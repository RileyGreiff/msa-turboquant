"""HuggingFace Transformers model backend.

Wraps AutoModelForCausalLM and AutoTokenizer behind the BaseModel interface.
Handles device placement, dtype casting, and output extraction.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from src.models.base_model import BaseModel, ModelOutput, TokenizedInput
from src.utils.config import ModelConfig, TokenizerConfig

logger = logging.getLogger("msa_turboquant.models.hf")

# Map string dtype names to torch dtypes
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "auto": torch.float16,  # default fallback
}


class HFModel(BaseModel):
    """HuggingFace Transformers model wrapper.

    Usage:
        model = HFModel(model_config, tokenizer_config)
        model.load()
        output = model.forward(input_ids, output_hidden_states=True)
        model.unload()
    """

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig | None = None,
    ) -> None:
        self._model_config = model_config
        self._tokenizer_config = tokenizer_config or TokenizerConfig()
        self._model: Any = None  # transformers.PreTrainedModel
        self._tokenizer: Any = None  # transformers.PreTrainedTokenizer
        self._loaded = False

    def load(self) -> None:
        """Load model and tokenizer from HuggingFace hub or local path."""
        if self._loaded:
            logger.warning("Model already loaded, skipping")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self._model_config.name
        tok_name = self._tokenizer_config.name or model_name
        target_dtype = _DTYPE_MAP.get(self._model_config.dtype, torch.float16)
        target_device = self._model_config.device

        logger.info(f"Loading tokenizer: {tok_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            tok_name,
            trust_remote_code=self._model_config.trust_remote_code,
            padding_side=self._tokenizer_config.padding_side,
        )
        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Set pad_token = eos_token")

        logger.info(f"Loading model: {model_name} (dtype={target_dtype}, device={target_device})")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=target_dtype,
            device_map=target_device if target_device != "cpu" else None,
            trust_remote_code=self._model_config.trust_remote_code,
            attn_implementation=self._model_config.attn_implementation,
        )
        if target_device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        self._loaded = True

        logger.info(
            f"Model loaded: {self.num_layers} layers, "
            f"{self.hidden_size} hidden, {self.num_heads} heads, "
            f"head_dim={self.head_dim}"
        )

    def unload(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

    # --- Properties ---

    @property
    def hidden_size(self) -> int:
        self._ensure_loaded()
        return self._model.config.hidden_size

    @property
    def num_layers(self) -> int:
        self._ensure_loaded()
        return self._model.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        self._ensure_loaded()
        return self._model.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        self._ensure_loaded()
        return self.hidden_size // self.num_heads

    @property
    def device(self) -> torch.device:
        self._ensure_loaded()
        return next(self._model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        self._ensure_loaded()
        return next(self._model.parameters()).dtype

    @property
    def vocab_size(self) -> int:
        self._ensure_loaded()
        return self._model.config.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self._model_config.max_seq_len

    @property
    def model_name(self) -> str:
        return self._model_config.name

    # --- Core methods ---

    def tokenize(
        self,
        texts: str | list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> TokenizedInput:
        """Tokenize text(s) using the HF tokenizer."""
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]

        max_len = max_length or self._tokenizer_config.max_length
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_len,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        num_tokens = attention_mask.sum(dim=1).tolist()

        return TokenizedInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_tokens=[int(n) for n in num_tokens],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_values: Any | None = None,
    ) -> ModelOutput:
        """Run a forward pass and extract requested outputs."""
        self._ensure_loaded()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            hf_output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )

        result = ModelOutput()

        # Logits
        if hasattr(hf_output, "logits") and hf_output.logits is not None:
            result.logits = hf_output.logits

        # Hidden states: tuple of (num_layers+1) tensors (including embedding layer)
        if output_hidden_states and hasattr(hf_output, "hidden_states"):
            result.hidden_states = hf_output.hidden_states
            # Last hidden state is the final layer's output (second to last in tuple
            # since HF includes embedding layer at index 0)
            result.last_hidden_state = hf_output.hidden_states[-1]

        # Attentions
        if output_attentions and hasattr(hf_output, "attentions"):
            result.attentions = hf_output.attentions

        # KV cache: tuple of (num_layers) tuples of (key, value)
        if use_cache and hasattr(hf_output, "past_key_values") and hf_output.past_key_values is not None:
            result.kv_cache = self._extract_kv_cache(hf_output.past_key_values)
            result.raw_kv_cache = hf_output.past_key_values

        return result

    def _extract_kv_cache(
        self, past_key_values: Any
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Extract KV tensors from HF past_key_values.

        HF models return past_key_values in different formats depending on
        the model and cache implementation. This method normalizes them.

        Returns:
            Tuple of (key, value) pairs per layer.
            Key/Value shape: (batch, num_kv_heads, seq_len, head_dim).

        NOTE: Some models use GQA (grouped query attention) where num_kv_heads
        may be less than num_attention_heads. The tensors are returned as-is
        without expanding to full head count.
        """
        # Modern HF (transformers >= 4.36) uses DynamicCache or similar objects.
        # Try multiple access patterns to handle different versions.

        # Pattern 1: .key_cache / .value_cache list attributes
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            key_cache = past_key_values.key_cache
            value_cache = past_key_values.value_cache
            if len(key_cache) > 0:
                layers = []
                for layer_idx in range(len(key_cache)):
                    layers.append((key_cache[layer_idx], value_cache[layer_idx]))
                return tuple(layers)

        # Pattern 2: iterable of (key, value) pairs (DynamicCache supports iteration)
        try:
            layers = []
            for layer_kv in past_key_values:
                if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                    layers.append((layer_kv[0], layer_kv[1]))
            if layers:
                return tuple(layers)
        except (TypeError, StopIteration):
            pass

        # Pattern 3: __len__ + __getitem__ (some cache implementations)
        if hasattr(past_key_values, "__len__") and hasattr(past_key_values, "__getitem__"):
            try:
                layers = []
                for i in range(len(past_key_values)):
                    item = past_key_values[i]
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        layers.append((item[0], item[1]))
                if layers:
                    return tuple(layers)
            except (IndexError, TypeError):
                pass

        logger.warning(
            f"Unknown past_key_values type: {type(past_key_values)}. "
            "Returning empty KV cache."
        )
        return ()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self._ensure_loaded()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                pad_token_id=self._tokenizer.pad_token_id,
                **{k: v for k, v in kwargs.items()
                   if k not in ("do_sample", "temperature", "top_p")},
            )
        return output_ids

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Decode token IDs back to text strings."""
        self._ensure_loaded()
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        return [
            self._tokenizer.decode(ids, skip_special_tokens=True)
            for ids in token_ids
        ]

    def generate_text(
        self,
        prompts: str | list[str],
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> list[str]:
        """Convenience: tokenize, generate, decode in one call."""
        tokens = self.tokenize(prompts)
        output_ids = self.generate(
            tokens.input_ids,
            tokens.attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        # Strip the prompt tokens from output
        prompt_len = tokens.input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        return self.decode(generated_ids)
