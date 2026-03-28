"""Abstract base class for LLM model wrappers.

Defines the interface that all model backends must implement. This keeps the
rest of the system (memory bank builder, evaluation, experiments) decoupled
from any specific model provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class ModelOutput:
    """Standardized output from a model forward pass.

    Attributes:
        hidden_states: Per-layer hidden states. Shape per layer: (batch, seq_len, hidden_dim).
            May be None if not requested.
        last_hidden_state: Final layer hidden state. Shape: (batch, seq_len, hidden_dim).
        logits: Output logits. Shape: (batch, seq_len, vocab_size). May be None.
        kv_cache: Per-layer (key, value) tuples. Shape per tensor:
            (batch, num_heads, seq_len, head_dim). May be None if not available.
        routing_vectors: Mean-pooled hidden states for retrieval routing.
            Shape: (batch, hidden_dim). May be None if not computed.
        attentions: Per-layer attention weights. May be None.
    """
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    kv_cache: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None
    raw_kv_cache: Any = None  # Backend-native cache (e.g. DynamicCache) for chunked prefill
    routing_vectors: Optional[torch.Tensor] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class TokenizedInput:
    """Standardized tokenizer output.

    Attributes:
        input_ids: Token IDs. Shape: (batch, seq_len).
        attention_mask: Attention mask. Shape: (batch, seq_len).
        num_tokens: Number of non-padding tokens per sequence.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    num_tokens: list[int] = field(default_factory=list)


class BaseModel(ABC):
    """Abstract interface for LLM model wrappers.

    All model backends (HuggingFace, GGUF, custom) should subclass this
    and implement the required methods.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer into memory/device."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free model from memory. Called during cleanup."""
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the model's hidden dimension size."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Return the number of attention heads."""
        ...

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Return the dimension per attention head."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device the model is on."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Return the model's compute dtype."""
        ...

    @abstractmethod
    def tokenize(
        self,
        texts: str | list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> TokenizedInput:
        """Tokenize one or more text strings.

        Args:
            texts: Single string or list of strings.
            max_length: Maximum sequence length (None = model default).
            padding: Whether to pad shorter sequences.
            truncation: Whether to truncate longer sequences.

        Returns:
            TokenizedInput with input_ids and attention_mask.
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_values: Any | None = None,
    ) -> ModelOutput:
        """Run a forward pass through the model.

        Args:
            input_ids: Token IDs. Shape: (batch, seq_len).
            attention_mask: Attention mask. Shape: (batch, seq_len).
            output_hidden_states: Whether to return all layer hidden states.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return KV cache.
            past_key_values: Previously computed KV cache for incremental
                decoding / chunked prefill. Format depends on backend.

        Returns:
            ModelOutput with requested outputs populated.
        """
        ...

    def get_hidden_states(
        self,
        texts: str | list[str],
        layer: int = -1,
    ) -> torch.Tensor:
        """Get hidden states for texts at a specific layer.

        Convenience method that tokenizes, runs forward, and extracts.

        Args:
            texts: Input text(s).
            layer: Which layer's hidden states to return (-1 = last).

        Returns:
            Hidden state tensor. Shape: (batch, seq_len, hidden_dim).
        """
        tokens = self.tokenize(texts)
        with torch.no_grad():
            output = self.forward(
                tokens.input_ids,
                tokens.attention_mask,
                output_hidden_states=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        return output.hidden_states[layer]

    def get_routing_vectors(
        self,
        texts: str | list[str],
        pooling: str = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        """Compute routing vectors for memory bank retrieval.

        Pools hidden states into a single vector per input text.

        Args:
            texts: Input text(s).
            pooling: Pooling strategy — "mean", "last", or "cls".
            layer: Which layer to pool from (-1 = last).

        Returns:
            Routing vectors. Shape: (batch, hidden_dim).
        """
        tokens = self.tokenize(texts)
        with torch.no_grad():
            output = self.forward(
                tokens.input_ids,
                tokens.attention_mask,
                output_hidden_states=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("Model did not return hidden states")

        hidden = output.hidden_states[layer]  # (batch, seq_len, hidden_dim)
        mask = tokens.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)

        if pooling == "mean":
            # Mean pool over non-padding tokens
            summed = (hidden * mask).sum(dim=1)  # (batch, hidden_dim)
            counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
            vectors = summed / counts
        elif pooling == "last":
            # Last non-padding token per sequence
            seq_lengths = tokens.attention_mask.sum(dim=1) - 1  # (batch,)
            vectors = hidden[torch.arange(hidden.size(0)), seq_lengths]
        elif pooling == "cls":
            # First token
            vectors = hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        return vectors

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs.
            attention_mask: Attention mask.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated token IDs (including prompt).
        """
        ...

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Decode token IDs back to text. Must be implemented by subclass."""
        raise NotImplementedError("decode() not implemented for this backend")
