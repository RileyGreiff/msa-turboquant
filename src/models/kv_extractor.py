"""KV cache extraction utilities.

Provides methods to extract, inspect, and manipulate key-value cache tensors
from model forward passes. Includes a fallback "hidden-state projection"
approach for when direct KV extraction is not feasible.

Key design note:
    Direct KV extraction via `use_cache=True` returns the actual K/V tensors
    that the attention mechanism uses. However, the exact format varies between
    model architectures (MHA vs GQA vs MQA, DynamicCache vs legacy tuples).

    The fallback approach uses hidden states projected through learned or
    random linear maps to approximate K/V. This is useful for:
    - Models that don't cleanly expose KV cache
    - Quick prototyping without full model loading
    - Ablation studies comparing real vs approximate KV
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel, ModelOutput

logger = logging.getLogger("msa_turboquant.models.kv_extractor")


@dataclass
class KVBlock:
    """Extracted KV tensors for a single text block.

    Attributes:
        block_id: Identifier matching the source TextBlock.
        keys: Key tensors per layer. Shape per tensor: (num_kv_heads, seq_len, head_dim).
        values: Value tensors per layer. Shape per tensor: (num_kv_heads, seq_len, head_dim).
        routing_vector: Mean-pooled hidden state for routing. Shape: (hidden_dim,).
        num_tokens: Number of real (non-padding) tokens in this block.
        layer_indices: Which layer indices are stored (may be a subset).
    """
    block_id: str
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    routing_vector: torch.Tensor
    num_tokens: int
    layer_indices: list[int]

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def total_kv_bytes(self) -> int:
        """Total bytes used by K and V tensors."""
        k_bytes = sum(k.nelement() * k.element_size() for k in self.keys)
        v_bytes = sum(v.nelement() * v.element_size() for v in self.values)
        return k_bytes + v_bytes

    def to_device(self, device: str | torch.device) -> "KVBlock":
        """Move all tensors to a device. Returns a new KVBlock."""
        return KVBlock(
            block_id=self.block_id,
            keys=[k.to(device) for k in self.keys],
            values=[v.to(device) for v in self.values],
            routing_vector=self.routing_vector.to(device),
            num_tokens=self.num_tokens,
            layer_indices=self.layer_indices,
        )

    def to_cpu(self) -> "KVBlock":
        """Move all tensors to CPU (for storage)."""
        return self.to_device("cpu")


class KVExtractor:
    """Extract KV cache from a model for memory bank construction.

    Supports two modes:
    1. Direct extraction via model.forward(use_cache=True) — uses real KV tensors
    2. Fallback via hidden state projection — approximates KV from hidden states

    Usage:
        extractor = KVExtractor(model, mode="direct")
        kv_block = extractor.extract("Some text to encode", block_id="blk_0")
    """

    def __init__(
        self,
        model: BaseModel,
        mode: str = "direct",
        layers: list[int] | None = None,
    ) -> None:
        """
        Args:
            model: A loaded BaseModel instance.
            mode: "direct" for real KV extraction, "hidden_state" for approximation.
            layers: Which layer indices to extract. None = all layers.
        """
        self._model = model
        self._mode = mode
        self._layers = layers
        self._fallback_projections: Optional[dict] = None

        if mode not in ("direct", "hidden_state"):
            raise ValueError(f"Unknown extraction mode: {mode}. Use 'direct' or 'hidden_state'.")

    def extract(
        self,
        text: str,
        block_id: str = "block_0",
    ) -> KVBlock:
        """Extract KV tensors for a single text block.

        Args:
            text: Input text for this block.
            block_id: Identifier for this block.

        Returns:
            KVBlock with extracted K/V tensors and routing vector.
        """
        tokens = self._model.tokenize(text)
        num_tokens = tokens.num_tokens[0]

        if self._mode == "direct":
            return self._extract_direct(tokens.input_ids, tokens.attention_mask, block_id, num_tokens)
        else:
            return self._extract_from_hidden_states(tokens.input_ids, tokens.attention_mask, block_id, num_tokens)

    def extract_batch(
        self,
        texts: list[str],
        block_ids: list[str] | None = None,
    ) -> list[KVBlock]:
        """Extract KV tensors for a batch of text blocks.

        Args:
            texts: List of input texts.
            block_ids: Optional list of block identifiers.

        Returns:
            List of KVBlock objects, one per input text.
        """
        if block_ids is None:
            block_ids = [f"block_{i}" for i in range(len(texts))]

        # For simplicity and memory safety, process one at a time
        # Batching KV extraction can OOM on large models
        results = []
        for text, bid in zip(texts, block_ids):
            kv_block = self.extract(text, block_id=bid)
            results.append(kv_block)
        return results

    def _extract_direct(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_id: str,
        num_tokens: int,
    ) -> KVBlock:
        """Extract real KV tensors using model's use_cache output."""
        output = self._model.forward(
            input_ids,
            attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )

        if output.kv_cache is None or len(output.kv_cache) == 0:
            logger.warning(
                "Direct KV extraction returned no cache. "
                "Falling back to hidden_state mode for this block."
            )
            return self._extract_from_hidden_states(
                input_ids, attention_mask, block_id, num_tokens
            )

        # Select which layers to keep
        all_layer_indices = list(range(len(output.kv_cache)))
        layer_indices = self._layers if self._layers is not None else all_layer_indices

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for li in layer_indices:
            if li < len(output.kv_cache):
                k, v = output.kv_cache[li]
                # Remove batch dim, trim to actual tokens, move to CPU
                keys.append(k[0, :, :num_tokens, :].cpu())
                values.append(v[0, :, :num_tokens, :].cpu())

        # Compute routing vector from last hidden state
        routing_vector = self._compute_routing_vector(output, attention_mask)

        return KVBlock(
            block_id=block_id,
            keys=keys,
            values=values,
            routing_vector=routing_vector.cpu(),
            num_tokens=num_tokens,
            layer_indices=layer_indices,
        )

    def _extract_from_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_id: str,
        num_tokens: int,
    ) -> KVBlock:
        """Approximate KV from hidden states using random projections.

        This is a fallback when direct KV extraction isn't available.
        It projects hidden states through fixed random matrices to create
        pseudo K/V tensors. The projections are deterministic (seeded)
        so the same hidden states always produce the same pseudo-KV.

        NOTE: These are NOT real attention K/V tensors. They preserve
        the information content of hidden states in a KV-shaped format
        suitable for compression and retrieval experiments.
        """
        output = self._model.forward(
            input_ids,
            attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        if output.hidden_states is None:
            raise RuntimeError("Model returned no hidden states")

        # Select layers (skip embedding layer at index 0)
        num_model_layers = len(output.hidden_states) - 1
        all_layer_indices = list(range(num_model_layers))
        layer_indices = self._layers if self._layers is not None else all_layer_indices

        # Initialize projection matrices lazily
        if self._fallback_projections is None:
            self._init_fallback_projections(num_model_layers)

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for li in layer_indices:
            # Hidden state at this layer (skip embedding at index 0)
            h = output.hidden_states[li + 1][0, :num_tokens, :]  # (seq_len, hidden_dim)

            # Project to pseudo K/V using fixed random projections
            # Reshape to (num_heads, seq_len, head_dim) to match KV format
            hidden_dim = h.shape[-1]
            num_heads = self._model.num_heads
            head_dim = hidden_dim // num_heads

            k_proj = self._fallback_projections[f"k_{li}"].to(device=h.device, dtype=h.dtype)
            v_proj = self._fallback_projections[f"v_{li}"].to(device=h.device, dtype=h.dtype)

            k = (h @ k_proj).view(num_tokens, num_heads, head_dim).permute(1, 0, 2)
            v = (h @ v_proj).view(num_tokens, num_heads, head_dim).permute(1, 0, 2)

            keys.append(k.cpu())
            values.append(v.cpu())

        routing_vector = self._compute_routing_vector(output, attention_mask)

        return KVBlock(
            block_id=block_id,
            keys=keys,
            values=values,
            routing_vector=routing_vector.cpu(),
            num_tokens=num_tokens,
            layer_indices=layer_indices,
        )

    def _init_fallback_projections(self, num_layers: int) -> None:
        """Initialize deterministic random projection matrices for pseudo-KV."""
        hidden_dim = self._model.hidden_size
        gen = torch.Generator().manual_seed(12345)

        self._fallback_projections = {}
        for li in range(num_layers):
            # Random orthogonal-ish projection (QR decomposition of random matrix)
            k_random = torch.randn(hidden_dim, hidden_dim, generator=gen)
            v_random = torch.randn(hidden_dim, hidden_dim, generator=gen)
            k_proj, _ = torch.linalg.qr(k_random)
            v_proj, _ = torch.linalg.qr(v_random)
            self._fallback_projections[f"k_{li}"] = k_proj
            self._fallback_projections[f"v_{li}"] = v_proj

        logger.info(
            f"Initialized fallback projections for {num_layers} layers "
            f"(hidden_dim={hidden_dim})"
        )

    def _compute_routing_vector(
        self,
        output: ModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a mean-pooled routing vector from the last hidden state."""
        hidden = output.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (batch, seq_len, 1)
        summed = (hidden * mask).sum(dim=1)  # (batch, hidden_dim)
        counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        routing = summed / counts  # (batch, hidden_dim)
        # Normalize for cosine similarity retrieval
        routing = F.normalize(routing, p=2, dim=-1)
        return routing[0]  # Return single vector (batch=1 expected)
