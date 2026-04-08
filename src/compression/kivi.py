"""KIVI-style asymmetric quantization: per-channel keys, per-token values.

Based on KIVI (ICML 2024): keys have high-magnitude outliers in specific
channels (head_dim positions), so per-channel quantization preserves them.
Values don't have this structure, so per-token quantization works fine.

This compressor is KV-cache-aware: it detects whether it's compressing keys
or values based on metadata and applies the appropriate quantization axis.

For KV cache tensors with shape (num_heads, seq_len, head_dim):
  - Keys: per-channel = one scale per (head, channel), quantize across seq_len
  - Values: per-token = one scale per (head, token), quantize across head_dim

This matches what ships in production systems (vLLM, etc.) and avoids the
correlated error problem that rotation-based methods (TQ-MSE) have with keys.
"""

from __future__ import annotations

import torch

from src.compression.base import BaseCompressor, CompressedTensor
from src.compression.bitpack import pack, unpack


class KIVICompressor(BaseCompressor):
    """KIVI-style asymmetric quantizer for KV cache.

    Args:
        bits: Quantization bits (2, 4, or 8).
        key_axis: Quantization axis for keys. "channel" (per head_dim channel,
            KIVI default) or "group" (standard groupwise along head_dim).
        value_axis: Quantization axis for values. "token" (per token,
            KIVI default) or "group" (standard groupwise along head_dim).
        group_size: Group size when using "group" axis. Ignored for
            "channel"/"token" modes.
    """

    def __init__(
        self,
        bits: int = 4,
        key_axis: str = "channel",
        value_axis: str = "token",
        group_size: int = 128,
    ) -> None:
        self._bits = bits
        self._key_axis = key_axis
        self._value_axis = value_axis
        self._group_size = group_size

        self._qmax = (1 << (bits - 1)) - 1
        self._qmin = -self._qmax

    def compress(self, tensor: torch.Tensor, **kwargs) -> CompressedTensor:
        """Compress a KV tensor with axis-appropriate quantization.

        Args:
            tensor: Shape (num_heads, seq_len, head_dim) for KV cache.
            **kwargs: Pass ``is_key=True`` for key tensors (per-channel),
                ``is_key=False`` for values (per-token). Defaults to True.
        """
        is_key = kwargs.get("is_key", True)
        original_dtype = tensor.dtype
        original_shape = tensor.shape
        t = tensor.float()

        axis = self._key_axis if is_key else self._value_axis

        if axis == "channel" and t.ndim == 3:
            # Per-channel: scale per (head, channel), quantize across seq_len
            # Shape: (num_heads, seq_len, head_dim)
            # Compute scale per (head, head_dim) -> amax over seq_len dim (1)
            max_vals = t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (H, 1, D)
            scales = max_vals / self._qmax
            quantized = (t / scales).round().clamp(self._qmin, self._qmax).to(torch.int8)
            scales = scales.squeeze(1)  # (H, D)
        elif axis == "token" and t.ndim == 3:
            # Per-token: scale per (head, token), quantize across head_dim
            # Compute scale per (head, seq_len) -> amax over head_dim dim (2)
            max_vals = t.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # (H, S, 1)
            scales = max_vals / self._qmax
            quantized = (t / scales).round().clamp(self._qmin, self._qmax).to(torch.int8)
            scales = scales.squeeze(2)  # (H, S)
        else:
            # Fallback: standard groupwise along last dim (same as INT4)
            max_vals = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scales = max_vals / self._qmax
            quantized = (t / scales).round().clamp(self._qmin, self._qmax).to(torch.int8)
            scales = scales.squeeze(-1)

        # Bit-pack for sub-8-bit
        if self._bits < 8:
            unsigned = (quantized + self._qmax).to(torch.uint8)
            quantized_shape = unsigned.shape
            packed = pack(unsigned, bits=self._bits)
            return CompressedTensor(
                data=packed,
                scales=scales,
                original_shape=original_shape,
                original_dtype=original_dtype,
                bits=self._bits,
                metadata={
                    "axis": axis,
                    "is_key": is_key,
                    "quantized_shape": quantized_shape,
                    "num_values": unsigned.numel(),
                    "qmax": self._qmax,
                },
            )

        return CompressedTensor(
            data=quantized,
            scales=scales,
            original_shape=original_shape,
            original_dtype=original_dtype,
            bits=self._bits,
            metadata={
                "axis": axis,
                "is_key": is_key,
                "qmax": self._qmax,
            },
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        meta = compressed.metadata
        axis = meta["axis"]
        qmax = meta["qmax"]
        scales = compressed.scales

        if compressed.bits < 8:
            unsigned = unpack(compressed.data, bits=compressed.bits, num_values=meta["num_values"])
            quantized = (unsigned.to(torch.int8) - qmax).float()
            quantized = quantized.reshape(meta["quantized_shape"])
        else:
            quantized = compressed.data.float()

        if axis == "channel":
            # scales shape: (H, D), quantized shape: (H, S, D)
            dequantized = quantized * scales.unsqueeze(1)  # broadcast over seq_len
        elif axis == "token":
            # scales shape: (H, S), quantized shape: (H, S, D)
            dequantized = quantized * scales.unsqueeze(2)  # broadcast over head_dim
        else:
            dequantized = quantized * scales.unsqueeze(-1)

        return dequantized.to(compressed.original_dtype)

    def estimate_bits_per_value(self) -> float:
        # bits per value + scale overhead
        # Per-channel: 32 bits per (head, channel), amortized over seq_len
        # Per-token: 32 bits per (head, token), amortized over head_dim
        # Both are small — assume typical seq_len=128, head_dim=128
        return float(self._bits) + 0.25

    @property
    def name(self) -> str:
        return f"kivi_{self._bits}b_k{self._key_axis}_v{self._value_axis}"
