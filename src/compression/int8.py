"""INT8 symmetric quantization compressor.

Quantizes float tensors to 8-bit integers using per-tensor or per-channel
symmetric quantization. Each value is mapped to [-127, 127] with a single
scale factor.

Quantization formula (symmetric):
    scale = max(|tensor|) / 127
    quantized = round(tensor / scale)
    dequantized = quantized * scale
"""

from __future__ import annotations

import torch

from src.compression.base import BaseCompressor, CompressedTensor


class Int8Compressor(BaseCompressor):
    """Symmetric INT8 quantization.

    Args:
        per_channel: If True, compute separate scale per last dimension
            (e.g., per head_dim for KV tensors). If False, single scale
            for the entire tensor.
    """

    def __init__(self, per_channel: bool = False, symmetric: bool = True) -> None:
        self._per_channel = per_channel
        self._symmetric = symmetric  # asymmetric not yet implemented

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        original_dtype = tensor.dtype
        original_shape = tensor.shape
        t = tensor.float()

        if self._per_channel and t.dim() >= 2:
            # Scale per last dimension: flatten all but last, compute max per column
            flat = t.reshape(-1, t.shape[-1])
            max_vals = flat.abs().amax(dim=0).clamp(min=1e-8)  # (last_dim,)
            scales = max_vals / 127.0
            quantized = (flat / scales.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
            quantized = quantized.reshape(original_shape)
        else:
            max_val = t.abs().max().clamp(min=1e-8)
            scales = (max_val / 127.0).unsqueeze(0)
            quantized = (t / scales).round().clamp(-127, 127).to(torch.int8)

        return CompressedTensor(
            data=quantized,
            scales=scales,
            original_shape=original_shape,
            original_dtype=original_dtype,
            bits=8,
            metadata={"per_channel": self._per_channel},
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        quantized = compressed.data.float()
        scales = compressed.scales

        if compressed.metadata.get("per_channel") and quantized.dim() >= 2:
            flat = quantized.reshape(-1, quantized.shape[-1])
            dequantized = flat * scales.unsqueeze(0)
            dequantized = dequantized.reshape(compressed.original_shape)
        else:
            dequantized = quantized * scales

        return dequantized.to(compressed.original_dtype)

    def estimate_bits_per_value(self) -> float:
        return 8.0

    @property
    def name(self) -> str:
        suffix = "_perchannel" if self._per_channel else ""
        return f"int8{suffix}"
