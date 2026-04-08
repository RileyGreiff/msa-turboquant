"""FP16 passthrough compressor (baseline — no actual compression).

Serves as the "no compression" baseline. Simply casts to fp16 and back.
Useful for measuring the overhead of the compression interface itself
and for fair comparisons (same code path, just no quantization).
"""

from __future__ import annotations

import torch

from src.compression.base import BaseCompressor, CompressedTensor


class FP16Compressor(BaseCompressor):
    """FP16 passthrough — stores data as float16 without quantization."""

    def compress(self, tensor: torch.Tensor, **kwargs) -> CompressedTensor:
        return CompressedTensor(
            data=tensor.to(torch.float16).clone(),
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            bits=16,
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        return compressed.data.to(compressed.original_dtype)

    def estimate_bits_per_value(self) -> float:
        return 16.0

    @property
    def name(self) -> str:
        return "fp16"
