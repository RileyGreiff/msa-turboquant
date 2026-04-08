"""Abstract compressor interface for KV cache compression.

All compressors implement a common interface:
- compress(tensor) -> CompressedTensor
- decompress(compressed) -> tensor
- estimate_bits_per_value() -> float

CompressedTensor is a container that holds the compressed data along with
metadata needed for decompression (scales, zero points, shape, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CompressedTensor:
    """Container for compressed tensor data.

    Attributes:
        data: The compressed data (quantized integers, packed bits, etc.).
        scales: Quantization scale factors (if applicable).
        zero_points: Quantization zero points (if applicable).
        original_shape: Shape of the original uncompressed tensor.
        original_dtype: dtype of the original tensor.
        bits: Bits per value in the compressed representation.
        metadata: Extra info needed for decompression.
    """
    data: torch.Tensor
    scales: torch.Tensor | None = None
    zero_points: torch.Tensor | None = None
    original_shape: tuple[int, ...] = ()
    original_dtype: torch.dtype = torch.float16
    bits: int = 16
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compressed_bytes(self) -> int:
        """Total bytes used by all compressed components."""
        total = self.data.nelement() * self.data.element_size()
        if self.scales is not None:
            total += self.scales.nelement() * self.scales.element_size()
        if self.zero_points is not None:
            total += self.zero_points.nelement() * self.zero_points.element_size()
        return total

    @property
    def original_bytes(self) -> int:
        """Bytes the original uncompressed tensor would use."""
        numel = 1
        for s in self.original_shape:
            numel *= s
        dtype_size = torch.tensor([], dtype=self.original_dtype).element_size()
        return numel * dtype_size

    @property
    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size."""
        cb = self.compressed_bytes
        return self.original_bytes / cb if cb > 0 else float("inf")


class BaseCompressor(ABC):
    """Abstract base class for tensor compressors."""

    @abstractmethod
    def compress(self, tensor: torch.Tensor, **kwargs) -> CompressedTensor:
        """Compress a tensor.

        Args:
            tensor: Input tensor of any shape. Typically (num_heads, seq_len, head_dim)
                for KV cache tensors, or (seq_len, hidden_dim) for hidden states.
            **kwargs: Compressor-specific options. KV-aware compressors (e.g. KIVI)
                accept ``is_key=True/False`` to apply different quantization axes
                for keys vs values. Other compressors ignore this.

        Returns:
            CompressedTensor containing the compressed data.
        """
        ...

    @abstractmethod
    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress back to a tensor.

        Args:
            compressed: CompressedTensor from a previous compress() call.

        Returns:
            Reconstructed tensor with the original shape and dtype.
        """
        ...

    @abstractmethod
    def estimate_bits_per_value(self) -> float:
        """Estimate the effective bits per value for this compressor.

        Returns:
            Bits per value (e.g., 16.0 for fp16, 8.0 for int8, 4.0 for int4).
            This is an estimate — actual compression ratio depends on overheads
            like scale factors.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this compressor."""
        ...

    def compute_reconstruction_error(self, tensor: torch.Tensor) -> dict[str, float]:
        """Compress and decompress, then measure reconstruction quality.

        Returns dict with:
            - mse: Mean squared error
            - max_abs_error: Maximum absolute error
            - cosine_sim: Cosine similarity (flattened)
            - snr_db: Signal-to-noise ratio in dB
        """
        compressed = self.compress(tensor)
        reconstructed = self.decompress(compressed)

        orig = tensor.float()
        recon = reconstructed.float()
        diff = orig - recon

        mse = (diff ** 2).mean().item()
        max_abs = diff.abs().max().item()

        # Cosine similarity on flattened tensors
        orig_flat = orig.flatten()
        recon_flat = recon.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()

        # SNR
        signal_power = (orig ** 2).mean().item()
        noise_power = mse
        if noise_power > 0:
            snr_db = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
        else:
            snr_db = float("inf")

        return {
            "mse": mse,
            "max_abs_error": max_abs,
            "cosine_sim": cos_sim,
            "snr_db": snr_db,
            "compression_ratio": compressed.compression_ratio,
            "bits_per_value": self.estimate_bits_per_value(),
        }
