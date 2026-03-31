"""INT4 groupwise symmetric quantization compressor.

Quantizes float tensors to 4-bit integers using groupwise quantization.
The tensor's last dimension is split into groups of `group_size`, and each
group gets its own scale factor. Values are mapped to [-7, 7].

Stored as int8 (since PyTorch has no native int4). The effective bits per
value is ~4 + overhead from scale factors.

Quantization formula (symmetric, per group):
    scale_g = max(|group_g|) / 7
    quantized_g = round(group_g / scale_g)
    dequantized_g = quantized_g * scale_g
"""

from __future__ import annotations

import torch

from src.compression.base import BaseCompressor, CompressedTensor
from src.compression.bitpack import pack, unpack


class Int4Compressor(BaseCompressor):
    """Groupwise symmetric INT4 quantization.

    Args:
        group_size: Number of values per quantization group along the last dim.
            Must evenly divide the last dimension of input tensors.
    """

    def __init__(self, group_size: int = 128) -> None:
        self._group_size = group_size

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        original_dtype = tensor.dtype
        original_shape = tensor.shape
        t = tensor.float()

        last_dim = t.shape[-1]
        gs = self._group_size

        # If last dim not divisible by group_size, pad
        pad_amount = (gs - last_dim % gs) % gs
        if pad_amount > 0:
            t = torch.nn.functional.pad(t, (0, pad_amount))

        # Reshape to (..., num_groups, group_size)
        leading_shape = t.shape[:-1]
        padded_dim = t.shape[-1]
        num_groups = padded_dim // gs
        grouped = t.reshape(*leading_shape, num_groups, gs)

        # Per-group symmetric quantization
        max_vals = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., ng, 1)
        scales = max_vals / 7.0  # (..., ng, 1)
        quantized = (grouped / scales).round().clamp(-7, 7).to(torch.int8)

        # Flatten back: (..., num_groups * group_size)
        quantized_flat = quantized.reshape(*leading_shape, padded_dim)
        scales_flat = scales.squeeze(-1)  # (..., num_groups)

        # Bit-pack: shift [-7,7] -> [0,14] unsigned, then pack 2 per byte
        quantized_shape = quantized_flat.shape
        unsigned = (quantized_flat + 7).to(torch.uint8)
        packed = pack(unsigned, bits=4)

        return CompressedTensor(
            data=packed,
            scales=scales_flat,
            original_shape=original_shape,
            original_dtype=original_dtype,
            bits=4,
            metadata={
                "group_size": gs,
                "pad_amount": pad_amount,
                "num_groups": num_groups,
                "quantized_shape": quantized_shape,
                "num_values": unsigned.numel(),
            },
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        # Unpack bit-packed data back to int8 quantized values
        meta = compressed.metadata
        gs = meta["group_size"]
        pad_amount = meta["pad_amount"]
        original_shape = compressed.original_shape
        quantized_shape = meta["quantized_shape"]
        num_values = meta["num_values"]

        unsigned = unpack(compressed.data, bits=4, num_values=num_values)
        quantized = (unsigned.to(torch.int8) - 7).float()
        quantized = quantized.reshape(quantized_shape)

        scales = compressed.scales  # (..., num_groups)
        leading_shape = quantized.shape[:-1]
        padded_dim = quantized.shape[-1]
        num_groups = padded_dim // gs

        # Reshape to groups
        grouped = quantized.reshape(*leading_shape, num_groups, gs)
        scales_expanded = scales.unsqueeze(-1)  # (..., ng, 1)

        # Dequantize
        dequantized = grouped * scales_expanded
        dequantized = dequantized.reshape(*leading_shape, padded_dim)

        # Remove padding
        if pad_amount > 0:
            dequantized = dequantized[..., :original_shape[-1]]

        return dequantized.to(compressed.original_dtype)

    def estimate_bits_per_value(self) -> float:
        # 4 bits per value + scale overhead (fp32 scale per group)
        # Overhead: 32 bits / group_size values
        overhead = 32.0 / self._group_size
        return 4.0 + overhead

    @property
    def name(self) -> str:
        return f"int4_g{self._group_size}"
