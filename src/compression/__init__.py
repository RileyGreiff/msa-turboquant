"""Compression module — quantization and compression for KV cache tensors.

Provides a factory function to create compressors by name or from config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.compression.base import BaseCompressor, CompressedTensor
from src.compression.fp16 import FP16Compressor
from src.compression.int4 import Int4Compressor
from src.compression.int8 import Int8Compressor
from src.compression.turboquant_like import TurboQuantLikeCompressor

if TYPE_CHECKING:
    from src.utils.config import CompressionConfig


def create_compressor(method: str = "none", **kwargs) -> BaseCompressor:
    """Factory function to create a compressor by method name.

    Args:
        method: One of "none", "fp16", "int8", "int4", "turboquant_like".
        **kwargs: Extra arguments passed to the compressor constructor.

    Returns:
        A BaseCompressor instance.
    """
    if method in ("none", "fp16"):
        return FP16Compressor()
    elif method == "int8":
        return Int8Compressor(
            per_channel=kwargs.get("per_channel", False),
        )
    elif method == "int4":
        return Int4Compressor(
            group_size=kwargs.get("group_size", 128),
        )
    elif method == "turboquant_like":
        return TurboQuantLikeCompressor(
            bits=kwargs.get("bits", 4),
            group_size=kwargs.get("group_size", 128),
            rotation=kwargs.get("rotation", "random_orthogonal"),
            seed=kwargs.get("seed", 42),
            residual_correction=kwargs.get("residual_correction", False),
        )
    else:
        raise ValueError(
            f"Unknown compression method: {method}. "
            f"Use 'none', 'fp16', 'int8', 'int4', or 'turboquant_like'."
        )


def create_compressor_from_config(config: CompressionConfig) -> BaseCompressor:
    """Create a compressor from a CompressionConfig object.

    Reads the method field and delegates to the appropriate sub-config.
    """
    method = config.method

    if method in ("none", "fp16"):
        return FP16Compressor()
    elif method == "int8":
        return Int8Compressor(symmetric=config.int8.symmetric)
    elif method == "int4":
        return Int4Compressor(group_size=config.int4.group_size)
    elif method == "turboquant_like":
        tq = config.turboquant_like
        return TurboQuantLikeCompressor(
            bits=tq.bits,
            group_size=tq.group_size,
            rotation=tq.rotation,
            seed=tq.seed,
            residual_correction=tq.residual_correction,
        )
    else:
        raise ValueError(f"Unknown compression method in config: {method}")
