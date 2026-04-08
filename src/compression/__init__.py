"""Compression module — quantization and compression for KV cache tensors.

Provides a factory function to create compressors by name or from config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.compression.base import BaseCompressor, CompressedTensor
from src.compression.fp16 import FP16Compressor
from src.compression.int4 import Int4Compressor
from src.compression.int8 import Int8Compressor
from src.compression.kivi import KIVICompressor
from src.compression.turboquant_mse import TurboQuantMSECompressor

if TYPE_CHECKING:
    from src.utils.config import CompressionConfig


def create_compressor(method: str = "none", **kwargs) -> BaseCompressor:
    """Factory function to create a compressor by method name.

    Args:
        method: One of "none", "fp16", "int8", "int4", "turboquant_mse",
            or parameterized like "turboquant_mse_3b", "turboquant_mse_2b".
        **kwargs: Extra arguments passed to the compressor constructor.

    Returns:
        A BaseCompressor instance.
    """
    # Parse parameterized TQ-MSE strings: "turboquant_mse_3b" -> method=turboquant_mse, bits=3
    parsed_method = method
    if method.startswith("turboquant_mse_") and method.endswith("b"):
        bits_str = method[len("turboquant_mse_"):-1]
        kwargs.setdefault("bits", int(bits_str))
        parsed_method = "turboquant_mse"

    if parsed_method in ("none", "fp16"):
        return FP16Compressor()
    elif parsed_method == "int8":
        return Int8Compressor(
            per_channel=kwargs.get("per_channel", False),
        )
    elif parsed_method == "int4":
        return Int4Compressor(
            group_size=kwargs.get("group_size", 128),
        )
    elif parsed_method == "turboquant_mse":
        return TurboQuantMSECompressor(
            bits=kwargs.get("bits", 4),
            rotation=kwargs.get("rotation", "random_orthogonal"),
            seed=kwargs.get("seed", 42),
        )
    elif parsed_method == "kivi":
        return KIVICompressor(
            bits=kwargs.get("bits", 4),
            key_axis=kwargs.get("key_axis", "channel"),
            value_axis=kwargs.get("value_axis", "token"),
        )
    else:
        raise ValueError(
            f"Unknown compression method: {method}. "
            f"Use 'none', 'fp16', 'int8', 'int4', 'turboquant_mse', "
            f"'kivi', or 'turboquant_mse_Nb' (e.g., 'turboquant_mse_3b')."
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
    elif method == "turboquant_mse":
        tq = config.turboquant_mse
        return TurboQuantMSECompressor(
            bits=tq.bits,
            rotation=tq.rotation,
            seed=tq.seed,
        )
    else:
        raise ValueError(f"Unknown compression method in config: {method}")
