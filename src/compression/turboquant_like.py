"""Backwards-compatibility shim — redirects to rotated_uniform.py.

The class formerly known as TurboQuantLikeCompressor has been renamed to
RotatedUniformCompressor to avoid confusion with the actual TurboQuant
algorithm (see turboquant_mse.py). This module re-exports everything
so existing imports continue to work.
"""

from src.compression.rotated_uniform import (  # noqa: F401
    RotatedUniformCompressor,
    TurboQuantLikeCompressor,
    _fast_walsh_hadamard_transform,
    _generate_random_orthogonal,
    _generate_random_signs,
    _next_power_of_2,
)

__all__ = [
    "RotatedUniformCompressor",
    "TurboQuantLikeCompressor",
    "_fast_walsh_hadamard_transform",
    "_generate_random_orthogonal",
    "_generate_random_signs",
    "_next_power_of_2",
]
