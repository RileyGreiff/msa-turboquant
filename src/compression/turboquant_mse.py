"""TurboQuant MSE compressor: norm separation + rotation + Lloyd-Max codebook quantization.

Implements the TurboQuant_MSE algorithm from the paper:
1. Separate each vector's L2 norm (store as fp32 scalar per vector)
2. Apply random rotation to the unit-normalized vector
3. Quantize each rotated coordinate independently using a precomputed
   Lloyd-Max codebook optimized for the standard Gaussian distribution
   (post-rotation coordinates are approximately N(0, 1/d))
4. On decode: dequantize via codebook lookup, inverse rotate, rescale by norm

Key differences from the RotatedUniformCompressor baseline:
- No group_size: the codebook is shared across ALL coordinates
- Norm-separated: vectors are normalized to unit length before quantization
- Lloyd-Max codebooks: optimal quantizer for Gaussian, not uniform max-abs scaling
- Better MSE guarantees from the paper's theoretical analysis
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch

from src.compression.base import BaseCompressor, CompressedTensor
from src.compression.bitpack import pack, unpack
from src.compression.rotated_uniform import (
    _fast_walsh_hadamard_transform,
    _generate_random_orthogonal,
    _generate_random_signs,
    _next_power_of_2,
)

logger = logging.getLogger("msa_turboquant.compression.turboquant_mse")


# ---------------------------------------------------------------------------
# Precomputed Lloyd-Max codebooks for standard Gaussian N(0,1)
# ---------------------------------------------------------------------------
# These are the optimal scalar quantizer centroids and decision boundaries
# for a standard Gaussian distribution at each bit-width.
# Sources: Lloyd (1982), Max (1960), widely tabulated.
#
# Format: {bits: (boundaries, centroids)}
#   boundaries: decision thresholds (len = 2^bits - 1), sorted ascending
#   centroids:  reconstruction levels (len = 2^bits), sorted ascending
#
# The quantizer assigns x to bin i if boundaries[i-1] < x <= boundaries[i]
# (with boundaries[-1] = -inf, boundaries[2^bits-1] = +inf).

_LLOYD_MAX_CODEBOOKS: dict[int, tuple[list[float], list[float]]] = {
    2: (
        # 4 levels, 3 boundaries
        [-0.9816, 0.0, 0.9816],
        [-1.5104, -0.4528, 0.4528, 1.5104],
    ),
    3: (
        # 8 levels, 7 boundaries
        [-1.7479, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7479],
        [-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519],
    ),
    4: (
        # 16 levels, 15 boundaries
        [
            -2.4008, -1.8441, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
            0.2582, 0.5224, 0.7996, 1.0993, 1.4371, 1.8441, 2.4008,
        ],
        [
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
        ],
    ),
    5: (
        # 32 levels, 31 boundaries
        [
            -2.9608, -2.4862, -2.1701, -1.9222, -1.7117, -1.5255, -1.3563, -1.1999,
            -1.0535, -0.9150, -0.7829, -0.6560, -0.5334, -0.4143, -0.2979, -0.1837,
            -0.0712, 0.0712, 0.1837, 0.2979, 0.4143, 0.5334, 0.6560, 0.7829,
            0.9150, 1.0535, 1.1999, 1.3563, 1.5255, 1.7117, 1.9222,
        ],
        [
            -3.2619, -2.6871, -2.3068, -2.0331, -1.8082, -1.6123, -1.4363, -1.2744,
            -1.1236, -0.9817, -0.8470, -0.7183, -0.5944, -0.4740, -0.3566, -0.2415,
            -0.1281, 0.0, 0.1281, 0.2415, 0.3566, 0.4740, 0.5944, 0.7183,
            0.8470, 0.9817, 1.1236, 1.2744, 1.4363, 1.6123, 1.8082, 2.0331,
        ],
    ),
    8: (
        # 256 levels — for 8-bit we fall back to uniform quantization
        # since Lloyd-Max at 256 levels is nearly identical to uniform for Gaussian
        # Placeholder: will use uniform path
        [],
        [],
    ),
}


def _build_codebook_tensors(
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (boundaries, centroids) as float32 tensors for the given bit-width.

    For 8-bit, returns empty tensors (caller should use uniform fallback).
    """
    if bits not in _LLOYD_MAX_CODEBOOKS:
        raise ValueError(
            f"No Lloyd-Max codebook for {bits} bits. "
            f"Available: {sorted(_LLOYD_MAX_CODEBOOKS.keys())}"
        )
    boundaries, centroids = _LLOYD_MAX_CODEBOOKS[bits]
    return (
        torch.tensor(boundaries, dtype=torch.float32),
        torch.tensor(centroids, dtype=torch.float32),
    )


class TurboQuantMSECompressor(BaseCompressor):
    """TurboQuant MSE compressor: norm + rotation + Lloyd-Max codebook.

    Implements Algorithm 1 from the TurboQuant paper for MSE-optimal
    quantization of vectors.

    Args:
        bits: Quantization bits (2-5 use Lloyd-Max, 8 uses uniform fallback).
        rotation: Rotation type — "random_orthogonal" or "hadamard".
        seed: Random seed for the rotation matrix / sign flips.
    """

    def __init__(
        self,
        bits: int = 4,
        rotation: Literal["random_orthogonal", "hadamard"] = "random_orthogonal",
        seed: int = 42,
    ) -> None:
        self._bits = bits
        self._rotation = rotation
        self._seed = seed

        # Build codebook tensors
        self._boundaries, self._centroids = _build_codebook_tensors(bits)
        self._use_uniform_fallback = len(self._centroids) == 0

        if self._use_uniform_fallback:
            self._qmax = (1 << (bits - 1)) - 1
            self._qmin = -self._qmax

        # Cache rotation matrices and sign vectors per dimension
        self._rotation_cache: dict[int, torch.Tensor | None] = {}
        self._sign_cache: dict[int, torch.Tensor] = {}

    def _get_rotation_matrix(self, dim: int, device: torch.device) -> torch.Tensor | None:
        """Get or create a rotation matrix for the given dimension."""
        if dim not in self._rotation_cache:
            if self._rotation == "random_orthogonal":
                self._rotation_cache[dim] = _generate_random_orthogonal(dim, self._seed)
            else:
                self._rotation_cache[dim] = None
        mat = self._rotation_cache[dim]
        if mat is not None:
            return mat.to(device)
        return None

    def _get_sign_vector(self, dim: int, device: torch.device) -> torch.Tensor:
        """Get or create the random sign vector for signed Hadamard."""
        if dim not in self._sign_cache:
            self._sign_cache[dim] = _generate_random_signs(dim, self._seed)
        return self._sign_cache[dim].to(device)

    def _rotate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply rotation to the last dimension."""
        dim = tensor.shape[-1]
        if self._rotation == "hadamard":
            p2 = _next_power_of_2(dim)
            if dim != p2:
                tensor = torch.nn.functional.pad(tensor, (0, p2 - dim))
            signs = self._get_sign_vector(tensor.shape[-1], tensor.device)
            tensor = tensor * signs
            return _fast_walsh_hadamard_transform(tensor)
        else:
            rot = self._get_rotation_matrix(dim, tensor.device)
            return tensor @ rot.to(tensor.dtype)

    def _inverse_rotate(self, tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
        """Apply inverse rotation to the last dimension."""
        if self._rotation == "hadamard":
            result = _fast_walsh_hadamard_transform(tensor)
            signs = self._get_sign_vector(tensor.shape[-1], tensor.device)
            result = result * signs
            if result.shape[-1] != original_dim:
                result = result[..., :original_dim]
            return result
        else:
            rot = self._get_rotation_matrix(original_dim, tensor.device)
            return tensor @ rot.to(tensor.dtype).T

    def _codebook_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize each element to the nearest Lloyd-Max centroid.

        Args:
            tensor: Float tensor, values expected to be ~ N(0, sigma).

        Returns:
            Int16 tensor of codebook indices (0 to 2^bits - 1).
        """
        boundaries = self._boundaries.to(tensor.device)
        # bucketize: returns index i such that boundaries[i-1] < x <= boundaries[i]
        # torch.bucketize returns the insertion index, which is our bin index
        indices = torch.bucketize(tensor, boundaries)
        return indices.to(torch.int16)

    def _codebook_dequantize(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Map codebook indices back to centroid values."""
        centroids = self._centroids.to(device)
        return centroids[indices.long()]

    def _uniform_quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback uniform symmetric quantization (for 8-bit)."""
        max_val = tensor.abs().amax().clamp(min=1e-8)
        scale = max_val / self._qmax
        quantized = (tensor / scale).round().clamp(self._qmin, self._qmax).to(torch.int8)
        return quantized, scale.unsqueeze(0)

    def _uniform_dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Fallback uniform dequantization."""
        return quantized.float() * scale

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """Compress: normalize -> rotate -> codebook quantize."""
        original_dtype = tensor.dtype
        original_shape = tensor.shape
        original_dim = tensor.shape[-1]
        t = tensor.float()

        # Step 1: Separate L2 norms along last dimension
        # Shape: (...,) — one norm per vector
        norms = torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=1e-12)
        t_normalized = t / norms
        # Squeeze keepdim for storage
        norms_flat = norms.squeeze(-1)

        # After normalization, each vector has unit norm.
        # The rotated coordinates will be ~ N(0, 1/sqrt(d)) by CLT.
        # Scale to standard normal for the codebook.
        d = original_dim
        scale_factor = math.sqrt(d)

        # Step 2: Apply rotation
        rotated = self._rotate(t_normalized)
        rotated_dim = rotated.shape[-1]

        # Scale rotated coordinates to ~ N(0, 1) for the codebook
        rotated_scaled = rotated * scale_factor

        # Step 3: Quantize
        if self._use_uniform_fallback:
            quantized, uniform_scale = self._uniform_quantize(rotated_scaled)
            return CompressedTensor(
                data=quantized,
                scales=uniform_scale,
                zero_points=norms_flat.to(torch.float32),
                original_shape=original_shape,
                original_dtype=original_dtype,
                bits=self._bits,
                metadata={
                    "rotation": self._rotation,
                    "seed": self._seed,
                    "original_dim": original_dim,
                    "rotated_dim": rotated_dim,
                    "scale_factor": scale_factor,
                    "uniform_fallback": True,
                },
            )

        indices = self._codebook_quantize(rotated_scaled)

        # Bit-pack codebook indices (already unsigned [0, 2^bits-1])
        indices_shape = indices.shape
        packed = pack(indices, bits=self._bits)

        return CompressedTensor(
            data=packed,
            scales=None,
            zero_points=norms_flat.to(torch.float32),
            original_shape=original_shape,
            original_dtype=original_dtype,
            bits=self._bits,
            metadata={
                "rotation": self._rotation,
                "seed": self._seed,
                "original_dim": original_dim,
                "rotated_dim": rotated_dim,
                "scale_factor": scale_factor,
                "uniform_fallback": False,
                "indices_shape": indices_shape,
                "num_values": indices.numel(),
            },
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress: dequantize -> inverse rotate -> rescale by norm."""
        meta = compressed.metadata
        original_dim = meta["original_dim"]
        scale_factor = meta["scale_factor"]
        norms = compressed.zero_points  # (...,) norms

        if meta["uniform_fallback"]:
            dequantized = self._uniform_dequantize(compressed.data, compressed.scales)
        else:
            # Unpack bit-packed indices
            indices = unpack(compressed.data, bits=compressed.bits, num_values=meta["num_values"])
            indices = indices.reshape(meta["indices_shape"]).to(torch.int16)
            dequantized = self._codebook_dequantize(indices, compressed.data.device)

        # Undo the scaling to standard normal
        dequantized = dequantized / scale_factor

        # Inverse rotation
        result = self._inverse_rotate(dequantized, original_dim)

        # Rescale by original norms
        result = result * norms.unsqueeze(-1)

        return result.to(compressed.original_dtype)

    def estimate_bits_per_value(self) -> float:
        # bits per coordinate + fp32 norm per vector (amortized over d)
        # For a typical d=128: 32/128 = 0.25 bits overhead
        return float(self._bits) + 0.25

    @property
    def name(self) -> str:
        return f"turboquant_mse_{self._rotation}_{self._bits}b"
