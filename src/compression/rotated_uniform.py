"""Rotated uniform compressor: rotation + groupwise scalar quantization.

This is a baseline compressor that applies a random orthogonal rotation
(or signed Hadamard transform) before groupwise uniform quantization.
The rotation spreads outlier magnitudes more evenly across dimensions,
reducing quantization error for the same bit budget.

This is NOT the TurboQuant algorithm from the paper. For that, see
turboquant_mse.py which uses Lloyd-Max codebooks on the rotated
Gaussian distribution.

Pipeline:
    Compress: tensor -> rotate -> groupwise scalar quantize -> store
    Decompress: load -> dequantize -> inverse rotate -> output

Design notes:
- The rotation is applied along the last dimension (head_dim for KV).
- For the Hadamard variant, we use random sign flips + fast Walsh-Hadamard
  transform (signed Hadamard), matching the paper's practical rotation.
- The rotation matrix is fixed and deterministic (seeded).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch

from src.compression.base import BaseCompressor, CompressedTensor

logger = logging.getLogger("msa_turboquant.compression.rotated_uniform")


def _generate_random_orthogonal(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a deterministic random orthogonal matrix via QR decomposition.

    Args:
        dim: Matrix dimension (dim x dim).
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix Q of shape (dim, dim) where Q @ Q.T = I.
    """
    gen = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(dim, dim, generator=gen)
    q, r = torch.linalg.qr(random_matrix)
    # Ensure deterministic sign (make diagonal of R positive)
    d = torch.diag(r)
    sign = torch.sign(d)
    sign[sign == 0] = 1.0
    q = q * sign.unsqueeze(0)
    return q


def _generate_random_signs(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a deterministic random ±1 sign vector.

    Used for signed Hadamard: multiply by random signs before WHT
    to break structured basis vectors.

    Args:
        dim: Vector length.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (dim,) with values in {-1, +1}.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1


def _fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply the Walsh-Hadamard transform along the last dimension.

    Requires the last dimension to be a power of 2. If not, the caller
    should pad first. The transform is its own inverse (up to a scaling
    factor of 1/sqrt(d)).

    Runs in O(d log d) time and O(1) extra memory.

    Args:
        x: Input tensor of shape (..., d) where d is a power of 2.

    Returns:
        Transformed tensor of the same shape, normalized by 1/sqrt(d).
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Last dim must be power of 2, got {d}"

    result = x.clone()
    h = 1
    while h < d:
        # Process pairs of elements at distance h
        half_size = h
        step = h * 2
        for i in range(0, d, step):
            j_range = slice(i, i + half_size)
            k_range = slice(i + half_size, i + step)
            a = result[..., j_range].clone()
            b = result[..., k_range].clone()
            result[..., j_range] = a + b
            result[..., k_range] = a - b
        h *= 2

    # Normalize
    result = result / math.sqrt(d)
    return result


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


class RotatedUniformCompressor(BaseCompressor):
    """Rotation + groupwise uniform quantization baseline.

    Applies a random orthogonal rotation or signed Hadamard transform
    before standard groupwise symmetric quantization.

    Args:
        bits: Quantization bits (typically 4 or 8).
        group_size: Number of values per quantization group.
        rotation: Rotation type — "random_orthogonal" or "hadamard".
        seed: Random seed for the rotation matrix / sign flips.
        residual_correction: If True, quantize the residual error for a second
            pass (NOT YET IMPLEMENTED — flag for future work).
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        rotation: Literal["random_orthogonal", "hadamard"] = "random_orthogonal",
        seed: int = 42,
        residual_correction: bool = False,
    ) -> None:
        self._bits = bits
        self._group_size = group_size
        self._rotation = rotation
        self._seed = seed
        self._residual_correction = residual_correction

        # Quantization range for symmetric quantization
        self._qmax = (1 << (bits - 1)) - 1  # e.g., 7 for 4-bit, 127 for 8-bit
        self._qmin = -self._qmax

        # Cache rotation matrices and sign vectors per dimension
        self._rotation_cache: dict[int, torch.Tensor] = {}
        self._sign_cache: dict[int, torch.Tensor] = {}

        if residual_correction:
            logger.warning(
                "residual_correction=True is flagged but NOT YET IMPLEMENTED. "
                "This will be a no-op until a future milestone."
            )

    def _get_rotation_matrix(self, dim: int, device: torch.device) -> torch.Tensor:
        """Get or create a rotation matrix for the given dimension."""
        if dim not in self._rotation_cache:
            if self._rotation == "random_orthogonal":
                self._rotation_cache[dim] = _generate_random_orthogonal(dim, self._seed)
            elif self._rotation == "hadamard":
                self._rotation_cache[dim] = None  # type: ignore
            else:
                raise ValueError(f"Unknown rotation type: {self._rotation}")

        mat = self._rotation_cache[dim]
        if mat is not None:
            return mat.to(device)
        return mat  # type: ignore  # None for hadamard

    def _get_sign_vector(self, dim: int, device: torch.device) -> torch.Tensor:
        """Get or create the random sign vector for signed Hadamard."""
        if dim not in self._sign_cache:
            self._sign_cache[dim] = _generate_random_signs(dim, self._seed)
        return self._sign_cache[dim].to(device)

    def _rotate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply rotation to the last dimension."""
        dim = tensor.shape[-1]

        if self._rotation == "hadamard":
            # Pad to power of 2 if needed
            p2 = _next_power_of_2(dim)
            if dim != p2:
                tensor = torch.nn.functional.pad(tensor, (0, p2 - dim))
            # Signed Hadamard: random sign flips then WHT
            signs = self._get_sign_vector(tensor.shape[-1], tensor.device)
            tensor = tensor * signs
            return _fast_walsh_hadamard_transform(tensor)
        else:
            rot = self._get_rotation_matrix(dim, tensor.device)
            return tensor @ rot.to(tensor.dtype)

    def _inverse_rotate(self, tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
        """Apply inverse rotation to the last dimension."""
        if self._rotation == "hadamard":
            # Inverse: WHT then undo sign flips (signs are self-inverse)
            result = _fast_walsh_hadamard_transform(tensor)
            signs = self._get_sign_vector(tensor.shape[-1], tensor.device)
            result = result * signs
            # Remove padding
            if result.shape[-1] != original_dim:
                result = result[..., :original_dim]
            return result
        else:
            rot = self._get_rotation_matrix(original_dim, tensor.device)
            return tensor @ rot.to(tensor.dtype).T

    def _groupwise_quantize(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Groupwise symmetric scalar quantization.

        Returns:
            (quantized_int, scales) where quantized is int8 and scales are float32.
        """
        last_dim = tensor.shape[-1]
        gs = self._group_size

        # Pad if needed
        pad_amount = (gs - last_dim % gs) % gs
        if pad_amount > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_amount))

        leading_shape = tensor.shape[:-1]
        padded_dim = tensor.shape[-1]
        num_groups = padded_dim // gs

        grouped = tensor.reshape(*leading_shape, num_groups, gs)
        max_vals = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = max_vals / self._qmax

        quantized = (grouped / scales).round().clamp(self._qmin, self._qmax).to(torch.int8)
        quantized_flat = quantized.reshape(*leading_shape, padded_dim)
        scales_flat = scales.squeeze(-1)  # (..., num_groups)

        return quantized_flat, scales_flat

    def _groupwise_dequantize(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        pad_amount: int,
    ) -> torch.Tensor:
        """Reverse groupwise quantization."""
        gs = self._group_size
        leading_shape = quantized.shape[:-1]
        padded_dim = quantized.shape[-1]
        num_groups = padded_dim // gs

        grouped = quantized.float().reshape(*leading_shape, num_groups, gs)
        scales_expanded = scales.unsqueeze(-1)
        dequantized = grouped * scales_expanded
        dequantized = dequantized.reshape(*leading_shape, padded_dim)

        if pad_amount > 0:
            dequantized = dequantized[..., :padded_dim - pad_amount]

        return dequantized

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """Compress: rotate -> quantize."""
        original_dtype = tensor.dtype
        original_shape = tensor.shape
        original_dim = tensor.shape[-1]
        t = tensor.float()

        # Step 1: Apply rotation
        rotated = self._rotate(t)
        rotated_dim = rotated.shape[-1]  # May differ from original if Hadamard padded

        # Step 2: Groupwise scalar quantization
        pad_amount = (self._group_size - rotated_dim % self._group_size) % self._group_size
        quantized, scales = self._groupwise_quantize(rotated)

        return CompressedTensor(
            data=quantized,
            scales=scales,
            original_shape=original_shape,
            original_dtype=original_dtype,
            bits=self._bits,
            metadata={
                "rotation": self._rotation,
                "seed": self._seed,
                "group_size": self._group_size,
                "pad_amount": pad_amount,
                "original_dim": original_dim,
                "rotated_dim": rotated_dim,
            },
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress: dequantize -> inverse rotate."""
        meta = compressed.metadata
        pad_amount = meta["pad_amount"]
        original_dim = meta["original_dim"]

        # Step 1: Dequantize
        dequantized = self._groupwise_dequantize(
            compressed.data, compressed.scales, pad_amount
        )

        # Step 2: Inverse rotation
        result = self._inverse_rotate(dequantized, original_dim)

        return result.to(compressed.original_dtype)

    def estimate_bits_per_value(self) -> float:
        overhead = 32.0 / self._group_size  # fp32 scale per group
        return float(self._bits) + overhead

    @property
    def name(self) -> str:
        return f"rotated_uniform_{self._rotation}_{self._bits}b_g{self._group_size}"

    def compute_dot_product_error(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
    ) -> dict[str, float]:
        """Measure how compression affects dot-product (attention) scores.

        Compresses keys, decompresses, then compares Q @ K^T with Q @ K_hat^T.

        Args:
            keys: Key tensor, shape (..., seq_len, dim).
            queries: Query tensor, same last dim as keys.

        Returns:
            Dict with dot-product MSE, max error, and rank agreement metrics.
        """
        compressed = self.compress(keys)
        keys_hat = self.decompress(compressed)

        q = queries.float()
        k_orig = keys.float()
        k_recon = keys_hat.float()

        scores_orig = q @ k_orig.transpose(-2, -1)
        scores_recon = q @ k_recon.transpose(-2, -1)

        diff = scores_orig - scores_recon
        mse = (diff ** 2).mean().item()
        max_err = diff.abs().max().item()

        k = min(5, scores_orig.shape[-1])
        _, orig_topk = scores_orig.topk(k, dim=-1)
        _, recon_topk = scores_recon.topk(k, dim=-1)

        matches = 0
        total = orig_topk.numel()
        for i in range(orig_topk.shape[-2] if orig_topk.dim() > 1 else 1):
            if orig_topk.dim() > 1:
                orig_set = set(orig_topk[..., i, :].flatten().tolist())
                recon_set = set(recon_topk[..., i, :].flatten().tolist())
            else:
                orig_set = set(orig_topk.flatten().tolist())
                recon_set = set(recon_topk.flatten().tolist())
            matches += len(orig_set & recon_set)
        rank_agreement = matches / total if total > 0 else 1.0

        return {
            "dot_product_mse": mse,
            "dot_product_max_error": max_err,
            "rank_agreement_top5": rank_agreement,
        }


# Backwards compatibility alias
TurboQuantLikeCompressor = RotatedUniformCompressor
