"""Tests for compression: RotatedUniform baseline, TurboQuantMSE, and factory."""

from __future__ import annotations

import math

import pytest
import torch

from src.compression import (
    create_compressor,
    create_compressor_from_config,
)
from src.compression.base import CompressedTensor
from src.compression.fp16 import FP16Compressor
from src.compression.int4 import Int4Compressor
from src.compression.int8 import Int8Compressor
from src.compression.rotated_uniform import (
    RotatedUniformCompressor,
    _fast_walsh_hadamard_transform,
    _generate_random_orthogonal,
    _generate_random_signs,
)
from src.compression.turboquant_mse import TurboQuantMSECompressor


# ---------------------------------------------------------------------------
# Rotation matrix tests
# ---------------------------------------------------------------------------

class TestRotationMatrix:
    """Tests for the random orthogonal matrix generation."""

    def test_orthogonality(self) -> None:
        """Q @ Q.T should be identity."""
        q = _generate_random_orthogonal(64, seed=42)
        eye = q @ q.T
        assert torch.allclose(eye, torch.eye(64), atol=1e-5)

    def test_deterministic(self) -> None:
        """Same seed produces same matrix."""
        q1 = _generate_random_orthogonal(32, seed=123)
        q2 = _generate_random_orthogonal(32, seed=123)
        assert torch.equal(q1, q2)

    def test_different_seeds(self) -> None:
        """Different seeds produce different matrices."""
        q1 = _generate_random_orthogonal(32, seed=1)
        q2 = _generate_random_orthogonal(32, seed=2)
        assert not torch.equal(q1, q2)


class TestWalshHadamard:
    """Tests for the fast Walsh-Hadamard transform."""

    def test_self_inverse(self) -> None:
        """WHT applied twice should return the original (it's involutory)."""
        x = torch.randn(4, 16)
        transformed = _fast_walsh_hadamard_transform(x)
        recovered = _fast_walsh_hadamard_transform(transformed)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_orthogonality(self) -> None:
        """WHT preserves vector norms (up to numerical precision)."""
        x = torch.randn(8, 32)
        y = _fast_walsh_hadamard_transform(x)
        orig_norms = torch.linalg.norm(x, dim=-1)
        trans_norms = torch.linalg.norm(y, dim=-1)
        assert torch.allclose(orig_norms, trans_norms, atol=1e-4)

    def test_requires_power_of_2(self) -> None:
        with pytest.raises(AssertionError):
            _fast_walsh_hadamard_transform(torch.randn(3, 10))


class TestSignedHadamard:
    """Tests for the signed Hadamard (random sign flips + WHT)."""

    def test_sign_vector_deterministic(self) -> None:
        s1 = _generate_random_signs(64, seed=42)
        s2 = _generate_random_signs(64, seed=42)
        assert torch.equal(s1, s2)

    def test_sign_vector_values(self) -> None:
        """Sign vector should contain only -1 and +1."""
        s = _generate_random_signs(128, seed=42)
        assert set(s.tolist()).issubset({-1.0, 1.0})

    def test_signed_hadamard_round_trip(self) -> None:
        """Signed Hadamard rotation should be invertible."""
        comp = RotatedUniformCompressor(bits=4, group_size=32, rotation="hadamard", seed=42)
        t = torch.randn(4, 32, 64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape
        # Should have reasonable reconstruction quality
        errors = comp.compute_reconstruction_error(t)
        assert errors["cosine_sim"] > 0.85


# ---------------------------------------------------------------------------
# RotatedUniformCompressor tests (baseline)
# ---------------------------------------------------------------------------

class TestRotatedUniformCompressor:
    """Tests for the rotation + groupwise uniform quantization baseline."""

    def test_round_trip_shape(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32, rotation="random_orthogonal")
        t = torch.randn(4, 32, 64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_round_trip_hadamard(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32, rotation="hadamard")
        t = torch.randn(4, 32, 64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_hadamard_non_power_of_2(self) -> None:
        """Hadamard with non-power-of-2 dim should still work (via padding)."""
        comp = RotatedUniformCompressor(bits=4, group_size=32, rotation="hadamard")
        t = torch.randn(4, 16, 50)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_dtype_preserved(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32)
        t = torch.randn(4, 16, 64, dtype=torch.float16)
        recovered = comp.decompress(comp.compress(t))
        assert recovered.dtype == torch.float16

    def test_compressed_is_int8(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32)
        compressed = comp.compress(torch.randn(4, 16, 64))
        assert compressed.data.dtype == torch.int8
        assert compressed.bits == 4

    def test_values_in_4bit_range(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32)
        compressed = comp.compress(torch.randn(4, 16, 64))
        assert compressed.data.min() >= -7
        assert compressed.data.max() <= 7

    def test_reconstruction_quality(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32, rotation="random_orthogonal")
        t = torch.randn(4, 32, 128)
        errors = comp.compute_reconstruction_error(t)
        assert errors["cosine_sim"] > 0.90
        assert errors["snr_db"] > 10

    def test_name(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=128, rotation="random_orthogonal")
        assert "rotated_uniform" in comp.name
        assert "4b" in comp.name
        assert "g128" in comp.name

    def test_bits_per_value(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=128)
        bpv = comp.estimate_bits_per_value()
        assert 4.0 < bpv < 5.0

    def test_deterministic_compression(self) -> None:
        t = torch.randn(4, 16, 64)
        comp1 = RotatedUniformCompressor(bits=4, group_size=32, seed=42)
        comp2 = RotatedUniformCompressor(bits=4, group_size=32, seed=42)
        c1 = comp1.compress(t)
        c2 = comp2.compress(t)
        assert torch.equal(c1.data, c2.data)

    def test_different_seeds_different_output(self) -> None:
        t = torch.randn(4, 16, 64)
        comp1 = RotatedUniformCompressor(bits=4, group_size=32, seed=1)
        comp2 = RotatedUniformCompressor(bits=4, group_size=32, seed=2)
        c1 = comp1.compress(t)
        c2 = comp2.compress(t)
        assert not torch.equal(c1.data, c2.data)


# ---------------------------------------------------------------------------
# TurboQuantMSE compressor tests
# ---------------------------------------------------------------------------

class TestTurboQuantMSECompressor:
    """Tests for the TurboQuant MSE compressor with Lloyd-Max codebooks."""

    def test_round_trip_shape(self) -> None:
        comp = TurboQuantMSECompressor(bits=4, rotation="random_orthogonal")
        t = torch.randn(4, 32, 128)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_round_trip_hadamard(self) -> None:
        comp = TurboQuantMSECompressor(bits=4, rotation="hadamard")
        t = torch.randn(4, 32, 64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_hadamard_non_power_of_2(self) -> None:
        comp = TurboQuantMSECompressor(bits=4, rotation="hadamard")
        t = torch.randn(4, 16, 50)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_dtype_preserved(self) -> None:
        comp = TurboQuantMSECompressor(bits=4)
        t = torch.randn(4, 16, 64, dtype=torch.float16)
        recovered = comp.decompress(comp.compress(t))
        assert recovered.dtype == torch.float16

    def test_compressed_data_is_bitpacked(self) -> None:
        """Codebook indices should be bit-packed into uint8."""
        comp = TurboQuantMSECompressor(bits=4)
        compressed = comp.compress(torch.randn(4, 16, 64))
        assert compressed.data.dtype == torch.uint8
        assert compressed.bits == 4
        # 4096 values at 4 bits = 2048 bytes
        assert compressed.data.numel() == (4 * 16 * 64) // 2

    def test_indices_roundtrip_in_codebook_range(self) -> None:
        """Unpacked indices should be in range [0, 2^bits - 1]."""
        from src.compression.bitpack import unpack
        comp = TurboQuantMSECompressor(bits=4)
        compressed = comp.compress(torch.randn(4, 16, 64))
        indices = unpack(compressed.data, bits=4, num_values=compressed.metadata["num_values"])
        assert indices.min() >= 0
        assert indices.max() <= 15  # 2^4 - 1

    def test_norms_stored_in_zero_points(self) -> None:
        """Norms should be stored in zero_points field."""
        comp = TurboQuantMSECompressor(bits=4)
        t = torch.randn(4, 16, 64)
        compressed = comp.compress(t)
        assert compressed.zero_points is not None
        # Norms should be shape (...) = (4, 16) — one per vector
        assert compressed.zero_points.shape == (4, 16)
        # Norms should be positive
        assert (compressed.zero_points > 0).all()

    def test_no_scales_for_codebook_path(self) -> None:
        """Codebook path (2-5 bits) should not use scales."""
        comp = TurboQuantMSECompressor(bits=4)
        compressed = comp.compress(torch.randn(4, 16, 64))
        assert compressed.scales is None

    def test_reconstruction_quality_4bit(self) -> None:
        comp = TurboQuantMSECompressor(bits=4, rotation="random_orthogonal")
        t = torch.randn(4, 32, 128)
        errors = comp.compute_reconstruction_error(t)
        assert errors["cosine_sim"] > 0.85
        assert errors["snr_db"] > 8

    def test_reconstruction_quality_improves_with_bits(self) -> None:
        t = torch.randn(4, 32, 128)
        comp2 = TurboQuantMSECompressor(bits=2)
        comp4 = TurboQuantMSECompressor(bits=4)
        err2 = comp2.compute_reconstruction_error(t)
        err4 = comp4.compute_reconstruction_error(t)
        assert err4["cosine_sim"] > err2["cosine_sim"]
        assert err4["mse"] < err2["mse"]

    def test_8bit_uniform_fallback(self) -> None:
        """8-bit should use uniform fallback and still work."""
        comp = TurboQuantMSECompressor(bits=8)
        t = torch.randn(4, 16, 64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape
        # 8-bit should be very high quality
        errors = comp.compute_reconstruction_error(t)
        assert errors["cosine_sim"] > 0.99

    def test_name(self) -> None:
        comp = TurboQuantMSECompressor(bits=4, rotation="random_orthogonal")
        assert "turboquant_mse" in comp.name
        assert "4b" in comp.name

    def test_bits_per_value(self) -> None:
        comp = TurboQuantMSECompressor(bits=4)
        bpv = comp.estimate_bits_per_value()
        assert 4.0 < bpv < 5.0

    def test_deterministic_compression(self) -> None:
        t = torch.randn(4, 16, 64)
        comp1 = TurboQuantMSECompressor(bits=4, seed=42)
        comp2 = TurboQuantMSECompressor(bits=4, seed=42)
        c1 = comp1.compress(t)
        c2 = comp2.compress(t)
        assert torch.equal(c1.data, c2.data)

    def test_different_seeds_different_output(self) -> None:
        t = torch.randn(4, 16, 64)
        comp1 = TurboQuantMSECompressor(bits=4, seed=1)
        comp2 = TurboQuantMSECompressor(bits=4, seed=2)
        c1 = comp1.compress(t)
        c2 = comp2.compress(t)
        assert not torch.equal(c1.data, c2.data)

    def test_compressed_bytes_structure(self) -> None:
        """Compressed data should be bit-packed uint8 with norms, no scales."""
        comp = TurboQuantMSECompressor(bits=4)
        t = torch.randn(4, 16, 128, dtype=torch.float16)
        compressed = comp.compress(t)
        # Indices bit-packed into uint8
        assert compressed.data.dtype == torch.uint8
        # Should be roughly half the element count (2 values per byte for 4-bit)
        assert compressed.data.numel() == (4 * 16 * 128) // 2
        # Norms stored as fp32
        assert compressed.zero_points is not None
        assert compressed.zero_points.dtype == torch.float32
        # No scales for codebook path
        assert compressed.scales is None
        # Compression ratio should now reflect actual 4-bit packing
        original_bytes = 4 * 16 * 128 * 2  # fp16
        packed_bytes = compressed.data.numel() + compressed.zero_points.numel() * 4
        assert compressed.compressed_bytes == packed_bytes
        assert original_bytes / packed_bytes > 3.5

    def test_all_supported_bit_widths(self) -> None:
        """Verify all Lloyd-Max bit widths work."""
        t = torch.randn(4, 16, 64)
        for bits in [2, 3, 4, 5]:
            comp = TurboQuantMSECompressor(bits=bits)
            recovered = comp.decompress(comp.compress(t))
            assert recovered.shape == t.shape

    def test_unsupported_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="No Lloyd-Max codebook"):
            TurboQuantMSECompressor(bits=6)


# ---------------------------------------------------------------------------
# Dot product error tests
# ---------------------------------------------------------------------------

class TestDotProductError:
    """Tests for dot-product distortion measurement."""

    def test_returns_expected_keys(self) -> None:
        comp = RotatedUniformCompressor(bits=4, group_size=32)
        keys = torch.randn(4, 32, 64)
        queries = torch.randn(4, 8, 64)
        result = comp.compute_dot_product_error(keys, queries)
        assert "dot_product_mse" in result
        assert "dot_product_max_error" in result
        assert "rank_agreement_top5" in result

    def test_higher_bits_lower_error(self) -> None:
        keys = torch.randn(4, 32, 64)
        queries = torch.randn(4, 8, 64)
        comp4 = RotatedUniformCompressor(bits=4, group_size=32)
        comp8 = RotatedUniformCompressor(bits=8, group_size=32)
        err4 = comp4.compute_dot_product_error(keys, queries)
        err8 = comp8.compute_dot_product_error(keys, queries)
        assert err8["dot_product_mse"] < err4["dot_product_mse"]


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCompressorFactory:
    """Tests for the compression factory functions."""

    def test_create_fp16(self) -> None:
        assert isinstance(create_compressor("fp16"), FP16Compressor)

    def test_create_none(self) -> None:
        assert isinstance(create_compressor("none"), FP16Compressor)

    def test_create_int8(self) -> None:
        assert isinstance(create_compressor("int8"), Int8Compressor)

    def test_create_int4(self) -> None:
        comp = create_compressor("int4", group_size=64)
        assert isinstance(comp, Int4Compressor)

    def test_create_turboquant_mse(self) -> None:
        comp = create_compressor("turboquant_mse", bits=4, rotation="hadamard")
        assert isinstance(comp, TurboQuantMSECompressor)

    def test_create_turboquant_mse_3b(self) -> None:
        comp = create_compressor("turboquant_mse_3b")
        assert isinstance(comp, TurboQuantMSECompressor)
        assert "3b" in comp.name

    def test_create_turboquant_mse_2b(self) -> None:
        comp = create_compressor("turboquant_mse_2b")
        assert isinstance(comp, TurboQuantMSECompressor)
        assert "2b" in comp.name

    def test_create_turboquant_mse_5b(self) -> None:
        comp = create_compressor("turboquant_mse_5b")
        assert isinstance(comp, TurboQuantMSECompressor)
        assert "5b" in comp.name

    def test_parameterized_bits_override_kwargs(self) -> None:
        # String-parsed bits should be used, not default
        comp = create_compressor("turboquant_mse_2b")
        assert comp.estimate_bits_per_value() == pytest.approx(2.25, abs=0.01)

    def test_create_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            create_compressor("invalid_method")

    def test_from_config(self) -> None:
        from src.utils.config import CompressionConfig
        config = CompressionConfig(method="int4")
        comp = create_compressor_from_config(config)
        assert isinstance(comp, Int4Compressor)

    def test_from_config_turboquant_mse(self) -> None:
        from src.utils.config import CompressionConfig, TurboQuantMSEConfig
        config = CompressionConfig(
            method="turboquant_mse",
            turboquant_mse=TurboQuantMSEConfig(bits=4, rotation="hadamard"),
        )
        comp = create_compressor_from_config(config)
        assert isinstance(comp, TurboQuantMSECompressor)
        assert "hadamard" in comp.name
