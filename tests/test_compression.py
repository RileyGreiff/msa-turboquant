"""Tests for compression modules: fp16, int8, int4."""

from __future__ import annotations

import pytest
import torch

from src.compression.base import BaseCompressor, CompressedTensor
from src.compression.fp16 import FP16Compressor
from src.compression.int4 import Int4Compressor
from src.compression.int8 import Int8Compressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kv_tensor(num_heads: int = 4, seq_len: int = 32, head_dim: int = 64) -> torch.Tensor:
    """Create a random tensor shaped like a KV cache layer."""
    return torch.randn(num_heads, seq_len, head_dim)


def _hidden_tensor(seq_len: int = 32, hidden_dim: int = 256) -> torch.Tensor:
    """Create a random tensor shaped like hidden states."""
    return torch.randn(seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# CompressedTensor tests
# ---------------------------------------------------------------------------

class TestCompressedTensor:
    """Tests for the CompressedTensor container."""

    def test_compressed_bytes(self) -> None:
        ct = CompressedTensor(
            data=torch.zeros(100, dtype=torch.int8),
            scales=torch.zeros(1, dtype=torch.float32),
            original_shape=(100,),
            original_dtype=torch.float16,
            bits=8,
        )
        assert ct.compressed_bytes == 100 * 1 + 1 * 4  # 100 int8 + 1 float32

    def test_original_bytes(self) -> None:
        ct = CompressedTensor(
            data=torch.zeros(100, dtype=torch.int8),
            original_shape=(100,),
            original_dtype=torch.float16,
            bits=8,
        )
        assert ct.original_bytes == 100 * 2  # 100 float16

    def test_compression_ratio(self) -> None:
        ct = CompressedTensor(
            data=torch.zeros(100, dtype=torch.int8),
            original_shape=(100,),
            original_dtype=torch.float32,
            bits=8,
        )
        # 400 bytes original / 100 bytes compressed = 4.0
        assert ct.compression_ratio == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# FP16 Compressor tests
# ---------------------------------------------------------------------------

class TestFP16Compressor:
    """Tests for FP16 passthrough."""

    def test_round_trip_exact(self) -> None:
        """FP16 from fp16 input should be lossless."""
        comp = FP16Compressor()
        t = torch.randn(4, 32, 64, dtype=torch.float16)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert torch.equal(t, recovered)

    def test_shape_preserved(self) -> None:
        comp = FP16Compressor()
        t = _kv_tensor()
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_bits_per_value(self) -> None:
        assert FP16Compressor().estimate_bits_per_value() == 16.0

    def test_name(self) -> None:
        assert FP16Compressor().name == "fp16"

    def test_reconstruction_error(self) -> None:
        comp = FP16Compressor()
        t = torch.randn(4, 32, 64, dtype=torch.float16)
        errors = comp.compute_reconstruction_error(t)
        assert errors["mse"] == 0.0
        assert errors["cosine_sim"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# INT8 Compressor tests
# ---------------------------------------------------------------------------

class TestInt8Compressor:
    """Tests for INT8 symmetric quantization."""

    def test_shape_preserved(self) -> None:
        comp = Int8Compressor()
        t = _kv_tensor()
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_dtype_preserved(self) -> None:
        comp = Int8Compressor()
        t = _kv_tensor().to(torch.float16)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.dtype == torch.float16

    def test_compressed_is_int8(self) -> None:
        comp = Int8Compressor()
        compressed = comp.compress(_kv_tensor())
        assert compressed.data.dtype == torch.int8
        assert compressed.bits == 8

    def test_values_in_range(self) -> None:
        comp = Int8Compressor()
        compressed = comp.compress(_kv_tensor())
        assert compressed.data.min() >= -127
        assert compressed.data.max() <= 127

    def test_low_reconstruction_error(self) -> None:
        comp = Int8Compressor()
        t = _kv_tensor()
        errors = comp.compute_reconstruction_error(t)
        assert errors["cosine_sim"] > 0.99
        assert errors["snr_db"] > 30  # int8 should have >30dB SNR

    def test_per_channel_mode(self) -> None:
        comp = Int8Compressor(per_channel=True)
        t = _kv_tensor(num_heads=4, seq_len=32, head_dim=64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape
        # Per-channel should have scale per head_dim column
        assert compressed.scales.shape[-1] == 64

    def test_bits_per_value(self) -> None:
        assert Int8Compressor().estimate_bits_per_value() == 8.0

    def test_name(self) -> None:
        assert Int8Compressor().name == "int8"
        assert Int8Compressor(per_channel=True).name == "int8_perchannel"

    def test_compression_ratio(self) -> None:
        comp = Int8Compressor()
        t = torch.randn(4, 32, 64, dtype=torch.float32)
        compressed = comp.compress(t)
        # Should be roughly 4x (float32 -> int8), minus scale overhead
        assert compressed.compression_ratio > 3.5

    def test_hidden_states(self) -> None:
        """Works on 2D hidden state tensors too."""
        comp = Int8Compressor()
        t = _hidden_tensor()
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_1d_tensor(self) -> None:
        comp = Int8Compressor()
        t = torch.randn(256)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape


# ---------------------------------------------------------------------------
# INT4 Compressor tests
# ---------------------------------------------------------------------------

class TestInt4Compressor:
    """Tests for INT4 groupwise quantization."""

    def test_shape_preserved(self) -> None:
        comp = Int4Compressor(group_size=32)
        t = _kv_tensor(num_heads=4, seq_len=32, head_dim=64)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_dtype_preserved(self) -> None:
        comp = Int4Compressor(group_size=32)
        t = _kv_tensor().to(torch.float16)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.dtype == torch.float16

    def test_data_is_bitpacked(self) -> None:
        comp = Int4Compressor(group_size=32)
        compressed = comp.compress(_kv_tensor())
        assert compressed.data.dtype == torch.uint8
        assert compressed.bits == 4
        # 2 values packed per byte
        expected_bytes = compressed.metadata["num_values"] // 2
        assert compressed.data.numel() == expected_bytes

    def test_reconstruction_quality(self) -> None:
        comp = Int4Compressor(group_size=32)
        t = _kv_tensor()
        errors = comp.compute_reconstruction_error(t)
        # INT4 has more error than INT8 but should still preserve structure
        assert errors["cosine_sim"] > 0.95
        assert errors["snr_db"] > 15

    def test_small_group_size_better_quality(self) -> None:
        """Smaller groups = more scales = better reconstruction."""
        t = _kv_tensor(num_heads=4, seq_len=32, head_dim=128)
        comp_small = Int4Compressor(group_size=32)
        comp_large = Int4Compressor(group_size=128)
        err_small = comp_small.compute_reconstruction_error(t)
        err_large = comp_large.compute_reconstruction_error(t)
        assert err_small["mse"] <= err_large["mse"]

    def test_handles_non_divisible_dim(self) -> None:
        """Padding handles last dims not divisible by group_size."""
        comp = Int4Compressor(group_size=32)
        t = torch.randn(4, 16, 50)  # 50 not divisible by 32
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape

    def test_bits_per_value(self) -> None:
        comp = Int4Compressor(group_size=128)
        bpv = comp.estimate_bits_per_value()
        assert 4.0 < bpv < 5.0  # 4 + overhead

    def test_name(self) -> None:
        assert Int4Compressor(group_size=128).name == "int4_g128"
        assert Int4Compressor(group_size=32).name == "int4_g32"

    def test_compression_ratio_vs_fp16(self) -> None:
        comp_fp16 = FP16Compressor()
        comp_int4 = Int4Compressor(group_size=128)
        t = torch.randn(4, 32, 128, dtype=torch.float16)

        cr_fp16 = comp_fp16.compress(t).compression_ratio
        cr_int4 = comp_int4.compress(t).compression_ratio
        assert cr_int4 > cr_fp16  # int4 should compress more

    def test_hidden_states(self) -> None:
        comp = Int4Compressor(group_size=64)
        t = _hidden_tensor(seq_len=32, hidden_dim=256)
        compressed = comp.compress(t)
        recovered = comp.decompress(compressed)
        assert recovered.shape == t.shape


# ---------------------------------------------------------------------------
# Cross-compressor comparison
# ---------------------------------------------------------------------------

class TestCompressorComparison:
    """Compare all compressors on the same tensor."""

    def test_error_ordering(self) -> None:
        """FP16 <= INT8 <= INT4 in terms of reconstruction error."""
        t = _kv_tensor(num_heads=4, seq_len=32, head_dim=128)
        compressors = [
            FP16Compressor(),
            Int8Compressor(),
            Int4Compressor(group_size=32),
        ]
        errors = [c.compute_reconstruction_error(t) for c in compressors]
        mses = [e["mse"] for e in errors]
        # FP16 should have 0 or near-0 MSE, INT8 less than INT4
        assert mses[0] <= mses[1] <= mses[2]

    def test_bits_per_value_ordering(self) -> None:
        """INT4 < INT8 < FP16 in bits per value."""
        compressors = [
            FP16Compressor(),
            Int8Compressor(),
            Int4Compressor(group_size=128),
        ]
        bpv = [c.estimate_bits_per_value() for c in compressors]
        # fp16=16, int8=8, int4~4.25
        assert bpv[0] > bpv[1] > bpv[2]

    def test_all_preserve_shape(self) -> None:
        """All compressors preserve tensor shape through round-trip."""
        t = _kv_tensor()
        for comp in [FP16Compressor(), Int8Compressor(), Int4Compressor(group_size=32)]:
            recovered = comp.decompress(comp.compress(t))
            assert recovered.shape == t.shape, f"{comp.name} changed shape"
