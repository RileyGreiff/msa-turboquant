"""Tests for the systems profiler and compression wiring in eval harness."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from src.compression.fp16 import FP16Compressor
from src.compression.int4 import Int4Compressor
from src.compression.int8 import Int8Compressor
from src.compression.rotated_uniform import RotatedUniformCompressor
from src.eval.profiler import (
    MemorySnapshot,
    PhaseRecord,
    ProfilingReport,
    RunProfiler,
    _take_memory_snapshot,
)
from src.eval.run_eval import EvalHarness, EvalSampleResult, score_answer
from src.eval.niah import generate_niah_sample

# Reuse DummyModel from test_models
from tests.test_models import DummyModel


# ---------------------------------------------------------------------------
# MemorySnapshot tests
# ---------------------------------------------------------------------------

class TestMemorySnapshot:

    def test_to_dict(self) -> None:
        snap = MemorySnapshot(ram_used_mb=1024.123, ram_available_mb=512.789)
        d = snap.to_dict()
        assert d["ram_used_mb"] == 1024.1
        assert d["ram_available_mb"] == 512.8

    def test_take_snapshot_returns_snapshot(self) -> None:
        snap = _take_memory_snapshot()
        assert isinstance(snap, MemorySnapshot)
        assert snap.ram_used_mb > 0
        assert snap.ram_available_mb > 0


# ---------------------------------------------------------------------------
# PhaseRecord tests
# ---------------------------------------------------------------------------

class TestPhaseRecord:

    def test_gpu_delta(self) -> None:
        rec = PhaseRecord(
            name="test",
            before=MemorySnapshot(gpu_allocated_mb=100),
            after=MemorySnapshot(gpu_allocated_mb=150),
        )
        assert rec.gpu_delta_mb == 50.0

    def test_ram_delta(self) -> None:
        rec = PhaseRecord(
            name="test",
            before=MemorySnapshot(ram_used_mb=1000),
            after=MemorySnapshot(ram_used_mb=1050),
        )
        assert rec.ram_delta_mb == 50.0

    def test_to_dict_has_expected_keys(self) -> None:
        rec = PhaseRecord(name="route", wall_time_ms=12.3)
        d = rec.to_dict()
        assert d["name"] == "route"
        assert d["wall_time_ms"] == 12.3
        assert "gpu_delta_mb" in d
        assert "before" in d
        assert "after" in d

    def test_counters_passed_through(self) -> None:
        rec = PhaseRecord(name="fetch", counters={"num_fetched": 5})
        assert rec.to_dict()["counters"]["num_fetched"] == 5


# ---------------------------------------------------------------------------
# ProfilingReport tests
# ---------------------------------------------------------------------------

class TestProfilingReport:

    def _make_report(self) -> ProfilingReport:
        return ProfilingReport(
            run_id="test_run",
            phases=[
                PhaseRecord(name="bank_build", wall_time_ms=100),
                PhaseRecord(name="route", wall_time_ms=20),
                PhaseRecord(name="generate", wall_time_ms=300),
            ],
            total_wall_time_ms=500,
            peak_gpu_mb=2048,
            peak_ram_used_mb=8192,
            total_bytes_fetched=1024000,
            total_tokens_generated=50,
            tokens_per_second=100.0,
            compression_info={"method": "int4_g128", "bits_per_value": 4.25},
        )

    def test_phase_time_found(self) -> None:
        report = self._make_report()
        assert report.phase_time("route") == 20.0

    def test_phase_time_not_found(self) -> None:
        report = self._make_report()
        assert report.phase_time("nonexistent") == 0.0

    def test_phase_times_dict(self) -> None:
        report = self._make_report()
        pt = report.phase_times()
        assert pt["bank_build"] == 100.0
        assert pt["generate"] == 300.0

    def test_to_dict(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert d["run_id"] == "test_run"
        assert d["total_wall_time_ms"] == 500.0
        assert d["peak_gpu_mb"] == 2048.0
        assert d["total_bytes_fetched"] == 1024000
        assert d["tokens_per_second"] == 100.0
        assert len(d["phases"]) == 3
        assert d["compression_info"]["method"] == "int4_g128"

    def test_summary_lines(self) -> None:
        report = self._make_report()
        lines = report.summary_lines()
        assert any("test_run" in line for line in lines)
        assert any("bank_build" in line for line in lines)
        assert any("Compression" in line for line in lines)

    def test_save(self, tmp_path: Path) -> None:
        report = self._make_report()
        path = report.save(tmp_path / "profile.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_run"


# ---------------------------------------------------------------------------
# RunProfiler tests
# ---------------------------------------------------------------------------

class TestRunProfiler:

    def test_phase_timing(self) -> None:
        profiler = RunProfiler(run_id="r1")
        profiler.start()
        with profiler.phase("compute"):
            time.sleep(0.01)
        report = profiler.report()
        assert len(report.phases) == 1
        assert report.phases[0].name == "compute"
        assert report.phases[0].wall_time_ms >= 5  # at least 5ms

    def test_multiple_phases(self) -> None:
        profiler = RunProfiler(run_id="r2")
        profiler.start()
        with profiler.phase("a"):
            time.sleep(0.005)
        with profiler.phase("b"):
            time.sleep(0.005)
        report = profiler.report()
        assert len(report.phases) == 2
        assert report.phases[0].name == "a"
        assert report.phases[1].name == "b"

    def test_disabled_profiler_no_phases(self) -> None:
        profiler = RunProfiler(enabled=False)
        profiler.start()
        with profiler.phase("work"):
            time.sleep(0.005)
        report = profiler.report()
        assert len(report.phases) == 0

    def test_bytes_and_tokens_tracking(self) -> None:
        profiler = RunProfiler(run_id="r3")
        profiler.start()
        profiler.add_bytes_fetched(5000)
        profiler.add_bytes_fetched(3000)
        profiler.add_tokens(20)
        profiler.add_generation_time(100.0)
        report = profiler.report()
        assert report.total_bytes_fetched == 8000
        assert report.total_tokens_generated == 20
        assert report.tokens_per_second == pytest.approx(200.0, rel=0.01)

    def test_compression_info(self) -> None:
        profiler = RunProfiler(run_id="r4")
        profiler.start()
        profiler.set_compression_info({"method": "int8", "bits": 8})
        report = profiler.report()
        assert report.compression_info["method"] == "int8"

    def test_extra_metadata(self) -> None:
        profiler = RunProfiler(run_id="r5")
        profiler.start()
        profiler.add_extra("model_name", "qwen2.5-3b")
        report = profiler.report()
        assert report.extra["model_name"] == "qwen2.5-3b"

    def test_total_wall_time(self) -> None:
        profiler = RunProfiler(run_id="r6")
        profiler.start()
        time.sleep(0.02)
        report = profiler.report()
        assert report.total_wall_time_ms >= 15

    def test_phase_counters(self) -> None:
        profiler = RunProfiler(run_id="r7")
        profiler.start()
        with profiler.phase("fetch") as ctx:
            ctx.set_counter("blocks", 5)
        report = profiler.report()
        assert report.phases[0].counters["blocks"] == 5

    def test_log_report(self, caplog) -> None:
        profiler = RunProfiler(run_id="r8")
        profiler.start()
        with profiler.phase("work"):
            pass
        import logging
        with caplog.at_level(logging.INFO, logger="msa_turboquant.eval.profiler"):
            report = profiler.log_report()
        assert isinstance(report, ProfilingReport)


# ---------------------------------------------------------------------------
# Compression wiring in EvalSampleResult
# ---------------------------------------------------------------------------

class TestEvalSampleResultCompression:

    def test_bytes_fetched_in_dict(self) -> None:
        r = EvalSampleResult(bytes_fetched=4096, compression_ratio=3.5)
        d = r.to_dict()
        assert d["bytes_fetched"] == 4096
        assert d["compression_ratio"] == 3.5

    def test_defaults_zero(self) -> None:
        r = EvalSampleResult()
        assert r.bytes_fetched == 0
        assert r.compression_ratio == 0.0


# ---------------------------------------------------------------------------
# Compression round-trip in _compress_kv_blocks
# ---------------------------------------------------------------------------

class TestCompressKVBlocks:
    """Test the _compress_kv_blocks method via EvalHarness."""

    def test_compress_kv_blocks_int8(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=2)
        compressor = Int8Compressor()
        harness = EvalHarness(
            model=model,
            mode="sparse_plus_compression",
            compressor=compressor,
        )

        # Create fake KV blocks
        from src.models.kv_extractor import KVBlock
        blocks = [
            KVBlock(
                block_id="b0",
                keys=[torch.randn(4, 8, 16) for _ in range(2)],
                values=[torch.randn(4, 8, 16) for _ in range(2)],
                routing_vector=torch.randn(64),
                num_tokens=8,
                layer_indices=[0, 1],
            ),
        ]

        ratio = harness._compress_kv_blocks(blocks)
        assert ratio > 0.0
        # KV tensors should be modified in-place (reconstructed)
        assert blocks[0].keys[0].dtype == torch.float32

    def test_compress_kv_blocks_int4(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=1)
        compressor = Int4Compressor(group_size=16)
        harness = EvalHarness(
            model=model,
            mode="compression_only",
            compressor=compressor,
        )

        from src.models.kv_extractor import KVBlock
        blocks = [
            KVBlock(
                block_id="b0",
                keys=[torch.randn(4, 8, 32)],
                values=[torch.randn(4, 8, 32)],
                routing_vector=torch.randn(64),
                num_tokens=8,
                layer_indices=[0],
            ),
        ]

        ratio = harness._compress_kv_blocks(blocks)
        assert ratio > 1.0  # int4 should compress well

    def test_compress_kv_blocks_turboquant(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=1)
        compressor = RotatedUniformCompressor(bits=4, group_size=32)
        harness = EvalHarness(
            model=model,
            mode="compression_only",
            compressor=compressor,
        )

        from src.models.kv_extractor import KVBlock
        blocks = [
            KVBlock(
                block_id="b0",
                keys=[torch.randn(4, 8, 64)],
                values=[torch.randn(4, 8, 64)],
                routing_vector=torch.randn(64),
                num_tokens=8,
                layer_indices=[0],
            ),
        ]

        ratio = harness._compress_kv_blocks(blocks)
        assert ratio > 1.0

    def test_no_compressor_returns_zero(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=1)
        harness = EvalHarness(model=model, mode="dense")
        assert harness._compress_kv_blocks([]) == 0.0

    def test_empty_blocks_returns_zero(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=1)
        compressor = Int8Compressor()
        harness = EvalHarness(
            model=model,
            mode="compression_only",
            compressor=compressor,
        )
        assert harness._compress_kv_blocks([]) == 0.0


# ---------------------------------------------------------------------------
# EvalHarness with profiler and compressor (integration)
# ---------------------------------------------------------------------------

class TestEvalHarnessWithProfiler:
    """Integration tests verifying profiler + compression in eval runs."""

    @staticmethod
    def _make_model() -> DummyModel:
        m = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
        m.load()
        m.decode = lambda ids: ["dummy answer 1234"] * ids.shape[0]
        return m

    def test_profiler_collects_phases_sparse(self) -> None:
        """Sparse mode should record bank_build, route, generate, score phases."""
        model = self._make_model()
        profiler = RunProfiler(run_id="test_sparse", enabled=True)
        harness = EvalHarness(
            model=model,
            mode="sparse",
            router_engine="torch_cosine",
            top_k=2,
            profiler=profiler,
        )

        sample = generate_niah_sample(num_blocks=5, block_chars=200, seed=42)
        result = harness.evaluate([sample], run_id="test_sparse")
        report = profiler.report()

        phase_names = [p.name for p in report.phases]
        assert "bank_build" in phase_names
        assert "route" in phase_names
        assert "generate" in phase_names
        assert "score" in phase_names
        assert report.total_wall_time_ms > 0

    def test_profiler_disabled_no_overhead(self) -> None:
        """Disabled profiler should produce empty report."""
        model = self._make_model()
        profiler = RunProfiler(enabled=False)
        harness = EvalHarness(
            model=model,
            mode="dense",
            profiler=profiler,
        )

        sample = generate_niah_sample(num_blocks=3, block_chars=100, seed=42)
        harness.evaluate([sample], run_id="test_dense")
        report = profiler.report()
        assert len(report.phases) == 0

    def test_sparse_plus_compression_tracks_bytes(self) -> None:
        model = self._make_model()
        compressor = Int8Compressor()
        profiler = RunProfiler(run_id="test_spc", enabled=True)
        harness = EvalHarness(
            model=model,
            mode="sparse_plus_compression",
            router_engine="torch_cosine",
            top_k=2,
            compressor=compressor,
            profiler=profiler,
        )

        sample = generate_niah_sample(num_blocks=5, block_chars=200, seed=42)
        result = harness.evaluate([sample], run_id="test_spc")
        report = profiler.report()

        # Should have compress phase
        phase_names = [p.name for p in report.phases]
        assert "compress" in phase_names

        # Should track bytes fetched
        assert report.total_bytes_fetched > 0

        # Compression info should be set
        assert report.compression_info.get("method") == "int8"

        # Sample result should have compression ratio
        assert result.sample_results[0].compression_ratio > 0
        assert result.sample_results[0].bytes_fetched > 0

    def test_compression_only_mode(self) -> None:
        model = self._make_model()
        compressor = Int4Compressor(group_size=32)
        profiler = RunProfiler(run_id="test_co", enabled=True)
        harness = EvalHarness(
            model=model,
            mode="compression_only",
            compressor=compressor,
            profiler=profiler,
        )

        sample = generate_niah_sample(num_blocks=3, block_chars=100, seed=42)
        result = harness.evaluate([sample], run_id="test_co")
        report = profiler.report()

        phase_names = [p.name for p in report.phases]
        assert "bank_build" in phase_names
        assert "compress" in phase_names
        assert result.sample_results[0].compression_ratio > 0

    def test_save_results_includes_profile(self, tmp_path: Path) -> None:
        model = self._make_model()
        profiler = RunProfiler(run_id="test_save", enabled=True)
        harness = EvalHarness(
            model=model,
            mode="dense",
            profiler=profiler,
        )

        sample = generate_niah_sample(num_blocks=3, block_chars=100, seed=42)
        result = harness.evaluate([sample], run_id="test_save")
        paths = harness.save_results(result, tmp_path)

        assert "profile" in paths
        assert paths["profile"].exists()
        data = json.loads(paths["profile"].read_text())
        assert data["run_id"] == "test_save"

    def test_config_snapshot_includes_compressor(self) -> None:
        model = self._make_model()
        compressor = RotatedUniformCompressor(bits=4, group_size=128)
        harness = EvalHarness(
            model=model,
            mode="dense",
            compressor=compressor,
        )

        sample = generate_niah_sample(num_blocks=3, block_chars=100, seed=42)
        result = harness.evaluate([sample], run_id="test_config")
        assert "compressor" in result.config
        assert "rotated_uniform" in result.config["compressor"]
        assert result.config["bits_per_value"] > 4.0
