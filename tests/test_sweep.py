"""Tests for scale sweep configuration, runner, and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.experiments.sweep_config import (
    SweepConfig,
    SweepResult,
    SweepRunRecord,
)
from src.experiments.run_scale_sweep import ScaleSweep
from src.experiments.sweep_plots import (
    generate_sweep_plots,
    plot_accuracy_vs_bank_size,
    plot_compression_comparison,
    plot_latency_vs_bank_size,
    plot_recall_vs_top_k,
)

# Reuse DummyModel from test_models
from tests.test_models import DummyModel


# ---------------------------------------------------------------------------
# SweepConfig tests
# ---------------------------------------------------------------------------

class TestSweepConfig:

    def test_defaults(self) -> None:
        cfg = SweepConfig()
        assert len(cfg.modes) > 0
        assert len(cfg.bank_sizes) > 0
        assert cfg.num_trials >= 1

    def test_parameter_grid_not_empty(self) -> None:
        cfg = SweepConfig(
            modes=["sparse"],
            bank_sizes=[10],
            block_chars=[200],
            top_k_values=[3],
            compression_methods=["none"],
        )
        grid = cfg.parameter_grid()
        assert len(grid) > 0
        assert grid[0]["mode"] == "sparse"
        assert grid[0]["bank_size"] == 10

    def test_dense_mode_collapses(self) -> None:
        """Dense mode should not multiply over top_k or compression."""
        cfg = SweepConfig(
            modes=["dense"],
            bank_sizes=[10, 20],
            block_chars=[200],
            top_k_values=[3, 5, 10],
            compression_methods=["none", "int4"],
        )
        grid = cfg.parameter_grid()
        # Dense only varies by bank_size — top_k and compression are collapsed
        assert len(grid) == 2  # 2 bank sizes

    def test_sparse_skips_compression(self) -> None:
        """Sparse mode should only run with 'none' compression."""
        cfg = SweepConfig(
            modes=["sparse"],
            bank_sizes=[10],
            block_chars=[200],
            top_k_values=[3],
            compression_methods=["none", "int4"],
        )
        grid = cfg.parameter_grid()
        methods = [g["compression_method"] for g in grid]
        assert all(m == "none" for m in methods)

    def test_compression_mode_skips_none(self) -> None:
        """sparse_plus_compression should skip 'none' compression."""
        cfg = SweepConfig(
            modes=["sparse_plus_compression"],
            bank_sizes=[10],
            block_chars=[200],
            top_k_values=[3],
            compression_methods=["none", "int4"],
        )
        grid = cfg.parameter_grid()
        methods = [g["compression_method"] for g in grid]
        assert all(m != "none" for m in methods)

    def test_total_runs(self) -> None:
        cfg = SweepConfig(
            modes=["sparse"],
            bank_sizes=[10, 20],
            block_chars=[200],
            top_k_values=[3],
            compression_methods=["none"],
            num_trials=3,
        )
        grid_len = len(cfg.parameter_grid())
        assert cfg.total_runs() == grid_len * 3

    def test_validation_positive(self) -> None:
        with pytest.raises(Exception):
            SweepConfig(bank_sizes=[0, 10])

    def test_validation_trials(self) -> None:
        with pytest.raises(Exception):
            SweepConfig(num_trials=0)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(Exception):
            SweepConfig(bogus_field=True)


# ---------------------------------------------------------------------------
# SweepRunRecord tests
# ---------------------------------------------------------------------------

class TestSweepRunRecord:

    def test_to_flat_dict(self) -> None:
        rec = SweepRunRecord(
            params={"mode": "sparse", "bank_size": 50},
            trial=0,
            run_id="test",
            accuracy=0.75,
            recall_at_k=0.8,
        )
        d = rec.to_flat_dict()
        assert d["mode"] == "sparse"
        assert d["bank_size"] == 50
        assert d["accuracy"] == 0.75
        assert d["trial"] == 0


# ---------------------------------------------------------------------------
# SweepResult tests
# ---------------------------------------------------------------------------

class TestSweepResult:

    def _make_records(self) -> list[SweepRunRecord]:
        return [
            SweepRunRecord(
                params={"mode": "sparse", "bank_size": 10, "compression_method": "none"},
                trial=0, accuracy=0.8, wall_time_ms=100,
            ),
            SweepRunRecord(
                params={"mode": "sparse", "bank_size": 10, "compression_method": "none"},
                trial=1, accuracy=0.9, wall_time_ms=110,
            ),
            SweepRunRecord(
                params={"mode": "sparse", "bank_size": 50, "compression_method": "none"},
                trial=0, accuracy=0.7, wall_time_ms=200,
            ),
        ]

    def test_summary_table(self) -> None:
        result = SweepResult(config=SweepConfig(), records=self._make_records())
        table = result.summary_table()
        assert len(table) == 3
        assert all("mode" in row for row in table)

    def test_averaged_summary(self) -> None:
        result = SweepResult(config=SweepConfig(), records=self._make_records())
        avg = result.averaged_summary()
        # 2 unique param combos (bank_size=10 x2 trials, bank_size=50 x1)
        assert len(avg) == 2
        # The bank_size=10 combo should have averaged accuracy 0.85
        for row in avg:
            if row["bank_size"] == 10:
                assert row["accuracy"] == pytest.approx(0.85, abs=0.01)
                assert row["num_trials"] == 2
            elif row["bank_size"] == 50:
                assert row["accuracy"] == pytest.approx(0.7, abs=0.01)
                assert row["num_trials"] == 1


# ---------------------------------------------------------------------------
# ScaleSweep runner tests (with DummyModel)
# ---------------------------------------------------------------------------

class TestScaleSweep:

    @staticmethod
    def _make_model() -> DummyModel:
        m = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
        m.load()
        m.decode = lambda ids: ["dummy answer 1234"] * ids.shape[0]
        return m

    def test_small_sweep_runs(self) -> None:
        """Minimal sweep completes without error."""
        model = self._make_model()
        config = SweepConfig(
            modes=["sparse"],
            bank_sizes=[5],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["none"],
            num_trials=1,
            router_engine="torch_cosine",
        )
        sweep = ScaleSweep(model=model, config=config)
        result = sweep.run()
        assert len(result.records) == 1
        assert result.records[0].params["mode"] == "sparse"

    def test_sweep_with_compression(self) -> None:
        model = self._make_model()
        config = SweepConfig(
            modes=["sparse_plus_compression"],
            bank_sizes=[5],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["int8"],
            num_trials=1,
            router_engine="torch_cosine",
        )
        sweep = ScaleSweep(model=model, config=config)
        result = sweep.run()
        assert len(result.records) == 1
        assert result.records[0].compression_ratio > 0

    def test_sweep_multiple_trials(self) -> None:
        model = self._make_model()
        config = SweepConfig(
            modes=["sparse"],
            bank_sizes=[5],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["none"],
            num_trials=2,
            router_engine="torch_cosine",
        )
        sweep = ScaleSweep(model=model, config=config)
        result = sweep.run()
        assert len(result.records) == 2
        assert result.records[0].trial == 0
        assert result.records[1].trial == 1

    def test_sweep_save(self, tmp_path: Path) -> None:
        model = self._make_model()
        config = SweepConfig(
            modes=["sparse"],
            bank_sizes=[5],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["none"],
            num_trials=1,
            router_engine="torch_cosine",
        )
        sweep = ScaleSweep(model=model, config=config)
        result = sweep.run()
        paths = sweep.save(result, tmp_path)

        assert "all_runs" in paths
        assert paths["all_runs"].exists()
        assert "averaged" in paths
        assert paths["averaged"].exists()
        assert "full" in paths
        data = json.loads(paths["full"].read_text())
        assert data["num_runs"] == 1
        assert "config" in paths

    def test_dense_mode_sweep(self) -> None:
        model = self._make_model()
        config = SweepConfig(
            modes=["dense"],
            bank_sizes=[3],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["none"],
            num_trials=1,
        )
        sweep = ScaleSweep(model=model, config=config)
        result = sweep.run()
        assert len(result.records) == 1
        assert result.records[0].params["mode"] == "dense"

    def test_compression_only_sweep(self) -> None:
        model = self._make_model()
        config = SweepConfig(
            modes=["compression_only"],
            bank_sizes=[3],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["int4"],
            num_trials=1,
        )
        sweep = ScaleSweep(model=model, config=config)
        result = sweep.run()
        assert len(result.records) == 1
        assert result.records[0].compression_ratio > 0

    def test_with_profiling(self) -> None:
        model = self._make_model()
        config = SweepConfig(
            modes=["sparse"],
            bank_sizes=[5],
            block_chars=[100],
            top_k_values=[2],
            compression_methods=["none"],
            num_trials=1,
            router_engine="torch_cosine",
        )
        sweep = ScaleSweep(model=model, config=config, enable_profiling=True)
        result = sweep.run()
        assert len(result.records) == 1
        assert result.records[0].wall_time_ms > 0


# ---------------------------------------------------------------------------
# Sweep plotting tests
# ---------------------------------------------------------------------------

class TestSweepPlots:

    @staticmethod
    def _sample_records() -> list[dict[str, Any]]:
        return [
            {"mode": "sparse", "bank_size": 10, "compression_method": "none",
             "top_k": 3, "accuracy": 0.8, "wall_time_ms": 100,
             "recall_at_k": 0.9, "compression_ratio": 0.0},
            {"mode": "sparse", "bank_size": 50, "compression_method": "none",
             "top_k": 3, "accuracy": 0.7, "wall_time_ms": 200,
             "recall_at_k": 0.85, "compression_ratio": 0.0},
            {"mode": "sparse_plus_compression", "bank_size": 10,
             "compression_method": "int4", "top_k": 3,
             "accuracy": 0.75, "wall_time_ms": 120,
             "recall_at_k": 0.9, "compression_ratio": 3.5},
            {"mode": "sparse_plus_compression", "bank_size": 50,
             "compression_method": "int4", "top_k": 5,
             "accuracy": 0.65, "wall_time_ms": 250,
             "recall_at_k": 0.8, "compression_ratio": 3.5},
        ]

    def test_accuracy_vs_bank_size(self, tmp_path: Path) -> None:
        p = tmp_path / "acc.png"
        plot_accuracy_vs_bank_size(self._sample_records(), p)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_latency_vs_bank_size(self, tmp_path: Path) -> None:
        p = tmp_path / "lat.png"
        plot_latency_vs_bank_size(self._sample_records(), p)
        assert p.exists()

    def test_compression_comparison(self, tmp_path: Path) -> None:
        p = tmp_path / "comp.png"
        plot_compression_comparison(self._sample_records(), p, metric="accuracy")
        assert p.exists()

    def test_recall_vs_top_k(self, tmp_path: Path) -> None:
        p = tmp_path / "recall.png"
        plot_recall_vs_top_k(self._sample_records(), p)
        assert p.exists()

    def test_generate_all_plots(self, tmp_path: Path) -> None:
        paths = generate_sweep_plots(self._sample_records(), tmp_path)
        assert len(paths) >= 3  # accuracy, latency, compression, optionally recall
        assert all(p.exists() for p in paths)

    def test_empty_records_no_crash(self, tmp_path: Path) -> None:
        paths = generate_sweep_plots([], tmp_path)
        assert paths == []
