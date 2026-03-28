"""Tests for the evaluation harness, interleave, and systems metrics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.eval.niah import generate_niah_sample
from src.eval.run_eval import (
    EvalHarness,
    EvalRunResult,
    EvalSampleResult,
    score_answer,
)
from src.eval.systems_metrics import (
    MetricsCollector,
    RunMetrics,
    SystemsSnapshot,
    take_snapshot,
)
from src.memory.interleave import (
    AssembledContext,
    assemble_context,
    assemble_dense_context,
)

# Reuse DummyModel
from tests.test_models import DummyModel


# ---------------------------------------------------------------------------
# Interleave tests
# ---------------------------------------------------------------------------

class TestAssembleContext:
    """Tests for context assembly."""

    def test_prepend_strategy(self) -> None:
        result = assemble_context(
            query="What is the code?",
            retrieved_texts=["Block one.", "Block two."],
            strategy="prepend",
        )
        assert isinstance(result, AssembledContext)
        assert "Block one." in result.text
        assert "Block two." in result.text
        assert "What is the code?" in result.text
        assert result.num_retrieved_blocks == 2
        assert result.strategy == "prepend"

    def test_interleave_strategy(self) -> None:
        result = assemble_context(
            query="What?",
            retrieved_texts=["Alpha.", "Beta."],
            strategy="interleave",
        )
        assert "[Document 1]" in result.text
        assert "[Document 2]" in result.text

    def test_summarize_prefix_strategy(self) -> None:
        result = assemble_context(
            query="What?",
            retrieved_texts=["Content here."],
            strategy="summarize_prefix",
        )
        assert "Read it carefully" in result.text

    def test_block_ids_tracked(self) -> None:
        result = assemble_context(
            query="Q?",
            retrieved_texts=["A.", "B."],
            retrieved_block_ids=["blk_5", "blk_12"],
        )
        assert result.retrieved_block_ids == ["blk_5", "blk_12"]

    def test_max_context_chars(self) -> None:
        long_text = "X" * 5000
        result = assemble_context(
            query="Short question?",
            retrieved_texts=[long_text],
            max_context_chars=500,
        )
        assert len(result.text) <= 510  # small tolerance for headers

    def test_empty_retrieved(self) -> None:
        result = assemble_context(
            query="What?",
            retrieved_texts=[],
        )
        assert "What?" in result.text
        assert result.num_retrieved_blocks == 0

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown assembly"):
            assemble_context(query="Q?", retrieved_texts=["A."], strategy="invalid")


class TestAssembleDenseContext:
    """Tests for dense (full) context assembly."""

    def test_basic(self) -> None:
        result = assemble_dense_context(
            query="What is 42?",
            full_context_blocks=["Block A.", "Block B.", "Block C."],
        )
        assert isinstance(result, AssembledContext)
        assert "Block A." in result.text
        assert "Block C." in result.text
        assert "What is 42?" in result.text
        assert result.strategy == "dense"
        assert result.num_retrieved_blocks == 3

    def test_truncation(self) -> None:
        blocks = ["Y" * 1000 for _ in range(10)]
        result = assemble_dense_context(
            query="Q?",
            full_context_blocks=blocks,
            max_context_chars=500,
        )
        assert len(result.text) <= 520


# ---------------------------------------------------------------------------
# Systems metrics tests
# ---------------------------------------------------------------------------

class TestSystemsMetrics:
    """Tests for systems metric collection."""

    def test_take_snapshot(self) -> None:
        snap = take_snapshot()
        assert isinstance(snap, SystemsSnapshot)
        assert snap.ram_used_mb > 0
        assert snap.ram_available_mb > 0

    def test_snapshot_to_dict(self) -> None:
        snap = take_snapshot()
        d = snap.to_dict()
        assert "ram_used_mb" in d
        assert "gpu_allocated_mb" in d

    def test_metrics_collector(self) -> None:
        collector = MetricsCollector(run_id="test", mode="dense")
        with collector:
            collector.add_generation_time(100.0)
            collector.add_tokens(50)
            collector.set_num_samples(5)
        metrics = collector.get_metrics()

        assert isinstance(metrics, RunMetrics)
        assert metrics.run_id == "test"
        assert metrics.mode == "dense"
        assert metrics.generation_time_ms == 100.0
        assert metrics.num_tokens_generated == 50
        assert metrics.wall_time_ms > 0

    def test_run_metrics_to_dict(self) -> None:
        collector = MetricsCollector(run_id="t1", mode="sparse")
        with collector:
            collector.add_retrieval_time(5.0)
        metrics = collector.get_metrics()
        d = metrics.to_dict()
        assert d["run_id"] == "t1"
        assert d["retrieval_time_ms"] == 5.0
        assert "peak_gpu_mb" in d

    def test_tokens_per_second(self) -> None:
        collector = MetricsCollector()
        with collector:
            collector.add_generation_time(1000.0)  # 1 second
            collector.add_tokens(100)
        metrics = collector.get_metrics()
        assert metrics.tokens_per_second == pytest.approx(100.0, rel=0.01)


# ---------------------------------------------------------------------------
# Score answer tests
# ---------------------------------------------------------------------------

class TestScoreAnswer:
    """Tests for answer scoring."""

    def test_exact_match(self) -> None:
        assert score_answer("4567", "4567") is True

    def test_substring_match(self) -> None:
        assert score_answer("The code is 4567.", "4567") is True

    def test_case_insensitive(self) -> None:
        assert score_answer("PARIS", "paris") is True

    def test_miss(self) -> None:
        assert score_answer("The answer is 1234", "5678") is False

    def test_empty_answer(self) -> None:
        assert score_answer("", "4567") is False


# ---------------------------------------------------------------------------
# EvalHarness tests (using DummyModel — no real generation)
# ---------------------------------------------------------------------------

class TestEvalHarness:
    """Tests for the evaluation harness using DummyModel.

    DummyModel generates random tokens, so accuracy will be ~0.
    These tests verify the pipeline runs end-to-end without errors.
    """

    @pytest.fixture
    def model(self):
        m = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
        m.load()
        # Add decode method that returns dummy text
        m.decode = lambda ids: ["dummy answer 1234"] * ids.shape[0]
        return m

    @pytest.fixture
    def samples(self):
        return [
            generate_niah_sample(num_blocks=5, block_chars=100, seed=i, sample_id=f"s_{i}")
            for i in range(3)
        ]

    def test_dense_mode(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="dense")
        result = harness.evaluate(samples, run_id="test_dense")

        assert isinstance(result, EvalRunResult)
        assert result.num_samples == 3
        assert len(result.sample_results) == 3
        assert result.mode == "dense"
        assert 0.0 <= result.accuracy <= 1.0

    def test_sparse_mode(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="sparse", top_k=3)
        result = harness.evaluate(samples, run_id="test_sparse")

        assert result.num_samples == 3
        assert result.mode == "sparse"
        # Retrieval metrics should be populated
        assert "recall_at_k" in result.retrieval_metrics

    def test_oracle_mode(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="oracle_plus_compression", top_k=3)
        result = harness.evaluate(samples, run_id="test_oracle")

        assert result.num_samples == 3
        # Oracle should have perfect recall
        assert result.retrieval_metrics.get("recall_at_k", 0) == 1.0

    def test_kv_inject_mode(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="kv_inject", top_k=3)
        result = harness.evaluate(samples, run_id="test_kv_inject")

        assert result.num_samples == 3
        assert result.mode == "kv_inject"
        assert "recall_at_k" in result.retrieval_metrics

    def test_kv_inject_compressed_mode(self, model, samples) -> None:
        from src.compression import create_compressor
        compressor = create_compressor("int8")
        harness = EvalHarness(
            model=model, mode="kv_inject_compressed", top_k=3,
            compressor=compressor,
        )
        result = harness.evaluate(samples, run_id="test_kv_inject_compressed")

        assert result.num_samples == 3
        assert result.mode == "kv_inject_compressed"
        assert "recall_at_k" in result.retrieval_metrics

    def test_compression_only_mode(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="compression_only")
        result = harness.evaluate(samples, run_id="test_compression")
        assert result.num_samples == 3

    def test_sample_result_structure(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="sparse", top_k=2)
        result = harness.evaluate(samples[:1], run_id="test_struct")
        sr = result.sample_results[0]

        assert isinstance(sr, EvalSampleResult)
        assert sr.sample_id == samples[0].sample_id
        assert sr.needle_answer != ""
        assert sr.context_chars > 0
        assert sr.generation_time_ms > 0

    def test_save_results(self, model, samples, tmp_path) -> None:
        harness = EvalHarness(model=model, mode="dense")
        result = harness.evaluate(samples[:2], run_id="test_save")
        paths = harness.save_results(result, tmp_path / "results")

        assert "full" in paths
        assert "summary" in paths
        assert paths["full"].exists()
        assert paths["summary"].exists()

    def test_result_to_dict(self, model, samples) -> None:
        harness = EvalHarness(model=model, mode="sparse", top_k=2)
        result = harness.evaluate(samples[:1])
        d = result.to_dict()
        assert "accuracy" in d
        assert "sample_results" in d
        assert "systems_metrics" in d
