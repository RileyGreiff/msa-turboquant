"""Tests for routing, fetching, and retrieval metrics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.eval.retrieval_metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    compute_metrics_from_result,
    compute_retrieval_metrics,
)
from src.memory.bank_builder import MemoryBankBuilder
from src.memory.bank_store import save_bank
from src.memory.chunking import chunk_text
from src.memory.fetcher import FetchResult, MemoryFetcher
from src.memory.router import (
    FaissRouter,
    OracleRouter,
    RetrievalResult,
    TorchCosineRouter,
    create_router,
)

# Reuse DummyModel
from tests.test_models import DummyModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vectors(num: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Create random L2-normalized vectors."""
    gen = torch.Generator().manual_seed(seed)
    vecs = torch.randn(num, dim, generator=gen)
    return F.normalize(vecs, p=2, dim=1)


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestFaissRouter:
    """Tests for FAISS-based router."""

    def test_build_and_query(self) -> None:
        router = FaissRouter()
        vecs = _make_vectors(100, 64)
        router.build_index(vecs)

        result = router.query(vecs[0], top_k=5)
        assert isinstance(result, RetrievalResult)
        assert len(result.block_indices) == 5
        assert len(result.scores) == 5
        # The query vector itself should be the top match
        assert result.block_indices[0] == 0
        assert result.scores[0] > 0.99

    def test_top_k_exceeds_bank(self) -> None:
        router = FaissRouter()
        vecs = _make_vectors(3, 64)
        router.build_index(vecs)
        result = router.query(vecs[0], top_k=10)
        assert len(result.block_indices) == 3

    def test_gold_evaluation(self) -> None:
        router = FaissRouter()
        vecs = _make_vectors(50, 64)
        router.build_index(vecs)
        result = router.query(vecs[0], top_k=5, gold_indices=[0])
        assert result.gold_retrieved == [True]
        assert result.hit

    def test_gold_miss(self) -> None:
        router = FaissRouter()
        vecs = _make_vectors(100, 64)
        router.build_index(vecs)
        # Query with vector 0 but gold is far away
        result = router.query(vecs[0], top_k=1, gold_indices=[99])
        # Likely a miss unless vectors happen to be similar
        assert len(result.gold_retrieved) == 1

    def test_block_ids(self) -> None:
        router = FaissRouter()
        vecs = _make_vectors(10, 64)
        ids = [f"blk_{i}" for i in range(10)]
        router.build_index(vecs, block_ids=ids)
        result = router.query(vecs[3], top_k=3)
        assert result.block_ids[0] == "blk_3"

    def test_query_batch(self) -> None:
        router = FaissRouter()
        vecs = _make_vectors(50, 64)
        router.build_index(vecs)
        queries = vecs[:5]
        results = router.query_batch(queries, top_k=3)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.block_indices[0] == i  # self is best match

    def test_not_built_raises(self) -> None:
        router = FaissRouter()
        with pytest.raises(RuntimeError, match="not built"):
            router.query(torch.randn(64), top_k=5)


class TestTorchCosineRouter:
    """Tests for pure-torch cosine router."""

    def test_build_and_query(self) -> None:
        router = TorchCosineRouter()
        vecs = _make_vectors(50, 64)
        router.build_index(vecs)

        result = router.query(vecs[0], top_k=5)
        assert len(result.block_indices) == 5
        assert result.block_indices[0] == 0
        assert result.scores[0] > 0.99

    def test_matches_faiss(self) -> None:
        """Torch and FAISS routers should return the same top-k."""
        vecs = _make_vectors(50, 64)
        query = vecs[10]

        faiss_router = FaissRouter()
        faiss_router.build_index(vecs)
        faiss_result = faiss_router.query(query, top_k=5)

        torch_router = TorchCosineRouter()
        torch_router.build_index(vecs)
        torch_result = torch_router.query(query, top_k=5)

        assert faiss_result.block_indices == torch_result.block_indices

    def test_not_built_raises(self) -> None:
        router = TorchCosineRouter()
        with pytest.raises(RuntimeError, match="not built"):
            router.query(torch.randn(64), top_k=5)


class TestOracleRouter:
    """Tests for oracle (perfect) router."""

    def test_returns_gold(self) -> None:
        router = OracleRouter()
        vecs = _make_vectors(20, 64)
        router.build_index(vecs)

        result = router.query(vecs[0], top_k=5, gold_indices=[7, 13])
        assert 7 in result.block_indices
        assert 13 in result.block_indices
        assert all(result.gold_retrieved)
        assert result.hit

    def test_pads_to_top_k(self) -> None:
        router = OracleRouter()
        vecs = _make_vectors(20, 64)
        router.build_index(vecs)

        result = router.query(vecs[0], top_k=5, gold_indices=[3])
        assert len(result.block_indices) == 5
        assert result.block_indices[0] == 3

    def test_no_gold_warns(self) -> None:
        router = OracleRouter()
        vecs = _make_vectors(10, 64)
        router.build_index(vecs)
        result = router.query(vecs[0], top_k=5)
        assert result.block_indices == []


class TestCreateRouter:
    """Tests for router factory."""

    def test_faiss(self) -> None:
        assert isinstance(create_router("faiss"), FaissRouter)

    def test_torch_cosine(self) -> None:
        assert isinstance(create_router("torch_cosine"), TorchCosineRouter)

    def test_oracle(self) -> None:
        assert isinstance(create_router("oracle"), OracleRouter)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            create_router("invalid_engine")


# ---------------------------------------------------------------------------
# Retrieval metrics tests
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    """Tests for retrieval quality metrics."""

    def test_perfect_retrieval(self) -> None:
        metrics = compute_retrieval_metrics([3, 7, 1, 9, 0], gold_indices=[3])
        assert metrics.recall_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.hit_rate == 1.0
        assert metrics.precision_at_k == 0.2  # 1 gold out of 5

    def test_miss(self) -> None:
        metrics = compute_retrieval_metrics([0, 1, 2, 3, 4], gold_indices=[99])
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.hit_rate == 0.0

    def test_partial_recall(self) -> None:
        metrics = compute_retrieval_metrics([0, 1, 2], gold_indices=[1, 5])
        assert metrics.recall_at_k == 0.5  # found 1 of 2
        assert metrics.mrr == 0.5  # gold at rank 2
        assert metrics.hit_rate == 1.0

    def test_mrr_rank_3(self) -> None:
        metrics = compute_retrieval_metrics([10, 20, 5, 30, 40], gold_indices=[5])
        assert metrics.mrr == pytest.approx(1.0 / 3)

    def test_multi_gold(self) -> None:
        metrics = compute_retrieval_metrics([0, 1, 2, 3, 4], gold_indices=[1, 3, 99])
        assert metrics.recall_at_k == pytest.approx(2 / 3)
        assert metrics.mrr == 0.5  # first gold (1) at rank 2

    def test_empty_gold(self) -> None:
        metrics = compute_retrieval_metrics([0, 1, 2], gold_indices=[])
        assert metrics.recall_at_k == 1.0  # vacuously true
        assert metrics.hit_rate == 1.0

    def test_from_retrieval_result(self) -> None:
        result = RetrievalResult(
            block_indices=[5, 3, 7],
            gold_indices=[3],
        )
        metrics = compute_metrics_from_result(result)
        assert metrics.recall_at_k == 1.0
        assert metrics.mrr == 0.5  # gold at rank 2

    def test_aggregate(self) -> None:
        m1 = RetrievalMetrics(recall_at_k=1.0, mrr=1.0, hit_rate=1.0, precision_at_k=0.2, num_gold=1, top_k=5)
        m2 = RetrievalMetrics(recall_at_k=0.0, mrr=0.0, hit_rate=0.0, precision_at_k=0.0, num_gold=1, top_k=5)
        agg = aggregate_metrics([m1, m2])
        assert agg.recall_at_k == 0.5
        assert agg.mrr == 0.5
        assert agg.hit_rate == 0.5

    def test_aggregate_empty(self) -> None:
        agg = aggregate_metrics([])
        assert agg.recall_at_k == 0.0


# ---------------------------------------------------------------------------
# Fetcher tests
# ---------------------------------------------------------------------------

class TestMemoryFetcher:
    """Tests for the retrieval + fetch pipeline."""

    @pytest.fixture
    def bank_and_model(self):
        model = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
        model.load()
        text = "Test content for memory bank. " * 100
        blocks = chunk_text(text, chunk_size=200, chunk_overlap=0, document_id="doc")
        builder = MemoryBankBuilder(model, extraction_mode="direct")
        bank = builder.build(blocks[:10], bank_id="fetch_test")
        return bank, model

    def test_from_bank(self, bank_and_model) -> None:
        bank, _ = bank_and_model
        fetcher = MemoryFetcher.from_bank(bank, engine="faiss")
        result = fetcher.fetch(bank.routing_vectors[0], top_k=3)

        assert isinstance(result, FetchResult)
        assert result.num_fetched == 3
        assert result.total_time_ms > 0
        assert result.retrieval.block_indices[0] == 0

    def test_from_bank_torch(self, bank_and_model) -> None:
        bank, _ = bank_and_model
        fetcher = MemoryFetcher.from_bank(bank, engine="torch_cosine")
        result = fetcher.fetch(bank.routing_vectors[0], top_k=3)
        assert result.num_fetched == 3

    def test_from_disk(self, bank_and_model, tmp_path) -> None:
        bank, _ = bank_and_model
        bank_dir = tmp_path / "bank"
        save_bank(bank, bank_dir)

        fetcher = MemoryFetcher.from_disk(bank_dir, engine="faiss")
        result = fetcher.fetch(bank.routing_vectors[0], top_k=3)
        assert result.num_fetched == 3
        assert result.kv_blocks[0].num_layers > 0

    def test_fetch_without_kv(self, bank_and_model) -> None:
        bank, _ = bank_and_model
        fetcher = MemoryFetcher.from_bank(bank, engine="faiss")
        result = fetcher.fetch(bank.routing_vectors[0], top_k=3, fetch_kv=False)
        assert result.num_fetched == 0
        assert len(result.retrieval.block_indices) == 3

    def test_fetch_with_gold(self, bank_and_model) -> None:
        bank, _ = bank_and_model
        fetcher = MemoryFetcher.from_bank(bank, engine="faiss")
        result = fetcher.fetch(bank.routing_vectors[0], top_k=5, gold_indices=[0])
        assert result.retrieval.gold_retrieved == [True]

    def test_fetch_batch(self, bank_and_model) -> None:
        bank, _ = bank_and_model
        fetcher = MemoryFetcher.from_bank(bank, engine="faiss")
        queries = bank.routing_vectors[:3]
        results = fetcher.fetch_batch(queries, top_k=2)
        assert len(results) == 3
        for r in results:
            assert r.num_fetched == 2

    def test_to_dict(self, bank_and_model) -> None:
        bank, _ = bank_and_model
        fetcher = MemoryFetcher.from_bank(bank, engine="faiss")
        result = fetcher.fetch(bank.routing_vectors[0], top_k=2, gold_indices=[0])
        d = result.to_dict()
        assert "block_indices" in d
        assert "total_kv_bytes" in d
        assert "route_time_ms" in d
