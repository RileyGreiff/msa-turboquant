"""Retrieval quality metrics for evaluating routing performance.

Computes standard information retrieval metrics given retrieved block indices
and ground-truth gold block indices.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.memory.router import RetrievalResult


@dataclass
class RetrievalMetrics:
    """Computed retrieval metrics for a single query or aggregated over many.

    Attributes:
        recall_at_k: Fraction of gold blocks found in top-k.
        mrr: Mean Reciprocal Rank of the first gold block in the retrieval list.
        hit_rate: 1.0 if any gold block is in top-k, else 0.0.
        precision_at_k: Fraction of top-k blocks that are gold.
        num_gold: Number of gold blocks.
        top_k: Number of blocks retrieved.
    """
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    precision_at_k: float = 0.0
    num_gold: int = 0
    top_k: int = 0

    def to_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "precision_at_k": self.precision_at_k,
            "num_gold": self.num_gold,
            "top_k": self.top_k,
        }


def compute_retrieval_metrics(
    retrieved_indices: list[int],
    gold_indices: list[int],
) -> RetrievalMetrics:
    """Compute retrieval metrics for a single query.

    Args:
        retrieved_indices: Ordered list of retrieved block indices (best first).
        gold_indices: Ground-truth block indices that should be retrieved.

    Returns:
        RetrievalMetrics for this query.
    """
    if not gold_indices:
        return RetrievalMetrics(
            recall_at_k=1.0,  # vacuously true
            mrr=0.0,
            hit_rate=1.0,
            precision_at_k=0.0,
            num_gold=0,
            top_k=len(retrieved_indices),
        )

    gold_set = set(gold_indices)
    k = len(retrieved_indices)

    # Recall@k: what fraction of gold blocks were retrieved
    retrieved_gold = sum(1 for idx in retrieved_indices if idx in gold_set)
    recall = retrieved_gold / len(gold_indices)

    # Precision@k: what fraction of retrieved blocks are gold
    precision = retrieved_gold / k if k > 0 else 0.0

    # Hit rate: binary — any gold block retrieved?
    hit = 1.0 if retrieved_gold > 0 else 0.0

    # MRR: reciprocal rank of the first gold block in the retrieval list
    mrr = 0.0
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in gold_set:
            mrr = 1.0 / rank
            break

    return RetrievalMetrics(
        recall_at_k=recall,
        mrr=mrr,
        hit_rate=hit,
        precision_at_k=precision,
        num_gold=len(gold_indices),
        top_k=k,
    )


def compute_metrics_from_result(result: RetrievalResult) -> RetrievalMetrics:
    """Compute metrics directly from a RetrievalResult."""
    return compute_retrieval_metrics(result.block_indices, result.gold_indices)


def aggregate_metrics(metrics_list: list[RetrievalMetrics]) -> RetrievalMetrics:
    """Compute mean metrics over a list of per-query metrics.

    Args:
        metrics_list: List of per-query RetrievalMetrics.

    Returns:
        Aggregated (mean) metrics.
    """
    if not metrics_list:
        return RetrievalMetrics()

    n = len(metrics_list)
    return RetrievalMetrics(
        recall_at_k=sum(m.recall_at_k for m in metrics_list) / n,
        mrr=sum(m.mrr for m in metrics_list) / n,
        hit_rate=sum(m.hit_rate for m in metrics_list) / n,
        precision_at_k=sum(m.precision_at_k for m in metrics_list) / n,
        num_gold=round(sum(m.num_gold for m in metrics_list) / n),
        top_k=metrics_list[0].top_k if metrics_list else 0,
    )
