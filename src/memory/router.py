"""Routing module for memory bank retrieval.

Routes a query vector to the top-k most similar memory bank blocks using
cosine similarity. Supports three backends:

1. FAISS (IndexFlatIP on L2-normalized vectors) — fast, recommended for >1K blocks
2. Torch cosine similarity — simple, no extra dependency, fine for small banks
3. Oracle mode — returns the known gold blocks (for synthetic task upper bounds)

All routers implement the same Router interface returning RetrievalResult objects.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger("msa_turboquant.memory.router")


@dataclass
class RetrievalResult:
    """Result from a routing query.

    Attributes:
        query_id: Identifier for the query (optional).
        block_indices: Top-k block indices in the bank, ordered by score descending.
        block_ids: Corresponding block IDs (if available).
        scores: Similarity scores for each retrieved block.
        gold_indices: Ground-truth block indices (if known, for evaluation).
        gold_retrieved: Whether each gold block was found in top-k.
    """
    query_id: str = ""
    block_indices: list[int] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    gold_indices: list[int] = field(default_factory=list)
    gold_retrieved: list[bool] = field(default_factory=list)

    @property
    def top_k(self) -> int:
        return len(self.block_indices)

    @property
    def hit(self) -> bool:
        """Whether any gold block was retrieved."""
        return any(self.gold_retrieved) if self.gold_retrieved else False

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "block_indices": self.block_indices,
            "block_ids": self.block_ids,
            "scores": self.scores,
            "gold_indices": self.gold_indices,
            "gold_retrieved": self.gold_retrieved,
            "hit": self.hit,
        }


class BaseRouter(ABC):
    """Abstract router interface."""

    @abstractmethod
    def build_index(
        self,
        routing_vectors: torch.Tensor,
        block_ids: list[str] | None = None,
    ) -> None:
        """Build a search index from routing vectors.

        Args:
            routing_vectors: Shape (num_blocks, hidden_dim). Should be L2-normalized.
            block_ids: Optional block ID strings corresponding to each row.
        """
        ...

    @abstractmethod
    def query(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        gold_indices: list[int] | None = None,
    ) -> RetrievalResult:
        """Retrieve top-k blocks for a query vector.

        Args:
            query_vector: Shape (hidden_dim,) or (1, hidden_dim). Should be L2-normalized.
            top_k: Number of blocks to retrieve.
            gold_indices: Optional ground-truth indices for evaluation.

        Returns:
            RetrievalResult with block indices, scores, and gold match info.
        """
        ...

    def query_batch(
        self,
        query_vectors: torch.Tensor,
        top_k: int = 5,
        gold_indices_list: list[list[int]] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k blocks for a batch of query vectors.

        Default implementation loops over queries. Subclasses may override
        for batched search.
        """
        results = []
        for i in range(query_vectors.shape[0]):
            gold = gold_indices_list[i] if gold_indices_list else None
            result = self.query(query_vectors[i], top_k=top_k, gold_indices=gold)
            result.query_id = f"query_{i}"
            results.append(result)
        return results


class FaissRouter(BaseRouter):
    """FAISS-based cosine similarity router using IndexFlatIP.

    Vectors must be L2-normalized before indexing so that inner product = cosine sim.
    """

    def __init__(self) -> None:
        self._index = None
        self._block_ids: list[str] = []
        self._num_blocks: int = 0

    def build_index(
        self,
        routing_vectors: torch.Tensor,
        block_ids: list[str] | None = None,
    ) -> None:
        import faiss

        vectors_np = routing_vectors.float().cpu().numpy()
        dim = vectors_np.shape[1]
        self._num_blocks = vectors_np.shape[0]
        self._block_ids = block_ids or [f"block_{i}" for i in range(self._num_blocks)]

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors_np)

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors_np)

        logger.info(f"FAISS index built: {self._num_blocks} vectors, dim={dim}")

    def query(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        gold_indices: list[int] | None = None,
    ) -> RetrievalResult:
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q = query_vector.float().cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(q)

        k = min(top_k, self._num_blocks)
        scores_np, indices_np = self._index.search(q, k)

        block_indices = indices_np[0].tolist()
        scores = scores_np[0].tolist()
        block_ids = [self._block_ids[i] for i in block_indices]

        # Evaluate against gold
        gold_indices = gold_indices or []
        gold_set = set(gold_indices)
        gold_retrieved = [gi in set(block_indices) for gi in gold_indices]

        return RetrievalResult(
            block_indices=block_indices,
            block_ids=block_ids,
            scores=scores,
            gold_indices=gold_indices,
            gold_retrieved=gold_retrieved,
        )

    def query_batch(
        self,
        query_vectors: torch.Tensor,
        top_k: int = 5,
        gold_indices_list: list[list[int]] | None = None,
    ) -> list[RetrievalResult]:
        """Batched FAISS search (more efficient than looping)."""
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q = query_vectors.float().cpu().numpy()
        faiss.normalize_L2(q)

        k = min(top_k, self._num_blocks)
        scores_np, indices_np = self._index.search(q, k)

        results = []
        for i in range(q.shape[0]):
            block_indices = indices_np[i].tolist()
            scores = scores_np[i].tolist()
            block_ids = [self._block_ids[idx] for idx in block_indices]

            gold = gold_indices_list[i] if gold_indices_list else []
            gold_retrieved = [gi in set(block_indices) for gi in gold]

            results.append(RetrievalResult(
                query_id=f"query_{i}",
                block_indices=block_indices,
                block_ids=block_ids,
                scores=scores,
                gold_indices=gold,
                gold_retrieved=gold_retrieved,
            ))

        return results


class TorchCosineRouter(BaseRouter):
    """Pure-torch cosine similarity router.

    Simple and dependency-free. Fine for small banks (<10K blocks).
    For larger banks, use FaissRouter.
    """

    def __init__(self) -> None:
        self._vectors: torch.Tensor | None = None
        self._block_ids: list[str] = []

    def build_index(
        self,
        routing_vectors: torch.Tensor,
        block_ids: list[str] | None = None,
    ) -> None:
        # Normalize and store on CPU
        self._vectors = F.normalize(routing_vectors.float().cpu(), p=2, dim=1)
        self._block_ids = block_ids or [f"block_{i}" for i in range(routing_vectors.shape[0])]
        logger.info(f"Torch cosine index built: {self._vectors.shape[0]} vectors")

    def query(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        gold_indices: list[int] | None = None,
    ) -> RetrievalResult:
        if self._vectors is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q = F.normalize(query_vector.float().unsqueeze(0), p=2, dim=1).cpu()  # (1, dim)
        similarities = (q @ self._vectors.T).squeeze(0)  # (num_blocks,)

        k = min(top_k, self._vectors.shape[0])
        top_scores, top_indices = torch.topk(similarities, k)

        block_indices = top_indices.tolist()
        scores = top_scores.tolist()
        block_ids = [self._block_ids[i] for i in block_indices]

        gold_indices = gold_indices or []
        gold_retrieved = [gi in set(block_indices) for gi in gold_indices]

        return RetrievalResult(
            block_indices=block_indices,
            block_ids=block_ids,
            scores=scores,
            gold_indices=gold_indices,
            gold_retrieved=gold_retrieved,
        )


class OracleRouter(BaseRouter):
    """Oracle router that always returns the correct gold blocks.

    Used as an upper bound in experiments — perfect retrieval to isolate
    the effect of compression or context injection from routing errors.

    Requires gold_indices to be passed at query time.
    """

    def __init__(self) -> None:
        self._vectors: torch.Tensor | None = None
        self._block_ids: list[str] = []

    def build_index(
        self,
        routing_vectors: torch.Tensor,
        block_ids: list[str] | None = None,
    ) -> None:
        self._vectors = routing_vectors
        self._block_ids = block_ids or [f"block_{i}" for i in range(routing_vectors.shape[0])]
        logger.info(f"Oracle router initialized: {routing_vectors.shape[0]} blocks")

    def query(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        gold_indices: list[int] | None = None,
    ) -> RetrievalResult:
        if gold_indices is None or len(gold_indices) == 0:
            logger.warning("OracleRouter called without gold_indices — returning empty result")
            return RetrievalResult()

        # Return gold blocks + pad with random others if needed
        block_indices = list(gold_indices)
        if len(block_indices) < top_k and self._vectors is not None:
            all_indices = set(range(self._vectors.shape[0]))
            remaining = list(all_indices - set(block_indices))
            pad_count = min(top_k - len(block_indices), len(remaining))
            block_indices.extend(remaining[:pad_count])

        block_indices = block_indices[:top_k]
        scores = [1.0 if i in set(gold_indices) else 0.0 for i in block_indices]
        block_ids = [self._block_ids[i] for i in block_indices]
        gold_retrieved = [True] * len(gold_indices)

        return RetrievalResult(
            block_indices=block_indices,
            block_ids=block_ids,
            scores=scores,
            gold_indices=list(gold_indices),
            gold_retrieved=gold_retrieved,
        )


def create_router(engine: str = "faiss") -> BaseRouter:
    """Factory function to create a router by engine name.

    Args:
        engine: "faiss", "torch_cosine", or "oracle".

    Returns:
        A BaseRouter instance.
    """
    if engine == "faiss":
        return FaissRouter()
    elif engine == "torch_cosine":
        return TorchCosineRouter()
    elif engine == "oracle":
        return OracleRouter()
    else:
        raise ValueError(f"Unknown router engine: {engine}. Use 'faiss', 'torch_cosine', or 'oracle'.")
