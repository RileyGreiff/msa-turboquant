"""Sparse retrieval pipeline: route query -> fetch top-k KV blocks.

Combines the router (for finding relevant blocks) with bank_store (for
loading their KV data) into a single retrieval interface. This is the
main entry point used by the evaluation harness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.memory.bank_builder import MemoryBank
from src.memory.bank_store import load_kv_for_blocks, load_routing_vectors
from src.memory.router import BaseRouter, RetrievalResult, create_router
from src.models.kv_extractor import KVBlock

logger = logging.getLogger("msa_turboquant.memory.fetcher")


@dataclass
class FetchResult:
    """Result of a sparse retrieval + fetch operation.

    Attributes:
        retrieval: The routing result (indices, scores, gold match info).
        kv_blocks: The fetched KVBlock objects for the retrieved blocks.
        fetch_time_ms: Time to fetch KV data from storage (ms).
        route_time_ms: Time for routing query (ms).
        total_time_ms: Total retrieval + fetch time (ms).
    """
    retrieval: RetrievalResult
    kv_blocks: list[KVBlock] = field(default_factory=list)
    fetch_time_ms: float = 0.0
    route_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def num_fetched(self) -> int:
        return len(self.kv_blocks)

    @property
    def total_kv_bytes(self) -> int:
        return sum(kb.total_kv_bytes for kb in self.kv_blocks)

    def to_dict(self) -> dict:
        return {
            **self.retrieval.to_dict(),
            "num_fetched": self.num_fetched,
            "total_kv_bytes": self.total_kv_bytes,
            "fetch_time_ms": self.fetch_time_ms,
            "route_time_ms": self.route_time_ms,
            "total_time_ms": self.total_time_ms,
        }


class MemoryFetcher:
    """Sparse retrieval pipeline: route + fetch.

    Can operate in two modes:
    1. In-memory: bank is fully loaded in RAM (faster, more memory)
    2. Disk-backed: only routing vectors in RAM, KV fetched from disk on demand

    Usage (in-memory):
        fetcher = MemoryFetcher.from_bank(bank, engine="faiss")
        result = fetcher.fetch(query_vector, top_k=5)

    Usage (disk-backed):
        fetcher = MemoryFetcher.from_disk(bank_dir, engine="faiss")
        result = fetcher.fetch(query_vector, top_k=5)
    """

    def __init__(
        self,
        router: BaseRouter,
        bank: MemoryBank | None = None,
        bank_dir: Path | None = None,
        kv_layers: list[int] | None = None,
    ) -> None:
        """
        Args:
            router: A built BaseRouter with index ready.
            bank: In-memory bank (for in-memory mode).
            bank_dir: Path to saved bank (for disk-backed mode).
            kv_layers: Which KV layers to fetch. None = all.
        """
        self._router = router
        self._bank = bank
        self._bank_dir = bank_dir
        self._kv_layers = kv_layers

    @classmethod
    def from_bank(
        cls,
        bank: MemoryBank,
        engine: str = "faiss",
        kv_layers: list[int] | None = None,
    ) -> "MemoryFetcher":
        """Create a fetcher from an in-memory bank."""
        router = create_router(engine)
        router.build_index(bank.routing_vectors, bank.get_block_ids())
        return cls(router=router, bank=bank, kv_layers=kv_layers)

    @classmethod
    def from_disk(
        cls,
        bank_dir: Path | str,
        engine: str = "faiss",
        kv_layers: list[int] | None = None,
    ) -> "MemoryFetcher":
        """Create a fetcher from a saved bank on disk.

        Only loads routing vectors into RAM. KV data is fetched on demand.
        """
        bank_dir = Path(bank_dir)
        routing_vectors, block_ids = load_routing_vectors(bank_dir)
        router = create_router(engine)
        router.build_index(routing_vectors, block_ids)
        return cls(router=router, bank_dir=bank_dir, kv_layers=kv_layers)

    def fetch(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        gold_indices: list[int] | None = None,
        fetch_kv: bool = True,
    ) -> FetchResult:
        """Route a query and fetch the top-k blocks' KV data.

        Args:
            query_vector: Shape (hidden_dim,) or (1, hidden_dim).
            top_k: Number of blocks to retrieve.
            gold_indices: Ground-truth block indices for evaluation.
            fetch_kv: Whether to actually load KV tensors (False = route only).

        Returns:
            FetchResult with retrieval info and KV blocks.
        """
        total_start = time.perf_counter()

        # 1. Route
        route_start = time.perf_counter()
        retrieval = self._router.query(query_vector, top_k=top_k, gold_indices=gold_indices)
        route_time = (time.perf_counter() - route_start) * 1000

        # 2. Fetch KV
        kv_blocks: list[KVBlock] = []
        fetch_time = 0.0

        if fetch_kv and retrieval.block_indices:
            fetch_start = time.perf_counter()
            kv_blocks = self._fetch_kv_blocks(retrieval.block_indices)
            fetch_time = (time.perf_counter() - fetch_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"Fetch: top_k={top_k}, fetched={len(kv_blocks)}, "
            f"route={route_time:.1f}ms, fetch={fetch_time:.1f}ms, "
            f"total={total_time:.1f}ms"
        )

        return FetchResult(
            retrieval=retrieval,
            kv_blocks=kv_blocks,
            fetch_time_ms=fetch_time,
            route_time_ms=route_time,
            total_time_ms=total_time,
        )

    def fetch_batch(
        self,
        query_vectors: torch.Tensor,
        top_k: int = 5,
        gold_indices_list: list[list[int]] | None = None,
        fetch_kv: bool = True,
    ) -> list[FetchResult]:
        """Fetch for a batch of queries."""
        results = []
        for i in range(query_vectors.shape[0]):
            gold = gold_indices_list[i] if gold_indices_list else None
            result = self.fetch(query_vectors[i], top_k=top_k, gold_indices=gold, fetch_kv=fetch_kv)
            result.retrieval.query_id = f"query_{i}"
            results.append(result)
        return results

    def _fetch_kv_blocks(self, block_indices: list[int]) -> list[KVBlock]:
        """Load KV blocks from bank or disk."""
        if self._bank is not None:
            # In-memory mode: index directly into the bank's kv_blocks
            return [self._bank.kv_blocks[i] for i in block_indices]
        elif self._bank_dir is not None:
            # Disk-backed mode: selective load
            return load_kv_for_blocks(
                self._bank_dir, block_indices, layers=self._kv_layers
            )
        else:
            logger.warning("No bank or bank_dir configured — returning empty KV blocks")
            return []
