"""Bank caching — build once, reuse across all experiments.

Maps (num_blocks, block_chars, seed) to a disk directory under a cache root.
Uses save_bank/load_bank from bank_store for persistence and load_kv_for_blocks
for selective on-demand KV loading at scale.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from src.eval.niah import NIAHSample, generate_niah_sample
from src.memory.bank_builder import MemoryBank, MemoryBankBuilder
from src.memory.bank_store import (
    load_bank,
    load_kv_for_blocks,
    load_routing_vectors,
    save_bank,
)
from src.memory.chunking import TextBlock
from src.models.base_model import BaseModel

logger = logging.getLogger("msa_turboquant.memory.bank_cache")


def _build_text_blocks_from_niah(sample: NIAHSample) -> list[TextBlock]:
    """Convert NIAH sample blocks into TextBlock objects for bank building."""
    blocks = []
    for i, text in enumerate(sample.blocks):
        is_needle = i in sample.needle_block_indices
        blocks.append(TextBlock(
            block_id=f"{sample.sample_id}_blk_{i}",
            document_id=sample.sample_id,
            block_index=i,
            text=text,
            char_start=0,
            char_end=len(text),
            metadata={"is_needle": is_needle},
        ))
    return blocks


class BankCache:
    """Cache pre-built memory banks to disk. Build once, reuse across experiments.

    Usage:
        cache = BankCache(Path("data/bank_cache"), model)
        bank = cache.get_or_build(num_blocks=100, block_chars=500, seed=42)
    """

    def __init__(
        self,
        cache_dir: Path,
        model: BaseModel,
        extraction_mode: str = "direct",
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._model = model
        self._extraction_mode = extraction_mode

    @staticmethod
    def cache_key(num_blocks: int, block_chars: int, seed: int) -> str:
        """Deterministic cache key for a bank configuration."""
        return f"bank_nb{num_blocks}_bc{block_chars}_s{seed}"

    def bank_dir(self, num_blocks: int, block_chars: int, seed: int) -> Path:
        """Path to the cached bank directory."""
        return self._cache_dir / self.cache_key(num_blocks, block_chars, seed)

    def exists(self, num_blocks: int, block_chars: int, seed: int) -> bool:
        """Check if a cached bank exists on disk."""
        bd = self.bank_dir(num_blocks, block_chars, seed)
        return (bd / "metadata.json").exists()

    def get_or_build(
        self,
        num_blocks: int,
        block_chars: int,
        seed: int,
        load_kv: bool = False,
        save_kv: bool = True,
    ) -> MemoryBank:
        """Return cached bank if it exists, otherwise build, save, and return.

        Args:
            num_blocks: Number of blocks in the NIAH sample.
            block_chars: Characters per block.
            seed: Random seed for NIAH sample generation.
            load_kv: Whether to load KV tensors into RAM. Default False
                to avoid OOM on large banks. KV injection modes re-encode
                context anyway, so pre-loaded KV is rarely needed.
            save_kv: Whether to save KV tensors to disk. False saves only
                routing vectors + metadata (~MB instead of ~GB). Use False
                for large banks when only routing vectors are needed.

        Returns:
            A MemoryBank loaded from cache or freshly built.
        """
        bd = self.bank_dir(num_blocks, block_chars, seed)

        if self.exists(num_blocks, block_chars, seed):
            logger.info(f"Loading cached bank from {bd}")
            start = time.perf_counter()
            bank = load_bank(bd, load_kv=load_kv, mmap_mode="r" if load_kv else None)
            elapsed = time.perf_counter() - start
            logger.info(f"Cached bank loaded: {bank.num_blocks} blocks in {elapsed:.1f}s")
            return bank

        # Build from scratch
        logger.info(f"Building bank: {num_blocks} blocks, {block_chars} chars, seed={seed}")
        sample = generate_niah_sample(
            num_blocks=num_blocks,
            block_chars=block_chars,
            seed=seed,
        )
        text_blocks = _build_text_blocks_from_niah(sample)

        builder = MemoryBankBuilder(self._model, extraction_mode=self._extraction_mode)
        start = time.perf_counter()
        bank = builder.build(text_blocks, bank_id=self.cache_key(num_blocks, block_chars, seed))
        build_time = time.perf_counter() - start

        # Save to disk (skip KV if requested — saves ~50 GB per 4000-block bank)
        save_bank(bank, bd, save_kv=save_kv)
        logger.info(f"Bank built and cached: {num_blocks} blocks in {build_time:.1f}s -> {bd}")

        # Clear KV from memory after saving — only routing vectors needed
        if not save_kv:
            import gc
            for kb in bank.kv_blocks:
                kb.keys.clear()
                kb.values.clear()
            gc.collect()
            torch.cuda.empty_cache()

        return bank

    def load_routing_only(
        self,
        num_blocks: int,
        block_chars: int,
        seed: int,
    ) -> tuple[torch.Tensor, list[str]]:
        """Load only routing vectors and block IDs (fast, for routing-only use).

        Useful for large banks where loading all KV into RAM is impractical.
        After routing selects top-k, use load_kv_for_selected() to fetch just those.

        Returns:
            Tuple of (routing_vectors tensor, list of block_ids).
        """
        bd = self.bank_dir(num_blocks, block_chars, seed)
        if not (bd / "routing_vectors.npy").exists():
            raise FileNotFoundError(
                f"No cached bank at {bd}. Call get_or_build() first."
            )
        return load_routing_vectors(bd)

    def load_kv_for_selected(
        self,
        num_blocks: int,
        block_chars: int,
        seed: int,
        block_indices: list[int],
    ) -> list:
        """Load KV tensors for specific blocks only (selective fetch from disk).

        Args:
            num_blocks: Bank configuration parameters.
            block_chars: Bank configuration parameters.
            seed: Bank configuration parameters.
            block_indices: Which block indices to load.

        Returns:
            List of KVBlock objects for the requested blocks only.
        """
        bd = self.bank_dir(num_blocks, block_chars, seed)
        return load_kv_for_blocks(bd, block_indices)
