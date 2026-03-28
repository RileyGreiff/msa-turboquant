"""Offline memory bank construction pipeline.

Takes text blocks (from chunking) and a model, produces a MemoryBank containing
routing vectors and KV representations for each block. The bank can then be
persisted via bank_store.

This is the main offline preprocessing step before retrieval experiments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from src.memory.chunking import TextBlock
from src.models.base_model import BaseModel
from src.models.kv_extractor import KVBlock, KVExtractor

logger = logging.getLogger("msa_turboquant.memory.bank_builder")


@dataclass
class MemoryBankMetadata:
    """Metadata for a memory bank.

    Attributes:
        bank_id: Unique identifier for this bank.
        num_blocks: Total number of blocks stored.
        hidden_dim: Dimension of routing vectors.
        num_layers: Number of KV layers stored per block.
        num_heads: Number of KV heads per layer.
        head_dim: Dimension per attention head.
        model_name: Name of the model used to build the bank.
        extraction_mode: "direct" or "hidden_state".
        layer_indices: Which model layers are stored.
        total_tokens: Sum of tokens across all blocks.
        build_time_seconds: Time taken to build the bank.
        extra: Arbitrary extra info.
    """
    bank_id: str = "default"
    num_blocks: int = 0
    hidden_dim: int = 0
    num_layers: int = 0
    num_heads: int = 0
    head_dim: int = 0
    model_name: str = ""
    extraction_mode: str = "direct"
    layer_indices: list[int] = field(default_factory=list)
    total_tokens: int = 0
    build_time_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bank_id": self.bank_id,
            "num_blocks": self.num_blocks,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "model_name": self.model_name,
            "extraction_mode": self.extraction_mode,
            "layer_indices": self.layer_indices,
            "total_tokens": self.total_tokens,
            "build_time_seconds": self.build_time_seconds,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryBankMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BlockMetadataEntry:
    """Per-block metadata stored in the bank index.

    Attributes:
        block_id: Unique block identifier.
        document_id: Source document identifier.
        block_index: Index within the source document.
        num_tokens: Number of tokens in this block.
        char_start: Character offset start in source document.
        char_end: Character offset end in source document.
        text_preview: First N characters of the block text.
        extra: Arbitrary extra metadata (e.g., is_needle, needle_id).
    """
    block_id: str
    document_id: str = ""
    block_index: int = 0
    num_tokens: int = 0
    char_start: int = 0
    char_end: int = 0
    text_preview: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "document_id": self.document_id,
            "block_index": self.block_index,
            "num_tokens": self.num_tokens,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text_preview": self.text_preview,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockMetadataEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryBank:
    """In-memory representation of a memory bank.

    Contains routing vectors and KV blocks for all stored text blocks,
    plus metadata for the bank and each individual block.

    Attributes:
        metadata: Bank-level metadata.
        block_metadata: Per-block metadata entries, ordered by index.
        routing_vectors: Stacked routing vectors. Shape: (num_blocks, hidden_dim).
        kv_blocks: List of KVBlock objects (K/V tensors on CPU).
    """
    metadata: MemoryBankMetadata
    block_metadata: list[BlockMetadataEntry]
    routing_vectors: torch.Tensor  # (num_blocks, hidden_dim)
    kv_blocks: list[KVBlock]

    @property
    def num_blocks(self) -> int:
        return len(self.kv_blocks)

    def get_block_ids(self) -> list[str]:
        return [bm.block_id for bm in self.block_metadata]

    def total_kv_bytes(self) -> int:
        return sum(kb.total_kv_bytes for kb in self.kv_blocks)


class MemoryBankBuilder:
    """Builds a MemoryBank from text blocks and a model.

    Usage:
        builder = MemoryBankBuilder(model, extraction_mode="direct")
        bank = builder.build(text_blocks, bank_id="my_bank")
    """

    def __init__(
        self,
        model: BaseModel,
        extraction_mode: str = "direct",
        layers: list[int] | None = None,
        text_preview_chars: int = 100,
    ) -> None:
        """
        Args:
            model: A loaded BaseModel instance.
            extraction_mode: "direct" or "hidden_state" for KV extraction.
            layers: Which model layers to extract. None = all.
            text_preview_chars: How many chars to store in block metadata preview.
        """
        self._model = model
        self._extractor = KVExtractor(model, mode=extraction_mode, layers=layers)
        self._extraction_mode = extraction_mode
        self._layers = layers
        self._preview_chars = text_preview_chars

    def build(
        self,
        text_blocks: list[TextBlock],
        bank_id: str = "default",
        extra_metadata: dict | None = None,
    ) -> MemoryBank:
        """Build a memory bank from text blocks.

        Args:
            text_blocks: List of TextBlock objects from chunking.
            bank_id: Unique identifier for this bank.
            extra_metadata: Extra info to store in bank metadata.

        Returns:
            A MemoryBank with routing vectors and KV blocks.
        """
        if not text_blocks:
            raise ValueError("Cannot build bank from empty text_blocks list")

        logger.info(f"Building memory bank '{bank_id}' from {len(text_blocks)} blocks")
        start_time = time.perf_counter()

        routing_vectors: list[torch.Tensor] = []
        kv_blocks: list[KVBlock] = []
        block_metadata: list[BlockMetadataEntry] = []

        for i, tb in enumerate(text_blocks):
            if i % 10 == 0:
                logger.info(f"  Processing block {i+1}/{len(text_blocks)}: {tb.block_id}")

            # Extract KV and routing vector
            kv_block = self._extractor.extract(
                text=tb.text,
                block_id=tb.block_id,
            )

            routing_vectors.append(kv_block.routing_vector)
            kv_blocks.append(kv_block.to_cpu())

            # Build per-block metadata
            entry = BlockMetadataEntry(
                block_id=tb.block_id,
                document_id=tb.document_id,
                block_index=tb.block_index,
                num_tokens=kv_block.num_tokens,
                char_start=tb.char_start,
                char_end=tb.char_end,
                text_preview=tb.text[:self._preview_chars],
                extra=dict(tb.metadata),
            )
            block_metadata.append(entry)

        # Stack routing vectors into a single tensor
        stacked_routing = torch.stack(routing_vectors, dim=0)  # (num_blocks, hidden_dim)

        build_time = time.perf_counter() - start_time

        # Build bank metadata
        sample_kv = kv_blocks[0]
        bank_meta = MemoryBankMetadata(
            bank_id=bank_id,
            num_blocks=len(kv_blocks),
            hidden_dim=stacked_routing.shape[1],
            num_layers=sample_kv.num_layers,
            num_heads=sample_kv.keys[0].shape[0] if sample_kv.keys else 0,
            head_dim=sample_kv.keys[0].shape[2] if sample_kv.keys else 0,
            model_name=getattr(self._model, "model_name", "unknown"),
            extraction_mode=self._extraction_mode,
            layer_indices=sample_kv.layer_indices,
            total_tokens=sum(kb.num_tokens for kb in kv_blocks),
            build_time_seconds=build_time,
            extra=extra_metadata or {},
        )

        bank = MemoryBank(
            metadata=bank_meta,
            block_metadata=block_metadata,
            routing_vectors=stacked_routing,
            kv_blocks=kv_blocks,
        )

        logger.info(
            f"Bank '{bank_id}' built: {bank.num_blocks} blocks, "
            f"{bank_meta.total_tokens} total tokens, "
            f"{bank.total_kv_bytes() / (1024**2):.1f} MB KV data, "
            f"{build_time:.1f}s"
        )

        return bank

    def build_from_texts(
        self,
        texts: list[str],
        document_id: str = "doc_0",
        bank_id: str = "default",
    ) -> MemoryBank:
        """Convenience: build a bank directly from raw text strings.

        Each string becomes one block (no chunking applied here).
        Use chunking.chunk_text() first if you need to split documents.
        """
        from src.memory.chunking import TextBlock, _content_hash

        text_blocks = []
        for i, text in enumerate(texts):
            tb = TextBlock(
                block_id=_content_hash(text, document_id, i),
                document_id=document_id,
                block_index=i,
                text=text,
                char_start=0,
                char_end=len(text),
            )
            text_blocks.append(tb)

        return self.build(text_blocks, bank_id=bank_id)
