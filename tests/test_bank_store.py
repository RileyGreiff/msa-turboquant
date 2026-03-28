"""Tests for memory bank building, saving, and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.memory.bank_builder import (
    BlockMetadataEntry,
    MemoryBank,
    MemoryBankBuilder,
    MemoryBankMetadata,
)
from src.memory.bank_store import (
    load_bank,
    load_kv_for_blocks,
    load_routing_vectors,
    save_bank,
)
from src.memory.chunking import TextBlock, chunk_text
from src.models.kv_extractor import KVBlock

# Import DummyModel from test_models for reuse
from tests.test_models import DummyModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_model():
    """A loaded DummyModel for bank building tests."""
    model = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
    model.load()
    return model


@pytest.fixture
def sample_text_blocks() -> list[TextBlock]:
    """Generate sample text blocks for testing."""
    text = "This is a test document. " * 200  # ~5000 chars
    return chunk_text(text, chunk_size=500, chunk_overlap=0, document_id="test_doc")


@pytest.fixture
def sample_bank(dummy_model, sample_text_blocks) -> MemoryBank:
    """Build a sample memory bank."""
    builder = MemoryBankBuilder(dummy_model, extraction_mode="direct")
    return builder.build(sample_text_blocks, bank_id="test_bank")


# ---------------------------------------------------------------------------
# MemoryBankBuilder tests
# ---------------------------------------------------------------------------

class TestMemoryBankBuilder:
    """Tests for building memory banks."""

    def test_build_basic(self, dummy_model, sample_text_blocks) -> None:
        """Builder produces a valid MemoryBank."""
        builder = MemoryBankBuilder(dummy_model, extraction_mode="direct")
        bank = builder.build(sample_text_blocks, bank_id="test")

        assert isinstance(bank, MemoryBank)
        assert bank.num_blocks == len(sample_text_blocks)
        assert bank.metadata.bank_id == "test"
        assert bank.metadata.num_blocks == len(sample_text_blocks)

    def test_routing_vectors_shape(self, sample_bank) -> None:
        """Routing vectors have correct shape."""
        assert sample_bank.routing_vectors.shape == (
            sample_bank.num_blocks,
            sample_bank.metadata.hidden_dim,
        )

    def test_routing_vectors_normalized(self, sample_bank) -> None:
        """Routing vectors are L2-normalized."""
        norms = torch.linalg.norm(sample_bank.routing_vectors, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_kv_blocks_present(self, sample_bank) -> None:
        """Each block has KV tensors."""
        for kb in sample_bank.kv_blocks:
            assert isinstance(kb, KVBlock)
            assert kb.num_layers > 0
            assert len(kb.keys) == kb.num_layers

    def test_block_metadata(self, sample_bank, sample_text_blocks) -> None:
        """Block metadata matches source text blocks."""
        assert len(sample_bank.block_metadata) == len(sample_text_blocks)
        for bm, tb in zip(sample_bank.block_metadata, sample_text_blocks):
            assert bm.block_id == tb.block_id
            assert bm.document_id == tb.document_id
            assert bm.text_preview == tb.text[:100]

    def test_bank_metadata_populated(self, sample_bank) -> None:
        """Bank-level metadata is correctly populated."""
        meta = sample_bank.metadata
        assert meta.hidden_dim == 64
        assert meta.num_layers == 2
        assert meta.num_heads == 4
        assert meta.head_dim == 16
        assert meta.total_tokens > 0
        assert meta.build_time_seconds > 0

    def test_hidden_state_mode(self, dummy_model, sample_text_blocks) -> None:
        """Building with hidden_state mode also works."""
        builder = MemoryBankBuilder(dummy_model, extraction_mode="hidden_state")
        bank = builder.build(sample_text_blocks[:3], bank_id="hs_test")
        assert bank.num_blocks == 3
        assert bank.kv_blocks[0].num_layers == 2

    def test_build_from_texts(self, dummy_model) -> None:
        """Convenience build_from_texts works."""
        builder = MemoryBankBuilder(dummy_model, extraction_mode="direct")
        bank = builder.build_from_texts(
            ["Hello world", "Foo bar baz"],
            document_id="test",
            bank_id="text_bank",
        )
        assert bank.num_blocks == 2

    def test_empty_blocks_raises(self, dummy_model) -> None:
        """Building from empty list raises ValueError."""
        builder = MemoryBankBuilder(dummy_model)
        with pytest.raises(ValueError, match="empty"):
            builder.build([], bank_id="empty")

    def test_total_kv_bytes(self, sample_bank) -> None:
        """total_kv_bytes returns a positive number."""
        assert sample_bank.total_kv_bytes() > 0


# ---------------------------------------------------------------------------
# Save/Load round-trip tests
# ---------------------------------------------------------------------------

class TestBankPersistence:
    """Tests for saving and loading memory banks."""

    def test_save_creates_files(self, sample_bank, tmp_path) -> None:
        """Saving a bank creates the expected file structure."""
        bank_dir = tmp_path / "test_bank"
        save_bank(sample_bank, bank_dir)

        assert (bank_dir / "metadata.json").exists()
        assert (bank_dir / "block_index.json").exists()
        assert (bank_dir / "routing_vectors.npy").exists()
        assert (bank_dir / "token_counts.npy").exists()
        assert (bank_dir / "shape_info.json").exists()
        assert (bank_dir / "kv" / "layer_0_keys.npy").exists()
        assert (bank_dir / "kv" / "layer_0_values.npy").exists()

    def test_round_trip_metadata(self, sample_bank, tmp_path) -> None:
        """Bank metadata survives save/load."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)
        loaded = load_bank(bank_dir)

        assert loaded.metadata.bank_id == sample_bank.metadata.bank_id
        assert loaded.metadata.num_blocks == sample_bank.metadata.num_blocks
        assert loaded.metadata.hidden_dim == sample_bank.metadata.hidden_dim
        assert loaded.metadata.num_layers == sample_bank.metadata.num_layers
        assert loaded.metadata.num_heads == sample_bank.metadata.num_heads
        assert loaded.metadata.head_dim == sample_bank.metadata.head_dim

    def test_round_trip_routing_vectors(self, sample_bank, tmp_path) -> None:
        """Routing vectors survive save/load (float32)."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)
        loaded = load_bank(bank_dir)

        assert loaded.routing_vectors.shape == sample_bank.routing_vectors.shape
        assert torch.allclose(
            loaded.routing_vectors, sample_bank.routing_vectors, atol=1e-6
        )

    def test_round_trip_kv_tensors(self, sample_bank, tmp_path) -> None:
        """KV tensors survive save/load (fp16 precision)."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)
        loaded = load_bank(bank_dir)

        assert len(loaded.kv_blocks) == len(sample_bank.kv_blocks)
        for orig, reloaded in zip(sample_bank.kv_blocks, loaded.kv_blocks):
            assert orig.block_id == reloaded.block_id
            assert orig.num_tokens == reloaded.num_tokens
            assert orig.num_layers == reloaded.num_layers
            for li in range(orig.num_layers):
                # fp16 round-trip: compare in fp16 space
                orig_k = orig.keys[li].to(torch.float16)
                assert torch.allclose(orig_k, reloaded.keys[li], atol=1e-3)

    def test_round_trip_block_metadata(self, sample_bank, tmp_path) -> None:
        """Block metadata index survives save/load."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)
        loaded = load_bank(bank_dir)

        assert len(loaded.block_metadata) == len(sample_bank.block_metadata)
        for orig, reloaded in zip(sample_bank.block_metadata, loaded.block_metadata):
            assert orig.block_id == reloaded.block_id
            assert orig.document_id == reloaded.document_id
            assert orig.num_tokens == reloaded.num_tokens

    def test_load_without_kv(self, sample_bank, tmp_path) -> None:
        """Loading with load_kv=False returns empty K/V but valid routing."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)
        loaded = load_bank(bank_dir, load_kv=False)

        assert loaded.num_blocks == sample_bank.num_blocks
        assert loaded.routing_vectors.shape == sample_bank.routing_vectors.shape
        # KV blocks exist but have empty keys/values
        for kb in loaded.kv_blocks:
            assert len(kb.keys) == 0

    def test_load_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_bank("/nonexistent/bank/dir")


# ---------------------------------------------------------------------------
# Selective loading tests
# ---------------------------------------------------------------------------

class TestSelectiveLoading:
    """Tests for loading specific blocks or routing vectors only."""

    def test_load_routing_vectors_only(self, sample_bank, tmp_path) -> None:
        """load_routing_vectors returns vectors and block IDs."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)

        vectors, block_ids = load_routing_vectors(bank_dir)
        assert vectors.shape == sample_bank.routing_vectors.shape
        assert len(block_ids) == sample_bank.num_blocks
        assert block_ids[0] == sample_bank.block_metadata[0].block_id

    def test_load_kv_for_specific_blocks(self, sample_bank, tmp_path) -> None:
        """load_kv_for_blocks loads only requested block indices."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)

        # Load blocks 0 and 2 only
        indices = [0, 2]
        blocks = load_kv_for_blocks(bank_dir, indices)

        assert len(blocks) == 2
        assert blocks[0].block_id == sample_bank.block_metadata[0].block_id
        assert blocks[1].block_id == sample_bank.block_metadata[2].block_id
        # KV tensors should be present
        assert blocks[0].num_layers > 0

    def test_load_kv_specific_layers(self, sample_bank, tmp_path) -> None:
        """load_kv_for_blocks can load a subset of layers."""
        bank_dir = tmp_path / "bank"
        save_bank(sample_bank, bank_dir)

        blocks = load_kv_for_blocks(bank_dir, [0], layers=[0])
        assert blocks[0].num_layers == 1
        assert blocks[0].layer_indices == [0]


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestDataStructures:
    """Tests for metadata serialization."""

    def test_bank_metadata_round_trip(self) -> None:
        meta = MemoryBankMetadata(
            bank_id="test",
            num_blocks=10,
            hidden_dim=64,
            model_name="test_model",
        )
        d = meta.to_dict()
        restored = MemoryBankMetadata.from_dict(d)
        assert restored.bank_id == "test"
        assert restored.num_blocks == 10

    def test_block_metadata_round_trip(self) -> None:
        entry = BlockMetadataEntry(
            block_id="blk_0",
            document_id="doc_0",
            block_index=5,
            num_tokens=128,
            text_preview="Hello world...",
            extra={"is_needle": True},
        )
        d = entry.to_dict()
        restored = BlockMetadataEntry.from_dict(d)
        assert restored.block_id == "blk_0"
        assert restored.extra["is_needle"] is True
