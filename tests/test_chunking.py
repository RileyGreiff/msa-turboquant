"""Tests for text chunking utilities."""

from __future__ import annotations

import pytest

from src.memory.chunking import TextBlock, chunk_by_tokens, chunk_text, merge_blocks


class TestChunkText:
    """Tests for character-based chunking."""

    def test_basic_chunking(self) -> None:
        """Chunking a string produces blocks covering the full text."""
        text = "A" * 1000
        blocks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        assert len(blocks) == 4  # 300+300+300+100
        assert all(isinstance(b, TextBlock) for b in blocks)

    def test_overlap_produces_more_blocks(self) -> None:
        """Overlap causes more blocks than non-overlapping chunking."""
        text = "A" * 2000
        no_overlap = chunk_text(text, chunk_size=300, chunk_overlap=0)
        with_overlap = chunk_text(text, chunk_size=300, chunk_overlap=100)
        assert len(with_overlap) > len(no_overlap)

    def test_single_block_short_text(self) -> None:
        """Text shorter than chunk_size produces one block."""
        text = "Hello world"
        blocks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        assert len(blocks) == 1
        assert blocks[0].text == text

    def test_empty_text_returns_empty(self) -> None:
        """Empty string produces no blocks."""
        assert chunk_text("", chunk_size=100, chunk_overlap=0) == []

    def test_block_metadata(self) -> None:
        """Blocks carry correct document_id and sequential indices."""
        text = "X" * 500
        blocks = chunk_text(text, chunk_size=200, chunk_overlap=0, document_id="mydoc")
        assert all(b.document_id == "mydoc" for b in blocks)
        assert [b.block_index for b in blocks] == list(range(len(blocks)))

    def test_char_offsets_correct(self) -> None:
        """char_start and char_end correctly reference the original text."""
        text = "ABCDEFGHIJ" * 50  # 500 chars
        blocks = chunk_text(text, chunk_size=200, chunk_overlap=0)
        for block in blocks:
            assert text[block.char_start:block.char_end] == block.text

    def test_custom_metadata_attached(self) -> None:
        """Custom metadata is passed through to each block."""
        blocks = chunk_text("test" * 100, chunk_size=50, chunk_overlap=0,
                            metadata={"source": "test", "is_needle": False})
        for b in blocks:
            assert b.metadata["source"] == "test"
            assert b.metadata["is_needle"] is False

    def test_block_ids_unique(self) -> None:
        """Each block gets a unique block_id."""
        text = "Hello world. " * 100
        blocks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        ids = [b.block_id for b in blocks]
        assert len(ids) == len(set(ids))

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=0, chunk_overlap=0)

    def test_overlap_exceeds_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("hello", chunk_size=10, chunk_overlap=10)

    def test_full_coverage(self) -> None:
        """All characters in the original text appear in at least one block."""
        text = "ABCDEFGHIJ" * 100
        blocks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        covered = set()
        for b in blocks:
            covered.update(range(b.char_start, b.char_end))
        assert covered == set(range(len(text)))


class TestChunkByTokens:
    """Tests for token-based chunking."""

    def test_basic_token_chunking(self) -> None:
        """Token chunking splits a sequence correctly."""
        tokens = list(range(100))
        blocks = chunk_by_tokens(tokens, chunk_size=30, chunk_overlap=0)
        assert len(blocks) == 4  # 30+30+30+10
        assert blocks[0].token_ids == list(range(30))

    def test_overlap_in_tokens(self) -> None:
        """Token overlap produces overlapping token ID sequences."""
        tokens = list(range(50))
        blocks = chunk_by_tokens(tokens, chunk_size=20, chunk_overlap=5)
        # First block: 0-19, second: 15-34, etc.
        assert blocks[0].token_ids[-5:] == blocks[1].token_ids[:5]

    def test_empty_tokens(self) -> None:
        assert chunk_by_tokens([], chunk_size=10, chunk_overlap=0) == []

    def test_num_tokens_property(self) -> None:
        """num_tokens property returns the correct count."""
        tokens = list(range(100))
        blocks = chunk_by_tokens(tokens, chunk_size=30, chunk_overlap=0)
        assert blocks[0].num_tokens == 30


class TestMergeBlocks:
    """Tests for block merging."""

    def test_merge_non_overlapping(self) -> None:
        """Merging non-overlapping blocks reconstructs original text."""
        text = "Hello World! This is a test."
        blocks = chunk_text(text, chunk_size=15, chunk_overlap=0)
        merged = merge_blocks(blocks)
        assert merged == text

    def test_merge_preserves_order(self) -> None:
        """Blocks are merged in block_index order regardless of input order."""
        text = "AABBCCDDEE"
        blocks = chunk_text(text, chunk_size=4, chunk_overlap=0)
        reversed_blocks = list(reversed(blocks))
        merged = merge_blocks(reversed_blocks)
        assert merged == merge_blocks(blocks)
