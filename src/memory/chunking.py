"""Text chunking utilities for splitting documents into memory bank blocks.

Supports both character-based and token-based chunking with configurable
overlap. Each chunk carries rich metadata for downstream tracing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class TextBlock:
    """A chunk of text with metadata for memory bank construction.

    Attributes:
        block_id: Unique identifier for this block (content hash).
        document_id: Identifier for the source document.
        block_index: Sequential index of this block within the document.
        text: The raw text content of this block.
        char_start: Character offset of the block start in the original document.
        char_end: Character offset of the block end in the original document.
        token_ids: Optional pre-computed token IDs for this block.
        metadata: Arbitrary extra metadata (e.g., needle label, is_distractor).
    """
    block_id: str
    document_id: str
    block_index: int
    text: str
    char_start: int
    char_end: int
    token_ids: Optional[list[int]] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return self.char_end - self.char_start

    @property
    def num_tokens(self) -> int | None:
        return len(self.token_ids) if self.token_ids is not None else None


def _content_hash(text: str, doc_id: str, index: int) -> str:
    """Generate a short deterministic hash for a block."""
    payload = f"{doc_id}::{index}::{text[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    document_id: str = "doc_0",
    metadata: dict | None = None,
) -> list[TextBlock]:
    """Split text into character-based chunks with overlap.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        document_id: Identifier for the source document.
        metadata: Extra metadata to attach to every block.

    Returns:
        List of TextBlock objects covering the full document.

    Raises:
        ValueError: If chunk_size < 1 or chunk_overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )
    if not text:
        return []

    base_meta = metadata or {}
    blocks: list[TextBlock] = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        block_text = text[start:end]
        blocks.append(TextBlock(
            block_id=_content_hash(block_text, document_id, index),
            document_id=document_id,
            block_index=index,
            text=block_text,
            char_start=start,
            char_end=end,
            metadata=dict(base_meta),
        ))
        index += 1
        start += step
        # Avoid creating a tiny trailing block that's just overlap
        if start < len(text) and (len(text) - start) < chunk_overlap:
            # Extend the last block to cover the remainder
            remainder = text[start:]
            last = blocks[-1]
            blocks[-1] = TextBlock(
                block_id=_content_hash(last.text + remainder, document_id, last.block_index),
                document_id=document_id,
                block_index=last.block_index,
                text=last.text + remainder,
                char_start=last.char_start,
                char_end=len(text),
                token_ids=last.token_ids,
                metadata=last.metadata,
            )
            break

    return blocks


def chunk_by_tokens(
    token_ids: Sequence[int],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    document_id: str = "doc_0",
    text: str | None = None,
    metadata: dict | None = None,
) -> list[TextBlock]:
    """Split a token sequence into fixed-size chunks with overlap.

    Args:
        token_ids: The full token ID sequence.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Overlapping tokens between consecutive chunks.
        document_id: Identifier for the source document.
        text: Optional original text (for reference only, not split here).
        metadata: Extra metadata to attach to every block.

    Returns:
        List of TextBlock objects with token_ids populated.
        The text field is set to a placeholder since we don't have a detokenizer here.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )
    if not token_ids:
        return []

    base_meta = metadata or {}
    blocks: list[TextBlock] = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0
    total = len(token_ids)

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = list(token_ids[start:end])

        blocks.append(TextBlock(
            block_id=_content_hash(str(chunk_tokens[:20]), document_id, index),
            document_id=document_id,
            block_index=index,
            text=text[start:end] if text else f"[tokens {start}:{end}]",
            char_start=start,  # token offsets, not char offsets
            char_end=end,
            token_ids=chunk_tokens,
            metadata={**base_meta, "chunking_mode": "token"},
        ))
        index += 1
        start += step

    return blocks


def merge_blocks(blocks: list[TextBlock], separator: str = "") -> str:
    """Reconstruct text from a list of non-overlapping TextBlocks.

    Warning: If blocks have overlap, the overlapping text will be duplicated.
    For non-overlapping sequential blocks this gives the original text.
    """
    return separator.join(b.text for b in sorted(blocks, key=lambda b: b.block_index))
