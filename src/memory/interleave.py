"""Context interleaving: inject retrieved memory blocks into generation context.

Assembles a prompt from retrieved text blocks and the user's query/question.
Supports multiple assembly strategies for how retrieved context is presented
to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("msa_turboquant.memory.interleave")


@dataclass
class AssembledContext:
    """The final context assembled for model generation.

    Attributes:
        text: The full assembled prompt text.
        num_retrieved_blocks: How many retrieved blocks were included.
        retrieved_block_ids: IDs of included blocks.
        context_chars: Total character count of the assembled context.
        query_text: The original query/question.
        strategy: Which assembly strategy was used.
    """
    text: str
    num_retrieved_blocks: int = 0
    retrieved_block_ids: list[str] = field(default_factory=list)
    context_chars: int = 0
    query_text: str = ""
    strategy: str = "prepend"

    def to_dict(self) -> dict:
        return {
            "num_retrieved_blocks": self.num_retrieved_blocks,
            "retrieved_block_ids": self.retrieved_block_ids,
            "context_chars": self.context_chars,
            "query_text": self.query_text,
            "strategy": self.strategy,
        }


def assemble_context(
    query: str,
    retrieved_texts: list[str],
    retrieved_block_ids: list[str] | None = None,
    strategy: Literal["prepend", "interleave", "summarize_prefix"] = "prepend",
    max_context_chars: int | None = None,
    block_separator: str = "\n\n",
    context_header: str = "Retrieved context:\n",
    query_header: str = "\nQuestion: ",
    answer_header: str = "\nAnswer: ",
) -> AssembledContext:
    """Assemble retrieved text blocks and a query into a generation prompt.

    Args:
        query: The user's question or prompt.
        retrieved_texts: Text content of retrieved memory blocks (ordered by relevance).
        retrieved_block_ids: Optional block IDs for tracking.
        strategy: How to combine context and query:
            - "prepend": Context blocks before query (most common).
            - "interleave": Numbered blocks with explicit references.
            - "summarize_prefix": Brief instruction prefix before context.
        max_context_chars: If set, truncate context to fit within this limit.
            The query is always included in full.
        block_separator: Separator between context blocks.
        context_header: Text before the context section.
        query_header: Text before the query.
        answer_header: Text after the query (prompts model to answer).

    Returns:
        AssembledContext with the full prompt and metadata.
    """
    block_ids = retrieved_block_ids or [f"block_{i}" for i in range(len(retrieved_texts))]

    if strategy == "prepend":
        context_section = block_separator.join(retrieved_texts)
        prompt = f"{context_header}{context_section}{query_header}{query}{answer_header}"

    elif strategy == "interleave":
        numbered_blocks = []
        for i, text in enumerate(retrieved_texts):
            numbered_blocks.append(f"[Document {i+1}]\n{text}")
        context_section = block_separator.join(numbered_blocks)
        prompt = (
            f"Below are relevant documents. Use them to answer the question.\n\n"
            f"{context_section}{query_header}{query}{answer_header}"
        )

    elif strategy == "summarize_prefix":
        context_section = block_separator.join(retrieved_texts)
        prompt = (
            f"You are given the following context. Read it carefully and "
            f"answer the question based only on the provided information.\n\n"
            f"{context_header}{context_section}{query_header}{query}{answer_header}"
        )
    else:
        raise ValueError(f"Unknown assembly strategy: {strategy}")

    # Truncate context if needed (preserve query)
    if max_context_chars is not None and len(prompt) > max_context_chars:
        query_part = f"{query_header}{query}{answer_header}"
        available = max_context_chars - len(query_part) - len(context_header) - 10
        if available > 0:
            context_section = context_section[:available] + "..."
            if strategy == "prepend":
                prompt = f"{context_header}{context_section}{query_header}{query}{answer_header}"
            else:
                prompt = prompt[:max_context_chars]
        else:
            # Not enough room for context, just use query
            prompt = f"{query_header}{query}{answer_header}"
            logger.warning("max_context_chars too small for any context, using query only")

    return AssembledContext(
        text=prompt,
        num_retrieved_blocks=len(retrieved_texts),
        retrieved_block_ids=block_ids,
        context_chars=len(prompt),
        query_text=query,
        strategy=strategy,
    )


def assemble_dense_context(
    query: str,
    full_context_blocks: list[str],
    query_header: str = "\nQuestion: ",
    answer_header: str = "\nAnswer: ",
    max_context_chars: int | None = None,
) -> AssembledContext:
    """Assemble a dense (non-retrieval) context — all blocks concatenated.

    Used as the dense baseline where the entire haystack is provided.

    Args:
        query: The question.
        full_context_blocks: All text blocks in order.
        query_header: Text before the query.
        answer_header: Text after the query.
        max_context_chars: Truncation limit.

    Returns:
        AssembledContext with the full dense prompt.
    """
    full_text = "\n\n".join(full_context_blocks)

    if max_context_chars is not None:
        query_part = f"{query_header}{query}{answer_header}"
        available = max_context_chars - len(query_part) - 10
        if available > 0 and len(full_text) > available:
            full_text = full_text[:available] + "..."

    prompt = f"{full_text}{query_header}{query}{answer_header}"

    return AssembledContext(
        text=prompt,
        num_retrieved_blocks=len(full_context_blocks),
        retrieved_block_ids=[],
        context_chars=len(prompt),
        query_text=query,
        strategy="dense",
    )
