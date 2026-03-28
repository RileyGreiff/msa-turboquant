"""Needle-in-a-Haystack (NIAH) synthetic dataset generator.

Generates long-context evaluation corpora where one or more "needles" (target
facts) are embedded in a large body of "haystack" distractor text. Supports:

- Single and multi-needle placement
- Configurable needle position (random, early, middle, late, specific depth)
- Configurable distractor count and block size
- Passkey retrieval variant
- Saving/loading datasets to disk as JSON

Each generated sample is a NIAHSample with full metadata for oracle evaluation.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from src.utils.io_utils import ensure_dir


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Needle:
    """A target fact to be hidden in the haystack.

    Attributes:
        needle_id: Unique identifier.
        text: The needle text content.
        answer: The expected answer (for evaluation).
        question: The question that should retrieve this needle.
        block_index: Which block position the needle was placed in (set during generation).
        depth: Relative position in the haystack (0.0 = start, 1.0 = end).
    """
    needle_id: str
    text: str
    answer: str
    question: str
    block_index: int = -1
    depth: float = 0.0


@dataclass
class NIAHSample:
    """A single needle-in-a-haystack evaluation sample.

    Attributes:
        sample_id: Unique identifier for this sample.
        blocks: Ordered list of text blocks forming the full context.
        needles: List of needles embedded in the context.
        needle_block_indices: Indices of blocks containing needles.
        total_chars: Total character count of the assembled context.
        num_blocks: Total number of blocks.
        metadata: Extra info (e.g., generation config).
    """
    sample_id: str
    blocks: list[str]
    needles: list[Needle]
    needle_block_indices: list[int]
    total_chars: int
    num_blocks: int
    metadata: dict = field(default_factory=dict)

    @property
    def full_context(self) -> str:
        """Concatenate all blocks into the full context string."""
        return "\n\n".join(self.blocks)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        d = {
            "sample_id": self.sample_id,
            "blocks": self.blocks,
            "needles": [asdict(n) for n in self.needles],
            "needle_block_indices": self.needle_block_indices,
            "total_chars": self.total_chars,
            "num_blocks": self.num_blocks,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NIAHSample":
        """Deserialize from a dict."""
        needles = [Needle(**n) for n in d["needles"]]
        return cls(
            sample_id=d["sample_id"],
            blocks=d["blocks"],
            needles=needles,
            needle_block_indices=d["needle_block_indices"],
            total_chars=d["total_chars"],
            num_blocks=d["num_blocks"],
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Text generators
# ---------------------------------------------------------------------------

# Collection of filler topics for generating diverse haystack text
_FILLER_TOPICS = [
    "The history of bridge construction spans thousands of years, from ancient "
    "stone arches to modern suspension bridges. Engineers have continuously "
    "improved materials and designs to create longer, stronger structures.",

    "Ocean currents play a vital role in regulating Earth's climate. The Gulf "
    "Stream, for instance, carries warm water from the tropics northward, "
    "moderating temperatures in Western Europe.",

    "The process of photosynthesis converts sunlight, water, and carbon dioxide "
    "into glucose and oxygen. This fundamental biological process sustains "
    "nearly all life on Earth.",

    "Medieval European castles served both as residences and military "
    "fortifications. Their design evolved from simple wooden structures to "
    "elaborate stone complexes with multiple layers of defense.",

    "The development of the printing press in the 15th century revolutionized "
    "information dissemination. Books became more accessible, literacy rates "
    "increased, and new ideas spread rapidly across Europe.",

    "Volcanic activity shapes landscapes through both destructive and creative "
    "forces. Lava flows can destroy existing terrain while simultaneously "
    "building new land masses and enriching soil with minerals.",

    "The study of ancient languages has revealed connections between seemingly "
    "unrelated cultures. Linguistic analysis helps historians trace migration "
    "patterns and cultural exchanges across millennia.",

    "Railroad networks transformed commerce and society in the 19th century. "
    "They enabled rapid transport of goods, created new towns, and connected "
    "remote regions to urban centers.",

    "Coral reefs support an extraordinary diversity of marine life despite "
    "covering less than one percent of the ocean floor. These ecosystems are "
    "sensitive indicators of ocean health.",

    "The invention of the telescope in the early 17th century opened new "
    "frontiers in astronomy. Galileo's observations of Jupiter's moons "
    "provided evidence supporting the heliocentric model of the solar system.",

    "Traditional fermentation techniques have been used for thousands of years "
    "to preserve food and create beverages. The biochemical processes involved "
    "were not understood until the work of Louis Pasteur.",

    "Desert ecosystems demonstrate remarkable adaptations to extreme "
    "conditions. Plants like cacti have evolved specialized water storage "
    "mechanisms, while animals often adopt nocturnal behaviors.",
]


def _generate_filler_block(
    target_chars: int,
    rng: random.Random,
) -> str:
    """Generate a block of plausible filler text of approximately target_chars length.

    Selects and repeats from a pool of filler paragraphs, then truncates or
    pads to reach the target length.
    """
    parts: list[str] = []
    current_len = 0
    while current_len < target_chars:
        topic = rng.choice(_FILLER_TOPICS)
        parts.append(topic)
        current_len += len(topic) + 1  # +1 for space

    text = " ".join(parts)
    # Truncate to target, but try to end at a sentence boundary
    if len(text) > target_chars:
        truncated = text[:target_chars]
        last_period = truncated.rfind(".")
        if last_period > target_chars * 0.7:
            text = truncated[: last_period + 1]
        else:
            text = truncated
    return text


def generate_passkey(length: int = 5, rng: random.Random | None = None) -> str:
    """Generate a random numeric passkey string."""
    rng = rng or random.Random()
    return "".join(rng.choices(string.digits, k=length))


# ---------------------------------------------------------------------------
# Default needle templates
# ---------------------------------------------------------------------------

def make_fact_needle(
    fact_id: int = 0,
    rng: random.Random | None = None,
) -> Needle:
    """Create a simple factual needle with a question-answer pair."""
    rng = rng or random.Random()
    city = rng.choice([
        "Paris", "Tokyo", "Berlin", "Mumbai", "Sydney",
        "Cairo", "Toronto", "Santiago", "Oslo", "Bangkok",
    ])
    color = rng.choice([
        "red", "blue", "green", "purple", "orange",
        "silver", "golden", "crimson", "turquoise", "amber",
    ])
    secret_number = rng.randint(1000, 9999)

    text = (
        f"IMPORTANT FACT: The secret code for the {color} door in {city} "
        f"is {secret_number}. Remember this number carefully."
    )
    question = f"What is the secret code for the {color} door in {city}?"
    answer = str(secret_number)

    return Needle(
        needle_id=f"fact_{fact_id}",
        text=text,
        answer=answer,
        question=question,
    )


def make_passkey_needle(
    passkey_id: int = 0,
    rng: random.Random | None = None,
) -> Needle:
    """Create a passkey retrieval needle."""
    rng = rng or random.Random()
    passkey = generate_passkey(5, rng)
    text = f"The passkey is {passkey}. Please remember it."
    return Needle(
        needle_id=f"passkey_{passkey_id}",
        text=text,
        answer=passkey,
        question="What is the passkey?",
    )


# ---------------------------------------------------------------------------
# Placement logic
# ---------------------------------------------------------------------------

def _compute_needle_positions(
    num_blocks: int,
    num_needles: int,
    position: Literal["random", "early", "middle", "late"] | float,
    rng: random.Random,
) -> list[int]:
    """Compute block indices for needle placement.

    Args:
        num_blocks: Total number of blocks in the haystack.
        num_needles: Number of needles to place.
        position: Placement strategy or a specific depth float (0.0-1.0).
        rng: Random number generator.

    Returns:
        Sorted list of block indices where needles will be placed.
    """
    if num_needles > num_blocks:
        raise ValueError(
            f"Cannot place {num_needles} needles in {num_blocks} blocks"
        )

    if isinstance(position, (int, float)) and not isinstance(position, bool):
        # Specific depth: cluster needles around this relative position
        center = int(position * (num_blocks - 1))
        indices = set()
        for i in range(num_needles):
            idx = min(max(center + i - num_needles // 2, 0), num_blocks - 1)
            indices.add(idx)
        # If we got duplicates from clamping, fill from nearby positions
        offset = 1
        while len(indices) < num_needles:
            for candidate in [center + offset, center - offset]:
                if 0 <= candidate < num_blocks and candidate not in indices:
                    indices.add(candidate)
                    if len(indices) >= num_needles:
                        break
            offset += 1
        return sorted(indices)

    if position == "random":
        return sorted(rng.sample(range(num_blocks), num_needles))
    elif position == "early":
        # First 25% of blocks
        end = max(num_needles, num_blocks // 4)
        return sorted(rng.sample(range(end), num_needles))
    elif position == "middle":
        # Middle 50% of blocks
        start = num_blocks // 4
        end = start + num_blocks // 2
        end = max(end, start + num_needles)
        return sorted(rng.sample(range(start, min(end, num_blocks)), num_needles))
    elif position == "late":
        # Last 25% of blocks
        start = max(0, num_blocks - max(num_needles, num_blocks // 4))
        return sorted(rng.sample(range(start, num_blocks), num_needles))
    else:
        raise ValueError(f"Unknown position strategy: {position}")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_niah_sample(
    num_blocks: int = 20,
    block_chars: int = 500,
    num_needles: int = 1,
    needle_type: Literal["fact", "passkey"] = "fact",
    position: Literal["random", "early", "middle", "late"] | float = "random",
    seed: int | None = None,
    sample_id: str = "niah_0",
) -> NIAHSample:
    """Generate a single needle-in-a-haystack sample.

    Args:
        num_blocks: Total number of text blocks (including needle blocks).
        block_chars: Approximate character count per filler block.
        num_needles: Number of needles to embed.
        needle_type: Type of needle to generate ("fact" or "passkey").
        position: Where to place needles in the haystack.
        seed: Random seed for reproducibility.
        sample_id: Unique identifier for this sample.

    Returns:
        A NIAHSample with blocks, needles, and full metadata.
    """
    rng = random.Random(seed)

    # Generate needles
    needles: list[Needle] = []
    for i in range(num_needles):
        if needle_type == "fact":
            needle = make_fact_needle(fact_id=i, rng=rng)
        elif needle_type == "passkey":
            needle = make_passkey_needle(passkey_id=i, rng=rng)
        else:
            raise ValueError(f"Unknown needle_type: {needle_type}")
        needles.append(needle)

    # Compute placement
    needle_indices = _compute_needle_positions(num_blocks, num_needles, position, rng)

    # Build blocks
    blocks: list[str] = []
    needle_iter = iter(needles)
    placed_needles: list[Needle] = []

    for block_idx in range(num_blocks):
        if block_idx in needle_indices:
            needle = next(needle_iter)
            # Embed needle in a filler block so it's not trivially findable by length
            prefix = _generate_filler_block(block_chars // 3, rng)
            suffix = _generate_filler_block(block_chars // 3, rng)
            block_text = f"{prefix} {needle.text} {suffix}"
            needle.block_index = block_idx
            needle.depth = block_idx / max(num_blocks - 1, 1)
            placed_needles.append(needle)
        else:
            block_text = _generate_filler_block(block_chars, rng)
        blocks.append(block_text)

    total_chars = sum(len(b) for b in blocks)

    return NIAHSample(
        sample_id=sample_id,
        blocks=blocks,
        needles=placed_needles,
        needle_block_indices=needle_indices,
        total_chars=total_chars,
        num_blocks=num_blocks,
        metadata={
            "num_needles": num_needles,
            "needle_type": needle_type,
            "position": position if isinstance(position, str) else float(position),
            "block_chars": block_chars,
            "seed": seed,
        },
    )


def generate_niah_dataset(
    num_samples: int = 10,
    num_blocks_list: list[int] | None = None,
    block_chars: int = 500,
    num_needles: int = 1,
    needle_type: Literal["fact", "passkey"] = "fact",
    position: Literal["random", "early", "middle", "late"] | float = "random",
    seed: int = 42,
) -> list[NIAHSample]:
    """Generate a dataset of NIAH samples, optionally sweeping over corpus sizes.

    If num_blocks_list is provided, generates one sample per entry (ignoring
    num_samples). Otherwise generates num_samples all with the same num_blocks.

    Args:
        num_samples: Number of samples to generate (if num_blocks_list is None).
        num_blocks_list: Optional list of block counts to sweep over.
        block_chars: Approximate characters per block.
        num_needles: Needles per sample.
        needle_type: "fact" or "passkey".
        position: Needle placement strategy.
        seed: Base random seed (incremented per sample).

    Returns:
        List of NIAHSample objects.
    """
    samples: list[NIAHSample] = []

    if num_blocks_list is not None:
        for i, nb in enumerate(num_blocks_list):
            sample = generate_niah_sample(
                num_blocks=nb,
                block_chars=block_chars,
                num_needles=num_needles,
                needle_type=needle_type,
                position=position,
                seed=seed + i,
                sample_id=f"niah_{i}_nb{nb}",
            )
            samples.append(sample)
    else:
        default_blocks = 20
        for i in range(num_samples):
            sample = generate_niah_sample(
                num_blocks=default_blocks,
                block_chars=block_chars,
                num_needles=num_needles,
                needle_type=needle_type,
                position=position,
                seed=seed + i,
                sample_id=f"niah_{i}",
            )
            samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_niah_dataset(
    samples: list[NIAHSample],
    path: Path | str,
) -> Path:
    """Save a NIAH dataset to a JSON file.

    Args:
        samples: List of NIAHSample objects.
        path: Output file path.

    Returns:
        The Path the dataset was saved to.
    """
    path = Path(path)
    ensure_dir(path.parent)
    data = {
        "num_samples": len(samples),
        "samples": [s.to_dict() for s in samples],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_niah_dataset(path: Path | str) -> list[NIAHSample]:
    """Load a NIAH dataset from a JSON file.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        List of NIAHSample objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIAH dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [NIAHSample.from_dict(s) for s in data["samples"]]
