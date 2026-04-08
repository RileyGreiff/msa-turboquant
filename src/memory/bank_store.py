"""Memory bank persistence using numpy memmap and JSON metadata.

Storage layout on disk:
    bank_dir/
    ├── metadata.json          # Bank-level metadata (MemoryBankMetadata)
    ├── block_index.json       # Per-block metadata list (BlockMetadataEntry[])
    ├── routing_vectors.npy    # Stacked routing vectors (num_blocks, hidden_dim)
    └── kv/
        ├── layer_0_keys.npy   # Keys for layer 0 (num_blocks, num_heads, max_seq, head_dim)
        ├── layer_0_values.npy # Values for layer 0
        ├── layer_1_keys.npy
        ├── layer_1_values.npy
        └── ...

KV tensors are stored as numpy memmap files so they can be accessed without
loading the entire bank into RAM. Routing vectors are stored as a regular
numpy array (small enough to fit in memory).

Token count varies per block, so KV arrays are padded to max_seq_len within
the bank. A token_counts array tracks the real length per block.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.memory.bank_builder import (
    BlockMetadataEntry,
    MemoryBank,
    MemoryBankMetadata,
)
from src.models.kv_extractor import KVBlock
from src.utils.io_utils import ensure_dir

logger = logging.getLogger("msa_turboquant.memory.bank_store")


def save_bank(bank: MemoryBank, bank_dir: Path | str, save_kv: bool = True) -> Path:
    """Save a MemoryBank to disk.

    Args:
        bank: The MemoryBank to save.
        bank_dir: Directory to save into (will be created).
        save_kv: Whether to save KV tensors. False saves only routing
            vectors and metadata (much smaller on disk). KV injection
            modes don't need stored KV since they re-encode from text.

    Returns:
        Path to the bank directory.
    """
    bank_dir = Path(bank_dir)
    ensure_dir(bank_dir)
    if save_kv:
        kv_dir = ensure_dir(bank_dir / "kv")

    num_blocks = bank.num_blocks
    meta = bank.metadata

    logger.info(f"Saving bank '{meta.bank_id}' to {bank_dir} ({num_blocks} blocks)")

    # 1. Save bank metadata
    with open(bank_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2, default=str)

    # 2. Save block index
    block_index = [bm.to_dict() for bm in bank.block_metadata]
    with open(bank_dir / "block_index.json", "w", encoding="utf-8") as f:
        json.dump(block_index, f, indent=2, default=str)

    # 3. Save routing vectors as regular numpy array
    routing_np = bank.routing_vectors.numpy().astype(np.float32)
    np.save(bank_dir / "routing_vectors.npy", routing_np)

    # 4. Save token counts
    token_counts = np.array([kb.num_tokens for kb in bank.kv_blocks], dtype=np.int32)
    np.save(bank_dir / "token_counts.npy", token_counts)

    # 5. Save shape info early (before KV, so routing-only banks are always loadable)
    shape_info = {
        "num_blocks": num_blocks,
        "num_layers": meta.num_layers,
        "num_heads": meta.num_heads,
        "head_dim": meta.head_dim,
        "max_tokens": max(kb.num_tokens for kb in bank.kv_blocks) if num_blocks > 0 else 0,
        "hidden_dim": meta.hidden_dim,
        "kv_dtype": "float16",
        "routing_dtype": "float32",
        "has_kv": save_kv,
    }
    with open(bank_dir / "shape_info.json", "w", encoding="utf-8") as f:
        json.dump(shape_info, f, indent=2)

    # 6. Save KV tensors per layer as memmap files
    if save_kv and num_blocks > 0 and bank.kv_blocks[0].num_layers > 0:
        max_tokens = max(kb.num_tokens for kb in bank.kv_blocks)
        num_layers = meta.num_layers
        num_heads = meta.num_heads
        head_dim = meta.head_dim

        for layer_idx in range(num_layers):
            # Allocate padded arrays
            keys_array = np.zeros(
                (num_blocks, num_heads, max_tokens, head_dim), dtype=np.float16
            )
            values_array = np.zeros(
                (num_blocks, num_heads, max_tokens, head_dim), dtype=np.float16
            )

            for block_idx, kb in enumerate(bank.kv_blocks):
                if layer_idx < kb.num_layers:
                    k = kb.keys[layer_idx].to(torch.float16).numpy()
                    v = kb.values[layer_idx].to(torch.float16).numpy()
                    seq_len = k.shape[1]
                    keys_array[block_idx, :, :seq_len, :] = k
                    values_array[block_idx, :, :seq_len, :] = v

            # Save as memmap-friendly npy files
            k_path = kv_dir / f"layer_{layer_idx}_keys.npy"
            v_path = kv_dir / f"layer_{layer_idx}_values.npy"
            np.save(k_path, keys_array)
            np.save(v_path, values_array)

        logger.info(
            f"Saved {num_layers} KV layers, "
            f"max_tokens={max_tokens}, shape per layer: "
            f"({num_blocks}, {num_heads}, {max_tokens}, {head_dim})"
        )

    total_bytes = sum(f.stat().st_size for f in bank_dir.rglob("*") if f.is_file())
    logger.info(f"Bank saved: {total_bytes / (1024**2):.1f} MB on disk")

    return bank_dir


def load_bank(
    bank_dir: Path | str,
    load_kv: bool = True,
    mmap_mode: str | None = "r",
) -> MemoryBank:
    """Load a MemoryBank from disk.

    Args:
        bank_dir: Directory containing the saved bank.
        load_kv: Whether to load KV tensors. Set False to load only
            routing vectors and metadata (much faster for routing-only use).
        mmap_mode: numpy memmap mode for KV arrays. "r" for read-only mmap,
            None to load fully into RAM. Use "r" for large banks.

    Returns:
        A MemoryBank instance.
    """
    bank_dir = Path(bank_dir)
    if not bank_dir.is_dir():
        raise FileNotFoundError(f"Bank directory not found: {bank_dir}")

    logger.info(f"Loading bank from {bank_dir} (load_kv={load_kv}, mmap_mode={mmap_mode})")

    # 1. Load metadata
    with open(bank_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = MemoryBankMetadata.from_dict(json.load(f))

    # 2. Load block index
    with open(bank_dir / "block_index.json", "r", encoding="utf-8") as f:
        block_index = [BlockMetadataEntry.from_dict(d) for d in json.load(f)]

    # 3. Load routing vectors
    routing_np = np.load(bank_dir / "routing_vectors.npy")
    routing_vectors = torch.from_numpy(routing_np.copy())

    # 4. Load token counts
    token_counts = np.load(bank_dir / "token_counts.npy")

    # 5. Load shape info
    with open(bank_dir / "shape_info.json", "r", encoding="utf-8") as f:
        shape_info = json.load(f)

    # 6. Build KV blocks
    kv_blocks: list[KVBlock] = []
    num_blocks = meta.num_blocks

    if load_kv and meta.num_layers > 0:
        kv_dir = bank_dir / "kv"

        # Load all layer arrays (memmap or full)
        layer_keys: list[np.ndarray] = []
        layer_values: list[np.ndarray] = []
        for layer_idx in range(meta.num_layers):
            k_path = kv_dir / f"layer_{layer_idx}_keys.npy"
            v_path = kv_dir / f"layer_{layer_idx}_values.npy"
            if mmap_mode is not None:
                layer_keys.append(np.load(k_path, mmap_mode=mmap_mode))
                layer_values.append(np.load(v_path, mmap_mode=mmap_mode))
            else:
                layer_keys.append(np.load(k_path))
                layer_values.append(np.load(v_path))

        # Assemble per-block KVBlocks
        for block_idx in range(num_blocks):
            ntok = int(token_counts[block_idx])
            keys = []
            values = []
            for layer_idx in range(meta.num_layers):
                # Slice to actual token count (remove padding)
                k = torch.from_numpy(
                    layer_keys[layer_idx][block_idx, :, :ntok, :].copy()
                ).to(torch.float16)
                v = torch.from_numpy(
                    layer_values[layer_idx][block_idx, :, :ntok, :].copy()
                ).to(torch.float16)
                keys.append(k)
                values.append(v)

            kv_blocks.append(KVBlock(
                block_id=block_index[block_idx].block_id,
                keys=keys,
                values=values,
                routing_vector=routing_vectors[block_idx],
                num_tokens=ntok,
                layer_indices=meta.layer_indices,
            ))
    else:
        # Create KV blocks with empty K/V but valid routing vectors
        for block_idx in range(num_blocks):
            kv_blocks.append(KVBlock(
                block_id=block_index[block_idx].block_id,
                keys=[],
                values=[],
                routing_vector=routing_vectors[block_idx],
                num_tokens=int(token_counts[block_idx]),
                layer_indices=[],
            ))

    logger.info(
        f"Bank loaded: {num_blocks} blocks, "
        f"{'with' if load_kv else 'without'} KV tensors"
    )

    return MemoryBank(
        metadata=meta,
        block_metadata=block_index,
        routing_vectors=routing_vectors,
        kv_blocks=kv_blocks,
    )


def load_routing_vectors(bank_dir: Path | str) -> tuple[torch.Tensor, list[str]]:
    """Load only routing vectors and block IDs (fast, for routing only).

    Returns:
        Tuple of (routing_vectors tensor, list of block_ids).
    """
    bank_dir = Path(bank_dir)
    routing_np = np.load(bank_dir / "routing_vectors.npy")
    routing_vectors = torch.from_numpy(routing_np.copy())

    with open(bank_dir / "block_index.json", "r", encoding="utf-8") as f:
        block_index = json.load(f)
    block_ids = [b["block_id"] for b in block_index]

    return routing_vectors, block_ids


def load_kv_for_blocks(
    bank_dir: Path | str,
    block_indices: list[int],
    layers: list[int] | None = None,
) -> list[KVBlock]:
    """Load KV tensors for specific blocks only (selective fetch).

    This is the key function for sparse retrieval — after routing selects
    the top-k blocks, only those blocks' KV tensors are loaded from disk.

    Args:
        bank_dir: Directory containing the saved bank.
        block_indices: Which block indices to load (0-based).
        layers: Which layer indices to load. None = all.

    Returns:
        List of KVBlock objects for the requested blocks.
    """
    bank_dir = Path(bank_dir)

    # Load shape info and metadata
    with open(bank_dir / "shape_info.json", "r", encoding="utf-8") as f:
        shape_info = json.load(f)
    with open(bank_dir / "block_index.json", "r", encoding="utf-8") as f:
        block_index = json.load(f)

    token_counts = np.load(bank_dir / "token_counts.npy")
    routing_np = np.load(bank_dir / "routing_vectors.npy")

    num_layers = shape_info["num_layers"]
    layer_indices = layers if layers is not None else list(range(num_layers))
    kv_dir = bank_dir / "kv"

    # Open memmap handles for requested layers
    layer_k_maps: dict[int, np.ndarray] = {}
    layer_v_maps: dict[int, np.ndarray] = {}
    for li in layer_indices:
        layer_k_maps[li] = np.load(kv_dir / f"layer_{li}_keys.npy", mmap_mode="r")
        layer_v_maps[li] = np.load(kv_dir / f"layer_{li}_values.npy", mmap_mode="r")

    # Build KVBlocks for requested indices only
    results: list[KVBlock] = []
    for bi in block_indices:
        ntok = int(token_counts[bi])
        keys = []
        values = []
        for li in layer_indices:
            k = torch.from_numpy(layer_k_maps[li][bi, :, :ntok, :].copy()).to(torch.float16)
            v = torch.from_numpy(layer_v_maps[li][bi, :, :ntok, :].copy()).to(torch.float16)
            keys.append(k)
            values.append(v)

        routing_vec = torch.from_numpy(routing_np[bi].copy())
        block_id = block_index[bi]["block_id"]

        results.append(KVBlock(
            block_id=block_id,
            keys=keys,
            values=values,
            routing_vector=routing_vec,
            num_tokens=ntok,
            layer_indices=layer_indices,
        ))

    logger.info(
        f"Loaded KV for {len(block_indices)} blocks "
        f"({len(layer_indices)} layers each) from {bank_dir}"
    )

    return results
