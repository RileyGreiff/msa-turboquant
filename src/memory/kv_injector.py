"""KV cache injection utilities.

Converts retrieved KVBlock objects into the arguments needed by
model.generate() for direct KV injection into attention. This is the
key module that makes compression quality affect answer accuracy.

Two approaches:
1. assemble_kv_injection() — concatenates pre-extracted KVBlocks (has RoPE
   discontinuity between blocks since each was extracted independently).
2. encode_context_to_kv() — runs a single contiguous forward pass on
   retrieved text so positions are correct, then returns KV for injection.
   This is the recommended approach for measuring compression quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from src.models.kv_extractor import KVBlock

if TYPE_CHECKING:
    from src.models.base_model import BaseModel

logger = logging.getLogger("msa_turboquant.memory.kv_injector")


@dataclass
class KVInjectionPayload:
    """Everything needed to call model.generate() with pre-filled KV cache.

    Attributes:
        past_key_values: DynamicCache or tuple-of-tuples for HF generate().
            None if no KV blocks were provided.
        attention_mask: Combined mask covering KV tokens + query tokens.
            Shape: (1, kv_seq_len + query_seq_len).
        position_ids: Position IDs for query tokens, offset by kv_seq_len.
            Shape: (1, query_seq_len). Critical for RoPE models.
        kv_seq_len: Total number of tokens in the injected KV cache.
    """
    past_key_values: Any
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_len: int = 0


def assemble_kv_injection(
    kv_blocks: list[KVBlock],
    query_token_ids: torch.Tensor,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVInjectionPayload:
    """Convert retrieved KVBlocks into generate() arguments for KV injection.

    Concatenates multiple KV blocks along the sequence dimension, constructs
    a DynamicCache (or tuple fallback), and builds the attention mask and
    position IDs needed for generation with pre-filled context.

    Args:
        kv_blocks: Retrieved KVBlock objects. May be empty.
        query_token_ids: Tokenized query. Shape: (1, query_seq_len).
        num_layers: Number of model layers (must match KV blocks).
        device: Target device for all tensors.
        dtype: Target dtype for KV tensors.

    Returns:
        KVInjectionPayload ready for model.generate().
    """
    query_seq_len = query_token_ids.shape[1]

    # Handle empty blocks — degrade to plain generation
    if not kv_blocks:
        logger.warning("No KV blocks for injection — generating without context")
        return KVInjectionPayload(
            past_key_values=None,
            attention_mask=torch.ones(1, query_seq_len, dtype=torch.long, device=device),
            position_ids=torch.arange(query_seq_len, dtype=torch.long, device=device).unsqueeze(0),
            kv_seq_len=0,
        )

    # Validate layer coverage
    block_num_layers = kv_blocks[0].num_layers
    if block_num_layers < num_layers:
        logger.warning(
            f"KV blocks have {block_num_layers} layers but model has {num_layers}. "
            f"Using {block_num_layers} layers for injection."
        )
        num_layers = block_num_layers

    # Concatenate KV blocks along sequence dimension for each layer
    # KVBlock stores: (num_kv_heads, seq_len, head_dim) — no batch dim
    # DynamicCache needs: (batch=1, num_kv_heads, seq_len, head_dim)
    kv_seq_len = sum(b.num_tokens for b in kv_blocks)

    all_keys: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []

    for layer_idx in range(num_layers):
        layer_keys = []
        layer_values = []
        for block in kv_blocks:
            # Add batch dim and move to target device/dtype
            k = block.keys[layer_idx].unsqueeze(0).to(device=device, dtype=dtype)
            v = block.values[layer_idx].unsqueeze(0).to(device=device, dtype=dtype)
            layer_keys.append(k)
            layer_values.append(v)

        # Concatenate along seq_len dimension (dim=2)
        # Shape: (1, num_kv_heads, total_seq_len, head_dim)
        all_keys.append(torch.cat(layer_keys, dim=2))
        all_values.append(torch.cat(layer_values, dim=2))

    # Build past_key_values — try DynamicCache first, fall back to tuples
    past_key_values = _build_past_key_values(all_keys, all_values)

    # Attention mask: ones for all KV tokens + all query tokens
    total_len = kv_seq_len + query_seq_len
    attention_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

    # Position IDs: query tokens start after the KV cache positions
    # This is critical for RoPE — without correct offsets, the model
    # can't relate query tokens to the cached context
    position_ids = torch.arange(
        kv_seq_len, kv_seq_len + query_seq_len,
        dtype=torch.long, device=device,
    ).unsqueeze(0)

    logger.info(
        f"KV injection assembled: {len(kv_blocks)} blocks, "
        f"{kv_seq_len} KV tokens + {query_seq_len} query tokens, "
        f"{num_layers} layers"
    )

    return KVInjectionPayload(
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_seq_len=kv_seq_len,
    )


def _build_past_key_values(
    all_keys: list[torch.Tensor],
    all_values: list[torch.Tensor],
) -> Any:
    """Construct past_key_values from assembled K/V tensors.

    Tries DynamicCache (transformers >= 4.36) first, falls back to
    tuple-of-tuples which is universally supported.

    Args:
        all_keys: Per-layer key tensors, each (1, num_kv_heads, seq_len, head_dim).
        all_values: Per-layer value tensors, same shape.

    Returns:
        DynamicCache or tuple of (key, value) pairs per layer.
    """
    try:
        from transformers import DynamicCache
        cache = DynamicCache()
        for layer_idx in range(len(all_keys)):
            cache.update(all_keys[layer_idx], all_values[layer_idx], layer_idx)
        return cache
    except ImportError:
        logger.info("DynamicCache not available, using tuple-of-tuples fallback")
        return tuple(
            (all_keys[i], all_values[i])
            for i in range(len(all_keys))
        )


def _chunked_forward(
    context_ids: torch.Tensor,
    model: BaseModel,
    chunk_size: int,
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], int]:
    """Run forward pass in chunks, accumulating KV cache incrementally.

    Processes context_ids in segments of chunk_size tokens, passing accumulated
    KV cache between chunks so positions stay contiguous (correct for RoPE).

    Args:
        context_ids: Full context token IDs. Shape: (1, total_seq_len).
        model: Loaded model.
        chunk_size: Maximum tokens per forward pass.

    Returns:
        Tuple of (kv_cache as tuple-of-tuples, total_seq_len).
    """
    total_seq_len = context_ids.shape[1]
    num_chunks = (total_seq_len + chunk_size - 1) // chunk_size
    raw_cache = None
    offset = 0

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_seq_len)
        chunk_ids = context_ids[:, start:end]

        # Attention mask covers all previous KV tokens + current chunk
        total_seen = offset + (end - start)
        attn_mask = torch.ones(1, total_seen, dtype=torch.long, device=model.device)

        output = model.forward(
            chunk_ids,
            attention_mask=attn_mask,
            use_cache=True,
            past_key_values=raw_cache,
        )

        # Use raw_kv_cache (DynamicCache) for accumulation if available,
        # otherwise fall back to kv_cache tuples
        raw_cache = output.raw_kv_cache if output.raw_kv_cache is not None else output.kv_cache
        offset = total_seen

    # Extract final KV as tuple-of-tuples
    if output.kv_cache is None:
        raise RuntimeError("Model forward pass did not return kv_cache")
    return output.kv_cache, total_seq_len


def encode_context_to_kv(
    context_text: str,
    query_text: str,
    model: BaseModel,
    compressor: Any | None = None,
    chunk_size: int | None = None,
) -> tuple[KVInjectionPayload, list[KVBlock], float]:
    """Encode context text, optionally compress, and build a KVInjectionPayload
    with correct contiguous positions.

    This is the recommended approach for KV injection evaluation because:
    - RoPE positions are contiguous (no discontinuity between blocks)
    - Compression roundtrip directly degrades the KV that feeds attention

    Args:
        context_text: Full context string (retrieved blocks concatenated).
        query_text: The question to answer.
        model: Loaded model with forward() and tokenize().
        compressor: Optional compressor. If provided, KV is compressed then
            decompressed in-place before injection (roundtrip degradation).
        chunk_size: If set, split context into chunks of this many tokens
            and process incrementally. Reduces peak activation memory.

    Returns:
        Tuple of (payload, kv_blocks, compression_ratio).
        kv_blocks contains a single KVBlock with the full context KV.
    """
    # Tokenize context (without query) to get KV via forward pass
    context_tokens = model.tokenize(context_text)
    context_ids = context_tokens.input_ids.to(model.device)
    context_seq_len = context_ids.shape[1]

    # Use chunked or single-pass forward
    if chunk_size is not None and context_seq_len > chunk_size:
        logger.info(
            f"Using chunked prefill: {context_seq_len} tokens in "
            f"{(context_seq_len + chunk_size - 1) // chunk_size} chunks of {chunk_size}"
        )
        past_kv, _ = _chunked_forward(context_ids, model, chunk_size)
    else:
        context_mask = context_tokens.attention_mask.to(model.device) if context_tokens.attention_mask is not None else None
        output = model.forward(
            context_ids,
            attention_mask=context_mask,
            use_cache=True,
        )
        past_kv = output.kv_cache

    # Extract KV from the cache
    if past_kv is None:
        raise RuntimeError("Model forward pass did not return kv_cache")

    # Convert to KVBlock format for compression
    keys_per_layer = []
    values_per_layer = []
    for layer_kv in past_kv:
        if isinstance(layer_kv, (tuple, list)):
            k, v = layer_kv[0], layer_kv[1]
        else:
            k, v = layer_kv
        # k, v shape: (batch=1, num_kv_heads, seq_len, head_dim)
        # KVBlock expects: (num_kv_heads, seq_len, head_dim)
        keys_per_layer.append(k.squeeze(0))
        values_per_layer.append(v.squeeze(0))

    kv_block = KVBlock(
        block_id="context_kv",
        keys=keys_per_layer,
        values=values_per_layer,
        routing_vector=torch.zeros(1),  # not needed for injection
        num_tokens=context_seq_len,
        layer_indices=list(range(len(keys_per_layer))),
    )

    # Compress/decompress roundtrip if compressor provided
    compression_ratio = 0.0
    if compressor is not None:
        original_bytes = kv_block.total_kv_bytes
        for i in range(kv_block.num_layers):
            compressed_k = compressor.compress(kv_block.keys[i])
            compressed_v = compressor.compress(kv_block.values[i])
            kv_block.keys[i] = compressor.decompress(compressed_k)
            kv_block.values[i] = compressor.decompress(compressed_v)
        decompressed_bytes = kv_block.total_kv_bytes
        if decompressed_bytes > 0:
            compression_ratio = original_bytes / decompressed_bytes

    # Build injection payload from the (possibly degraded) KV
    query_tokens = model.tokenize(query_text)
    query_ids = query_tokens.input_ids.to(model.device)

    # Rebuild past_key_values from the KVBlock (adding batch dim back)
    all_keys = [kv_block.keys[i].unsqueeze(0).to(device=model.device, dtype=model.dtype)
                for i in range(kv_block.num_layers)]
    all_values = [kv_block.values[i].unsqueeze(0).to(device=model.device, dtype=model.dtype)
                  for i in range(kv_block.num_layers)]
    past_key_values = _build_past_key_values(all_keys, all_values)

    # Attention mask and position IDs
    query_seq_len = query_ids.shape[1]
    total_len = context_seq_len + query_seq_len
    attention_mask = torch.ones(1, total_len, dtype=torch.long, device=model.device)
    position_ids = torch.arange(
        context_seq_len, context_seq_len + query_seq_len,
        dtype=torch.long, device=model.device,
    ).unsqueeze(0)

    payload = KVInjectionPayload(
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_seq_len=context_seq_len,
    )

    logger.info(
        f"Context KV encoded: {context_seq_len} context tokens + "
        f"{query_seq_len} query tokens, {kv_block.num_layers} layers"
        + (f", compression_ratio={compression_ratio:.2f}" if compressor else "")
    )

    return payload, [kv_block], compression_ratio
