"""Tests for KV cache injection utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from src.memory.kv_injector import (
    KVInjectionPayload,
    assemble_kv_injection,
    encode_context_to_kv,
    _chunked_forward,
)
from src.models.kv_extractor import KVBlock

# Import DummyModel from test_models
sys.path.insert(0, str(Path(__file__).parent))
from test_models import DummyModel


def _make_kv_block(
    block_id: str = "blk_0",
    num_layers: int = 4,
    num_kv_heads: int = 2,
    seq_len: int = 16,
    head_dim: int = 32,
) -> KVBlock:
    """Create a KVBlock with random tensors for testing."""
    keys = [torch.randn(num_kv_heads, seq_len, head_dim) for _ in range(num_layers)]
    values = [torch.randn(num_kv_heads, seq_len, head_dim) for _ in range(num_layers)]
    routing = torch.randn(num_kv_heads * head_dim)
    return KVBlock(
        block_id=block_id,
        keys=keys,
        values=values,
        routing_vector=routing,
        num_tokens=seq_len,
        layer_indices=list(range(num_layers)),
    )


class TestAssembleKVInjection:
    """Tests for assemble_kv_injection()."""

    def test_single_block(self) -> None:
        block = _make_kv_block(num_layers=4, num_kv_heads=2, seq_len=16, head_dim=32)
        query = torch.randint(0, 100, (1, 8))

        payload = assemble_kv_injection(
            kv_blocks=[block],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert payload.kv_seq_len == 16
        assert payload.attention_mask.shape == (1, 16 + 8)
        assert payload.position_ids.shape == (1, 8)
        assert payload.position_ids[0, 0].item() == 16  # starts after KV
        assert payload.past_key_values is not None

    def test_multi_block_concatenation(self) -> None:
        blocks = [
            _make_kv_block(block_id=f"blk_{i}", seq_len=10 + i * 5)
            for i in range(3)
        ]
        total_seq = 10 + 15 + 20  # 45
        query = torch.randint(0, 100, (1, 6))

        payload = assemble_kv_injection(
            kv_blocks=blocks,
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert payload.kv_seq_len == total_seq
        assert payload.attention_mask.shape == (1, total_seq + 6)
        assert payload.position_ids[0, 0].item() == total_seq

    def test_attention_mask_all_ones(self) -> None:
        block = _make_kv_block(seq_len=20)
        query = torch.randint(0, 100, (1, 5))

        payload = assemble_kv_injection(
            kv_blocks=[block],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert payload.attention_mask.sum().item() == 25  # all ones

    def test_position_ids_contiguous(self) -> None:
        block = _make_kv_block(seq_len=50)
        query = torch.randint(0, 100, (1, 10))

        payload = assemble_kv_injection(
            kv_blocks=[block],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        expected = torch.arange(50, 60)
        assert torch.equal(payload.position_ids.squeeze(0), expected)

    def test_empty_blocks_returns_none(self) -> None:
        query = torch.randint(0, 100, (1, 8))

        payload = assemble_kv_injection(
            kv_blocks=[],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert payload.past_key_values is None
        assert payload.kv_seq_len == 0
        assert payload.attention_mask.shape == (1, 8)
        assert payload.position_ids.shape == (1, 8)
        assert payload.position_ids[0, 0].item() == 0

    def test_dtype_casting(self) -> None:
        block = _make_kv_block(seq_len=8)
        query = torch.randint(0, 100, (1, 4))

        payload = assemble_kv_injection(
            kv_blocks=[block],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )

        # Verify KV tensors are in the requested dtype
        past_kv = payload.past_key_values
        # Access via iteration (works for both DynamicCache and tuples)
        for layer_kv in past_kv:
            if isinstance(layer_kv, (tuple, list)):
                k, v = layer_kv[0], layer_kv[1]
            else:
                k, v = layer_kv
            assert k.dtype == torch.float16
            assert v.dtype == torch.float16

    def test_fewer_layers_than_model(self) -> None:
        """Blocks with fewer layers than model should use block layer count."""
        block = _make_kv_block(num_layers=2, seq_len=8)
        query = torch.randint(0, 100, (1, 4))

        # Model has 4 layers but blocks only have 2
        payload = assemble_kv_injection(
            kv_blocks=[block],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert payload.past_key_values is not None
        # Should have only 2 layers
        count = 0
        for _ in payload.past_key_values:
            count += 1
        assert count == 2

    def test_past_key_values_shape(self) -> None:
        """Verify the shape of KV tensors in the cache."""
        block = _make_kv_block(num_layers=4, num_kv_heads=2, seq_len=16, head_dim=32)
        query = torch.randint(0, 100, (1, 4))

        payload = assemble_kv_injection(
            kv_blocks=[block],
            query_token_ids=query,
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Check first layer
        past_kv = payload.past_key_values
        for layer_kv in past_kv:
            if isinstance(layer_kv, (tuple, list)):
                k, v = layer_kv[0], layer_kv[1]
            else:
                k, v = layer_kv
            assert k.shape == (1, 2, 16, 32)  # (batch, kv_heads, seq, head_dim)
            assert v.shape == (1, 2, 16, 32)
            break  # just check first layer


class TestEncodeContextToKV:
    """Tests for encode_context_to_kv() and chunked prefill."""

    def _make_model(self):
        model = DummyModel(hidden_size=64, num_layers=4, num_heads=4)
        model.load()
        return model

    def test_basic_encode(self) -> None:
        model = self._make_model()
        payload, kv_blocks, ratio = encode_context_to_kv(
            context_text="Some context text",
            query_text="What is the answer?",
            model=model,
        )
        assert payload.past_key_values is not None
        assert payload.kv_seq_len == 16  # DummyModel fixed seq_len
        assert len(kv_blocks) == 1
        assert ratio == 0.0  # no compression

    def test_chunked_basic(self) -> None:
        """Chunked prefill produces a payload with correct structure."""
        model = self._make_model()
        payload, kv_blocks, ratio = encode_context_to_kv(
            context_text="Some context text",
            query_text="What is the answer?",
            model=model,
            chunk_size=4,  # Force chunking (DummyModel produces 16 tokens)
        )
        assert payload.past_key_values is not None
        assert payload.kv_seq_len == 16
        assert len(kv_blocks) == 1
        # KV should cover all 16 context tokens (accumulated across chunks)
        first_layer = list(payload.past_key_values)[0]
        if isinstance(first_layer, (tuple, list)):
            k = first_layer[0]
        else:
            k = first_layer
        assert k.shape[2] == 16  # total KV seq_len

    def test_chunked_fallthrough(self) -> None:
        """When context is smaller than chunk_size, no chunking occurs."""
        model = self._make_model()
        # DummyModel returns 16 tokens; chunk_size=32 should skip chunking
        payload, _, _ = encode_context_to_kv(
            context_text="Short text",
            query_text="Question?",
            model=model,
            chunk_size=32,
        )
        assert payload.past_key_values is not None
        assert payload.kv_seq_len == 16

    def test_encode_with_compression(self) -> None:
        """Compression roundtrip works within encode_context_to_kv."""
        from src.compression import create_compressor
        model = self._make_model()
        compressor = create_compressor("int8")
        payload, kv_blocks, ratio = encode_context_to_kv(
            context_text="Some context",
            query_text="Question?",
            model=model,
            compressor=compressor,
        )
        assert payload.past_key_values is not None
        assert len(kv_blocks) == 1
        # INT8 roundtrip should not change shape
        assert kv_blocks[0].num_tokens == 16
