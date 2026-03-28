"""Tests for model wrappers and KV extraction.

Unit tests use mock objects to avoid downloading real models.
Integration tests (marked @pytest.mark.slow) require a real model on disk/hub.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.base_model import BaseModel, ModelOutput, TokenizedInput
from src.models.kv_extractor import KVBlock, KVExtractor


# ---------------------------------------------------------------------------
# Concrete test implementation of BaseModel
# ---------------------------------------------------------------------------

class DummyModel(BaseModel):
    """Minimal BaseModel implementation for testing the abstract interface."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 4, num_heads: int = 4):
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._hidden_size // self._num_heads

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def tokenize(self, texts, max_length=None, padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        seq_len = 16  # fixed for testing
        return TokenizedInput(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
            num_tokens=[seq_len] * batch_size,
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                output_attentions=False, use_cache=False, past_key_values=None):
        batch, seq_len = input_ids.shape
        h = self._hidden_size
        num_heads = self._num_heads
        head_dim = h // num_heads

        # When past_key_values provided, total KV length = past + current
        past_seq_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            if isinstance(first_layer, (tuple, list)):
                past_seq_len = first_layer[0].shape[2]
            else:
                past_seq_len = first_layer.shape[2]
        total_kv_len = past_seq_len + seq_len

        result = ModelOutput()
        result.logits = torch.randn(batch, seq_len, 1000)
        result.last_hidden_state = torch.randn(batch, seq_len, h)

        if output_hidden_states:
            # num_layers + 1 (including embedding layer)
            result.hidden_states = tuple(
                torch.randn(batch, seq_len, h)
                for _ in range(self._num_layers + 1)
            )

        if use_cache:
            result.kv_cache = tuple(
                (
                    torch.randn(batch, num_heads, total_kv_len, head_dim),
                    torch.randn(batch, num_heads, total_kv_len, head_dim),
                )
                for _ in range(self._num_layers)
            )

        return result

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128, **kwargs):
        batch, seq_len = input_ids.shape
        new_tokens = torch.randint(0, 1000, (batch, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)


# ---------------------------------------------------------------------------
# BaseModel interface tests
# ---------------------------------------------------------------------------

class TestBaseModel:
    """Tests for the BaseModel abstract interface via DummyModel."""

    def test_load_unload(self) -> None:
        model = DummyModel()
        model.load()
        assert model._loaded
        model.unload()
        assert not model._loaded

    def test_properties(self) -> None:
        model = DummyModel(hidden_size=128, num_layers=6, num_heads=8)
        model.load()
        assert model.hidden_size == 128
        assert model.num_layers == 6
        assert model.num_heads == 8
        assert model.head_dim == 16

    def test_tokenize(self) -> None:
        model = DummyModel()
        model.load()
        tokens = model.tokenize("Hello world")
        assert isinstance(tokens, TokenizedInput)
        assert tokens.input_ids.shape[0] == 1  # batch of 1
        assert tokens.attention_mask.shape == tokens.input_ids.shape

    def test_tokenize_batch(self) -> None:
        model = DummyModel()
        model.load()
        tokens = model.tokenize(["Hello", "World"])
        assert tokens.input_ids.shape[0] == 2

    def test_forward_basic(self) -> None:
        model = DummyModel()
        model.load()
        tokens = model.tokenize("Test")
        output = model.forward(tokens.input_ids, tokens.attention_mask)
        assert isinstance(output, ModelOutput)
        assert output.logits is not None

    def test_forward_with_hidden_states(self) -> None:
        model = DummyModel(num_layers=4)
        model.load()
        tokens = model.tokenize("Test")
        output = model.forward(tokens.input_ids, tokens.attention_mask, output_hidden_states=True)
        assert output.hidden_states is not None
        assert len(output.hidden_states) == 5  # 4 layers + embedding

    def test_forward_with_kv_cache(self) -> None:
        model = DummyModel(num_layers=4, num_heads=4, hidden_size=64)
        model.load()
        tokens = model.tokenize("Test")
        output = model.forward(tokens.input_ids, tokens.attention_mask, use_cache=True)
        assert output.kv_cache is not None
        assert len(output.kv_cache) == 4
        k, v = output.kv_cache[0]
        assert k.shape[-1] == 16  # head_dim = 64/4

    def test_get_hidden_states(self) -> None:
        model = DummyModel(hidden_size=64)
        model.load()
        hidden = model.get_hidden_states("Test", layer=-1)
        assert hidden.shape[-1] == 64

    def test_get_routing_vectors_mean(self) -> None:
        model = DummyModel(hidden_size=64)
        model.load()
        vectors = model.get_routing_vectors("Test", pooling="mean")
        assert vectors.shape == (1, 64)

    def test_get_routing_vectors_last(self) -> None:
        model = DummyModel(hidden_size=64)
        model.load()
        vectors = model.get_routing_vectors("Test", pooling="last")
        assert vectors.shape == (1, 64)

    def test_get_routing_vectors_cls(self) -> None:
        model = DummyModel(hidden_size=64)
        model.load()
        vectors = model.get_routing_vectors("Test", pooling="cls")
        assert vectors.shape == (1, 64)

    def test_get_routing_vectors_invalid_pooling(self) -> None:
        model = DummyModel()
        model.load()
        with pytest.raises(ValueError, match="Unknown pooling"):
            model.get_routing_vectors("Test", pooling="invalid")

    def test_forward_with_past_key_values(self) -> None:
        model = DummyModel(num_layers=4, num_heads=4, hidden_size=64)
        model.load()
        tokens = model.tokenize("Test")
        # First forward: get initial KV cache
        out1 = model.forward(tokens.input_ids, tokens.attention_mask, use_cache=True)
        assert out1.kv_cache is not None
        initial_kv_len = out1.kv_cache[0][0].shape[2]
        # Second forward with past_key_values: KV should accumulate
        out2 = model.forward(
            tokens.input_ids, tokens.attention_mask,
            use_cache=True, past_key_values=out1.kv_cache,
        )
        assert out2.kv_cache is not None
        new_kv_len = out2.kv_cache[0][0].shape[2]
        assert new_kv_len == initial_kv_len + tokens.input_ids.shape[1]

    def test_generate(self) -> None:
        model = DummyModel()
        model.load()
        tokens = model.tokenize("Test")
        output = model.generate(tokens.input_ids, tokens.attention_mask, max_new_tokens=10)
        assert output.shape[1] == tokens.input_ids.shape[1] + 10


# ---------------------------------------------------------------------------
# KVBlock tests
# ---------------------------------------------------------------------------

class TestKVBlock:
    """Tests for KVBlock data structure."""

    def _make_kv_block(self, num_layers=2, num_heads=4, seq_len=16, head_dim=16):
        return KVBlock(
            block_id="test_block",
            keys=[torch.randn(num_heads, seq_len, head_dim) for _ in range(num_layers)],
            values=[torch.randn(num_heads, seq_len, head_dim) for _ in range(num_layers)],
            routing_vector=torch.randn(num_heads * head_dim),
            num_tokens=seq_len,
            layer_indices=list(range(num_layers)),
        )

    def test_num_layers(self) -> None:
        block = self._make_kv_block(num_layers=3)
        assert block.num_layers == 3

    def test_total_kv_bytes(self) -> None:
        block = self._make_kv_block(num_layers=2, num_heads=4, seq_len=16, head_dim=16)
        # 2 layers * 2 (k+v) * 4 heads * 16 seq * 16 head_dim * 4 bytes (float32)
        expected = 2 * 2 * 4 * 16 * 16 * 4
        assert block.total_kv_bytes == expected

    def test_to_cpu(self) -> None:
        block = self._make_kv_block()
        cpu_block = block.to_cpu()
        assert all(k.device.type == "cpu" for k in cpu_block.keys)
        assert cpu_block.routing_vector.device.type == "cpu"

    def test_to_device(self) -> None:
        block = self._make_kv_block()
        moved = block.to_device("cpu")
        assert moved.block_id == block.block_id
        assert moved.num_tokens == block.num_tokens


# ---------------------------------------------------------------------------
# KVExtractor tests (using DummyModel)
# ---------------------------------------------------------------------------

class TestKVExtractor:
    """Tests for KV extraction using the DummyModel."""

    def test_direct_extraction(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=4, num_heads=4)
        model.load()
        extractor = KVExtractor(model, mode="direct")
        block = extractor.extract("Test text", block_id="blk_0")

        assert isinstance(block, KVBlock)
        assert block.block_id == "blk_0"
        assert block.num_layers == 4
        assert block.routing_vector.shape == (64,)
        # Routing vector should be L2-normalized
        norm = torch.linalg.norm(block.routing_vector)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_hidden_state_extraction(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=4, num_heads=4)
        model.load()
        extractor = KVExtractor(model, mode="hidden_state")
        block = extractor.extract("Test text", block_id="blk_1")

        assert isinstance(block, KVBlock)
        assert block.num_layers == 4
        # K/V should have correct shape: (num_heads, seq_len, head_dim)
        assert block.keys[0].shape[0] == 4  # num_heads
        assert block.keys[0].shape[2] == 16  # head_dim = 64/4

    def test_layer_selection(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=4, num_heads=4)
        model.load()
        extractor = KVExtractor(model, mode="direct", layers=[0, 2])
        block = extractor.extract("Test")

        assert block.num_layers == 2
        assert block.layer_indices == [0, 2]

    def test_extract_batch(self) -> None:
        model = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
        model.load()
        extractor = KVExtractor(model, mode="direct")
        blocks = extractor.extract_batch(
            ["Text A", "Text B", "Text C"],
            block_ids=["a", "b", "c"],
        )
        assert len(blocks) == 3
        assert blocks[0].block_id == "a"
        assert blocks[2].block_id == "c"

    def test_invalid_mode_raises(self) -> None:
        model = DummyModel()
        model.load()
        with pytest.raises(ValueError, match="Unknown extraction mode"):
            KVExtractor(model, mode="invalid")

    def test_routing_vector_normalized(self) -> None:
        """Routing vectors should be L2-normalized for cosine similarity."""
        model = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
        model.load()
        for mode in ("direct", "hidden_state"):
            extractor = KVExtractor(model, mode=mode)
            block = extractor.extract("Test")
            norm = torch.linalg.norm(block.routing_vector)
            assert abs(norm.item() - 1.0) < 1e-5, f"mode={mode}: norm={norm.item()}"


# ---------------------------------------------------------------------------
# HFModel import test (no model download)
# ---------------------------------------------------------------------------

class TestHFModelImport:
    """Verify HFModel can be imported and instantiated (without loading)."""

    def test_import(self) -> None:
        from src.models.hf_model import HFModel
        from src.utils.config import ModelConfig
        model = HFModel(ModelConfig(name="test/model"))
        assert model.model_name == "test/model"

    def test_not_loaded_raises(self) -> None:
        from src.models.hf_model import HFModel
        from src.utils.config import ModelConfig
        model = HFModel(ModelConfig(name="test/model"))
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = model.hidden_size
