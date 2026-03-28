"""Tests for the needle-in-a-haystack dataset generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.niah import (
    NIAHSample,
    generate_niah_dataset,
    generate_niah_sample,
    load_niah_dataset,
    make_fact_needle,
    make_passkey_needle,
    save_niah_dataset,
)


class TestNeedleGeneration:
    """Tests for individual needle creation."""

    def test_fact_needle_has_answer(self) -> None:
        needle = make_fact_needle(fact_id=0)
        assert needle.answer.isdigit()
        assert len(needle.answer) == 4
        assert needle.question
        assert needle.answer in needle.text

    def test_passkey_needle_has_answer(self) -> None:
        needle = make_passkey_needle(passkey_id=0)
        assert needle.answer.isdigit()
        assert len(needle.answer) == 5
        assert needle.answer in needle.text

    def test_reproducibility_with_seed(self) -> None:
        """Same seed produces identical needles."""
        import random
        n1 = make_fact_needle(0, rng=random.Random(42))
        n2 = make_fact_needle(0, rng=random.Random(42))
        assert n1.answer == n2.answer
        assert n1.text == n2.text


class TestGenerateNIAHSample:
    """Tests for single sample generation."""

    def test_basic_generation(self) -> None:
        sample = generate_niah_sample(num_blocks=10, block_chars=200, seed=42)
        assert isinstance(sample, NIAHSample)
        assert sample.num_blocks == 10
        assert len(sample.blocks) == 10
        assert len(sample.needles) == 1
        assert len(sample.needle_block_indices) == 1

    def test_needle_placed_in_correct_block(self) -> None:
        """The needle text appears in the block at needle_block_indices."""
        sample = generate_niah_sample(num_blocks=10, seed=42)
        for needle in sample.needles:
            block_text = sample.blocks[needle.block_index]
            assert needle.answer in block_text

    def test_multi_needle(self) -> None:
        sample = generate_niah_sample(num_blocks=20, num_needles=3, seed=42)
        assert len(sample.needles) == 3
        assert len(sample.needle_block_indices) == 3
        # All needles should be in distinct blocks
        assert len(set(sample.needle_block_indices)) == 3

    def test_position_early(self) -> None:
        """Early position places needles in the first quarter."""
        sample = generate_niah_sample(num_blocks=40, num_needles=1, position="early", seed=42)
        idx = sample.needle_block_indices[0]
        assert idx < 40 // 4 + 1

    def test_position_late(self) -> None:
        """Late position places needles in the last quarter."""
        sample = generate_niah_sample(num_blocks=40, num_needles=1, position="late", seed=42)
        idx = sample.needle_block_indices[0]
        assert idx >= 40 * 3 // 4 - 1

    def test_position_float_depth(self) -> None:
        """Float position places needle near the specified depth."""
        sample = generate_niah_sample(num_blocks=100, num_needles=1, position=0.5, seed=42)
        idx = sample.needle_block_indices[0]
        # Should be roughly in the middle (allow 20% tolerance)
        assert 30 <= idx <= 70

    def test_passkey_type(self) -> None:
        sample = generate_niah_sample(num_blocks=5, needle_type="passkey", seed=42)
        assert sample.needles[0].answer.isdigit()
        assert len(sample.needles[0].answer) == 5

    def test_total_chars_correct(self) -> None:
        sample = generate_niah_sample(num_blocks=5, block_chars=200, seed=42)
        actual = sum(len(b) for b in sample.blocks)
        assert sample.total_chars == actual

    def test_reproducibility(self) -> None:
        """Same seed produces identical samples."""
        s1 = generate_niah_sample(num_blocks=10, seed=123)
        s2 = generate_niah_sample(num_blocks=10, seed=123)
        assert s1.blocks == s2.blocks
        assert s1.needles[0].answer == s2.needles[0].answer

    def test_too_many_needles_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot place"):
            generate_niah_sample(num_blocks=2, num_needles=5, seed=42)


class TestGenerateNIAHDataset:
    """Tests for batch dataset generation."""

    def test_fixed_count(self) -> None:
        dataset = generate_niah_dataset(num_samples=5, seed=42)
        assert len(dataset) == 5

    def test_sweep_over_block_counts(self) -> None:
        dataset = generate_niah_dataset(
            num_blocks_list=[10, 20, 50],
            seed=42,
        )
        assert len(dataset) == 3
        assert dataset[0].num_blocks == 10
        assert dataset[1].num_blocks == 20
        assert dataset[2].num_blocks == 50


class TestNIAHPersistence:
    """Tests for save/load round-trip."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Saving and loading preserves all sample data."""
        dataset = generate_niah_dataset(num_samples=3, seed=42)
        path = tmp_path / "test_niah.json"
        save_niah_dataset(dataset, path)
        assert path.exists()

        loaded = load_niah_dataset(path)
        assert len(loaded) == 3
        assert loaded[0].sample_id == dataset[0].sample_id
        assert loaded[0].needles[0].answer == dataset[0].needles[0].answer
        assert loaded[0].blocks == dataset[0].blocks

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_niah_dataset(tmp_path / "nonexistent.json")

    def test_serialization_round_trip(self) -> None:
        """to_dict/from_dict preserves all fields."""
        sample = generate_niah_sample(num_blocks=5, num_needles=2, seed=42)
        d = sample.to_dict()
        restored = NIAHSample.from_dict(d)
        assert restored.sample_id == sample.sample_id
        assert len(restored.needles) == 2
        assert restored.needles[0].answer == sample.needles[0].answer
        assert restored.needle_block_indices == sample.needle_block_indices
