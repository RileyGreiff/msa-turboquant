"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ExperimentConfig, load_config


class TestLoadConfig:
    """Tests for load_config and ExperimentConfig."""

    def test_load_defaults(self, config_dir: Path) -> None:
        """Loading from default configs returns a valid ExperimentConfig."""
        config = load_config(config_dir)
        assert isinstance(config, ExperimentConfig)
        assert config.name == "baseline_qwen25_3b"
        assert config.seed == 42

    def test_model_config_loaded(self, sample_config: ExperimentConfig) -> None:
        """Model config is populated from model.yaml."""
        assert sample_config.model.name == "Qwen/Qwen2.5-3B-Instruct"
        assert sample_config.model.dtype == "float16"
        assert sample_config.model.max_seq_len == 32768

    def test_retrieval_config_loaded(self, sample_config: ExperimentConfig) -> None:
        """Retrieval config is populated from retrieval.yaml."""
        assert sample_config.retrieval.engine == "faiss"
        assert sample_config.retrieval.top_k == 5
        assert sample_config.retrieval.chunk_size == 512

    def test_compression_config_loaded(self, sample_config: ExperimentConfig) -> None:
        """Compression config is populated from compression.yaml."""
        assert sample_config.compression.method == "none"
        assert sample_config.compression.int4.group_size == 128

    def test_benchmarks_loaded(self, sample_config: ExperimentConfig) -> None:
        """Benchmark tasks are loaded from benchmarks.yaml."""
        assert len(sample_config.benchmarks.tasks) > 0
        task_names = [t.name for t in sample_config.benchmarks.tasks]
        assert "needle_in_haystack" in task_names

    def test_tokenizer_defaults_to_model_name(self, sample_config: ExperimentConfig) -> None:
        """Tokenizer name defaults to model name when null."""
        assert sample_config.tokenizer.name == sample_config.model.name

    def test_override_simple_value(self, config_dir: Path) -> None:
        """CLI overrides change config values."""
        config = load_config(config_dir, overrides=["model.max_seq_len=4096"])
        assert config.model.max_seq_len == 4096

    def test_override_boolean(self, config_dir: Path) -> None:
        """CLI overrides handle boolean values."""
        config = load_config(config_dir, overrides=["model.trust_remote_code=false"])
        assert config.model.trust_remote_code is False

    def test_override_string(self, config_dir: Path) -> None:
        """CLI overrides handle string values."""
        config = load_config(config_dir, overrides=["compression.method=int8"])
        assert config.compression.method == "int8"

    def test_extra_field_rejected(self, config_dir: Path) -> None:
        """extra='forbid' catches unknown config keys."""
        with pytest.raises(Exception):
            load_config(config_dir, overrides=["model.nonexistent_field=123"])

    def test_invalid_config_dir(self) -> None:
        """Missing config directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/to/configs")

    def test_paths_resolve(self, sample_config: ExperimentConfig, tmp_path: Path) -> None:
        """PathsConfig.resolve creates directories."""
        resolved = sample_config.paths.resolve(tmp_path)
        for key, path in resolved.items():
            assert path.exists(), f"{key} directory was not created: {path}"
            assert path.is_dir()
