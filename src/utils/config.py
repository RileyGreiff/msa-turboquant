"""Configuration models and loader for experiment configs.

All config sections are Pydantic v2 BaseModels with extra="forbid" to catch typos.
The load_config() function merges all YAML files from a config directory into a
single ExperimentConfig instance, with optional dot-path overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.utils.io_utils import load_yaml


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """LLM model configuration."""
    model_config = ConfigDict(extra="forbid")

    name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "float16"
    device: str = "cuda:0"
    max_seq_len: int = 32768
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None  # defaults to model name if None
    padding_side: str = "left"
    max_length: int = 32768


class RetrievalConfig(BaseModel):
    """Retrieval and routing configuration."""
    model_config = ConfigDict(extra="forbid")

    engine: Literal["faiss", "torch_cosine"] = "faiss"
    index_type: str = "IndexFlatIP"
    nprobe: int = 10
    top_k: int = 5
    normalize_embeddings: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 64
    routing_vector: str = "mean_hidden_state"
    routing_dim: Optional[int] = None


class Int8Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symmetric: bool = True


class Int4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_size: int = 128
    symmetric: bool = True


class TurboQuantMSEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rotation: Literal["random_orthogonal", "hadamard"] = "random_orthogonal"
    bits: int = 4
    seed: int = 42


# Backwards compat aliases
TurboQuantLikeConfig = TurboQuantMSEConfig
RotatedUniformConfig = TurboQuantMSEConfig


class CompressionConfig(BaseModel):
    """Compression method configuration."""
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "none", "fp16", "int8", "int4", "turboquant_mse",
    ] = "none"
    int8: Int8Config = Int8Config()
    int4: Int4Config = Int4Config()
    turboquant_mse: TurboQuantMSEConfig = TurboQuantMSEConfig()


class BenchmarkTaskConfig(BaseModel):
    """Single benchmark task configuration."""
    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    context_lengths: list[int] = [4096, 8192, 16384]
    num_needles: int = 1
    num_distractors: int = 10
    num_trials: int = 3
    needle_position: str = "random"
    dataset: Optional[str] = None
    max_samples: Optional[int] = None


class BenchmarkConfig(BaseModel):
    """Benchmark suite configuration."""
    model_config = ConfigDict(extra="forbid")

    tasks: list[BenchmarkTaskConfig] = []
    metrics: list[str] = ["accuracy", "latency_ms", "peak_vram_mb"]
    output_dir: str = "results/"
    save_predictions: bool = True
    save_retrieved_blocks: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    console_format: Literal["rich", "plain"] = "rich"
    file_output: str = "results/logs/{experiment_name}_{timestamp}.log"
    json_logs: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"Invalid log level '{v}', must be one of {valid}")
        return v_upper


class DeviceConfig(BaseModel):
    """Device and parallelism configuration."""
    model_config = ConfigDict(extra="forbid")

    gpu_id: int = 0
    pin_memory: bool = True
    num_workers: int = 0  # 0 for Windows compatibility


class PathsConfig(BaseModel):
    """Project path configuration. Paths are relative to project root."""
    model_config = ConfigDict(extra="forbid")

    data_dir: str = "data/"
    raw_dir: str = "data/raw/"
    processed_dir: str = "data/processed/"
    memory_banks_dir: str = "data/memory_banks/"
    results_dir: str = "results/"
    configs_dir: str = "configs/"

    def resolve(self, root: Path) -> dict[str, Path]:
        """Resolve all paths relative to a root directory and ensure they exist."""
        resolved = {}
        for field_name in self.__class__.model_fields:
            rel_path = getattr(self, field_name)
            abs_path = (root / rel_path).resolve()
            abs_path.mkdir(parents=True, exist_ok=True)
            resolved[field_name] = abs_path
        return resolved


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Top-level experiment configuration, merging all sub-configs."""
    model_config = ConfigDict(extra="forbid")

    # Experiment metadata
    name: str = "default_experiment"
    seed: int = 42
    mode: str = "dense"

    # Sub-configs
    logging: LoggingConfig = LoggingConfig()
    device: DeviceConfig = DeviceConfig()
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    compression: CompressionConfig = CompressionConfig()
    benchmarks: BenchmarkConfig = BenchmarkConfig()

    @model_validator(mode="after")
    def set_tokenizer_default(self) -> "ExperimentConfig":
        """Default tokenizer name to model name if not set."""
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        return self


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base. Overlay values take precedence."""
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-path overrides like 'model.max_seq_len=4096' to a nested dict.

    Values are auto-cast: 'true'/'false' -> bool, integers -> int, floats -> float,
    'null'/'none' -> None, otherwise str.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in 'key.subkey=value' format, got: '{override}'")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")

        # Auto-cast the value
        value: Any
        lower = raw_value.strip().lower()
        if lower in ("true",):
            value = True
        elif lower in ("false",):
            value = False
        elif lower in ("null", "none"):
            value = None
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip()

        # Navigate to the target and set value
        target = config_dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    return config_dict


# Map YAML filenames to their expected top-level keys
_YAML_FILES = [
    "experiment.yaml",
    "model.yaml",
    "retrieval.yaml",
    "compression.yaml",
    "benchmarks.yaml",
]


def load_config(
    config_dir: Path | str = "configs",
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load and merge all YAML config files into an ExperimentConfig.

    Args:
        config_dir: Directory containing YAML config files.
        overrides: Optional list of 'key.subkey=value' strings.

    Returns:
        A validated ExperimentConfig instance.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # Load and merge all YAML files
    merged: dict[str, Any] = {}
    for filename in _YAML_FILES:
        filepath = config_dir / filename
        if filepath.exists():
            data = load_yaml(filepath)
            merged = _deep_merge(merged, data)

    # Flatten the 'experiment' key up to top level if present
    # (experiment.yaml wraps its content under 'experiment:')
    if "experiment" in merged:
        experiment_data = merged.pop("experiment")
        # Merge experiment fields into top level (name, seed, mode, logging, device, paths)
        merged = _deep_merge(experiment_data, merged)

    # Apply CLI overrides
    if overrides:
        merged = _apply_overrides(merged, overrides)

    return ExperimentConfig(**merged)
