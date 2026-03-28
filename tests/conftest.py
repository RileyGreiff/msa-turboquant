"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ExperimentConfig, load_config


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the path to the configs directory."""
    return project_root / "configs"


@pytest.fixture
def sample_config(config_dir: Path) -> ExperimentConfig:
    """Load the default experiment config."""
    return load_config(config_dir)


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Create and return a temporary results directory."""
    results = tmp_path / "results"
    results.mkdir()
    return results
