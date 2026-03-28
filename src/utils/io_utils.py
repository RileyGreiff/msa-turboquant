"""I/O utilities for file operations, path management, and data persistence."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml in any parent directory)"
    )


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_str() -> str:
    """Return a filesystem-safe timestamp string: YYYYMMDD_HHMMSS."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str, max_length: int = 200) -> str:
    """Sanitize a string for use as a filename on Windows/Linux.

    Replaces problematic characters, truncates to max_length.
    """
    # Replace any non-alphanumeric, non-dash, non-underscore, non-dot with underscore
    cleaned = re.sub(r"[^\w\-.]", "_", name)
    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Strip leading/trailing underscores and dots
    cleaned = cleaned.strip("_.")
    return cleaned[:max_length]


# --- JSON ---

def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """Save data as JSON with pretty-printing."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str, ensure_ascii=False)


def load_json(path: Path | str) -> Any:
    """Load JSON from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- CSV ---

def save_csv(rows: list[dict[str, Any]], path: Path | str) -> None:
    """Save a list of dicts as a CSV file. Keys from first row become headers."""
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path | str) -> list[dict[str, str]]:
    """Load a CSV file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# --- YAML ---

def save_yaml(data: dict[str, Any], path: Path | str) -> None:
    """Save data as YAML."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load YAML from a file using safe_load."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML file to contain a mapping, got {type(data).__name__}")
    return data
