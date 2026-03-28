"""Tests for I/O utilities."""

from __future__ import annotations

from pathlib import Path

from src.utils.io_utils import (
    load_csv,
    load_json,
    load_yaml,
    project_root,
    safe_filename,
    save_csv,
    save_json,
    save_yaml,
    timestamp_str,
)


class TestJsonRoundTrip:
    """Tests for JSON save/load."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """JSON round-trip preserves data."""
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent JSON file raises FileNotFoundError."""
        import pytest
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")


class TestCsvRoundTrip:
    """Tests for CSV save/load."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """CSV round-trip preserves data (values returned as strings)."""
        rows = [
            {"name": "alpha", "score": "0.95"},
            {"name": "beta", "score": "0.87"},
        ]
        path = tmp_path / "test.csv"
        save_csv(rows, path)
        loaded = load_csv(path)
        assert len(loaded) == 2
        assert loaded[0]["name"] == "alpha"
        assert loaded[1]["score"] == "0.87"

    def test_empty_rows_no_file(self, tmp_path: Path) -> None:
        """Saving empty rows does not create a file."""
        path = tmp_path / "empty.csv"
        save_csv([], path)
        assert not path.exists()


class TestYamlRoundTrip:
    """Tests for YAML save/load."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """YAML round-trip preserves data."""
        data = {"model": {"name": "test", "layers": 12}}
        path = tmp_path / "test.yaml"
        save_yaml(data, path)
        loaded = load_yaml(path)
        assert loaded == data


class TestProjectRoot:
    """Tests for project_root."""

    def test_finds_pyproject_toml(self) -> None:
        """project_root returns a directory containing pyproject.toml."""
        root = project_root()
        assert (root / "pyproject.toml").exists()
        assert root.is_dir()


class TestSafeFilename:
    """Tests for safe_filename."""

    def test_replaces_special_chars(self) -> None:
        assert safe_filename("hello world!@#") == "hello_world"

    def test_truncates_long_names(self) -> None:
        long_name = "a" * 300
        assert len(safe_filename(long_name)) <= 200

    def test_normal_name_unchanged(self) -> None:
        assert safe_filename("experiment_01") == "experiment_01"


class TestTimestampStr:
    """Tests for timestamp_str."""

    def test_format(self) -> None:
        """Timestamp string matches YYYYMMDD_HHMMSS pattern."""
        ts = timestamp_str()
        assert len(ts) == 15  # 8 + 1 + 6
        assert ts[8] == "_"
