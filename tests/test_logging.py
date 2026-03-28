"""Tests for logging utilities."""

from __future__ import annotations

import json
import logging

from src.utils.config import LoggingConfig
from src.utils.logging_utils import JsonFormatter, setup_logging


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_returns_logger(self) -> None:
        """setup_logging returns a Logger instance."""
        config = LoggingConfig(
            level="DEBUG",
            console_format="plain",
            file_output="",
            json_logs=False,
        )
        logger = setup_logging(config, experiment_name="test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "msa_turboquant"

    def test_log_level_set(self) -> None:
        """Logger level matches config."""
        config = LoggingConfig(
            level="WARNING",
            console_format="plain",
            file_output="",
            json_logs=False,
        )
        logger = setup_logging(config)
        assert logger.level == logging.WARNING

    def test_file_output(self, tmp_path) -> None:
        """File handler creates a log file."""
        log_path = str(tmp_path / "{experiment_name}_{timestamp}.log")
        config = LoggingConfig(
            level="INFO",
            console_format="plain",
            file_output=log_path,
            json_logs=True,
        )
        logger = setup_logging(config, experiment_name="test_exp")
        logger.info("Test message")

        # Check a log file was created
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) >= 1


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_produces_valid_json(self) -> None:
        """JsonFormatter output is valid JSON with expected keys."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Hello world"
        assert "timestamp" in parsed
        assert "logger" in parsed
