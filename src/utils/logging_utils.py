"""Logging utilities with structured JSON file output and rich console formatting."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.io_utils import ensure_dir, timestamp_str

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig, LoggingConfig


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON lines for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }
        # Include any extra fields attached to the record
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry, default=str)


def setup_logging(
    config: LoggingConfig,
    experiment_name: str = "",
) -> logging.Logger:
    """Configure the root logger with console and optional file handlers.

    Args:
        config: LoggingConfig with level, format, and file output settings.
        experiment_name: Used to template the log file path.

    Returns:
        The configured root logger.
    """
    root_logger = logging.getLogger("msa_turboquant")
    root_logger.setLevel(getattr(logging, config.level))

    # Remove existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    # --- Console handler ---
    if config.console_format == "rich":
        try:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                level=config.level,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
            )
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root_logger.addHandler(console_handler)

    # --- File handler (JSON lines) ---
    if config.file_output:
        ts = timestamp_str()
        log_path = config.file_output.format(
            experiment_name=experiment_name or "unnamed",
            timestamp=ts,
        )
        log_path = Path(log_path)
        ensure_dir(log_path.parent)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        if config.json_logs:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
        root_logger.addHandler(file_handler)
        root_logger.info(f"Log file: {log_path}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the msa_turboquant namespace."""
    return logging.getLogger(f"msa_turboquant.{name}")


def log_gpu_memory(logger: logging.Logger) -> None:
    """Log current GPU memory usage if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            logger.info(
                f"GPU memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved"
            )
        else:
            logger.debug("CUDA not available, skipping GPU memory log")
    except ImportError:
        logger.debug("PyTorch not installed, skipping GPU memory log")


def log_config(logger: logging.Logger, config: ExperimentConfig) -> None:
    """Log the full experiment configuration as formatted text."""
    logger.info(f"Experiment: {config.name} (seed={config.seed}, mode={config.mode})")
    logger.info(f"Model: {config.model.name} ({config.model.dtype})")
    logger.info(f"Retrieval: engine={config.retrieval.engine}, top_k={config.retrieval.top_k}")
    logger.info(f"Compression: {config.compression.method}")
    enabled_tasks = [t.name for t in config.benchmarks.tasks if t.enabled]
    logger.info(f"Benchmarks: {enabled_tasks}")
