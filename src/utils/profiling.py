"""Profiling utilities: timers, GPU memory tracking, and system info logging."""

from __future__ import annotations

import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import Generator


@dataclass
class TimerResult:
    """Stores the result of a timed operation."""
    name: str
    elapsed: float = 0.0  # seconds


class Timer:
    """Context manager and decorator for measuring wall-clock time.

    Usage as context manager:
        with Timer("forward_pass") as t:
            output = model(input_ids)
        print(t.elapsed)  # seconds

    Usage as decorator:
        @Timer("my_func")
        def my_func():
            ...
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __call__(self, func):
        """Use as a decorator."""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


@dataclass
class GPUMemorySnapshot:
    """Snapshot of GPU memory usage in MB."""
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    delta_mb: float = 0.0


class GPUMemoryTracker:
    """Context manager that tracks GPU memory changes.

    Usage:
        with GPUMemoryTracker() as tracker:
            tensor = torch.randn(1000, 1000, device="cuda")
        print(tracker.snapshot.delta_mb)
    """

    def __init__(self) -> None:
        self.snapshot = GPUMemorySnapshot()
        self._before: float = 0.0

    def __enter__(self) -> "GPUMemoryTracker":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                self._before = torch.cuda.memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / (1024 ** 2)
                self.snapshot = GPUMemorySnapshot(
                    allocated_mb=after,
                    reserved_mb=torch.cuda.memory_reserved() / (1024 ** 2),
                    max_allocated_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
                    delta_mb=after - self._before,
                )
        except ImportError:
            pass


def get_gpu_memory_summary() -> dict[str, float]:
    """Return current GPU memory stats as a dict (values in MB)."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            }
    except ImportError:
        pass
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}


def log_system_info(logger: Logger) -> None:
    """Log system hardware and software information."""
    import psutil

    logger.info("=" * 60)
    logger.info("SYSTEM INFO")
    logger.info("=" * 60)

    # Python
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")

    # CPU & RAM
    logger.info(f"CPU cores: {psutil.cpu_count(logical=True)}")
    mem = psutil.virtual_memory()
    logger.info(f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")

    # PyTorch & CUDA
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name}")
            logger.info(f"VRAM: {props.total_memory / (1024**3):.1f} GB")
        else:
            logger.info("CUDA: not available")
    except ImportError:
        logger.info("PyTorch: not installed")

    # Transformers
    try:
        import transformers
        logger.info(f"Transformers: {transformers.__version__}")
    except ImportError:
        logger.info("Transformers: not installed")

    logger.info("=" * 60)
