"""Systems-level metrics collection for experiment runs.

Collects VRAM, RAM, latency, throughput, and disk I/O metrics.
Designed to be used as a context manager around experiment sections.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger("msa_turboquant.eval.systems_metrics")


@dataclass
class SystemsSnapshot:
    """A single snapshot of system resource usage.

    All memory values are in MB. Times are in milliseconds.
    """
    timestamp: float = 0.0
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "ram_used_mb": round(self.ram_used_mb, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
            "gpu_allocated_mb": round(self.gpu_allocated_mb, 1),
            "gpu_reserved_mb": round(self.gpu_reserved_mb, 1),
            "gpu_max_allocated_mb": round(self.gpu_max_allocated_mb, 1),
        }


@dataclass
class RunMetrics:
    """Collected metrics for a single experiment run.

    Attributes:
        run_id: Identifier for this run.
        mode: Experiment mode (dense, sparse, etc.).
        config: Subset of config relevant to this run.
        before: System snapshot taken before the run.
        after: System snapshot taken after the run.
        wall_time_ms: Total wall-clock time in ms.
        retrieval_time_ms: Time spent on routing + fetching.
        generation_time_ms: Time spent on model generation.
        eval_time_ms: Time spent on evaluation/scoring.
        num_samples: Number of samples evaluated.
        num_tokens_generated: Total tokens generated.
        tokens_per_second: Generation throughput.
        retrieval_metrics: Retrieval quality metrics dict.
        answer_metrics: Answer quality metrics dict.
        extra: Arbitrary extra metrics.
    """
    run_id: str = ""
    mode: str = ""
    config: dict = field(default_factory=dict)
    before: SystemsSnapshot = field(default_factory=SystemsSnapshot)
    after: SystemsSnapshot = field(default_factory=SystemsSnapshot)
    wall_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    num_samples: int = 0
    num_tokens_generated: int = 0
    tokens_per_second: float = 0.0
    retrieval_metrics: dict = field(default_factory=dict)
    answer_metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def peak_gpu_mb(self) -> float:
        return self.after.gpu_max_allocated_mb

    @property
    def gpu_delta_mb(self) -> float:
        return self.after.gpu_allocated_mb - self.before.gpu_allocated_mb

    @property
    def ram_delta_mb(self) -> float:
        return self.after.ram_used_mb - self.before.ram_used_mb

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "wall_time_ms": round(self.wall_time_ms, 1),
            "retrieval_time_ms": round(self.retrieval_time_ms, 1),
            "generation_time_ms": round(self.generation_time_ms, 1),
            "eval_time_ms": round(self.eval_time_ms, 1),
            "num_samples": self.num_samples,
            "num_tokens_generated": self.num_tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "peak_gpu_mb": round(self.peak_gpu_mb, 1),
            "gpu_delta_mb": round(self.gpu_delta_mb, 1),
            "ram_delta_mb": round(self.ram_delta_mb, 1),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "retrieval_metrics": self.retrieval_metrics,
            "answer_metrics": self.answer_metrics,
            "config": self.config,
            "extra": self.extra,
        }


def take_snapshot() -> SystemsSnapshot:
    """Take a snapshot of current system resource usage."""
    mem = psutil.virtual_memory()
    snap = SystemsSnapshot(
        timestamp=time.time(),
        ram_used_mb=mem.used / (1024 ** 2),
        ram_available_mb=mem.available / (1024 ** 2),
    )

    try:
        import torch
        if torch.cuda.is_available():
            snap.gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            snap.gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            snap.gpu_max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass

    return snap


def reset_gpu_peak_stats() -> None:
    """Reset GPU peak memory tracking for accurate per-run measurement."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class MetricsCollector:
    """Context manager for collecting systems metrics around an operation.

    Usage:
        collector = MetricsCollector(run_id="run_0", mode="sparse")
        with collector:
            # ... run experiment ...
            collector.add_retrieval_time(15.2)
            collector.add_generation_time(230.5)
        metrics = collector.get_metrics()
    """

    def __init__(self, run_id: str = "", mode: str = "", config: dict | None = None):
        self._run_id = run_id
        self._mode = mode
        self._config = config or {}
        self._start_time: float = 0.0
        self._before: SystemsSnapshot = SystemsSnapshot()
        self._retrieval_time_ms: float = 0.0
        self._generation_time_ms: float = 0.0
        self._eval_time_ms: float = 0.0
        self._tokens_generated: int = 0
        self._retrieval_metrics: dict = {}
        self._answer_metrics: dict = {}
        self._num_samples: int = 0
        self._extra: dict = {}
        self._finished = False

    def __enter__(self) -> "MetricsCollector":
        reset_gpu_peak_stats()
        self._before = take_snapshot()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._finished = True

    def add_retrieval_time(self, ms: float) -> None:
        self._retrieval_time_ms += ms

    def add_generation_time(self, ms: float) -> None:
        self._generation_time_ms += ms

    def add_eval_time(self, ms: float) -> None:
        self._eval_time_ms += ms

    def add_tokens(self, count: int) -> None:
        self._tokens_generated += count

    def set_retrieval_metrics(self, metrics: dict) -> None:
        self._retrieval_metrics = metrics

    def set_answer_metrics(self, metrics: dict) -> None:
        self._answer_metrics = metrics

    def set_num_samples(self, n: int) -> None:
        self._num_samples = n

    def add_extra(self, key: str, value: Any) -> None:
        self._extra[key] = value

    def get_metrics(self) -> RunMetrics:
        """Build the final RunMetrics object."""
        after = take_snapshot()
        wall_time = (time.perf_counter() - self._start_time) * 1000 if self._start_time else 0.0
        gen_seconds = self._generation_time_ms / 1000 if self._generation_time_ms > 0 else 0.0
        tps = self._tokens_generated / gen_seconds if gen_seconds > 0 else 0.0

        return RunMetrics(
            run_id=self._run_id,
            mode=self._mode,
            config=self._config,
            before=self._before,
            after=after,
            wall_time_ms=wall_time,
            retrieval_time_ms=self._retrieval_time_ms,
            generation_time_ms=self._generation_time_ms,
            eval_time_ms=self._eval_time_ms,
            num_samples=self._num_samples,
            num_tokens_generated=self._tokens_generated,
            tokens_per_second=tps,
            retrieval_metrics=self._retrieval_metrics,
            answer_metrics=self._answer_metrics,
            extra=self._extra,
        )
