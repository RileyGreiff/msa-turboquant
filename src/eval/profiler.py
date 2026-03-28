"""Per-phase systems profiler for experiment runs.

Provides fine-grained timing and memory measurement across the evaluation
pipeline.  Each phase (bank build, routing, fetching, compression,
generation, scoring) is tracked independently so callers can identify
bottlenecks without an external profiler.

Usage:
    profiler = RunProfiler(enabled=True)
    with profiler.phase("bank_build"):
        bank = builder.build(blocks)
    with profiler.phase("generate"):
        output = model.generate(...)
    report = profiler.report()
    report.save(output_dir / "profile.json")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from src.utils.io_utils import save_json

logger = logging.getLogger("msa_turboquant.eval.profiler")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """RAM + GPU memory at a point in time (values in MB)."""
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "ram_used_mb": round(self.ram_used_mb, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
            "gpu_allocated_mb": round(self.gpu_allocated_mb, 1),
            "gpu_reserved_mb": round(self.gpu_reserved_mb, 1),
            "gpu_max_allocated_mb": round(self.gpu_max_allocated_mb, 1),
        }


def _take_memory_snapshot() -> MemorySnapshot:
    mem = psutil.virtual_memory()
    snap = MemorySnapshot(
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


@dataclass
class PhaseRecord:
    """Timing + memory for a single phase of the pipeline."""
    name: str
    wall_time_ms: float = 0.0
    before: MemorySnapshot = field(default_factory=MemorySnapshot)
    after: MemorySnapshot = field(default_factory=MemorySnapshot)
    counters: dict[str, Any] = field(default_factory=dict)

    @property
    def gpu_delta_mb(self) -> float:
        return self.after.gpu_allocated_mb - self.before.gpu_allocated_mb

    @property
    def ram_delta_mb(self) -> float:
        return self.after.ram_used_mb - self.before.ram_used_mb

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_time_ms": round(self.wall_time_ms, 2),
            "gpu_delta_mb": round(self.gpu_delta_mb, 1),
            "ram_delta_mb": round(self.ram_delta_mb, 1),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "counters": self.counters,
        }


@dataclass
class ProfilingReport:
    """Full profiling report for an experiment run.

    Attributes:
        run_id: Run identifier.
        phases: Per-phase timing records in execution order.
        total_wall_time_ms: End-to-end wall time.
        peak_gpu_mb: Maximum GPU allocation observed across all phases.
        peak_ram_used_mb: Maximum RAM used observed across all phases.
        total_bytes_fetched: KV bytes fetched from bank.
        total_tokens_generated: Tokens produced by generation.
        tokens_per_second: Generation throughput.
        compression_info: Compression method stats (ratio, bits/value).
        extra: Arbitrary extra fields.
    """
    run_id: str = ""
    phases: list[PhaseRecord] = field(default_factory=list)
    total_wall_time_ms: float = 0.0
    peak_gpu_mb: float = 0.0
    peak_ram_used_mb: float = 0.0
    total_bytes_fetched: int = 0
    total_tokens_generated: int = 0
    tokens_per_second: float = 0.0
    compression_info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def phase_time(self, name: str) -> float:
        """Get wall time in ms for a phase by name. Returns 0 if not found."""
        for p in self.phases:
            if p.name == name:
                return p.wall_time_ms
        return 0.0

    def phase_times(self) -> dict[str, float]:
        """Return {phase_name: wall_time_ms} for all phases."""
        return {p.name: round(p.wall_time_ms, 2) for p in self.phases}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_wall_time_ms": round(self.total_wall_time_ms, 2),
            "peak_gpu_mb": round(self.peak_gpu_mb, 1),
            "peak_ram_used_mb": round(self.peak_ram_used_mb, 1),
            "total_bytes_fetched": self.total_bytes_fetched,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "phase_times": self.phase_times(),
            "phases": [p.to_dict() for p in self.phases],
            "compression_info": self.compression_info,
            "extra": self.extra,
        }

    def summary_lines(self) -> list[str]:
        """Human-readable summary lines."""
        lines = [
            f"=== Profiling Report: {self.run_id} ===",
            f"Total wall time: {self.total_wall_time_ms:.1f} ms",
            f"Peak GPU: {self.peak_gpu_mb:.1f} MB",
            f"Peak RAM: {self.peak_ram_used_mb:.1f} MB",
            f"Bytes fetched: {self.total_bytes_fetched:,}",
            f"Tokens generated: {self.total_tokens_generated}",
            f"Tokens/sec: {self.tokens_per_second:.1f}",
            "",
            "Phase breakdown:",
        ]
        for p in self.phases:
            pct = (p.wall_time_ms / self.total_wall_time_ms * 100) if self.total_wall_time_ms > 0 else 0
            lines.append(
                f"  {p.name:20s}  {p.wall_time_ms:8.1f} ms  ({pct:5.1f}%)  "
                f"GPU Δ={p.gpu_delta_mb:+.1f}MB  RAM Δ={p.ram_delta_mb:+.1f}MB"
            )
        if self.compression_info:
            lines.append("")
            lines.append("Compression:")
            for k, v in self.compression_info.items():
                lines.append(f"  {k}: {v}")
        return lines

    def save(self, path: Path | str) -> Path:
        """Save report as JSON."""
        path = Path(path)
        save_json(self.to_dict(), path)
        logger.info(f"Profiling report saved to {path}")
        return path


# ---------------------------------------------------------------------------
# RunProfiler
# ---------------------------------------------------------------------------

class _PhaseContext:
    """Context manager for timing a single phase."""

    def __init__(self, profiler: "RunProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._start: float = 0.0
        self._before: MemorySnapshot = MemorySnapshot()
        self._counters: dict[str, Any] = {}

    def set_counter(self, key: str, value: Any) -> None:
        """Attach a counter (e.g. bytes_fetched) to this phase."""
        self._counters[key] = value

    def __enter__(self) -> "_PhaseContext":
        if self._profiler.enabled:
            self._before = _take_memory_snapshot()
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._profiler.enabled:
            elapsed = (time.perf_counter() - self._start) * 1000
            after = _take_memory_snapshot()
            record = PhaseRecord(
                name=self._name,
                wall_time_ms=elapsed,
                before=self._before,
                after=after,
                counters=dict(self._counters),
            )
            self._profiler._records.append(record)


class RunProfiler:
    """Per-phase profiler for experiment runs.

    When ``enabled=False``, all operations are no-ops so the profiler can
    stay in calling code without any performance overhead.

    Args:
        run_id: Identifier for the run being profiled.
        enabled: If False, all phase tracking is skipped.
    """

    def __init__(self, run_id: str = "", enabled: bool = True) -> None:
        self.run_id = run_id
        self.enabled = enabled
        self._records: list[PhaseRecord] = []
        self._start_time: float = 0.0
        self._bytes_fetched: int = 0
        self._tokens_generated: int = 0
        self._generation_time_ms: float = 0.0
        self._compression_info: dict[str, Any] = {}
        self._extra: dict[str, Any] = {}

    def start(self) -> None:
        """Mark the start of the profiled run."""
        if self.enabled:
            self._start_time = time.perf_counter()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

    def phase(self, name: str) -> _PhaseContext:
        """Return a context manager that times the named phase."""
        return _PhaseContext(self, name)

    def add_bytes_fetched(self, n: int) -> None:
        """Record bytes fetched from the memory bank."""
        self._bytes_fetched += n

    def add_tokens(self, n: int) -> None:
        """Record tokens generated."""
        self._tokens_generated += n

    def add_generation_time(self, ms: float) -> None:
        """Accumulate generation time for tokens/sec calculation."""
        self._generation_time_ms += ms

    def set_compression_info(self, info: dict[str, Any]) -> None:
        """Record compression method details."""
        self._compression_info = info

    def add_extra(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata."""
        self._extra[key] = value

    def report(self) -> ProfilingReport:
        """Build the final profiling report."""
        total_wall = (time.perf_counter() - self._start_time) * 1000 if self._start_time else 0.0

        # Compute peaks across all phase snapshots
        peak_gpu = 0.0
        peak_ram = 0.0
        for rec in self._records:
            peak_gpu = max(peak_gpu, rec.after.gpu_max_allocated_mb)
            peak_ram = max(peak_ram, rec.after.ram_used_mb)

        # Also check current state
        final_snap = _take_memory_snapshot()
        peak_gpu = max(peak_gpu, final_snap.gpu_max_allocated_mb)
        peak_ram = max(peak_ram, final_snap.ram_used_mb)

        # Tokens/sec from generation time
        gen_seconds = self._generation_time_ms / 1000 if self._generation_time_ms > 0 else 0.0
        tps = self._tokens_generated / gen_seconds if gen_seconds > 0 else 0.0

        return ProfilingReport(
            run_id=self.run_id,
            phases=list(self._records),
            total_wall_time_ms=total_wall,
            peak_gpu_mb=peak_gpu,
            peak_ram_used_mb=peak_ram,
            total_bytes_fetched=self._bytes_fetched,
            total_tokens_generated=self._tokens_generated,
            tokens_per_second=tps,
            compression_info=self._compression_info,
            extra=self._extra,
        )

    def log_report(self) -> ProfilingReport:
        """Build report and log the human-readable summary."""
        rep = self.report()
        for line in rep.summary_lines():
            logger.info(line)
        return rep
