"""Configuration for scale sweep experiments.

A SweepConfig defines the axes to sweep over and produces the full
cross-product of parameter combinations.  Each combination becomes a
single evaluation run executed by the ScaleSweep runner.

Example YAML::

    sweep:
      modes: ["sparse", "sparse_plus_compression"]
      bank_sizes: [10, 50, 100]
      block_chars: [200, 500]
      top_k_values: [3, 5, 10]
      compression_methods: ["none", "int8", "int4", "turboquant_like"]
      num_trials: 3
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class SweepConfig(BaseModel):
    """Defines the parameter space for a scale sweep experiment.

    Each list attribute is one sweep axis.  The runner generates the full
    cross-product of all axes.

    Attributes:
        modes: Eval modes to test.
        bank_sizes: Number of blocks in the memory bank.
        block_chars: Character count per block (controls block size).
        top_k_values: Number of blocks to retrieve.
        compression_methods: Compression method names ("none", "fp16",
            "int8", "int4", "turboquant_like").
        num_trials: Number of repeated trials per combination (averaged).
        seed_base: Base random seed; each trial offsets from this.
        router_engine: Router backend for sparse modes.
        max_new_tokens: Max tokens to generate per sample.
        max_context_chars: Max context chars for assembly.
    """

    model_config = ConfigDict(extra="forbid")

    modes: list[str] = ["dense", "sparse", "sparse_plus_compression"]
    bank_sizes: list[int] = [10, 50]
    block_chars: list[int] = [200]
    top_k_values: list[int] = [3, 5]
    compression_methods: list[str] = ["none", "int4"]
    num_trials: int = 1
    seed_base: int = 42
    router_engine: str = "torch_cosine"
    max_new_tokens: int = 64
    max_context_chars: int | None = None
    exclude_dense_above: int = 30

    @field_validator("bank_sizes", "block_chars", "top_k_values")
    @classmethod
    def _positive_lists(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            raise ValueError("All values must be positive")
        return v

    @field_validator("num_trials")
    @classmethod
    def _positive_trials(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_trials must be >= 1")
        return v

    def parameter_grid(self) -> list[dict[str, Any]]:
        """Generate the full cross-product of sweep parameters.

        Returns:
            List of dicts, each representing one parameter combination.
            Keys: mode, bank_size, block_chars, top_k, compression_method.
        """
        combos = list(itertools.product(
            self.modes,
            self.bank_sizes,
            self.block_chars,
            self.top_k_values,
            self.compression_methods,
        ))

        grid = []
        for mode, bank_size, bchars, top_k, comp in combos:
            # Skip combos that don't make sense
            # Dense mode infeasible above a certain bank size (exceeds max_seq_len)
            if mode == "dense" and bank_size > self.exclude_dense_above:
                continue
            # Dense mode doesn't use top_k or compression
            if mode == "dense" and (top_k != self.top_k_values[0] or comp != self.compression_methods[0]):
                continue
            # Non-compression modes should only run with "none" compression
            _no_comp_modes = ("dense", "sparse", "sparse_text", "kv_inject", "oracle_kv_inject")
            if mode in _no_comp_modes and comp != "none":
                continue
            # Compression modes need an actual compressor
            _comp_modes = ("kv_inject_compressed", "oracle_kv_inject_compressed",
                           "compression_only", "sparse_plus_compression", "oracle_plus_compression")
            if mode in _comp_modes and comp == "none":
                continue

            grid.append({
                "mode": mode,
                "bank_size": bank_size,
                "block_chars": bchars,
                "top_k": top_k,
                "compression_method": comp,
            })

        return grid

    def total_runs(self) -> int:
        """Total number of runs (grid size * num_trials)."""
        return len(self.parameter_grid()) * self.num_trials


@dataclass
class SweepRunRecord:
    """Results from a single sweep run (one parameter combo, one trial).

    Attributes:
        params: The parameter combination for this run.
        trial: Trial index (0-based).
        run_id: EvalRunResult.run_id.
        accuracy: Answer accuracy.
        recall_at_k: Retrieval recall.
        mrr: Mean reciprocal rank.
        hit_rate: Fraction of samples with at least one gold block retrieved.
        wall_time_ms: Total wall time.
        peak_gpu_mb: Peak GPU allocation.
        peak_ram_mb: Peak RAM usage.
        bytes_fetched: Total KV bytes fetched.
        compression_ratio: Average compression ratio.
        tokens_per_second: Generation throughput.
    """
    params: dict[str, Any] = field(default_factory=dict)
    trial: int = 0
    run_id: str = ""
    accuracy: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    wall_time_ms: float = 0.0
    peak_gpu_mb: float = 0.0
    peak_ram_mb: float = 0.0
    bytes_fetched: int = 0
    compression_ratio: float = 0.0
    tokens_per_second: float = 0.0

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten params + metrics into a single dict for CSV export."""
        d = dict(self.params)
        d.update({
            "trial": self.trial,
            "run_id": self.run_id,
            "accuracy": round(self.accuracy, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "wall_time_ms": round(self.wall_time_ms, 2),
            "peak_gpu_mb": round(self.peak_gpu_mb, 1),
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "bytes_fetched": self.bytes_fetched,
            "compression_ratio": round(self.compression_ratio, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
        })
        return d


@dataclass
class SweepResult:
    """Aggregated results from a full scale sweep.

    Attributes:
        config: The SweepConfig used.
        records: All individual run records.
    """
    config: SweepConfig
    records: list[SweepRunRecord] = field(default_factory=list)

    def summary_table(self) -> list[dict[str, Any]]:
        """Return records as a list of flat dicts (one per run)."""
        return [r.to_flat_dict() for r in self.records]

    def averaged_summary(self) -> list[dict[str, Any]]:
        """Average metrics across trials for each parameter combo.

        Returns:
            List of dicts keyed by param combo with averaged metric values.
        """
        from collections import defaultdict

        groups: dict[str, list[SweepRunRecord]] = defaultdict(list)
        for rec in self.records:
            key = str(sorted(rec.params.items()))
            groups[key].append(rec)

        averaged = []
        metric_keys = [
            "accuracy", "recall_at_k", "mrr", "hit_rate",
            "wall_time_ms", "peak_gpu_mb", "peak_ram_mb",
            "bytes_fetched", "compression_ratio", "tokens_per_second",
        ]
        for group_recs in groups.values():
            n = len(group_recs)
            row = dict(group_recs[0].params)
            row["num_trials"] = n
            for key in metric_keys:
                vals = [getattr(r, key) for r in group_recs]
                row[key] = round(sum(vals) / n, 4) if n > 0 else 0.0
            averaged.append(row)

        return averaged
