"""Scale sweep runner — executes experiments across parameter combinations.

Takes a SweepConfig, generates the parameter grid, runs EvalHarness for
each combination, and collects results into a SweepResult with summary
tables and optional plots.

Usage::

    from src.experiments.run_scale_sweep import ScaleSweep
    from src.experiments.sweep_config import SweepConfig

    config = SweepConfig(
        modes=["sparse", "sparse_plus_compression"],
        bank_sizes=[10, 50, 100],
        top_k_values=[3, 5],
        compression_methods=["none", "int4"],
    )
    sweep = ScaleSweep(model=model, config=config)
    result = sweep.run()
    sweep.save(result, output_dir="results/sweep_001")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.compression import create_compressor
from src.eval.niah import generate_niah_sample
from src.eval.profiler import RunProfiler
from src.eval.run_eval import EvalHarness
from src.experiments.sweep_config import SweepConfig, SweepResult, SweepRunRecord
from src.memory.bank_cache import BankCache
from src.models.base_model import BaseModel
from src.utils.io_utils import ensure_dir, save_csv, save_json, timestamp_str

logger = logging.getLogger("msa_turboquant.experiments.scale_sweep")


class ScaleSweep:
    """Runs a config-driven scale sweep experiment.

    Args:
        model: A loaded BaseModel for evaluation.
        config: SweepConfig defining the parameter space.
        enable_profiling: If True, each run is profiled.
    """

    def __init__(
        self,
        model: BaseModel,
        config: SweepConfig,
        enable_profiling: bool = False,
        bank_cache: BankCache | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._enable_profiling = enable_profiling
        self._bank_cache = bank_cache

    def run(self) -> SweepResult:
        """Execute the full sweep and return aggregated results."""
        grid = self._config.parameter_grid()
        total = len(grid) * self._config.num_trials
        logger.info(
            f"Starting scale sweep: {len(grid)} combos x "
            f"{self._config.num_trials} trials = {total} runs"
        )

        result = SweepResult(config=self._config)

        for combo_idx, params in enumerate(grid):
            for trial in range(self._config.num_trials):
                run_num = combo_idx * self._config.num_trials + trial + 1
                seed = self._config.seed_base + trial
                run_id = (
                    f"sweep_{params['mode']}_b{params['bank_size']}"
                    f"_bc{params['block_chars']}_k{params['top_k']}"
                    f"_{params['compression_method']}_t{trial}"
                )

                logger.info(
                    f"  Run {run_num}/{total}: {run_id}"
                )

                record = self._run_single(params, trial, seed, run_id)
                result.records.append(record)

        logger.info(f"Sweep complete: {len(result.records)} runs recorded")
        return result

    def _run_single(
        self,
        params: dict[str, Any],
        trial: int,
        seed: int,
        run_id: str,
    ) -> SweepRunRecord:
        """Execute a single sweep run."""
        mode = params["mode"]
        bank_size = params["bank_size"]
        block_chars = params["block_chars"]
        top_k = params["top_k"]
        comp_method = params["compression_method"]

        # Create compressor if needed
        compressor = None
        if comp_method != "none":
            compressor = create_compressor(comp_method)

        # Create profiler
        profiler = RunProfiler(run_id=run_id, enabled=self._enable_profiling)

        # Create harness
        harness = EvalHarness(
            model=self._model,
            mode=mode,
            router_engine=self._config.router_engine,
            top_k=top_k,
            max_new_tokens=self._config.max_new_tokens,
            max_context_chars=self._config.max_context_chars,
            compressor=compressor,
            profiler=profiler,
        )

        # Generate sample
        sample = generate_niah_sample(
            num_blocks=bank_size,
            block_chars=block_chars,
            seed=seed,
        )

        # Use cached bank if available (avoids rebuilding per run)
        prebuilt_bank = None
        if self._bank_cache is not None and mode != "dense":
            prebuilt_bank = self._bank_cache.get_or_build(
                num_blocks=bank_size,
                block_chars=block_chars,
                seed=seed,
            )

        # Run evaluation
        eval_result = harness.evaluate([sample], run_id=run_id, prebuilt_bank=prebuilt_bank)

        # Extract metrics
        retrieval = eval_result.retrieval_metrics
        systems = eval_result.systems_metrics

        # Compute avg bytes_fetched and compression_ratio from sample results
        avg_bytes = 0
        avg_cr = 0.0
        if eval_result.sample_results:
            avg_bytes = sum(s.bytes_fetched for s in eval_result.sample_results) // len(eval_result.sample_results)
            crs = [s.compression_ratio for s in eval_result.sample_results if s.compression_ratio > 0]
            avg_cr = sum(crs) / len(crs) if crs else 0.0

        # Get profiling data
        prof_report = profiler.report()

        return SweepRunRecord(
            params=params,
            trial=trial,
            run_id=run_id,
            accuracy=eval_result.accuracy,
            recall_at_k=retrieval.get("recall_at_k", 0.0),
            mrr=retrieval.get("mrr", 0.0),
            hit_rate=retrieval.get("hit_rate", 0.0),
            wall_time_ms=systems.get("wall_time_ms", prof_report.total_wall_time_ms),
            peak_gpu_mb=prof_report.peak_gpu_mb,
            peak_ram_mb=prof_report.peak_ram_used_mb,
            bytes_fetched=avg_bytes,
            compression_ratio=avg_cr,
            tokens_per_second=prof_report.tokens_per_second,
        )

    def save(
        self,
        result: SweepResult,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Save sweep results to disk.

        Saves:
        - All individual run records as CSV
        - Averaged summary as CSV
        - Full results as JSON
        - Sweep config as JSON

        Args:
            result: The SweepResult to save.
            output_dir: Directory for output files.

        Returns:
            Dict mapping file type to path.
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        paths: dict[str, Path] = {}

        # All runs CSV
        all_runs_path = output_dir / "sweep_all_runs.csv"
        save_csv(result.summary_table(), all_runs_path)
        paths["all_runs"] = all_runs_path

        # Averaged summary CSV
        avg_path = output_dir / "sweep_averaged.csv"
        save_csv(result.averaged_summary(), avg_path)
        paths["averaged"] = avg_path

        # Full JSON
        full_path = output_dir / "sweep_full.json"
        save_json({
            "config": self._config.model_dump(),
            "num_runs": len(result.records),
            "records": result.summary_table(),
            "averaged": result.averaged_summary(),
        }, full_path)
        paths["full"] = full_path

        # Config JSON
        config_path = output_dir / "sweep_config.json"
        save_json(self._config.model_dump(), config_path)
        paths["config"] = config_path

        logger.info(f"Sweep results saved to {output_dir}")
        return paths
