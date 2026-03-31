"""MSA TurboQuant Local — Main entrypoint.

Usage:
    python -m src.main --dry-run
    python -m src.main --config-dir configs --override model.max_seq_len=4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MSA TurboQuant Local — extreme-context memory benchmark harness",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML config files (default: configs/)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key.subkey=value format",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config, log system info, and exit without running experiments",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    """Main entry point: load config, setup logging, dispatch experiments."""
    from src.utils.config import load_config
    from src.utils.logging_utils import log_config, log_gpu_memory, setup_logging
    from src.utils.profiling import log_system_info

    args = parse_args()

    # 1. Load configuration
    try:
        config = load_config(args.config_dir, overrides=args.override)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        return 1

    # 2. Setup logging
    logger = setup_logging(config.logging, experiment_name=config.name)

    # 3. Log system info and config
    log_system_info(logger)
    log_config(logger, config)
    log_gpu_memory(logger)

    # 4. Set seed
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")

    # 5. Resolve paths
    resolved_paths = config.paths.resolve(Path.cwd())
    logger.info(f"Results dir: {resolved_paths['results_dir']}")

    # 6. Dry run check
    if args.dry_run:
        logger.info("Dry run complete. Config loaded and validated successfully.")
        return 0

    # 7. Experiment dispatch
    logger.info(f"Experiment mode: {config.mode}")

    if config.mode == "sweep":
        from src.experiments.run_scale_sweep import ScaleSweep
        from src.experiments.sweep_config import SweepConfig
        from src.models.hf_model import HFModel

        model = HFModel(config.model)
        model.load()

        sweep = ScaleSweep(
            model=model,
            config=SweepConfig(),
            enable_profiling=True,
        )
        result = sweep.run()
        sweep.save(result, resolved_paths["results_dir"])
        logger.info(
            f"Sweep complete: {len(result.records)} runs "
            f"saved to {resolved_paths['results_dir']}"
        )
    else:
        logger.info(
            f"Mode '{config.mode}' is not a runnable experiment. "
            "Set mode='sweep' for a default scale sweep, "
            "or use demo_scale.py for custom configurations."
        )

    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required on Windows
    sys.exit(main())
