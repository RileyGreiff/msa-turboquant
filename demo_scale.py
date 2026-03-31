"""Scale-up experiment — bit-budget + multi-needle stress test for TurboQuant.

Run with: python demo_scale.py

Tests two things:
1. Bit-budget sweep: TQ-MSE at 4b/3b/2b vs INT4/INT8 on single-needle NIAH
2. Multi-needle with distractors: harder task that stresses dot-product geometry

Expected runtime: ~15 min with cached banks.
"""

import gc
import time
from pathlib import Path

import torch

from src.compression import create_compressor
from src.eval.niah import generate_niah_sample
from src.eval.profiler import RunProfiler
from src.eval.run_eval import EvalHarness
from src.experiments.sweep_config import SweepConfig
from src.experiments.run_scale_sweep import ScaleSweep
from src.experiments.sweep_plots import (
    generate_sweep_plots,
    plot_accuracy_vs_context_size,
    plot_max_context_comparison,
)
from src.memory.bank_cache import BankCache
from src.models.hf_model import HFModel
from src.utils.config import ModelConfig, TokenizerConfig
from src.utils.io_utils import ensure_dir, save_csv, save_json

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
BANK_SIZES = [500, 1000, 2000, 4000]
BLOCK_CHARS = 500
TOP_K_VALUES = [10, 20, 50]
COMPRESSION_METHODS = [
    "none", "int8", "int4",
    "turboquant_mse",       # 4-bit TQ-MSE
    "turboquant_mse_3b",    # 3-bit TQ-MSE (INT4 can't go here)
    "turboquant_mse_2b",    # 2-bit TQ-MSE (aggressive compression)
]
TASK_TYPES = ["single_needle", "multi_needle_distractor"]
TOKENS_PER_BLOCK = 125  # Approximate for 500 chars
SEED = 42
OUTPUT_DIR = Path("results/scale_sweep")
CACHE_DIR = Path("data/bank_cache")

# -----------------------------------------------------------------------
# Step 1: Load model
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 1: LOADING MODEL (Qwen/Qwen2.5-3B, float16, CUDA)")
print("=" * 70)

model_config = ModelConfig(
    name="Qwen/Qwen2.5-3B",
    dtype="float16",
    device="cuda:0",
    max_seq_len=4096,
    trust_remote_code=True,
    attn_implementation="sdpa",
)

model = HFModel(model_config)
model.load()

print(f"  Model: {model.model_name}")
print(f"  Layers: {model.num_layers}, Hidden: {model.hidden_size}, Heads: {model.num_heads}")
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

num_kv_heads = 2  # Qwen2.5-3B uses GQA with 2 KV heads
head_dim = model.hidden_size // model.num_heads
estimated_kv_per_block = model.num_layers * 2 * num_kv_heads * TOKENS_PER_BLOCK * head_dim * 2
print(f"  Estimated KV per block: {estimated_kv_per_block / 1024**2:.1f} MB (FP16)")
print()

# -----------------------------------------------------------------------
# Step 2: Pre-build and cache banks
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 2: PRE-BUILDING MEMORY BANKS (cached to disk)")
print("=" * 70)

cache = BankCache(CACHE_DIR, model, extraction_mode="direct")
bank_build_times = {}

for num_blocks in BANK_SIZES:
    if cache.exists(num_blocks, BLOCK_CHARS, SEED):
        print(f"  {num_blocks:4d} blocks: CACHED (skipping build)")
        bank_build_times[num_blocks] = 0.0
    else:
        start = time.perf_counter()
        print(f"  {num_blocks:4d} blocks: building...", end="", flush=True)
        cache.get_or_build(num_blocks, BLOCK_CHARS, SEED)
        elapsed = time.perf_counter() - start
        bank_build_times[num_blocks] = elapsed
        print(f" done in {elapsed:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

print()
total_build = sum(bank_build_times.values())
print(f"  Total bank build time: {total_build:.1f}s")
print()

# -----------------------------------------------------------------------
# Step 3: Run sweep
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 3: RUNNING SCALE SWEEP")
print("=" * 70)

config = SweepConfig(
    modes=[
        "kv_inject",
        "kv_inject_compressed",
        "oracle_kv_inject",
        "oracle_kv_inject_compressed",
    ],
    bank_sizes=BANK_SIZES,
    block_chars=[BLOCK_CHARS],
    top_k_values=TOP_K_VALUES,
    compression_methods=COMPRESSION_METHODS,
    task_types=TASK_TYPES,
    num_trials=5,
    seed_base=SEED,
    router_engine="torch_cosine",
    max_new_tokens=32,
    exclude_dense_above=30,
)

grid_size = len(config.parameter_grid())
print(f"  Grid size: {grid_size} parameter combos")
print(f"  Bank sizes: {BANK_SIZES}")
print(f"  Compression: {COMPRESSION_METHODS}")
print(f"  Task types: {TASK_TYPES}")
print(f"  Top-k: {TOP_K_VALUES}")
print(f"  Trials: {config.num_trials}")
print(f"  Total runs: {config.total_runs()}")
print()

sweep = ScaleSweep(model=model, config=config, enable_profiling=True, bank_cache=cache)
sweep_start = time.perf_counter()
sweep_result = sweep.run()
sweep_elapsed = time.perf_counter() - sweep_start

print()
print(f"  Sweep completed: {len(sweep_result.records)} runs in {sweep_elapsed:.1f}s")
print()

# -----------------------------------------------------------------------
# Step 4: Save results
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 4: SAVING RESULTS")
print("=" * 70)

ensure_dir(OUTPUT_DIR)
saved = sweep.save(sweep_result, OUTPUT_DIR)
for name, path in saved.items():
    print(f"  {name}: {path}")
print()

# -----------------------------------------------------------------------
# Step 5: Generate plots
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 5: GENERATING PLOTS")
print("=" * 70)

averaged = sweep_result.averaged_summary()
plot_dir = OUTPUT_DIR / "plots"
plot_paths = generate_sweep_plots(averaged, plot_dir)
for p in plot_paths:
    print(f"  {p}")
print()

# -----------------------------------------------------------------------
# Step 6: Summary tables
# -----------------------------------------------------------------------
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

from collections import defaultdict as _ddict

# Helper to filter averaged results
def _get(task, mode, comp, bank, topk=10):
    for r in averaged:
        if (r.get("task_type") == task and r.get("mode") == mode
                and r.get("compression_method") == comp
                and r.get("bank_size") == bank and r.get("top_k") == topk):
            return r
    return None

# --- Table A: Bit-budget comparison (single-needle, oracle, top_k=10) ---
print()
print("  A. BIT-BUDGET: Oracle KV inject accuracy (single_needle, top_k=10)")
print(f"  {'Bank':>6s}  {'FP16':>6s}  {'INT8':>6s}  {'INT4':>6s}  {'TQ-4b':>6s}  {'TQ-3b':>6s}  {'TQ-2b':>6s}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
for bs in BANK_SIZES:
    row = f"  {bs:6d}"
    for mode, comp in [
        ("oracle_kv_inject", "none"),
        ("oracle_kv_inject_compressed", "int8"),
        ("oracle_kv_inject_compressed", "int4"),
        ("oracle_kv_inject_compressed", "turboquant_mse"),
        ("oracle_kv_inject_compressed", "turboquant_mse_3b"),
        ("oracle_kv_inject_compressed", "turboquant_mse_2b"),
    ]:
        r = _get("single_needle", mode, comp, bs)
        if r:
            row += f"  {r['accuracy']:6.0%}"
        else:
            row += f"  {'N/A':>6s}"
    print(row)

# --- Table B: Multi-needle distractor (oracle, top_k=10) ---
print()
print("  B. MULTI-NEEDLE DISTRACTOR: Oracle KV inject (top_k=10)")
print(f"  {'Bank':>6s}  {'FP16':>6s}  {'INT8':>6s}  {'INT4':>6s}  {'TQ-4b':>6s}  {'TQ-3b':>6s}  {'TQ-2b':>6s}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
for bs in BANK_SIZES:
    row = f"  {bs:6d}"
    for mode, comp in [
        ("oracle_kv_inject", "none"),
        ("oracle_kv_inject_compressed", "int8"),
        ("oracle_kv_inject_compressed", "int4"),
        ("oracle_kv_inject_compressed", "turboquant_mse"),
        ("oracle_kv_inject_compressed", "turboquant_mse_3b"),
        ("oracle_kv_inject_compressed", "turboquant_mse_2b"),
    ]:
        r = _get("multi_needle_distractor", mode, comp, bs)
        if r:
            row += f"  {r['needle_accuracy']:6.0%}"
        else:
            row += f"  {'N/A':>6s}"
    print(row)

# --- Table C: Distractor confusions ---
print()
print("  C. DISTRACTOR CONFUSIONS (oracle, avg across bank sizes, top_k=10)")
confusion_groups = _ddict(list)
for r in averaged:
    if (r.get("task_type") == "multi_needle_distractor"
            and r.get("mode") in ("oracle_kv_inject", "oracle_kv_inject_compressed")
            and r.get("top_k") == 10):
        label = "FP16" if r["mode"] == "oracle_kv_inject" else r["compression_method"]
        confusion_groups[label].append(r.get("distractor_confusions", 0))

print(f"  {'Method':<25s}  {'Avg Confusions':>15s}")
print(f"  {'-'*25}  {'-'*15}")
for label in ["FP16", "int8", "int4", "turboquant_mse", "turboquant_mse_3b", "turboquant_mse_2b"]:
    if label in confusion_groups:
        vals = confusion_groups[label]
        avg = sum(vals) / len(vals)
        print(f"  {label:<25s}  {avg:15.2f}")

# --- Table D: Compression ratios ---
print()
print("  D. COMPRESSION RATIOS (avg across all runs)")
ratio_groups = _ddict(list)
for r in averaged:
    if r.get("mode") == "oracle_kv_inject_compressed" and r.get("task_type") == "single_needle":
        ratio_groups[r["compression_method"]].append(r.get("compression_ratio", 0))

print(f"  {'Method':<25s}  {'Avg Ratio':>10s}  {'Effective bpv':>15s}")
print(f"  {'-'*25}  {'-'*10}  {'-'*15}")
for label in ["int8", "int4", "turboquant_mse", "turboquant_mse_3b", "turboquant_mse_2b"]:
    if label in ratio_groups:
        vals = ratio_groups[label]
        avg = sum(vals) / len(vals)
        comp = create_compressor(label)
        bpv = comp.estimate_bits_per_value()
        print(f"  {label:<25s}  {avg:10.2f}x  {bpv:15.2f}")

# --- Table E: Average accuracy by method (single_needle, all conditions) ---
print()
print("  E. AVERAGE ACCURACY across all bank_sizes and top_k")
for task in TASK_TYPES:
    metric = "accuracy" if task == "single_needle" else "needle_accuracy"
    print(f"  Task: {task} (metric: {metric})")
    acc_groups = _ddict(list)
    for r in averaged:
        if r.get("task_type") != task:
            continue
        m = r.get("mode", "")
        if m in ("oracle_kv_inject", "oracle_kv_inject_compressed"):
            label = "FP16" if m == "oracle_kv_inject" else r.get("compression_method", "")
            acc_groups[label].append(r.get(metric, 0))
    for label in ["FP16", "int8", "int4", "turboquant_mse", "turboquant_mse_3b", "turboquant_mse_2b"]:
        if label in acc_groups:
            vals = acc_groups[label]
            avg = sum(vals) / len(vals)
            print(f"    {label:<25s}  {avg:8.1%}  (n={len(vals)})")
    print()

print()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
print()
print(f"  Results saved to: {OUTPUT_DIR}")
print(f"  Plots saved to: {plot_dir}")
print()
print("Done!")
