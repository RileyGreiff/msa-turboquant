"""Stretch experiment — how far can sparse attention + KV compression extend context?

Run with: python demo_stretch.py

Core research question: at large bank sizes (8K-16K blocks), can TQ-MSE with
higher top-k match FP16 at lower top-k? i.e., does compression let us inject
more context for the same VRAM budget?

Reports both accuracy (with normalization) and exact_match (clean generation
without tokenization artifacts like "550 6" for "5506").
"""

import gc
import time
from pathlib import Path

import torch

from src.compression import create_compressor
from src.experiments.sweep_config import SweepConfig
from src.experiments.run_scale_sweep import ScaleSweep
from src.experiments.sweep_plots import generate_sweep_plots
from src.memory.bank_cache import BankCache
from src.models.hf_model import HFModel
from src.utils.config import ModelConfig
from src.utils.io_utils import ensure_dir, save_csv, save_json

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
BANK_SIZES = [4000, 8000, 16000]
BLOCK_CHARS = 500
TOP_K_VALUES = [10, 50, 100, 200]
COMPRESSION_METHODS = ["none", "int8", "int4", "turboquant_mse", "kivi"]
TASK_TYPES = ["single_needle"]
TOKENS_PER_BLOCK = 125  # ~500 chars
SEED = 42
NUM_TRIALS = 5
OUTPUT_DIR = Path("results/stretch_sweep")
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
print()

# -----------------------------------------------------------------------
# Step 2: Pre-build and cache banks
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 2: PRE-BUILDING MEMORY BANKS (cached to disk)")
print("=" * 70)

cache = BankCache(CACHE_DIR, model, extraction_mode="direct")

for num_blocks in BANK_SIZES:
    for trial in range(NUM_TRIALS):
        seed = SEED + trial
        if cache.exists(num_blocks, BLOCK_CHARS, seed):
            print(f"  {num_blocks:5d} blocks (seed={seed}): CACHED")
        else:
            start = time.perf_counter()
            print(f"  {num_blocks:5d} blocks (seed={seed}): building (routing-only)...", end="", flush=True)
            # save_kv=False: only save routing vectors (~63 MB per 8K blocks vs ~100 GB KV).
            # KV injection modes re-encode from text, so stored KV is never used.
            cache.get_or_build(num_blocks, BLOCK_CHARS, seed, save_kv=False)
            elapsed = time.perf_counter() - start
            print(f" done in {elapsed:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

print()

# -----------------------------------------------------------------------
# Step 3: Run stretch sweep
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 3: RUNNING STRETCH SWEEP")
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
    num_trials=NUM_TRIALS,
    seed_base=SEED,
    router_engine="torch_cosine",
    max_new_tokens=32,
    exclude_dense_above=30,
)

grid_size = len(config.parameter_grid())
print(f"  Grid size: {grid_size} parameter combos")
print(f"  Bank sizes: {BANK_SIZES}")
print(f"  Top-k: {TOP_K_VALUES}")
print(f"  Compression: {COMPRESSION_METHODS}")
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


def _get(mode, comp, bank, topk):
    for r in averaged:
        if (r.get("mode") == mode
                and r.get("compression_method") == comp
                and r.get("bank_size") == bank
                and r.get("top_k") == topk):
            return r
    return None


# --- Table A: Accuracy by top-k and compression (oracle, single-needle) ---
print()
print("  A. ORACLE ACCURACY: Does compression hold up at high top-k?")
for bs in BANK_SIZES:
    print(f"\n  Bank size = {bs} ({bs * TOKENS_PER_BLOCK:,} tokens stored)")
    print(f"  {'top_k':>6s}  {'FP16':>6s}  {'INT8':>6s}  {'INT4':>6s}  {'TQ-4b':>6s}  {'KIVI':>6s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for k in TOP_K_VALUES:
        injected_tokens = k * TOKENS_PER_BLOCK
        row = f"  {k:6d}"
        for mode, comp in [
            ("oracle_kv_inject", "none"),
            ("oracle_kv_inject_compressed", "int8"),
            ("oracle_kv_inject_compressed", "int4"),
            ("oracle_kv_inject_compressed", "turboquant_mse"),
            ("oracle_kv_inject_compressed", "kivi"),
        ]:
            r = _get(mode, comp, bs, k)
            if r:
                row += f"  {r['accuracy']:6.0%}"
            else:
                row += f"  {'N/A':>6s}"
        print(f"{row}  ({injected_tokens:,} tokens injected)")

# --- Table B: Exact match (clean generation quality) ---
print()
print("  B. EXACT MATCH (no normalization needed — clean generation)")
for bs in BANK_SIZES:
    print(f"\n  Bank size = {bs}")
    print(f"  {'top_k':>6s}  {'FP16':>6s}  {'INT8':>6s}  {'INT4':>6s}  {'TQ-4b':>6s}  {'KIVI':>6s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for k in TOP_K_VALUES:
        row = f"  {k:6d}"
        for mode, comp in [
            ("oracle_kv_inject", "none"),
            ("oracle_kv_inject_compressed", "int8"),
            ("oracle_kv_inject_compressed", "int4"),
            ("oracle_kv_inject_compressed", "turboquant_mse"),
            ("oracle_kv_inject_compressed", "kivi"),
        ]:
            r = _get(mode, comp, bs, k)
            if r:
                row += f"  {r['exact_match']:6.0%}"
            else:
                row += f"  {'N/A':>6s}"
        print(row)

# --- Table C: Sparse retrieval — does the router find the needle at scale? ---
print()
print("  C. SPARSE RETRIEVAL: Recall at scale (kv_inject, no compression)")
print(f"  {'Bank':>6s}  {'k=10':>6s}  {'k=50':>6s}  {'k=100':>6s}  {'k=200':>6s}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
for bs in BANK_SIZES:
    row = f"  {bs:6d}"
    for k in TOP_K_VALUES:
        r = _get("kv_inject", "none", bs, k)
        if r:
            row += f"  {r['recall_at_k']:6.0%}"
        else:
            row += f"  {'N/A':>6s}"
    print(row)

# --- Table D: Sparse + compression accuracy ---
print()
print("  D. SPARSE + COMPRESSION: End-to-end accuracy (retrieval + compression)")
for bs in BANK_SIZES:
    print(f"\n  Bank size = {bs}")
    print(f"  {'top_k':>6s}  {'FP16':>6s}  {'INT8':>6s}  {'INT4':>6s}  {'TQ-4b':>6s}  {'KIVI':>6s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for k in TOP_K_VALUES:
        row = f"  {k:6d}"
        for mode, comp in [
            ("kv_inject", "none"),
            ("kv_inject_compressed", "int8"),
            ("kv_inject_compressed", "int4"),
            ("kv_inject_compressed", "turboquant_mse"),
            ("kv_inject_compressed", "kivi"),
        ]:
            r = _get(mode, comp, bs, k)
            if r:
                row += f"  {r['accuracy']:6.0%}"
            else:
                row += f"  {'N/A':>6s}"
        print(row)

# --- Table E: VRAM budget comparison ---
print()
print("  E. VRAM BUDGET: Effective context at same memory cost")
print("  (FP16 KV per token ~= 4.4 KB for Qwen2.5-3B)")
num_kv_heads = 2
head_dim = model.hidden_size // model.num_heads
kv_per_token = model.num_layers * 2 * num_kv_heads * head_dim * 2  # bytes
for comp_name, ratio_label in [("none", "1x"), ("int8", "2x"), ("int4", "~4x"), ("turboquant_mse", "~4x"), ("kivi", "~4x")]:
    comp = create_compressor(comp_name) if comp_name != "none" else None
    bpv = comp.estimate_bits_per_value() if comp else 16.0
    effective_ratio = 16.0 / bpv
    for k in [50, 100, 200]:
        kv_bytes = k * TOKENS_PER_BLOCK * kv_per_token / effective_ratio
        print(f"  {comp_name:20s}  top_k={k:3d}  -> {k * TOKENS_PER_BLOCK:6,} tokens, ~{kv_bytes / 1024**2:.0f} MB KV")

print()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
print()
print(f"  Results saved to: {OUTPUT_DIR}")
print(f"  Plots saved to: {plot_dir}")
print()
print("Done!")
