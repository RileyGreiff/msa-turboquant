"""Scale-up experiment — find max context length per compression method.

Run with: python demo_scale.py

This loads Qwen2.5-3B, pre-builds memory banks at increasing sizes,
then runs NIAH evaluation across compression methods to find each method's
practical context ceiling on your GPU.

Expected runtime: ~6 minutes (75s bank building + 3 min sweep + plotting).
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
COMPRESSION_METHODS = ["none", "int8", "int4", "turboquant_like"]
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

# Compute KV bytes per block
# Each block: num_layers * 2 (K+V) * num_kv_heads * tokens * head_dim * 2 bytes (fp16)
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

    # Free memory between builds
    gc.collect()
    torch.cuda.empty_cache()

print()
total_build = sum(bank_build_times.values())
print(f"  Total bank build time: {total_build:.1f}s")
print()

# -----------------------------------------------------------------------
# Step 3: Run progressive stress test via ScaleSweep
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 3: RUNNING SCALE SWEEP")
print("=" * 70)

config = SweepConfig(
    modes=[
        "sparse_text",
        "kv_inject",
        "kv_inject_compressed",
        "oracle_kv_inject",
        "oracle_kv_inject_compressed",
    ],
    bank_sizes=BANK_SIZES,
    block_chars=[BLOCK_CHARS],
    top_k_values=TOP_K_VALUES,
    compression_methods=COMPRESSION_METHODS,
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
print(f"  Top-k: {TOP_K_VALUES}")
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

# Save max context summary
averaged = sweep_result.averaged_summary()

# Find max context per compression method where accuracy >= 80%
accuracy_threshold = 0.8
max_context: dict[str, dict] = {}
for r in averaged:
    comp = r.get("compression_method", "none")
    label = comp if comp != "none" else "FP16"
    acc = r.get("accuracy", 0)
    bank_size = r["bank_size"]
    total_tokens = bank_size * TOKENS_PER_BLOCK
    total_chars = bank_size * BLOCK_CHARS

    if acc >= accuracy_threshold:
        if label not in max_context or total_tokens > max_context[label]["max_tokens"]:
            max_context[label] = {
                "method": label,
                "max_blocks": bank_size,
                "max_tokens": total_tokens,
                "max_chars": total_chars,
                "accuracy_at_max": acc,
            }

summary_path = OUTPUT_DIR / "max_context_summary.json"
save_json({
    "accuracy_threshold": accuracy_threshold,
    "tokens_per_block": TOKENS_PER_BLOCK,
    "block_chars": BLOCK_CHARS,
    "methods": max_context,
}, summary_path)
print(f"  max_context_summary: {summary_path}")
print()

# -----------------------------------------------------------------------
# Step 5: Generate plots
# -----------------------------------------------------------------------
print("=" * 70)
print("STEP 5: GENERATING PLOTS")
print("=" * 70)

plot_dir = OUTPUT_DIR / "plots"
plot_paths = generate_sweep_plots(averaged, plot_dir)
for p in plot_paths:
    print(f"  {p}")
print()

# -----------------------------------------------------------------------
# Step 6: Summary
# -----------------------------------------------------------------------
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Table: Sparse text retrieval baseline (top_k=10)
print()
print(f"  NIAH Accuracy - Sparse text retrieval baseline (top_k=10)")
print(f"  {'Bank Size':>10s}  {'Accuracy':>10s}  {'Recall@k':>10s}")
print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
for bs in BANK_SIZES:
    matching = [r for r in averaged
                if r["bank_size"] == bs and r.get("top_k") == 10
                and r.get("mode") == "sparse_text"]
    if matching:
        acc = matching[0].get("accuracy", 0)
        recall = matching[0].get("recall_at_k", 0)
        print(f"  {bs:10d}  {acc:10.1%}  {recall:10.1%}")

# Table: KV injection — sparse retrieval (top_k=10)
print()
print(f"  NIAH Accuracy - KV injection, sparse retrieval (top_k=10)")
print(f"  {'Bank Size':>10s}  {'FP16':>8s}  {'INT8':>8s}  {'INT4':>8s}  {'TurboQ':>8s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
for bs in BANK_SIZES:
    row = f"  {bs:10d}"
    for mode, comp in [("kv_inject", "none"), ("kv_inject_compressed", "int8"),
                        ("kv_inject_compressed", "int4"), ("kv_inject_compressed", "turboquant_like")]:
        matching = [r for r in averaged
                    if r["bank_size"] == bs and r.get("top_k") == 10
                    and r.get("compression_method") == comp and r.get("mode") == mode]
        if matching:
            row += f"  {matching[0].get('accuracy', 0):8.1%}"
        else:
            row += f"  {'N/A':>8s}"
    print(row)

# Table: KV injection — oracle retrieval (top_k=10, isolates compression)
print()
print(f"  NIAH Accuracy - KV injection, oracle retrieval (top_k=10)")
print(f"  {'Bank Size':>10s}  {'FP16':>8s}  {'INT8':>8s}  {'INT4':>8s}  {'TurboQ':>8s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
for bs in BANK_SIZES:
    row = f"  {bs:10d}"
    for mode, comp in [("oracle_kv_inject", "none"), ("oracle_kv_inject_compressed", "int8"),
                        ("oracle_kv_inject_compressed", "int4"), ("oracle_kv_inject_compressed", "turboquant_like")]:
        matching = [r for r in averaged
                    if r["bank_size"] == bs and r.get("top_k") == 10
                    and r.get("compression_method") == comp and r.get("mode") == mode]
        if matching:
            row += f"  {matching[0].get('accuracy', 0):8.1%}"
        else:
            row += f"  {'N/A':>8s}"
    print(row)

# Summary: average accuracy across all bank_sizes and top_k
print()
print(f"  Average accuracy across all bank_sizes and top_k")
from collections import defaultdict as _ddict
for mode_label, mode_filter in [("Sparse KV inject", "kv_inject"),
                                  ("Oracle KV inject", "oracle_kv_inject")]:
    print(f"  {mode_label}:")
    _kv_groups = _ddict(list)
    for r in averaged:
        m = r.get("mode", "")
        if m.startswith(mode_filter):
            label = "FP16" if m == mode_filter else r.get("compression_method", "")
            _kv_groups[label].append(r.get("accuracy", 0))
    for label in ["FP16", "int8", "int4", "turboquant_like"]:
        if label in _kv_groups:
            accs = _kv_groups[label]
            avg = sum(accs) / len(accs)
            print(f"    {label:<20s}  {avg:8.1%}  (n={len(accs)})")

print()
print(f"  Maximum viable context (accuracy >= {accuracy_threshold:.0%}):")
print(f"  {'Method':<20s}  {'Max Blocks':>12s}  {'Max Tokens':>12s}  {'Max Chars':>12s}  {'Accuracy':>10s}")
print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
for label in ["FP16", "int8", "int4", "turboquant_like"]:
    if label in max_context:
        mc = max_context[label]
        print(f"  {label:<20s}  {mc['max_blocks']:12,d}  {mc['max_tokens']:12,d}  {mc['max_chars']:12,d}  {mc['accuracy_at_max']:10.1%}")
    else:
        print(f"  {label:<20s}  {'(below threshold)':>12s}")

print()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
print()
print(f"  Results saved to: {OUTPUT_DIR}")
print(f"  Plots saved to: {plot_dir}")
print()
print("Done!")
