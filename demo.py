"""Demo script — shows what the project actually does.

Run with: python demo.py
"""

import torch
from src.compression import create_compressor
from src.eval.niah import generate_niah_sample
from src.eval.profiler import RunProfiler
from src.eval.run_eval import EvalHarness
from src.experiments.sweep_config import SweepConfig
from src.experiments.run_scale_sweep import ScaleSweep
from src.experiments.sweep_plots import generate_sweep_plots
from tests.test_models import DummyModel

print("=" * 60)
print("PART 1: Compression quality comparison")
print("=" * 60)

t = torch.randn(4, 32, 64)
for method in ["fp16", "int8", "int4", "turboquant_mse"]:
    comp = create_compressor(method)
    errors = comp.compute_reconstruction_error(t)
    print(f"  {comp.name:40s}  cosine={errors['cosine_sim']:.4f}  snr={errors['snr_db']:.1f}dB  bpv={comp.estimate_bits_per_value():.2f}")

print()
print("=" * 60)
print("PART 2: Generate a needle-in-a-haystack sample")
print("=" * 60)

sample = generate_niah_sample(num_blocks=10, block_chars=200, seed=42)
print(f"  Total blocks: {len(sample.blocks)}")
print(f"  Needle hidden in block(s): {sample.needle_block_indices}")
print(f"  Question: {sample.needles[0].question}")
print(f"  Answer: {sample.needles[0].answer}")
print(f"  Haystack block preview: {sample.blocks[0][:100]}...")
print(f"  Needle block preview:   {sample.blocks[sample.needle_block_indices[0]][:100]}...")

print()
print("=" * 60)
print("PART 3: Run eval pipeline (sparse + compression)")
print("=" * 60)

model = DummyModel(hidden_size=64, num_layers=2, num_heads=4)
model.load()
model.decode = lambda ids: ["dummy answer"] * ids.shape[0]

profiler = RunProfiler(run_id="demo", enabled=True)
harness = EvalHarness(
    model=model,
    mode="sparse_plus_compression",
    router_engine="torch_cosine",
    top_k=3,
    compressor=create_compressor("int4"),
    profiler=profiler,
)
result = harness.evaluate([sample], run_id="demo")

print(f"  Accuracy: {result.accuracy}")
print(f"  Recall@k: {result.retrieval_metrics.get('recall_at_k', 0):.2f}")
print(f"  Compression ratio: {result.sample_results[0].compression_ratio:.1f}x")
print(f"  Bytes fetched: {result.sample_results[0].bytes_fetched:,}")

report = profiler.report()
print()
for line in report.summary_lines():
    print(f"  {line}")

print()
print("=" * 60)
print("PART 4: Scale sweep (saves CSV + plots to results/demo_sweep/)")
print("=" * 60)

config = SweepConfig(
    modes=["sparse", "sparse_plus_compression"],
    bank_sizes=[5, 10],
    top_k_values=[2, 3],
    compression_methods=["none", "int4"],
    num_trials=1,
    router_engine="torch_cosine",
)

sweep = ScaleSweep(model=model, config=config)
sweep_result = sweep.run()
paths = sweep.save(sweep_result, "results/demo_sweep")
plot_paths = generate_sweep_plots(sweep_result.averaged_summary(), "results/demo_sweep/plots")

print(f"  Runs completed: {len(sweep_result.records)}")
print()
print("  Files saved:")
for name, path in paths.items():
    print(f"    {name}: {path}")
print()
print("  Plots saved:")
for p in plot_paths:
    print(f"    {p}")

print()
print("  Averaged results:")
for row in sweep_result.averaged_summary():
    print(f"    mode={row['mode']:30s} bank={row['bank_size']:3d}  top_k={row['top_k']}  acc={row['accuracy']:.2f}  recall={row['recall_at_k']:.2f}  time={row['wall_time_ms']:.0f}ms")

print()
print("Done! Check results/demo_sweep/ for CSV tables and PNG plots.")
