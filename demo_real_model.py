"""Demo with a real model — Qwen2.5-3B on GPU.

Run with: python demo_real_model.py

This loads a real model onto your RTX 5060 Ti and runs the full pipeline
with meaningful results. Expect ~6GB VRAM usage and ~30-60 seconds total.
"""

import torch
from src.compression import create_compressor
from src.eval.niah import generate_niah_sample
from src.eval.profiler import RunProfiler
from src.eval.run_eval import EvalHarness
from src.models.hf_model import HFModel
from src.utils.config import ModelConfig, TokenizerConfig

# -----------------------------------------------------------------------
# Step 1: Load the real model
# -----------------------------------------------------------------------
print("=" * 60)
print("LOADING MODEL (Qwen/Qwen2.5-3B, float16, CUDA)")
print("=" * 60)

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
print(f"  Device: {model.device}, Dtype: {model.dtype}")
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

# -----------------------------------------------------------------------
# Step 2: Quick sanity check — can it generate text?
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("SANITY CHECK: Generate text")
print("=" * 60)

test_prompt = "The capital of France is"
response = model.generate_text(test_prompt, max_new_tokens=20)
print(f"  Prompt: {test_prompt}")
print(f"  Response: {response[0]}")

# -----------------------------------------------------------------------
# Step 3: Compression quality on REAL hidden states
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("COMPRESSION QUALITY (on real model hidden states)")
print("=" * 60)

# Get real hidden states from the model
tokens = model.tokenize("The quick brown fox jumps over the lazy dog. " * 10)
output = model.forward(tokens.input_ids, tokens.attention_mask, output_hidden_states=True, use_cache=True)
real_tensor = output.hidden_states[-1].float().cpu()  # Last layer hidden states

print(f"  Tensor shape: {real_tensor.shape}")
print(f"  Tensor range: [{real_tensor.min():.3f}, {real_tensor.max():.3f}]")
print()

for method in ["fp16", "int8", "int4", "turboquant_mse"]:
    comp = create_compressor(method)
    errors = comp.compute_reconstruction_error(real_tensor)
    print(f"  {comp.name:45s}  cosine={errors['cosine_sim']:.4f}  snr={errors['snr_db']:.1f}dB  bpv={comp.estimate_bits_per_value():.2f}")

# Also test on real KV cache tensors
if output.kv_cache:
    kv_key = output.kv_cache[0][0].float().cpu()  # Layer 0 keys
    print()
    print(f"  KV key tensor shape: {kv_key.shape}")
    print(f"  KV key range: [{kv_key.min():.3f}, {kv_key.max():.3f}]")
    print()
    for method in ["fp16", "int8", "int4", "turboquant_mse"]:
        comp = create_compressor(method)
        errors = comp.compute_reconstruction_error(kv_key)
        print(f"  {comp.name:45s}  cosine={errors['cosine_sim']:.4f}  snr={errors['snr_db']:.1f}dB")

# -----------------------------------------------------------------------
# Step 4: NIAH evaluation — can the model find the needle?
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("NIAH EVALUATION: Dense mode (all context visible)")
print("=" * 60)

sample = generate_niah_sample(num_blocks=8, block_chars=300, seed=42)
print(f"  Blocks: {len(sample.blocks)}")
print(f"  Needle in block: {sample.needle_block_indices}")
print(f"  Question: {sample.needles[0].question}")
print(f"  Expected answer: {sample.needles[0].answer}")

profiler = RunProfiler(run_id="dense_real", enabled=True)
harness_dense = EvalHarness(
    model=model,
    mode="dense",
    max_new_tokens=32,
    profiler=profiler,
)
result_dense = harness_dense.evaluate([sample], run_id="dense_real")

print(f"  Model answer: {result_dense.sample_results[0].model_answer[:200]}")
print(f"  Correct: {result_dense.sample_results[0].correct}")
print(f"  Accuracy: {result_dense.accuracy:.0%}")

report = profiler.report()
print(f"  Wall time: {report.total_wall_time_ms:.0f} ms")
print(f"  Peak GPU: {report.peak_gpu_mb:.0f} MB")

# -----------------------------------------------------------------------
# Step 5: Sparse retrieval — can the router find the needle block?
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("NIAH EVALUATION: Sparse retrieval (top-3 from 8 blocks)")
print("=" * 60)

profiler_sparse = RunProfiler(run_id="sparse_real", enabled=True)
harness_sparse = EvalHarness(
    model=model,
    mode="sparse",
    router_engine="torch_cosine",
    top_k=3,
    max_new_tokens=32,
    profiler=profiler_sparse,
)
result_sparse = harness_sparse.evaluate([sample], run_id="sparse_real")

sr = result_sparse.sample_results[0]
print(f"  Model answer: {sr.model_answer[:200]}")
print(f"  Correct: {sr.correct}")
print(f"  Recall@3: {result_sparse.retrieval_metrics.get('recall_at_k', 0):.2f}")
print(f"  MRR: {result_sparse.retrieval_metrics.get('mrr', 0):.2f}")
print(f"  Bytes fetched: {sr.bytes_fetched:,}")

report_s = profiler_sparse.report()
print()
for line in report_s.summary_lines():
    print(f"  {line}")

# -----------------------------------------------------------------------
# Step 6: Sparse + Compression — the realistic scenario
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("NIAH EVALUATION: Sparse + INT4 compression")
print("=" * 60)

profiler_spc = RunProfiler(run_id="spc_real", enabled=True)
harness_spc = EvalHarness(
    model=model,
    mode="sparse_plus_compression",
    router_engine="torch_cosine",
    top_k=3,
    max_new_tokens=32,
    compressor=create_compressor("int4"),
    profiler=profiler_spc,
)
result_spc = harness_spc.evaluate([sample], run_id="spc_real")

sr_spc = result_spc.sample_results[0]
print(f"  Model answer: {sr_spc.model_answer[:200]}")
print(f"  Correct: {sr_spc.correct}")
print(f"  Compression ratio: {sr_spc.compression_ratio:.1f}x")
print(f"  Recall@3: {result_spc.retrieval_metrics.get('recall_at_k', 0):.2f}")

# -----------------------------------------------------------------------
# Step 7: Sparse + TurboQuant — does rotation help?
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("NIAH EVALUATION: Sparse + TurboQuant-like compression")
print("=" * 60)

profiler_tq = RunProfiler(run_id="tq_real", enabled=True)
harness_tq = EvalHarness(
    model=model,
    mode="sparse_plus_compression",
    router_engine="torch_cosine",
    top_k=3,
    max_new_tokens=32,
    compressor=create_compressor("turboquant_mse", bits=4, group_size=128, rotation="hadamard"),
    profiler=profiler_tq,
)
result_tq = harness_tq.evaluate([sample], run_id="tq_real")

sr_tq = result_tq.sample_results[0]
print(f"  Model answer: {sr_tq.model_answer[:200]}")
print(f"  Correct: {sr_tq.correct}")
print(f"  Compression ratio: {sr_tq.compression_ratio:.1f}x")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Mode':<35s} {'Correct':<10s} {'Answer'}")
print(f"  {'-'*35} {'-'*10} {'-'*40}")
print(f"  {'Dense (all context)':<35s} {str(result_dense.sample_results[0].correct):<10s} {result_dense.sample_results[0].model_answer[:40]}")
print(f"  {'Sparse (top-3)':<35s} {str(sr.correct):<10s} {sr.model_answer[:40]}")
print(f"  {'Sparse + INT4':<35s} {str(sr_spc.correct):<10s} {sr_spc.model_answer[:40]}")
print(f"  {'Sparse + TurboQuant (hadamard)':<35s} {str(sr_tq.correct):<10s} {sr_tq.model_answer[:40]}")

print()
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
print()
print("Done!")
