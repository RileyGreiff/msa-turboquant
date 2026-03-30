# MSA TurboQuant Local

A local research harness for benchmarking sparse external memory retrieval and KV cache compression on extreme-length context tasks. Designed for rigorous experimentation on a single desktop GPU.

---

## What This Project Is (and Is Not)

**This IS** a research evaluation harness that:

- Maximizes **stored** memory size (thousands of text blocks in an external bank)
- Keeps the **active attended context** small (fits in GPU VRAM)
- Measures quality and performance degradation as stored memory grows
- Compares retrieval and compression strategies via direct KV cache injection
- Provides config-driven scale sweeps with automatic plotting

**This is NOT:**

- A claim of true dense attention over billions of tokens
- A production-ready inference system
- An exact reproduction of any published paper (TurboQuant-inspired compression is clearly labeled as *inspired by*, not a reproduction)

The core research question:

> *As stored memory grows, how do sparse retrieval and KV compression affect retrieval recall, answer quality, and systems performance?*

---

## System Design

### Architecture

```
                        +------------------+
                        |   Config (YAML)  |
                        +--------+---------+
                                 |
                        +--------v---------+
                        |  Main CLI / Sweep |
                        |    Runner         |
                        +--------+---------+
                                 |
            +--------------------+--------------------+
            |                    |                    |
   +--------v--------+  +-------v--------+  +--------v--------+
   |  Model Wrapper   |  |  Memory Bank   |  |  Compression    |
   |  (HF backend,    |  |  Builder       |  |  (fp16, int8,   |
   |   KV extraction,  |  |  (chunking,    |  |   int4, TQ-like)|
   |   hidden states)  |  |   routing vecs,|  +--------+--------+
   +--------+---------+  |   KV storage)  |           |
            |             +-------+--------+           |
            |                     |                    |
            +----------+----------+--------------------+
                       |
              +--------v---------+
              |  Retrieval        |
              |  (FAISS / cosine, |
              |   oracle router)  |
              +--------+---------+
                       |
              +--------v---------+
              |  KV Injection     |
              |  (encode context, |
              |   chunked prefill,|
              |   compress/inject)|
              +--------+---------+
                       |
              +--------v---------+
              |  Eval Harness     |
              |  (NIAH, scoring,  |
              |   profiler)       |
              +--------+---------+
                       |
              +--------v---------+
              |  Results          |
              |  (JSON, CSV,      |
              |   plots, profiles)|
              +-------------------+
```

### Data Flow

1. **Offline bank building**: Text is chunked into blocks. Each block is passed through the model to extract KV cache tensors and a routing vector (mean-pooled hidden state). These are stored in a memory bank on disk via numpy memmap. Banks are cached and reused across experiments.

2. **At query time**: The query is encoded into a routing vector, which is compared against all bank routing vectors via FAISS IndexFlatIP or torch cosine similarity. The top-k block indices are returned.

3. **KV injection**: The retrieved blocks' text is concatenated and re-encoded via a contiguous forward pass (with chunked prefill for long contexts), producing a KV cache with correct RoPE positions. This KV is optionally compressed and decompressed (roundtrip degradation), then injected as `past_key_values` into the model's generate call.

4. **Evaluation**: The model generates an answer conditioned on the injected KV. Correctness is checked via substring match against the NIAH ground truth. Retrieval quality, systems metrics, and profiling data are recorded per sample.

---

## Evaluation Modes

The harness supports several modes that isolate different variables. The primary modes for research are the KV injection variants:

| Mode | Retrieval | KV Injection | Compression | Purpose |
|------|-----------|-------------|-------------|---------|
| **sparse_text** | Top-k (sparse) | No | No | Baseline: text-only generation from retrieved blocks |
| **kv_inject** | Top-k (sparse) | Yes | No (FP16) | Does KV injection match text-based generation? |
| **kv_inject_compressed** | Top-k (sparse) | Yes | Yes | The core research result: compression degradation at scale |
| **oracle_kv_inject** | Perfect (gold) | Yes | No (FP16) | Upper bound: perfect retrieval + FP16 KV injection |
| **oracle_kv_inject_compressed** | Perfect (gold) | Yes | Yes | Isolates compression quality (no retrieval noise) |
| **dense** | None (full context) | No | No | Dense attention baseline (infeasible at large bank sizes) |

**Why KV injection matters:** Earlier text-based compression modes (`sparse_plus_compression`, `compression_only`) had a critical flaw: KV was compressed and decompressed but generation still used plain text, so compression had zero effect on accuracy. The KV injection modes fix this by feeding the (potentially degraded) KV cache directly into attention.

**Chunked prefill:** For large retrieved contexts, `encode_context_to_kv()` splits the context into chunks and processes them incrementally, passing accumulated KV between chunks. This keeps activation memory bounded while maintaining contiguous RoPE positions.

---

## Compression Pipeline

Four compression strategies, from no-op to aggressive:

| Method | Bits/Value | Technique |
|--------|-----------|-----------|
| **FP16** | 16 | Cast to float16 (lossless from fp16 input) |
| **INT8** | 8 | Symmetric per-tensor or per-channel quantization |
| **INT4** | ~4.25 | Groupwise symmetric quantization with per-group scales |
| **TurboQuant-like** | ~4.25 | Random orthogonal rotation (or fast Walsh-Hadamard transform) before groupwise quantization. Spreads outlier magnitudes more evenly, reducing quantization error for the same bit budget. |

The TurboQuant-like compressor pipeline:

```
Compress:   tensor -> rotate (QR or WHT) -> groupwise scalar quantize -> store as int8
Decompress: load   -> dequantize          -> inverse rotate            -> output
```

The rotation matrix is deterministic (seeded), so it is not stored per tensor.

---

## Experiment Design

### Scale Sweep Parameters

The `SweepConfig` defines axes to sweep over:

| Axis | Default Values | What It Tests |
|------|---------------|---------------|
| `bank_sizes` | [500, 1000, 2000, 4000] | How quality degrades as stored memory grows |
| `block_chars` | [500] | Characters per block (controls block granularity) |
| `top_k_values` | [10, 20, 50] | Retrieval breadth vs. noise |
| `compression_methods` | [none, int8, int4, turboquant_like] | Compression vs. quality tradeoff |
| `modes` | [sparse_text, kv_inject, kv_inject_compressed, oracle_kv_inject, oracle_kv_inject_compressed] | Strategy comparison |
| `num_trials` | 5 | Repeated trials per combo (averaged for statistical signal) |

The runner generates the cross-product (with smart pruning of nonsensical combos like compression on non-compression modes), executes each with the eval harness, and produces summary tables and plots.

### Metrics Collected

**Quality metrics:**
- Accuracy (substring match on NIAH answers)
- Retrieval recall@k, MRR, hit rate, precision@k

**Systems metrics (per-phase profiling):**
- Wall clock time (total and per-phase: bank_build, route, kv_encode, compress, generate, score)
- Peak GPU VRAM (MB)
- Peak RAM usage (MB)
- Bytes fetched from memory bank
- Tokens generated per second
- Compression ratio

---

## Results

*Pending. Run `python demo_scale.py` to generate results.*

Results will be saved to `results/scale_sweep/` including:
- `sweep_full.json` — all individual run records
- `sweep_summary.csv` — averaged metrics per parameter combo
- `max_context_summary.json` — maximum viable context per compression method
- `plots/` — accuracy vs. context size, max context comparison, latency breakdown

---

## How to Interpret Results

### What to look for

1. **Accuracy vs. bank size**: As the bank grows, does sparse retrieval maintain accuracy? A flat line means the router is effective; a declining curve means relevant blocks get lost in noise.

2. **Sparse vs. oracle KV injection**: The gap isolates retrieval quality. If oracle is much better, the bottleneck is retrieval, not compression.

3. **Oracle KV compressed across methods**: With perfect retrieval, the only variable is compression. This directly measures how INT8/INT4/TurboQuant degrade KV quality.

4. **TurboQuant vs. INT4**: Both use ~4.25 bits. If TurboQuant-like maintains higher accuracy, the rotation is doing its job of spreading outlier magnitudes.

5. **Latency breakdown**: The phase-level profiler shows where time is spent. At small bank sizes, generation dominates. At large bank sizes, KV encoding and routing may become significant.

### Common pitfalls

- **Small bank sizes don't stress the system.** With 10 blocks and top_k=5, you're retrieving half the bank — of course recall is high. Use bank_size >> top_k for meaningful measurements.
- **Synthetic data is not real data.** NIAH tasks test retrieval precision but don't capture the distributional complexity of real documents. Results here are necessary but not sufficient.
- **DummyModel tests verify the pipeline, not quality.** The test suite uses a DummyModel with random weights. Real quality measurements require a loaded model (e.g., Qwen2.5-3B).

---

## Limitations

1. **No actual model inference in tests.** All 309 tests use a DummyModel with random tensors. This validates the pipeline end-to-end but says nothing about real model quality.

2. **Residual correction not implemented.** The TurboQuant-like compressor has a `residual_correction` flag, but it's a no-op placeholder for future work.

3. **Single-GPU, CPU-FAISS only.** FAISS GPU is unavailable on this platform. For bank sizes in the thousands, CPU FAISS with IndexFlatIP is fine; for millions, approximate indices would be needed.

4. **No streaming/incremental bank updates.** The bank is built offline in one pass. Incremental append and eviction are future work.

5. **Windows-specific.** Tested on Windows 11 with `multiprocessing.freeze_support()` and `num_workers=0`. Linux should work but is untested.

---

## How Not to Overclaim

This harness is for **measuring** retrieval and compression, not for claiming breakthrough context lengths. Specific non-claims:

- "1M-token context" means 1M tokens are **stored** in the external bank. The model's active attended context is still bounded by its native window (e.g., 32K for Qwen2.5-3B).
- Compression ratios are measured on KV tensors, not on end-to-end throughput or cost savings.
- Accuracy on synthetic NIAH tasks does not generalize to real-world document QA without further validation.
- The TurboQuant-like compressor is inspired by the paper's core idea (rotation before quantization) but is not a faithful reproduction of the full method.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup (loads config, logs system info, exits)
python -m src.main --dry-run

# Run tests (309 tests, ~35s)
pytest

# Run the full scale sweep on a real model (requires CUDA GPU)
python demo_scale.py
```

---

## Project Structure

```
msa_turboquant_local/
├── configs/                # YAML experiment configs
│   ├── model.yaml          #   Model name, dtype, device, sequence length
│   ├── retrieval.yaml      #   FAISS settings, chunk size, top-k, routing
│   ├── compression.yaml    #   Compression method and parameters
│   ├── benchmarks.yaml     #   Task definitions, metrics, output settings
│   └── experiment.yaml     #   Experiment name, seed, logging, paths
├── data/                   # Raw data, processed data, memory banks, bank cache
├── docs/
│   └── PROJECT_SPEC.md     # Full project specification (11 milestones)
├── src/
│   ├── main.py             # CLI entrypoint with argparse
│   ├── models/
│   │   ├── base_model.py   #   BaseModel ABC, ModelOutput, TokenizedInput
│   │   ├── hf_model.py     #   HuggingFace model wrapper
│   │   └── kv_extractor.py #   KVBlock, KVExtractor (direct + hidden_state modes)
│   ├── memory/
│   │   ├── chunking.py     #   TextBlock, chunk_text, chunk_by_tokens
│   │   ├── bank_builder.py #   MemoryBankBuilder, MemoryBank, MemoryBankMetadata
│   │   ├── bank_store.py   #   save_bank, load_bank, load_routing_vectors, load_kv_for_blocks
│   │   ├── bank_cache.py   #   BankCache (build once, lazy KV loading)
│   │   ├── router.py       #   FaissRouter, TorchCosineRouter, OracleRouter
│   │   ├── fetcher.py      #   MemoryFetcher (in-memory + disk-backed)
│   │   ├── kv_injector.py  #   KV injection: encode_context_to_kv, chunked prefill
│   │   └── interleave.py   #   assemble_context (prepend/interleave/summarize_prefix)
│   ├── compression/
│   │   ├── base.py         #   BaseCompressor ABC, CompressedTensor
│   │   ├── fp16.py         #   FP16Compressor (passthrough)
│   │   ├── int8.py         #   Int8Compressor (symmetric, per-channel)
│   │   ├── int4.py         #   Int4Compressor (groupwise symmetric)
│   │   └── turboquant_like.py  # Rotation + quantization (QR or WHT)
│   ├── eval/
│   │   ├── niah.py         #   NIAH + passkey dataset generation
│   │   ├── retrieval_metrics.py  # recall@k, MRR, hit_rate, precision@k
│   │   ├── systems_metrics.py    # SystemsSnapshot, RunMetrics, MetricsCollector
│   │   ├── profiler.py     #   RunProfiler, PhaseRecord, ProfilingReport
│   │   └── run_eval.py     #   EvalHarness (KV injection + legacy modes)
│   ├── experiments/
│   │   ├── sweep_config.py #   SweepConfig, SweepRunRecord, SweepResult
│   │   ├── run_scale_sweep.py  # ScaleSweep runner
│   │   └── sweep_plots.py  #   Accuracy, latency, compression plots
│   └── utils/
│       ├── config.py       #   Pydantic config models, load_config
│       ├── logging_utils.py #   setup_logging, JSON formatter
│       ├── profiling.py    #   Timer, GPUMemoryTracker, log_system_info
│       ├── plotting.py     #   matplotlib helpers
│       └── io_utils.py     #   JSON/CSV/YAML I/O, path helpers
├── tests/                  # 309 pytest tests
│   ├── conftest.py
│   ├── test_config.py      #   Config loading, validation, overrides
│   ├── test_logging.py     #   Logger setup, JSON formatting
│   ├── test_profiling.py   #   Timer, GPUMemoryTracker
│   ├── test_io_utils.py    #   JSON/CSV/YAML round-trips
│   ├── test_chunking.py    #   Text chunking, token chunking
│   ├── test_niah.py        #   NIAH dataset generation
│   ├── test_models.py      #   DummyModel, BaseModel interface
│   ├── test_bank_store.py  #   Bank persistence, selective loading
│   ├── test_router.py      #   FAISS/cosine/oracle routers, fetcher
│   ├── test_compression.py #   FP16, INT8, INT4 compressors
│   ├── test_turboquant.py  #   TurboQuant-like rotation + quantization
│   ├── test_eval.py        #   Eval harness (KV injection + legacy modes)
│   ├── test_kv_injector.py #   KV injection, chunked prefill, compression roundtrip
│   ├── test_profiler.py    #   RunProfiler, compression wiring
│   └── test_sweep.py       #   SweepConfig, ScaleSweep, sweep plots
├── demo.py                 # Quick pipeline demo (DummyModel)
├── demo_real_model.py      # Single-run demo with real model
├── demo_scale.py           # Full scale sweep (Qwen2.5-3B, 500-4000 blocks)
├── notebooks/              # Exploration notebooks
├── results/                # Experiment outputs
├── pyproject.toml
├── requirements.txt
└── README.md               # This file
```

---

## Configuration

All behavior is config-driven via YAML files in `configs/`:

| File | Controls |
|------|----------|
| `model.yaml` | Model name, dtype, device, max sequence length |
| `retrieval.yaml` | FAISS engine, index type, top-k, chunk size, routing vector method |
| `compression.yaml` | Compression method (none/fp16/int8/int4/turboquant_like) and sub-configs |
| `benchmarks.yaml` | NIAH and passkey task definitions, metrics list, output settings |
| `experiment.yaml` | Experiment name, random seed, logging level, device, paths |

CLI overrides: `python -m src.main --override model.max_seq_len=4096 --dry-run`

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5060 Ti (16GB VRAM) |
| RAM | 24GB system memory |
| OS | Windows 11 Pro |
| Python | 3.14 |
| PyTorch | 2.10+ with CUDA 12.8 |

---

## Milestones

- [x] M1: Project scaffolding (config, logging, profiling, CLI, tests)
- [x] M2: Chunking and NIAH synthetic dataset generation
- [x] M3: Model wrapper and KV extraction (direct + hidden state fallback)
- [x] M4: Memory bank builder and disk-backed storage
- [x] M5: Router and retrieval (FAISS, cosine, oracle)
- [x] M6: Baseline evaluation harness (5 modes, scoring, result saving)
- [x] M7: Compression baselines (FP16, INT8, INT4)
- [x] M8: TurboQuant-inspired compressor (rotation + groupwise quantization)
- [x] M9: Systems profiling (per-phase timing, memory tracking, bytes fetched)
- [x] M10: Scale sweeps (config-driven parameter grid, summary tables, auto plots)
- [x] M11: KV injection pipeline (chunked prefill, oracle variants, compression roundtrip)
- [ ] M12: Full scale sweep results and analysis

---

## License

Research use only. Not intended for production deployment.
