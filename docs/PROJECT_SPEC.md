# MSA TurboQuant Local — Full Project Specification

> This file preserves the complete project scope, milestones, and requirements.
> Saved as a reference in case conversation context is lost.

---

## Project Framing

Build a local extreme-context memory benchmark harness for sparse retrieval + compressed KV storage.

## Important Context

- Hardware: Windows desktop with NVIDIA RTX 5060 Ti 16GB GPU
- Goal: practical local prototype, not a giant distributed system
- NOT claiming true dense 1B-token attention
- Goal IS to maximize STORED memory size, keep active attended context small, and measure quality/performance degradation as stored memory grows
- Clean, rigorous evaluation harness
- Compare:
  1. Dense local baseline
  2. Sparse retrieval only
  3. Compression only
  4. Sparse retrieval + compression
  5. Oracle router + compression
- Test long-context scaling behavior as memory bank size grows

## Core Ideas

- Use a small open model that fits comfortably on 16GB VRAM
- Build an EXTERNAL memory bank:
  - routing vectors per block
  - content K tensors
  - content V tensors
  - metadata
- Store the external memory bank mostly in CPU RAM / memmap / disk
- At inference time:
  - compute a routing vector from current query/context
  - retrieve top-k memory blocks
  - fetch those blocks only
  - inject or append them into active generation context
- Implement compression modules:
  - fp16 baseline
  - int8 baseline
  - int4/groupwise baseline
  - TurboQuant-inspired compression:
    - fixed random rotation or Hadamard-style transform
    - scalar quantization after rotation
    - optional residual correction path for keys later
- Build synthetic and real long-context evaluation tasks
- Log memory usage, retrieval quality, throughput, latency, and answer quality

## Technical Constraints

- Prioritize correctness, modularity, and instrumentation over fancy optimizations
- Write production-quality Python with clear comments, typing, error handling, and docstrings
- Use config-driven experiments
- Make it easy to run sweeps over memory size, top-k, block size, and compression method
- Target local iteration speed first
- Avoid overengineering the first version
- If there are tradeoffs, favor a clean, inspectable harness over a maximally optimized system

## Preferred Stack

- Python
- PyTorch
- Hugging Face Transformers
- Accelerate
- FAISS or torch cosine similarity for routing
- numpy memmap / safetensors / parquet / sqlite as needed
- matplotlib for plots
- pydantic or dataclasses for config if useful
- pytest for at least basic tests

## Suggested Starter Model

- Start with a 3B-4B instruct model if feasible
- Keep the model abstraction generic enough to swap later
- Do NOT hardcode one vendor/model too deeply

## Repository Structure

```
msa_turboquant_local/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── model.yaml
│   ├── retrieval.yaml
│   ├── compression.yaml
│   ├── benchmarks.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── memory_banks/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── hf_model.py
│   │   └── kv_extractor.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── bank_builder.py
│   │   ├── bank_store.py
│   │   ├── router.py
│   │   ├── fetcher.py
│   │   └── interleave.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fp16.py
│   │   ├── int8.py
│   │   ├── int4.py
│   │   └── turboquant_like.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── niah.py
│   │   ├── long_qa.py
│   │   ├── retrieval_metrics.py
│   │   ├── systems_metrics.py
│   │   └── run_eval.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_scale_sweep.py
│   │   ├── run_ablations.py
│   │   └── run_oracle_router.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── profiling.py
│       ├── io_utils.py
│       └── plotting.py
├── tests/
│   ├── test_chunking.py
│   ├── test_compression.py
│   ├── test_router.py
│   └── test_bank_store.py
├── notebooks/
└── results/
```

## High-Level Milestones

### MILESTONE 1 — Project scaffolding
- Create repo structure
- Create requirements and pyproject
- Create config loader
- Create logging utilities
- Create README with project purpose and quickstart
- Create main entrypoint
- Create minimal tests that run

### MILESTONE 2 — Chunking and synthetic dataset generation
- Implement block chunking utilities
- Implement synthetic needle-in-a-haystack data generator
- Allow generation of corpora with:
  - one needle
  - multiple distractors
  - configurable block size
  - configurable memory size
- Save generated datasets to disk
- Add tests

### MILESTONE 3 — Model wrapper and KV extraction
- Implement a generic model wrapper interface
- Implement Hugging Face model backend
- Add methods for:
  - tokenize
  - forward hidden states
  - extract per-layer K/V where feasible
  - mean-pool hidden states for routing vectors
- If exact K/V extraction is tricky initially, implement a clean fallback abstraction and document it
- Make this robust and inspectable

### MILESTONE 4 — Memory bank builder and storage
- Build offline memory bank creation pipeline
- For each text block, store:
  - block_id
  - document_id
  - token span
  - routing vector
  - content representations (initially hidden state or simplified K/V representation if needed)
- Implement bank storage using memmap or another efficient local format
- Implement ability to reload bank
- Add metadata index
- Add tests

### MILESTONE 5 — Router and retrieval
- Implement cosine similarity router
- Implement top-k retrieval
- Implement oracle router mode for synthetic tasks
- Log:
  - retrieved block IDs
  - similarity scores
  - whether gold block was retrieved
- Add retrieval metrics:
  - recall@k
  - MRR
  - hit rate

### MILESTONE 6 — Baseline evaluation harness
- Implement evaluation loop for synthetic needle task
- Support modes:
  - dense
  - sparse_full_precision
  - compression_only
  - sparse_plus_compression
  - oracle_router_plus_compression
- Log structured metrics to JSON/CSV
- Add plotting utilities for:
  - recall vs memory size
  - latency vs memory size
  - VRAM/RAM vs memory size

### MILESTONE 7 — Compression baselines
- Implement fp16 passthrough
- Implement int8 compressor
- Implement int4/groupwise compressor
- Provide a common interface:
  - compress()
  - decompress()
  - estimate_bits_per_value()
- Add tests comparing reconstruction error and shape correctness

### MILESTONE 8 — TurboQuant-inspired compressor
- Implement a first-pass TurboQuant-like module:
  - fixed random orthogonal rotation or fast approximate transform
  - scalar quantization in rotated space
  - decompression path
  - metrics for reconstruction error and dot-product distortion
- Do NOT overclaim that this exactly reproduces the paper
- Clearly label it as "TurboQuant-inspired"
- Add experiments comparing:
  - reconstruction MSE
  - cosine similarity
  - dot-product error
  - retrieval rank agreement

### MILESTONE 9 — Systems profiling
- Measure:
  - runtime
  - tokens/sec if generation loop exists
  - retrieval time
  - peak VRAM
  - RAM usage
  - bytes fetched from bank
- Save results per run
- Make profiling optional but easy to enable

### MILESTONE 10 — Scale sweeps
- Implement config-driven sweeps over:
  - memory bank size
  - block size
  - top-k
  - compression method
- Create scripts that run repeated experiments and save summary tables
- Generate summary plots automatically

### MILESTONE 11 — Long-form README and research framing
- Update README with:
  - system design
  - limitations
  - experiment plan
  - how to interpret results
  - how not to overclaim
- Include an example research question:
  "As stored memory grows, how do sparse retrieval and KV compression affect retrieval recall, answer quality, and systems performance?"

## Implementation Preferences

- Use dataclasses or pydantic models for experiment configs
- Keep modules decoupled
- Prefer simple interfaces and composability
- Use pathlib, not brittle raw string paths
- Add a small CLI entrypoint where practical
- Ensure Windows compatibility
- Avoid Linux-only assumptions
- Include comments where model/KV internals are uncertain

## Starter File Expectations

When creating files, do not just create empty placeholders.
Each file should contain enough starter implementation that I can run, inspect, and extend it.

Specific starter behaviors wanted:
- `chunking.py`: text to token blocks or simple text blocks, metadata-rich block objects
- `niah.py`: generate synthetic long-context corpora with configurable needle placement
- `router.py`: cosine similarity retrieval and oracle mode
- `compression/base.py`: abstract compressor interface
- `compression/int8.py` and `compression/int4.py`: simple working quantization baselines
- `compression/turboquant_like.py`: rotation + quantization based compressor
- `bank_store.py`: save/load arrays and metadata
- `run_scale_sweep.py`: take config, run experiments, save outputs
- `plotting.py`: load metrics files and create basic matplotlib charts

## Development Style

- After each milestone, stop and summarize:
  - files created/updated
  - how to run
  - what assumptions were made
  - what technical debt remains
- If a milestone depends on a design decision, choose a sensible default and proceed
- If exact transformer KV extraction becomes a bottleneck, build a clean approximation layer first

## Key Principles

- Build this as a RESEARCH HARNESS, not a polished product
- Make results reproducible
- Keep experiment configuration explicit
- Make it easy to compare baselines fairly
- Emphasize instrumentation and ablations
