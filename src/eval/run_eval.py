"""Evaluation harness for long-context memory experiments.

Orchestrates the full evaluation pipeline:
1. Generate or load synthetic data (NIAH)
2. Build memory bank (if needed for sparse/oracle modes)
3. For each sample: route, fetch, assemble context, generate answer, score
4. Collect and save all metrics

Supports modes:
- dense: Full context window, no retrieval
- sparse / sparse_text: Sparse retrieval from memory bank (text generation)
- kv_inject: Sparse retrieval with direct KV injection into attention
- kv_inject_compressed: Sparse retrieval + compressed KV injection
- oracle_kv_inject: Oracle routing + KV injection (upper bound for retrieval)
- oracle_kv_inject_compressed: Oracle routing + compressed KV injection

Legacy modes (compression has no effect on accuracy):
- compression_only: Full context, KV compressed via compressor
- sparse_plus_compression: Sparse retrieval with compressed KV (text gen)
- oracle_plus_compression: Oracle routing with compressed KV (text gen)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from src.compression.base import BaseCompressor
from src.eval.niah import NIAHSample, generate_niah_dataset
from src.eval.profiler import RunProfiler
from src.eval.retrieval_metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    compute_retrieval_metrics,
)
from src.eval.systems_metrics import MetricsCollector, RunMetrics
from src.memory.bank_builder import MemoryBank, MemoryBankBuilder
from src.memory.chunking import TextBlock, chunk_text
from src.memory.fetcher import FetchResult, MemoryFetcher
from src.memory.interleave import assemble_context, assemble_dense_context
from src.memory.kv_injector import assemble_kv_injection, encode_context_to_kv
from src.memory.router import create_router
from src.models.base_model import BaseModel
from src.utils.io_utils import ensure_dir, save_csv, save_json, timestamp_str

logger = logging.getLogger("msa_turboquant.eval.run_eval")


EvalMode = Literal[
    "dense",
    "sparse",
    "sparse_text",
    "compression_only",
    "sparse_plus_compression",
    "oracle_plus_compression",
    "kv_inject",
    "kv_inject_compressed",
    "oracle_kv_inject",
    "oracle_kv_inject_compressed",
]

_LEGACY_MODES = {"sparse_plus_compression", "oracle_plus_compression", "compression_only"}


@dataclass
class EvalSampleResult:
    """Result for a single evaluation sample.

    Attributes:
        sample_id: Source sample identifier.
        mode: Evaluation mode used.
        needle_answer: Expected answer.
        model_answer: Model's generated answer.
        correct: Whether the model answer contains the expected answer.
        retrieval: Retrieval metrics (if applicable).
        context_chars: Total chars in assembled context.
        num_retrieved: Number of blocks retrieved.
        retrieval_time_ms: Routing + fetch time.
        generation_time_ms: Model generation time.
    """
    sample_id: str = ""
    mode: str = ""
    needle_answer: str = ""
    model_answer: str = ""
    correct: bool = False
    exact_match: bool = False
    retrieval: dict = field(default_factory=dict)
    context_chars: int = 0
    num_retrieved: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    bytes_fetched: int = 0
    compression_ratio: float = 0.0
    # Multi-needle fields
    task_type: str = "single_needle"
    needles_found: int = 0
    needles_total: int = 0
    needle_accuracy: float = 0.0
    distractor_confusions: int = 0

    def to_dict(self) -> dict:
        d = {
            "sample_id": self.sample_id,
            "mode": self.mode,
            "needle_answer": self.needle_answer,
            "model_answer": self.model_answer,
            "correct": self.correct,
            "context_chars": self.context_chars,
            "num_retrieved": self.num_retrieved,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "bytes_fetched": self.bytes_fetched,
            "compression_ratio": round(self.compression_ratio, 2),
            "task_type": self.task_type,
            "needles_found": self.needles_found,
            "needles_total": self.needles_total,
            "needle_accuracy": round(self.needle_accuracy, 4),
            "distractor_confusions": self.distractor_confusions,
            **{f"retrieval_{k}": v for k, v in self.retrieval.items()},
        }
        return d


@dataclass
class EvalRunResult:
    """Aggregated results for a full evaluation run.

    Attributes:
        run_id: Unique run identifier.
        mode: Evaluation mode.
        num_samples: Number of samples evaluated.
        accuracy: Fraction of samples where model answer was correct.
        sample_results: Per-sample results.
        retrieval_metrics: Aggregated retrieval metrics.
        systems_metrics: Systems-level metrics for the run.
        config: Config snapshot for reproducibility.
    """
    run_id: str = ""
    mode: str = ""
    num_samples: int = 0
    accuracy: float = 0.0
    sample_results: list[EvalSampleResult] = field(default_factory=list)
    retrieval_metrics: dict = field(default_factory=dict)
    systems_metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "num_samples": self.num_samples,
            "accuracy": round(self.accuracy, 4),
            "retrieval_metrics": self.retrieval_metrics,
            "systems_metrics": self.systems_metrics,
            "config": self.config,
            "sample_results": [s.to_dict() for s in self.sample_results],
        }


def score_answer(model_answer: str, expected_answer: str) -> bool:
    """Check if the model's answer contains the expected answer.

    Normalizes whitespace and punctuation before matching so that
    tokenization artifacts (e.g. "550 6" for "5506") don't cause
    false negatives.  For numeric answers (common in NIAH), also
    strips all spaces/punctuation from both strings before comparing.
    """
    return score_answer_detailed(model_answer, expected_answer)[0]


def score_answer_detailed(model_answer: str, expected_answer: str) -> tuple[bool, bool]:
    """Score with detail: returns (correct, exact_match).

    exact_match is True only if the expected answer appears as-is (no normalization).
    correct is True if it matches after whitespace/punctuation normalization.
    """
    answer = model_answer.lower().strip()
    expected = expected_answer.lower().strip()

    # Direct substring match
    if expected in answer:
        return True, True

    # Normalize: collapse whitespace and strip punctuation, then retry.
    # Handles "550 6" -> "5506", "55,06" -> "5506", etc.
    import re
    answer_norm = re.sub(r"[\s.,;:!?'\"-]+", "", answer)
    expected_norm = re.sub(r"[\s.,;:!?'\"-]+", "", expected)
    if expected_norm in answer_norm:
        return True, False

    return False, False


def _build_text_blocks_from_niah(sample: NIAHSample) -> list[TextBlock]:
    """Convert NIAH sample blocks into TextBlock objects for bank building."""
    blocks = []
    for i, text in enumerate(sample.blocks):
        is_needle = i in sample.needle_block_indices
        blocks.append(TextBlock(
            block_id=f"{sample.sample_id}_blk_{i}",
            document_id=sample.sample_id,
            block_index=i,
            text=text,
            char_start=0,
            char_end=len(text),
            metadata={"is_needle": is_needle},
        ))
    return blocks


class EvalHarness:
    """Main evaluation harness orchestrating the full pipeline.

    Usage:
        harness = EvalHarness(model=model, mode="sparse", top_k=5)
        result = harness.evaluate(samples)
        harness.save_results(result, output_dir)
    """

    def __init__(
        self,
        model: BaseModel,
        mode: EvalMode = "dense",
        router_engine: str = "faiss",
        top_k: int = 5,
        max_new_tokens: int = 64,
        max_context_chars: int | None = None,
        assembly_strategy: str = "prepend",
        compressor: BaseCompressor | None = None,
        profiler: RunProfiler | None = None,
    ) -> None:
        """
        Args:
            model: A loaded BaseModel instance.
            mode: Evaluation mode.
            router_engine: Router backend ("faiss", "torch_cosine", "oracle").
            top_k: Number of blocks to retrieve in sparse modes.
            max_new_tokens: Max tokens to generate per sample.
            max_context_chars: Truncation limit for assembled context.
            assembly_strategy: How to combine context + query.
            compressor: Optional compressor for compression modes. When
                provided, KV blocks are compressed/decompressed during
                evaluation to measure quality impact.
            profiler: Optional RunProfiler for per-phase timing. When None,
                profiling is skipped with no overhead.
        """
        self._model = model
        self._mode = mode
        self._router_engine = router_engine
        self._top_k = top_k
        self._max_new_tokens = max_new_tokens
        self._max_context_chars = max_context_chars
        self._assembly_strategy = assembly_strategy
        self._compressor = compressor
        self._profiler = profiler or RunProfiler(enabled=False)

    def evaluate(
        self,
        samples: list[NIAHSample],
        run_id: str | None = None,
        prebuilt_bank: MemoryBank | None = None,
    ) -> EvalRunResult:
        """Run evaluation over a list of NIAH samples.

        Args:
            samples: List of NIAHSample objects to evaluate.
            run_id: Optional run identifier (auto-generated if None).
            prebuilt_bank: Optional pre-built MemoryBank to reuse across
                evaluations, avoiding expensive bank rebuilding per sample.

        Returns:
            EvalRunResult with all metrics.
        """
        run_id = run_id or f"eval_{self._mode}_{timestamp_str()}"
        logger.info(f"Starting eval run '{run_id}': mode={self._mode}, samples={len(samples)}")

        config_snapshot = {
            "mode": self._mode,
            "router_engine": self._router_engine,
            "top_k": self._top_k,
            "max_new_tokens": self._max_new_tokens,
            "num_samples": len(samples),
        }
        if self._compressor is not None:
            config_snapshot["compressor"] = self._compressor.name
            config_snapshot["bits_per_value"] = self._compressor.estimate_bits_per_value()

        self._profiler.run_id = run_id
        self._profiler.start()

        collector = MetricsCollector(run_id=run_id, mode=self._mode, config=config_snapshot)
        sample_results: list[EvalSampleResult] = []
        per_sample_retrieval: list[RetrievalMetrics] = []

        with collector:
            for i, sample in enumerate(samples):
                logger.info(f"  Sample {i+1}/{len(samples)}: {sample.sample_id}")
                result = self._evaluate_single(sample, collector, prebuilt_bank=prebuilt_bank)
                sample_results.append(result)

                if result.retrieval:
                    per_sample_retrieval.append(
                        RetrievalMetrics(**{k: v for k, v in result.retrieval.items()
                                           if k in RetrievalMetrics.__dataclass_fields__})
                    )

            collector.set_num_samples(len(samples))

        # Aggregate
        num_correct = sum(1 for r in sample_results if r.correct)
        accuracy = num_correct / len(samples) if samples else 0.0

        agg_retrieval = aggregate_metrics(per_sample_retrieval) if per_sample_retrieval else RetrievalMetrics()
        systems = collector.get_metrics()

        collector.set_retrieval_metrics(agg_retrieval.to_dict())
        collector.set_answer_metrics({"accuracy": accuracy})

        # Attach compression info to profiler
        if self._compressor is not None:
            self._profiler.set_compression_info({
                "method": self._compressor.name,
                "bits_per_value": self._compressor.estimate_bits_per_value(),
            })

        run_result = EvalRunResult(
            run_id=run_id,
            mode=self._mode,
            num_samples=len(samples),
            accuracy=accuracy,
            sample_results=sample_results,
            retrieval_metrics=agg_retrieval.to_dict(),
            systems_metrics=systems.to_dict(),
            config=config_snapshot,
        )

        logger.info(
            f"Run '{run_id}' complete: accuracy={accuracy:.2%}, "
            f"recall@{self._top_k}={agg_retrieval.recall_at_k:.2%}, "
            f"wall_time={systems.wall_time_ms:.0f}ms"
        )

        return run_result

    def _evaluate_single(
        self,
        sample: NIAHSample,
        collector: MetricsCollector,
        prebuilt_bank: MemoryBank | None = None,
    ) -> EvalSampleResult:
        """Evaluate a single NIAH sample."""
        # Warn on legacy modes
        if self._mode in _LEGACY_MODES:
            import warnings
            warnings.warn(
                f"Mode '{self._mode}' is legacy: compression has no effect on "
                f"accuracy because generation uses plain text. Use "
                f"'kv_inject_compressed' or 'oracle_kv_inject_compressed' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        needle = sample.needles[0]  # Primary needle
        question = needle.question
        expected = needle.answer
        gold_indices = sample.needle_block_indices

        retrieval_metrics_dict: dict = {}
        retrieval_time = 0.0
        num_retrieved = 0
        bytes_fetched = 0
        compression_ratio = 0.0
        use_compression = (
            self._compressor is not None
            and self._mode in ("compression_only", "sparse_plus_compression", "oracle_plus_compression")
        )

        # --- KV injection modes (the research-relevant modes) ---
        if self._mode in ("kv_inject", "kv_inject_compressed",
                          "oracle_kv_inject", "oracle_kv_inject_compressed"):
            return self._evaluate_kv_inject_mode(
                sample, collector, prebuilt_bank,
            )

        # --- Text-based modes ---
        if self._mode == "dense":
            # Dense: use all blocks as context
            assembled = assemble_dense_context(
                query=question,
                full_context_blocks=sample.blocks,
                max_context_chars=self._max_context_chars,
            )

        elif self._mode in ("sparse", "sparse_text", "sparse_plus_compression"):
            # Build a bank and retrieve (or reuse prebuilt)
            if prebuilt_bank is not None:
                bank = prebuilt_bank
                fetcher = MemoryFetcher.from_bank(bank, engine=self._router_engine)
            else:
                with self._profiler.phase("bank_build"):
                    text_blocks = _build_text_blocks_from_niah(sample)
                    bank_builder = MemoryBankBuilder(self._model, extraction_mode="direct")
                    bank = bank_builder.build(text_blocks, bank_id=sample.sample_id)
                    fetcher = MemoryFetcher.from_bank(bank, engine=self._router_engine)

            # Get query routing vector and retrieve
            with self._profiler.phase("route") as phase_ctx:
                query_vec = self._model.get_routing_vectors(question, pooling="mean")[0]
                fetch_result = fetcher.fetch(
                    query_vec, top_k=self._top_k, gold_indices=gold_indices
                )
                phase_ctx.set_counter("num_fetched", fetch_result.num_fetched)

            retrieval_time = fetch_result.total_time_ms
            collector.add_retrieval_time(retrieval_time)
            num_retrieved = fetch_result.num_fetched
            bytes_fetched = fetch_result.total_kv_bytes
            self._profiler.add_bytes_fetched(bytes_fetched)

            # Compress/decompress KV blocks to measure quality impact
            if use_compression:
                with self._profiler.phase("compress"):
                    compression_ratio = self._compress_kv_blocks(fetch_result.kv_blocks)

            # Compute retrieval metrics
            ret_metrics = compute_retrieval_metrics(
                fetch_result.retrieval.block_indices, gold_indices
            )
            retrieval_metrics_dict = ret_metrics.to_dict()

            # Assemble context from retrieved blocks' text
            retrieved_texts = [
                sample.blocks[idx] for idx in fetch_result.retrieval.block_indices
            ]
            assembled = assemble_context(
                query=question,
                retrieved_texts=retrieved_texts,
                retrieved_block_ids=fetch_result.retrieval.block_ids,
                strategy=self._assembly_strategy,
                max_context_chars=self._max_context_chars,
            )

        elif self._mode == "oracle_plus_compression":
            # Oracle: use gold blocks directly (or reuse prebuilt)
            if prebuilt_bank is not None:
                bank = prebuilt_bank
                fetcher = MemoryFetcher.from_bank(bank, engine="oracle")
            else:
                with self._profiler.phase("bank_build"):
                    text_blocks = _build_text_blocks_from_niah(sample)
                    bank_builder = MemoryBankBuilder(self._model, extraction_mode="direct")
                    bank = bank_builder.build(text_blocks, bank_id=sample.sample_id)
                    fetcher = MemoryFetcher.from_bank(bank, engine="oracle")

            with self._profiler.phase("route") as phase_ctx:
                query_vec = self._model.get_routing_vectors(question, pooling="mean")[0]
                fetch_result = fetcher.fetch(
                    query_vec, top_k=self._top_k, gold_indices=gold_indices
                )
                phase_ctx.set_counter("num_fetched", fetch_result.num_fetched)

            retrieval_time = fetch_result.total_time_ms
            num_retrieved = fetch_result.num_fetched
            bytes_fetched = fetch_result.total_kv_bytes
            self._profiler.add_bytes_fetched(bytes_fetched)

            if use_compression:
                with self._profiler.phase("compress"):
                    compression_ratio = self._compress_kv_blocks(fetch_result.kv_blocks)

            ret_metrics = compute_retrieval_metrics(
                fetch_result.retrieval.block_indices, gold_indices
            )
            retrieval_metrics_dict = ret_metrics.to_dict()

            retrieved_texts = [
                sample.blocks[idx] for idx in fetch_result.retrieval.block_indices
            ]
            assembled = assemble_context(
                query=question,
                retrieved_texts=retrieved_texts,
                strategy=self._assembly_strategy,
                max_context_chars=self._max_context_chars,
            )

        elif self._mode == "compression_only":
            # Compression only: build bank from all blocks, compress KV
            if use_compression:
                if prebuilt_bank is not None:
                    bank = prebuilt_bank
                else:
                    with self._profiler.phase("bank_build"):
                        text_blocks = _build_text_blocks_from_niah(sample)
                        bank_builder = MemoryBankBuilder(self._model, extraction_mode="direct")
                        bank = bank_builder.build(text_blocks, bank_id=sample.sample_id)

                with self._profiler.phase("compress"):
                    compression_ratio = self._compress_kv_blocks(bank.kv_blocks)
                    bytes_fetched = sum(kb.total_kv_bytes for kb in bank.kv_blocks)
                    self._profiler.add_bytes_fetched(bytes_fetched)

            assembled = assemble_dense_context(
                query=question,
                full_context_blocks=sample.blocks,
                max_context_chars=self._max_context_chars,
            )

        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        # Generate answer (text-based modes)
        with self._profiler.phase("generate") as gen_phase:
            gen_start = time.perf_counter()
            tokens = self._model.tokenize(assembled.text)
            output_ids = self._model.generate(
                tokens.input_ids,
                tokens.attention_mask,
                max_new_tokens=self._max_new_tokens,
            )
            gen_time = (time.perf_counter() - gen_start) * 1000
            collector.add_generation_time(gen_time)
            self._profiler.add_generation_time(gen_time)

        # Decode answer (only the generated part)
        prompt_len = tokens.input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        num_tokens = generated_ids.shape[1]
        collector.add_tokens(num_tokens)
        self._profiler.add_tokens(num_tokens)
        gen_phase.set_counter("tokens_generated", num_tokens)
        model_answer = self._model.decode(generated_ids)[0]

        # Score
        with self._profiler.phase("score"):
            correct, exact_match = score_answer_detailed(model_answer, expected)

        return EvalSampleResult(
            sample_id=sample.sample_id,
            mode=self._mode,
            needle_answer=expected,
            model_answer=model_answer,
            correct=correct,
            exact_match=exact_match,
            retrieval=retrieval_metrics_dict,
            context_chars=assembled.context_chars,
            num_retrieved=num_retrieved,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=gen_time,
            bytes_fetched=bytes_fetched,
            compression_ratio=compression_ratio,
        )

    def _evaluate_kv_inject_mode(
        self,
        sample: NIAHSample,
        collector: MetricsCollector,
        prebuilt_bank: MemoryBank | None = None,
    ) -> EvalSampleResult:
        """Evaluate using KV injection: encode retrieved context as KV cache,
        optionally compress/decompress, then inject into attention for generation.

        Handles all four KV injection modes and both single/multi-needle tasks.
        For multi-needle: builds KV once from retrieved context, then asks each
        needle's question against that same KV payload.
        """
        gold_indices = sample.needle_block_indices
        task_type = getattr(sample, "task_type", "single_needle")
        is_multi = len(sample.needles) > 1

        # For routing query, use first needle's question
        primary_question = sample.needles[0].question

        # Determine engine and compression based on mode
        use_oracle = self._mode in ("oracle_kv_inject", "oracle_kv_inject_compressed")
        use_compression = self._mode in ("kv_inject_compressed", "oracle_kv_inject_compressed")
        fetcher_engine = "oracle" if use_oracle else self._router_engine

        # Build or reuse bank
        if prebuilt_bank is not None:
            bank = prebuilt_bank
            fetcher = MemoryFetcher.from_bank(bank, engine=fetcher_engine)
        else:
            with self._profiler.phase("bank_build"):
                text_blocks = _build_text_blocks_from_niah(sample)
                bank_builder = MemoryBankBuilder(self._model, extraction_mode="direct")
                bank = bank_builder.build(text_blocks, bank_id=sample.sample_id)
                fetcher = MemoryFetcher.from_bank(bank, engine=fetcher_engine)

        # Route and fetch
        with self._profiler.phase("route") as phase_ctx:
            query_vec = self._model.get_routing_vectors(primary_question, pooling="mean")[0]
            fetch_result = fetcher.fetch(
                query_vec, top_k=self._top_k, gold_indices=gold_indices
            )
            phase_ctx.set_counter("num_fetched", fetch_result.num_fetched)

        retrieval_time = fetch_result.total_time_ms
        collector.add_retrieval_time(retrieval_time)
        num_retrieved = fetch_result.num_fetched
        bytes_fetched = fetch_result.total_kv_bytes
        self._profiler.add_bytes_fetched(bytes_fetched)

        # Retrieval metrics
        ret_metrics = compute_retrieval_metrics(
            fetch_result.retrieval.block_indices, gold_indices
        )
        retrieval_metrics_dict = ret_metrics.to_dict()

        # Assemble context text, encode to KV in a contiguous forward pass
        retrieved_texts = [
            sample.blocks[idx] for idx in fetch_result.retrieval.block_indices
        ]
        context_text = "Retrieved context:\n" + "\n\n".join(retrieved_texts)

        # For multi-needle, ask all questions in one framed query
        if is_multi:
            questions_text = "\n".join(
                f"Q{i+1}: {n.question}" for i, n in enumerate(sample.needles)
            )
            framed_query = f"\n{questions_text}\nAnswer each question with just the number:\n"
        else:
            framed_query = f"\nQuestion: {primary_question}\nAnswer:"

        compressor_for_inject = self._compressor if use_compression else None
        with self._profiler.phase("kv_encode"):
            payload, kv_blocks, compression_ratio = encode_context_to_kv(
                context_text=context_text,
                query_text=framed_query,
                model=self._model,
                compressor=compressor_for_inject,
                chunk_size=2048,
            )

        # Generate with injected KV
        query_tokens = self._model.tokenize(framed_query)
        with self._profiler.phase("generate") as gen_phase:
            gen_start = time.perf_counter()
            output_ids = self._model.generate(
                query_tokens.input_ids,
                attention_mask=payload.attention_mask,
                max_new_tokens=self._max_new_tokens * (len(sample.needles) if is_multi else 1),
                past_key_values=payload.past_key_values,
                position_ids=payload.position_ids,
            )
            gen_time = (time.perf_counter() - gen_start) * 1000
            collector.add_generation_time(gen_time)
            self._profiler.add_generation_time(gen_time)

        # Decode (only generated tokens)
        prompt_len = query_tokens.input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        num_tokens = generated_ids.shape[1]
        collector.add_tokens(num_tokens)
        self._profiler.add_tokens(num_tokens)
        gen_phase.set_counter("tokens_generated", num_tokens)
        model_answer = self._model.decode(generated_ids)[0]

        # Score
        with self._profiler.phase("score"):
            if is_multi:
                needles_found = 0
                exact_needles = 0
                for n in sample.needles:
                    c, e = score_answer_detailed(model_answer, n.answer)
                    needles_found += int(c)
                    exact_needles += int(e)
                needles_total = len(sample.needles)
                needle_accuracy = needles_found / needles_total if needles_total > 0 else 0.0
                # Detect distractor confusions
                distractor_confusions = 0
                distractors = getattr(sample, "distractors", [])
                if distractors:
                    for d in distractors:
                        if score_answer(model_answer, d.answer):
                            distractor_confusions += 1
                # "correct" = all needles found for multi-needle
                correct = (needles_found == needles_total)
                exact_match = (exact_needles == needles_total)
                expected = "; ".join(n.answer for n in sample.needles)
            else:
                expected = sample.needles[0].answer
                correct, exact_match = score_answer_detailed(model_answer, expected)
                needles_found = 1 if correct else 0
                needles_total = 1
                needle_accuracy = float(correct)
                distractor_confusions = 0

        return EvalSampleResult(
            sample_id=sample.sample_id,
            mode=self._mode,
            needle_answer=expected,
            model_answer=model_answer,
            correct=correct,
            exact_match=exact_match,
            retrieval=retrieval_metrics_dict,
            context_chars=len(context_text),
            num_retrieved=num_retrieved,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=gen_time,
            bytes_fetched=bytes_fetched,
            compression_ratio=compression_ratio,
            task_type=task_type,
            needles_found=needles_found,
            needles_total=needles_total,
            needle_accuracy=needle_accuracy,
            distractor_confusions=distractor_confusions,
        )

    def _compress_kv_blocks(self, kv_blocks: list) -> float:
        """Compress and decompress KV blocks in-place, return avg compression ratio.

        This round-trips the KV tensors through the compressor to simulate
        the quality degradation that would occur in a production system.
        The blocks are modified in-place with the reconstructed (lossy) values.

        Returns:
            Average compression ratio across all tensors.
        """
        if not self._compressor or not kv_blocks:
            return 0.0

        ratios: list[float] = []
        for block in kv_blocks:
            for i, key_tensor in enumerate(block.keys):
                compressed = self._compressor.compress(key_tensor, is_key=True)
                ratios.append(compressed.compression_ratio)
                block.keys[i] = self._compressor.decompress(compressed)
            for i, val_tensor in enumerate(block.values):
                compressed = self._compressor.compress(val_tensor, is_key=False)
                ratios.append(compressed.compression_ratio)
                block.values[i] = self._compressor.decompress(compressed)

        return sum(ratios) / len(ratios) if ratios else 0.0

    def get_profiling_report(self) -> "ProfilingReport":
        """Build and return the profiling report for the last run.

        Only meaningful after ``evaluate()`` has been called with profiling
        enabled.
        """
        from src.eval.profiler import ProfilingReport  # avoid circular at module level
        return self._profiler.report()

    def save_results(
        self,
        result: EvalRunResult,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Save evaluation results to disk.

        Saves:
        - Full results as JSON
        - Per-sample results as CSV
        - Summary as JSON
        - Profiling report as JSON (if profiler was enabled)

        Args:
            result: The EvalRunResult to save.
            output_dir: Directory to save into.

        Returns:
            Dict mapping file type to path.
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        paths: dict[str, Path] = {}

        # Full results JSON
        full_path = output_dir / f"{result.run_id}_full.json"
        save_json(result.to_dict(), full_path)
        paths["full"] = full_path

        # Per-sample CSV
        csv_path = output_dir / f"{result.run_id}_samples.csv"
        if result.sample_results:
            save_csv([s.to_dict() for s in result.sample_results], csv_path)
            paths["samples"] = csv_path

        # Summary JSON
        summary = {
            "run_id": result.run_id,
            "mode": result.mode,
            "num_samples": result.num_samples,
            "accuracy": result.accuracy,
            "retrieval_metrics": result.retrieval_metrics,
            "systems_summary": {
                "wall_time_ms": result.systems_metrics.get("wall_time_ms", 0),
                "peak_gpu_mb": result.systems_metrics.get("peak_gpu_mb", 0),
            },
        }
        summary_path = output_dir / f"{result.run_id}_summary.json"
        save_json(summary, summary_path)
        paths["summary"] = summary_path

        # Profiling report (if enabled)
        if self._profiler.enabled:
            profile_path = output_dir / f"{result.run_id}_profile.json"
            self._profiler.report().save(profile_path)
            paths["profile"] = profile_path

        logger.info(f"Results saved to {output_dir}: {list(paths.keys())}")
        return paths
