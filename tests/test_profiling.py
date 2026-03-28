"""Tests for profiling utilities."""

from __future__ import annotations

import time

import pytest

from src.utils.profiling import GPUMemoryTracker, Timer, get_gpu_memory_summary


class TestTimer:
    """Tests for the Timer context manager."""

    def test_measures_time(self) -> None:
        """Timer records elapsed time greater than zero."""
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed > 0
        assert t.elapsed < 1.0  # sanity: sleep(0.01) shouldn't take > 1s

    def test_name_stored(self) -> None:
        """Timer stores the provided name."""
        with Timer("my_operation") as t:
            pass
        assert t.name == "my_operation"

    def test_decorator_usage(self) -> None:
        """Timer works as a decorator."""
        @Timer("decorated")
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        assert result == 42


class TestGPUMemoryTracker:
    """Tests for GPU memory tracking."""

    def test_tracker_runs_without_cuda(self) -> None:
        """GPUMemoryTracker works even without CUDA (graceful no-op)."""
        with GPUMemoryTracker() as tracker:
            _ = 1 + 1
        # Should not raise; snapshot has default values
        assert tracker.snapshot.allocated_mb >= 0

    @pytest.mark.gpu
    def test_tracker_with_cuda(self) -> None:
        """GPUMemoryTracker records memory changes on GPU."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with GPUMemoryTracker() as tracker:
            _ = torch.randn(1000, 1000, device="cuda")
        assert tracker.snapshot.allocated_mb > 0


class TestGetGPUMemorySummary:
    """Tests for get_gpu_memory_summary."""

    def test_returns_dict(self) -> None:
        """Returns a dict with expected keys regardless of CUDA availability."""
        summary = get_gpu_memory_summary()
        assert "allocated_mb" in summary
        assert "reserved_mb" in summary
        assert "max_allocated_mb" in summary
