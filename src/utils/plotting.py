"""Plotting utilities for experiment results visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def setup_matplotlib() -> None:
    """Configure matplotlib defaults for publication-quality charts."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_context_length(
    results: list[dict[str, Any]],
    metric_key: str,
    save_path: Path | str,
    title: str | None = None,
    ylabel: str | None = None,
    group_by: str = "method",
) -> None:
    """Plot a metric (y-axis) vs context length (x-axis), grouped by method.

    Args:
        results: List of dicts, each with 'context_length', metric_key, and group_by key.
        metric_key: Key in results dicts for the y-axis value.
        save_path: Where to save the figure.
        title: Plot title (defaults to metric_key).
        ylabel: Y-axis label (defaults to metric_key).
        group_by: Key to group lines by (e.g., "method", "compression").
    """
    setup_matplotlib()
    fig, ax = plt.subplots()

    # Group results
    groups: dict[str, list[tuple[int, float]]] = {}
    for r in results:
        group = r.get(group_by, "default")
        ctx_len = r.get("context_length", 0)
        val = r.get(metric_key, 0)
        groups.setdefault(group, []).append((ctx_len, val))

    for group_name, points in sorted(groups.items()):
        points.sort(key=lambda x: x[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=group_name)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or f"{metric_key} vs Context Length")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    save_figure(fig, save_path)


def plot_latency_vs_context(
    results: list[dict[str, Any]],
    save_path: Path | str,
) -> None:
    """Plot latency (ms) vs context length."""
    plot_metric_vs_context_length(
        results, "latency_ms", save_path,
        title="Latency vs Context Length",
        ylabel="Latency (ms)",
    )


def plot_memory_usage(
    results: list[dict[str, Any]],
    save_path: Path | str,
) -> None:
    """Plot peak VRAM usage vs context length."""
    plot_metric_vs_context_length(
        results, "peak_vram_mb", save_path,
        title="Peak VRAM vs Context Length",
        ylabel="Peak VRAM (MB)",
    )


def plot_accuracy_heatmap(
    results: list[dict[str, Any]],
    save_path: Path | str,
    x_key: str = "context_length",
    y_key: str = "needle_depth",
    value_key: str = "accuracy",
) -> None:
    """Plot accuracy as a heatmap (e.g., needle depth vs context length).

    Args:
        results: List of dicts containing x_key, y_key, and value_key.
        save_path: Where to save the figure.
        x_key: Key for x-axis bins.
        y_key: Key for y-axis bins.
        value_key: Key for cell values.
    """
    setup_matplotlib()

    if not results:
        return

    # Build grid data
    x_vals = sorted(set(r[x_key] for r in results))
    y_vals = sorted(set(r[y_key] for r in results))
    grid = [[0.0] * len(x_vals) for _ in range(len(y_vals))]

    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    for r in results:
        xi = x_idx[r[x_key]]
        yi = y_idx[r[y_key]]
        grid[yi][xi] = r.get(value_key, 0.0)

    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v/1000:.0f}K" if isinstance(v, (int, float)) and v > 100 else str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.set_title(f"{value_key.replace('_', ' ').title()} Heatmap")
    fig.colorbar(im, ax=ax)

    save_figure(fig, save_path)
