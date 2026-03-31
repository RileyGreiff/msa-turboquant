"""Plotting functions for scale sweep results.

Generates summary visualizations from SweepResult data, including:
- Accuracy vs bank size grouped by mode
- Latency vs bank size
- Compression ratio comparison bar chart
- Multi-metric dashboard
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.plotting import save_figure, setup_matplotlib


def plot_accuracy_vs_bank_size(
    records: list[dict[str, Any]],
    save_path: Path | str,
    title: str = "Accuracy vs Memory Bank Size",
) -> None:
    """Plot accuracy (y) vs bank_size (x) grouped by mode.

    Args:
        records: List of flat dicts from SweepResult.averaged_summary().
        save_path: Path to save the figure.
        title: Plot title.
    """
    setup_matplotlib()
    fig, ax = plt.subplots()

    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        label = r.get("mode", "unknown")
        if r.get("compression_method", "none") != "none":
            label += f" + {r['compression_method']}"
        groups[label].append((r["bank_size"], r.get("accuracy", 0)))

    for label, points in sorted(groups.items()):
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("Memory Bank Size (blocks)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, save_path)


def plot_latency_vs_bank_size(
    records: list[dict[str, Any]],
    save_path: Path | str,
    title: str = "Latency vs Memory Bank Size",
) -> None:
    """Plot wall_time_ms (y) vs bank_size (x) grouped by mode."""
    setup_matplotlib()
    fig, ax = plt.subplots()

    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        label = r.get("mode", "unknown")
        if r.get("compression_method", "none") != "none":
            label += f" + {r['compression_method']}"
        groups[label].append((r["bank_size"], r.get("wall_time_ms", 0)))

    for label, points in sorted(groups.items()):
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="s", label=label)

    ax.set_xlabel("Memory Bank Size (blocks)")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, save_path)


def plot_compression_comparison(
    records: list[dict[str, Any]],
    save_path: Path | str,
    metric: str = "accuracy",
    title: str | None = None,
) -> None:
    """Bar chart comparing a metric across compression methods.

    Averages across all bank sizes for each (mode, compression) pair.

    Args:
        records: Flat dicts from averaged_summary().
        save_path: Path to save.
        metric: Which metric to compare.
        title: Plot title.
    """
    setup_matplotlib()
    fig, ax = plt.subplots()

    # Group by compression_method, average the metric
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        comp = r.get("compression_method", "none")
        mode = r.get("mode", "unknown")
        label = f"{mode}\n{comp}"
        groups[label].append(r.get(metric, 0))

    labels = sorted(groups.keys())
    means = [sum(groups[l]) / len(groups[l]) for l in labels]

    bars = ax.bar(range(len(labels)), means, color=plt.cm.Set2.colors[:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by Mode + Compression")

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    save_figure(fig, save_path)


def plot_recall_vs_top_k(
    records: list[dict[str, Any]],
    save_path: Path | str,
    title: str = "Recall@k vs Top-k",
) -> None:
    """Plot recall_at_k (y) vs top_k (x) grouped by mode."""
    setup_matplotlib()
    fig, ax = plt.subplots()

    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        if r.get("mode") == "dense":
            continue
        label = r.get("mode", "unknown")
        groups[label].append((r.get("top_k", 0), r.get("recall_at_k", 0)))

    for label, points in sorted(groups.items()):
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="^", label=label)

    ax.set_xlabel("Top-k")
    ax.set_ylabel("Recall@k")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, save_path)


def plot_accuracy_vs_context_size(
    records: list[dict[str, Any]],
    save_path: Path | str,
    tokens_per_block: int = 125,
    title: str = "Accuracy vs Total Context Size",
) -> None:
    """Plot accuracy vs total token count, one line per compression method.

    This is the headline chart showing how much context each method supports.

    Args:
        records: Flat dicts from averaged_summary().
        save_path: Path to save.
        tokens_per_block: Approximate tokens per block for x-axis conversion.
        title: Plot title.
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        if r.get("mode") == "dense":
            continue
        comp = r.get("compression_method", "none")
        mode = r.get("mode", "unknown")
        # Group by compression method (across modes)
        label = comp if comp != "none" else "FP16 (no compression)"
        total_tokens = r["bank_size"] * tokens_per_block
        groups[label].append((total_tokens, r.get("accuracy", 0)))

    colors = {"FP16 (no compression)": "#1f77b4", "int8": "#ff7f0e",
              "int4": "#2ca02c", "turboquant_mse": "#d62728"}
    for label, points in sorted(groups.items()):
        # Average accuracy at each token count
        by_size: dict[int, list[float]] = defaultdict(list)
        for tokens, acc in points:
            by_size[tokens].append(acc)
        xs = sorted(by_size.keys())
        ys = [sum(by_size[x]) / len(by_size[x]) for x in xs]
        color = colors.get(label, None)
        ax.plot(xs, ys, marker="o", label=label, linewidth=2, color=color)

    ax.set_xlabel("Total Context Size (tokens)")
    ax.set_ylabel("NIAH Accuracy")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, save_path)


def plot_max_context_comparison(
    records: list[dict[str, Any]],
    save_path: Path | str,
    accuracy_threshold: float = 0.8,
    tokens_per_block: int = 125,
    title: str = "Maximum Viable Context per Compression Method",
) -> None:
    """Bar chart: max context length per compression method at a given accuracy threshold.

    Args:
        records: Flat dicts from averaged_summary().
        save_path: Path to save.
        accuracy_threshold: Minimum accuracy to consider "viable".
        tokens_per_block: Approximate tokens per block.
        title: Plot title.
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Find max bank_size where accuracy >= threshold for each compression method
    method_max: dict[str, int] = {}
    for r in records:
        if r.get("mode") == "dense":
            continue
        comp = r.get("compression_method", "none")
        label = comp if comp != "none" else "FP16"
        acc = r.get("accuracy", 0)
        bank_size = r["bank_size"]
        if acc >= accuracy_threshold:
            total_tokens = bank_size * tokens_per_block
            method_max[label] = max(method_max.get(label, 0), total_tokens)

    if not method_max:
        # If no method meets threshold, show all with their best accuracy
        plt.close(fig)
        return

    labels = sorted(method_max.keys())
    values = [method_max[l] for l in labels]
    colors = {"FP16": "#1f77b4", "int8": "#ff7f0e", "int4": "#2ca02c", "turboquant_mse": "#d62728"}
    bar_colors = [colors.get(l, "#888888") for l in labels]

    bars = ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Max Context (tokens)")
    ax.set_title(f"{title}\n(accuracy >= {accuracy_threshold:.0%})")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")
    save_figure(fig, save_path)


def plot_recall_vs_bank_size(
    records: list[dict[str, Any]],
    save_path: Path | str,
    title: str = "Retrieval Recall@k vs Bank Size",
) -> None:
    """Plot recall_at_k vs bank_size, grouped by top_k value.

    Shows how retrieval quality degrades as the haystack grows.
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        if r.get("mode") == "dense" or r.get("recall_at_k", 0) == 0:
            continue
        top_k = r.get("top_k", 0)
        groups[top_k].append((r["bank_size"], r.get("recall_at_k", 0)))

    for top_k, points in sorted(groups.items()):
        # Average recall at each bank_size
        by_size: dict[int, list[float]] = defaultdict(list)
        for bs, recall in points:
            by_size[bs].append(recall)
        xs = sorted(by_size.keys())
        ys = [sum(by_size[x]) / len(by_size[x]) for x in xs]
        ax.plot(xs, ys, marker="^", label=f"top_k={top_k}", linewidth=2)

    ax.set_xlabel("Memory Bank Size (blocks)")
    ax.set_ylabel("Recall@k")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, save_path)


def plot_compression_error_at_scale(
    records: list[dict[str, Any]],
    save_path: Path | str,
    title: str = "Compression Quality at Scale: INT4 vs TurboQuant",
) -> None:
    """Compare accuracy of INT4 vs TurboQuant across bank sizes.

    Shows whether TurboQuant's advantage grows at larger scale.
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    target_methods = {"int4", "turboquant_mse"}
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        comp = r.get("compression_method", "none")
        if comp not in target_methods:
            continue
        groups[comp].append((r["bank_size"], r.get("accuracy", 0)))

    colors = {"int4": "#2ca02c", "turboquant_mse": "#d62728"}
    for label, points in sorted(groups.items()):
        by_size: dict[int, list[float]] = defaultdict(list)
        for bs, acc in points:
            by_size[bs].append(acc)
        xs = sorted(by_size.keys())
        ys = [sum(by_size[x]) / len(by_size[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=label, linewidth=2, color=colors.get(label))

    ax.set_xlabel("Memory Bank Size (blocks)")
    ax.set_ylabel("NIAH Accuracy")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, save_path)


def generate_sweep_plots(
    records: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Generate all standard sweep plots and return their paths.

    Args:
        records: Flat dicts from SweepResult.averaged_summary().
        output_dir: Directory to save plots.

    Returns:
        List of paths to generated plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not records:
        return paths

    p = output_dir / "accuracy_vs_bank_size.png"
    plot_accuracy_vs_bank_size(records, p)
    paths.append(p)

    p = output_dir / "latency_vs_bank_size.png"
    plot_latency_vs_bank_size(records, p)
    paths.append(p)

    p = output_dir / "compression_comparison.png"
    plot_compression_comparison(records, p, metric="accuracy")
    paths.append(p)

    # Only plot recall if there are sparse modes with recall data
    has_recall = any(r.get("recall_at_k", 0) > 0 for r in records)
    if has_recall:
        p = output_dir / "recall_vs_top_k.png"
        plot_recall_vs_top_k(records, p)
        paths.append(p)

    # Scale-specific plots (when multiple bank sizes present)
    bank_sizes = set(r.get("bank_size", 0) for r in records)
    if len(bank_sizes) >= 3:
        p = output_dir / "accuracy_vs_context_size.png"
        plot_accuracy_vs_context_size(records, p)
        paths.append(p)

        p = output_dir / "max_context_comparison.png"
        plot_max_context_comparison(records, p)
        paths.append(p)

        if has_recall:
            p = output_dir / "recall_vs_bank_size.png"
            plot_recall_vs_bank_size(records, p)
            paths.append(p)

        has_compression = any(r.get("compression_method") in ("int4", "turboquant_mse") for r in records)
        if has_compression:
            p = output_dir / "compression_error_at_scale.png"
            plot_compression_error_at_scale(records, p)
            paths.append(p)

    return paths
