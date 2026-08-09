"""Render the FC1/FC2 runtime lookup A/B result from tracked evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EDGE_COLOR = "#192133"
METRICS = (
    ("Generation throughput", "generation_tokens_per_second_per_gpu", True),
    ("End-to-end throughput", "e2e_tokens_per_second_per_gpu", True),
    ("Total step time", "total_step_seconds", False),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _ratio_rows(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, key, higher_is_better in METRICS:
        aggregate = summary["aggregate"][key]
        rows.extend(
            (
                {
                    "Metric": label,
                    "Configuration": "Stock",
                    "Ratio": 1.0,
                    "Higher is better": higher_is_better,
                },
                {
                    "Metric": label,
                    "Configuration": "Lookup",
                    "Ratio": aggregate["candidate_over_stock"],
                    "Higher is better": higher_is_better,
                },
            )
        )
    return pd.DataFrame(rows)


def _replicate_ratios(summary: dict[str, Any], key: str) -> list[float]:
    return [run["candidate"][key] / run["stock"][key] for run in summary["runs"]]


def render(summary_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text())
    frame = _ratio_rows(summary)
    palette = sns.color_palette("Paired", n_colors=2)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharey=True)

    for ax, (label, key, higher_is_better) in zip(axes, METRICS, strict=True):
        subset = frame[frame["Metric"] == label]
        sns.barplot(
            data=subset,
            x="Configuration",
            y="Ratio",
            hue="Configuration",
            palette=palette,
            dodge=False,
            edgecolor=EDGE_COLOR,
            linewidth=2.0,
            legend=False,
            errorbar=None,
            zorder=10,
            ax=ax,
        )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1, zorder=2)
        ax.grid(True, axis="y", linestyle="--", dashes=(6, 6), linewidth=1.0, zorder=0)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0.985, 1.018)
        ax.set_ylabel("Candidate / Stock" if ax is axes[0] else "", fontsize=11)
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_linewidth(1.8)
            ax.spines[side].set_color("black")

        ratio = float(summary["aggregate"][key]["candidate_over_stock"])
        direction = 1.0 if higher_is_better else -1.0
        improvement = direction * (ratio - 1.0) * 100.0
        labels = ["1.000x", f"{ratio:.4f}x\n({improvement:+.2f}%)"]
        for x, y, text in zip((0, 1), (1.0, ratio), labels, strict=True):
            ax.annotate(
                text,
                (x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        replicate_ratios = _replicate_ratios(summary, key)
        ax.scatter(
            [1] * len(replicate_ratios),
            replicate_ratios,
            color=EDGE_COLOR,
            marker="D",
            s=22,
            zorder=20,
            label="Independent A/B repetitions",
        )

    fig.text(
        0.5,
        0.01,
        "Qwen3-30B-A3B, 16 GB200 GPUs, CUDA Graph enabled; diamonds show two independent A/B repetitions",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = _parse_args()
    render(args.summary, args.output)
