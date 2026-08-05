#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BACKEND_LABELS = {
    "flashinfer_cutedsl": "CuTeDSL",
    "flashinfer_cutlass": "CUTLASS",
    "flashinfer_trtllm": "TRTLLM",
}
BACKEND_ORDER = list(BACKEND_LABELS.values())
EDGE_COLOR = "#192133"


def _load_summary(path: Path) -> pd.DataFrame:
    raw: dict[str, dict[str, Any]] = json.loads(path.read_text())
    rows = [
        {
            "Backend": BACKEND_LABELS[backend],
            "Generation throughput": metrics["generation_tokens_per_sec_per_gpu_mean"],
            "Generation time": metrics["generation_seconds_mean"],
            "E2E throughput": metrics["e2e_tokens_per_sec_per_gpu_mean"],
        }
        for backend, metrics in raw.items()
    ]
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(
        True,
        linestyle="--",
        dashes=(6, 6),
        linewidth=1.1,
        axis="y",
        zorder=0,
    )
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_linewidth(2.0)
        ax.spines[side].set_color("black")


def _draw_bar(ax: plt.Axes, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    sns.barplot(
        data=data,
        x="Backend",
        y=metric,
        order=BACKEND_ORDER,
        hue="Backend",
        hue_order=BACKEND_ORDER,
        palette=sns.color_palette("Paired", n_colors=len(BACKEND_ORDER)),
        edgecolor=EDGE_COLOR,
        linewidth=2.0,
        legend=False,
        zorder=10,
        ax=ax,
    )
    _style_axis(ax, ylabel)


def plot_absolute(data: pd.DataFrame, output_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    _draw_bar(
        axes[0],
        data,
        "Generation throughput",
        "Generation tokens/s/GPU",
    )
    _draw_bar(axes[1], data, "Generation time", "Generation time (s, lower)")
    _draw_bar(axes[2], data, "E2E throughput", "E2E tokens/s/GPU")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            bbox_inches="tight",
            dpi=600,
        )
    plt.close(fig)


def plot_normalized(data: pd.DataFrame, output_base: Path) -> None:
    baseline = data.set_index("Backend").loc["CuTeDSL"]
    normalized = data.copy()
    normalized["Generation throughput"] /= baseline["Generation throughput"]
    normalized["Generation speed"] = (
        baseline["Generation time"] / normalized["Generation time"]
    )
    normalized["E2E throughput"] /= baseline["E2E throughput"]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    panels = (
        ("Generation throughput", "Generation throughput / CuTeDSL"),
        ("Generation speed", "Generation speed / CuTeDSL"),
        ("E2E throughput", "E2E throughput / CuTeDSL"),
    )
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        _draw_bar(ax, normalized, metric, ylabel)
        ax.axhline(y=1, linestyle="--", linewidth=1.1, color="black", zorder=2)
        ax.set_ylim(0.85, 1.05)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            bbox_inches="tight",
            dpi=600,
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    data = _load_summary(args.summary)
    plot_absolute(data, args.output_dir / "qwen30b_mxfp8_linear_backends_absolute")
    plot_normalized(
        data,
        args.output_dir / "qwen30b_mxfp8_linear_backends_normalized",
    )


if __name__ == "__main__":
    main()
