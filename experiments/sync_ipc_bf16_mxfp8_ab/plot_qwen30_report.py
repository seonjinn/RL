#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


NAVY = "#192133"
BF16_COLOR = "#A6CEE3"
MXFP8_COLOR = "#33A02C"


def _metric(data: pd.DataFrame, arm: str, metric: str) -> float:
    row = data[
        (data["model"] == "qwen30") & (data["arm"] == arm) & (data["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one qwen30/{arm}/{metric} row, found {len(row)}")
    aggregate = row.iloc[0]["aggregate"]
    return float(aggregate if pd.notna(aggregate) else row.iloc[0]["mean"])


def _annotate(ax: plt.Axes, bars: list, values: list[float], unit: str) -> None:
    upper = max(values)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + upper * 0.025,
            f"{value:,.0f}{unit}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=NAVY,
            zorder=20,
        )


def plot(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(summary_path)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    colors = [BF16_COLOR, MXFP8_COLOR]
    labels = ["BF16 rollout", "MXFP8 rollout"]

    throughput_metrics = [
        ("E2E", "e2e_throughput"),
        ("Generation", "generation_throughput"),
    ]
    x = list(range(len(throughput_metrics)))
    width = 0.34
    for arm_index, arm in enumerate(("bf16", "mxfp8")):
        values = [_metric(data, arm, metric) for _, metric in throughput_metrics]
        bars = axes[0].bar(
            [position + (arm_index - 0.5) * width for position in x],
            values,
            width=width,
            color=colors[arm_index],
            edgecolor=NAVY,
            linewidth=1.6,
            label=labels[arm_index],
            zorder=10,
        )
        _annotate(axes[0], list(bars), values, "")

    axes[0].set_xticks(x, [name for name, _ in throughput_metrics])
    axes[0].set_ylabel("Throughput (tokens/s/GPU)")
    axes[0].set_title("Throughput")
    axes[0].set_ylim(0, 9000)
    axes[0].text(
        0,
        2825,
        "0.949x",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=MXFP8_COLOR,
        zorder=20,
    )
    axes[0].text(
        1,
        8125,
        "1.207x",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=MXFP8_COLOR,
        zorder=20,
    )

    latency_metrics = [
        ("Generation", "generation_time"),
        ("Logprob", "logprob_time"),
        ("Refit", "refit_total_time"),
    ]
    baseline = [_metric(data, "bf16", metric) for _, metric in latency_metrics]
    optimized = [_metric(data, "mxfp8", metric) for _, metric in latency_metrics]
    normalized = [
        value / reference for value, reference in zip(optimized, baseline, strict=True)
    ]
    bars = axes[1].bar(
        range(len(latency_metrics)),
        normalized,
        width=0.58,
        color=MXFP8_COLOR,
        edgecolor=NAVY,
        linewidth=1.6,
        zorder=10,
    )
    axes[1].axhline(1.0, color=NAVY, linestyle="--", linewidth=1.5, zorder=5)
    for bar, ratio in zip(bars, normalized, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.045,
            f"{ratio:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=NAVY,
            zorder=20,
        )
    axes[1].set_xticks(
        range(len(latency_metrics)), [name for name, _ in latency_metrics]
    )
    axes[1].set_ylabel("Time normalized to BF16")
    axes[1].set_title("Stage time (lower is better)")
    axes[1].set_ylim(0, 2.15)

    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.55, zorder=0)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_color(NAVY)
            spine.set_linewidth(1.2)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Qwen3-30B-A3B: BF16 vs routed-expert MXFP8 rollout",
        y=1.13,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.06,
        "20-step matched run on 16 GB200 GPUs; reward 0.526 -> 0.528; gen KL 0.00189 -> 0.00398",
        ha="center",
        va="top",
        fontsize=8.5,
        color=NAVY,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
