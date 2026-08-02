#!/usr/bin/env python3
"""Aggregate and plot matched Qwen3-30B-A3B NeMo-RL A/B runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


def _arm_by_name(payload: dict[str, Any], arm: str) -> dict[str, Any]:
    matches = [run for run in payload["runs"] if run.get("arm") == arm]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {arm} run")
    return matches[0]


def _validate_run(run: dict[str, Any]) -> None:
    if run.get("complete") is not True:
        raise ValueError(f"incomplete {run.get('arm')} run")
    if run.get("measurement_scope") != "generation_calls":
        raise ValueError("expected direct generation-call timing")
    if int(run.get("generation_calls", 0)) != 1:
        raise ValueError("expected exactly one generation call")
    if float(run.get("tokens_per_second_per_gpu", 0.0)) <= 0.0:
        raise ValueError("invalid tokens/sec/GPU")


def aggregate_summaries(paths: Sequence[Path]) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = _arm_by_name(payload, "baseline")
        adaptive = _arm_by_name(payload, "adaptive")
        _validate_run(baseline)
        _validate_run(adaptive)
        if int(baseline["output_tokens"]) != int(adaptive["output_tokens"]):
            raise ValueError(f"output token mismatch in {path}")
        if int(baseline["gpu_count"]) != int(adaptive["gpu_count"]):
            raise ValueError(f"GPU count mismatch in {path}")
        baseline_tps = float(baseline["tokens_per_second_per_gpu"])
        adaptive_tps = float(adaptive["tokens_per_second_per_gpu"])
        repeats.append(
            {
                "repeat": index,
                "source": str(path),
                "output_tokens": int(baseline["output_tokens"]),
                "gpu_count": int(baseline["gpu_count"]),
                "baseline_generation_seconds": float(
                    baseline["generation_seconds"]
                ),
                "adaptive_generation_seconds": float(
                    adaptive["generation_seconds"]
                ),
                "baseline_tokens_per_second_per_gpu": baseline_tps,
                "adaptive_tokens_per_second_per_gpu": adaptive_tps,
                "paired_speedup": adaptive_tps / baseline_tps,
            }
        )
    if not repeats:
        raise ValueError("at least one summary is required")
    return {
        "repeats": repeats,
        "median": {
            "baseline_tokens_per_second_per_gpu": statistics.median(
                row["baseline_tokens_per_second_per_gpu"] for row in repeats
            ),
            "adaptive_tokens_per_second_per_gpu": statistics.median(
                row["adaptive_tokens_per_second_per_gpu"] for row in repeats
            ),
            "paired_speedup": statistics.median(
                row["paired_speedup"] for row in repeats
            ),
        },
    }


def _style_axis(axis: Any, *, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel, fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
    axis.tick_params(axis="x", labelsize=12)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(
        True,
        linestyle="--",
        dashes=(6, 6),
        linewidth=1.1,
        axis="y",
        zorder=0,
    )
    for side in ("left", "right", "top", "bottom"):
        axis.spines[side].set_linewidth(2.0)
        axis.spines[side].set_color("black")


def _save_grouped_plot(
    rows: list[dict[str, object]],
    *,
    output_base: Path,
    ylabel: str,
    baseline_line: bool,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    palette = sns.color_palette("Paired", n_colors=2)
    sns.barplot(
        data=frame,
        x="Group",
        y="Value",
        hue="Config",
        order=["Repeat 1", "Repeat 2", "Repeat 3", "Median"],
        hue_order=["MXFP8 baseline", "Adaptive selection"],
        palette=palette,
        edgecolor="#192133",
        linewidth=2.0,
        zorder=10,
        ax=axis,
    )
    if baseline_line:
        axis.axhline(y=1.0, linestyle="--", linewidth=1.1, color="black", zorder=5)
        axis.set_ylim(0.98, 1.005)
    _style_axis(axis, xlabel="Run", ylabel=ylabel)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend().remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    repeats = result["repeats"]
    median = result["median"]
    absolute_rows: list[dict[str, object]] = []
    relative_rows: list[dict[str, object]] = []
    for row in repeats:
        group = f"Repeat {row['repeat']}"
        absolute_rows.extend(
            [
                {
                    "Group": group,
                    "Config": "MXFP8 baseline",
                    "Value": row["baseline_tokens_per_second_per_gpu"],
                },
                {
                    "Group": group,
                    "Config": "Adaptive selection",
                    "Value": row["adaptive_tokens_per_second_per_gpu"],
                },
            ]
        )
        relative_rows.extend(
            [
                {"Group": group, "Config": "MXFP8 baseline", "Value": 1.0},
                {
                    "Group": group,
                    "Config": "Adaptive selection",
                    "Value": row["paired_speedup"],
                },
            ]
        )
    absolute_rows.extend(
        [
            {
                "Group": "Median",
                "Config": "MXFP8 baseline",
                "Value": median["baseline_tokens_per_second_per_gpu"],
            },
            {
                "Group": "Median",
                "Config": "Adaptive selection",
                "Value": median["adaptive_tokens_per_second_per_gpu"],
            },
        ]
    )
    relative_rows.extend(
        [
            {"Group": "Median", "Config": "MXFP8 baseline", "Value": 1.0},
            {
                "Group": "Median",
                "Config": "Adaptive selection",
                "Value": median["paired_speedup"],
            },
        ]
    )
    _save_grouped_plot(
        absolute_rows,
        output_base=output_dir / "qwen30_nemorl_ab_tokens_per_second_per_gpu",
        ylabel="Throughput (tokens/sec/GPU)",
        baseline_line=False,
    )
    _save_grouped_plot(
        relative_rows,
        output_base=output_dir / "qwen30_nemorl_ab_relative_throughput",
        ylabel="Throughput / MXFP8 baseline",
        baseline_line=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_outputs(aggregate_summaries(args.summary), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
