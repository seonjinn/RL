#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = {
    "e2e_throughput": "performance/tokens_per_sec_per_gpu",
    "generation_throughput": "performance/generation_tokens_per_sec_per_gpu",
    "total_step_time": "timing/train/total_step_time",
    "generation_time": "timing/train/generation",
    "logprob_time": "timing/train/policy_and_reference_logprobs",
    "training_time": "timing/train/policy_training",
    "refit_total_time": "timing/train/prepare_for_generation/total",
    "refit_transfer_time": "timing/train/prepare_for_generation/transfer_and_update_weights",
    "total_tokens": "train/total_num_tokens",
    "mean_generation_tokens": "train/mean_gen_tokens_per_sample",
    "reward": "train/reward",
    "generation_kl": "train/gen_kl_error",
    "policy_kl": "train/policy_kl_error",
}

THROUGHPUT_METRICS = {"e2e_throughput", "generation_throughput"}
LATENCY_METRICS = {
    "total_step_time",
    "generation_time",
    "logprob_time",
    "training_time",
    "refit_total_time",
    "refit_transfer_time",
}


@dataclass(frozen=True)
class Summary:
    model: str
    arm: str
    metric: str
    mean: float
    median: float
    stdev: float
    count: int
    aggregate: float | None = None


def select_steps(values: dict[str, float], first_step: int = 2) -> list[float]:
    return [
        float(value)
        for step, value in sorted(values.items(), key=lambda item: int(item[0]))
        if int(step) >= first_step
    ]


def throughput_speedup(bf16: float, mxfp8: float) -> float:
    return mxfp8 / bf16


def latency_speedup(bf16: float, mxfp8: float) -> float:
    return bf16 / mxfp8


def aggregate_throughput(
    tokens: dict[str, float],
    seconds: dict[str, float],
    gpu_count: int,
    first_step: int = 2,
) -> float:
    steps = sorted(
        {
            int(step)
            for step in tokens.keys() & seconds.keys()
            if int(step) >= first_step
        }
    )
    total_tokens = sum(float(tokens[str(step)]) for step in steps)
    total_seconds = sum(float(seconds[str(step)]) for step in steps)
    return total_tokens / total_seconds / gpu_count


def read_gpu_count(metadata_path: Path) -> int:
    values = dict(
        line.split("=", maxsplit=1)
        for line in metadata_path.read_text().splitlines()
        if "=" in line
    )
    return int(values["nodes"]) * int(values["gpus_per_node"])


def summarize_run(
    metrics_path: Path,
    metadata_path: Path,
    model: str,
    arm: str,
    first_step: int,
) -> list[Summary]:
    payload: dict[str, Any] = json.loads(metrics_path.read_text())
    gpu_count = read_gpu_count(metadata_path)
    summaries: list[Summary] = []
    for name, key in METRICS.items():
        raw_values = payload.get(key)
        if not isinstance(raw_values, dict):
            continue
        values = select_steps(raw_values, first_step)
        if not values:
            continue
        aggregate = None
        if name == "e2e_throughput":
            aggregate = aggregate_throughput(
                payload["train/total_num_tokens"],
                payload["timing/train/total_step_time"],
                gpu_count,
                first_step,
            )
        elif name == "generation_throughput":
            aggregate = aggregate_throughput(
                payload["train/total_num_tokens"],
                payload["timing/train/generation"],
                gpu_count,
                first_step,
            )
        summaries.append(
            Summary(
                model=model,
                arm=arm,
                metric=name,
                mean=statistics.fmean(values),
                median=statistics.median(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                count=len(values),
                aggregate=aggregate,
            )
        )
    return summaries


def find_runs(results_root: Path, first_step: int) -> list[Summary]:
    summaries: list[Summary] = []
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for arm in ("bf16", "mxfp8"):
            metrics_path = model_dir / arm / "metrics.json"
            metadata_path = model_dir / arm / "metadata.env"
            if metrics_path.exists() and metadata_path.exists():
                summaries.extend(
                    summarize_run(
                        metrics_path,
                        metadata_path,
                        model_dir.name,
                        arm,
                        first_step,
                    )
                )
    return summaries


def write_summary_csv(summaries: list[Summary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            ("model", "arm", "metric", "mean", "median", "stdev", "count", "aggregate")
        )
        for item in summaries:
            writer.writerow(
                (
                    item.model,
                    item.arm,
                    item.metric,
                    f"{item.mean:.9f}",
                    f"{item.median:.9f}",
                    f"{item.stdev:.9f}",
                    item.count,
                    "" if item.aggregate is None else f"{item.aggregate:.9f}",
                )
            )


def comparison_rows(summaries: list[Summary]) -> list[dict[str, float | str]]:
    by_key = {(item.model, item.arm, item.metric): item for item in summaries}
    rows: list[dict[str, float | str]] = []
    models = sorted({item.model for item in summaries})
    for model in models:
        for metric in sorted(THROUGHPUT_METRICS | LATENCY_METRICS):
            bf16 = by_key.get((model, "bf16", metric))
            mxfp8 = by_key.get((model, "mxfp8", metric))
            if bf16 is None or mxfp8 is None:
                continue
            bf16_value = bf16.aggregate if bf16.aggregate is not None else bf16.mean
            mxfp8_value = mxfp8.aggregate if mxfp8.aggregate is not None else mxfp8.mean
            speedup = (
                throughput_speedup(bf16_value, mxfp8_value)
                if metric in THROUGHPUT_METRICS
                else latency_speedup(bf16_value, mxfp8_value)
            )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bf16_mean": bf16_value,
                    "mxfp8_mean": mxfp8_value,
                    "mxfp8_speedup": speedup,
                    "mxfp8_change_percent": (speedup - 1.0) * 100.0,
                }
            )
    return rows


def write_comparison_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "model",
        "metric",
        "bf16_mean",
        "mxfp8_mean",
        "mxfp8_speedup",
        "mxfp8_change_percent",
    )
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_speedups(rows: list[dict[str, float | str]], output_base: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    selected = {
        "e2e_throughput": "E2E throughput",
        "generation_throughput": "Generation throughput",
        "total_step_time": "Step time",
        "refit_transfer_time": "Refit transfer",
    }
    frame = pd.DataFrame([row for row in rows if row["metric"] in selected])
    if frame.empty:
        return
    frame["Metric"] = frame["metric"].map(selected)
    frame["Model"] = (
        frame["model"]
        .map({"qwen30": "Qwen3-30B-A3B", "nano": "Nemotron3 Nano"})
        .fillna(frame["model"])
    )

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    models = list(frame["Model"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False
    )
    metric_order = list(selected.values())
    for axis, model in zip(axes[0], models, strict=True):
        subset = frame[frame["Model"] == model]
        sns.barplot(
            data=subset,
            x="Metric",
            y="mxfp8_speedup",
            order=metric_order,
            color=sns.color_palette("Paired", n_colors=2)[1],
            edgecolor="#192133",
            linewidth=2.0,
            errorbar=None,
            zorder=10,
            ax=axis,
        )
        axis.axhline(y=1, linestyle="--", linewidth=1.1, color="black", zorder=2)
        axis.set_title(model, fontsize=14)
        axis.set_xlabel("")
        axis.set_ylabel("MXFP8 speedup over BF16", fontsize=13)
        axis.tick_params(axis="x", labelrotation=20, labelsize=10)
        axis.tick_params(axis="y", labelsize=11)
        axis.grid(
            True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0
        )
        for side in ("left", "right", "top", "bottom"):
            axis.spines[side].set_linewidth(2.0)
            axis.spines[side].set_color("black")
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"), bbox_inches="tight", dpi=600
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = find_runs(args.results_root, args.first_step)
    rows = comparison_rows(summaries)
    write_summary_csv(summaries, args.report_root / "summary.csv")
    write_comparison_csv(rows, args.report_root / "comparison.csv")
    plot_speedups(rows, args.report_root / "plots" / "sync_ipc_mxfp8_speedup")


if __name__ == "__main__":
    main()
