#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


WARMUP_END_STEP = 4
MEASUREMENT_END_STEP = 29
REQUIRED_STEPS = tuple(range(WARMUP_END_STEP + 1, MEASUREMENT_END_STEP + 1))
METRICS: dict[str, tuple[str, ...]] = {
    "e2e": ("timing/train/total_step_time",),
    "policy": ("timing/train/policy_training",),
    "logprob": ("timing/train/policy_and_reference_logprobs",),
    "refit": ("timing/train/prepare_for_generation/total",),
    "generation": ("timing/train/generation",),
    "tokens": ("train/total_num_tokens", "train/global_valid_toks"),
    "generation_tps": ("performance/generation_tokens_per_sec_per_gpu",),
    "acceptance": ("train/vllm/spec_acceptance_rate",),
    "draft_loss": ("train/draft_loss",),
    "draft_grad_norm": ("train/draft_grad_norm",),
    "peak_memory": (
        "train/peak_memory_allocated_mb",
        "performance/peak_memory_allocated_mb",
        "system/peak_memory_allocated_mb",
    ),
}
PAIRED_METRICS: tuple[tuple[str, str], ...] = (
    ("e2e_seconds_mean", "E2E seconds/step"),
    ("policy_seconds_mean", "policy seconds/step"),
    ("refit_seconds_mean", "refit seconds/step"),
    ("logprob_seconds_mean", "logprob seconds/step"),
    ("generation_tokens_per_second_per_gpu", "generation TPS/GPU"),
    ("acceptance_rate_mean", "acceptance rate"),
)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _value(row: Mapping[str, Any], metric: str) -> float | None:
    for key in METRICS[metric]:
        value = _numeric(row.get(key))
        if value is not None:
            return value
    return None


def _required_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values = [_value(row, metric) for row in rows]
    missing_steps = [
        int(float(row["_step"]))
        for row, value in zip(rows, values, strict=True)
        if value is None
    ]
    if missing_steps:
        raise ValueError(
            f"{metric} missing numeric values at steps {missing_steps}; "
            f"required W&B keys {METRICS[metric]}"
        )
    return [float(value) for value in values if value is not None]


def _optional_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [value for row in rows if (value := _value(row, metric)) is not None]


def _merge_by_step(rows: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in rows:
        step_value = _numeric(row.get("_step"))
        if step_value is None or not step_value.is_integer():
            continue
        step = int(step_value)
        if step not in REQUIRED_STEPS:
            continue
        merged.setdefault(step, {"_step": step}).update(row)
    return merged


def summarize_history(
    rows: Iterable[Mapping[str, Any]],
    *,
    cell: str,
    gbs: int,
    arm: str,
    replicate: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    merged = _merge_by_step(rows)
    missing_steps = [step for step in REQUIRED_STEPS if step not in merged]
    if missing_steps:
        raise ValueError(
            f"{cell} replicate {replicate} missing required steps: {missing_steps}"
        )
    measured = [merged[step] for step in REQUIRED_STEPS]
    e2e = _required_values(measured, "e2e")
    policy = _required_values(measured, "policy")
    logprob = _required_values(measured, "logprob")
    refit = _required_values(measured, "refit")
    generation = _required_values(measured, "generation")
    tokens = _required_values(measured, "tokens")
    acceptance = _required_values(measured, "acceptance")
    draft_loss = _optional_values(measured, "draft_loss")
    draft_grad_norm = _optional_values(measured, "draft_grad_norm")
    peak_memory = _optional_values(measured, "peak_memory")
    logged_generation_tps = _required_values(measured, "generation_tps")
    total_tokens = sum(tokens)
    generation_tps = sum(logged_generation_tps) / len(logged_generation_tps)
    refit_count = int(evidence.get("draft_refit_count", 0))
    update_count = int(evidence.get("draft_update_count", 0))
    positive_grad_steps = sum(value > 0 for value in draft_grad_norm)
    if arm == "online":
        update_refit_correct = (
            update_count >= len(measured) or positive_grad_steps == len(measured)
        ) and refit_count >= len(measured)
    else:
        update_refit_correct = update_count == 0 and refit_count == 0
    return {
        "cell": cell,
        "shape": cell.removesuffix(f"_{arm}"),
        "arm": arm,
        "replicate": replicate,
        "steps": len(measured),
        "included_steps": list(REQUIRED_STEPS),
        "missing_steps": missing_steps,
        "valid_counts": {
            metric: len(_optional_values(measured, metric)) for metric in METRICS
        },
        "first_step": min(int(float(row["_step"])) for row in measured),
        "last_step": max(int(float(row["_step"])) for row in measured),
        "e2e_seconds_per_sample": sum(e2e) / (len(measured) * gbs),
        "e2e_seconds_per_token": sum(e2e) / total_tokens,
        "policy_seconds_mean": sum(policy) / len(policy),
        "logprob_seconds_mean": sum(logprob) / len(logprob),
        "refit_seconds_mean": sum(refit) / len(refit),
        "e2e_seconds_mean": sum(e2e) / len(e2e),
        "generation_seconds_mean": sum(generation) / len(generation),
        "generation_tokens_per_second_per_gpu": generation_tps,
        "acceptance_rate_mean": sum(acceptance) / len(acceptance),
        "peak_memory_allocated_mb": max(peak_memory) if peak_memory else None,
        "draft_loss_mean": (sum(draft_loss) / len(draft_loss) if draft_loss else None),
        "draft_grad_norm_positive_steps": positive_grad_steps,
        "draft_update_count": update_count,
        "draft_refit_count": refit_count,
        "update_refit_correct": update_refit_correct,
    }


def compare_pair(fixed: Mapping[str, Any], online: Mapping[str, Any]) -> dict[str, Any]:
    if fixed["replicate"] != online["replicate"]:
        raise ValueError("Fixed and online summaries must have the same replicate")
    fixed_value = float(fixed["e2e_seconds_per_token"])
    online_value = float(online["e2e_seconds_per_token"])
    paired_metrics = {
        metric: {
            "fixed": float(fixed[metric]),
            "online": float(online[metric]),
            "delta": float(online[metric]) - float(fixed[metric]),
        }
        for metric, _label in PAIRED_METRICS
    }
    return {
        "shape": str(fixed["cell"]).removesuffix("_fixed"),
        "replicate": int(fixed["replicate"]),
        "fixed_e2e_seconds_per_token": fixed_value,
        "online_e2e_seconds_per_token": online_value,
        "paired_delta_e2e_seconds_per_token": online_value - fixed_value,
        "online_overhead_percent": 100 * (online_value / fixed_value - 1),
        "paired_metrics": paired_metrics,
    }


def aggregate_pairs(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(comparisons) != 3:
        raise ValueError(f"Expected 3 paired replicates, received {len(comparisons)}")
    shapes = {str(item["shape"]) for item in comparisons}
    replicates = sorted(int(item["replicate"]) for item in comparisons)
    if len(shapes) != 1 or replicates != [1, 2, 3]:
        raise ValueError(f"Expected one shape with replicates 1, 2, 3: {comparisons}")
    ordered = sorted(comparisons, key=lambda item: int(item["replicate"]))
    deltas = [float(item["paired_delta_e2e_seconds_per_token"]) for item in ordered]
    overheads = [float(item["online_overhead_percent"]) for item in ordered]
    mean_delta = statistics.mean(deltas)
    stdev_delta = statistics.stdev(deltas)
    margin = 4.303 * stdev_delta / math.sqrt(len(deltas))
    metric_statistics: dict[str, dict[str, Any]] = {}
    for metric, _label in PAIRED_METRICS:
        metric_deltas = [
            float(item["paired_metrics"][metric]["delta"]) for item in ordered
        ]
        metric_mean = statistics.mean(metric_deltas)
        metric_stdev = statistics.stdev(metric_deltas)
        metric_margin = 4.303 * metric_stdev / math.sqrt(len(metric_deltas))
        metric_statistics[metric] = {
            "paired_deltas": metric_deltas,
            "mean": metric_mean,
            "sample_stdev": metric_stdev,
            "95pct_ci": [metric_mean - metric_margin, metric_mean + metric_margin],
        }
    return {
        "shape": shapes.pop(),
        "replicates": replicates,
        "paired_deltas_e2e_seconds_per_token": deltas,
        "paired_delta_mean": mean_delta,
        "paired_delta_sample_stdev": stdev_delta,
        "paired_delta_95pct_ci": [mean_delta - margin, mean_delta + margin],
        "online_overhead_percent_values": overheads,
        "online_overhead_percent_mean": statistics.mean(overheads),
        "online_overhead_percent_sample_stdev": statistics.stdev(overheads),
        "metric_statistics": metric_statistics,
    }


def _runtime_contract() -> ModuleType:
    path = Path(__file__).with_name("runtime_contract.py")
    spec = importlib.util.spec_from_file_location("matrix_runtime_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runtime contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_history(run_path: str) -> list[dict[str, Any]]:
    import wandb

    run = wandb.Api(timeout=120).run(run_path)
    return [
        dict(row)
        for row in run.scan_history(
            min_step=REQUIRED_STEPS[0],
            max_step=REQUIRED_STEPS[-1] + 1,
            page_size=1000,
        )
    ]


def _write_reports(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "summaries": summaries,
        "comparisons": comparisons,
        "aggregates": aggregates,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Qwen3-8B DFlash refit performance matrix",
        "",
        "Steps 0-4 are excluded. Every run includes exact 25/25 required W&B steps 5-29. Time-per-sample/token uses sums, not means of ratios.",
        "",
        "| Cell | Rep | E2E s/token | E2E s/step | Policy s | Refit s | Logprob s | Gen tok/s/GPU | Acceptance | Peak MiB | Correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in summaries:
        peak = item["peak_memory_allocated_mb"]
        lines.append(
            f"| {item['cell']} | {item['replicate']} | {item['e2e_seconds_per_token']:.6f} | "
            f"{item['e2e_seconds_mean']:.3f} | {item['policy_seconds_mean']:.3f} | "
            f"{item['refit_seconds_mean']:.3f} | {item['logprob_seconds_mean']:.3f} | "
            f"{item['generation_tokens_per_second_per_gpu']:.2f} | "
            f"{item['acceptance_rate_mean']:.4f} | "
            f"{peak:.1f} | {item['update_refit_correct']} |"
            if peak is not None
            else f"| {item['cell']} | {item['replicate']} | {item['e2e_seconds_per_token']:.6f} | "
            f"{item['e2e_seconds_mean']:.3f} | {item['policy_seconds_mean']:.3f} | "
            f"{item['refit_seconds_mean']:.3f} | {item['logprob_seconds_mean']:.3f} | "
            f"{item['generation_tokens_per_second_per_gpu']:.2f} | "
            f"{item['acceptance_rate_mean']:.4f} | n/a | "
            f"{item['update_refit_correct']} |"
        )
    lines.extend(
        [
            "",
            "## Paired replicates",
            "",
            "| Shape | Rep | Paired delta E2E s/token | Online overhead % |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['shape']} | {item['replicate']} | "
            f"{item['paired_delta_e2e_seconds_per_token']:.6f} | "
            f"{item['online_overhead_percent']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired replicate metrics",
            "",
            "| Shape | Rep | Metric | Fixed | Online | Online - fixed |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        for metric, label in PAIRED_METRICS:
            values = item["paired_metrics"][metric]
            lines.append(
                f"| {item['shape']} | {item['replicate']} | {label} | "
                f"{values['fixed']:.6f} | {values['online']:.6f} | "
                f"{values['delta']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Paired delta mean ± sample stdev and 95% CI",
            "",
            "| Shape | Mean E2E delta | Sample stdev | 95% CI | Mean overhead % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in aggregates:
        ci_low, ci_high = item["paired_delta_95pct_ci"]
        lines.append(
            f"| {item['shape']} | {item['paired_delta_mean']:.6f} | "
            f"{item['paired_delta_sample_stdev']:.6f} | "
            f"[{ci_low:.6f}, {ci_high:.6f}] | "
            f"{item['online_overhead_percent_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired metric delta statistics",
            "",
            "| Shape | Metric | Deltas (r1, r2, r3) | Mean | Sample stdev | 95% CI |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in aggregates:
        for metric, label in PAIRED_METRICS:
            stats = item["metric_statistics"][metric]
            deltas = ", ".join(f"{value:.6f}" for value in stats["paired_deltas"])
            ci_low, ci_high = stats["95pct_ci"]
            lines.append(
                f"| {item['shape']} | {label} | {deltas} | "
                f"{stats['mean']:.6f} | {stats['sample_stdev']:.6f} | "
                f"[{ci_low:.6f}, {ci_high:.6f}] |"
            )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    contract = _runtime_contract()
    summaries: list[dict[str, Any]] = []
    for entry in manifest["runs"]:
        cell = contract.resolve_cell(entry["cell"])
        evidence_path = Path(entry["evidence"])
        summaries.append(
            summarize_history(
                _load_history(entry["run_path"]),
                cell=cell.name,
                gbs=cell.gbs,
                arm=cell.arm,
                replicate=int(entry["replicate"]),
                evidence=json.loads(evidence_path.read_text()),
            )
        )
    by_cell = {(item["cell"], item["replicate"]): item for item in summaries}
    comparisons = [
        compare_pair(
            by_cell[(f"{shape}_fixed", replicate)],
            by_cell[(f"{shape}_online", replicate)],
        )
        for shape in ("gbs32_mbs1", "gbs64_mbs1", "gbs64_mbs2")
        for replicate in (1, 2, 3)
    ]
    aggregates = [
        aggregate_pairs([item for item in comparisons if item["shape"] == shape])
        for shape in ("gbs32_mbs1", "gbs64_mbs1", "gbs64_mbs2")
    ]
    _write_reports(args.output_dir, summaries, comparisons, aggregates)


if __name__ == "__main__":
    main()
