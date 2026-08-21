#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


WARMUP_END_STEP = 4
MEASUREMENT_END_STEP = 49
METRICS: dict[str, tuple[str, ...]] = {
    "e2e": ("timing/train/total_step_time",),
    "policy": ("timing/train/policy_training",),
    "refit": (
        "timing/train/weight_sync",
        "timing/train/prepare_for_generation/total",
    ),
    "generation": ("timing/train/generation",),
    "tokens": ("train/total_num_tokens", "train/global_valid_toks"),
    "generation_tps": ("performance/generation_tokens_per_sec",),
    "acceptance": ("train/vllm/spec_acceptance_rate",),
    "draft_loss": ("train/draft_loss",),
    "draft_grad_norm": ("train/draft_grad_norm",),
    "peak_memory": (
        "train/peak_memory_allocated_mb",
        "performance/peak_memory_allocated_mb",
        "system/peak_memory_allocated_mb",
    ),
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _value(row: Mapping[str, Any], metric: str) -> float | None:
    for key in METRICS[metric]:
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def _required_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values = [_value(row, metric) for row in rows]
    if any(value is None for value in values):
        raise ValueError(
            f"Missing finite {metric} metric; tried aliases {METRICS[metric]}"
        )
    return [float(value) for value in values if value is not None]


def _optional_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [value for row in rows if (value := _value(row, metric)) is not None]


def summarize_history(
    rows: Iterable[Mapping[str, Any]],
    *,
    cell: str,
    gbs: int,
    arm: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    measured = [
        dict(row)
        for row in rows
        if WARMUP_END_STEP < (_finite(row.get("_step")) or 0) <= MEASUREMENT_END_STEP
    ]
    if not measured:
        raise ValueError(f"No measured rows after step {WARMUP_END_STEP}: {cell}")
    e2e = _required_values(measured, "e2e")
    policy = _required_values(measured, "policy")
    refit = _required_values(measured, "refit")
    generation = _required_values(measured, "generation")
    tokens = _required_values(measured, "tokens")
    acceptance = _required_values(measured, "acceptance")
    draft_loss = _optional_values(measured, "draft_loss")
    draft_grad_norm = _optional_values(measured, "draft_grad_norm")
    peak_memory = _optional_values(measured, "peak_memory")
    logged_generation_tps = _optional_values(measured, "generation_tps")
    total_tokens = sum(tokens)
    total_generation_time = sum(generation)
    generation_tps = (
        total_tokens / total_generation_time
        if total_generation_time > 0
        else (
            sum(logged_generation_tps) / len(logged_generation_tps)
            if logged_generation_tps
            else None
        )
    )
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
        "steps": len(measured),
        "first_step": min(int(float(row["_step"])) for row in measured),
        "last_step": max(int(float(row["_step"])) for row in measured),
        "e2e_seconds_per_sample": sum(e2e) / (len(measured) * gbs),
        "e2e_seconds_per_token": sum(e2e) / total_tokens,
        "policy_seconds_mean": sum(policy) / len(policy),
        "refit_seconds_mean": sum(refit) / len(refit),
        "e2e_seconds_mean": sum(e2e) / len(e2e),
        "generation_tokens_per_second": generation_tps,
        "acceptance_rate_mean": sum(acceptance) / len(acceptance),
        "peak_memory_allocated_mb": max(peak_memory) if peak_memory else None,
        "draft_loss_mean": (sum(draft_loss) / len(draft_loss) if draft_loss else None),
        "draft_grad_norm_positive_steps": positive_grad_steps,
        "draft_update_count": update_count,
        "draft_refit_count": refit_count,
        "update_refit_correct": update_refit_correct,
    }


def compare_pair(fixed: Mapping[str, Any], online: Mapping[str, Any]) -> dict[str, Any]:
    fixed_value = float(fixed["e2e_seconds_per_token"])
    online_value = float(online["e2e_seconds_per_token"])
    return {
        "shape": str(fixed["cell"]).removesuffix("_fixed"),
        "fixed_e2e_seconds_per_token": fixed_value,
        "online_e2e_seconds_per_token": online_value,
        "online_overhead_percent": 100 * (online_value / fixed_value - 1),
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

    keys = ["_step", *(key for aliases in METRICS.values() for key in aliases)]
    run = wandb.Api(timeout=120).run(run_path)
    return [dict(row) for row in run.scan_history(keys=keys, page_size=1000)]


def _write_reports(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {"summaries": summaries, "comparisons": comparisons}
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
        "Steps 0-4 are excluded. Time-per-sample/token uses sums, not means of ratios.",
        "",
        "| Cell | E2E s/token | Policy s | Refit s | Gen tok/s | Acceptance | Peak MiB | Correct |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in summaries:
        peak = item["peak_memory_allocated_mb"]
        lines.append(
            f"| {item['cell']} | {item['e2e_seconds_per_token']:.6f} | "
            f"{item['policy_seconds_mean']:.3f} | {item['refit_seconds_mean']:.3f} | "
            f"{item['generation_tokens_per_second']:.2f} | "
            f"{item['acceptance_rate_mean']:.4f} | "
            f"{peak:.1f} | {item['update_refit_correct']} |"
            if peak is not None
            else f"| {item['cell']} | {item['e2e_seconds_per_token']:.6f} | "
            f"{item['policy_seconds_mean']:.3f} | {item['refit_seconds_mean']:.3f} | "
            f"{item['generation_tokens_per_second']:.2f} | "
            f"{item['acceptance_rate_mean']:.4f} | n/a | "
            f"{item['update_refit_correct']} |"
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
                evidence=json.loads(evidence_path.read_text()),
            )
        )
    by_cell = {item["cell"]: item for item in summaries}
    comparisons = [
        compare_pair(by_cell[f"{shape}_fixed"], by_cell[f"{shape}_online"])
        for shape in ("gbs32_mbs1", "gbs64_mbs1", "gbs64_mbs2")
    ]
    _write_reports(args.output_dir, summaries, comparisons)


if __name__ == "__main__":
    main()
