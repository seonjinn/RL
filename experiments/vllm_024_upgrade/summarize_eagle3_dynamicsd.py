from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Protocol, TypeGuard, cast


METRIC_KEYS = {
    "generation_time_s": "timing/train/generation",
    "e2e_step_time_s": "timing/train/total_step_time",
    "generation_throughput": "performance/generation_tokens_per_sec_per_gpu",
    "e2e_throughput": "performance/tokens_per_sec_per_gpu",
    "acceptance_rate": "train/vllm/spec_acceptance_rate",
    "mean_acceptance_length": "train/vllm/spec_acceptance_length",
    "reward": "train/reward",
    "mean_response_length": "train/mean_gen_tokens_per_sample",
    "approx_kl": "train/gen_kl_error",
}
EXPECTED_STEPS = set(range(2, 21))
SPECDEC_VARIANTS = frozenset({"eagle3_k5", "eagle3_k7", "eagle3_k9", "dynamic"})
SPECDEC_METRICS = frozenset({"acceptance_rate", "mean_acceptance_length"})
SPECDEC_COUNTER_KEYS = {
    "num_drafts": "train/vllm/spec_num_drafts",
    "num_draft_tokens": "train/vllm/spec_num_draft_tokens",
    "num_accepted_tokens": "train/vllm/spec_num_accepted_tokens",
}
POSITIVE_METRICS = frozenset(
    {
        "generation_time_s",
        "e2e_step_time_s",
        "generation_throughput",
        "e2e_throughput",
    }
)
MATCHED_SETUP_FIELDS = (
    "recipe",
    "nodes",
    "segment",
    "commit",
    "container",
    "container_sha256",
    "max_steps",
)


@dataclass(frozen=True)
class RunSummary:
    model: str
    variant: str
    complete: bool
    reason: str
    measured_steps: list[int]
    generation_time_s: float | None
    e2e_step_time_s: float | None
    generation_throughput: float | None
    e2e_throughput: float | None
    acceptance_rate: float | None
    mean_acceptance_length: float | None
    reward: float | None
    mean_response_length: float | None
    approx_kl: float | None


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    variant: str
    complete: bool
    reason: str
    measured_steps: list[int]
    generation_time_s: float | None
    e2e_step_time_s: float | None
    generation_throughput: float | None
    e2e_throughput: float | None
    acceptance_rate: float | None
    mean_acceptance_length: float | None
    reward: float | None
    mean_response_length: float | None
    approx_kl: float | None
    generation_time_speedup_vs_baseline: float | None
    e2e_step_time_speedup_vs_baseline: float | None
    generation_throughput_speedup_vs_baseline: float | None
    e2e_throughput_speedup_vs_baseline: float | None
    generation_time_speedup_vs_fixed: float | None
    e2e_step_time_speedup_vs_fixed: float | None
    generation_throughput_speedup_vs_fixed: float | None
    e2e_throughput_speedup_vs_fixed: float | None
    reward_health_passed: bool | None
    response_length_health_passed: bool | None
    kl_health_passed: bool | None
    health_gate_passed: bool | None


class WandbRun(Protocol):
    url: str

    def scan_history(self, *, keys: list[str]) -> Iterable[Mapping[str, object]]: ...


class WandbApi(Protocol):
    def run(self, path: str) -> WandbRun: ...


def _empty_summary(
    model: str, variant: str, reason: str, measured_steps: list[int]
) -> RunSummary:
    return RunSummary(
        model=model,
        variant=variant,
        complete=False,
        reason=reason,
        measured_steps=measured_steps,
        **{metric_name: None for metric_name in METRIC_KEYS},
    )


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_history(
    model: str, variant: str, history: Iterable[Mapping[str, object]]
) -> RunSummary:
    """Aggregate the steady-state W&B records for one submitted variant."""
    records_by_step: dict[int, Mapping[str, object]] = {}
    for record in history:
        step = record.get("_step")
        if isinstance(step, int) and not isinstance(step, bool) and step in EXPECTED_STEPS:
            records_by_step[step] = record

    measured_steps = sorted(records_by_step)
    missing_steps = sorted(EXPECTED_STEPS - records_by_step.keys())
    if missing_steps:
        return _empty_summary(
            model, variant, f"missing_steps:{','.join(map(str, missing_steps))}", measured_steps
        )

    required_metrics = set(METRIC_KEYS) - SPECDEC_METRICS
    values: dict[str, list[float]] = {metric_name: [] for metric_name in METRIC_KEYS}
    counter_values: dict[str, list[float]] = {
        metric_name: [] for metric_name in SPECDEC_COUNTER_KEYS
    }
    for step in sorted(EXPECTED_STEPS):
        record = records_by_step[step]
        for metric_name in required_metrics:
            wandb_key = METRIC_KEYS[metric_name]
            value = record.get(wandb_key)
            if not _is_finite_number(value):
                return _empty_summary(
                    model, variant, f"non_finite_metrics:{metric_name}:{step}", measured_steps
                )
            values[metric_name].append(float(value))
        if variant in SPECDEC_VARIANTS:
            for metric_name, wandb_key in SPECDEC_COUNTER_KEYS.items():
                value = record.get(wandb_key)
                if not _is_finite_number(value):
                    return _empty_summary(
                        model, variant, f"non_finite_metrics:{metric_name}:{step}", measured_steps
                    )
                counter_values[metric_name].append(float(value))

    averages = {
        metric_name: statistics.fmean(metric_values) if metric_values else None
        for metric_name, metric_values in values.items()
    }
    for metric_name in POSITIVE_METRICS:
        value = averages[metric_name]
        if value is None or value <= 0.0:
            return _empty_summary(
                model, variant, f"non_positive_metric:{metric_name}", measured_steps
            )
    if variant in SPECDEC_VARIANTS:
        total_drafts = sum(counter_values["num_drafts"])
        total_draft_tokens = sum(counter_values["num_draft_tokens"])
        total_accepted_tokens = sum(counter_values["num_accepted_tokens"])
        if total_drafts <= 0.0 or total_draft_tokens <= 0.0 or total_accepted_tokens <= 0.0:
            return _empty_summary(model, variant, "missing_specdec_evidence", measured_steps)
        averages["acceptance_rate"] = total_accepted_tokens / total_draft_tokens
        averages["mean_acceptance_length"] = 1.0 + total_accepted_tokens / total_drafts

    return RunSummary(
        model=model,
        variant=variant,
        complete=True,
        reason="",
        measured_steps=measured_steps,
        **averages,
    )


def _speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _health_metric(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    if baseline == 0.0:
        return math.isclose(candidate, 0.0, abs_tol=1e-8)
    return abs(candidate - baseline) / abs(baseline) <= 0.10


def _comparison_row(
    summary: RunSummary, baseline: RunSummary, fixed: RunSummary | None
) -> ComparisonRow:
    reward_health = _health_metric(summary.reward, baseline.reward)
    response_length_health = _health_metric(
        summary.mean_response_length, baseline.mean_response_length
    )
    kl_health = _health_metric(summary.approx_kl, baseline.approx_kl)
    return ComparisonRow(
        **asdict(summary),
        generation_time_speedup_vs_baseline=_speedup(
            baseline.generation_time_s, summary.generation_time_s
        ),
        e2e_step_time_speedup_vs_baseline=_speedup(
            baseline.e2e_step_time_s, summary.e2e_step_time_s
        ),
        generation_throughput_speedup_vs_baseline=_speedup(
            summary.generation_throughput, baseline.generation_throughput
        ),
        e2e_throughput_speedup_vs_baseline=_speedup(
            summary.e2e_throughput, baseline.e2e_throughput
        ),
        generation_time_speedup_vs_fixed=(
            _speedup(fixed.generation_time_s, summary.generation_time_s)
            if summary.variant == "dynamic" and fixed is not None
            else None
        ),
        e2e_step_time_speedup_vs_fixed=(
            _speedup(fixed.e2e_step_time_s, summary.e2e_step_time_s)
            if summary.variant == "dynamic" and fixed is not None
            else None
        ),
        generation_throughput_speedup_vs_fixed=(
            _speedup(summary.generation_throughput, fixed.generation_throughput)
            if summary.variant == "dynamic" and fixed is not None
            else None
        ),
        e2e_throughput_speedup_vs_fixed=(
            _speedup(summary.e2e_throughput, fixed.e2e_throughput)
            if summary.variant == "dynamic" and fixed is not None
            else None
        ),
        reward_health_passed=reward_health,
        response_length_health_passed=response_length_health,
        kl_health_passed=kl_health,
        health_gate_passed=reward_health and response_length_health and kl_health,
    )


def build_comparison_rows(summaries: Iterable[RunSummary]) -> list[ComparisonRow]:
    """Build model-matched baseline and fixed-K comparisons from complete runs."""
    grouped: dict[str, list[RunSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.model, []).append(summary)

    comparison_rows: list[ComparisonRow] = []
    for model_summaries in grouped.values():
        variants = [summary.variant for summary in model_summaries]
        if len(variants) != len(set(variants)):
            raise ValueError(f"duplicate variants for model {model_summaries[0].model}")
        baselines = [summary for summary in model_summaries if summary.variant == "baseline"]
        if not baselines:
            raise ValueError(f"missing baseline for model {model_summaries[0].model}")
        if len(baselines) != 1:
            raise ValueError(f"expected one baseline for model {model_summaries[0].model}")
        baseline = baselines[0]
        if not baseline.complete:
            raise ValueError(f"incomplete baseline for model {baseline.model}: {baseline.reason}")

        fixed_runs = [
            summary
            for summary in model_summaries
            if summary.variant == "eagle3_k5" and summary.complete
        ]
        if len(fixed_runs) > 1:
            raise ValueError(f"expected one fixed-K run for model {baseline.model}")
        fixed = fixed_runs[0] if fixed_runs else None
        comparison_rows.extend(
            _comparison_row(summary, baseline, fixed) for summary in model_summaries
        )
    return comparison_rows


def _unmatched_row(summary: RunSummary) -> dict[str, object]:
    row = asdict(summary)
    for field in fields(ComparisonRow):
        row.setdefault(field.name, None)
    return row


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _validate_manifest_rows(rows: list[dict[str, str]]) -> str | None:
    seen: set[tuple[str, str]] = set()
    setup_by_model: dict[str, tuple[str, ...]] = {}
    for row in rows:
        model = row.get("model", "")
        variant = row.get("variant", "")
        key = (model, variant)
        if key in seen:
            return f"duplicate variant {variant} for model {model}"
        seen.add(key)
        setup = tuple(row.get(field, "") for field in MATCHED_SETUP_FIELDS)
        previous = setup_by_model.setdefault(model, setup)
        if setup != previous:
            return f"mismatched setup for model {model}"
    return None


def _write_json_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [field.name for field in fields(ComparisonRow)] + [
        "job_id",
        "wandb_run_id",
        "wandb_url",
    ]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _history_keys(variant: str) -> list[str]:
    keys = [
        wandb_key
        for metric_name, wandb_key in METRIC_KEYS.items()
        if metric_name not in SPECDEC_METRICS
    ]
    if variant in SPECDEC_VARIANTS:
        keys.extend(SPECDEC_COUNTER_KEYS.values())
    return ["_step", *keys]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Eagle-3 DynamicSD W&B runs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="nvidia")
    parser.add_argument("--project", default="nemorl-vllm024-dynamicsd-aws-dfw")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _create_wandb_api() -> WandbApi:
    wandb = importlib.import_module("wandb")
    return cast(WandbApi, wandb.Api())


def main(argv: list[str] | None = None, *, api: WandbApi | None = None) -> int:
    args = _parse_args(argv)
    manifest_rows = _read_manifest(args.manifest)
    manifest_error = _validate_manifest_rows(manifest_rows)
    client = api if api is not None else _create_wandb_api()
    summaries: list[RunSummary] = []
    metadata: list[dict[str, str]] = []

    for manifest_row in manifest_rows:
        model = manifest_row.get("model", "")
        variant = manifest_row.get("variant", "")
        run_id = manifest_row.get("wandb_run_id", "")
        if not model or not variant or not run_id:
            summaries.append(_empty_summary(model, variant, "missing_manifest_fields", []))
            metadata.append(manifest_row)
            continue
        try:
            run = client.run(f"{args.entity}/{args.project}/{run_id}")
            summary = summarize_history(
                model,
                variant,
                run.scan_history(keys=_history_keys(variant)),
            )
            if not manifest_row.get("wandb_url"):
                manifest_row = {**manifest_row, "wandb_url": run.url}
        except Exception as error:  # W&B errors are converted into an incomplete row.
            summary = _empty_summary(model, variant, f"wandb_fetch_failed:{type(error).__name__}", [])
        summaries.append(summary)
        metadata.append(manifest_row)

    comparison_error: ValueError | None = None
    try:
        if manifest_error is not None:
            raise ValueError(manifest_error)
        rows = [asdict(row) for row in build_comparison_rows(summaries)]
        metadata_by_run = {
            (summary.model, summary.variant): manifest_row
            for summary, manifest_row in zip(summaries, metadata, strict=True)
        }
        row_metadata = [
            metadata_by_run[(row["model"], row["variant"])] for row in rows
        ]
    except ValueError as error:
        comparison_error = error
        rows = [_unmatched_row(summary) for summary in summaries]
        for row in rows:
            row["complete"] = False
            row["reason"] = f"comparison_failed:{error}"
        row_metadata = metadata
    for row, manifest_row in zip(rows, row_metadata, strict=True):
        row["job_id"] = manifest_row.get("job_id", "")
        row["wandb_run_id"] = manifest_row.get("wandb_run_id", "")
        row["wandb_url"] = manifest_row.get("wandb_url", "")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.output_dir / "summary.json", rows)
    _write_csv(args.output_dir / "summary.csv", rows)
    return int(
        comparison_error is not None
        or any(not bool(row.get("complete")) for row in rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
