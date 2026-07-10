# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect validated tail-gated SpecDec cohorts from W&B history."""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterable, Mapping, Protocol, TypeGuard, cast


EXPECTED_STEPS = set(range(2, 21))
METRIC_KEYS = {
    "e2e_time": "timing/train/total_step_time",
    "generation_time": "timing/train/generation",
    "e2e_tps_gpu": "performance/tokens_per_sec_per_gpu",
    "generation_tps_gpu": "performance/generation_tokens_per_sec_per_gpu",
    "policy_time": "timing/train/policy",
    "logprob_time": "timing/train/logprob",
    "acceptance_rate": "train/vllm/spec_acceptance_rate",
    "mean_accept_len": "train/vllm/spec_acceptance_length",
    "gate_enabled_ratio": "train/vllm/spec_gate_enabled_ratio",
    "activation_batch": "train/vllm/spec_gate_activation_batch",
    "activation_seq_len": "train/vllm/spec_gate_activation_seq_len",
    "predicted_speedup": "train/vllm/spec_predicted_speedup",
    "target_graph_ratio": "train/vllm/target_graph_ratio",
    "draft_prefill_graph_ratio": "train/vllm/draft_prefill_graph_ratio",
    "draft_decode_graph_ratio": "train/vllm/draft_decode_graph_ratio",
    "reward": "train/reward",
    "response_length": "train/mean_gen_tokens_per_sample",
    "approx_kl": "train/gen_kl_error",
    "policy_loss": "train/policy_loss",
}
HEALTH_METRICS = ("reward", "response_length", "approx_kl", "policy_loss")
GRAPH_METRICS = (
    "target_graph_ratio",
    "draft_prefill_graph_ratio",
    "draft_decode_graph_ratio",
)
REQUIRED_MANIFEST_FIELDS = (
    "model",
    "runner",
    "variant",
    "graph_mode",
    "recipe",
    "commit",
    "container",
    "container_sha256",
    "job_id",
    "wandb_run_id",
)
VARIABLE_MANIFEST_FIELDS = frozenset(
    {
        "timestamp",
        "variant",
        "gate_mode",
        "k",
        "threshold",
        "consecutive_checks",
        "roofline_config_sha256",
        "job_id",
        "wandb_run_id",
        "wandb_url",
        "command",
        "source",
    }
)


@dataclass(frozen=True)
class RunSummary:
    model: str
    runner: str
    variant: str
    gate_mode: str
    K: str
    steps: list[int]
    job_id: str
    wandb_url: str
    e2e_time: float | None
    generation_time: float | None
    e2e_tps_gpu: float | None
    generation_tps_gpu: float | None
    policy_time: float | None
    logprob_time: float | None
    acceptance_rate: float | None
    mean_accept_len: float | None
    gate_enabled_ratio: float | None
    activation_batch: float | None
    activation_seq_len: float | None
    predicted_speedup: float | None
    target_graph_ratio: float | None
    draft_prefill_graph_ratio: float | None
    draft_decode_graph_ratio: float | None
    reward: float | None
    response_length: float | None
    approx_kl: float | None
    policy_loss: float | None
    status: str
    source: str
    reason: str
    graph_mode: str
    recipe: str
    commit: str
    container: str
    container_sha256: str
    wandb_run_id: str
    provenance: str
    comparison_key: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "comparison_key"
        }


@dataclass(frozen=True)
class ComparisonRow:
    summary: RunSummary
    generation_time_speedup_vs_baseline: float | None
    e2e_time_speedup_vs_baseline: float | None
    generation_tps_gpu_speedup_vs_baseline: float | None
    e2e_tps_gpu_speedup_vs_baseline: float | None
    generation_time_speedup_vs_always_on: float | None
    e2e_time_speedup_vs_always_on: float | None
    generation_tps_gpu_speedup_vs_always_on: float | None
    e2e_tps_gpu_speedup_vs_always_on: float | None
    reward_health_passed: bool | None
    response_length_health_passed: bool | None
    approx_kl_health_passed: bool | None
    policy_loss_health_passed: bool | None
    cuda_graph_health_passed: bool | None
    health_gate_passed: bool | None

    def __getattr__(self, name: str) -> object:
        return getattr(self.summary, name)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.summary.to_dict(),
            **{
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.name != "summary"
            },
        }


REQUIRED_ROW_FIELDS = tuple(
    field.name for field in fields(RunSummary) if field.name != "comparison_key"
) + tuple(field.name for field in fields(ComparisonRow) if field.name != "summary")


class WandbRun(Protocol):
    url: str

    def scan_history(self, *, keys: list[str]) -> Iterable[Mapping[str, object]]: ...


class WandbApi(Protocol):
    def run(self, path: str) -> WandbRun: ...


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _comparison_key(metadata: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (key, value)
            for key, value in metadata.items()
            if key not in VARIABLE_MANIFEST_FIELDS
        )
    )


def _provenance(metadata: Mapping[str, str]) -> str:
    return json.dumps(
        dict(sorted(metadata.items())), separators=(",", ":"), sort_keys=True
    )


def _empty_summary(
    metadata: Mapping[str, str], reason: str, steps: list[int]
) -> RunSummary:
    return RunSummary(
        model=metadata.get("model", ""),
        runner=metadata.get("runner", ""),
        variant=metadata.get("variant", ""),
        gate_mode=metadata.get("gate_mode", ""),
        K=metadata.get("k", ""),
        steps=steps,
        job_id=metadata.get("job_id", ""),
        wandb_url=metadata.get("wandb_url", ""),
        **{metric_name: None for metric_name in METRIC_KEYS},
        status="partial",
        source=metadata.get("source", ""),
        reason=reason,
        graph_mode=metadata.get("graph_mode", ""),
        recipe=metadata.get("recipe", ""),
        commit=metadata.get("commit", ""),
        container=metadata.get("container", ""),
        container_sha256=metadata.get("container_sha256", ""),
        wandb_run_id=metadata.get("wandb_run_id", ""),
        provenance=_provenance(metadata),
        comparison_key=_comparison_key(metadata),
    )


def summarize_history(
    metadata: Mapping[str, str], history: Iterable[Mapping[str, object]]
) -> RunSummary:
    """Average all required measurements from the stable Step 2-20 window."""
    records_by_step: dict[int, Mapping[str, object]] = {}
    for record in history:
        step = record.get("_step")
        if (
            isinstance(step, int)
            and not isinstance(step, bool)
            and step in EXPECTED_STEPS
        ):
            records_by_step[step] = record

    steps = sorted(records_by_step)
    missing_steps = sorted(EXPECTED_STEPS - records_by_step.keys())
    if missing_steps:
        return _empty_summary(
            metadata, f"missing_steps:{','.join(map(str, missing_steps))}", steps
        )

    values: dict[str, list[float]] = {metric_name: [] for metric_name in METRIC_KEYS}
    for step in sorted(EXPECTED_STEPS):
        record = records_by_step[step]
        for metric_name, wandb_key in METRIC_KEYS.items():
            value = record.get(wandb_key)
            if not _is_finite_number(value):
                return _empty_summary(
                    metadata, f"non_finite_metrics:{metric_name}:{step}", steps
                )
            values[metric_name].append(float(value))

    averages = {
        metric_name: statistics.fmean(metric_values)
        for metric_name, metric_values in values.items()
    }
    for metric_name in (
        "e2e_time",
        "generation_time",
        "e2e_tps_gpu",
        "generation_tps_gpu",
    ):
        if averages[metric_name] <= 0.0:
            return _empty_summary(metadata, f"non_positive_metric:{metric_name}", steps)
    return RunSummary(
        model=metadata.get("model", ""),
        runner=metadata.get("runner", ""),
        variant=metadata.get("variant", ""),
        gate_mode=metadata.get("gate_mode", ""),
        K=metadata.get("k", ""),
        steps=steps,
        job_id=metadata.get("job_id", ""),
        wandb_url=metadata.get("wandb_url", ""),
        **averages,
        status="final",
        source=metadata.get("source", ""),
        reason="",
        graph_mode=metadata.get("graph_mode", ""),
        recipe=metadata.get("recipe", ""),
        commit=metadata.get("commit", ""),
        container=metadata.get("container", ""),
        container_sha256=metadata.get("container_sha256", ""),
        wandb_run_id=metadata.get("wandb_run_id", ""),
        provenance=_provenance(metadata),
        comparison_key=_comparison_key(metadata),
    )


def _speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _within_ten_percent(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    if baseline == 0.0:
        return math.isclose(candidate, 0.0, abs_tol=1e-8)
    return abs(candidate - baseline) / abs(baseline) <= 0.10


def _graph_health(candidate: RunSummary, baseline: RunSummary) -> bool:
    return all(
        math.isclose(
            cast(float, getattr(candidate, metric_name)),
            cast(float, getattr(baseline, metric_name)),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for metric_name in GRAPH_METRICS
        if getattr(candidate, metric_name) is not None
        and getattr(baseline, metric_name) is not None
    ) and all(
        getattr(candidate, metric_name) is not None for metric_name in GRAPH_METRICS
    )


def _comparison_row(
    summary: RunSummary, baseline: RunSummary, always_on: RunSummary | None
) -> ComparisonRow:
    if summary.status != "final":
        return ComparisonRow(summary, *([None] * 14))
    health = {
        metric_name: _within_ten_percent(
            cast(float | None, getattr(summary, metric_name)),
            cast(float | None, getattr(baseline, metric_name)),
        )
        for metric_name in HEALTH_METRICS
    }
    graph_health = _graph_health(summary, baseline)
    speedups = (
        _speedup(baseline.generation_time, summary.generation_time),
        _speedup(baseline.e2e_time, summary.e2e_time),
        _speedup(summary.generation_tps_gpu, baseline.generation_tps_gpu),
        _speedup(summary.e2e_tps_gpu, baseline.e2e_tps_gpu),
    )
    always_speedups = (
        _speedup(always_on.generation_time, summary.generation_time)
        if always_on
        else None,
        _speedup(always_on.e2e_time, summary.e2e_time) if always_on else None,
        _speedup(summary.generation_tps_gpu, always_on.generation_tps_gpu)
        if always_on
        else None,
        _speedup(summary.e2e_tps_gpu, always_on.e2e_tps_gpu) if always_on else None,
    )
    return ComparisonRow(
        summary,
        *speedups,
        *always_speedups,
        health["reward"],
        health["response_length"],
        health["approx_kl"],
        health["policy_loss"],
        graph_health,
        all(health.values()) and graph_health,
    )


def build_comparison_rows(summaries: Iterable[RunSummary]) -> list[ComparisonRow]:
    """Build comparisons without allowing any cross-runner/config baseline match."""
    grouped: dict[tuple[tuple[str, str], ...], list[RunSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.comparison_key, []).append(summary)

    rows: list[ComparisonRow] = []
    for key, cohort in grouped.items():
        duplicate_variants = [
            variant
            for variant in {summary.variant for summary in cohort}
            if sum(summary.variant == variant for summary in cohort) > 1
        ]
        if duplicate_variants:
            raise ValueError(
                f"duplicate variants in cohort: {','.join(sorted(duplicate_variants))}"
            )
        runner = cohort[0].runner
        baselines = [
            summary for summary in cohort if summary.variant == f"baseline_{runner}"
        ]
        if len(baselines) != 1 or baselines[0].status != "final":
            raise ValueError(f"missing matched baseline for cohort:{dict(key)}")
        baseline = baselines[0]
        always_on = next(
            (
                summary
                for summary in cohort
                if summary.variant == f"always_on_{runner}_k5"
                and summary.status == "final"
            ),
            None,
        )
        rows.extend(_comparison_row(summary, baseline, always_on) for summary in cohort)
    return sorted(
        rows, key=lambda row: (row.runner, row.model, row.variant, row.job_id)
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _validate_manifest_rows(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return "empty manifest"
    seen: set[tuple[tuple[tuple[str, str], ...], str]] = set()
    for row in rows:
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not row.get(field)]
        if missing:
            return f"missing manifest fields:{','.join(missing)}"
        runner = row["runner"]
        if runner not in {"v1", "v2"}:
            return f"invalid runner:{runner}"
        if (
            not row["variant"].endswith(f"_{runner}")
            and f"_{runner}_" not in row["variant"]
        ):
            return f"variant runner mismatch:{row['variant']}:{runner}"
        key = (_comparison_key(row), row["variant"])
        if key in seen:
            return f"duplicate variant in cohort:{row['variant']}"
        seen.add(key)
    return None


def _write_atomic(path: Path, contents: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=REQUIRED_ROW_FIELDS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _format_metric(value: object) -> str:
    return "-" if value is None else f"{cast(float, value):.3f}"


def _render_html(rows: list[dict[str, object]]) -> str:
    fragments = [
        '<section class="tail-gated-specdec">',
        "<style>.tail-gated-specdec{font:13px sans-serif}.tail-gated-specdec table{border-collapse:collapse;width:100%;margin:8px 0 14px}.tail-gated-specdec th,.tail-gated-specdec td{border:1px solid #c9c9c9;padding:4px;text-align:right}.tail-gated-specdec th:first-child,.tail-gated-specdec td:first-child{text-align:left}.tail-gated-specdec .partial{background:#fff4cc}.tail-gated-specdec .legend{text-align:center;margin:4px}.tail-gated-specdec .bar{height:9px;background:#76b900;display:inline-block}</style>",
    ]
    final_rows = [row for row in rows if row["status"] == "final"]
    for runner in ("v1", "v2"):
        runner_rows = [row for row in rows if row["runner"] == runner]
        if not runner_rows:
            continue
        fragments.append(f"<h3>Model Runner {runner.upper()}</h3>")
        fragments.append(
            '<div class="legend">Bars: E2E time speedup vs matched baseline<br>Table: final and partial cohort rows</div>'
        )
        for model in sorted({cast(str, row["model"]) for row in runner_rows}):
            model_rows = [row for row in runner_rows if row["model"] == model]
            fragments.append(f"<h4>{html.escape(model)}</h4><div>")
            for row in model_rows:
                speedup = row["e2e_time_speedup_vs_baseline"]
                width = (
                    0
                    if speedup is None
                    else min(100, max(0, (cast(float, speedup) - 0.5) * 80))
                )
                fragments.append(
                    f'<span class="bar" style="width:{width:.0f}px"></span> {html.escape(cast(str, row["variant"]))} {_format_metric(speedup)}<br>'
                )
            fragments.append("</div>")
        fragments.append(
            "<table><thead><tr><th>Variant</th><th>Status</th><th>E2E x</th><th>Gen x</th><th>Health</th><th>W&B</th></tr></thead><tbody>"
        )
        for row in runner_rows:
            row_class = "partial" if row["status"] == "partial" else ""
            url = html.escape(cast(str, row["wandb_url"]), quote=True)
            link = f'<a href="{url}">run</a>' if url else "-"
            fragments.append(
                f'<tr class="{row_class}"><td>{html.escape(cast(str, row["variant"]))}</td><td>{html.escape(cast(str, row["status"]))}</td><td>{_format_metric(row["e2e_time_speedup_vs_baseline"])}</td><td>{_format_metric(row["generation_time_speedup_vs_baseline"])}</td><td>{row["health_gate_passed"] if row["health_gate_passed"] is not None else "-"}</td><td>{link}</td></tr>'
            )
        fragments.append("</tbody></table>")
    if final_rows:
        best = max(
            final_rows,
            key=lambda row: cast(float, row["e2e_time_speedup_vs_baseline"] or 0.0),
        )
        best_variant = html.escape(cast(str, best["variant"]))
        best_speedup = _format_metric(best["e2e_time_speedup_vs_baseline"])
        fragments.append(
            f"<p>Final finding: {best_variant} has the highest matched E2E time speedup ({best_speedup}x).</p>"
        )
    fragments.append("</section>\n")
    return "\n".join(fragments)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tail-gated SpecDec W&B runs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="nvidia")
    parser.add_argument("--project", default="nemorl-vllm024-tail-gated-lyris")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _create_wandb_api() -> WandbApi:
    wandb = importlib.import_module("wandb")
    return cast(WandbApi, wandb.Api())


def main(argv: list[str] | None = None, *, api: WandbApi | None = None) -> int:
    args = _parse_args(argv)
    manifest_rows = _read_manifest(args.manifest)
    manifest_error = _validate_manifest_rows(manifest_rows)
    if manifest_error:
        raise ValueError(manifest_error)
    output_paths = tuple(
        args.output_dir / filename
        for filename in ("summary.json", "summary.csv", "tail_gated_specdec.html")
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite historical cohort output")

    client = api if api is not None else _create_wandb_api()
    summaries: list[RunSummary] = []
    for manifest_row in manifest_rows:
        metadata = {
            **manifest_row,
            "source": manifest_row.get("source") or args.manifest.name,
        }
        try:
            run = client.run(f"{args.entity}/{args.project}/{metadata['wandb_run_id']}")
            if not metadata.get("wandb_url"):
                metadata["wandb_url"] = run.url
            summaries.append(
                summarize_history(
                    metadata, run.scan_history(keys=["_step", *METRIC_KEYS.values()])
                )
            )
        except (
            Exception
        ) as error:  # W&B failures become provenance-preserving partial rows.
            summaries.append(
                _empty_summary(
                    metadata, f"wandb_fetch_failed:{type(error).__name__}", []
                )
            )

    cohorts: dict[tuple[tuple[str, str], ...], list[RunSummary]] = {}
    for summary in summaries:
        cohorts.setdefault(summary.comparison_key, []).append(summary)
    comparison_rows: list[ComparisonRow] = []
    for key in sorted(cohorts):
        cohort = cohorts[key]
        try:
            comparison_rows.extend(build_comparison_rows(cohort))
        except ValueError as error:
            comparison_rows.extend(
                ComparisonRow(
                    replace(
                        summary,
                        status="partial",
                        reason=f"comparison_failed:{error}",
                    ),
                    *([None] * 14),
                )
                for summary in cohort
            )
    rows = [
        row.to_dict()
        for row in sorted(
            comparison_rows,
            key=lambda row: (row.runner, row.model, row.variant, row.job_id),
        )
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        args.output_dir / "summary.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(args.output_dir / "summary.csv", rows)
    _write_atomic(args.output_dir / "tail_gated_specdec.html", _render_html(rows))
    return int(
        any(
            row["status"] != "final" or row["health_gate_passed"] is False
            for row in rows
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
