"""Collect and render stable Q30 fixed-versus-always W&B results."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


GROUP = "q30ba3b-fixed-vs-always-stable-200step-20260827"
VARIANTS = frozenset({"dflash-fixed", "dflash-always", "dspark-fixed", "dspark-always"})
RUN_NAME = re.compile(
    r"q30ba3b-stable-200step-(dflash|dspark)-(fixed|always)-k5-[0-9a-f]{32}"
)
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "e2e_throughput_per_gpu": ("performance/tokens_per_sec_per_gpu",),
    "generation_throughput_per_gpu": ("performance/generation_tokens_per_sec_per_gpu",),
    "e2e_step_time_s": ("timing/train/total_step_time",),
    "generation_time_s": ("timing/train/generation",),
    "policy_training_time_s": ("timing/train/policy_training",),
    "policy_and_reference_logprob_time_s": (
        "timing/train/policy_and_reference_logprobs",
    ),
    "refit_time_s": ("timing/train/prepare_for_generation/total",),
    "acceptance_rate": (
        "train/vllm/spec_acceptance_rate",
        "vllm/spec_acceptance_rate",
        "train/spec_acceptance_rate",
        "spec_acceptance_rate",
    ),
    "mean_accepted_length": (
        "train/vllm/spec_acceptance_length",
        "vllm/spec_acceptance_length",
    ),
}
COMPARISON_METRICS = (
    "generation_throughput_per_gpu",
    "generation_time_s",
    "e2e_throughput_per_gpu",
    "e2e_step_time_s",
)


def _finite(row: Mapping[str, object], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        value = row.get(alias)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    return None


def aggregate_history(
    rows: Sequence[Mapping[str, object]], start_step: int, end_step: int
) -> dict[str, object]:
    """Aggregate finite metric values in a closed W&B step interval."""
    if start_step > end_step:
        raise ValueError("start_step must be less than or equal to end_step")

    merged_by_step: dict[int, dict[str, object]] = {}
    for row in rows:
        step = row.get("_step")
        if type(step) is int and start_step <= step <= end_step:
            merged_by_step.setdefault(step, {}).update(row)

    metric_values: dict[str, list[float]] = {name: [] for name in METRIC_ALIASES}
    for row in merged_by_step.values():
        for name, aliases in METRIC_ALIASES.items():
            value = _finite(row, aliases)
            if value is not None:
                metric_values[name].append(value)

    included_steps = sorted(merged_by_step)
    missing_steps = [
        step for step in range(start_step, end_step + 1) if step not in merged_by_step
    ]
    return {
        "window": {
            "start_step": start_step,
            "end_step": end_step,
            "step_count": end_step - start_step + 1,
        },
        "included_steps": included_steps,
        "missing_steps": missing_steps,
        "completed": not missing_steps,
        "metrics": {
            name: {
                "mean": sum(values) / len(values) if values else None,
                "valid_count": len(values),
            }
            for name, values in metric_values.items()
        },
    }


def _variant(run: Mapping[str, object]) -> str | None:
    explicit = run.get("variant")
    if isinstance(explicit, str) and explicit in VARIANTS:
        return explicit
    config = run.get("config")
    if isinstance(config, Mapping):
        configured = config.get("variant")
        if isinstance(configured, str) and configured in VARIANTS:
            return configured
    name = run.get("name")
    if isinstance(name, str):
        match = RUN_NAME.fullmatch(name)
        if match is not None:
            return f"{match.group(1)}-{match.group(2)}"
    return None


def _metric(summary: Mapping[str, object], name: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    field = metrics.get(name)
    value = field.get("mean") if isinstance(field, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric > 0.0:
            return numeric
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / denominator


def _has_full_metric(summary: Mapping[str, object], name: str) -> bool:
    """Return whether every requested step has a valid value for one metric."""
    window = summary.get("window")
    metrics = summary.get("metrics")
    if not isinstance(window, Mapping) or not isinstance(metrics, Mapping):
        return False
    step_count = window.get("step_count")
    metric = metrics.get(name)
    return bool(
        type(step_count) is int
        and isinstance(metric, Mapping)
        and metric.get("valid_count") == step_count
        and _metric(summary, name) is not None
    )


def _run_ready(run: Mapping[str, object], summary: Mapping[str, object]) -> bool:
    """Return whether a run can participate in a final comparison."""
    return bool(
        run.get("state") == "finished"
        and summary.get("completed") is True
        and all(_has_full_metric(summary, name) for name in COMPARISON_METRICS)
    )


def build_comparisons(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Compare always-online only with the matched fixed drafter."""
    by_variant = {
        variant: run for run in runs if (variant := _variant(run)) is not None
    }
    comparisons: list[dict[str, object]] = []
    for drafter in ("dflash", "dspark"):
        variant = f"{drafter}-always"
        baseline_name = f"{drafter}-fixed"
        run = by_variant.get(variant)
        baseline = by_variant.get(baseline_name)
        if run is None:
            continue
        if baseline is None:
            comparisons.append(
                {
                    "variant": variant,
                    "fixed_baseline": baseline_name,
                    "status": "waiting fixed baseline",
                    "generation_throughput_speedup": None,
                    "generation_time_speedup": None,
                    "e2e_throughput_speedup": None,
                    "e2e_step_time_speedup": None,
                }
            )
            continue
        summary = run.get("summary")
        baseline_summary = baseline.get("summary")
        if not isinstance(summary, Mapping) or not isinstance(
            baseline_summary, Mapping
        ):
            raise TypeError("comparison runs require normalized summaries")
        ready = _run_ready(run, summary) and _run_ready(baseline, baseline_summary)
        ratios = {
            "generation_throughput_speedup": _ratio(
                _metric(summary, "generation_throughput_per_gpu"),
                _metric(baseline_summary, "generation_throughput_per_gpu"),
            ),
            "generation_time_speedup": _ratio(
                _metric(baseline_summary, "generation_time_s"),
                _metric(summary, "generation_time_s"),
            ),
            "e2e_throughput_speedup": _ratio(
                _metric(summary, "e2e_throughput_per_gpu"),
                _metric(baseline_summary, "e2e_throughput_per_gpu"),
            ),
            "e2e_step_time_speedup": _ratio(
                _metric(baseline_summary, "e2e_step_time_s"),
                _metric(summary, "e2e_step_time_s"),
            ),
        }
        ready = ready and all(value is not None for value in ratios.values())
        comparisons.append(
            {
                "variant": variant,
                "fixed_baseline": baseline_name,
                "status": "ready" if ready else "preliminary",
                **{name: value if ready else None for name, value in ratios.items()},
            }
        )
    return comparisons


def build_report(
    runs: Sequence[Mapping[str, object]], *, entity: str, project: str
) -> dict[str, object]:
    """Normalize retries and aggregate the closed steps 3 through 200."""
    selected: dict[str, Mapping[str, object]] = {}
    for run in runs:
        variant = _variant(run)
        if variant is None:
            continue
        current = selected.get(variant)
        retry_key = (str(run.get("created_at", "")), str(run.get("id", "")))
        current_key = (
            (
                str(current.get("created_at", "")),
                str(current.get("id", "")),
            )
            if current is not None
            else ("", "")
        )
        if current is None or retry_key > current_key:
            selected[variant] = run

    normalized: list[dict[str, object]] = []
    for variant in sorted(selected):
        run = selected[variant]
        rows = run.get("history")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            rows = []
        normalized.append(
            {
                "id": str(run.get("id", "")),
                "name": str(run.get("name", "")),
                "variant": variant,
                "state": str(run.get("state", "unknown")),
                "created_at": str(run.get("created_at", "")),
                "summary": aggregate_history(rows, 3, 200),
            }
        )
    return {
        "entity": entity,
        "project": project,
        "group": GROUP,
        "methodology": "closed steps 3-200; canonical logged throughput only",
        "runs": normalized,
        "comparisons": build_comparisons(normalized),
    }


def _format(value: object, suffix: str = "") -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f"{numeric:.3f}{suffix}"
    return "n/a"


def render_html(report: Mapping[str, object]) -> str:
    """Render a self-contained HTML report with audit metadata."""
    run_rows: list[str] = []
    runs = report.get("runs")
    if isinstance(runs, Sequence):
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            summary = run.get("summary")
            if not isinstance(summary, Mapping):
                continue
            metrics = summary.get("metrics")
            metric_map = metrics if isinstance(metrics, Mapping) else {}

            def displayed(name: str) -> tuple[object, object]:
                field = metric_map.get(name)
                if isinstance(field, Mapping):
                    return field.get("mean"), field.get("valid_count")
                return None, 0

            generation, generation_count = displayed("generation_throughput_per_gpu")
            e2e, e2e_count = displayed("e2e_throughput_per_gpu")
            generation_time, generation_time_count = displayed("generation_time_s")
            e2e_time, e2e_time_count = displayed("e2e_step_time_s")
            included = summary.get("included_steps")
            missing = summary.get("missing_steps")
            readiness = "ready" if _run_ready(run, summary) else "preliminary"
            full_metric_count = sum(
                _has_full_metric(summary, name) for name in COMPARISON_METRICS
            )
            included_text = (
                ", ".join(map(str, included)) if isinstance(included, list) else "n/a"
            )
            missing_text = (
                ", ".join(map(str, missing)) if isinstance(missing, list) else "n/a"
            )
            run_rows.append(
                "<tr>"
                f"<td>{html.escape(str(run.get('variant', 'unknown')))}</td>"
                f"<td>{readiness}</td>"
                f"<td>{html.escape(str(run.get('state', 'unknown')))}</td>"
                f"<td>{full_metric_count}/{len(COMPARISON_METRICS)} full-window metrics</td>"
                f"<td>{_format(generation)}<br><small>valid={html.escape(str(generation_count))}</small></td>"
                f"<td>{_format(generation_time, ' s')}<br><small>valid={html.escape(str(generation_time_count))}</small></td>"
                f"<td>{_format(e2e)}<br><small>valid={html.escape(str(e2e_count))}</small></td>"
                f"<td>{_format(e2e_time, ' s')}<br><small>valid={html.escape(str(e2e_time_count))}</small></td>"
                f"<td><small>Included steps: {html.escape(included_text)}<br>Missing steps: {html.escape(missing_text)}</small></td>"
                "</tr>"
            )

    comparison_rows: list[str] = []
    comparisons = report.get("comparisons")
    if isinstance(comparisons, Sequence):
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                continue
            comparison_rows.append(
                "<tr>"
                f"<td>{html.escape(str(comparison.get('variant', 'unknown')))}</td>"
                f"<td>{html.escape(str(comparison.get('fixed_baseline', 'unknown')))}</td>"
                f"<td>{html.escape(str(comparison.get('status', 'unknown')))}</td>"
                f"<td>{_format(comparison.get('generation_throughput_speedup'), 'x')}</td>"
                f"<td>{_format(comparison.get('generation_time_speedup'), 'x')}</td>"
                f"<td>{_format(comparison.get('e2e_throughput_speedup'), 'x')}</td>"
                f"<td>{_format(comparison.get('e2e_step_time_speedup'), 'x')}</td>"
                "</tr>"
            )

    return "".join(
        (
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Q30 fixed vs always</title><style>",
            "body{font-family:system-ui;margin:2rem;background:#f4f6f8;color:#17202a}",
            "main{max-width:1280px;margin:auto;background:white;padding:2rem;border-radius:12px}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d8dee4;padding:.6rem;text-align:left}th{background:#edf2f7}small{color:#52606d}",
            "</style></head><body><main>",
            "<h1>Qwen3-30B-A3B: fixed vs always-online drafter training</h1>",
            "<p>Means use closed steps 3–200 and canonical logged W&amp;B throughput. fixed means frozen drafter training while generation SpecDec remains enabled. Comparisons are always-online relative to the same drafter's fixed arm, not relative to a no-SpecDec baseline.</p>",
            "<h2>Run metrics</h2><table><thead><tr><th>Variant</th><th>Readiness</th><th>W&amp;B state</th><th>Comparison metric coverage</th><th>Generation tok/s/GPU</th><th>Generation time</th><th>E2E tok/s/GPU</th><th>E2E step time</th><th>Audit</th></tr></thead><tbody>",
            "".join(run_rows),
            "</tbody></table><h2>Always-online / fixed comparison</h2><table><thead><tr><th>Variant</th><th>Fixed arm</th><th>Status</th><th>Generation throughput</th><th>Generation time</th><th>E2E throughput</th><th>E2E step time</th></tr></thead><tbody>",
            "".join(comparison_rows),
            "</tbody></table></main></body></html>",
        )
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _wandb_runs(entity: str, project: str, group: str) -> list[dict[str, object]]:
    wandb = importlib.import_module("wandb")
    api = wandb.Api()
    keys = [
        "_step",
        *(alias for aliases in METRIC_ALIASES.values() for alias in aliases),
    ]
    runs: list[dict[str, object]] = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        rows = list(run.scan_history(keys=keys, min_step=3, max_step=201))
        runs.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "config": dict(run.config),
                "history": rows,
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-json", type=Path)
    parser.add_argument("--entity", default="nvidia")
    parser.add_argument("--project", default="sna-specdec")
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    args = parser.parse_args()

    if args.history_json is not None:
        payload = json.loads(args.history_json.read_text())
        runs = payload.get("runs", [])
    else:
        runs = _wandb_runs(args.entity, args.project, args.group)
    if not isinstance(runs, list):
        raise TypeError("report input must contain a runs list")
    report = build_report(runs, entity=args.entity, project=args.project)
    _atomic_write(args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.html_output, render_html(report))
    print(
        json.dumps(
            {"runs": len(report["runs"]), "comparisons": len(report["comparisons"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
