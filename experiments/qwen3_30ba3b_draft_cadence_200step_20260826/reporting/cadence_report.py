"""Collect and render the Qwen3-30B-A3B cadence experiment results."""

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


CADENCE_VARIANTS = frozenset(
    {
        "dflash-static",
        "dflash-always",
        "dflash-fixed10",
        "dspark-static",
        "dspark-always",
        "dspark-fixed10",
    }
)
RUN_NAME_PATTERN = re.compile(
    r"q30ba3b-200step-(dflash|dspark)-(static|always|fixed10)-k5-[0-9a-f]{32}"
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
ACCEPTED_COUNT_ALIASES = (
    "train/vllm/spec_num_accepted_tokens",
    "vllm/spec_num_accepted_tokens",
    "train/vllm/spec_accepted_tokens",
    "vllm/spec_accepted_tokens",
)
DRAFT_COUNT_ALIASES = (
    "train/vllm/spec_num_draft_tokens",
    "vllm/spec_num_draft_tokens",
    "train/vllm/spec_draft_tokens",
    "vllm/spec_draft_tokens",
)


def _finite_number(row: Mapping[str, object], aliases: Sequence[str]) -> float | None:
    """Return the first finite numeric value for any documented metric alias."""
    for alias in aliases:
        value = row.get(alias)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    return None


def _acceptance_rate(row: Mapping[str, object]) -> float | None:
    """Read a logged acceptance rate or derive one from the logged count pair."""
    direct = _finite_number(row, METRIC_ALIASES["acceptance_rate"])
    if direct is not None:
        return direct
    accepted = _finite_number(row, ACCEPTED_COUNT_ALIASES)
    drafted = _finite_number(row, DRAFT_COUNT_ALIASES)
    if accepted is None or drafted is None or drafted <= 0.0:
        return None
    return accepted / drafted


def _metric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return the arithmetic mean and observation count for one metric."""
    return {
        "mean": sum(values) / len(values) if values else None,
        "valid_count": len(values),
    }


def aggregate_history(
    rows: Sequence[Mapping[str, object]], start_step: int, end_step: int
) -> dict[str, object]:
    """Aggregate finite W&B observations in a closed step interval.

    Throughput deliberately has only its canonical W&B keys in ``METRIC_ALIASES``:
    it is never reconstructed from timing or token-count fields.
    """
    if start_step > end_step:
        raise ValueError("start_step must be less than or equal to end_step")

    window_rows: list[Mapping[str, object]] = []
    included_steps: set[int] = set()
    for row in rows:
        step = row.get("_step")
        if type(step) is int and start_step <= step <= end_step:
            window_rows.append(row)
            included_steps.add(step)

    metric_values: dict[str, list[float]] = {
        metric_name: [] for metric_name in METRIC_ALIASES
    }
    for row in window_rows:
        for metric_name, aliases in METRIC_ALIASES.items():
            value = (
                _acceptance_rate(row)
                if metric_name == "acceptance_rate"
                else _finite_number(row, aliases)
            )
            if value is not None:
                metric_values[metric_name].append(value)

    missing_steps = [
        step for step in range(start_step, end_step + 1) if step not in included_steps
    ]
    return {
        "window": {
            "start_step": start_step,
            "end_step": end_step,
            "step_count": end_step - start_step + 1,
        },
        "included_steps": sorted(included_steps),
        "missing_steps": missing_steps,
        "completed": not missing_steps,
        "metrics": {
            metric_name: _metric_summary(values)
            for metric_name, values in metric_values.items()
        },
        "cadence_reason_counts": {},
        "cadence_reason_counts_provenance": None,
    }


def _variant(run: Mapping[str, object]) -> str | None:
    """Extract an allowlisted variant, falling back to the exact launcher contract."""
    explicit = run.get("variant")
    if isinstance(explicit, str) and explicit in CADENCE_VARIANTS:
        return explicit
    config = run.get("config")
    if isinstance(config, Mapping):
        for key in ("cadence_variant", "variant"):
            configured = config.get(key)
            if isinstance(configured, str) and configured in CADENCE_VARIANTS:
                return configured
    name = run.get("name")
    if isinstance(name, str):
        if name in CADENCE_VARIANTS:
            return name
        match = RUN_NAME_PATTERN.fullmatch(name)
        if match is not None:
            return f"{match.group(1)}-{match.group(2)}"
    return None


def _retry_key(run: Mapping[str, object]) -> tuple[str, str]:
    """Return a stable newest-attempt key from allowlisted W&B metadata."""
    created_at = run.get("created_at")
    created_key = created_at if isinstance(created_at, str) else ""
    return created_key, str(run.get("id", ""))


def _select_latest_retries(
    runs: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Select exactly one deterministic, latest launcher attempt per variant."""
    selected: dict[str, Mapping[str, object]] = {}
    for run in runs:
        variant = _variant(run)
        if variant is None:
            continue
        incumbent = selected.get(variant)
        if incumbent is None or _retry_key(run) > _retry_key(incumbent):
            selected[variant] = run
    return [selected[variant] for variant in sorted(selected)]


def _summary(run: Mapping[str, object]) -> Mapping[str, object]:
    """Return a normalized run summary, accepting direct summaries for callers."""
    summary = run.get("summary")
    return summary if isinstance(summary, Mapping) else run


def _metric_mean(summary: Mapping[str, object], metric_name: str) -> float | None:
    """Read a finite metric mean from one normalized run summary."""
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    if isinstance(metric, Mapping):
        value = metric.get("mean")
    else:
        value = metric
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _positive_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Divide two positive finite metric means after explicit type narrowing."""
    if numerator is None or denominator is None:
        return None
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return numerator / denominator


def build_comparisons(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Compare each always/fixed10 arm with its same-drafter static arm."""
    static_runs = {
        variant: run
        for run in runs
        if (variant := _variant(run)) in {"dflash-static", "dspark-static"}
    }
    comparisons: list[dict[str, object]] = []
    for run in runs:
        variant = _variant(run)
        if variant is None or not variant.endswith(("-always", "-fixed10")):
            continue
        drafter = variant.split("-", maxsplit=1)[0]
        baseline_name = f"{drafter}-static"
        baseline = static_runs.get(baseline_name)
        if baseline is None:
            comparisons.append(
                {
                    "variant": variant,
                    "static_baseline": baseline_name,
                    "status": "waiting static baseline",
                    "e2e_throughput_speedup": None,
                    "generation_throughput_speedup": None,
                }
            )
            continue

        summary = _summary(run)
        baseline_summary = _summary(baseline)
        e2e = _metric_mean(summary, "e2e_throughput_per_gpu")
        baseline_e2e = _metric_mean(baseline_summary, "e2e_throughput_per_gpu")
        generation = _metric_mean(summary, "generation_throughput_per_gpu")
        baseline_generation = _metric_mean(
            baseline_summary, "generation_throughput_per_gpu"
        )
        e2e_speedup = _positive_ratio(e2e, baseline_e2e)
        generation_speedup = _positive_ratio(generation, baseline_generation)
        ready = (
            e2e_speedup is not None
            and generation_speedup is not None
            and summary.get("completed") is True
            and baseline_summary.get("completed") is True
        )
        comparisons.append(
            {
                "variant": variant,
                "static_baseline": baseline_name,
                "status": "ready" if ready else "preliminary",
                "e2e_throughput_speedup": e2e_speedup if ready else None,
                "generation_throughput_speedup": generation_speedup if ready else None,
            }
        )
    return comparisons


def _format_number(value: object, *, suffix: str = "") -> str:
    """Format a report metric without exposing arbitrary source objects."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f"{numeric:.3f}{suffix}"
    return "n/a"


def _display_run_row(run: Mapping[str, object]) -> str:
    """Render one bounded, escaped run row for the report table."""
    variant = _variant(run) or "unnamed run"
    summary = _summary(run)
    metrics = summary.get("metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    values = {
        metric_name: _metric_mean(summary, metric_name)
        for metric_name in METRIC_ALIASES
    }
    status = "complete" if summary.get("completed") is True else "preliminary"
    valid_counts: dict[str, object] = {}
    for metric_name in METRIC_ALIASES:
        metric = metric_map.get(metric_name)
        if isinstance(metric, Mapping):
            valid_counts[metric_name] = metric.get("valid_count", 0)
        else:
            valid_counts[metric_name] = 0
    included = summary.get("included_steps")
    included_text = (
        ", ".join(str(step) for step in included)
        if isinstance(included, list)
        else "n/a"
    )
    missing = summary.get("missing_steps")
    missing_text = (
        ", ".join(str(step) for step in missing) if isinstance(missing, list) else "n/a"
    )
    reasons = summary.get("cadence_reason_counts")
    reason_provenance = summary.get("cadence_reason_counts_provenance")
    reason_text = ""
    if isinstance(reasons, Mapping):
        reason_text = ", ".join(
            f"{html.escape(str(reason))}: {html.escape(str(count))}"
            for reason, count in sorted(reasons.items(), key=lambda item: str(item[0]))
        )
    provenance_text = ""
    if isinstance(reason_provenance, Mapping):
        source = reason_provenance.get("source")
        path = reason_provenance.get("path")
        if isinstance(source, str) and isinstance(path, str):
            provenance_text = (
                f"<br>Source: {html.escape(source)}<br>Path: {html.escape(path)}"
            )
    return "".join(
        (
            "<tr>",
            f"<td>{html.escape(variant)}</td>",
            f"<td>{status}</td>",
            f"<td>{_format_number(values['generation_throughput_per_gpu'])}</td>",
            f"<td>{_format_number(values['generation_time_s'], suffix=' s')}</td>",
            f"<td>{_format_number(values['e2e_throughput_per_gpu'])}</td>",
            f"<td>{_format_number(values['e2e_step_time_s'], suffix=' s')}</td>",
            f"<td>{_format_number(values['policy_training_time_s'], suffix=' s')}</td>",
            f"<td>{_format_number(values['policy_and_reference_logprob_time_s'], suffix=' s')}</td>",
            f"<td>{_format_number(values['refit_time_s'], suffix=' s')}</td>",
            f"<td>{_format_number(values['acceptance_rate'])}</td>",
            f"<td>{_format_number(values['mean_accepted_length'])}</td>",
            "<td>"
            f"Included steps: {html.escape(included_text)}<br>"
            f"Missing steps: {html.escape(missing_text)}"
            "</td>",
            "<td>"
            + "; ".join(
                f"{html.escape(metric_name)}={html.escape(str(valid_counts[metric_name]))}"
                for metric_name in METRIC_ALIASES
            )
            + "</td>",
            f"<td>{reason_text or 'n/a'}{provenance_text}</td>",
            "</tr>",
        )
    )


def render_html(report: Mapping[str, object]) -> str:
    """Render a self-contained, cadence-relative Q30 results page."""
    runs = report.get("runs")
    run_rows = runs if isinstance(runs, Sequence) and not isinstance(runs, str) else []
    comparisons = report.get("comparisons")
    comparison_rows = (
        comparisons
        if isinstance(comparisons, Sequence) and not isinstance(comparisons, str)
        else []
    )
    rendered_runs = (
        "".join(_display_run_row(run) for run in run_rows if isinstance(run, Mapping))
        or '<tr><td colspan="14">No matching runs collected.</td></tr>'
    )
    rendered_comparisons = (
        "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('variant', 'unnamed run')))}</td>"
            f"<td>{html.escape(str(row.get('static_baseline', 'n/a')))}</td>"
            f"<td>{html.escape(str(row.get('status', 'preliminary')))}</td>"
            f"<td>{_format_number(row.get('generation_throughput_speedup'), suffix='x')}</td>"
            f"<td>{_format_number(row.get('e2e_throughput_speedup'), suffix='x')}</td>"
            "</tr>"
            for row in comparison_rows
            if isinstance(row, Mapping)
        )
        or '<tr><td colspan="5">No cadence comparisons available.</td></tr>'
    )
    entity = html.escape(str(report.get("entity", "")))
    project = html.escape(str(report.get("project", "")))
    group = html.escape(str(report.get("group", "")))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-30B-A3B cadence report</title>
<style>
body {{ background: #10151c; color: #e6edf3; font: 15px system-ui, sans-serif; margin: 2rem; }}
h1, h2 {{ color: #8bd5ff; }}
table {{ border-collapse: collapse; display: block; max-width: 100%; overflow-x: auto; }}
th, td {{ border: 1px solid #3d4854; padding: .45rem; text-align: left; white-space: nowrap; }}
th {{ background: #1c2733; }}
.notice {{ background: #202d3a; border-left: 4px solid #8bd5ff; padding: 1rem; }}
</style>
</head>
<body>
<h1>Qwen3-30B-A3B draft cadence, steps 3–200</h1>
<p>Entity: {entity} · Project: {project} · Group: {group}</p>
<p class="notice">This is a cadence-relative comparison. Always and fixed-10 arms are compared only with their same-drafter static arm. The matrix has no matched no-SpecDec baseline, so it does not claim SpecDec-versus-baseline speedup.</p>
<h2>Run summaries</h2>
<p>Incomplete histories remain visible and are labelled preliminary. Throughput is the canonical logged W&amp;B value, never reconstructed from timing.</p>
<table>
<thead><tr><th>Arm</th><th>Status</th><th>Generation throughput</th><th>Generation time</th><th>E2E throughput</th><th>E2E step time</th><th>Policy training</th><th>Policy/reference logprob</th><th>Refit</th><th>Acceptance rate</th><th>Mean accepted length</th><th>Window steps</th><th>Valid metric observations</th><th>Cadence reasons</th></tr></thead>
<tbody>{rendered_runs}</tbody>
</table>
<h2>Matched static comparisons</h2>
<table>
<thead><tr><th>Arm</th><th>Static baseline</th><th>Status</th><th>Generation throughput speedup</th><th>E2E throughput speedup</th></tr></thead>
<tbody>{rendered_comparisons}</tbody>
</table>
</body>
</html>
"""


def _terminal_cadence_summary(
    artifact_root: Path, run: Mapping[str, object]
) -> tuple[dict[str, int], dict[str, object]] | None:
    """Read the terminal, ledger-validated cadence summary for one launcher run."""
    name = run.get("name")
    if not isinstance(name, str) or RUN_NAME_PATTERN.fullmatch(name) is None:
        return None
    path = artifact_root / "artifacts" / name / "cadence" / "schedule-runtime.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"terminal cadence artifact is not an object: {path}")
    raw_counts = payload.get("decision_reason_counts")
    current_step = payload.get("current_step")
    decision_count = payload.get("decision_count")
    if (
        not isinstance(raw_counts, Mapping)
        or type(current_step) is not int
        or type(decision_count) is not int
    ):
        raise ValueError(f"terminal cadence artifact lacks summary fields: {path}")
    counts: dict[str, int] = {}
    for reason, count in raw_counts.items():
        if not isinstance(reason, str) or type(count) is not int or count < 0:
            raise ValueError(
                f"terminal cadence artifact has invalid reason counts: {path}"
            )
        counts[reason] = count
    if sum(counts.values()) != decision_count:
        raise ValueError(f"terminal cadence artifact reason counts do not sum: {path}")
    return (
        dict(sorted(counts.items())),
        {
            "source": "terminal cadence schedule-runtime.json",
            "path": str(path.resolve()),
            "current_step": current_step,
            "decision_count": decision_count,
        },
    )


def _normalize_run(
    run: Mapping[str, object], cadence_artifact_root: Path | None = None
) -> dict[str, object]:
    """Normalize allowed run metadata and aggregate its supplied history."""
    history = run.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("each run must provide a history sequence")
    rows = [row for row in history if isinstance(row, Mapping)]
    variant = _variant(run)
    if variant is None:
        raise ValueError("each run must provide a variant or name")
    summary = aggregate_history(rows, start_step=3, end_step=200)
    if cadence_artifact_root is not None:
        terminal_summary = _terminal_cadence_summary(cadence_artifact_root, run)
        if terminal_summary is not None:
            reason_counts, provenance = terminal_summary
            summary["cadence_reason_counts"] = reason_counts
            summary["cadence_reason_counts_provenance"] = provenance
    return {
        "id": str(run.get("id", "")),
        "name": str(run.get("name", "")),
        "variant": variant,
        "group": str(run.get("group", "")),
        "state": str(run.get("state", "")),
        "created_at": str(run.get("created_at", "")),
        "summary": summary,
    }


def build_report(
    runs: Sequence[Mapping[str, object]],
    *,
    entity: str,
    project: str,
    group: str,
    cadence_artifact_root: Path | None = None,
) -> dict[str, object]:
    """Build deterministic report data from normalized W&B-style run records."""
    normalized_runs = [
        _normalize_run(run, cadence_artifact_root)
        for run in _select_latest_retries(runs)
    ]
    return {
        "entity": entity,
        "project": project,
        "group": group,
        "window": {"start_step": 3, "end_step": 200},
        "runs": normalized_runs,
        "comparisons": build_comparisons(normalized_runs),
    }


def collect_wandb_runs(
    *, entity: str, project: str, group: str
) -> list[dict[str, object]]:
    """Collect only the named W&B project/group with the required history window."""
    # Import dynamically so offline reporting and static checks do not require W&B.
    wandb = importlib.import_module("wandb")

    api = wandb.Api()
    runs: list[dict[str, object]] = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        history = list(run.scan_history(min_step=3, max_step=201))
        runs.append(
            {
                "id": run.id,
                "name": run.name,
                "group": run.group,
                "state": run.state,
                "created_at": str(run.created_at),
                "config": dict(run.config),
                "history": history,
            }
        )
    return runs


def _write_atomically(path: Path, content: str) -> None:
    """Replace one report artifact only after its complete content is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _offline_runs(path: Path) -> list[Mapping[str, object]]:
    """Load the portable JSON fixture format used by offline verification."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("offline history JSON must be an object with a runs list")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, Mapping) for run in runs):
        raise ValueError("offline history JSON must contain a runs list")
    return runs


def main() -> None:
    """Collect the requested W&B group or render an equivalent offline fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="sna")
    parser.add_argument("--project", default="sna-specdec")
    parser.add_argument("--group", default="q30ba3b-draft-cadence-200step-20260826")
    parser.add_argument("--history-json", type=Path)
    parser.add_argument(
        "--cadence-artifact-root",
        type=Path,
        help="durable experiment root containing artifacts/<run>/cadence",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args()
    if args.json_output is None and args.html_output is None:
        parser.error("provide --json-output, --html-output, or both")

    runs = (
        _offline_runs(args.history_json)
        if args.history_json is not None
        else collect_wandb_runs(
            entity=args.entity, project=args.project, group=args.group
        )
    )
    report = build_report(
        runs,
        entity=args.entity,
        project=args.project,
        group=args.group,
        cadence_artifact_root=args.cadence_artifact_root,
    )
    if args.json_output is not None:
        _write_atomically(
            args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    if args.html_output is not None:
        _write_atomically(args.html_output, render_html(report))


if __name__ == "__main__":
    main()
