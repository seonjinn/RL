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

"""Validate the two-step synchronous GRPO tail-gate smoke."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, cast

from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    REQUIRED_ROW_FIELDS,
    ComparisonRow,
    RunSummary,
    WandbApi,
    _claim_output_directory,
    _empty_comparison_row,
    _empty_summary,
    _history_keys,
    _is_finite_number,
    _read_manifest,
    _validate_manifest_rows,
    _write_atomic,
    _write_csv,
    build_comparison_rows,
    summarize_history,
)


MINI_STEPS = {1, 2}
MINI_THRESHOLD = 32
MINI_METRIC_KEYS = {
    "tail_gate_k0_steps": "train/vllm/tail_gate_k_0_steps",
    "tail_gate_k5_steps": "train/vllm/tail_gate_k_5_steps",
}
MINI_ROW_FIELDS = (
    *REQUIRED_ROW_FIELDS,
    "mini_health_passed",
    *MINI_METRIC_KEYS,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="nvidia")
    parser.add_argument("--project", default="nemorl-vllm024-mini-sync-grpo")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _create_wandb_api() -> WandbApi:
    wandb = importlib.import_module("wandb")
    return cast(WandbApi, wandb.Api())


def _records_by_step(
    history: Iterable[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    return {
        step: record
        for record in history
        if isinstance((step := record.get("_step")), int)
        and not isinstance(step, bool)
        and step in MINI_STEPS
    }


def _positive_metric(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    return _is_finite_number(value) and value > 0.0


def _mini_failure(
    summary: RunSummary,
    history: Iterable[Mapping[str, object]],
    metadata: Mapping[str, str],
) -> str | None:
    if summary.status != "final":
        return summary.reason
    records = _records_by_step(history)
    if set(records) != MINI_STEPS:
        return (
            f"missing_steps:{','.join(map(str, sorted(MINI_STEPS - records.keys())))}"
        )
    for record in records.values():
        if not _positive_metric(record, "timing/train/policy_training"):
            return "policy_training"
        if not _positive_metric(record, "timing/train/policy_and_reference_logprobs"):
            return "policy_and_reference_logprobs"

    if metadata.get("gate_mode") != "threshold":
        return None
    try:
        threshold = int(metadata["threshold"])
    except (KeyError, ValueError):
        return "threshold"
    if threshold != MINI_THRESHOLD:
        return "threshold"
    for record in records.values():
        checks = (
            (
                "tail_gate_activations",
                "train/vllm/tail_gate_activations",
                lambda value: value == 1.0,
            ),
            (
                "activation_tick",
                "train/vllm/tail_gate_activation_tick",
                lambda value: value > 0.0,
            ),
            (
                "activation_batch",
                "train/vllm/tail_gate_activation_batch",
                lambda value: 1.0 <= value <= threshold,
            ),
            (
                "gate_enabled_ratio",
                "train/vllm/tail_gate_enabled_step_ratio",
                lambda value: 0.0 < value < 1.0,
            ),
            (
                "gate_advance_only_ratio",
                "train/vllm/tail_gate_advance_only_step_ratio",
                lambda value: 0.0 < value < 1.0,
            ),
            (
                "tail_gate_k0_steps",
                MINI_METRIC_KEYS["tail_gate_k0_steps"],
                lambda value: value > 0.0,
            ),
            (
                "tail_gate_k5_steps",
                MINI_METRIC_KEYS["tail_gate_k5_steps"],
                lambda value: value > 0.0,
            ),
            ("num_drafts", "train/vllm/spec_num_drafts", lambda value: value > 0.0),
            (
                "num_accepted_tokens",
                "train/vllm/spec_num_accepted_tokens",
                lambda value: value > 0.0,
            ),
        )
        for name, key, predicate in checks:
            value = record.get(key)
            if not _is_finite_number(value) or not predicate(float(value)):
                return name
    return None


def _activation_events(
    metadata: Mapping[str, str], history: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    if metadata.get("gate_mode") != "threshold":
        return []
    events: list[dict[str, object]] = []
    for step, record in sorted(_records_by_step(history).items()):
        tick = record.get("train/vllm/tail_gate_activation_tick")
        batch = record.get("train/vllm/tail_gate_activation_batch")
        activations = record.get("train/vllm/tail_gate_activations")
        if (
            _is_finite_number(tick)
            and _is_finite_number(batch)
            and _is_finite_number(activations)
            and activations > 0.0
        ):
            events.append(
                {
                    "job_id": metadata["job_id"],
                    "step": step,
                    "tick": float(tick),
                    "batch": float(batch),
                    "variant": metadata["variant"],
                }
            )
    return events


def _render_activation_scatter(events: list[dict[str, object]]) -> str:
    ordered = sorted(
        events,
        key=lambda event: (
            cast(str, event["variant"]),
            cast(str, event["job_id"]),
            cast(int, event["step"]),
            cast(float, event["tick"]),
            cast(float, event["batch"]),
        ),
    )
    width = 460
    height = 220
    left = 52
    top = 18
    plot_width = 382
    plot_height = 152
    max_tick = max([MINI_THRESHOLD, *(cast(float, event["tick"]) for event in ordered)])
    max_batch = max(
        [float(MINI_THRESHOLD), *(cast(float, event["batch"]) for event in ordered)]
    )

    def x(value: float) -> float:
        return left + plot_width * value / max_tick

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / max_batch)

    threshold_y = y(float(MINI_THRESHOLD))
    fragments = [
        '<section class="tail-gate-activation-events">',
        "<style>.tail-gate-activation-events{font:13px sans-serif}.tail-gate-activation-events svg{border:1px solid #c9c9c9}.tail-gate-activation-events .axis{stroke:#333}.tail-gate-activation-events .threshold{stroke:#c55;stroke-dasharray:4 3}.tail-gate-activation-events .event{fill:#76b900}.tail-gate-activation-events text{fill:#222}</style>",
        "<p>This two-step smoke makes no speedup claim.</p>",
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Tail-gate activation events">',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<line class="threshold" x1="{left}" y1="{threshold_y:.1f}" x2="{left + plot_width}" y2="{threshold_y:.1f}"/>',
        f'<text x="{left + 4}" y="{threshold_y - 4:.1f}">threshold=32</text>',
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle">Scheduler tick</text>',
        f'<text x="14" y="{top + plot_height / 2:.1f}" transform="rotate(-90 14 {top + plot_height / 2:.1f})" text-anchor="middle">Inflight batch</text>',
    ]
    for event in ordered:
        tick = cast(float, event["tick"])
        batch = cast(float, event["batch"])
        label = (
            f"OFF-to-ON: {html.escape(cast(str, event['variant']))} "
            f"step={cast(int, event['step'])} tick={tick:g} batch={batch:g}"
        )
        fragments.append(
            f'<circle class="event" cx="{x(tick):.1f}" cy="{y(batch):.1f}" r="4"><title>{label}</title></circle>'
        )
        fragments.append(
            f'<text x="{x(tick) + 6:.1f}" y="{y(batch) - 6:.1f}">{label}</text>'
        )
    fragments.extend(["</svg>", "</section>\n"])
    return "\n".join(fragments)


def _mini_row(
    comparison: ComparisonRow,
    failure: str | None,
    history: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    row = comparison.to_dict()
    records = _records_by_step(history)
    row["mini_health_passed"] = row["status"] == "final" and failure is None
    for name, key in MINI_METRIC_KEYS.items():
        values = [record.get(key) for record in records.values()]
        finite_values = [float(value) for value in values if _is_finite_number(value)]
        row[name] = (
            statistics.fmean(finite_values)
            if len(finite_values) == len(MINI_STEPS)
            else None
        )
    return row


def main(argv: list[str] | None = None, *, api: WandbApi | None = None) -> int:
    """Validate all mini-smoke manifest rows and render deterministic artifacts."""
    args = _parse_args(argv)
    manifest_rows = _read_manifest(args.manifest)
    manifest_error = _validate_manifest_rows(manifest_rows)
    if manifest_error:
        raise ValueError(manifest_error)
    _claim_output_directory(args.output_dir)

    client = api if api is not None else _create_wandb_api()
    summaries: list[RunSummary] = []
    histories: dict[str, list[Mapping[str, object]]] = {}
    failures: dict[str, str | None] = {}
    events: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        metadata = {
            **manifest_row,
            "source": manifest_row.get("source") or args.manifest.name,
        }
        job_id = metadata["job_id"]
        try:
            run = client.run(f"{args.entity}/{args.project}/{metadata['wandb_run_id']}")
            if not metadata.get("wandb_url"):
                metadata["wandb_url"] = run.url
            history = list(
                run.scan_history(
                    keys=[*_history_keys(metadata), *MINI_METRIC_KEYS.values()]
                )
            )
            histories[job_id] = history
            summary = summarize_history(metadata, history, expected_steps=MINI_STEPS)
            failure = _mini_failure(summary, history, metadata)
            if summary.status == "final" and failure is not None:
                summary = replace(
                    summary,
                    status="health_failed",
                    reason=f"mini_health_failed:{failure}",
                )
            summaries.append(summary)
            failures[job_id] = failure
            events.extend(_activation_events(metadata, history))
        except Exception as error:  # Preserve every failed W&B row in the report.
            summaries.append(
                _empty_summary(
                    metadata, f"wandb_fetch_failed:{type(error).__name__}", []
                )
            )
            histories[job_id] = []
            failures[job_id] = f"wandb_fetch_failed:{type(error).__name__}"

    try:
        comparisons = build_comparison_rows(summaries)
    except ValueError as error:
        comparisons = [
            _empty_comparison_row(
                replace(summary, status="partial", reason=f"comparison_failed:{error}")
            )
            for summary in summaries
        ]
    rows = [
        _mini_row(
            comparison,
            failures[comparison.summary.job_id],
            histories[comparison.summary.job_id],
        )
        for comparison in sorted(
            comparisons,
            key=lambda comparison: (
                comparison.summary.runner,
                comparison.summary.model,
                comparison.summary.variant,
                comparison.summary.job_id,
            ),
        )
    ]
    _write_atomic(
        args.output_dir / "mini_summary.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(args.output_dir / "mini_summary.csv", rows, fieldnames=MINI_ROW_FIELDS)
    _write_atomic(
        args.output_dir / "tail_gate_activation_events.html",
        _render_activation_scatter(events),
    )
    return int(
        any(row["status"] != "final" or not row["mini_health_passed"] for row in rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
