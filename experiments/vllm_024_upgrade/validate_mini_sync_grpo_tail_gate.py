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
import re
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, cast
from urllib.parse import unquote, urlparse

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
DEFAULT_WANDB_ENTITY = "nvidia"
DEFAULT_WANDB_PROJECT = "nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche"
REQUIRED_COMMON_CONFIG = {
    "model": "qwen32b",
    "cluster": "pre-tyche",
    "runtime": "nemo-rl",
    "recipe": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
    "target_tp": "2",
    "draft_tp": "1",
    "dp": "8",
    "ep": "1",
    "temperature": "1.0",
    "top_p": "1.0",
    "max_osl": "1024",
    "max_model_len": "1056",
    "max_sequence_length": "1024",
    "num_prompts": "16",
    "num_generations": "4",
    "train_gbs": "64",
    "max_num_batched_tokens": "16384",
    "max_num_seqs": "1024",
    "runner": "v2",
    "graph_mode": "FULL_AND_PIECEWISE",
    "sampling": "standard",
}
REQUIRED_VARIANT_CONFIG = {
    "baseline_v2": {
        "gate_mode": "off",
        "k": "0",
        "threshold": "",
        "consecutive_checks": "",
        "draft_sample_method": "not_applicable",
    },
    "always_on_v2_k5": {
        "gate_mode": "off",
        "k": "5",
        "threshold": "",
        "consecutive_checks": "",
        "draft_sample_method": "probabilistic",
    },
    "fastrl_threshold_v2_k5": {
        "gate_mode": "threshold",
        "k": "5",
        "consecutive_checks": "10",
        "draft_sample_method": "probabilistic",
    },
}
MINI_METRIC_KEYS = {
    "tail_gate_k0_steps": "train/vllm/tail_gate_k_0_steps",
    "tail_gate_k5_steps": "train/vllm/tail_gate_k_5_steps",
}
MINI_ROW_FIELDS = (
    *REQUIRED_ROW_FIELDS,
    "mini_health_passed",
    *MINI_METRIC_KEYS,
)
SLURM_LOG_FIELDS = ("slurm_log", "slurm_log_path", "log_path")
SLURM_FAILURE_PATTERNS = (
    (
        "stale_draft_id",
        re.compile(
            r"(?<!no )\bstale draft (?:ids?|token ids?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "invalid_token",
        re.compile(
            r"(?:\b(?:valueerror|runtimeerror|error|exception|assertionerror)\b.*"
            r"\binvalid tokens?(?: ids?)?\b|(?<!no )\binvalid tokens?"
            r"(?: ids?)?\b.*\b(?:generated|detected|found|observed|returned|"
            r"produced)\b)",
            re.IGNORECASE,
        ),
    ),
    ("tokens_left_for_obs", re.compile(r"\btokens_left_for_obs\s*=\s*-\d+\b")),
    (
        "nan",
        re.compile(
            r"(?:\b(?:loss|reward|logprobs?|gradients?|metrics?)\b\s*"
            r"(?:is|are|=|:|contains?)\s*nan\b|\bnan detected in\b.*"
            r"\b(?:loss|reward|logprobs?|gradients?|metrics?)\b|"
            r"\b(?:found|detected|encountered)\s+nan\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "oom",
        re.compile(
            r"\b(?:cuda out of memory|outofmemoryerror|oom-kill(?:er)?|"
            r"killed process.*out of memory|(?:runtimeerror|error|exception)"
            r"\b.*\boom\b)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "nccl",
        re.compile(
            r"(?:\bnccl\b.*\b(?:watchdog\s+)?(?:timed out|timeout detected|"
            r"hang detected|hung|aborted)\b|\bwatchdog caught collective "
            r"operation timeout\b.*\bworknccl\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "q_cache",
        re.compile(
            r"(?:\bq[-_ ]?cache\b.*\b(?:error|mismatch|failed|failure|corrupt)"
            r"\w*\b|\b(?:assertionerror|runtimeerror|error|exception)\b.*"
            r"\bq[-_ ]?cache\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "cuda_graph_fallback",
        re.compile(
            r"(?:\bcuda[ _]?graphs?\b.*\b(?:fallback to eager|"
            r"falling back to eager|uncaptured)\b|\buncaptured "
            r"cuda[ _]?graphs?\b|\bcuda[ _]?graphs?\s+fallback\b"
            r"(?!\s+(?:count\s*[:=]\s*0|disabled)\b)|\beager[ _-]?fallback"
            r"(?![ _]?count\s*[:=]\s*0\b)(?:\s+(?:detected|used|occurred)\b|"
            r"[ _]?count\s*[:=]\s*[1-9]\d*\b))",
            re.IGNORECASE,
        ),
    ),
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _create_wandb_api() -> WandbApi:
    wandb = importlib.import_module("wandb")
    return cast(WandbApi, wandb.Api())


def _wandb_run_path_from_url(url: str, *, variant: str, expected_run_id: str) -> str:
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or len(parts) != 4
        or parts[2] != "runs"
        or not parts[0]
        or not parts[1]
        or not parts[3]
    ):
        raise ValueError(f"invalid wandb_url:{variant}:{url}")
    entity, project, _, run_id = parts
    if run_id != expected_run_id:
        raise ValueError(
            f"wandb_url run ID mismatch:{variant}:{run_id}:{expected_run_id}"
        )
    return f"{entity}/{project}/{run_id}"


def _command_override_values(command: str, key: str) -> set[str]:
    pattern = re.compile(rf"(?:^|\s){re.escape(key)}=([^\s]+)(?=\s|$)")
    return {match.group(1) for match in pattern.finditer(command)}


def _validate_mini_manifest_rows(rows: list[dict[str, str]]) -> str | None:
    variants = sorted(row.get("variant", "") for row in rows)
    required_variants = sorted(REQUIRED_VARIANT_CONFIG)
    if variants != required_variants:
        return (
            "mini manifest variants must be exactly:"
            f"{','.join(required_variants)}:got:{','.join(variants)}"
        )
    for row in rows:
        variant = row["variant"]
        for field, expected in REQUIRED_COMMON_CONFIG.items():
            actual = row.get(field, "")
            if actual != expected:
                return (
                    f"invalid mini manifest field:{variant}:{field}:{actual}:{expected}"
                )
        for field, expected in REQUIRED_VARIANT_CONFIG[variant].items():
            actual = row.get(field, "")
            if actual != expected:
                return (
                    f"invalid mini manifest field:{variant}:{field}:{actual}:{expected}"
                )
        if variant == "fastrl_threshold_v2_k5":
            try:
                threshold = int(row.get("threshold", ""))
            except ValueError:
                return f"invalid mini manifest field:{variant}:threshold"
            if threshold <= 0:
                return f"invalid mini manifest field:{variant}:threshold"
        checkpointing_enabled = row.get("checkpointing_enabled", "")
        command = row.get("command", "")
        if checkpointing_enabled and checkpointing_enabled.lower() != "false":
            return f"invalid mini manifest provenance:{variant}:checkpointing_enabled"
        if command:
            if _command_override_values(command, "checkpointing.enabled") != {"false"}:
                return f"invalid mini manifest provenance:{variant}:command"
            if _command_override_values(command, "grpo.max_num_steps") != {"2"}:
                return f"invalid mini manifest provenance:{variant}:command"
        if not checkpointing_enabled and not command:
            return f"invalid mini manifest provenance:{variant}:checkpointing"
        if row.get("wandb_url"):
            try:
                _wandb_run_path_from_url(
                    row["wandb_url"],
                    variant=variant,
                    expected_run_id=row["wandb_run_id"],
                )
            except ValueError as error:
                return str(error)
    return None


def _mini_threshold(rows: Iterable[Mapping[str, str]]) -> int:
    threshold_row = next(
        row for row in rows if row["variant"] == "fastrl_threshold_v2_k5"
    )
    return int(threshold_row["threshold"])


def _wandb_run_path(
    metadata: Mapping[str, str], *, fallback_entity: str, fallback_project: str
) -> str:
    url = metadata.get("wandb_url", "")
    if url:
        return _wandb_run_path_from_url(
            url,
            variant=metadata["variant"],
            expected_run_id=metadata["wandb_run_id"],
        )
    return f"{fallback_entity}/{fallback_project}/{metadata['wandb_run_id']}"


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


def _slurm_log_path(manifest: Path, metadata: Mapping[str, str]) -> Path:
    for field in SLURM_LOG_FIELDS:
        value = metadata.get(field, "")
        if value:
            path = Path(value.replace("%j", metadata["job_id"]))
            return path if path.is_absolute() else manifest.parent / path
    return (
        manifest.parent
        / metadata["model"]
        / metadata["variant"]
        / f"slurm-{metadata['job_id']}.out"
    )


def _slurm_log_failure(manifest: Path, metadata: Mapping[str, str]) -> str | None:
    path = _slurm_log_path(manifest, metadata)
    if not path.is_file():
        return "slurm_log_missing"
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for reason, pattern in SLURM_FAILURE_PATTERNS:
                if pattern.search(line):
                    return f"slurm_log:{reason}"
    return None


def _mini_failure(
    summary: RunSummary,
    history: Iterable[Mapping[str, object]],
    metadata: Mapping[str, str],
    *,
    threshold: int,
    slurm_log_failure: str | None,
) -> str | None:
    if summary.status != "final":
        return summary.reason
    if slurm_log_failure is not None:
        return slurm_log_failure
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


def _render_activation_scatter(
    events: list[dict[str, object]], *, threshold: int
) -> str:
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
    max_tick = max(
        [float(threshold), *(cast(float, event["tick"]) for event in ordered)]
    )
    max_batch = max(
        [float(threshold), *(cast(float, event["batch"]) for event in ordered)]
    )

    def x(value: float) -> float:
        return left + plot_width * value / max_tick

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / max_batch)

    threshold_y = y(float(threshold))
    fragments = [
        '<section class="tail-gate-activation-events">',
        "<style>.tail-gate-activation-events{font:13px sans-serif}.tail-gate-activation-events svg{border:1px solid #c9c9c9}.tail-gate-activation-events .axis{stroke:#333}.tail-gate-activation-events .threshold{stroke:#c55;stroke-dasharray:4 3}.tail-gate-activation-events .event{fill:#76b900}.tail-gate-activation-events text{fill:#222}</style>",
        "<p>This two-step smoke makes no speedup claim.</p>",
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Tail-gate activation events">',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<line class="threshold" x1="{left}" y1="{threshold_y:.1f}" x2="{left + plot_width}" y2="{threshold_y:.1f}"/>',
        f'<text x="{left + 4}" y="{threshold_y - 4:.1f}">threshold={threshold}</text>',
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
    mini_manifest_error = _validate_mini_manifest_rows(manifest_rows)
    if mini_manifest_error:
        raise ValueError(mini_manifest_error)
    threshold = _mini_threshold(manifest_rows)
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
            run = client.run(
                _wandb_run_path(
                    metadata,
                    fallback_entity=args.entity,
                    fallback_project=args.project,
                )
            )
            if not metadata.get("wandb_url"):
                metadata["wandb_url"] = run.url
            history = list(
                run.scan_history(
                    keys=[*_history_keys(metadata), *MINI_METRIC_KEYS.values()]
                )
            )
            histories[job_id] = history
            summary = summarize_history(metadata, history, expected_steps=MINI_STEPS)
            failure = _mini_failure(
                summary,
                history,
                metadata,
                threshold=threshold,
                slurm_log_failure=_slurm_log_failure(args.manifest, metadata),
            )
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
        _render_activation_scatter(events, threshold=threshold),
    )
    return int(
        any(row["status"] != "final" or not row["mini_health_passed"] for row in rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
