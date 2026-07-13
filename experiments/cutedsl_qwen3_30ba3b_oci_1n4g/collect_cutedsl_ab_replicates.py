#!/usr/bin/env python3
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

"""Collect deterministic paired CuTeDSL ON/OFF replicate statistics."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    """A measured raw-timing field included in the paired aggregate."""

    name: str
    category: str
    field: str


METRIC_SPECS = (
    MetricSpec("e2e_duration", "duration", "total_step_seconds"),
    MetricSpec("generation_duration", "duration", "generation_seconds"),
    MetricSpec(
        "generation_finalize_duration", "duration", "generation_finalize_seconds"
    ),
    MetricSpec("logprob_duration", "duration", "logprob_seconds"),
    MetricSpec("policy_training_duration", "duration", "policy_training_seconds"),
    MetricSpec("refit_duration", "duration", "refit_transfer_update_seconds"),
    MetricSpec("e2e_throughput", "throughput", "e2e_tokens_per_sec_per_gpu"),
    MetricSpec(
        "generation_throughput",
        "throughput",
        "generation_tokens_per_sec_per_gpu",
    ),
    MetricSpec(
        "logprob_throughput",
        "throughput",
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    ),
    MetricSpec(
        "policy_training_throughput",
        "throughput",
        "policy_training_tokens_per_sec_per_gpu",
    ),
    MetricSpec(
        "refit_effective_throughput",
        "throughput",
        "refit_effective_tokens_per_sec_per_gpu",
    ),
)
REQUIRED_CANONICAL_METRICS = frozenset(
    {
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/generation_finalize",
        "timing/train/get_logprobs",
        "timing/train/policy_training",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "train/total_num_tokens",
        "train/global_valid_toks",
        "train/mean_prompt_length",
        "train/num_valid_samples",
        "train/total_turns",
    }
)
WORKLOAD_TOKEN_FIELDS = ("total_num_tokens", "global_valid_toks")
WORKLOAD_EXACT_OBSERVED_FIELDS = (
    "mean_prompt_length",
    "num_valid_samples",
    "total_turns",
)
WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT = 0.01
WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT = 0.02
THROUGHPUT_DURATION_FIELDS = {
    "e2e_tokens_per_sec_per_gpu": "total_step_seconds",
    "generation_tokens_per_sec_per_gpu": "generation_seconds",
    "policy_and_reference_logprobs_tokens_per_sec_per_gpu": "logprob_seconds",
    "policy_training_tokens_per_sec_per_gpu": "policy_training_seconds",
    "refit_effective_tokens_per_sec_per_gpu": "refit_transfer_update_seconds",
}
RAW_FIELD_BY_CANONICAL_METRIC = {
    "timing/train/total_step_time": "total_step_seconds",
    "timing/train/generation": "generation_seconds",
    "timing/train/generation_finalize": "generation_finalize_seconds",
    "timing/train/get_logprobs": "logprob_seconds",
    "timing/train/policy_training": "policy_training_seconds",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "refit_transfer_update_seconds"
    ),
    "performance/tokens_per_sec_per_gpu": "e2e_tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu": (
        "generation_tokens_per_sec_per_gpu"
    ),
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu"
    ),
    "performance/policy_training_tokens_per_sec_per_gpu": (
        "policy_training_tokens_per_sec_per_gpu"
    ),
    "train/total_num_tokens": "total_num_tokens",
    "train/global_valid_toks": "global_valid_toks",
    "train/mean_prompt_length": "mean_prompt_length",
    "train/num_valid_samples": "num_valid_samples",
    "train/total_turns": "total_turns",
}
VALID_ORDERS = frozenset({"on,off", "off,on"})
ORDERED_TIMING_ORDERS = ("on,off", "off,on")
RATIO_DEFINITION = "median(on measured steps) / median(off measured steps)"
CSV_FIELDS = (
    "scope",
    "metric",
    "category",
    "ratio_definition",
    "replicate_count",
    "replicate_index",
    "job_id",
    "timing_order",
    "ratio",
    "median_ratio",
    "replicate_median_cv_percent",
    "ci95_lower",
    "ci95_upper",
    "extend_to_six",
    "recommendation_reasons",
)


class CollectorError(ValueError):
    """Raised when submitted benchmark evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class Replicate:
    """Validated inputs and paired ratios for one submitted job."""

    replicate_index: int
    job_id: str
    run_id: str
    result_dir: Path
    timing_order: str
    profile_enabled: bool
    submission_group: str
    source_identity: str
    image_identity: str
    workload_identity: str
    metric_identity: str
    measured_workload_identity: str
    workload_equivalence: dict[str, Any]
    ratios: dict[str, float]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CollectorError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CollectorError(f"{label} {path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CollectorError(f"{label} must be a nonempty string")
    return value


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CollectorError(f"{label} must be a non-negative integer")
    return value


def _require_profile_flag(value: Any, label: str) -> bool:
    if value not in (0, 1, False, True):
        raise CollectorError(f"{label} must be 0 or 1")
    return bool(value)


def _load_submission(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise CollectorError(f"cannot read submission JSONL {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CollectorError(
                f"submission JSONL {path}:{line_number} is invalid: {error}"
            ) from error
        if not isinstance(record, dict):
            raise CollectorError(
                f"submission JSONL {path}:{line_number} must contain an object"
            )
        records.append(record)
    return records


def _contained_file(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise CollectorError(f"{label} escapes benchmark result root")
    if not resolved.is_file():
        raise CollectorError(f"{label} does not exist as a regular file: {path}")
    return resolved


def _safe_artifact(root: Path, job_dir: Path, relative_path: Any, label: str) -> Path:
    relative = Path(_require_string(relative_path, label))
    if relative.is_absolute():
        raise CollectorError(f"{label} must be relative to job result directory")
    job_root = job_dir.resolve()
    path = _contained_file(root, job_dir / relative, label)
    if path == job_root or job_root not in path.parents:
        raise CollectorError(f"{label} escapes job result directory: {relative}")
    return path


def _require_sha(value: Any, *, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None
    ):
        raise CollectorError(f"{label} must be a {length}-character hexadecimal SHA")
    return value


def _validate_artifact_revisions(manifest: dict[str, Any], job_id: str) -> None:
    revisions = manifest.get("artifact_revisions")
    expected = {
        "model": ("Qwen/Qwen3-30B-A3B", None),
        "dataset": ("nvidia/OpenMathInstruct-2", "dataset"),
    }
    if not isinstance(revisions, dict) or set(revisions) != set(expected):
        raise CollectorError(
            f"job {job_id} artifact_revisions must contain model and dataset"
        )
    for label, (repo_id, repo_type) in expected.items():
        repository = revisions[label]
        if not isinstance(repository, dict):
            raise CollectorError(f"job {job_id} {label} revision must be an object")
        if (
            repository.get("repo_id") != repo_id
            or repository.get("repo_type") != repo_type
        ):
            raise CollectorError(f"job {job_id} {label} repository identity differs")
        _require_sha(
            repository.get("revision"),
            length=40,
            label=f"job {job_id} {label} revision",
        )
        if label == "dataset":
            if repository.get("split") != "train_1M":
                raise CollectorError(f"job {job_id} dataset split differs")
            if (
                _require_nonnegative_integer(
                    repository.get("num_rows"),
                    f"job {job_id} dataset num_rows",
                )
                == 0
            ):
                raise CollectorError(f"job {job_id} dataset num_rows must be positive")


def _validate_manifest_identity(manifest: dict[str, Any], job_id: str) -> None:
    if (
        manifest.get("functional_gate") is True
        or manifest.get("performance_eligible") is False
    ):
        raise CollectorError(
            f"job {job_id} functional-gate evidence is not performance eligible"
        )
    _require_sha(manifest.get("source_sha"), length=40, label="source_sha")
    _require_string(manifest.get("upstream_ref"), "upstream_ref")
    _require_sha(manifest.get("upstream_sha"), length=40, label="upstream_sha")
    _require_string(manifest.get("image"), "image")
    _require_sha(manifest.get("image_sha256"), length=64, label="image_sha256")
    _require_sha(
        manifest.get("base_config_sha256"),
        length=64,
        label="base_config_sha256",
    )
    _validate_artifact_revisions(manifest, job_id)
    _require_string(manifest.get("recipe"), "recipe")
    warmup_updates = _require_nonnegative_integer(
        manifest.get("warmup_updates"), f"job {job_id} warmup_updates"
    )
    measured_updates = _require_nonnegative_integer(
        manifest.get("measured_updates"), f"job {job_id} measured_updates"
    )
    total_updates = _require_nonnegative_integer(
        manifest.get("total_updates"), f"job {job_id} total_updates"
    )
    if warmup_updates == 0 or measured_updates == 0:
        raise CollectorError(
            f"job {job_id} warmup and measured updates must be positive"
        )
    if total_updates != warmup_updates + measured_updates:
        raise CollectorError(
            f"job {job_id} total_updates must equal warmup_updates + measured_updates"
        )
    topology = manifest.get("topology")
    if not isinstance(topology, dict) or not topology:
        raise CollectorError("topology must be a nonempty object")
    fixed_config = manifest.get("fixed_config_evidence")
    if (
        not isinstance(fixed_config, dict)
        or set(fixed_config) != {"on", "off"}
        or any(
            not isinstance(fixed_config[arm], dict) or not fixed_config[arm]
            for arm in ("on", "off")
        )
    ):
        raise CollectorError("fixed_config_evidence must contain ON/OFF objects")
    if fixed_config["on"] != fixed_config["off"]:
        raise CollectorError("ON/OFF fixed_config_evidence must match")


def _validate_manifest_metrics(
    manifest: dict[str, Any], job_id: str
) -> dict[str, dict[str, str]]:
    metrics = manifest.get("resolved_metric_names")
    if not isinstance(metrics, dict) or set(metrics) != {"on", "off"}:
        raise CollectorError(
            f"job {job_id} manifest must contain ON/OFF resolved metric names"
        )
    validated = {}
    for arm in ("on", "off"):
        mapping = metrics[arm]
        if not isinstance(mapping, dict) or set(mapping) != REQUIRED_CANONICAL_METRICS:
            raise CollectorError(
                f"job {job_id} {arm.upper()} resolved metric names must contain "
                "the exact canonical metric set"
            )
        for canonical_name, source_name in mapping.items():
            if not isinstance(source_name, str) or not source_name:
                raise CollectorError(
                    f"job {job_id} resolved metric name for {canonical_name!r} "
                    "must be a nonempty string"
                )
        validated[arm] = mapping
    if validated["on"] != validated["off"]:
        raise CollectorError(f"job {job_id} ON/OFF resolved metric names must match")
    return validated


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CollectorError(f"{label} must be finite and positive")
    return result


def _validate_metric_names(
    raw: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    resolved = raw.get("resolved_metric_names")
    if not isinstance(resolved, dict):
        raise CollectorError(f"{label} lacks resolved_metric_names")
    missing = sorted(REQUIRED_CANONICAL_METRICS - resolved.keys())
    if missing:
        raise CollectorError(f"{label} is missing resolved metric names: {missing}")
    if resolved != expected:
        raise CollectorError(f"{label} resolved metric names differ from manifest")


def _validate_component_series(
    raw: dict[str, Any],
    workload: list[dict[str, Any]],
    expected_steps: list[int],
    job_id: str,
    arm: str,
) -> None:
    series_by_metric = raw.get("measured_component_series")
    if not isinstance(series_by_metric, dict) or set(series_by_metric) != set(
        RAW_FIELD_BY_CANONICAL_METRIC
    ):
        raise CollectorError(
            f"job {job_id} {arm.upper()} measured_component_series must contain "
            "the exact canonical metric set"
        )
    for canonical_name, row_field in RAW_FIELD_BY_CANONICAL_METRIC.items():
        series = series_by_metric[canonical_name]
        if not isinstance(series, list) or len(series) != len(workload):
            raise CollectorError(
                f"job {job_id} {arm.upper()} {canonical_name} series length differs"
            )
        series_steps = []
        for row_index, (point, row) in enumerate(zip(series, workload, strict=True)):
            if not isinstance(point, dict):
                raise CollectorError(
                    f"job {job_id} {arm.upper()} {canonical_name} point must be an object"
                )
            step = _require_nonnegative_integer(
                point.get("step"),
                f"job {job_id} {arm.upper()} {canonical_name} series step",
            )
            series_steps.append(step)
            series_value = _numeric(
                point.get("value"),
                f"job {job_id} {arm.upper()} {canonical_name} series value",
            )
            row_value = _numeric(
                row.get(row_field),
                f"job {job_id} {arm.upper()} {row_field} row {row_index}",
            )
            if series_value != row_value:
                raise CollectorError(
                    f"job {job_id} {arm.upper()} {canonical_name} measured row "
                    "differs from component series"
                )
        if series_steps != expected_steps:
            raise CollectorError(
                f"job {job_id} {arm.upper()} {canonical_name} series step window "
                "differs from manifest"
            )
    policy_seconds = raw.get("policy_training_seconds")
    expected_policy_seconds = [row["policy_training_seconds"] for row in workload]
    if policy_seconds != expected_policy_seconds:
        raise CollectorError(
            f"job {job_id} {arm.upper()} policy_training_seconds differs from measured rows"
        )


def _load_raw_timing(
    root: Path,
    job_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    job_id: str,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    raw_files = summary.get("raw_timing_files")
    if not isinstance(raw_files, list) or len(raw_files) != 2:
        raise CollectorError(
            f"job {job_id} must reference exactly two raw timing files"
        )
    manifest_metrics = _validate_manifest_metrics(manifest, job_id)
    measured_updates = _require_nonnegative_integer(
        manifest.get("measured_updates"), f"job {job_id} manifest measured_updates"
    )
    warmup_updates = _require_nonnegative_integer(
        manifest.get("warmup_updates"), f"job {job_id} manifest warmup_updates"
    )
    total_updates = _require_nonnegative_integer(
        manifest.get("total_updates"), f"job {job_id} manifest total_updates"
    )
    if total_updates != warmup_updates + measured_updates:
        raise CollectorError(
            f"job {job_id} total_updates must equal warmup_updates plus measured_updates"
        )
    expected_steps = list(range(warmup_updates + 1, total_updates + 1))
    topology = manifest.get("topology")
    if not isinstance(topology, dict):
        raise CollectorError(f"job {job_id} manifest topology must be an object")
    expected_training_gpu_count = _require_nonnegative_integer(
        topology.get("num_nodes"), f"job {job_id} topology num_nodes"
    ) * _require_nonnegative_integer(
        topology.get("gpus_per_node"), f"job {job_id} topology gpus_per_node"
    )

    by_arm = {}
    order_indices = set()
    for raw_file in raw_files:
        path = _safe_artifact(root, job_dir, raw_file, f"job {job_id} raw timing file")
        raw = _read_json(path, f"job {job_id} raw timing")
        arm = raw.get("arm")
        if arm not in ("on", "off") or arm in by_arm:
            raise CollectorError(
                f"job {job_id} raw timing arms must be exactly ON and OFF"
            )
        if raw.get("run_id") != run_id:
            raise CollectorError(f"job {job_id} raw timing run_id does not match run")
        order_index = _require_nonnegative_integer(
            raw.get("order_index"), f"job {job_id} {arm.upper()} raw order_index"
        )
        timing_order = summary["timing_order"]
        if (
            order_index not in (0, 1)
            or order_index in order_indices
            or timing_order[order_index] != arm
        ):
            raise CollectorError(
                f"job {job_id} raw timing order_index does not evidence timing order"
            )
        order_indices.add(order_index)
        _validate_metric_names(
            raw,
            manifest_metrics[arm],
            f"job {job_id} {arm.upper()} raw timing",
        )
        workload = raw.get("measured_step_workload")
        if not isinstance(workload, list) or len(workload) != measured_updates:
            raise CollectorError(
                f"job {job_id} {arm.upper()} measured_step_workload must contain "
                f"exactly {measured_updates} rows"
            )
        if raw.get("measured_updates") != measured_updates:
            raise CollectorError(
                f"job {job_id} {arm.upper()} raw measured_updates differs from manifest"
            )
        if raw.get("warmup_updates") != warmup_updates:
            raise CollectorError(
                f"job {job_id} {arm.upper()} raw warmup_updates differs from manifest"
            )
        if raw.get("training_gpu_count") != expected_training_gpu_count:
            raise CollectorError(
                f"job {job_id} {arm.upper()} training_gpu_count differs from topology"
            )
        raw_steps = [row.get("step") for row in workload if isinstance(row, dict)]
        if raw_steps != expected_steps:
            raise CollectorError(
                f"job {job_id} {arm.upper()} measured step sequence differs from "
                "manifest window"
            )
        _validate_component_series(raw, workload, expected_steps, job_id, arm)
        by_arm[arm] = raw
    if set(by_arm) != {"on", "off"}:
        raise CollectorError(f"job {job_id} raw timing arms must be exactly ON and OFF")
    if order_indices != {0, 1}:
        raise CollectorError(
            f"job {job_id} raw timing order_index does not evidence timing order"
        )
    return by_arm


def _workload_equivalence(
    by_arm: dict[str, dict[str, Any]], job_id: str
) -> tuple[dict[str, Any], str]:
    steps_by_arm = {}
    exact_values_by_arm = {}
    for arm in ("on", "off"):
        rows = by_arm[arm]["measured_step_workload"]
        steps = []
        exact_rows = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CollectorError(
                    f"job {job_id} {arm.upper()} workload row {row_index} must be an object"
                )
            step = _require_nonnegative_integer(
                row.get("step"), f"job {job_id} {arm.upper()} workload step"
            )
            steps.append(step)
            exact_rows.append(
                tuple(
                    _numeric(
                        row.get(field),
                        f"job {job_id} {arm.upper()} {field} step {step}",
                    )
                    for field in WORKLOAD_EXACT_OBSERVED_FIELDS
                )
            )
            for field, value in zip(
                WORKLOAD_EXACT_OBSERVED_FIELDS, exact_rows[-1], strict=True
            ):
                if (
                    field in ("num_valid_samples", "total_turns")
                    and not value.is_integer()
                ):
                    raise CollectorError(
                        f"job {job_id} {arm.upper()} {field} step {step} "
                        "must be an integral count"
                    )
        if steps != sorted(set(steps)):
            raise CollectorError(
                f"job {job_id} {arm.upper()} measured step sequence must be ordered and unique"
            )
        steps_by_arm[arm] = steps
        exact_values_by_arm[arm] = exact_rows
    if steps_by_arm["on"] != steps_by_arm["off"]:
        raise CollectorError(
            f"job {job_id} measured step sequence differs across ON/OFF arms"
        )
    exact_invariants_observed = exact_values_by_arm["on"] == exact_values_by_arm["off"]
    metrics = {}
    observed = exact_invariants_observed
    for field in WORKLOAD_TOKEN_FIELDS:
        values = {}
        for arm in ("on", "off"):
            values[arm] = [
                _numeric(
                    row.get(field),
                    f"job {job_id} {arm.upper()} {field} row {row_index}",
                )
                for row_index, row in enumerate(by_arm[arm]["measured_step_workload"])
            ]
            if any(value <= 0.0 for value in values[arm]):
                raise CollectorError(
                    f"job {job_id} {arm.upper()} {field} must be positive"
                )
            if any(not value.is_integer() for value in values[arm]):
                raise CollectorError(
                    f"job {job_id} {arm.upper()} {field} must contain integral counts"
                )
        on_total = sum(values["on"])
        off_total = sum(values["off"])
        arm_total_relative_delta = abs(on_total - off_total) / (
            (on_total + off_total) / 2.0
        )
        max_paired_step_relative_delta = max(
            abs(on_value - off_value) / ((on_value + off_value) / 2.0)
            for on_value, off_value in zip(values["on"], values["off"], strict=True)
        )
        metrics[field] = {
            "on_total": on_total,
            "off_total": off_total,
            "arm_total_relative_delta": arm_total_relative_delta,
            "max_paired_step_relative_delta": max_paired_step_relative_delta,
        }
        observed = observed and (
            arm_total_relative_delta <= WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT
            and max_paired_step_relative_delta
            <= WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT
        )
    for arm in ("on", "off"):
        for row_index, row in enumerate(by_arm[arm]["measured_step_workload"]):
            total_tokens = _numeric(
                row.get("total_num_tokens"),
                f"job {job_id} {arm.upper()} total_num_tokens row {row_index}",
            )
            valid_tokens = _numeric(
                row.get("global_valid_toks"),
                f"job {job_id} {arm.upper()} global_valid_toks row {row_index}",
            )
            if valid_tokens > total_tokens:
                raise CollectorError(
                    f"job {job_id} {arm.upper()} global_valid_toks exceeds "
                    f"total_num_tokens at row {row_index}"
                )
    equivalence = {
        "schema_version": 2,
        "relative_delta_formula": "abs(on-off)/mean(on,off)",
        "required": True,
        "observed": observed,
        "actual_token_normalization_required": True,
        "normalization_metric": "train/total_num_tokens",
        "exact_observed_invariants": {
            "fields": list(WORKLOAD_EXACT_OBSERVED_FIELDS),
            "observed": exact_invariants_observed,
        },
        "prompt_sequence_identity_verified": False,
        "limits": {
            "arm_total_relative_delta": WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT,
            "paired_step_relative_delta": WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT,
        },
        "metrics": metrics,
    }
    identity = _canonical_json(
        {
            "steps": steps_by_arm["on"],
            "exact_observed_fields": list(WORKLOAD_EXACT_OBSERVED_FIELDS),
            "exact_observed_values": exact_values_by_arm["on"],
            "prompt_sequence_identity_verified": False,
            "limits": equivalence["limits"],
            "actual_token_normalization_required": True,
            "normalization_metric": "train/total_num_tokens",
            "relative_delta_formula": "abs(on-off)/mean(on,off)",
        }
    )
    return equivalence, identity


def _paired_ratios(by_arm: dict[str, dict[str, Any]], job_id: str) -> dict[str, float]:
    ratios = {}
    for spec in METRIC_SPECS:
        medians = {}
        for arm in ("on", "off"):
            rows = by_arm[arm]["measured_step_workload"]
            values = [
                _numeric(
                    row.get(spec.field),
                    f"job {job_id} {arm.upper()} {spec.field} row {row_index}",
                )
                for row_index, row in enumerate(rows)
            ]
            medians[arm] = statistics.median(values)
        ratios[spec.name] = medians["on"] / medians["off"]
    return ratios


def _validate_actual_token_normalization(
    by_arm: dict[str, dict[str, Any]], job_id: str
) -> None:
    for arm in ("on", "off"):
        training_gpu_count = _numeric(
            by_arm[arm].get("training_gpu_count"),
            f"job {job_id} {arm.upper()} training_gpu_count",
        )
        for row_index, row in enumerate(by_arm[arm]["measured_step_workload"]):
            total_tokens = _numeric(
                row.get("total_num_tokens"),
                f"job {job_id} {arm.upper()} total_num_tokens row {row_index}",
            )
            for throughput_field, duration_field in THROUGHPUT_DURATION_FIELDS.items():
                duration = _numeric(
                    row.get(duration_field),
                    f"job {job_id} {arm.upper()} {duration_field} row {row_index}",
                )
                observed = _numeric(
                    row.get(throughput_field),
                    f"job {job_id} {arm.upper()} {throughput_field} row {row_index}",
                )
                expected = total_tokens / duration / training_gpu_count
                if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
                    raise CollectorError(
                        f"job {job_id} {arm.upper()} {throughput_field} row "
                        f"{row_index} is not normalized by actual total_num_tokens"
                    )


def _validate_summary_projections(
    summary: dict[str, Any],
    by_arm: dict[str, dict[str, Any]],
    job_id: str,
) -> None:
    expected_tokens = {
        arm: [row["total_num_tokens"] for row in by_arm[arm]["measured_step_workload"]]
        for arm in ("on", "off")
    }
    if summary.get("measured_total_num_tokens") != expected_tokens:
        raise CollectorError(
            f"job {job_id} measured_total_num_tokens differs from raw timing"
        )
    expected_policy_medians = {
        arm: statistics.median(by_arm[arm]["policy_training_seconds"])
        for arm in ("on", "off")
    }
    if summary.get("median_policy_training_seconds") != expected_policy_medians:
        raise CollectorError(
            f"job {job_id} median_policy_training_seconds differs from raw timing"
        )
    expected_throughput_medians = {
        arm: statistics.median(
            row["policy_training_tokens_per_sec_per_gpu"]
            for row in by_arm[arm]["measured_step_workload"]
        )
        for arm in ("on", "off")
    }
    if summary.get("median_normalized_throughput") != expected_throughput_medians:
        raise CollectorError(
            f"job {job_id} median_normalized_throughput differs from raw timing"
        )


def _validate_profile_attribution(root: Path, job_dir: Path, job_id: str) -> None:
    attribution_paths = sorted(job_dir.rglob("kernel_attribution.json"))
    if len(attribution_paths) != 1:
        raise CollectorError(
            f"designated profile job {job_id} expected exactly one "
            f"kernel_attribution.json, found {len(attribution_paths)}"
        )
    attribution_path = _contained_file(
        root,
        attribution_paths[0],
        f"designated profile job {job_id} kernel attribution",
    )
    attribution = _read_json(
        attribution_path, f"designated profile job {job_id} kernel attribution"
    )
    if attribution.get("passed") is not True:
        raise CollectorError(
            f"designated profile job {job_id} kernel attribution did not pass"
        )
    arms = attribution.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"on", "off"}:
        raise CollectorError(
            f"designated profile job {job_id} attribution must contain ON/OFF arms"
        )
    for arm in ("on", "off"):
        result = arms[arm]
        if not isinstance(result, dict):
            raise CollectorError(
                f"designated profile job {job_id} {arm.upper()} attribution is invalid"
            )
        _safe_artifact(
            root,
            job_dir,
            result.get("kernel_evidence"),
            f"designated profile job {job_id} {arm.upper()} kernel evidence",
        )
        if (
            _require_nonnegative_integer(
                result.get("grouped_gemm_match_count"),
                f"designated profile job {job_id} {arm.upper()} grouped GEMM match count",
            )
            == 0
        ):
            raise CollectorError(
                f"designated profile job {job_id} {arm.upper()} lacks grouped GEMM attribution"
            )
    for field in ("fused_glu_match_count", "fused_dglu_match_count"):
        on_count = _require_nonnegative_integer(
            arms["on"].get(field),
            f"designated profile job {job_id} ON {field}",
        )
        off_count = _require_nonnegative_integer(
            arms["off"].get(field),
            f"designated profile job {job_id} OFF {field}",
        )
        if on_count == 0 or off_count != 0:
            raise CollectorError(
                f"designated profile job {job_id} has invalid {field} attribution"
            )

    profile_paths = sorted(job_dir.glob("profiles/*/profile_summary.json"))
    if len(profile_paths) != 2:
        raise CollectorError(
            f"designated profile job {job_id} expected two profile summaries, "
            f"found {len(profile_paths)}"
        )
    profile_arms = set()
    for path in profile_paths:
        path = _contained_file(
            root, path, f"designated profile job {job_id} profile summary"
        )
        profile = _read_json(path, f"designated profile job {job_id} profile summary")
        arm = profile.get("arm")
        if arm not in ("on", "off") or arm in profile_arms:
            raise CollectorError(
                f"designated profile job {job_id} profile arms must be ON and OFF"
            )
        profile_arms.add(arm)
        if (
            _require_nonnegative_integer(
                profile.get("nsight_report_count"),
                f"designated profile job {job_id} {arm.upper()} Nsight report count",
            )
            == 0
        ):
            raise CollectorError(
                f"designated profile job {job_id} {arm.upper()} has no Nsight report"
            )
        _safe_artifact(
            root,
            path.parent,
            profile.get("kernel_evidence"),
            f"designated profile job {job_id} {arm.upper()} profile kernel evidence",
        )


def _find_completed_run(root: Path, job_id: str) -> tuple[Path, str]:
    if (
        Path(job_id).is_absolute()
        or Path(job_id).name != job_id
        or job_id in (".", "..")
        or "/" in job_id
        or "\\" in job_id
    ):
        raise CollectorError("submission job_id must be a safe single path component")
    root = root.resolve()
    if not root.is_dir():
        raise CollectorError(f"benchmark result root is not a directory: {root}")
    candidate_pattern = re.compile(rf"{re.escape(job_id)}(?:-r[1-9][0-9]*)?")
    candidates = sorted(
        path for path in root.iterdir() if candidate_pattern.fullmatch(path.name)
    )
    if not candidates:
        raise CollectorError(
            f"submitted job {job_id} result directory is missing under {root}"
        )

    successful = []
    failed_exit_codes = []
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if (
            resolved_candidate == root
            or root not in resolved_candidate.parents
            or not resolved_candidate.is_dir()
        ):
            raise CollectorError(
                f"job {job_id} result directory escapes benchmark root"
            )
        status_path = _contained_file(
            root, candidate / "status.json", f"job {job_id} status"
        )
        status = _read_json(status_path, f"job {job_id} status")
        if status.get("job_id") != job_id or status.get("run_id") != candidate.name:
            raise CollectorError(
                f"job {job_id} status identity does not match submission"
            )
        exit_code = status.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise CollectorError(f"job {job_id} status exit_code must be an integer")
        if exit_code == 0:
            successful.append((resolved_candidate, candidate.name))
        else:
            failed_exit_codes.append(exit_code)
    if len(successful) > 1:
        raise CollectorError(
            f"job {job_id} has multiple successfully completed run directories"
        )
    if not successful:
        detail = (
            failed_exit_codes[0] if len(failed_exit_codes) == 1 else failed_exit_codes
        )
        raise CollectorError(
            f"job {job_id} is not completed successfully: exit_code={detail!r}"
        )
    return successful[0]


def _load_replicate(root: Path, record: dict[str, Any]) -> Replicate:
    replicate_index = _require_nonnegative_integer(
        record.get("replicate_index"), "submission replicate_index"
    )
    job_id = _require_string(record.get("job_id"), "submission job_id")
    timing_order = _require_string(
        record.get("timing_order"), f"submission timing order for job {job_id}"
    )
    expected_order = "on,off" if replicate_index % 2 == 0 else "off,on"
    if timing_order != expected_order:
        raise CollectorError(
            f"submission timing order for replicate {replicate_index} must be "
            f"{expected_order}, found {timing_order}"
        )
    profile_enabled = _require_profile_flag(
        record.get("profile_enabled"), f"submission profile_enabled for job {job_id}"
    )
    submission_group = _require_string(
        record.get("submission_group"),
        f"submission group for job {job_id}",
    )
    job_dir, run_id = _find_completed_run(root, job_id)

    manifest_path = job_dir / "benchmark_manifest.json"
    manifest_path = _contained_file(
        root, manifest_path, f"job {job_id} benchmark manifest"
    )
    manifest = _read_json(manifest_path, f"job {job_id} benchmark manifest")
    manifest_contracts = {
        "run_id": run_id,
        "replicate_index": replicate_index,
        "timing_order": timing_order.split(","),
        "profile_enabled": profile_enabled,
        "submission_group": submission_group,
    }
    for field, expected in manifest_contracts.items():
        if manifest.get(field) != expected:
            raise CollectorError(
                f"job {job_id} manifest {field} differs from submission: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    _validate_manifest_identity(manifest, job_id)

    timing_paths = sorted(job_dir.rglob("timing_summary.json"))
    if len(timing_paths) != 1:
        raise CollectorError(
            f"job {job_id} expected exactly one timing_summary.json, "
            f"found {len(timing_paths)}"
        )
    summary_path = _contained_file(
        root, timing_paths[0], f"job {job_id} timing summary"
    )
    summary = _read_json(summary_path, f"job {job_id} timing summary")
    if summary.get("run_id") != run_id:
        raise CollectorError(f"job {job_id} timing summary run_id does not match")
    if summary.get("timing_order") != timing_order.split(","):
        raise CollectorError(
            f"job {job_id} timing summary order differs from submission"
        )
    if summary.get("workload_metric") != "train/total_num_tokens":
        raise CollectorError(
            f"job {job_id} workload metric must be train/total_num_tokens"
        )

    by_arm = _load_raw_timing(root, job_dir, summary, manifest, job_id, run_id)
    workload_equivalence, measured_workload_identity = _workload_equivalence(
        by_arm, job_id
    )
    if _canonical_json(summary.get("workload_equivalence")) != _canonical_json(
        workload_equivalence
    ):
        raise CollectorError(
            f"job {job_id} workload equivalence summary does not match raw timing"
        )
    if not workload_equivalence["observed"]:
        raise CollectorError(f"job {job_id} workload equivalence limits exceeded")
    _validate_actual_token_normalization(by_arm, job_id)
    _validate_summary_projections(summary, by_arm, job_id)
    ratios = _paired_ratios(by_arm, job_id)
    source_identity = _canonical_json(
        {
            field: manifest.get(field)
            for field in ("source_sha", "upstream_ref", "upstream_sha")
        }
    )
    image_identity = _canonical_json(
        {field: manifest.get(field) for field in ("image", "image_sha256")}
    )
    workload_identity = _canonical_json(
        {
            field: manifest.get(field)
            for field in (
                "recipe",
                "base_config_sha256",
                "artifact_revisions",
                "warmup_updates",
                "measured_updates",
                "total_updates",
                "topology",
                "fixed_config_evidence",
            )
        }
    )
    metric_identity = _canonical_json(manifest.get("resolved_metric_names"))
    return Replicate(
        replicate_index=replicate_index,
        job_id=job_id,
        run_id=run_id,
        result_dir=job_dir,
        timing_order=timing_order,
        profile_enabled=profile_enabled,
        submission_group=submission_group,
        source_identity=source_identity,
        image_identity=image_identity,
        workload_identity=workload_identity,
        metric_identity=metric_identity,
        measured_workload_identity=measured_workload_identity,
        workload_equivalence=workload_equivalence,
        ratios=ratios,
    )


def _clean_float(value: float) -> float:
    return round(value, 12)


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] + fraction * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )


def _bootstrap_ci(
    ratios: list[float], *, samples: int, seed: int, stream_name: str
) -> dict[str, float]:
    digest = hashlib.sha256(f"{seed}:{stream_name}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    count = len(ratios)
    estimates = sorted(
        statistics.median(generator.choices(ratios, k=count)) for _ in range(samples)
    )
    return {
        "lower": _clean_float(_percentile(estimates, 0.025)),
        "upper": _clean_float(_percentile(estimates, 0.975)),
    }


def _summarize_ratios(
    ratios: list[float], *, samples: int, seed: int, stream_name: str
) -> dict[str, Any]:
    median_ratio = _clean_float(statistics.median(ratios))
    cv_percent = None
    if len(ratios) > 1:
        cv_percent = _clean_float(
            statistics.stdev(ratios) / statistics.mean(ratios) * 100.0
        )
    ci = _bootstrap_ci(ratios, samples=samples, seed=seed, stream_name=stream_name)
    reasons = []
    if cv_percent is not None and cv_percent > 5.0:
        reasons.append("CV exceeds 5%")
    if ci["lower"] <= 1.0 <= ci["upper"]:
        reasons.append("CI crosses 1")
    return {
        "replicate_count": len(ratios),
        "median_ratio": median_ratio,
        "replicate_median_cv_percent": cv_percent,
        "paired_bootstrap_ci95": ci,
        "recommendation": {
            "extend_to_six": bool(reasons),
            "reasons": reasons,
        },
    }


def _validate_replicates(replicates: list[Replicate]) -> Replicate:
    job_ids = [replicate.job_id for replicate in replicates]
    replicate_indices = [replicate.replicate_index for replicate in replicates]
    if len(replicates) < 3 or len(set(job_ids)) != len(replicates):
        raise CollectorError(
            "at least 3 distinct completed replicate jobs are required"
        )
    if len(set(replicate_indices)) != len(replicates):
        raise CollectorError("replicate indices must be distinct")
    orders = {replicate.timing_order for replicate in replicates}
    if orders != VALID_ORDERS:
        raise CollectorError(
            f"alternating timing orders must both be represented; found {sorted(orders)}"
        )
    profiles = [replicate for replicate in replicates if replicate.profile_enabled]
    if len(profiles) != 1:
        raise CollectorError(
            f"exactly one designated profile replicate is required; found {len(profiles)}"
        )
    identity_contracts = (
        ("submission group differs", "submission_group"),
        ("source identity differs", "source_identity"),
        ("image identity differs", "image_identity"),
        ("workload identity differs", "workload_identity"),
        ("resolved metric names differ", "metric_identity"),
        ("measured workload differs", "measured_workload_identity"),
    )
    for failure, attribute in identity_contracts:
        values = {getattr(replicate, attribute) for replicate in replicates}
        if len(values) != 1:
            raise CollectorError(f"{failure} across replicates")
    return profiles[0]


def collect(
    submission_path: Path,
    result_root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate replicate artifacts and return JSON and long-form CSV records."""
    records = _load_submission(submission_path)
    replicates = sorted(
        (_load_replicate(result_root, record) for record in records),
        key=lambda replicate: replicate.replicate_index,
    )
    profile_replicate = _validate_replicates(replicates)
    _validate_profile_attribution(
        result_root,
        profile_replicate.result_dir,
        profile_replicate.job_id,
    )

    metrics = {}
    csv_rows = []
    extension_metrics = []
    for spec in METRIC_SPECS:
        replicate_values = [replicate.ratios[spec.name] for replicate in replicates]
        summary = _summarize_ratios(
            replicate_values,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            stream_name=f"aggregate:{spec.name}",
        )
        replicate_rows = [
            {
                "replicate_index": replicate.replicate_index,
                "job_id": replicate.job_id,
                "timing_order": replicate.timing_order,
                "ratio": _clean_float(replicate.ratios[spec.name]),
            }
            for replicate in replicates
        ]
        order_summaries = {}
        for order in ORDERED_TIMING_ORDERS:
            order_values = [
                replicate.ratios[spec.name]
                for replicate in replicates
                if replicate.timing_order == order
            ]
            order_summaries[order] = _summarize_ratios(
                order_values,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                stream_name=f"order:{order}:{spec.name}",
            )
        metrics[spec.name] = {
            "category": spec.category,
            "measured_field": spec.field,
            "replicates": replicate_rows,
            **summary,
            "order_stratified": order_summaries,
        }
        if summary["recommendation"]["extend_to_six"]:
            extension_metrics.append(spec.name)

        for replicate_row in replicate_rows:
            csv_rows.append(
                {
                    "scope": "replicate",
                    "metric": spec.name,
                    "category": spec.category,
                    "ratio_definition": RATIO_DEFINITION,
                    "replicate_count": 1,
                    **replicate_row,
                }
            )
        csv_rows.append(
            _summary_csv_row(
                scope="aggregate",
                metric=spec.name,
                category=spec.category,
                timing_order="",
                summary=summary,
            )
        )
        for order, order_summary in order_summaries.items():
            csv_rows.append(
                _summary_csv_row(
                    scope="order",
                    metric=spec.name,
                    category=spec.category,
                    timing_order=order,
                    summary=order_summary,
                )
            )

    first = replicates[0]
    aggregate = {
        "schema_version": 2,
        "submission_jsonl": str(submission_path),
        "benchmark_result_root": str(result_root),
        "ratio_definition": RATIO_DEFINITION,
        "duration_ratio_interpretation": "values below 1 favor ON",
        "throughput_ratio_interpretation": "values above 1 favor ON",
        "replicate_count": len(replicates),
        "replicate_indices": [replicate.replicate_index for replicate in replicates],
        "job_ids": [replicate.job_id for replicate in replicates],
        "run_ids": [replicate.run_id for replicate in replicates],
        "timing_orders": list(ORDERED_TIMING_ORDERS),
        "profile_replicate": {
            "replicate_index": profile_replicate.replicate_index,
            "job_id": profile_replicate.job_id,
            "run_id": profile_replicate.run_id,
        },
        "submission_group": first.submission_group,
        "source": json.loads(first.source_identity),
        "image": json.loads(first.image_identity),
        "workload": json.loads(first.workload_identity),
        "workload_equivalence": [
            {
                "replicate_index": replicate.replicate_index,
                "job_id": replicate.job_id,
                **replicate.workload_equivalence,
            }
            for replicate in replicates
        ],
        "resolved_metric_names": json.loads(first.metric_identity),
        "bootstrap": {
            "method": "paired replicate resampling of median ratios",
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "recommendation": {
            "extend_to_six": bool(extension_metrics),
            "metrics": extension_metrics,
            "rule": "extend when replicate-median CV exceeds 5% or paired CI crosses 1",
        },
        "metrics": metrics,
    }
    return aggregate, csv_rows


def _summary_csv_row(
    *,
    scope: str,
    metric: str,
    category: str,
    timing_order: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    recommendation = summary["recommendation"]
    return {
        "scope": scope,
        "metric": metric,
        "category": category,
        "ratio_definition": RATIO_DEFINITION,
        "replicate_count": summary["replicate_count"],
        "replicate_index": "",
        "job_id": "",
        "timing_order": timing_order,
        "ratio": "",
        "median_ratio": summary["median_ratio"],
        "replicate_median_cv_percent": (
            ""
            if summary["replicate_median_cv_percent"] is None
            else summary["replicate_median_cv_percent"]
        ),
        "ci95_lower": summary["paired_bootstrap_ci95"]["lower"],
        "ci95_upper": summary["paired_bootstrap_ci95"]["upper"],
        "extend_to_six": recommendation["extend_to_six"],
        "recommendation_reasons": "; ".join(recommendation["reasons"]),
    }


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(content)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_jsonl", type=Path)
    parser.add_argument("benchmark_result_root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2606)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    if args.output_json is None:
        args.output_json = args.submission_jsonl.with_suffix(".aggregate.json")
    if args.output_csv is None:
        args.output_csv = args.submission_jsonl.with_suffix(".aggregate.csv")
    if args.output_json.resolve() == args.output_csv.resolve():
        parser.error("--output-json and --output-csv must differ")
    if args.submission_jsonl.resolve() in {
        args.output_json.resolve(),
        args.output_csv.resolve(),
    }:
        parser.error("output paths must not overwrite the submission JSONL")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the replicate collector command-line interface."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        aggregate, csv_rows = collect(
            args.submission_jsonl,
            args.benchmark_result_root,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        _atomic_write(
            args.output_json,
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(args.output_csv, _render_csv(csv_rows))
    except (CollectorError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[INFO] Wrote aggregate JSON: {args.output_json}")
    print(f"[INFO] Wrote aggregate CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
