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

"""Collect a claim-safe dependency-constrained NeMo 26.06 factorial cohort."""

import argparse
import hashlib
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


class CollectorError(ValueError):
    """Raised when factorial evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ComponentSpec:
    """Duration and throughput fields for a reported component."""

    name: str
    duration_field: str
    throughput_field: str


COMPONENT_SPECS = (
    ComponentSpec("e2e", "total_step_seconds", "e2e_tokens_per_sec_per_gpu"),
    ComponentSpec(
        "generation",
        "generation_seconds",
        "generation_tokens_per_sec_per_gpu",
    ),
    ComponentSpec(
        "policy_training",
        "policy_training_seconds",
        "policy_training_tokens_per_sec_per_gpu",
    ),
    ComponentSpec(
        "refit",
        "refit_transfer_update_seconds",
        "refit_effective_tokens_per_sec_per_gpu",
    ),
)
CONTEXTS = ("g0a0", "g1a0", "g0a1", "g1a1")
CONTEXT_FLAGS = {
    "g0a0": (False, False),
    "g1a0": (True, False),
    "g0a1": (False, True),
    "g1a1": (True, True),
}
DIRECT_PATH_TARGETS = {
    "cutedsl_only": ("g0a0", "on"),
    "a2a_only_without_cutedsl": ("g0a1", "off"),
    "cutedsl_a2a": ("g0a1", "on"),
    "cutedsl_full_cg": ("g1a0", "on"),
    "all_three_combined": ("g1a1", "on"),
}
REPRESENTATIVE_LIMITATION_FRAGMENT = (
    "only one representative process/rank analyzed; no all-rank aggregation"
)
A2A_CONFIG_FIELDS = (
    "policy.megatron_cfg.overlap_moe_expert_parallel_comm",
    "policy.megatron_cfg.high_priority_a2a_comm_stream",
    "policy.megatron_cfg.delay_wgrad_compute",
)
REQUIRED_CANONICAL_METRICS = frozenset(
    {
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/generation_finalize",
        "timing/train/policy_training",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "train/total_num_tokens",
        "train/global_valid_toks",
        "train/mean_prompt_length",
        "train/num_valid_samples",
        "train/total_turns",
    }
)
OPTIONAL_ZERO_LOGPROB_METRICS = {
    "timing/train/get_logprobs": "logprob_seconds",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu"
    ),
}
RAW_FIELD_BY_CANONICAL_METRIC = {
    "timing/train/total_step_time": "total_step_seconds",
    "timing/train/generation": "generation_seconds",
    "timing/train/generation_finalize": "generation_finalize_seconds",
    "timing/train/policy_training": "policy_training_seconds",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "refit_transfer_update_seconds"
    ),
    "performance/tokens_per_sec_per_gpu": "e2e_tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu": (
        "generation_tokens_per_sec_per_gpu"
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
THROUGHPUT_DURATION_FIELDS = {
    "e2e_tokens_per_sec_per_gpu": "total_step_seconds",
    "generation_tokens_per_sec_per_gpu": "generation_seconds",
    "policy_training_tokens_per_sec_per_gpu": "policy_training_seconds",
    "refit_effective_tokens_per_sec_per_gpu": "refit_transfer_update_seconds",
}
WORKLOAD_EXACT_FIELDS = ("mean_prompt_length", "num_valid_samples", "total_turns")
WORKLOAD_TOKEN_FIELDS = ("total_num_tokens", "global_valid_toks")
WORKLOAD_TOTAL_DELTA_LIMIT = 0.01
WORKLOAD_STEP_DELTA_LIMIT = 0.02
SUBMISSION_RECORD_FIELDS = frozenset(
    {
        "factorial_context",
        "full_cg_enabled",
        "a2a_enabled",
        "replicate_index",
        "timing_order",
        "profile_enabled",
        "job_id",
        "submission_group",
    }
)


@dataclass(frozen=True)
class ArmObservation:
    """Validated component and workload values for one timing arm."""

    arm: str
    rows: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    components: dict[str, dict[str, float]]


@dataclass(frozen=True)
class FactorialRun:
    """Validated evidence for one context and replicate."""

    context: str
    replicate_index: int
    job_id: str
    run_id: str
    timing_order: tuple[str, ...]
    profile_enabled: bool
    source_identity: str
    image_identity: str
    workload_identity: str
    invariant_config_identity: str
    metric_identity: str
    arms: dict[str, ArmObservation]
    evidence_digest: str
    provisional_reasons: tuple[str, ...]
    a2a_overlap_ratio: float | None
    temporal_representative_only: bool


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CollectorError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CollectorError(f"{label} {path} must contain a JSON object")
    return value


def _safe_file(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    lexical = Path(os.path.abspath(path))
    if lexical == root or root not in lexical.parents:
        raise CollectorError(f"{label} escapes benchmark result root")
    current = root
    for part in lexical.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise CollectorError(f"{label} must not contain symlinks")
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise CollectorError(f"{label} escapes benchmark result root")
    if not resolved.is_file():
        raise CollectorError(f"{label} is missing: {path}")
    return resolved


def _safe_relative_file(root: Path, job_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CollectorError(f"{label} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise CollectorError(f"{label} must be relative")
    path = _safe_file(root, job_dir / relative, label)
    job_root = job_dir.resolve()
    if path == job_root or job_root not in path.parents:
        raise CollectorError(f"{label} escapes job result directory")
    return path


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CollectorError(f"{label} must be a non-negative integer")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CollectorError(f"{label} must be finite and positive")
    return result


def _require_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CollectorError(f"{label} must be finite and non-negative")
    return result


def _require_sha(value: Any, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
    ):
        raise CollectorError(f"{label} must be a lowercase {length}-character SHA")
    return value


def _clean(value: float) -> float:
    return round(value, 12)


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
        if not isinstance(record, dict) or set(record) != SUBMISSION_RECORD_FIELDS:
            raise CollectorError(
                f"submission record {line_number} must contain exact factorial fields"
            )
        records.append(record)
    return records


def _validate_submission_matrix(
    records: list[dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    by_context: dict[str, dict[int, dict[str, Any]]] = {
        context: {} for context in CONTEXTS
    }
    seen_job_ids: set[str] = set()
    submission_groups: set[str] = set()
    for record in records:
        context = record["factorial_context"]
        if context not in CONTEXT_FLAGS:
            raise CollectorError(f"unknown factorial context: {context!r}")
        expected_full_cg, expected_a2a = CONTEXT_FLAGS[context]
        if (
            record["full_cg_enabled"] is not expected_full_cg
            or record["a2a_enabled"] is not expected_a2a
        ):
            raise CollectorError(f"submission flags do not match context {context}")
        replicate_index = _require_int(
            record["replicate_index"], f"{context} replicate_index"
        )
        job_id = record["job_id"]
        if (
            not isinstance(job_id, str)
            or not job_id
            or Path(job_id).name != job_id
            or "/" in job_id
            or "\\" in job_id
        ):
            raise CollectorError("submission job_id must be a safe path component")
        if job_id in seen_job_ids:
            raise CollectorError(f"submission job_id is reused: {job_id}")
        if replicate_index in by_context[context]:
            raise CollectorError(
                f"duplicate {context} replicate index {replicate_index}"
            )
        seen_job_ids.add(job_id)
        by_context[context][replicate_index] = record
        submission_group = record["submission_group"]
        if not isinstance(submission_group, str) or not submission_group:
            raise CollectorError("submission_group must be a nonempty string")
        submission_groups.add(submission_group)

    if len(submission_groups) != 1:
        raise CollectorError("submission group differs across factorial cohort")
    index_sets = {context: set(values) for context, values in by_context.items()}
    for context, indices in index_sets.items():
        if len(indices) < 3:
            raise CollectorError(f"context {context} requires at least 3 replicas")
    if len({tuple(sorted(indices)) for indices in index_sets.values()}) != 1:
        raise CollectorError("factorial contexts must have identical replicate indices")

    indices = sorted(next(iter(index_sets.values())))
    expected_record_count = len(indices) * len(CONTEXTS)
    if len(records) != expected_record_count:
        raise CollectorError("submission matrix contains unexpected records")
    for block_index, replicate_index in enumerate(indices):
        block = records[block_index * 4 : (block_index + 1) * 4]
        if {record["replicate_index"] for record in block} != {replicate_index}:
            raise CollectorError("context submission order is not replica-blocked")
        offset = replicate_index % len(CONTEXTS)
        expected_order = (*CONTEXTS[offset:], *CONTEXTS[:offset])
        observed_order = tuple(record["factorial_context"] for record in block)
        if observed_order != expected_order:
            raise CollectorError(
                f"context submission order for replica {replicate_index} is not balanced"
            )

    for context, context_records in by_context.items():
        full_cg_enabled, _ = CONTEXT_FLAGS[context]
        for replicate_index, record in context_records.items():
            expected_order = (
                "on"
                if full_cg_enabled
                else ("on,off" if replicate_index % 2 == 0 else "off,on")
            )
            if record["timing_order"] != expected_order:
                raise CollectorError(
                    f"timing order for {context} replica {replicate_index} must be "
                    f"{expected_order}"
                )
            if record["profile_enabled"] not in (True, False, 0, 1):
                raise CollectorError("profile_enabled must be boolean")
    return by_context


def _find_run(root: Path, job_id: str) -> tuple[Path, str]:
    root = root.resolve()
    if not root.is_dir():
        raise CollectorError(f"benchmark result root is not a directory: {root}")
    pattern = re.compile(rf"{re.escape(job_id)}(?:-r[1-9][0-9]*)?")
    candidates = sorted(path for path in root.iterdir() if pattern.fullmatch(path.name))
    successful = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == root or root not in resolved.parents or not resolved.is_dir():
            raise CollectorError(f"job {job_id} result directory escapes root")
        status = _read_json(
            _safe_file(root, candidate / "status.json", f"job {job_id} status"),
            f"job {job_id} status",
        )
        if status.get("job_id") != job_id or status.get("run_id") != candidate.name:
            raise CollectorError(f"job {job_id} status identity differs")
        exit_code = status.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise CollectorError(f"job {job_id} exit_code must be an integer")
        if exit_code == 0:
            successful.append((resolved, candidate.name, status))
    if len(successful) != 1:
        raise CollectorError(
            f"job {job_id} requires exactly one successful run; found {len(successful)}"
        )
    job_dir, run_id, _ = successful[0]
    return job_dir, run_id


def _validate_identity(manifest: dict[str, Any], job_id: str) -> None:
    if manifest.get("functional_gate") is not False:
        raise CollectorError(f"job {job_id} is functional evidence, not performance")
    if manifest.get("performance_eligible") is not True:
        raise CollectorError(f"job {job_id} is not performance eligible")
    _require_sha(manifest.get("source_sha"), 40, f"job {job_id} source_sha")
    _require_sha(manifest.get("upstream_sha"), 40, f"job {job_id} upstream_sha")
    _require_sha(manifest.get("image_sha256"), 64, f"job {job_id} image_sha256")
    _require_sha(
        manifest.get("base_config_sha256"), 64, f"job {job_id} base_config_sha256"
    )
    if (
        not isinstance(manifest.get("upstream_ref"), str)
        or not manifest["upstream_ref"]
    ):
        raise CollectorError(f"job {job_id} upstream_ref must be nonempty")
    if not isinstance(manifest.get("image"), str) or not manifest["image"]:
        raise CollectorError(f"job {job_id} image must be nonempty")
    revisions = manifest.get("artifact_revisions")
    if not isinstance(revisions, dict) or set(revisions) != {"model", "dataset"}:
        raise CollectorError(f"job {job_id} artifact revisions are incomplete")
    expected = {
        "model": ("Qwen/Qwen3-30B-A3B", None),
        "dataset": ("nvidia/OpenMathInstruct-2", "dataset"),
    }
    for label, (repo_id, repo_type) in expected.items():
        record = revisions[label]
        if (
            not isinstance(record, dict)
            or record.get("repo_id") != repo_id
            or record.get("repo_type") != repo_type
        ):
            raise CollectorError(f"job {job_id} {label} repository identity differs")
        _require_sha(record.get("revision"), 40, f"job {job_id} {label} revision")
    dataset = revisions["dataset"]
    if dataset.get("split") != "train_1M" or dataset.get("num_rows") != 1_000_000:
        raise CollectorError(f"job {job_id} dataset identity differs")


def _validate_contract(manifest: dict[str, Any], context: str, job_id: str) -> None:
    expected_full_cg, expected_a2a = CONTEXT_FLAGS[context]
    if (
        manifest.get("feature_context") != context
        or manifest.get("full_cg_enabled") is not expected_full_cg
        or manifest.get("a2a_enabled") is not expected_a2a
    ):
        raise CollectorError(f"job {job_id} manifest feature context differs")
    topology = manifest.get("topology")
    expected_topology = {
        "num_nodes": 2,
        "gpus_per_node": 4,
        "segment_size": None,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 4,
    }
    if topology != expected_topology:
        raise CollectorError(
            f"job {job_id} resolved topology must match noncolocated EP4 contract"
        )
    workload = manifest.get("workload")
    if not isinstance(workload, dict):
        raise CollectorError(f"job {job_id} workload must be an object")
    if workload.get("train_global_batch_size") != 8:
        raise CollectorError(f"job {job_id} train_global_batch_size must equal 8")
    if workload.get("train_micro_batch_size") != 1:
        raise CollectorError(f"job {job_id} train_micro_batch_size must equal 1")

    expected_arms = tuple(manifest.get("timing_order", ()))
    if expected_full_cg:
        if expected_arms != ("on",) or manifest.get("available_arms") != ["on"]:
            raise CollectorError(f"job {job_id} full-CG context must be ON-only")
        if set(manifest.get("not_applicable_arms", {})) != {"off"}:
            raise CollectorError(
                f"job {job_id} must explicitly mark OFF not applicable"
            )
    elif set(expected_arms) != {"on", "off"}:
        raise CollectorError(f"job {job_id} g0 context must contain ON/OFF arms")

    fixed_config = manifest.get("fixed_config_evidence")
    if not isinstance(fixed_config, dict) or set(fixed_config) != set(expected_arms):
        raise CollectorError(f"job {job_id} fixed config arms differ")
    for arm in expected_arms:
        arm_config = fixed_config[arm]
        if not isinstance(arm_config, dict):
            raise CollectorError(f"job {job_id} {arm} fixed config must be an object")
        if arm_config.get("policy.megatron_cfg.moe_grouped_gemm") is not True:
            raise CollectorError(f"job {job_id} CuTeDSL ON grouped GEMM is not enabled")
        if (
            arm_config.get("policy.megatron_cfg.env_vars.CUDA_DEVICE_MAX_CONNECTIONS")
            != "32"
        ):
            raise CollectorError(
                f"job {job_id} CUDA_DEVICE_MAX_CONNECTIONS must equal string '32'"
            )
        if arm_config.get("policy.train_global_batch_size") != 8:
            raise CollectorError(f"job {job_id} fixed-config GBS must equal 8")
        if arm_config.get("policy.train_micro_batch_size") != 1:
            raise CollectorError(f"job {job_id} fixed-config MBS must equal 1")
        logprob_contract = {
            "loss_fn.force_on_policy_ratio": True,
            "grpo.seq_logprob_error_threshold": None,
            "grpo.skip_reference_policy_logprobs_calculation": True,
            "loss_fn.reference_policy_kl_penalty": 0.0,
        }
        if any(
            arm_config.get(field) != expected
            for field, expected in logprob_contract.items()
        ):
            raise CollectorError(
                f"job {job_id} logprob N/A contract does not match supported full-CG slice"
            )
        for field in A2A_CONFIG_FIELDS:
            if arm_config.get(field) is not expected_a2a:
                raise CollectorError(
                    f"job {job_id} A2A config {field} does not match context"
                )

    graph_config = manifest.get("full_cg_config_evidence")
    if not isinstance(graph_config, dict) or set(graph_config) != set(expected_arms):
        raise CollectorError(f"job {job_id} full-CG config evidence differs")
    for arm in expected_arms:
        record = graph_config[arm]
        expected_impl = "full_iteration" if expected_full_cg else "none"
        if (
            not isinstance(record, dict)
            or record.get("cuda_graph_impl") != expected_impl
        ):
            raise CollectorError(f"job {job_id} cuda_graph_impl differs from context")
        if expected_full_cg and record.get("cuda_graph_use_single_mempool") is not True:
            raise CollectorError(f"job {job_id} full-CG single mempool is not enabled")


def _validate_metric_mapping(mapping: Any, *, job_id: str, arm: str) -> dict[str, str]:
    allowed_sets = {
        REQUIRED_CANONICAL_METRICS,
        REQUIRED_CANONICAL_METRICS | frozenset(OPTIONAL_ZERO_LOGPROB_METRICS),
    }
    if not isinstance(mapping, dict) or set(mapping) not in allowed_sets:
        raise CollectorError(
            f"job {job_id} {arm} resolved metrics must contain the canonical set"
        )
    if any(not isinstance(value, str) or not value for value in mapping.values()):
        raise CollectorError(f"job {job_id} {arm} resolved metric name is invalid")
    if len(set(mapping.values())) != len(mapping):
        raise CollectorError(
            f"job {job_id} {arm} resolved metric source names must be one-to-one"
        )
    return mapping


def _validate_counter_series(
    metrics: dict[str, Any], metric_name: str, expected_steps: list[int], job_id: str
) -> list[int]:
    series = metrics.get(metric_name)
    if not isinstance(series, dict):
        raise CollectorError(f"job {job_id} lacks {metric_name} evidence")
    try:
        steps = sorted(int(step) for step in series)
    except ValueError as error:
        raise CollectorError(f"job {job_id} {metric_name} step is invalid") from error
    if steps != expected_steps:
        raise CollectorError(f"job {job_id} {metric_name} has missing steps")
    values = []
    for step in steps:
        value = series[str(step)]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CollectorError(f"job {job_id} {metric_name} value is invalid")
        numeric = float(value)
        if not numeric.is_integer() or numeric < 0:
            raise CollectorError(f"job {job_id} {metric_name} value is invalid")
        values.append(int(numeric))
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise CollectorError(f"job {job_id} {metric_name} is not monotonic")
    return values


def _load_arm(
    root: Path,
    job_dir: Path,
    raw_path: Path,
    manifest: dict[str, Any],
    job_id: str,
    run_id: str,
    expected_order_index: int,
) -> tuple[ArmObservation, dict[str, Any]]:
    raw = _read_json(raw_path, f"job {job_id} raw timing")
    arm = raw.get("arm")
    if arm not in {"on", "off"}:
        raise CollectorError(f"job {job_id} raw arm is invalid")
    if raw.get("run_id") != run_id or raw.get("order_index") != expected_order_index:
        raise CollectorError(f"job {job_id} raw timing order identity differs")
    warmup = _require_int(manifest.get("warmup_updates"), f"job {job_id} warmup")
    measured = _require_int(
        manifest.get("measured_updates"), f"job {job_id} measured updates"
    )
    total = _require_int(manifest.get("total_updates"), f"job {job_id} total updates")
    if warmup < 5:
        raise CollectorError(f"job {job_id} warmup_updates must be at least 5")
    if measured < 20:
        raise CollectorError(f"job {job_id} measured_updates must be at least 20")
    if total != warmup + measured:
        raise CollectorError(f"job {job_id} update window is invalid")
    if raw.get("warmup_updates") != warmup or raw.get("measured_updates") != measured:
        raise CollectorError(f"job {job_id} {arm} raw update window differs")
    expected_steps = list(range(warmup + 1, total + 1))
    rows = raw.get("measured_step_workload")
    if not isinstance(rows, list) or len(rows) != measured:
        raise CollectorError(f"job {job_id} {arm} measured workload is incomplete")
    if [row.get("step") for row in rows if isinstance(row, dict)] != expected_steps:
        raise CollectorError(f"job {job_id} {arm} has missing measured steps")
    if raw.get("training_gpu_count") != 4:
        raise CollectorError(f"job {job_id} policy training GPU count must equal 4")

    manifest_metrics = manifest.get("resolved_metric_names")
    if not isinstance(manifest_metrics, dict) or arm not in manifest_metrics:
        raise CollectorError(f"job {job_id} manifest metric evidence is incomplete")
    expected_mapping = _validate_metric_mapping(
        manifest_metrics[arm], job_id=job_id, arm=arm
    )
    raw_mapping = _validate_metric_mapping(
        raw.get("resolved_metric_names"), job_id=job_id, arm=arm
    )
    if raw_mapping != expected_mapping:
        raise CollectorError(f"job {job_id} {arm} resolved metrics differ")
    component_series = raw.get("measured_component_series")
    if not isinstance(component_series, dict) or set(component_series) != set(
        expected_mapping
    ):
        raise CollectorError(f"job {job_id} {arm} component series are incomplete")
    raw_fields = {**RAW_FIELD_BY_CANONICAL_METRIC, **OPTIONAL_ZERO_LOGPROB_METRICS}
    for canonical_name in expected_mapping:
        field = raw_fields[canonical_name]
        series = component_series[canonical_name]
        if not isinstance(series, list) or len(series) != measured:
            raise CollectorError(
                f"job {job_id} {arm} {canonical_name} has missing measured steps"
            )
        for row, point, expected_step in zip(rows, series, expected_steps, strict=True):
            if not isinstance(row, dict) or not isinstance(point, dict):
                raise CollectorError(f"job {job_id} {arm} measured row is invalid")
            if point.get("step") != expected_step:
                raise CollectorError(
                    f"job {job_id} {arm} {canonical_name} has missing measured steps"
                )
            validator = (
                _require_nonnegative_number
                if canonical_name in OPTIONAL_ZERO_LOGPROB_METRICS
                else _require_number
            )
            row_value = validator(row.get(field), f"job {job_id} {arm} {field}")
            point_value = validator(
                point.get("value"), f"job {job_id} {arm} {canonical_name}"
            )
            if canonical_name in OPTIONAL_ZERO_LOGPROB_METRICS and (
                row_value != 0.0 or point_value != 0.0
            ):
                raise CollectorError(
                    f"job {job_id} {arm} optional Logprob metrics must be exactly zero"
                )
            if row_value != point_value:
                raise CollectorError(
                    f"job {job_id} {arm} row differs from component series"
                )
    for row_index, row in enumerate(rows):
        total_tokens = _require_number(
            row.get("total_num_tokens"),
            f"job {job_id} {arm} total_num_tokens row {row_index}",
        )
        valid_tokens = _require_number(
            row.get("global_valid_toks"),
            f"job {job_id} {arm} global_valid_toks row {row_index}",
        )
        if valid_tokens > total_tokens:
            raise CollectorError(f"job {job_id} {arm} valid tokens exceed total tokens")
        for throughput_field, duration_field in THROUGHPUT_DURATION_FIELDS.items():
            duration = _require_number(
                row.get(duration_field), f"job {job_id} {arm} {duration_field}"
            )
            observed = _require_number(
                row.get(throughput_field), f"job {job_id} {arm} {throughput_field}"
            )
            expected = total_tokens / duration / 4.0
            if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise CollectorError(
                    f"job {job_id} {arm} {throughput_field} is not actual-token normalized"
                )

    components = {
        spec.name: {
            "duration": statistics.median(
                _require_number(row[spec.duration_field], spec.duration_field)
                for row in rows
            ),
            "throughput": statistics.median(
                _require_number(row[spec.throughput_field], spec.throughput_field)
                for row in rows
            ),
        }
        for spec in COMPONENT_SPECS
    }
    metrics_path = _safe_file(
        root, raw_path.parent / "metrics.json", f"job {job_id} {arm} metrics"
    )
    metrics = _read_json(metrics_path, f"job {job_id} {arm} metrics")
    for canonical_name, source_name in expected_mapping.items():
        source_series = metrics.get(source_name)
        if not isinstance(source_series, dict):
            raise CollectorError(f"job {job_id} {arm} metrics.json lacks {source_name}")
        for point in component_series[canonical_name]:
            step = str(point["step"])
            observed = source_series.get(step)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or float(observed) != float(point["value"])
            ):
                raise CollectorError(
                    f"job {job_id} {arm} metrics.json differs from raw measured series"
                )
    return (
        ArmObservation(
            arm=arm,
            rows=tuple(rows),
            metrics=metrics,
            components=components,
        ),
        {"raw": raw, "metrics": metrics},
    )


def _relative_delta(left: float, right: float) -> float:
    return abs(left - right) / ((left + right) / 2.0)


def _validate_workload_pair(
    baseline: ArmObservation,
    candidate: ArmObservation,
    label: str,
) -> dict[str, Any]:
    if len(baseline.rows) != len(candidate.rows):
        raise CollectorError(f"{label} workload row count differs")
    observed = True
    token_evidence = {}
    for field in WORKLOAD_EXACT_FIELDS:
        left = [row[field] for row in baseline.rows]
        right = [row[field] for row in candidate.rows]
        if left != right:
            observed = False
    for field in WORKLOAD_TOKEN_FIELDS:
        left = [
            _require_number(row[field], f"{label} baseline {field}")
            for row in baseline.rows
        ]
        right = [
            _require_number(row[field], f"{label} candidate {field}")
            for row in candidate.rows
        ]
        total_delta = _relative_delta(sum(left), sum(right))
        max_step_delta = max(
            _relative_delta(a, b) for a, b in zip(left, right, strict=True)
        )
        token_evidence[field] = {
            "total_relative_delta": _clean(total_delta),
            "max_paired_step_relative_delta": _clean(max_step_delta),
        }
        observed = observed and total_delta <= WORKLOAD_TOTAL_DELTA_LIMIT
        observed = observed and max_step_delta <= WORKLOAD_STEP_DELTA_LIMIT
    if not observed:
        raise CollectorError(f"{label} workload equivalence failed")
    return {"observed": True, "token_evidence": token_evidence}


def _validate_profile_evidence(
    root: Path,
    job_dir: Path,
    context: str,
    job_id: str,
    timing_order: tuple[str, ...],
) -> tuple[list[str], dict[str, Any]]:
    full_cg_enabled, a2a_enabled = CONTEXT_FLAGS[context]
    feature_path = _safe_file(
        root,
        job_dir / "feature_attribution.json",
        f"job {job_id} feature attribution",
    )
    feature = _read_json(feature_path, f"job {job_id} feature attribution")
    if (
        feature.get("feature_context") != context
        or feature.get("full_cg_enabled") is not full_cg_enabled
        or feature.get("a2a_enabled") is not a2a_enabled
        or feature.get("kernel_presence_passed") is not True
    ):
        raise CollectorError(f"job {job_id} feature attribution differs")
    counts = feature.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(timing_order):
        raise CollectorError(f"job {job_id} feature attribution arms differ")
    for arm in timing_order:
        record = counts[arm]
        if not isinstance(record, dict):
            raise CollectorError(f"job {job_id} feature counts are invalid")
        a2a_count = _require_int(
            record.get("nccl_a2a_kernel"), f"job {job_id} NCCL A2A kernel count"
        )
        if a2a_count == 0:
            raise CollectorError(f"job {job_id} lacks NCCL A2A kernel presence")

    evidence: dict[str, Any] = {"feature_attribution": feature}
    if not full_cg_enabled:
        kernel_path = _safe_file(
            root,
            job_dir / "kernel_attribution.json",
            f"job {job_id} CuTeDSL kernel attribution",
        )
        kernel = _read_json(kernel_path, f"job {job_id} CuTeDSL kernel attribution")
        arms = kernel.get("arms")
        if (
            kernel.get("passed") is not True
            or not isinstance(arms, dict)
            or set(arms) != {"on", "off"}
        ):
            raise CollectorError(f"job {job_id} CuTeDSL kernel attribution differs")
        fused_fields = (
            "fused_glu_match_count",
            "fused_dglu_match_count",
            "fused_quant_match_count",
            "fused_grouped_gemm_match_count",
        )
        for field in fused_fields:
            on_count = _require_int(arms["on"].get(field), f"job {job_id} ON {field}")
            off_count = _require_int(
                arms["off"].get(field), f"job {job_id} OFF {field}"
            )
            if on_count == 0 or off_count != 0:
                raise CollectorError(
                    f"job {job_id} CuTeDSL kernel attribution {field} differs"
                )
        if (
            _require_int(
                arms["off"].get("baseline_expert_gemm_match_count"),
                f"job {job_id} OFF baseline expert GEMM count",
            )
            == 0
        ):
            raise CollectorError(
                f"job {job_id} CuTeDSL kernel attribution lacks OFF baseline GEMM"
            )
        evidence["cutedsl_kernel_attribution"] = kernel
    return [], evidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_a2a_temporal_evidence(
    root: Path,
    job_dir: Path,
    context: str,
    job_id: str,
    *,
    required: bool,
    require_verified: bool,
) -> tuple[list[str], dict[str, Any]]:
    analyzer_path = job_dir / "a2a_temporal_overlap.json"
    if not analyzer_path.exists() and not analyzer_path.is_symlink():
        if required:
            return [f"{context} job {job_id} lacks A2A temporal-overlap analysis"], {}
        return [], {}
    analyzer_path = _safe_file(
        root, analyzer_path, f"job {job_id} A2A temporal-overlap analysis"
    )
    analyzer = _read_json(analyzer_path, f"job {job_id} A2A temporal-overlap analysis")
    expected_fields = {
        "schema_version",
        "source_profile_sha256",
        "a2a_interval_count",
        "expert_gemm_interval_count",
        "overlap_duration_ns",
        "a2a_overlap_ratio",
        "gemm_overlap_ratio",
        "temporal_overlap_verified",
        "limitations",
    }
    if set(analyzer) != expected_fields or analyzer.get("schema_version") != 1:
        raise CollectorError(f"job {job_id} A2A analyzer schema differs")
    source_digest = _require_sha(
        analyzer.get("source_profile_sha256"),
        64,
        f"job {job_id} A2A source profile SHA",
    )
    profile_paths = sorted(job_dir.glob("profiles/**/*.nsys-rep"))
    matching_profiles = []
    for profile_path in profile_paths:
        profile_path = _safe_file(
            root, profile_path, f"job {job_id} A2A source profile"
        )
        if _sha256_file(profile_path) == source_digest:
            matching_profiles.append(profile_path)
    if len(matching_profiles) != 1:
        raise CollectorError(
            f"job {job_id} A2A source profile digest must match exactly one artifact"
        )
    for field in ("a2a_interval_count", "expert_gemm_interval_count"):
        if _require_int(analyzer.get(field), f"job {job_id} {field}") == 0:
            raise CollectorError(f"job {job_id} {field} must be positive")
    overlap_duration = analyzer.get("overlap_duration_ns")
    if (
        isinstance(overlap_duration, bool)
        or not isinstance(overlap_duration, (int, float))
        or not math.isfinite(float(overlap_duration))
        or float(overlap_duration) < 0.0
        or (require_verified and float(overlap_duration) == 0.0)
    ):
        expected = "positive" if require_verified else "non-negative"
        raise CollectorError(f"job {job_id} overlap_duration_ns must be {expected}")
    for field in ("a2a_overlap_ratio", "gemm_overlap_ratio"):
        value = analyzer.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            or (require_verified and float(value) == 0.0)
        ):
            interval = "(0, 1]" if require_verified else "[0, 1]"
            raise CollectorError(f"job {job_id} {field} must be in {interval}")
    limitations = analyzer.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise CollectorError(f"job {job_id} A2A limitations must be strings")
    verified = analyzer.get("temporal_overlap_verified")
    if not isinstance(verified, bool):
        raise CollectorError(f"job {job_id} temporal_overlap_verified must be boolean")
    observed_overlap = float(overlap_duration) > 0.0 and all(
        float(analyzer[field]) > 0.0
        for field in ("a2a_overlap_ratio", "gemm_overlap_ratio")
    )
    if verified is not observed_overlap:
        raise CollectorError(
            f"job {job_id} temporal_overlap_verified is inconsistent with "
            "overlap evidence"
        )
    reasons = []
    if require_verified and not verified:
        reasons.append(f"{context} job {job_id} A2A temporal overlap is not verified")
    if any(REPRESENTATIVE_LIMITATION_FRAGMENT in item for item in limitations):
        reasons.append(
            f"{context} job {job_id} A2A analysis is representative-only; "
            "no all-rank aggregation"
        )
    return reasons, {"a2a_temporal_overlap": analyzer}


def _load_run(
    root: Path,
    record: dict[str, Any],
) -> FactorialRun:
    context = record["factorial_context"]
    replicate_index = record["replicate_index"]
    job_id = record["job_id"]
    job_dir, run_id = _find_run(root, job_id)
    manifest = _read_json(
        _safe_file(root, job_dir / "benchmark_manifest.json", f"job {job_id} manifest"),
        f"job {job_id} manifest",
    )
    expected_manifest = {
        "run_id": run_id,
        "feature_context": context,
        "replicate_index": replicate_index,
        "submission_group": record["submission_group"],
        "timing_order": record["timing_order"].split(","),
        "profile_enabled": bool(record["profile_enabled"]),
        "full_cg_enabled": record["full_cg_enabled"],
        "a2a_enabled": record["a2a_enabled"],
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise CollectorError(
                f"job {job_id} manifest {field} differs from submission"
            )
    _validate_identity(manifest, job_id)
    _validate_contract(manifest, context, job_id)

    summaries = sorted(job_dir.rglob("timing_summary.json"))
    if len(summaries) != 1:
        raise CollectorError(
            f"job {job_id} requires exactly one timing_summary.json; found {len(summaries)}"
        )
    summary = _read_json(
        _safe_file(root, summaries[0], f"job {job_id} timing summary"),
        f"job {job_id} timing summary",
    )
    timing_order = tuple(record["timing_order"].split(","))
    if (
        summary.get("run_id") != run_id
        or tuple(summary.get("timing_order", ())) != timing_order
        or summary.get("available_arms") != list(timing_order)
        or summary.get("workload_metric") != "train/total_num_tokens"
    ):
        raise CollectorError(f"job {job_id} timing summary identity differs")
    raw_files = summary.get("raw_timing_files")
    if not isinstance(raw_files, list) or len(raw_files) != len(timing_order):
        raise CollectorError(f"job {job_id} timing summary raw files differ")
    actual_raw_paths = sorted(job_dir.glob("timing/*/raw_timing.json"))
    if len(actual_raw_paths) != len(raw_files):
        raise CollectorError(f"job {job_id} has unexpected raw timing artifacts")
    arms = {}
    digest_arms = {}
    for order_index, (expected_arm, raw_file) in enumerate(
        zip(timing_order, raw_files, strict=True)
    ):
        raw_path = _safe_relative_file(
            root, job_dir, raw_file, f"job {job_id} raw timing"
        )
        arm, digest_evidence = _load_arm(
            root,
            job_dir,
            raw_path,
            manifest,
            job_id,
            run_id,
            order_index,
        )
        if arm.arm != expected_arm or arm.arm in arms:
            raise CollectorError(f"job {job_id} raw timing arms differ")
        arms[arm.arm] = arm
        digest_arms[arm.arm] = digest_evidence
    if set(arms) != set(timing_order):
        raise CollectorError(f"job {job_id} raw timing arms are incomplete")

    equivalence = summary.get("workload_equivalence")
    if not isinstance(equivalence, dict) or equivalence.get("observed") is not True:
        raise CollectorError(f"job {job_id} workload equivalence summary is invalid")
    if CONTEXT_FLAGS[context][0]:
        if equivalence.get("required") is not False:
            raise CollectorError(f"job {job_id} single-arm equivalence must be N/A")
    else:
        if equivalence.get("required") is not True:
            raise CollectorError(f"job {job_id} ON/OFF equivalence must be required")
        _validate_workload_pair(arms["on"], arms["off"], f"job {job_id} ON/OFF")

    provisional_reasons = []
    profile_evidence: dict[str, Any] = {}
    if bool(record["profile_enabled"]):
        provisional_reasons, profile_evidence = _validate_profile_evidence(
            root, job_dir, context, job_id, timing_order
        )
    temporal_reasons, temporal_evidence = _validate_a2a_temporal_evidence(
        root,
        job_dir,
        context,
        job_id,
        required=bool(record["profile_enabled"]),
        require_verified=CONTEXT_FLAGS[context][1],
    )
    provisional_reasons.extend(temporal_reasons)
    profile_evidence.update(temporal_evidence)
    if CONTEXT_FLAGS[context][0]:
        metrics = arms["on"].metrics
        all_steps = list(
            range(
                1,
                _require_int(manifest.get("total_updates"), f"job {job_id} total") + 1,
            )
        )
        counters = {
            name: _validate_counter_series(metrics, metric, all_steps, job_id)
            for name, metric in {
                "warmup": "train/full_cuda_graph_warmup_calls",
                "capture": "train/full_cuda_graph_capture_calls",
                "replay": "train/full_cuda_graph_replay_calls",
                "reset": "train/full_cuda_graph_reset_calls",
            }.items()
        }
        if counters["warmup"][-1] != 3:
            raise CollectorError(f"job {job_id} warmup_calls must equal 3")
        capture = counters["capture"]
        replay = counters["replay"]
        if capture[-1] != 1:
            raise CollectorError(f"job {job_id} capture_calls must equal 1")
        if replay[-1] < 2:
            raise CollectorError(f"job {job_id} replay_calls must be at least 2")
        if counters["reset"][-1] != 0:
            raise CollectorError(f"job {job_id} reset_calls must equal 0")

    fixed_on = manifest["fixed_config_evidence"]["on"]
    invariant_config = {
        key: value for key, value in fixed_on.items() if key not in A2A_CONFIG_FIELDS
    }
    digest_payload = {
        "status": _read_json(job_dir / "status.json", f"job {job_id} status"),
        "manifest": manifest,
        "timing_summary": summary,
        "arms": digest_arms,
        "profile": profile_evidence,
    }
    evidence_digest = hashlib.sha256(
        _canonical_json(digest_payload).encode()
    ).hexdigest()
    temporal_analyzer = temporal_evidence.get("a2a_temporal_overlap")
    a2a_overlap_ratio = None
    temporal_representative_only = False
    if isinstance(temporal_analyzer, dict):
        a2a_overlap_ratio = float(temporal_analyzer["a2a_overlap_ratio"])
        temporal_representative_only = any(
            REPRESENTATIVE_LIMITATION_FRAGMENT in item
            for item in temporal_analyzer["limitations"]
        )
    return FactorialRun(
        context=context,
        replicate_index=replicate_index,
        job_id=job_id,
        run_id=run_id,
        timing_order=timing_order,
        profile_enabled=bool(record["profile_enabled"]),
        source_identity=_canonical_json(
            {
                key: manifest.get(key)
                for key in ("source_sha", "upstream_ref", "upstream_sha")
            }
        ),
        image_identity=_canonical_json(
            {key: manifest.get(key) for key in ("image", "image_sha256")}
        ),
        workload_identity=_canonical_json(
            {
                key: manifest.get(key)
                for key in (
                    "recipe",
                    "base_config_sha256",
                    "artifact_revisions",
                    "warmup_updates",
                    "measured_updates",
                    "total_updates",
                    "topology",
                    "workload",
                )
            }
        ),
        invariant_config_identity=_canonical_json(invariant_config),
        metric_identity=_canonical_json(manifest["resolved_metric_names"]["on"]),
        arms=arms,
        evidence_digest=evidence_digest,
        provisional_reasons=tuple(provisional_reasons),
        a2a_overlap_ratio=a2a_overlap_ratio,
        temporal_representative_only=temporal_representative_only,
    )


def _validate_runs(runs: list[FactorialRun]) -> list[int]:
    by_context = {
        context: sorted(
            (run for run in runs if run.context == context),
            key=lambda run: run.replicate_index,
        )
        for context in CONTEXTS
    }
    profiles = {
        context: [run for run in context_runs if run.profile_enabled]
        for context, context_runs in by_context.items()
    }
    for context, context_profiles in profiles.items():
        if CONTEXT_FLAGS[context][1] and not context_profiles:
            raise CollectorError(
                f"context {context} requires at least one profile replicate"
            )
        if not CONTEXT_FLAGS[context][1] and len(context_profiles) != 1:
            raise CollectorError(
                f"context {context} requires exactly one profile replicate"
            )
    identity_fields = (
        "source_identity",
        "image_identity",
        "workload_identity",
        "invariant_config_identity",
        "metric_identity",
    )
    for field in identity_fields:
        values = {getattr(run, field) for run in runs}
        if len(values) != 1:
            raise CollectorError(f"{field} differs across factorial cohort")
    indices = sorted({run.replicate_index for run in runs})
    for replicate_index in indices:
        block = {
            run.context: run for run in runs if run.replicate_index == replicate_index
        }
        if set(block) != set(CONTEXTS):
            raise CollectorError(f"replica {replicate_index} lacks a factorial context")
        baseline = block["g0a0"].arms["on"]
        for context in CONTEXTS[1:]:
            _validate_workload_pair(
                baseline,
                block[context].arms["on"],
                f"replica {replicate_index} g0a0/{context}",
            )
        _validate_workload_pair(
            block["g0a0"].arms["off"],
            block["g0a1"].arms["off"],
            f"replica {replicate_index} g0a0/g0a1 OFF",
        )
    return indices


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _summarize_factors(
    values: list[tuple[int, float]],
    *,
    samples: int,
    seed: int,
    stream: str,
    claim_status: str,
) -> dict[str, Any]:
    factors = [factor for _, factor in values]
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    bootstrap = sorted(
        statistics.median(generator.choices(factors, k=len(factors)))
        for _ in range(samples)
    )
    to_percent = lambda factor: _clean((factor - 1.0) * 100.0)
    return {
        "claim_status": claim_status,
        "replicate_count": len(values),
        "replicates": [
            {
                "replicate_index": index,
                "factor": _clean(factor),
                "percent": to_percent(factor),
            }
            for index, factor in values
        ],
        "median_factor": _clean(statistics.median(factors)),
        "median_percent": to_percent(statistics.median(factors)),
        "min_percent": to_percent(min(factors)),
        "max_percent": to_percent(max(factors)),
        "bootstrap_ci95_percent": {
            "lower": to_percent(_percentile(bootstrap, 0.025)),
            "upper": to_percent(_percentile(bootstrap, 0.975)),
        },
    }


def _benefit_factor(
    baseline: float,
    feature: float,
    measurement: str,
) -> float:
    return baseline / feature if measurement == "duration" else feature / baseline


def _factorial_effect_values(
    cells: dict[str, float], measurement: str
) -> dict[str, float]:
    full_cg_at_a0 = _benefit_factor(cells["g0a0"], cells["g1a0"], measurement)
    full_cg_at_a1 = _benefit_factor(cells["g0a1"], cells["g1a1"], measurement)
    a2a_at_g0 = _benefit_factor(cells["g0a0"], cells["g0a1"], measurement)
    a2a_at_g1 = _benefit_factor(cells["g1a0"], cells["g1a1"], measurement)
    return {
        "full_cg_at_a0": full_cg_at_a0,
        "full_cg_at_a1": full_cg_at_a1,
        "a2a_at_g0": a2a_at_g0,
        "a2a_at_g1": a2a_at_g1,
        "full_cg_main": math.sqrt(full_cg_at_a0 * full_cg_at_a1),
        "a2a_main": math.sqrt(a2a_at_g0 * a2a_at_g1),
        "interaction": a2a_at_g1 / a2a_at_g0,
    }


def _factorial_effects(
    runs: list[FactorialRun],
    indices: list[int],
    *,
    samples: int,
    seed: int,
    claim_status: str,
) -> dict[str, Any]:
    by_key = {(run.context, run.replicate_index): run for run in runs}
    output = {}
    for spec in COMPONENT_SPECS:
        output[spec.name] = {}
        for measurement in ("duration", "throughput"):
            values_by_effect: dict[str, list[tuple[int, float]]] = {}
            for replicate_index in indices:
                cells = {
                    context: by_key[(context, replicate_index)]
                    .arms["on"]
                    .components[spec.name][measurement]
                    for context in CONTEXTS
                }
                effects = _factorial_effect_values(cells, measurement)
                for name, factor in effects.items():
                    values_by_effect.setdefault(name, []).append(
                        (replicate_index, factor)
                    )
            output[spec.name][measurement] = {
                name: _summarize_factors(
                    values,
                    samples=samples,
                    seed=seed,
                    stream=f"factorial:{spec.name}:{measurement}:{name}",
                    claim_status=claim_status,
                )
                for name, values in values_by_effect.items()
            }
    output["logprob"] = {
        "status": "not_applicable",
        "reason": "disabled_by_supported_full_cg_slice",
    }
    return output


def _cutedsl_effects(
    runs: list[FactorialRun],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    for context in ("g0a0", "g0a1"):
        context_runs = sorted(
            (run for run in runs if run.context == context),
            key=lambda run: run.replicate_index,
        )
        output[context] = {}
        for spec in COMPONENT_SPECS:
            output[context][spec.name] = {}
            for measurement in ("duration", "throughput"):
                values = []
                for run in context_runs:
                    baseline = run.arms["off"].components[spec.name][measurement]
                    feature = run.arms["on"].components[spec.name][measurement]
                    values.append(
                        (
                            run.replicate_index,
                            _benefit_factor(baseline, feature, measurement),
                        )
                    )
                output[context][spec.name][measurement] = _summarize_factors(
                    values,
                    samples=samples,
                    seed=seed,
                    stream=f"cutedsl:{context}:{spec.name}:{measurement}",
                    claim_status="claim_ready",
                )
        output[context]["logprob"] = {
            "status": "not_applicable",
            "reason": "requires_separate_eager_paired_cohort",
        }
    return output


def _direct_path_effects(
    runs: list[FactorialRun],
    indices: list[int],
    *,
    samples: int,
    seed: int,
    claim_status: str,
) -> dict[str, Any]:
    by_key = {(run.context, run.replicate_index): run for run in runs}
    output = {}
    for spec in COMPONENT_SPECS:
        output[spec.name] = {}
        for measurement in ("duration", "throughput"):
            values_by_effect = {name: [] for name in DIRECT_PATH_TARGETS}
            for replicate_index in indices:
                baseline = (
                    by_key[("g0a0", replicate_index)]
                    .arms["off"]
                    .components[spec.name][measurement]
                )
                for name, (context, arm) in DIRECT_PATH_TARGETS.items():
                    feature = (
                        by_key[(context, replicate_index)]
                        .arms[arm]
                        .components[spec.name][measurement]
                    )
                    values_by_effect[name].append(
                        (
                            replicate_index,
                            _benefit_factor(baseline, feature, measurement),
                        )
                    )
            output[spec.name][measurement] = {
                name: _summarize_factors(
                    values,
                    samples=samples,
                    seed=seed,
                    stream=f"direct:{spec.name}:{measurement}:{name}",
                    claim_status=claim_status,
                )
                for name, values in values_by_effect.items()
            }
    output["full_cg_without_cutedsl"] = {
        "status": "unsupported_dependency",
        "reason": (
            "full-iteration CUDA Graph requires device-initiated CuTeDSL kernels"
        ),
    }
    output["logprob"] = {
        "status": "not_applicable",
        "reason": "disabled_by_supported_full_cg_slice",
    }
    return output


def _a2a_overlap_ratio_contrasts(
    runs: list[FactorialRun],
) -> tuple[dict[str, Any], list[str]]:
    output = {}
    reasons = []
    for level, full_cg_enabled in (("g0", False), ("g1", True)):
        baseline_context = f"{level}a0"
        overlap_context = f"{level}a1"
        baseline = {
            run.replicate_index: run
            for run in runs
            if run.context == baseline_context and run.profile_enabled
        }
        overlap = {
            run.replicate_index: run
            for run in runs
            if run.context == overlap_context and run.profile_enabled
        }
        paired_indices = sorted(set(baseline) & set(overlap))
        if not paired_indices:
            raise CollectorError(
                f"{level} requires a paired profile replicate for A2A overlap contrast"
            )
        pairs = []
        pair_increases = []
        available_pairs = []
        representative_only = False
        for replicate_index in paired_indices:
            baseline_run = baseline[replicate_index]
            overlap_run = overlap[replicate_index]
            baseline_ratio = baseline_run.a2a_overlap_ratio
            overlap_ratio = overlap_run.a2a_overlap_ratio
            if baseline_ratio is None or overlap_ratio is None:
                pairs.append(
                    {
                        "replicate_index": replicate_index,
                        "status": "unavailable",
                    }
                )
                continue
            increase = overlap_ratio - baseline_ratio
            increased = increase > 0.0
            if not increased:
                reasons.append(
                    f"{level} profile replica {replicate_index} "
                    "A2A overlap ratio did not increase"
                )
            pair_increases.append(increased)
            representative_only = representative_only or (
                baseline_run.temporal_representative_only
                or overlap_run.temporal_representative_only
            )
            pair = {
                "replicate_index": replicate_index,
                "baseline_ratio": _clean(baseline_ratio),
                "overlap_ratio": _clean(overlap_ratio),
                "absolute_increase": _clean(increase),
                "relative_factor": (
                    _clean(overlap_ratio / baseline_ratio)
                    if baseline_ratio > 0.0
                    else None
                ),
                "increased": increased,
            }
            pairs.append(pair)
            available_pairs.append(pair)
        all_pairs_increased = len(available_pairs) == len(paired_indices) and all(
            pair_increases
        )
        output[level] = {
            "full_cg_enabled": full_cg_enabled,
            "baseline_context": baseline_context,
            "overlap_context": overlap_context,
            "evidence_scope": "representative_process_exploratory",
            "paired_profile_replicates": pairs,
            "all_pairs_increased": all_pairs_increased,
            "median_baseline_ratio": (
                _clean(
                    statistics.median(
                        pair["baseline_ratio"] for pair in available_pairs
                    )
                )
                if available_pairs
                else None
            ),
            "median_overlap_ratio": (
                _clean(
                    statistics.median(pair["overlap_ratio"] for pair in available_pairs)
                )
                if available_pairs
                else None
            ),
            "median_absolute_increase": (
                _clean(
                    statistics.median(
                        pair["absolute_increase"] for pair in available_pairs
                    )
                )
                if available_pairs
                else None
            ),
            "claim_status": (
                "provisional"
                if representative_only or not all_pairs_increased
                else "claim_ready"
            ),
        }
    return output, reasons


def collect(
    submission_path: Path,
    result_root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Validate and summarize one dependency-constrained factorial cohort."""
    records = _load_submission(submission_path)
    by_context = _validate_submission_matrix(records)
    runs = [
        _load_run(result_root, by_context[context][replicate_index])
        for replicate_index in sorted(by_context[CONTEXTS[0]])
        for context in CONTEXTS
    ]
    indices = _validate_runs(runs)
    overlap_contrasts, contrast_reasons = _a2a_overlap_ratio_contrasts(runs)
    direct_path_limitation = (
        "a2a_only_without_cutedsl lacks OFF-arm temporal overlap proof; "
        "direct timing effect is associative only"
    )
    provisional_reasons = sorted(
        {
            direct_path_limitation,
            *contrast_reasons,
            *(reason for run in runs for reason in run.provisional_reasons),
        }
    )
    claim_status = "provisional" if provisional_reasons else "claim_ready"
    digests = [
        {
            "factorial_context": run.context,
            "replicate_index": run.replicate_index,
            "job_id": run.job_id,
            "run_id": run.run_id,
            "evidence_digest": run.evidence_digest,
        }
        for run in sorted(runs, key=lambda item: (item.replicate_index, item.context))
    ]
    cohort_digest = hashlib.sha256(_canonical_json(digests).encode()).hexdigest()
    return {
        "schema_version": 1,
        "claim_status": claim_status,
        "claim_ready": claim_status == "claim_ready",
        "provisional_reasons": provisional_reasons,
        "submission_jsonl": str(submission_path),
        "benchmark_result_root": str(result_root),
        "paired_replicate_indices": indices,
        "context_replicate_counts": {
            context: len(by_context[context]) for context in CONTEXTS
        },
        "contract": {
            "policy_training_gpu_count": 4,
            "train_global_batch_size": 8,
            "train_micro_batch_size": 1,
            "expert_model_parallel_size": 4,
            "cross_context_arm": "on",
            "cutedsl_off_scope": ["g0a0", "g0a1"],
        },
        "effect_definitions": {
            "duration": "baseline / feature - 1",
            "throughput": "feature / baseline - 1",
            "main_effect": (
                "geometric mean of the two dependency-valid conditional factors"
            ),
            "interaction": (
                "A2A optimization bundle factor with full-CG / "
                "factor without full-CG - 1"
            ),
        },
        "direct_path_effect_definitions": {
            "baseline": "g0a0 OFF (CuTeDSL OFF, full-CG OFF, A2A OFF)",
            "cutedsl_only": "g0a0 OFF -> g0a0 ON",
            "a2a_only_without_cutedsl": (
                "g0a0 OFF -> g0a1 OFF (A2A optimization bundle)"
            ),
            "cutedsl_a2a": ("g0a0 OFF -> g0a1 ON (A2A optimization bundle)"),
            "cutedsl_full_cg": "g0a0 OFF -> g1a0 ON",
            "all_three_combined": "g0a0 OFF -> g1a1 ON",
        },
        "dependency_notes": {
            "a2a_optimization_bundle": (
                "expert-parallel overlap + high-priority A2A stream + "
                "delayed wgrad compute"
            ),
            "common_baseline": "all three features disabled in g0a0 OFF",
            "a2a_only_without_cutedsl": (
                "timing effect is computed, but OFF-arm temporal overlap is not "
                "profiled; mechanistic attribution is provisional"
            ),
            "full_cg_without_cutedsl": (
                "unsupported because full-iteration CUDA Graph requires "
                "device-initiated CuTeDSL kernels"
            ),
            "incremental_effects": (
                "factorial_effects and cutedsl_g0_effects remain available"
            ),
            "profile_overlap_contrasts": (
                "representative-process exploratory evidence; not all-rank causal proof"
            ),
        },
        "a2a_overlap_ratio_contrasts": overlap_contrasts,
        "evidence_digest_method": (
            "SHA256 of canonical validated status, manifest, timing, metrics, and "
            "profile evidence"
        ),
        "job_evidence_digests": digests,
        "cohort_evidence_digest": cohort_digest,
        "bootstrap": {
            "method": "replicate-index paired resampling of median benefit factors",
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "factorial_effects": _factorial_effects(
            runs,
            indices,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            claim_status=claim_status,
        ),
        "direct_path_effects": _direct_path_effects(
            runs,
            indices,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            claim_status=claim_status,
        ),
        "cutedsl_g0_effects": _cutedsl_effects(
            runs, samples=bootstrap_samples, seed=bootstrap_seed
        ),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_jsonl", type=Path)
    parser.add_argument("benchmark_result_root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2606)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    if args.output_json is None:
        args.output_json = args.submission_jsonl.with_suffix(".factorial.json")
    if args.output_json.resolve() == args.submission_jsonl.resolve():
        parser.error("output JSON must not overwrite submission JSONL")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the dependency-constrained collector CLI."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        aggregate = collect(
            args.submission_jsonl,
            args.benchmark_result_root,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        _atomic_write(
            args.output_json,
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
    except (CollectorError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[INFO] Wrote factorial aggregate JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
