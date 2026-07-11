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
APPROVED_GRAPH_ON_RUNTIME_COMMIT = "539cfb96f3944ea6e32616ec43e10f4d1cf20491"
MIN_GRAPH_CALL_RATIO = 0.99
MIN_ROOFLINE_PREDICTED_SPEEDUP = 1.05
MIN_SPECDEC_HEADROOM_TOKENS = 32
CUDAGRAPH_ROLES = (
    "target",
    "draft",
    "draft_prefill",
    "draft_decode",
    "draft_query",
)
CUDAGRAPH_FALLBACK_REASONS = (
    "uninitialized",
    "disabled",
    "missing_capture_limit",
    "oversize",
    "mode_restricted",
    "missing_key",
    "incompatible",
    "empty",
)
CUDAGRAPH_FALLBACK_KEYS = tuple(
    f"train/vllm/cudagraph_{role}_fallback_{reason}"
    for role in CUDAGRAPH_ROLES
    for reason in CUDAGRAPH_FALLBACK_REASONS
)

BASE_METRIC_KEYS = {
    "e2e_time": "timing/train/total_step_time",
    "generation_time": "timing/train/generation",
    "e2e_tps_gpu": "performance/tokens_per_sec_per_gpu",
    "generation_tps_gpu": "performance/generation_tokens_per_sec_per_gpu",
    "policy_time": "timing/train/policy_training",
    "logprob_time": "timing/train/policy_and_reference_logprobs",
    "reward": "train/reward",
    "response_length": "train/mean_gen_tokens_per_sample",
    "approx_kl": "train/gen_kl_error",
    "policy_loss": "train/loss",
    "target_graph_ratio": "train/vllm/cudagraph_target_graph_call_ratio",
}
SPECDEC_METRIC_KEYS = {
    "num_drafts": "train/vllm/spec_num_drafts",
    "num_draft_tokens": "train/vllm/spec_num_draft_tokens",
    "num_accepted_tokens": "train/vllm/spec_num_accepted_tokens",
    "acceptance_rate": "train/vllm/spec_acceptance_rate",
    "mean_accept_len": "train/vllm/spec_acceptance_length",
}
V1_DRAFT_GRAPH_METRIC_KEYS = {
    "draft_graph_ratio": "train/vllm/cudagraph_draft_graph_call_ratio",
}
V2_DRAFT_GRAPH_METRIC_KEYS = {
    "draft_prefill_graph_ratio": (
        "train/vllm/cudagraph_draft_prefill_graph_call_ratio"
    ),
    "draft_decode_graph_ratio": "train/vllm/cudagraph_draft_decode_graph_call_ratio",
}
TAIL_GATE_METRIC_KEYS = {
    "gate_decisions": "train/vllm/tail_gate_decisions",
    "gate_activations": "train/vllm/tail_gate_activations",
    "gate_enabled_ratio": "train/vllm/tail_gate_enabled_step_ratio",
    "gate_advance_only_ratio": "train/vllm/tail_gate_advance_only_step_ratio",
    "activation_tick": "train/vllm/tail_gate_activation_tick",
    "activation_batch": "train/vllm/tail_gate_activation_batch",
    "activation_seq_len": "train/vllm/tail_gate_activation_seq_len",
    "predicted_speedup": "train/vllm/tail_gate_predicted_speedup",
}
ROOFLINE_ACTIVATION_METRIC_KEYS = {
    "activation_predicted_speedup": (
        "train/vllm/tail_gate_activation_predicted_speedup"
    ),
}
METRIC_KEYS = {
    **BASE_METRIC_KEYS,
    **SPECDEC_METRIC_KEYS,
    **V1_DRAFT_GRAPH_METRIC_KEYS,
    **V2_DRAFT_GRAPH_METRIC_KEYS,
    **TAIL_GATE_METRIC_KEYS,
    **ROOFLINE_ACTIVATION_METRIC_KEYS,
}
HEALTH_METRICS = ("reward", "response_length", "approx_kl", "policy_loss")
COHORT_FIELDS = (
    "model",
    "cluster",
    "runtime",
    "runtime_version",
    "runtime_commit",
    "ablation_behavior_revision",
    "vllm_version",
    "vllm_commit",
    "target_tp",
    "draft_tp",
    "dp",
    "ep",
    "temperature",
    "top_p",
    "max_osl",
    "max_model_len",
    "max_sequence_length",
    "num_prompts",
    "num_generations",
    "train_gbs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "recipe",
    "container",
    "container_sha256",
    "runner",
    "graph_mode",
    "cuda_graph_enabled",
    "enforce_eager",
    "sampling",
    "draft_sample_method",
)
CAPTURE_PROFILE_COHORT_FIELDS = ("cudagraph_max_requests",)
GRAPH_ABLATION_COHORT_FIELDS = (
    "runtime_commit",
    "graph_mode",
    "cuda_graph_enabled",
    "enforce_eager",
    *CAPTURE_PROFILE_COHORT_FIELDS,
)
SAME_VARIANT_GRAPH_MODE_FIELDS = (
    *(field for field in COHORT_FIELDS if field not in GRAPH_ABLATION_COHORT_FIELDS),
    "target_checkpoint",
    "target_checkpoint_revision",
    "draft_checkpoint",
    "gate_mode",
    "k",
    "threshold",
    "consecutive_checks",
    "roofline_config_sha256",
)
LEGACY_COHORT_FIELDS = tuple(
    field for field in COHORT_FIELDS if field != "draft_sample_method"
)
LOG_PROVENANCE_FIELDS = (
    "run_dir",
    "slurm_log_path",
    "ray_driver_log_path",
    "ray_log_dir",
)
LEGACY_UNRECORDED_MAX_MODEL_LEN = "legacy_unrecorded"
LEGACY_UNRECORDED_CAPTURE_PROFILE = "legacy_unrecorded"
LEGACY_UNRECORDED_CUDA_GRAPH_STATE = "legacy_unrecorded"
LEGACY_UNRECORDED_ABLATION_BEHAVIOR = "legacy_unrecorded"
LEGACY_CUDA_GRAPH_STATE_DEFAULTS = {
    "cuda_graph_enabled": LEGACY_UNRECORDED_CUDA_GRAPH_STATE,
    "enforce_eager": LEGACY_UNRECORDED_CUDA_GRAPH_STATE,
}
LEGACY_C78A93C8_MANIFEST_HEADER = (
    "timestamp",
    "model",
    "variant",
    "gate_mode",
    "k",
    "threshold",
    "consecutive_checks",
    "roofline_config_sha256",
    "cluster",
    "runtime",
    "runtime_version",
    "runtime_commit",
    "vllm_version",
    "vllm_commit",
    "target_tp",
    "draft_tp",
    "dp",
    "ep",
    "temperature",
    "top_p",
    "max_osl",
    "max_sequence_length",
    "num_prompts",
    "num_generations",
    "train_gbs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "recipe",
    "container",
    "container_sha256",
    "runner",
    "graph_mode",
    "sampling",
    "job_id",
    "wandb_run_id",
    "wandb_url",
    "command",
)
MINI_PROVENANCE_MANIFEST_HEADER = (
    "timestamp",
    "model",
    "variant",
    "gate_mode",
    "k",
    "threshold",
    "consecutive_checks",
    "roofline_config_sha256",
    "cluster",
    "runtime",
    "runtime_version",
    "runtime_commit",
    "vllm_version",
    "vllm_commit",
    "target_tp",
    "draft_tp",
    "dp",
    "ep",
    "temperature",
    "top_p",
    "max_osl",
    "max_model_len",
    "max_sequence_length",
    "num_prompts",
    "num_generations",
    "train_gbs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "recipe",
    "container",
    "container_sha256",
    "runner",
    "graph_mode",
    "sampling",
    "draft_sample_method",
    "job_id",
    "wandb_run_id",
    "wandb_url",
    "run_dir",
    "slurm_log_path",
    "ray_driver_log_path",
    "ray_log_dir",
    "launcher_command",
    "command",
)
STRUCTURED_PROVENANCE_MANIFEST_HEADER = (
    *MINI_PROVENANCE_MANIFEST_HEADER,
    "checkout_path",
    "ray_sub_path",
    "draft_checkpoint",
    "command_argv_json",
    "launcher_argv_json",
)
_MINI_GRAPH_MODE_INDEX = MINI_PROVENANCE_MANIFEST_HEADER.index("graph_mode")
CUDA_GRAPH_ABLATION_MINI_PROVENANCE_MANIFEST_HEADER = (
    *MINI_PROVENANCE_MANIFEST_HEADER[: _MINI_GRAPH_MODE_INDEX + 1],
    "cuda_graph_enabled",
    "enforce_eager",
    *MINI_PROVENANCE_MANIFEST_HEADER[_MINI_GRAPH_MODE_INDEX + 1 :],
)
CUDA_GRAPH_ABLATION_MINI_PROVENANCE_V4_MANIFEST_HEADER = (
    CUDA_GRAPH_ABLATION_MINI_PROVENANCE_MANIFEST_HEADER
)
_MINI_RUNTIME_COMMIT_INDEX = CUDA_GRAPH_ABLATION_MINI_PROVENANCE_V4_MANIFEST_HEADER.index(
    "runtime_commit"
)
CUDA_GRAPH_ABLATION_MINI_PROVENANCE_MANIFEST_HEADER = (
    *CUDA_GRAPH_ABLATION_MINI_PROVENANCE_V4_MANIFEST_HEADER[
        : _MINI_RUNTIME_COMMIT_INDEX + 1
    ],
    "ablation_behavior_revision",
    *CUDA_GRAPH_ABLATION_MINI_PROVENANCE_V4_MANIFEST_HEADER[
        _MINI_RUNTIME_COMMIT_INDEX + 1 :
    ],
)
IMMUTABLE_CHECKPOINT_MANIFEST_HEADER = (
    *STRUCTURED_PROVENANCE_MANIFEST_HEADER[:-3],
    "target_checkpoint",
    "target_checkpoint_revision",
    *STRUCTURED_PROVENANCE_MANIFEST_HEADER[-3:],
)
_MAX_NUM_SEQS_INDEX = IMMUTABLE_CHECKPOINT_MANIFEST_HEADER.index("max_num_seqs")
CAPTURE_PROFILE_MANIFEST_HEADER = (
    *IMMUTABLE_CHECKPOINT_MANIFEST_HEADER[: _MAX_NUM_SEQS_INDEX + 1],
    "cudagraph_max_requests",
    "cudagraph_max_tokens",
    "cudagraph_capture_sizes",
    *IMMUTABLE_CHECKPOINT_MANIFEST_HEADER[_MAX_NUM_SEQS_INDEX + 1 :],
)
_GRAPH_MODE_INDEX = CAPTURE_PROFILE_MANIFEST_HEADER.index("graph_mode")
CUDA_GRAPH_ABLATION_V4_MANIFEST_HEADER = (
    *CAPTURE_PROFILE_MANIFEST_HEADER[: _GRAPH_MODE_INDEX + 1],
    "cuda_graph_enabled",
    "enforce_eager",
    *CAPTURE_PROFILE_MANIFEST_HEADER[_GRAPH_MODE_INDEX + 1 :],
)
_RUNTIME_COMMIT_INDEX = CUDA_GRAPH_ABLATION_V4_MANIFEST_HEADER.index(
    "runtime_commit"
)
CUDA_GRAPH_ABLATION_MANIFEST_HEADER = (
    *CUDA_GRAPH_ABLATION_V4_MANIFEST_HEADER[: _RUNTIME_COMMIT_INDEX + 1],
    "ablation_behavior_revision",
    *CUDA_GRAPH_ABLATION_V4_MANIFEST_HEADER[_RUNTIME_COMMIT_INDEX + 1 :],
)
REQUIRED_MANIFEST_FIELDS = (
    *LEGACY_COHORT_FIELDS,
    "variant",
    "gate_mode",
    "k",
    "job_id",
    "wandb_run_id",
)
MINI_REQUIRED_MANIFEST_FIELDS = (
    *COHORT_FIELDS,
    *LOG_PROVENANCE_FIELDS,
    "variant",
    "gate_mode",
    "k",
    "job_id",
    "wandb_run_id",
    "launcher_command",
    "command",
    "checkout_path",
    "ray_sub_path",
    "draft_checkpoint",
    "target_checkpoint",
    "target_checkpoint_revision",
    "command_argv_json",
    "launcher_argv_json",
)


@dataclass(frozen=True)
class ManifestSchema:
    name: str
    header: tuple[str, ...]
    legacy_defaults: Mapping[str, str]


MANIFEST_SCHEMAS = {
    LEGACY_C78A93C8_MANIFEST_HEADER: ManifestSchema(
        "c78a93c8",
        LEGACY_C78A93C8_MANIFEST_HEADER,
        {
            "max_model_len": LEGACY_UNRECORDED_MAX_MODEL_LEN,
            "draft_sample_method": "",
            "cudagraph_max_requests": LEGACY_UNRECORDED_CAPTURE_PROFILE,
            "ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR,
            **LEGACY_CUDA_GRAPH_STATE_DEFAULTS,
        },
    ),
    MINI_PROVENANCE_MANIFEST_HEADER: ManifestSchema(
        "mini_provenance_v1",
        MINI_PROVENANCE_MANIFEST_HEADER,
        {
            "cudagraph_max_requests": LEGACY_UNRECORDED_CAPTURE_PROFILE,
            "ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR,
            **LEGACY_CUDA_GRAPH_STATE_DEFAULTS,
        },
    ),
    CUDA_GRAPH_ABLATION_MINI_PROVENANCE_V4_MANIFEST_HEADER: ManifestSchema(
        "cuda_graph_ablation_mini_provenance_v4",
        CUDA_GRAPH_ABLATION_MINI_PROVENANCE_V4_MANIFEST_HEADER,
        {
            "cudagraph_max_requests": LEGACY_UNRECORDED_CAPTURE_PROFILE,
            "ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR,
        },
    ),
    CUDA_GRAPH_ABLATION_MINI_PROVENANCE_MANIFEST_HEADER: ManifestSchema(
        "cuda_graph_ablation_mini_provenance_v5",
        CUDA_GRAPH_ABLATION_MINI_PROVENANCE_MANIFEST_HEADER,
        {"cudagraph_max_requests": LEGACY_UNRECORDED_CAPTURE_PROFILE},
    ),
    STRUCTURED_PROVENANCE_MANIFEST_HEADER: ManifestSchema(
        "structured_provenance_v1",
        STRUCTURED_PROVENANCE_MANIFEST_HEADER,
        {
            "cudagraph_max_requests": LEGACY_UNRECORDED_CAPTURE_PROFILE,
            "ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR,
            **LEGACY_CUDA_GRAPH_STATE_DEFAULTS,
        },
    ),
    IMMUTABLE_CHECKPOINT_MANIFEST_HEADER: ManifestSchema(
        "immutable_checkpoint_v2",
        IMMUTABLE_CHECKPOINT_MANIFEST_HEADER,
        {
            "cudagraph_max_requests": LEGACY_UNRECORDED_CAPTURE_PROFILE,
            "ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR,
            **LEGACY_CUDA_GRAPH_STATE_DEFAULTS,
        },
    ),
    CAPTURE_PROFILE_MANIFEST_HEADER: ManifestSchema(
        "capture_profile_v3",
        CAPTURE_PROFILE_MANIFEST_HEADER,
        {
            **LEGACY_CUDA_GRAPH_STATE_DEFAULTS,
            "ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR,
        },
    ),
    CUDA_GRAPH_ABLATION_V4_MANIFEST_HEADER: ManifestSchema(
        "cuda_graph_ablation_v4",
        CUDA_GRAPH_ABLATION_V4_MANIFEST_HEADER,
        {"ablation_behavior_revision": LEGACY_UNRECORDED_ABLATION_BEHAVIOR},
    ),
    CUDA_GRAPH_ABLATION_MANIFEST_HEADER: ManifestSchema(
        "cuda_graph_ablation_v5", CUDA_GRAPH_ABLATION_MANIFEST_HEADER, {}
    ),
}
LEGACY_GRAPH_STATE_SCHEMA_NAMES = frozenset(
    schema.name
    for schema in MANIFEST_SCHEMAS.values()
    if schema.legacy_defaults.get("cuda_graph_enabled")
    == LEGACY_UNRECORDED_CUDA_GRAPH_STATE
)
LEGACY_BEHAVIOR_SCHEMA_NAMES = frozenset(
    schema.name
    for schema in MANIFEST_SCHEMAS.values()
    if schema.legacy_defaults.get("ablation_behavior_revision")
    == LEGACY_UNRECORDED_ABLATION_BEHAVIOR
)


@dataclass(frozen=True)
class RunSummary:
    model: str
    runner: str
    variant: str
    gate_mode: str
    K: str
    threshold: str
    consecutive_checks: str
    roofline_config_sha256: str
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
    num_drafts: float | None
    num_draft_tokens: float | None
    num_accepted_tokens: float | None
    gate_decisions: float | None
    gate_activations: float | None
    gate_enabled_ratio: float | None
    gate_advance_only_ratio: float | None
    activation_tick: float | None
    activation_batch: float | None
    activation_seq_len: float | None
    predicted_speedup: float | None
    activation_predicted_speedup: float | None
    target_graph_ratio: float | None
    draft_graph_ratio: float | None
    draft_prefill_graph_ratio: float | None
    draft_decode_graph_ratio: float | None
    cuda_graph_fallback_count: float | None
    cuda_graph_evidence: str
    reward: float | None
    response_length: float | None
    approx_kl: float | None
    policy_loss: float | None
    status: str
    source: str
    reason: str
    cluster: str
    runtime: str
    runtime_version: str
    runtime_commit: str
    ablation_behavior_revision: str
    vllm_version: str
    vllm_commit: str
    target_tp: str
    draft_tp: str
    dp: str
    ep: str
    temperature: str
    top_p: str
    max_osl: str
    max_model_len: str
    max_sequence_length: str
    num_prompts: str
    num_generations: str
    train_gbs: str
    max_num_batched_tokens: str
    max_num_seqs: str
    recipe: str
    container: str
    container_sha256: str
    graph_mode: str
    cuda_graph_enabled: str
    enforce_eager: str
    sampling: str
    draft_sample_method: str
    wandb_run_id: str
    run_dir: str
    slurm_log_path: str
    ray_driver_log_path: str
    ray_log_dir: str
    launcher_command: str
    command: str
    target_checkpoint: str
    target_checkpoint_revision: str
    draft_checkpoint: str
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
    generation_time_speedup_vs_graph_on: float | None
    e2e_time_speedup_vs_graph_on: float | None
    generation_tps_gpu_speedup_vs_graph_on: float | None
    e2e_tps_gpu_speedup_vs_graph_on: float | None
    reward_health_passed: bool | None
    response_length_health_passed: bool | None
    approx_kl_health_passed: bool | None
    policy_loss_health_passed: bool | None
    cuda_graph_health_passed: bool | None
    gate_activation_health_passed: bool | None
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

    def scan_history(
        self, *, keys: list[str] | None = None
    ) -> Iterable[Mapping[str, object]]: ...


class WandbApi(Protocol):
    def run(self, path: str) -> WandbRun: ...


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_specdec(metadata: Mapping[str, str]) -> bool:
    return not metadata.get("variant", "").startswith("baseline_")


def _is_gated(metadata: Mapping[str, str]) -> bool:
    return metadata.get("gate_mode") in {"threshold", "roofline"}


def _required_metric_keys(metadata: Mapping[str, str]) -> dict[str, str]:
    required = dict(BASE_METRIC_KEYS)
    if metadata.get("cuda_graph_enabled") != "true":
        required.pop("target_graph_ratio")
    if _is_specdec(metadata):
        required.update(SPECDEC_METRIC_KEYS)
        if metadata.get("cuda_graph_enabled") == "true":
            required.update(
                V1_DRAFT_GRAPH_METRIC_KEYS
                if metadata.get("runner") == "v1"
                else V2_DRAFT_GRAPH_METRIC_KEYS
            )
    if _is_gated(metadata):
        required.update(TAIL_GATE_METRIC_KEYS)
    if metadata.get("gate_mode") == "roofline":
        required.update(ROOFLINE_ACTIVATION_METRIC_KEYS)
    return required


def _history_keys(metadata: Mapping[str, str]) -> list[str]:
    fallback_roles = ["target"] if metadata.get("cuda_graph_enabled") == "true" else []
    if _is_specdec(metadata) and fallback_roles:
        fallback_roles.extend(
            ["draft"]
            if metadata.get("runner") == "v1"
            else ["draft_prefill", "draft_decode", "draft_query"]
        )
    fallback_keys = [
        f"train/vllm/cudagraph_{role}_fallback_{reason}"
        for role in fallback_roles
        for reason in CUDAGRAPH_FALLBACK_REASONS
    ]
    return [
        "_step",
        *_required_metric_keys(metadata).values(),
        *fallback_keys,
    ]


def _scan_sparse_history(
    run: WandbRun, fallback_keys: list[str]
) -> list[Mapping[str, object]]:
    """Read sparse W&B rows without requiring every optional metric key."""
    try:
        history = list(run.scan_history())
    except TypeError:
        history = []
    if history and all("_step" in row for row in history):
        return history
    history = list(run.scan_history(keys=[]))
    if history and all("_step" in row for row in history):
        return history
    return list(run.scan_history(keys=fallback_keys))


def _comparison_key(metadata: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            field,
            "matched_specdec_method"
            if field == "draft_sample_method"
            else metadata.get(field, ""),
        )
        for field in (*COHORT_FIELDS, *CAPTURE_PROFILE_COHORT_FIELDS)
    )


def same_variant_graph_mode_key(
    summary: RunSummary,
) -> tuple[tuple[str, str], ...]:
    """Match graph ablations only within an otherwise identical variant."""
    return (
        ("variant", summary.variant),
        *(
            (field, getattr(summary, "K" if field == "k" else field))
            for field in SAME_VARIANT_GRAPH_MODE_FIELDS
        ),
    )


def _provenance(metadata: Mapping[str, str]) -> str:
    return json.dumps(
        dict(sorted(metadata.items())), separators=(",", ":"), sort_keys=True
    )


def _draft_sample_method(metadata: Mapping[str, str]) -> str:
    recorded = metadata.get("draft_sample_method", "")
    if recorded:
        return recorded
    if metadata.get("variant", "").startswith("baseline_"):
        return "not_applicable"
    return "legacy_unspecified"


def _make_summary(
    metadata: Mapping[str, str],
    steps: list[int],
    metrics: Mapping[str, float | None],
    status: str,
    reason: str,
) -> RunSummary:
    fallback_count = metrics.get("cuda_graph_fallback_count")
    fallback_display = (
        str(int(fallback_count))
        if fallback_count is not None and fallback_count.is_integer()
        else str(fallback_count)
    )
    if metadata.get("cuda_graph_enabled") != "true":
        cuda_graph_evidence = "not applicable: CUDA graphs disabled"
    else:
        cuda_graph_evidence = (
            "fallback counters unavailable; measured graph-call ratio threshold >= 0.99"
            if fallback_count is None
            else f"observed fallback counters={fallback_display}"
        )
    return RunSummary(
        model=metadata.get("model", ""),
        runner=metadata.get("runner", ""),
        variant=metadata.get("variant", ""),
        gate_mode=metadata.get("gate_mode", ""),
        K=metadata.get("k", ""),
        threshold=metadata.get("threshold", ""),
        consecutive_checks=metadata.get("consecutive_checks", ""),
        roofline_config_sha256=metadata.get("roofline_config_sha256", ""),
        steps=steps,
        job_id=metadata.get("job_id", ""),
        wandb_url=metadata.get("wandb_url", ""),
        e2e_time=metrics.get("e2e_time"),
        generation_time=metrics.get("generation_time"),
        e2e_tps_gpu=metrics.get("e2e_tps_gpu"),
        generation_tps_gpu=metrics.get("generation_tps_gpu"),
        policy_time=metrics.get("policy_time"),
        logprob_time=metrics.get("logprob_time"),
        acceptance_rate=metrics.get("acceptance_rate"),
        mean_accept_len=metrics.get("mean_accept_len"),
        num_drafts=metrics.get("num_drafts"),
        num_draft_tokens=metrics.get("num_draft_tokens"),
        num_accepted_tokens=metrics.get("num_accepted_tokens"),
        gate_decisions=metrics.get("gate_decisions"),
        gate_activations=metrics.get("gate_activations"),
        gate_enabled_ratio=metrics.get("gate_enabled_ratio"),
        gate_advance_only_ratio=metrics.get("gate_advance_only_ratio"),
        activation_tick=metrics.get("activation_tick"),
        activation_batch=metrics.get("activation_batch"),
        activation_seq_len=metrics.get("activation_seq_len"),
        predicted_speedup=metrics.get("predicted_speedup"),
        activation_predicted_speedup=metrics.get("activation_predicted_speedup"),
        target_graph_ratio=metrics.get("target_graph_ratio"),
        draft_graph_ratio=metrics.get("draft_graph_ratio"),
        draft_prefill_graph_ratio=metrics.get("draft_prefill_graph_ratio"),
        draft_decode_graph_ratio=metrics.get("draft_decode_graph_ratio"),
        cuda_graph_fallback_count=fallback_count,
        cuda_graph_evidence=cuda_graph_evidence,
        reward=metrics.get("reward"),
        response_length=metrics.get("response_length"),
        approx_kl=metrics.get("approx_kl"),
        policy_loss=metrics.get("policy_loss"),
        status=status,
        source=metadata.get("source", ""),
        reason=reason,
        cluster=metadata.get("cluster", ""),
        runtime=metadata.get("runtime", ""),
        runtime_version=metadata.get("runtime_version", ""),
        runtime_commit=metadata.get("runtime_commit", ""),
        ablation_behavior_revision=metadata.get("ablation_behavior_revision", ""),
        vllm_version=metadata.get("vllm_version", ""),
        vllm_commit=metadata.get("vllm_commit", ""),
        target_tp=metadata.get("target_tp", ""),
        draft_tp=metadata.get("draft_tp", ""),
        dp=metadata.get("dp", ""),
        ep=metadata.get("ep", ""),
        temperature=metadata.get("temperature", ""),
        top_p=metadata.get("top_p", ""),
        max_osl=metadata.get("max_osl", ""),
        max_model_len=metadata.get("max_model_len", ""),
        max_sequence_length=metadata.get("max_sequence_length", ""),
        num_prompts=metadata.get("num_prompts", ""),
        num_generations=metadata.get("num_generations", ""),
        train_gbs=metadata.get("train_gbs", ""),
        max_num_batched_tokens=metadata.get("max_num_batched_tokens", ""),
        max_num_seqs=metadata.get("max_num_seqs", ""),
        recipe=metadata.get("recipe", ""),
        container=metadata.get("container", ""),
        container_sha256=metadata.get("container_sha256", ""),
        graph_mode=metadata.get("graph_mode", ""),
        cuda_graph_enabled=metadata.get("cuda_graph_enabled", ""),
        enforce_eager=metadata.get("enforce_eager", ""),
        sampling=metadata.get("sampling", ""),
        draft_sample_method=_draft_sample_method(metadata),
        wandb_run_id=metadata.get("wandb_run_id", ""),
        run_dir=metadata.get("run_dir", ""),
        slurm_log_path=metadata.get("slurm_log_path", ""),
        ray_driver_log_path=metadata.get("ray_driver_log_path", ""),
        ray_log_dir=metadata.get("ray_log_dir", ""),
        launcher_command=metadata.get("launcher_command", ""),
        command=metadata.get("command", ""),
        target_checkpoint=metadata.get("target_checkpoint", ""),
        target_checkpoint_revision=metadata.get("target_checkpoint_revision", ""),
        draft_checkpoint=metadata.get("draft_checkpoint", ""),
        provenance=_provenance(metadata),
        comparison_key=_comparison_key(metadata),
    )


def _empty_summary(
    metadata: Mapping[str, str], reason: str, steps: list[int]
) -> RunSummary:
    return _make_summary(
        metadata,
        steps,
        {metric_name: None for metric_name in METRIC_KEYS},
        "partial",
        reason,
    )


def summarize_history(
    metadata: Mapping[str, str],
    history: Iterable[Mapping[str, object]],
    *,
    expected_steps: set[int] = EXPECTED_STEPS,
) -> RunSummary:
    """Average required production W&B metrics over the requested steps."""
    records_by_step: dict[int, Mapping[str, object]] = {}
    for record in history:
        step = record.get("_step")
        if (
            isinstance(step, int)
            and not isinstance(step, bool)
            and step in expected_steps
        ):
            records_by_step[step] = record

    steps = sorted(records_by_step)
    missing_steps = sorted(expected_steps - records_by_step.keys())
    if missing_steps:
        return _empty_summary(
            metadata, f"missing_steps:{','.join(map(str, missing_steps))}", steps
        )

    required = _required_metric_keys(metadata)
    values: dict[str, list[float]] = {metric_name: [] for metric_name in required}
    for step in sorted(expected_steps):
        record = records_by_step[step]
        for metric_name, wandb_key in required.items():
            if wandb_key not in record:
                return _empty_summary(
                    metadata, f"missing_metric:{metric_name}:{step}", steps
                )
            value = record[wandb_key]
            if not _is_finite_number(value):
                return _empty_summary(
                    metadata, f"non_finite_metric:{metric_name}:{step}", steps
                )
            values[metric_name].append(float(value))

    averages: dict[str, float | None] = {
        metric_name: None for metric_name in METRIC_KEYS
    }
    averages.update(
        {
            metric_name: statistics.fmean(metric_values)
            for metric_name, metric_values in values.items()
        }
    )
    observed_fallbacks: list[float] = []
    for step in sorted(expected_steps):
        record = records_by_step[step]
        for key in CUDAGRAPH_FALLBACK_KEYS:
            if key not in record:
                continue
            value = record[key]
            if not _is_finite_number(value):
                return _empty_summary(
                    metadata, f"non_finite_metric:cuda_graph_fallback:{step}", steps
                )
            observed_fallbacks.append(float(value))
    averages["cuda_graph_fallback_count"] = (
        sum(observed_fallbacks) if observed_fallbacks else None
    )
    for metric_name in (
        "e2e_time",
        "generation_time",
        "e2e_tps_gpu",
        "generation_tps_gpu",
    ):
        if cast(float, averages[metric_name]) <= 0.0:
            return _empty_summary(metadata, f"non_positive_metric:{metric_name}", steps)
    if _is_specdec(metadata):
        for metric_name in ("num_drafts", "num_draft_tokens"):
            if cast(float, averages[metric_name]) <= 0.0:
                return _empty_summary(
                    metadata, f"missing_specdec_evidence:{metric_name}", steps
                )
    return _make_summary(metadata, steps, averages, "final", "")


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


def _required_graph_metrics(summary: RunSummary) -> tuple[str, ...]:
    if summary.variant.startswith("baseline_"):
        return ("target_graph_ratio",)
    if summary.runner == "v1":
        return ("target_graph_ratio", "draft_graph_ratio")
    return (
        "target_graph_ratio",
        "draft_prefill_graph_ratio",
        "draft_decode_graph_ratio",
    )


def _graph_health(summary: RunSummary) -> bool | None:
    if summary.cuda_graph_enabled != "true":
        return None
    if (
        summary.cuda_graph_fallback_count is not None
        and summary.cuda_graph_fallback_count > 0.0
    ):
        return False
    for metric_name in _required_graph_metrics(summary):
        value = cast(float | None, getattr(summary, metric_name))
        if (
            value is None
            or (
                value < MIN_GRAPH_CALL_RATIO
                and not math.isclose(value, MIN_GRAPH_CALL_RATIO, abs_tol=1e-12)
            )
            or value > 1.0
        ):
            return False
    return True


def _gate_activation_health(summary: RunSummary) -> bool | None:
    if summary.gate_mode not in {"threshold", "roofline"}:
        return None
    values = (
        summary.gate_decisions,
        summary.gate_activations,
        summary.gate_enabled_ratio,
        summary.gate_advance_only_ratio,
        summary.activation_tick,
        summary.activation_batch,
        summary.activation_seq_len,
    )
    if any(value is None for value in values):
        return False
    try:
        max_worker_ranks = float(summary.dp) * float(summary.target_tp)
    except ValueError:
        return False
    if (
        cast(float, summary.gate_decisions) <= 0.0
        or not 0.0 < cast(float, summary.gate_activations) <= max_worker_ranks
        or not 0.0 < cast(float, summary.gate_enabled_ratio) < 1.0
        or not 0.0 < cast(float, summary.gate_advance_only_ratio) < 1.0
        or cast(float, summary.activation_tick) <= 0.0
        or cast(float, summary.activation_batch) <= 0.0
        or cast(float, summary.activation_seq_len) <= 0.0
        or not math.isclose(
            cast(float, summary.gate_enabled_ratio)
            + cast(float, summary.gate_advance_only_ratio),
            1.0,
            abs_tol=0.01,
        )
    ):
        return False
    return not (
        summary.gate_mode == "roofline"
        and (
            summary.activation_predicted_speedup is None
            or summary.activation_predicted_speedup < MIN_ROOFLINE_PREDICTED_SPEEDUP
        )
    )


def _empty_comparison_row(summary: RunSummary) -> ComparisonRow:
    return ComparisonRow(summary, *([None] * 19))


def _comparison_row(
    summary: RunSummary, baseline: RunSummary, always_on: RunSummary | None
) -> ComparisonRow:
    if summary.status != "final":
        return _empty_comparison_row(summary)
    health = {
        metric_name: _within_ten_percent(
            cast(float | None, getattr(summary, metric_name)),
            cast(float | None, getattr(baseline, metric_name)),
        )
        for metric_name in HEALTH_METRICS
    }
    graph_health = _graph_health(summary)
    gate_health = _gate_activation_health(summary)
    overall_health = (
        all(health.values())
        and graph_health is not False
        and gate_health is not False
    )
    if not overall_health:
        failed = [name for name, passed in health.items() if not passed]
        if graph_health is False:
            failed.append("cuda_graph")
        if gate_health is False:
            failed.append("gate_activation")
        summary = replace(
            summary,
            status="health_failed",
            reason=f"health_gate_failed:{','.join(failed)}",
        )
    return ComparisonRow(
        summary,
        _speedup(baseline.generation_time, summary.generation_time),
        _speedup(baseline.e2e_time, summary.e2e_time),
        _speedup(summary.generation_tps_gpu, baseline.generation_tps_gpu),
        _speedup(summary.e2e_tps_gpu, baseline.e2e_tps_gpu),
        _speedup(always_on.generation_time, summary.generation_time)
        if always_on
        else None,
        _speedup(always_on.e2e_time, summary.e2e_time) if always_on else None,
        _speedup(summary.generation_tps_gpu, always_on.generation_tps_gpu)
        if always_on
        else None,
        _speedup(summary.e2e_tps_gpu, always_on.e2e_tps_gpu) if always_on else None,
        None,
        None,
        None,
        None,
        health["reward"],
        health["response_length"],
        health["approx_kl"],
        health["policy_loss"],
        graph_health,
        gate_health,
        overall_health,
    )


def build_comparison_rows(summaries: Iterable[RunSummary]) -> list[ComparisonRow]:
    """Build speedups only inside the required explicit cohort key."""
    grouped: dict[tuple[tuple[str, str], ...], list[RunSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.comparison_key, []).append(summary)

    rows: list[ComparisonRow] = []
    for key, cohort in grouped.items():
        variants = [summary.variant for summary in cohort]
        if len(variants) != len(set(variants)):
            raise ValueError(f"duplicate variants in cohort:{dict(key)}")
        baseline_methods = {
            summary.draft_sample_method
            for summary in cohort
            if summary.variant.startswith("baseline_")
        }
        if baseline_methods and baseline_methods != {"not_applicable"}:
            raise ValueError(f"invalid baseline draft_sample_method:{dict(key)}")
        specdec_methods = {
            summary.draft_sample_method
            for summary in cohort
            if not summary.variant.startswith("baseline_")
        }
        if len(specdec_methods) > 1:
            raise ValueError(f"mixed draft_sample_method:{dict(key)}")
        runner = cohort[0].runner
        baselines = [
            summary for summary in cohort if summary.variant == f"baseline_{runner}"
        ]
        if len(baselines) != 1 or baselines[0].status != "final":
            raise ValueError(f"missing matched baseline for cohort:{dict(key)}")
        baseline = baselines[0]
        baseline_row = _comparison_row(baseline, baseline, None)
        if baseline_row.health_gate_passed is not True:
            raise ValueError(f"unhealthy matched baseline for cohort:{dict(key)}")

        always_on = next(
            (
                summary
                for summary in cohort
                if summary.variant == f"always_on_{runner}_k5"
                and summary.status == "final"
            ),
            None,
        )
        if any(_is_gated({"gate_mode": summary.gate_mode}) for summary in cohort):
            if always_on is None:
                raise ValueError(f"missing matched always-on for cohort:{dict(key)}")
            if (
                _comparison_row(always_on, baseline, None).health_gate_passed
                is not True
            ):
                raise ValueError(f"unhealthy matched always-on for cohort:{dict(key)}")
        rows.extend(_comparison_row(summary, baseline, always_on) for summary in cohort)
    return sorted(
        rows, key=lambda row: (row.runner, row.model, row.variant, row.job_id)
    )


def add_graph_ablation_deltas(rows: Iterable[ComparisonRow]) -> list[ComparisonRow]:
    """Attach graph-off versus matched graph-on effects to report rows."""
    materialized = list(rows)
    grouped: dict[tuple[tuple[str, str], ...], list[ComparisonRow]] = {}
    for row in materialized:
        grouped.setdefault(same_variant_graph_mode_key(row.summary), []).append(row)

    updated: list[ComparisonRow] = []
    for row in materialized:
        if row.cuda_graph_enabled != "false":
            updated.append(row)
            continue
        matches = grouped[same_variant_graph_mode_key(row.summary)]
        graph_on = [candidate for candidate in matches if candidate.cuda_graph_enabled == "true"]
        if len(graph_on) != 1 or row.status != "final" or graph_on[0].status != "final":
            updated.append(row)
            continue
        reference = graph_on[0]
        updated.append(
            replace(
                row,
                generation_time_speedup_vs_graph_on=_speedup(
                    reference.generation_time, row.generation_time
                ),
                e2e_time_speedup_vs_graph_on=_speedup(
                    reference.e2e_time, row.e2e_time
                ),
                generation_tps_gpu_speedup_vs_graph_on=_speedup(
                    row.generation_tps_gpu, reference.generation_tps_gpu
                ),
                e2e_tps_gpu_speedup_vs_graph_on=_speedup(
                    row.e2e_tps_gpu, reference.e2e_tps_gpu
                ),
            )
        )
    return updated


def _read_manifest(path: Path) -> tuple[ManifestSchema, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header = tuple(reader.fieldnames or ())
        schema = MANIFEST_SCHEMAS.get(header)
        if schema is None:
            raise ValueError(f"unsupported manifest header:{','.join(header)}")
        rows = []
        for raw_row in reader:
            row = {
                **schema.legacy_defaults,
                **raw_row,
                "_manifest_schema": schema.name,
            }
            if (
                row.get("runtime_commit") == APPROVED_GRAPH_ON_RUNTIME_COMMIT
                and row.get("graph_mode") in {"FULL_AND_PIECEWISE", "PIECEWISE"}
            ):
                if "cuda_graph_enabled" not in schema.header:
                    row["cuda_graph_enabled"] = "true"
                    row["enforce_eager"] = "false"
                if row.get("cuda_graph_enabled") == "true" and row.get(
                    "ablation_behavior_revision"
                ) == LEGACY_UNRECORDED_ABLATION_BEHAVIOR:
                    row["ablation_behavior_revision"] = (
                        APPROVED_GRAPH_ON_RUNTIME_COMMIT
                    )
            rows.append(row)
        return schema, rows


def _validate_manifest_rows(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return "empty manifest"
    seen: set[tuple[tuple[tuple[str, str], ...], str]] = set()
    seen_job_ids: set[str] = set()
    seen_wandb_run_ids: set[str] = set()
    for row in rows:
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not row.get(field)]
        if missing:
            return f"missing manifest fields:{','.join(missing)}"
        graph_state = row["cuda_graph_enabled"]
        enforce_eager = row["enforce_eager"]
        legacy_schema = row.get("_manifest_schema", "")
        has_legacy_state = LEGACY_UNRECORDED_CUDA_GRAPH_STATE in {
            graph_state,
            enforce_eager,
        }
        if (
            has_legacy_state
            and legacy_schema not in LEGACY_GRAPH_STATE_SCHEMA_NAMES
        ):
            return "legacy graph state requires legacy schema"
        behavior_revision = row["ablation_behavior_revision"]
        if (
            behavior_revision == LEGACY_UNRECORDED_ABLATION_BEHAVIOR
            and legacy_schema not in LEGACY_BEHAVIOR_SCHEMA_NAMES
        ):
            return "legacy ablation behavior requires legacy schema"
        if graph_state == "true":
            if enforce_eager != "false":
                return "invalid graph-on enforce_eager"
            expected_mode = (
                "FULL_AND_PIECEWISE" if row["runner"] == "v2" else "PIECEWISE"
            )
            if row["graph_mode"] != expected_mode:
                return "invalid graph-on graph_mode"
            if row["runner"] == "v2" and any(
                row.get(field) in {"", "not_applicable", LEGACY_UNRECORDED_CAPTURE_PROFILE}
                for field in (
                    "cudagraph_max_requests",
                    "cudagraph_max_tokens",
                    "cudagraph_capture_sizes",
                )
            ):
                return "invalid graph-on capture profile"
        elif graph_state == "false":
            if enforce_eager != "true":
                return "invalid graph-off enforce_eager"
            if row["graph_mode"] != "NONE":
                return "invalid graph-off graph_mode"
            if any(
                row.get(field) != "not_applicable"
                for field in (
                    "cudagraph_max_requests",
                    "cudagraph_max_tokens",
                    "cudagraph_capture_sizes",
                )
            ):
                return "invalid graph-off capture profile"
        elif not has_legacy_state:
            return f"invalid cuda_graph_enabled:{graph_state}"
        if row["job_id"] in seen_job_ids:
            return f"duplicate job_id:{row['job_id']}"
        seen_job_ids.add(row["job_id"])
        if row["wandb_run_id"] in seen_wandb_run_ids:
            return f"duplicate wandb_run_id:{row['wandb_run_id']}"
        seen_wandb_run_ids.add(row["wandb_run_id"])
        capture_profile = row.get(
            "cudagraph_max_requests", LEGACY_UNRECORDED_CAPTURE_PROFILE
        )
        if capture_profile != LEGACY_UNRECORDED_CAPTURE_PROFILE:
            missing_capture_fields = [
                field
                for field in (
                    "cudagraph_max_requests",
                    "cudagraph_max_tokens",
                    "cudagraph_capture_sizes",
                )
                if not row.get(field)
            ]
            if missing_capture_fields:
                return (
                    "missing CUDA graph capture profile fields:"
                    f"{','.join(missing_capture_fields)}"
                )
        draft_sample_method = _draft_sample_method(row)
        if row["variant"].startswith("baseline_"):
            if draft_sample_method != "not_applicable":
                return f"invalid baseline draft_sample_method:{draft_sample_method}"
        elif row.get("draft_sample_method") and draft_sample_method not in {
            "greedy",
            "probabilistic",
        }:
            return f"invalid SpecDec draft_sample_method:{draft_sample_method}"
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
        try:
            max_osl = int(row["max_osl"])
            max_model_len = row["max_model_len"]
            if max_model_len == LEGACY_UNRECORDED_MAX_MODEL_LEN:
                continue
            max_model_len_int = int(max_model_len)
        except ValueError:
            return "max_osl and max_model_len must be integers"
        if max_model_len_int < max_osl + MIN_SPECDEC_HEADROOM_TOKENS:
            return (
                "max_model_len must be at least max_osl plus "
                f"{MIN_SPECDEC_HEADROOM_TOKENS}:{max_model_len_int}:{max_osl}"
            )
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


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: tuple[str, ...] = REQUIRED_ROW_FIELDS,
) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
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
        "<style>.tail-gated-specdec{font:13px sans-serif}.tail-gated-specdec table{border-collapse:collapse;width:100%;margin:8px 0 14px}.tail-gated-specdec th,.tail-gated-specdec td{border:1px solid #c9c9c9;padding:4px;text-align:right}.tail-gated-specdec th:first-child,.tail-gated-specdec td:first-child{text-align:left}.tail-gated-specdec .partial,.tail-gated-specdec .health_failed{background:#fff4cc}.tail-gated-specdec .legend{text-align:center;margin:4px}.tail-gated-specdec .bar{height:9px;background:#76b900;display:inline-block}</style>",
    ]
    final_rows = [
        row
        for row in rows
        if row["status"] == "final" and row["health_gate_passed"] is True
    ]
    for runner in ("v1", "v2"):
        runner_rows = [row for row in rows if row["runner"] == runner]
        if not runner_rows:
            continue
        fragments.append(f"<h3>Model Runner {runner.upper()}</h3>")
        fragments.append(
            '<div class="legend">Bars: E2E time speedup vs matched baseline<br>Table: validated final and non-final cohort rows</div>'
        )
        for model in sorted({cast(str, row["model"]) for row in runner_rows}):
            model_rows = [row for row in runner_rows if row["model"] == model]
            fragments.append(f"<h4>{html.escape(model)}</h4><div>")
            for row in model_rows:
                speedup = row["e2e_time_speedup_vs_baseline"]
                width = (
                    0
                    if speedup is None
                    else min(100.0, max(0.0, (cast(float, speedup) - 0.5) * 80))
                )
                fragments.append(
                    f'<span class="bar" style="width:{width:.0f}px"></span> {html.escape(cast(str, row["variant"]))} {_format_metric(speedup)}<br>'
                )
            fragments.append("</div>")
        fragments.append(
            "<table><thead><tr><th>Variant</th><th>Status</th><th>E2E x</th><th>Gen x</th><th>Health</th><th>W&B</th></tr></thead><tbody>"
        )
        for row in runner_rows:
            status = cast(str, row["status"])
            row_class = status if status != "final" else ""
            url = html.escape(cast(str, row["wandb_url"]), quote=True)
            link = f'<a href="{url}">run</a>' if url else "-"
            fragments.append(
                f'<tr class="{row_class}"><td>{html.escape(cast(str, row["variant"]))}</td><td>{html.escape(status)}</td><td>{_format_metric(row["e2e_time_speedup_vs_baseline"])}</td><td>{_format_metric(row["generation_time_speedup_vs_baseline"])}</td><td>{row["health_gate_passed"] if row["health_gate_passed"] is not None else "-"}</td><td>{link}</td></tr>'
            )
        fragments.append("</tbody></table>")
    graph_effect_rows = [
        row
        for row in rows
        if row["cuda_graph_enabled"] == "false"
        and row["e2e_time_speedup_vs_graph_on"] is not None
    ]
    if graph_effect_rows:
        fragments.append(
            "<h3>CUDA Graph Effect</h3><table><thead><tr><th>Variant</th>"
            "<th>Runner</th><th>E2E x vs graph on</th>"
            "<th>Generation x vs graph on</th></tr></thead><tbody>"
        )
        for row in graph_effect_rows:
            fragments.append(
                f"<tr><td>{html.escape(cast(str, row['variant']))}</td>"
                f"<td>{html.escape(cast(str, row['runner']))}</td>"
                f"<td>{_format_metric(row['e2e_time_speedup_vs_graph_on'])}</td>"
                f"<td>{_format_metric(row['generation_time_speedup_vs_graph_on'])}</td></tr>"
            )
        fragments.append("</tbody></table>")
    finding_rows = [
        row
        for row in final_rows
        if not cast(str, row["variant"]).startswith("baseline_")
    ]
    if finding_rows:
        best = max(
            finding_rows,
            key=lambda row: cast(float, row["e2e_time_speedup_vs_baseline"] or 0.0),
        )
        fragments.append(
            f"<p>Final finding: {html.escape(cast(str, best['variant']))} has the highest matched E2E time speedup ({_format_metric(best['e2e_time_speedup_vs_baseline'])}x).</p>"
        )
    fragments.append("</section>\n")
    return "\n".join(fragments)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tail-gated SpecDec W&B runs."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--manifest-project",
        action="append",
        default=[],
        metavar="MANIFEST=PROJECT",
        help="repeatable manifest-to-W&B-project binding",
    )
    parser.add_argument("--entity", default="nvidia")
    parser.add_argument("--project", default="nemorl-vllm024-tail-gated-lyris")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest is None and not args.manifest_project:
        parser.error("one of --manifest or --manifest-project is required")
    if args.manifest is not None and args.manifest_project:
        parser.error("--manifest cannot be combined with --manifest-project")
    return args


def _manifest_project_bindings(args: argparse.Namespace) -> list[tuple[Path, str]]:
    if args.manifest is not None:
        return [(args.manifest, args.project)]
    bindings = []
    for value in args.manifest_project:
        manifest, separator, project = value.rpartition("=")
        if not separator or not manifest or not project:
            raise ValueError(
                f"invalid --manifest-project binding (expected MANIFEST=PROJECT):{value}"
            )
        bindings.append((Path(manifest), project))
    return bindings


def _create_wandb_api() -> WandbApi:
    wandb = importlib.import_module("wandb")
    return cast(WandbApi, wandb.Api())


def _claim_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite or share claimed cohort output: {path}"
        ) from error


def main(argv: list[str] | None = None, *, api: WandbApi | None = None) -> int:
    args = _parse_args(argv)
    bound_rows: list[tuple[Path, str, dict[str, str]]] = []
    for manifest, project in _manifest_project_bindings(args):
        _, manifest_rows = _read_manifest(manifest)
        manifest_error = _validate_manifest_rows(manifest_rows)
        if manifest_error:
            raise ValueError(f"{manifest}:{manifest_error}")
        bound_rows.extend((manifest, project, row) for row in manifest_rows)
    _claim_output_directory(args.output_dir)

    client = api if api is not None else _create_wandb_api()
    summaries: list[RunSummary] = []
    for manifest, project, manifest_row in bound_rows:
        metadata = {
            **manifest_row,
            "source": manifest_row.get("source") or manifest.name,
        }
        try:
            run = client.run(f"{args.entity}/{project}/{metadata['wandb_run_id']}")
            if not metadata.get("wandb_url"):
                metadata["wandb_url"] = run.url
            summaries.append(
                summarize_history(
                    metadata, _scan_sparse_history(run, _history_keys(metadata))
                )
            )
        except Exception as error:  # W&B failures become provenance-preserving rows.
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
                _empty_comparison_row(
                    replace(
                        summary,
                        status="partial",
                        reason=f"comparison_failed:{error}",
                    )
                )
                for summary in cohort
            )
    comparison_rows = add_graph_ablation_deltas(comparison_rows)
    rows = [
        row.to_dict()
        for row in sorted(
            comparison_rows,
            key=lambda row: (row.runner, row.model, row.variant, row.job_id),
        )
    ]
    _write_atomic(
        args.output_dir / "summary.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(args.output_dir / "summary.csv", rows)
    _write_atomic(args.output_dir / "tail_gated_specdec.html", _render_html(rows))
    return int(any(row["status"] != "final" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
