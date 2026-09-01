"""Immutable result records for one Q30 benchmark engine."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .contract import ExperimentContract, MethodPlan


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRACE_SOURCE_KINDS = frozenset({"nsys", "ncu", "torch_profiler", "kineto"})
_DRAFTER_MAX_CONFIGURED_K = {"dflash": 7, "dspark": 8}


@dataclass(frozen=True, slots=True)
class CudaGraphEvidence:
    """Observed CUDA Graph capture modes for target and drafter runners."""

    mode: str
    target_full: bool
    target_piecewise: bool
    drafter_full: bool | None
    drafter_piecewise: bool | None
    drafter_decode_full: bool | None

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "target_full": self.target_full,
            "target_piecewise": self.target_piecewise,
            "drafter_full": self.drafter_full,
            "drafter_piecewise": self.drafter_piecewise,
            "drafter_decode_full": self.drafter_decode_full,
        }


@dataclass(frozen=True, slots=True)
class DrafterTraceEvidence:
    """Profiler artifact with monotonic offsets from generation-run start."""

    run_id: str
    source_kind: str
    clock_domain: str
    artifact_uri: str
    artifact_sha256: str
    artifact_size_bytes: int
    capture_start_offset_seconds: float
    capture_end_offset_seconds: float
    capture_duration_seconds: float
    draft_kernel_count: int
    draft_kernel_time_seconds: float
    observed_query_width: int
    observed_output_width: int

    @property
    def observed_execution(self) -> bool:
        return self.draft_kernel_count > 0

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "source_kind": self.source_kind,
            "clock_domain": self.clock_domain,
            "artifact_uri": self.artifact_uri,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "capture_start_offset_seconds": self.capture_start_offset_seconds,
            "capture_end_offset_seconds": self.capture_end_offset_seconds,
            "capture_duration_seconds": self.capture_duration_seconds,
            "draft_kernel_count": self.draft_kernel_count,
            "draft_kernel_time_seconds": self.draft_kernel_time_seconds,
            "observed_query_width": self.observed_query_width,
            "observed_output_width": self.observed_output_width,
        }


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Sealed identity and topology reported by the live worker runtime."""

    container_path: str
    container_digest: str
    vllm_version: str
    vllm_commit: str
    target_path: str
    target_config_sha256: str
    drafter_path: str | None
    drafter_config_sha256: str | None
    prompt_manifest_sha256: str
    tensor_parallel_size: int
    data_parallel_size: int
    external_engine_count: int
    worker_index: int
    slurm_job_id: str
    cuda_graph_evidence: CudaGraphEvidence

    def to_payload(self) -> dict[str, object]:
        return {
            "container_path": self.container_path,
            "container_digest": self.container_digest,
            "vllm_version": self.vllm_version,
            "vllm_commit": self.vllm_commit,
            "target_path": self.target_path,
            "target_config_sha256": self.target_config_sha256,
            "drafter_path": self.drafter_path,
            "drafter_config_sha256": self.drafter_config_sha256,
            "prompt_manifest_sha256": self.prompt_manifest_sha256,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "external_engine_count": self.external_engine_count,
            "worker_index": self.worker_index,
            "slurm_job_id": self.slurm_job_id,
            "cuda_graph_evidence": self.cuda_graph_evidence.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class SpecDecodeMetrics:
    """SpecDec counters plus per-sequence K decisions and drafter evidence."""

    proposed_tokens: int
    accepted_tokens: int
    draft_iterations: int
    selected_k_histogram: Mapping[int, int]
    selected_verifier_k: int | None
    configured_draft_k: int | None
    drafter_trace: DrafterTraceEvidence | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_k_histogram",
            MappingProxyType(dict(self.selected_k_histogram)),
        )

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @property
    def mean_accepted_length(self) -> float:
        if self.draft_iterations == 0:
            return 0.0
        return 1.0 + self.accepted_tokens / self.draft_iterations

    @property
    def observed_drafter_execution(self) -> bool | None:
        if self.drafter_trace is None:
            return None
        return self.drafter_trace.observed_execution

    def to_payload(self) -> dict[str, object]:
        return {
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "draft_iterations": self.draft_iterations,
            "acceptance_rate": self.acceptance_rate,
            "mean_accepted_length": self.mean_accepted_length,
            "selected_k_histogram": {
                str(key): value
                for key, value in sorted(self.selected_k_histogram.items())
            },
            "selected_verifier_k": self.selected_verifier_k,
            "configured_draft_k": self.configured_draft_k,
            "observed_drafter_execution": self.observed_drafter_execution,
            "drafter_trace": (
                None if self.drafter_trace is None else self.drafter_trace.to_payload()
            ),
        }


@dataclass(frozen=True, slots=True)
class RequestResult:
    """Concrete completion and finish timing for one deterministic request."""

    request_id: str
    global_request_index: int
    prompt_index: int
    generation_index: int
    seed: int
    max_tokens: int
    ignore_eos: bool
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    finish_seconds: float

    def to_payload(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "global_request_index": self.global_request_index,
            "prompt_index": self.prompt_index,
            "generation_index": self.generation_index,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "output_tokens": len(self.token_ids),
            "finish_reason": self.finish_reason,
            "finish_seconds": self.finish_seconds,
        }


@dataclass(frozen=True, slots=True)
class WorkerSummary:
    """Literal elapsed/token summary for one externally coordinated engine."""

    elapsed_seconds: float
    barrier_seconds: float
    request_count: int
    output_tokens: int
    output_tokens_per_second: float

    def to_payload(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "barrier_seconds": self.barrier_seconds,
            "request_count": self.request_count,
            "output_tokens": self.output_tokens,
            "output_tokens_per_second": self.output_tokens_per_second,
        }


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """One immutable, publication-ready engine result."""

    schema_version: int
    status: str
    run_id: str
    attempt_index: int
    method_plan: MethodPlan
    max_tokens: int
    temperature: float
    top_p: float
    seed_policy: str
    natural_eos: bool
    runtime_provenance: RuntimeProvenance
    summary: WorkerSummary
    spec_decode: SpecDecodeMetrics
    rows: tuple[RequestResult, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "run_id": self.run_id,
            "attempt_index": self.attempt_index,
            "method_plan": {
                "key": self.method_plan.key,
                "stage": self.method_plan.stage,
                "drafter": self.method_plan.drafter,
                "method": self.method_plan.method,
                "controller": self.method_plan.controller,
                "batch_size": self.method_plan.batch_size,
                "verifier_k": self.method_plan.verifier_k,
                "physical_block_size": self.method_plan.physical_block_size,
            },
            "sampling": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed_policy": self.seed_policy,
                "natural_eos": self.natural_eos,
            },
            "runtime_provenance": self.runtime_provenance.to_payload(),
            "summary": self.summary.to_payload(),
            "spec_decode": self.spec_decode.to_payload(),
            "rows": [row.to_payload() for row in self.rows],
        }


def _require_sha256(value: str, field_name: str, *, prefixed: bool = False) -> None:
    digest = value.removeprefix("sha256:") if prefixed else value
    if prefixed and not value.startswith("sha256:"):
        raise ValueError(f"{field_name} must use a sha256: prefix")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must contain a lowercase SHA256 digest")


def _validate_runtime_provenance(
    provenance: RuntimeProvenance,
    *,
    contract: ExperimentContract,
    plan: MethodPlan,
    prompt_manifest_sha256: str,
) -> None:
    pinned_values = {
        "container_path": contract.container_path,
        "vllm_version": contract.vllm_version,
        "vllm_commit": contract.vllm_commit,
        "target_path": contract.target_path,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "tensor_parallel_size": contract.tensor_parallel_size,
        "data_parallel_size": contract.data_parallel_size,
        "external_engine_count": contract.engine_count,
    }
    for field_name, expected in pinned_values.items():
        if getattr(provenance, field_name) != expected:
            raise ValueError(f"runtime {field_name} does not match the contract")
    expected_drafter = (
        None if plan.drafter is None else contract.drafter_paths[plan.drafter]
    )
    if provenance.drafter_path != expected_drafter:
        raise ValueError("runtime drafter_path does not match the method plan")
    if not 0 <= provenance.worker_index < contract.engine_count:
        raise ValueError("runtime worker_index is outside the external engine topology")
    if not provenance.slurm_job_id:
        raise ValueError("runtime slurm_job_id must be recorded")
    _require_sha256(
        provenance.container_digest,
        "container_digest",
        prefixed=True,
    )
    _require_sha256(provenance.target_config_sha256, "target_config_sha256")
    if plan.drafter is None:
        if provenance.drafter_config_sha256 is not None:
            raise ValueError("baseline drafter_config_sha256 must be null")
    elif provenance.drafter_config_sha256 is None:
        raise ValueError("drafter_config_sha256 must be recorded")
    else:
        _require_sha256(
            provenance.drafter_config_sha256,
            "drafter_config_sha256",
        )
    _require_sha256(
        provenance.prompt_manifest_sha256,
        "prompt_manifest_sha256",
    )

    graph = provenance.cuda_graph_evidence
    if (
        graph.mode != contract.cuda_graph_mode
        or graph.target_full is not True
        or graph.target_piecewise is not True
    ):
        raise ValueError("missing target FULL_AND_PIECEWISE CUDA Graph evidence")
    if plan.drafter is None:
        if any(
            value is not None
            for value in (
                graph.drafter_full,
                graph.drafter_piecewise,
                graph.drafter_decode_full,
            )
        ):
            raise ValueError("baseline must not report drafter CUDA Graph evidence")
    elif not all(
        value is True
        for value in (
            graph.drafter_full,
            graph.drafter_piecewise,
            graph.drafter_decode_full,
        )
    ):
        raise ValueError("missing drafter FULL_AND_PIECEWISE CUDA Graph evidence")


def _expected_request_indices(
    contract: ExperimentContract,
    plan: MethodPlan,
    worker_index: int,
) -> tuple[int, ...]:
    if plan.stage == "calibration":
        if plan.batch_size is None:
            raise ValueError("calibration plan is missing batch_size")
        return tuple(range(plan.batch_size))
    first_index = worker_index * contract.requests_per_engine
    return tuple(range(first_index, first_index + contract.requests_per_engine))


def _validate_rows_and_summary(
    result: WorkerResult,
    *,
    contract: ExperimentContract,
    expected_indices: tuple[int, ...],
) -> None:
    actual_indices = tuple(row.global_request_index for row in result.rows)
    if actual_indices != expected_indices:
        raise ValueError("result rows do not contain the exact request work")
    if len({row.request_id for row in result.rows}) != len(result.rows):
        raise ValueError("result rows contain duplicate request IDs")
    finish_times = tuple(row.finish_seconds for row in result.rows)
    if any(
        not math.isfinite(value)
        or not 0.0 <= value <= result.summary.elapsed_seconds
        for value in finish_times
    ):
        raise ValueError(
            "row finish_seconds must be finite and within generation timing"
        )

    for row in result.rows:
        expected_prompt, expected_generation = divmod(
            row.global_request_index,
            contract.generations_per_prompt,
        )
        if row.request_id != f"request-{row.global_request_index:04d}":
            raise ValueError("request_id does not match global_request_index")
        if (
            row.prompt_index != expected_prompt
            or row.generation_index != expected_generation
        ):
            raise ValueError("row prompt partition does not match global request work")
        if row.seed != contract.seed_for_request(row.global_request_index):
            raise ValueError("row seed does not match the deterministic seed policy")
        if row.max_tokens != contract.max_tokens or row.max_tokens > 1_024:
            raise ValueError("row max_tokens exceeds the OSL1K contract")
        if row.ignore_eos is not False:
            raise ValueError("row must retain natural EOS")
        if not 0 <= len(row.token_ids) <= row.max_tokens:
            raise ValueError("row output_tokens exceed max_tokens")
        if any(type(token_id) is not int for token_id in row.token_ids):
            raise ValueError("row token_ids must contain integers")

    output_tokens = sum(len(row.token_ids) for row in result.rows)
    summary = result.summary
    if not math.isfinite(summary.elapsed_seconds) or summary.elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive and finite")
    if summary.barrier_seconds != summary.elapsed_seconds:
        raise ValueError("one-engine barrier_seconds must equal elapsed_seconds")
    if summary.request_count != len(expected_indices):
        raise ValueError("summary request_count does not match exact request work")
    if summary.output_tokens != output_tokens:
        raise ValueError("summary output_tokens do not match completion rows")
    expected_tps = output_tokens / summary.elapsed_seconds
    if not math.isclose(
        summary.output_tokens_per_second,
        expected_tps,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("summary output_tokens_per_second is inconsistent")


def _validate_trace_evidence(trace: DrafterTraceEvidence) -> None:
    if not trace.run_id:
        raise ValueError("drafter trace run_id must be recorded")
    if trace.source_kind not in _TRACE_SOURCE_KINDS:
        raise ValueError("drafter trace source_kind is not a supported profiler")
    if trace.clock_domain != "monotonic":
        raise ValueError("drafter trace clock_domain must be monotonic")
    if not trace.artifact_uri:
        raise ValueError("drafter trace artifact_uri must be recorded")
    _require_sha256(trace.artifact_sha256, "drafter trace artifact_sha256")
    if type(trace.artifact_size_bytes) is not int or trace.artifact_size_bytes <= 0:
        raise ValueError("drafter trace artifact_size_bytes must be positive")
    interval_values = (
        trace.capture_start_offset_seconds,
        trace.capture_end_offset_seconds,
        trace.capture_duration_seconds,
        trace.draft_kernel_time_seconds,
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in interval_values
    ) or any(
        value < 0
        for value in (
            trace.capture_end_offset_seconds,
            trace.capture_duration_seconds,
            trace.draft_kernel_time_seconds,
        )
    ):
        raise ValueError("drafter trace timing must be finite and valid")
    if (
        trace.capture_end_offset_seconds
        <= trace.capture_start_offset_seconds
        or trace.capture_duration_seconds <= 0
        or not math.isclose(
            trace.capture_end_offset_seconds
            - trace.capture_start_offset_seconds,
            trace.capture_duration_seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("drafter trace capture interval and duration conflict")
    if trace.draft_kernel_time_seconds > trace.capture_duration_seconds:
        raise ValueError("drafter trace kernel time exceeds its capture interval")
    integer_fields = (
        trace.draft_kernel_count,
        trace.observed_query_width,
        trace.observed_output_width,
    )
    if any(type(value) is not int or value < 0 for value in integer_fields):
        raise ValueError("drafter trace counts and widths must be nonnegative integers")
    if trace.draft_kernel_count == 0:
        if (
            trace.draft_kernel_time_seconds != 0
            or trace.observed_query_width != 0
            or trace.observed_output_width != 0
        ):
            raise ValueError("drafter trace absence conflicts with time or widths")
    elif (
        trace.draft_kernel_time_seconds <= 0
        or trace.observed_query_width <= 0
        or trace.observed_output_width <= 0
    ):
        raise ValueError("drafter trace execution lacks positive time or widths")


def validate_spec_decode_metrics(
    metrics: SpecDecodeMetrics,
    plan: MethodPlan,
) -> SpecDecodeMetrics:
    """Validate verifier selection separately from physical drafter evidence."""
    integer_metrics = {
        "proposed_tokens": metrics.proposed_tokens,
        "accepted_tokens": metrics.accepted_tokens,
        "draft_iterations": metrics.draft_iterations,
    }
    for field_name, value in integer_metrics.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"spec_decode {field_name} must be nonnegative")
    if metrics.accepted_tokens > metrics.proposed_tokens:
        raise ValueError("accepted_tokens cannot exceed proposed_tokens")
    if any(
        type(key) is not int
        or key < 0
        or type(value) is not int
        or value <= 0
        for key, value in metrics.selected_k_histogram.items()
    ):
        raise ValueError("selected-K histogram must contain positive integer counts")

    if plan.drafter is None:
        if (
            metrics.proposed_tokens != 0
            or metrics.accepted_tokens != 0
            or metrics.draft_iterations != 0
            or metrics.selected_k_histogram
            or metrics.selected_verifier_k is not None
            or metrics.configured_draft_k is not None
            or metrics.drafter_trace is not None
        ):
            raise ValueError("baseline must contain no speculative or trace evidence")
        return metrics

    if plan.controller == "k0_diagnostic" and (
        metrics.selected_verifier_k is None
        or metrics.configured_draft_k is None
        or metrics.drafter_trace is None
    ):
        raise ValueError("K0 diagnostic evidence is incomplete")
    max_configured_k = _DRAFTER_MAX_CONFIGURED_K[plan.drafter]
    if (
        type(metrics.configured_draft_k) is not int
        or not 1 <= metrics.configured_draft_k <= max_configured_k
    ):
        drafter_name = "DFlash" if plan.drafter == "dflash" else "DSpark"
        raise ValueError(
            f"{drafter_name} configured K exceeds the s4166 checkpoint capability"
        )
    if metrics.drafter_trace is None:
        if plan.controller == "k0_diagnostic":
            raise ValueError("K0 diagnostic requires reproducible drafter trace evidence")
        raise ValueError("drafter result requires reproducible trace evidence")
    trace = metrics.drafter_trace
    _validate_trace_evidence(trace)
    if trace.observed_execution:
        expected_query_width = (
            metrics.configured_draft_k + 1
            if plan.drafter == "dflash"
            else metrics.configured_draft_k
        )
        if (
            trace.observed_query_width != expected_query_width
            or trace.observed_output_width != metrics.configured_draft_k
        ):
            raise ValueError("drafter trace physical widths conflict with configured K")

    histogram_proposals = sum(
        selected_k * count
        for selected_k, count in metrics.selected_k_histogram.items()
    )
    if metrics.proposed_tokens != histogram_proposals:
        raise ValueError(
            "proposed_tokens must equal per-sequence selected-K proposal decisions"
        )
    histogram_iterations = sum(metrics.selected_k_histogram.values())
    if metrics.draft_iterations != histogram_iterations:
        raise ValueError(
            "draft_iterations must equal selected-K proposal decision count"
        )
    if (
        plan.controller != "k0_diagnostic"
        and histogram_proposals > 0
        and not trace.observed_execution
    ):
        raise ValueError("positive speculative metrics require drafter execution")

    if plan.controller == "k0_diagnostic":
        if metrics.selected_verifier_k != 0 or metrics.configured_draft_k <= 0:
            raise ValueError("K0 diagnostic evidence is incomplete")
        if (
            metrics.proposed_tokens != 0
            or metrics.accepted_tokens != 0
        ):
            raise ValueError("K0 diagnostic verifier token counters must remain zero")
        if (
            set(metrics.selected_k_histogram) != {0}
            or metrics.selected_k_histogram[0] <= 0
        ):
            raise ValueError("K0 diagnostic selected-K histogram must contain only K0")
    elif plan.method == "fixed":
        fixed_k = plan.verifier_k
        if fixed_k is None:
            raise ValueError("fixed method plan must materialize verifier K")
        if (
            metrics.selected_verifier_k != fixed_k
            or metrics.configured_draft_k != fixed_k
        ):
            raise ValueError("selected_verifier_k does not match the fixed method plan")
        if (
            set(metrics.selected_k_histogram) != {fixed_k}
            or metrics.selected_k_histogram[fixed_k] <= 0
        ):
            raise ValueError("selected-K histogram does not match the fixed method plan")
    else:
        histogram = metrics.selected_k_histogram
        if metrics.selected_verifier_k is not None or not histogram:
            raise ValueError("DynamicSD selected-K histogram must be nonempty")
        if any(
            key > metrics.configured_draft_k or key > max_configured_k
            for key in histogram
        ):
            raise ValueError("DynamicSD selected-K exceeds configured K or capability")
    return metrics


def worker_run_id(
    provenance: RuntimeProvenance,
    plan: MethodPlan,
    attempt_index: int,
) -> str:
    """Build the stable job/worker/method/attempt correlation identity."""
    return (
        f"{provenance.slurm_job_id}:worker-{provenance.worker_index}:"
        f"{plan.key}:attempt-{attempt_index}"
    )


def _validate_run_identity_and_trace_span(result: WorkerResult) -> None:
    if type(result.attempt_index) is not int or result.attempt_index < 0:
        raise ValueError("attempt_index must be a nonnegative integer")
    if result.run_id != worker_run_id(
        result.runtime_provenance,
        result.method_plan,
        result.attempt_index,
    ):
        raise ValueError("run_id does not match job, worker, method, and attempt")
    trace = result.spec_decode.drafter_trace
    if trace is None:
        return
    if trace.run_id != result.run_id:
        raise ValueError("drafter trace run_id does not match result run_id")
    if (
        trace.capture_start_offset_seconds > 0
        or trace.capture_end_offset_seconds < result.summary.elapsed_seconds
    ):
        raise ValueError("drafter trace does not cover the full generation run")


def validate_worker_result(
    result: WorkerResult,
    *,
    contract: ExperimentContract,
    plan: MethodPlan,
    prompt_manifest_sha256: str,
) -> WorkerResult:
    """Reject identity drift, incomplete work, or unproven runtime behavior."""
    if result.schema_version != 1 or result.status != "complete":
        raise ValueError("worker result must be a complete schema-version-1 payload")
    if result.method_plan != plan:
        raise ValueError("worker result method_plan does not match the requested plan")
    if result.max_tokens != contract.max_tokens or result.max_tokens > 1_024:
        raise ValueError("result max_tokens exceeds the OSL1K contract")
    if (
        result.temperature != contract.temperature
        or result.top_p != contract.top_p
        or result.seed_policy != contract.seed_policy
    ):
        raise ValueError("result sampling settings do not match the contract")
    if result.natural_eos is not True:
        raise ValueError("result must preserve natural EOS")
    _validate_runtime_provenance(
        result.runtime_provenance,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=prompt_manifest_sha256,
    )
    expected_indices = _expected_request_indices(
        contract,
        plan,
        result.runtime_provenance.worker_index,
    )
    _validate_rows_and_summary(
        result,
        contract=contract,
        expected_indices=expected_indices,
    )
    validate_spec_decode_metrics(result.spec_decode, plan)
    _validate_run_identity_and_trace_span(result)
    return result


def _require_keys(
    payload: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field_name} fields are incomplete or unsupported")


def _mapping_field(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _list_field(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _str_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_str_field(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int_field(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer or null")
    return value


def _bool_field(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_bool_field(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean or null")
    return value


def _float_field(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _graph_from_payload(payload: Mapping[str, object]) -> CudaGraphEvidence:
    _require_keys(
        payload,
        {
            "mode",
            "target_full",
            "target_piecewise",
            "drafter_full",
            "drafter_piecewise",
            "drafter_decode_full",
        },
        "cuda_graph_evidence",
    )
    return CudaGraphEvidence(
        mode=_str_field(payload, "mode"),
        target_full=_bool_field(payload, "target_full"),
        target_piecewise=_bool_field(payload, "target_piecewise"),
        drafter_full=_optional_bool_field(payload, "drafter_full"),
        drafter_piecewise=_optional_bool_field(payload, "drafter_piecewise"),
        drafter_decode_full=_optional_bool_field(payload, "drafter_decode_full"),
    )


def _provenance_from_payload(payload: Mapping[str, object]) -> RuntimeProvenance:
    _require_keys(
        payload,
        {
            "container_path",
            "container_digest",
            "vllm_version",
            "vllm_commit",
            "target_path",
            "target_config_sha256",
            "drafter_path",
            "drafter_config_sha256",
            "prompt_manifest_sha256",
            "tensor_parallel_size",
            "data_parallel_size",
            "external_engine_count",
            "worker_index",
            "slurm_job_id",
            "cuda_graph_evidence",
        },
        "runtime_provenance",
    )
    return RuntimeProvenance(
        container_path=_str_field(payload, "container_path"),
        container_digest=_str_field(payload, "container_digest"),
        vllm_version=_str_field(payload, "vllm_version"),
        vllm_commit=_str_field(payload, "vllm_commit"),
        target_path=_str_field(payload, "target_path"),
        target_config_sha256=_str_field(payload, "target_config_sha256"),
        drafter_path=_optional_str_field(payload, "drafter_path"),
        drafter_config_sha256=_optional_str_field(
            payload,
            "drafter_config_sha256",
        ),
        prompt_manifest_sha256=_str_field(payload, "prompt_manifest_sha256"),
        tensor_parallel_size=_int_field(payload, "tensor_parallel_size"),
        data_parallel_size=_int_field(payload, "data_parallel_size"),
        external_engine_count=_int_field(payload, "external_engine_count"),
        worker_index=_int_field(payload, "worker_index"),
        slurm_job_id=_str_field(payload, "slurm_job_id"),
        cuda_graph_evidence=_graph_from_payload(
            _mapping_field(payload, "cuda_graph_evidence")
        ),
    )


def _trace_from_payload(payload: Mapping[str, object]) -> DrafterTraceEvidence:
    _require_keys(
        payload,
        {
            "run_id",
            "source_kind",
            "clock_domain",
            "artifact_uri",
            "artifact_sha256",
            "artifact_size_bytes",
            "capture_start_offset_seconds",
            "capture_end_offset_seconds",
            "capture_duration_seconds",
            "draft_kernel_count",
            "draft_kernel_time_seconds",
            "observed_query_width",
            "observed_output_width",
        },
        "drafter_trace",
    )
    return DrafterTraceEvidence(
        run_id=_str_field(payload, "run_id"),
        source_kind=_str_field(payload, "source_kind"),
        clock_domain=_str_field(payload, "clock_domain"),
        artifact_uri=_str_field(payload, "artifact_uri"),
        artifact_sha256=_str_field(payload, "artifact_sha256"),
        artifact_size_bytes=_int_field(payload, "artifact_size_bytes"),
        capture_start_offset_seconds=_float_field(
            payload,
            "capture_start_offset_seconds",
        ),
        capture_end_offset_seconds=_float_field(
            payload,
            "capture_end_offset_seconds",
        ),
        capture_duration_seconds=_float_field(
            payload,
            "capture_duration_seconds",
        ),
        draft_kernel_count=_int_field(payload, "draft_kernel_count"),
        draft_kernel_time_seconds=_float_field(
            payload,
            "draft_kernel_time_seconds",
        ),
        observed_query_width=_int_field(payload, "observed_query_width"),
        observed_output_width=_int_field(payload, "observed_output_width"),
    )


def _spec_decode_from_payload(payload: Mapping[str, object]) -> SpecDecodeMetrics:
    _require_keys(
        payload,
        {
            "proposed_tokens",
            "accepted_tokens",
            "draft_iterations",
            "acceptance_rate",
            "mean_accepted_length",
            "selected_k_histogram",
            "selected_verifier_k",
            "configured_draft_k",
            "observed_drafter_execution",
            "drafter_trace",
        },
        "spec_decode",
    )
    raw_histogram = _mapping_field(payload, "selected_k_histogram")
    histogram: dict[int, int] = {}
    for key, value in raw_histogram.items():
        if not isinstance(key, str) or not key.isdecimal() or type(value) is not int:
            raise ValueError("selected_k_histogram must map integer strings to integers")
        histogram[int(key)] = value
    raw_trace = payload.get("drafter_trace")
    if raw_trace is not None and not isinstance(raw_trace, Mapping):
        raise ValueError("drafter_trace must be an object or null")
    metrics = SpecDecodeMetrics(
        proposed_tokens=_int_field(payload, "proposed_tokens"),
        accepted_tokens=_int_field(payload, "accepted_tokens"),
        draft_iterations=_int_field(payload, "draft_iterations"),
        selected_k_histogram=histogram,
        selected_verifier_k=_optional_int_field(payload, "selected_verifier_k"),
        configured_draft_k=_optional_int_field(payload, "configured_draft_k"),
        drafter_trace=(
            None if raw_trace is None else _trace_from_payload(raw_trace)
        ),
    )
    if _optional_bool_field(
        payload,
        "observed_drafter_execution",
    ) != metrics.observed_drafter_execution:
        raise ValueError("observed_drafter_execution conflicts with drafter trace")
    if not math.isclose(
        _float_field(payload, "acceptance_rate"),
        metrics.acceptance_rate,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("acceptance_rate does not match SpecDec counters")
    if not math.isclose(
        _float_field(payload, "mean_accepted_length"),
        metrics.mean_accepted_length,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("mean_accepted_length does not match SpecDec counters")
    return metrics


def _row_from_payload(payload: Mapping[str, object]) -> RequestResult:
    _require_keys(
        payload,
        {
            "request_id",
            "global_request_index",
            "prompt_index",
            "generation_index",
            "seed",
            "max_tokens",
            "ignore_eos",
            "text",
            "token_ids",
            "output_tokens",
            "finish_reason",
            "finish_seconds",
        },
        "request row",
    )
    raw_token_ids = _list_field(payload, "token_ids")
    if any(type(value) is not int for value in raw_token_ids):
        raise ValueError("token_ids must contain integers")
    token_ids = tuple(value for value in raw_token_ids if type(value) is int)
    if _int_field(payload, "output_tokens") != len(token_ids):
        raise ValueError("row output_tokens do not match token_ids")
    return RequestResult(
        request_id=_str_field(payload, "request_id"),
        global_request_index=_int_field(payload, "global_request_index"),
        prompt_index=_int_field(payload, "prompt_index"),
        generation_index=_int_field(payload, "generation_index"),
        seed=_int_field(payload, "seed"),
        max_tokens=_int_field(payload, "max_tokens"),
        ignore_eos=_bool_field(payload, "ignore_eos"),
        text=_str_field(payload, "text"),
        token_ids=token_ids,
        finish_reason=_str_field(payload, "finish_reason"),
        finish_seconds=_float_field(payload, "finish_seconds"),
    )


def validate_result_payload(
    payload: Mapping[str, object],
    *,
    contract: ExperimentContract,
    plan: MethodPlan,
    prompt_manifest_sha256: str,
) -> WorkerResult:
    """Reconstruct and validate an untrusted JSON worker-result payload."""
    _require_keys(
        payload,
        {
            "schema_version",
            "status",
            "run_id",
            "attempt_index",
            "method_plan",
            "sampling",
            "runtime_provenance",
            "summary",
            "spec_decode",
            "rows",
        },
        "worker result",
    )
    method_payload = _mapping_field(payload, "method_plan")
    expected_method_payload = {
        "key": plan.key,
        "stage": plan.stage,
        "drafter": plan.drafter,
        "method": plan.method,
        "controller": plan.controller,
        "batch_size": plan.batch_size,
        "verifier_k": plan.verifier_k,
        "physical_block_size": plan.physical_block_size,
    }
    if method_payload != expected_method_payload:
        raise ValueError("worker result method_plan does not match the requested plan")

    sampling = _mapping_field(payload, "sampling")
    _require_keys(
        sampling,
        {"max_tokens", "temperature", "top_p", "seed_policy", "natural_eos"},
        "sampling",
    )
    summary_payload = _mapping_field(payload, "summary")
    _require_keys(
        summary_payload,
        {
            "elapsed_seconds",
            "barrier_seconds",
            "request_count",
            "output_tokens",
            "output_tokens_per_second",
        },
        "summary",
    )
    raw_rows = _list_field(payload, "rows")
    rows = tuple(
        _row_from_payload(row)
        for row in raw_rows
        if isinstance(row, Mapping)
    )
    if len(rows) != len(raw_rows):
        raise ValueError("rows must contain only objects")
    result = WorkerResult(
        schema_version=_int_field(payload, "schema_version"),
        status=_str_field(payload, "status"),
        run_id=_str_field(payload, "run_id"),
        attempt_index=_int_field(payload, "attempt_index"),
        method_plan=plan,
        max_tokens=_int_field(sampling, "max_tokens"),
        temperature=_float_field(sampling, "temperature"),
        top_p=_float_field(sampling, "top_p"),
        seed_policy=_str_field(sampling, "seed_policy"),
        natural_eos=_bool_field(sampling, "natural_eos"),
        runtime_provenance=_provenance_from_payload(
            _mapping_field(payload, "runtime_provenance")
        ),
        summary=WorkerSummary(
            elapsed_seconds=_float_field(summary_payload, "elapsed_seconds"),
            barrier_seconds=_float_field(summary_payload, "barrier_seconds"),
            request_count=_int_field(summary_payload, "request_count"),
            output_tokens=_int_field(summary_payload, "output_tokens"),
            output_tokens_per_second=_float_field(
                summary_payload,
                "output_tokens_per_second",
            ),
        ),
        spec_decode=_spec_decode_from_payload(
            _mapping_field(payload, "spec_decode")
        ),
        rows=rows,
    )
    return validate_worker_result(
        result,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=prompt_manifest_sha256,
    )


def publish_worker_result(
    path: Path,
    result: WorkerResult,
    *,
    contract: ExperimentContract,
    plan: MethodPlan,
    prompt_manifest_sha256: str,
) -> None:
    """Validate and atomically publish JSON without replacing an existing file."""
    validated = validate_worker_result(
        result,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=prompt_manifest_sha256,
    )
    temporary = path.with_name(
        f"{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(temporary, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(validated.to_payload(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
