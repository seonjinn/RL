"""One-engine execution for the Q30 synchronous DynamicSD benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .contract import ExperimentContract, MethodPlan
from .results import (
    DrafterTraceEvidence,
    RequestResult,
    RuntimeProvenance,
    SpecDecodeMetrics,
    WorkerResult,
    WorkerSummary,
    validate_worker_result,
    worker_run_id,
)


@dataclass(frozen=True, slots=True)
class PromptManifest:
    """A canonical prompt list sealed by its SHA256 digest."""

    prompts: tuple[str, ...]
    sha256: str


def _prompt_manifest_digest(prompts: tuple[str, ...]) -> str:
    encoded = json.dumps(
        {"schema_version": 1, "prompts": prompts},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_prompt_manifest(prompts: Sequence[str]) -> PromptManifest:
    """Copy and seal a prompt sequence for deterministic partitioning."""
    frozen_prompts = tuple(prompts)
    return PromptManifest(
        prompts=frozen_prompts,
        sha256=_prompt_manifest_digest(frozen_prompts),
    )


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Engine-independent generation input carrying the sampling contract."""

    request_id: str
    global_request_index: int
    prompt_index: int
    generation_index: int
    prompt: str
    seed: int
    max_tokens: int
    temperature: float
    top_p: float
    ignore_eos: bool


@dataclass(frozen=True, slots=True)
class EngineCompletion:
    """Engine-independent concrete completion from a local adapter."""

    request_id: str
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    finish_seconds: float


@dataclass(frozen=True, slots=True)
class EngineRun:
    """Concrete completions plus adapter-supplied metric evidence."""

    completions: tuple[EngineCompletion, ...]
    metric_evidence: Mapping[str, object]


class LocalEngine(Protocol):
    """Boundary implemented by the live vLLM adapter or a unit-test fake."""

    def generate(self, requests: Sequence[GenerationRequest]) -> EngineRun:
        """Generate every supplied request and return normalized completions."""
        ...


def build_worker_requests(
    contract: ExperimentContract,
    plan: MethodPlan,
    prompt_manifest: PromptManifest,
    worker_index: int,
) -> tuple[GenerationRequest, ...]:
    """Partition exact global work and materialize deterministic requests."""
    if (
        len(prompt_manifest.prompts) != contract.prompt_count
        or any(
            not isinstance(prompt, str) or not prompt
            for prompt in prompt_manifest.prompts
        )
        or prompt_manifest.sha256
        != _prompt_manifest_digest(prompt_manifest.prompts)
    ):
        raise ValueError("prompt manifest is incomplete or its seal is invalid")
    if type(worker_index) is not int or not 0 <= worker_index < contract.engine_count:
        raise ValueError("worker_index is outside the external engine topology")
    if plan.stage == "calibration" and worker_index != 0:
        raise ValueError("calibration runs use worker_index zero")
    request_count = (
        plan.batch_size
        if plan.stage == "calibration"
        else contract.requests_per_engine
    )
    if request_count is None:
        raise ValueError("request count is unavailable for this method plan")
    first_global_index = (
        0
        if plan.stage == "calibration"
        else worker_index * contract.requests_per_engine
    )
    requests = []
    for offset in range(request_count):
        global_index = first_global_index + offset
        prompt_index, generation_index = divmod(
            global_index,
            contract.generations_per_prompt,
        )
        requests.append(
            GenerationRequest(
                request_id=f"request-{global_index:04d}",
                global_request_index=global_index,
                prompt_index=prompt_index,
                generation_index=generation_index,
                prompt=prompt_manifest.prompts[prompt_index],
                seed=contract.seed_for_request(global_index),
                max_tokens=contract.max_tokens,
                temperature=contract.temperature,
                top_p=contract.top_p,
                ignore_eos=contract.ignore_eos,
            )
        )
    return tuple(requests)


def extract_spec_decode_metrics(
    evidence: Mapping[str, object],
) -> SpecDecodeMetrics:
    """Normalize adapter evidence without importing vLLM in this module."""
    histogram_payload = evidence.get("selected_k_histogram", {})
    if not isinstance(histogram_payload, Mapping):
        raise ValueError("selected_k_histogram must be a mapping")
    histogram: dict[int, int] = {}
    for key, value in histogram_payload.items():
        if type(value) is not int:
            raise ValueError("selected_k_histogram values must be integers")
        if type(key) is int:
            normalized_key = key
        elif isinstance(key, str) and key.isdecimal():
            normalized_key = int(key)
        else:
            raise ValueError("selected_k_histogram keys must be nonnegative integers")
        histogram[normalized_key] = value
    trace_payload = evidence.get("drafter_trace")
    if trace_payload is not None and not isinstance(trace_payload, Mapping):
        raise ValueError("drafter_trace must be a mapping or null")
    return SpecDecodeMetrics(
        proposed_tokens=_required_int(evidence.get("proposed_tokens", 0)),
        accepted_tokens=_required_int(evidence.get("accepted_tokens", 0)),
        draft_iterations=_required_int(evidence.get("draft_iterations", 0)),
        selected_k_histogram=histogram,
        selected_verifier_k=_optional_int(evidence.get("selected_verifier_k")),
        configured_draft_k=_optional_int(evidence.get("configured_draft_k")),
        drafter_trace=(
            None
            if trace_payload is None
            else _extract_drafter_trace(trace_payload)
        ),
    )


def _extract_drafter_trace(
    evidence: Mapping[str, object],
) -> DrafterTraceEvidence:
    return DrafterTraceEvidence(
        run_id=_required_str(evidence.get("run_id")),
        source_kind=_required_str(evidence.get("source_kind")),
        clock_domain=_required_str(evidence.get("clock_domain")),
        artifact_uri=_required_str(evidence.get("artifact_uri")),
        artifact_sha256=_required_str(evidence.get("artifact_sha256")),
        artifact_size_bytes=_required_int(evidence.get("artifact_size_bytes")),
        capture_start_offset_seconds=_required_float(
            evidence.get("capture_start_offset_seconds")
        ),
        capture_end_offset_seconds=_required_float(
            evidence.get("capture_end_offset_seconds")
        ),
        capture_duration_seconds=_required_float(
            evidence.get("capture_duration_seconds")
        ),
        draft_kernel_count=_required_int(evidence.get("draft_kernel_count")),
        draft_kernel_time_seconds=_required_float(
            evidence.get("draft_kernel_time_seconds")
        ),
        observed_query_width=_required_int(evidence.get("observed_query_width")),
        observed_output_width=_required_int(
            evidence.get("observed_output_width")
        ),
    )


def _required_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("SpecDec counter evidence must be integer-valued")
    return value


def _required_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("drafter trace timing must be numeric")
    return float(value)


def _required_str(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("drafter trace text fields must be strings")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _required_int(value)


def run_one_engine(
    *,
    contract: ExperimentContract,
    plan: MethodPlan,
    prompt_manifest: PromptManifest,
    worker_index: int,
    engine: LocalEngine,
    runtime_provenance: RuntimeProvenance,
    metric_extractor: Callable[
        [Mapping[str, object]], SpecDecodeMetrics
    ] = extract_spec_decode_metrics,
    clock: Callable[[], float] = time.perf_counter,
    attempt_index: int = 0,
) -> WorkerResult:
    """Run one local engine and retain exact timing and completion evidence."""
    requests = build_worker_requests(
        contract,
        plan,
        prompt_manifest,
        worker_index,
    )
    start = clock()
    engine_run = engine.generate(requests)
    end = clock()
    elapsed = end - start
    if elapsed <= 0:
        raise ValueError("generation elapsed time must be positive")
    if tuple(
        completion.request_id for completion in engine_run.completions
    ) != tuple(request.request_id for request in requests):
        raise ValueError("engine completions do not contain the exact request work")
    requests_by_id = {request.request_id: request for request in requests}
    rows = tuple(
        RequestResult(
            request_id=completion.request_id,
            global_request_index=requests_by_id[
                completion.request_id
            ].global_request_index,
            prompt_index=requests_by_id[completion.request_id].prompt_index,
            generation_index=requests_by_id[
                completion.request_id
            ].generation_index,
            seed=requests_by_id[completion.request_id].seed,
            max_tokens=requests_by_id[completion.request_id].max_tokens,
            ignore_eos=requests_by_id[completion.request_id].ignore_eos,
            text=completion.text,
            token_ids=completion.token_ids,
            finish_reason=completion.finish_reason,
            finish_seconds=completion.finish_seconds,
        )
        for completion in engine_run.completions
    )
    output_tokens = sum(len(row.token_ids) for row in rows)
    summary = WorkerSummary(
        elapsed_seconds=elapsed,
        barrier_seconds=elapsed,
        request_count=len(rows),
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens / elapsed,
    )
    result = WorkerResult(
        schema_version=1,
        status="complete",
        run_id=worker_run_id(runtime_provenance, plan, attempt_index),
        attempt_index=attempt_index,
        method_plan=plan,
        max_tokens=contract.max_tokens,
        temperature=contract.temperature,
        top_p=contract.top_p,
        seed_policy=contract.seed_policy,
        natural_eos=not contract.ignore_eos,
        runtime_provenance=runtime_provenance,
        summary=summary,
        spec_decode=metric_extractor(engine_run.metric_evidence),
        rows=rows,
    )
    return validate_worker_result(
        result,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=prompt_manifest.sha256,
    )
