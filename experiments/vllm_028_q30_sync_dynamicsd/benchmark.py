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
    RequestResult,
    RuntimeProvenance,
    SpecDecodeMetrics,
    WorkerResult,
    WorkerSummary,
    validate_worker_result,
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
    finished_at_seconds: float


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
    return SpecDecodeMetrics(
        proposed_tokens=_required_int(evidence.get("proposed_tokens", 0)),
        accepted_tokens=_required_int(evidence.get("accepted_tokens", 0)),
        draft_iterations=_required_int(evidence.get("draft_iterations", 0)),
        selected_k_histogram=histogram,
        selected_verifier_k=_optional_int(evidence.get("selected_verifier_k")),
        configured_draft_width=_optional_int(
            evidence.get("configured_draft_width")
        ),
        physical_draft_width=_optional_int(evidence.get("physical_draft_width")),
        observed_drafter_execution=_optional_bool(
            evidence.get("observed_drafter_execution")
        ),
        drafter_execution_evidence_source=_optional_str(
            evidence.get("drafter_execution_evidence_source")
        ),
        drafter_execution_count=_optional_int(
            evidence.get("drafter_execution_count")
        ),
    )


def _required_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("SpecDec counter evidence must be integer-valued")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _required_int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError("drafter execution observation must be boolean")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("drafter execution evidence source must be a string")
    return value


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
            finish_seconds=completion.finished_at_seconds - start,
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
