#!/usr/bin/env python3
"""Runtime-only vLLM 0.28 adapter for Q30 offline generation.

The importable adapter deliberately has no vLLM dependency. The executable
imports vLLM only after its arguments and evidence inputs have been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmark import (
    EngineCompletion,
    EngineRun,
    GenerationRequest,
    PromptManifest,
    build_worker_requests,
    seal_prompt_manifest,
)
from .contract import ExperimentContract, MethodPlan


class EvidenceUnavailableError(RuntimeError):
    """Raised rather than synthesizing evidence vLLM does not expose."""


def _extract_prompt(row: Mapping[str, object]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "assistant":
                break
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content)
        if parts:
            return "\n".join(parts)
    for key in ("prompt", "question", "problem", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("prompt row does not contain a usable user prompt")


def load_real_prompt_manifest(path: Path) -> tuple[PromptManifest, str]:
    """Load and seal exactly the first 64 real prompts and hash the source."""
    source_bytes = path.read_bytes()
    prompts: list[str] = []
    for line_number, raw_line in enumerate(source_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prompt JSON on line {line_number}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"prompt line {line_number} must be a JSON object")
        prompts.append(_extract_prompt(payload))
        if len(prompts) == ExperimentContract().prompt_count:
            break
    if len(prompts) != ExperimentContract().prompt_count:
        raise ValueError("prompt source does not contain exactly 64 usable prompts")
    return seal_prompt_manifest(prompts), hashlib.sha256(source_bytes).hexdigest()


def _validate_schedule(schedule: Sequence[Sequence[int]]) -> list[list[int]]:
    normalized = [list(row) for row in schedule]
    if not normalized or any(len(row) != 3 for row in normalized):
        raise ValueError("DynamicSD schedule must contain start/end/K triples")
    previous_end = 0
    for index, row in enumerate(normalized):
        start, end, verifier_k = row
        if any(type(value) is not int for value in row):
            raise ValueError("DynamicSD schedule values must be integers")
        if start != previous_end + 1 or end < start or not 0 <= verifier_k <= 8:
            raise ValueError("DynamicSD schedule must be contiguous and valid")
        if index and verifier_k > normalized[index - 1][2]:
            raise ValueError("DynamicSD K must be non-increasing")
        previous_end = end
    if normalized[0][0] != 1 or normalized[-1][1] != 128:
        raise ValueError("DynamicSD schedule must cover BS1 through BS128")
    return normalized


def build_speculative_config(
    plan: MethodPlan,
    schedule: Sequence[Sequence[int]] | None = None,
    *,
    drafter_path: str | None = None,
) -> dict[str, object] | None:
    """Build the exact vLLM speculative config for one method plan."""
    if plan.drafter is None:
        return None
    contract = ExperimentContract()
    model_path = drafter_path or contract.drafter_paths[plan.drafter]
    table: list[list[int]] | None = None
    if plan.method == "dynamic":
        if schedule is None:
            raise ValueError("DynamicSD requires an explicit calibrated schedule")
        table = _validate_schedule(schedule)
        configured_k = max(row[2] for row in table)
        if configured_k <= 0:
            configured_k = 7 if plan.drafter == "dflash" else 8
    elif plan.method == "adaptive":
        if schedule is not None:
            raise ValueError("DSpark adaptive verification must not mix with DynamicSD")
        configured_k = 8
    elif plan.verifier_k == 0:
        configured_k = 7 if plan.drafter == "dflash" else 8
    elif plan.verifier_k is not None:
        configured_k = plan.verifier_k
    else:
        raise ValueError("fixed method is missing its calibrated K")
    config: dict[str, object] = {
        "method": plan.drafter,
        "model": model_path,
        "num_speculative_tokens": configured_k,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN",
        "max_model_len": 4096,
    }
    if plan.method == "dynamic":
        if table is None:
            raise AssertionError("validated DynamicSD schedule missing")
        config["num_speculative_tokens_per_batch_size"] = table
    if plan.controller == "k0_diagnostic":
        config["num_speculative_tokens_per_batch_size"] = [[1, 128, 0]]
    if plan.method == "adaptive":
        config["enable_adaptive_verification"] = True
    return config


def _metric_value(metrics: Sequence[object], name: str) -> int | None:
    values = [
        int(getattr(metric, "value"))
        for metric in metrics
        if getattr(metric, "name", None) == name
    ]
    return sum(values) if values else None


class VllmOfflineEngine:
    """Adapt offline ``LLM.generate`` outputs to the Task-3 engine boundary."""

    def __init__(
        self,
        llm: Any,
        *,
        sampling_params_factory: Callable[..., object],
        monotonic: Callable[[], float] = time.monotonic,
        plan: MethodPlan,
        trace_evidence: Mapping[str, object] | None = None,
    ) -> None:
        self._llm = llm
        self._sampling_params_factory = sampling_params_factory
        self._monotonic = monotonic
        self._plan = plan
        self._trace_evidence = trace_evidence

    def _metrics(self) -> list[object]:
        metrics = self._llm.get_metrics()
        return [] if metrics is None else list(metrics)

    def normalize_metric_evidence(
        self,
        *,
        before: Sequence[object],
        after: Sequence[object],
    ) -> dict[str, object]:
        if self._plan.drafter is None:
            return {
                "proposed_tokens": 0,
                "accepted_tokens": 0,
                "draft_iterations": 0,
                "selected_k_histogram": {},
                "selected_verifier_k": None,
                "configured_draft_k": None,
                "drafter_trace": None,
            }
        if self._plan.method in {"dynamic", "adaptive"}:
            raise EvidenceUnavailableError(
                "vLLM 0.28 offline metrics do not expose an exact selected-K histogram"
            )
        names = {
            "drafts": "vllm:spec_decode_num_drafts",
            "proposed": "vllm:spec_decode_num_draft_tokens",
            "accepted": "vllm:spec_decode_num_accepted_tokens",
        }
        diffs: dict[str, int] = {}
        for key, name in names.items():
            current = _metric_value(after, name)
            initial = _metric_value(before, name)
            if current is None or initial is None:
                raise EvidenceUnavailableError(f"missing required vLLM metric {name}")
            diffs[key] = current - initial
        selected_k = self._plan.verifier_k
        if selected_k is None:
            raise EvidenceUnavailableError("fixed plan lacks selected verifier K")
        if diffs["proposed"] != selected_k * diffs["drafts"]:
            raise EvidenceUnavailableError(
                "aggregate counters cannot prove fixed-K work"
            )
        configured_k = 7 if self._plan.drafter == "dflash" else 8
        if selected_k > 0:
            configured_k = selected_k
        return {
            "proposed_tokens": diffs["proposed"],
            "accepted_tokens": diffs["accepted"],
            "draft_iterations": diffs["drafts"],
            "selected_k_histogram": {str(selected_k): diffs["drafts"]},
            "selected_verifier_k": selected_k,
            "configured_draft_k": configured_k,
            "drafter_trace": self._trace_evidence,
        }

    def generate_raw(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[tuple[EngineCompletion, ...], list[object], list[object]]:
        """Run generation while retaining the raw before/after metric snapshots."""
        prompts = [request.prompt for request in requests]
        sampling = [
            self._sampling_params_factory(
                n=1,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                seed=request.seed,
                ignore_eos=request.ignore_eos,
            )
            for request in requests
        ]
        before = self._metrics()
        anchor = self._monotonic()
        outputs = self._llm.generate(prompts, sampling, use_tqdm=False)
        after = self._metrics()
        if len(outputs) != len(requests):
            raise ValueError("vLLM returned an incomplete output set")
        completions: list[EngineCompletion] = []
        for request, output in zip(requests, outputs, strict=True):
            candidates = getattr(output, "outputs", None)
            if not isinstance(candidates, list) or len(candidates) != 1:
                raise ValueError("each explicit request must return exactly one output")
            candidate = candidates[0]
            metrics = getattr(output, "metrics", None)
            last_token_ts = getattr(metrics, "last_token_ts", None)
            if not isinstance(last_token_ts, (int, float)) or isinstance(
                last_token_ts, bool
            ):
                raise EvidenceUnavailableError(
                    "RequestOutput.metrics.last_token_ts missing"
                )
            finish_seconds = float(last_token_ts) - anchor
            if not math.isfinite(finish_seconds) or finish_seconds < 0:
                raise ValueError("vLLM request finish timing is invalid")
            completions.append(
                EngineCompletion(
                    request_id=request.request_id,
                    text=str(getattr(candidate, "text", "")),
                    token_ids=tuple(int(token) for token in candidate.token_ids),
                    finish_reason=str(candidate.finish_reason),
                    finish_seconds=finish_seconds,
                )
            )
        return tuple(completions), before, after

    def generate(self, requests: Sequence[GenerationRequest]) -> EngineRun:
        completions, before, after = self.generate_raw(requests)
        return EngineRun(
            completions=completions,
            metric_evidence=self.normalize_metric_evidence(before=before, after=after),
        )


def _metric_snapshot(metrics: Sequence[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        name = getattr(metric, "name", None)
        value = getattr(metric, "value", None)
        if isinstance(name, str) and isinstance(value, (int, float)):
            rows.append({"name": name, "value": value})
    return rows


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _plan_from_json(raw: str) -> MethodPlan:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("method plan JSON must contain an object")
    return MethodPlan(**payload)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsupported-receipt", type=Path)
    parser.add_argument("--reason", default="")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--external-engine-count", type=int)
    parser.add_argument("--requests-per-engine", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--speculative-config-json")
    parser.add_argument("--plan-json")
    parser.add_argument("--target-path")
    parser.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--cuda-graph-mode")
    parser.add_argument("--moe-backend")
    parser.add_argument("--evidence-status")
    parser.add_argument("--runtime-drafter-path", default="")
    parser.add_argument("--runtime-drafter-config-sha256", default="")
    parser.add_argument("--source-drafter-config-sha256", default="")
    parser.add_argument("--dtype")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-model-len", type=int)
    parsed = parser.parse_args()
    required = (
        parsed.plan_json,
        parsed.target_path,
        parsed.prompt_jsonl,
        parsed.result_dir,
        parsed.worker_index,
    )
    if any(value is None for value in required):
        parser.error(
            "live execution requires plan, target, prompts, result, and worker"
        )
    if parsed.unsupported_receipt is None:
        parsed.unsupported_receipt = parsed.result_dir / "unsupported-receipt.json"
    contract = ExperimentContract()
    plan = _plan_from_json(parsed.plan_json)
    expected_scalars = {
        "target_path": (parsed.target_path, contract.target_path),
        "max_tokens": (parsed.max_tokens, contract.max_tokens),
        "temperature": (parsed.temperature, contract.temperature),
        "top_p": (parsed.top_p, contract.top_p),
        "data_parallel_size": (
            parsed.data_parallel_size,
            contract.data_parallel_size,
        ),
        "cuda_graph_mode": (parsed.cuda_graph_mode, contract.cuda_graph_mode),
        "moe_backend": (parsed.moe_backend, "flashinfer_trtllm"),
        "dtype": (parsed.dtype, "bfloat16"),
        "gpu_memory_utilization": (parsed.gpu_memory_utilization, 0.9),
        "max_num_batched_tokens": (parsed.max_num_batched_tokens, 32_768),
        "max_model_len": (parsed.max_model_len, 4_096),
    }
    for name, (actual, expected) in expected_scalars.items():
        if actual != expected:
            raise ValueError(f"{name} must be {expected!r}")
    actual_request_count = (
        plan.batch_size if plan.stage == "calibration" else contract.requests_per_engine
    )
    if parsed.requests_per_engine != actual_request_count:
        raise ValueError(f"requests_per_engine must be {actual_request_count!r}")
    if not parsed.disable_prefix_caching or not parsed.enable_chunked_prefill:
        raise ValueError("prefix caching must be disabled and chunked prefill enabled")
    expected_evidence_status = (
        "baseline_no_speculation"
        if plan.drafter is None
        else "aggregate_fixed_k_counters_only"
        if plan.method == "fixed"
        and plan.verifier_k is not None
        and plan.verifier_k > 0
        else "selected_k_and_physical_trace_unavailable"
    )
    if parsed.evidence_status != expected_evidence_status:
        raise ValueError("evidence_status is inconsistent with the method plan")
    expected_engine_counts = {1} if plan.stage == "calibration" else {1, 16}
    if parsed.external_engine_count not in expected_engine_counts:
        raise ValueError("external_engine_count is inconsistent with the method stage")
    prompt_manifest, source_sha = load_real_prompt_manifest(parsed.prompt_jsonl)
    requests = build_worker_requests(
        contract, plan, prompt_manifest, parsed.worker_index
    )

    from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]

    speculative = (
        None
        if parsed.speculative_config_json is None
        else json.loads(parsed.speculative_config_json)
    )
    if plan.drafter is None:
        if speculative is not None:
            raise ValueError("target-only execution must not enable speculation")
    else:
        if not isinstance(speculative, dict):
            raise ValueError("drafter execution requires speculative config JSON")
        model = speculative.get("model")
        if not isinstance(model, str):
            raise ValueError("speculative model path is missing")
        raw_schedule = speculative.get("num_speculative_tokens_per_batch_size")
        schedule = raw_schedule if plan.method == "dynamic" else None
        if speculative != build_speculative_config(
            plan,
            schedule,
            drafter_path=model,
        ):
            raise ValueError("speculative config does not match the method plan")
    if plan.drafter is None:
        if any(
            (
                parsed.runtime_drafter_path,
                parsed.runtime_drafter_config_sha256,
                parsed.source_drafter_config_sha256,
            )
        ):
            raise ValueError("target-only execution must not record a drafter")
    elif not all(
        (
            parsed.runtime_drafter_path,
            parsed.runtime_drafter_config_sha256,
            parsed.source_drafter_config_sha256,
        )
    ):
        raise ValueError("drafter path and config hashes are required")
    llm_kwargs: dict[str, object] = {
        "model": parsed.target_path,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": parsed.dtype,
        "gpu_memory_utilization": parsed.gpu_memory_utilization,
        "max_model_len": parsed.max_model_len,
        "max_num_seqs": max(1, len(requests)),
        "max_num_batched_tokens": parsed.max_num_batched_tokens,
        "enable_prefix_caching": not parsed.disable_prefix_caching,
        "enable_chunked_prefill": parsed.enable_chunked_prefill,
        "seed": contract.base_seed,
        "disable_log_stats": False,
        "compilation_config": {"cudagraph_mode": parsed.cuda_graph_mode},
        "kernel_config": {"moe_backend": parsed.moe_backend},
    }
    if speculative is not None:
        llm_kwargs["speculative_config"] = speculative
    engine = VllmOfflineEngine(
        LLM(**llm_kwargs),
        sampling_params_factory=SamplingParams,
        plan=plan,
    )
    started = time.perf_counter()
    completions, before, after = engine.generate_raw(requests)
    elapsed = time.perf_counter() - started
    unresolved = expected_evidence_status == "selected_k_and_physical_trace_unavailable"
    aggregate_counter_evidence: Mapping[str, object] | None = None
    if expected_evidence_status == "aggregate_fixed_k_counters_only":
        try:
            normalized = engine.normalize_metric_evidence(
                before=before,
                after=after,
            )
        except EvidenceUnavailableError as exc:
            aggregate_counter_evidence = {
                "status": "unavailable",
                "reason": str(exc),
                "physical_trace_validated": False,
                "cuda_graph_validated": False,
            }
        else:
            aggregate_counter_evidence = {
                **normalized,
                "status": "validated_aggregate_only",
                "physical_trace_validated": False,
                "cuda_graph_validated": False,
            }
    runtime_knobs: dict[str, object] = {
        "dtype": parsed.dtype,
        "gpu_memory_utilization": parsed.gpu_memory_utilization,
        "max_num_batched_tokens": parsed.max_num_batched_tokens,
        "enable_prefix_caching": not parsed.disable_prefix_caching,
        "enable_chunked_prefill": parsed.enable_chunked_prefill,
        "max_model_len": parsed.max_model_len,
    }
    _atomic_json(
        parsed.result_dir / "runtime-provenance.json",
        {
            "schema_version": 1,
            "worker_index": parsed.worker_index,
            "runtime_drafter_path": parsed.runtime_drafter_path or None,
            "runtime_drafter_config_sha256": (
                parsed.runtime_drafter_config_sha256 or None
            ),
            "source_drafter_config_sha256": (
                parsed.source_drafter_config_sha256 or None
            ),
            "runtime_knobs": runtime_knobs,
        },
    )
    raw_path = parsed.result_dir / (
        "unvalidated_raw.json" if unresolved else "complete_raw.json"
    )
    _atomic_json(
        raw_path,
        {
            "schema_version": 1,
            "status": "unvalidated_raw" if unresolved else "complete_raw",
            "promotion_allowed": False,
            "evidence_status": expected_evidence_status,
            "method_plan": json.loads(parsed.plan_json),
            "worker_index": parsed.worker_index,
            "prompt_source_sha256": source_sha,
            "prompt_manifest_sha256": prompt_manifest.sha256,
            "elapsed_seconds": elapsed,
            "request_count": len(completions),
            "output_tokens": sum(len(row.token_ids) for row in completions),
            "runtime_drafter_path": parsed.runtime_drafter_path or None,
            "runtime_drafter_config_sha256": (
                parsed.runtime_drafter_config_sha256 or None
            ),
            "source_drafter_config_sha256": (
                parsed.source_drafter_config_sha256 or None
            ),
            "runtime_knobs": runtime_knobs,
            "metrics_before": _metric_snapshot(before),
            "metrics_after": _metric_snapshot(after),
            "aggregate_counter_evidence": aggregate_counter_evidence,
            "rows": [
                {
                    "request_id": row.request_id,
                    "text": row.text,
                    "token_ids": row.token_ids,
                    "finish_reason": row.finish_reason,
                    "finish_seconds": row.finish_seconds,
                }
                for row in completions
            ],
        },
    )
    receipt_path = (
        parsed.unsupported_receipt
        if unresolved
        else parsed.result_dir / "evidence-receipt.json"
    )
    _atomic_json(
        receipt_path,
        {
            "schema_version": 1,
            "status": "unsupported" if unresolved else "complete_raw",
            "evidence_status": expected_evidence_status,
            "reason": (
                parsed.reason
                if unresolved
                else "strict Task-3 promotion not attempted by live runner"
            ),
            "raw_artifact": str(raw_path),
            "aggregate_counter_evidence": aggregate_counter_evidence,
            "evidence_fabricated": False,
            "promotion_allowed": False,
        },
    )
    return 2 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(_main())
