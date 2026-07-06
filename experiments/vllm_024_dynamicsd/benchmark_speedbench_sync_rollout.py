#!/usr/bin/env python3
"""Run SPEED-Bench official and AsyncLLM synchronous-rollout overlay cohorts."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from benchmark import (
    DEFAULT_DYNAMIC_SCHEDULE,
    build_speculative_config,
    diff_spec_decode_counters,
    parse_dynamic_schedule,
    read_spec_decode_counters,
    runtime_metadata,
    sum_spec_decode_counters,
    write_json_atomic,
)
from speedbench_dataset import SpeedBenchRecord, select_sync_overlay_rows
from sync_rollout_core import RequestPlan, load_request_plan, resolve_request_plan


@dataclass(frozen=True, slots=True)
class OverlayPrompt:
    prompt_id: str
    prompt_token_ids: list[int]
    prompt_sha256: str
    source_prompt_sha256: str | None
    category: str
    dataset_config: str
    turn_count: int
    multiturn: bool

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(frozen=True, slots=True)
class OverlayRequest:
    request_id: str
    prompt_id: str
    prompt_sha256: str
    source_prompt_sha256: str | None
    category: str
    sample_index: int
    seed: int
    prompt_token_ids: list[int]
    max_tokens: int
    min_tokens: int
    ignore_eos: bool


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    request: OverlayRequest
    output: Any
    output_token_ids: list[int]
    ttft_s: float
    finished_at_s: float
    completion_time_s: float
    finish_reason: str


def token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _string_field(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string field {field_name!r}")
    return value


def _optional_string_field(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected optional string field {field_name!r}")
    return value


def _bool_field(row: Mapping[str, Any], field_name: str, default: bool) -> bool:
    value = row.get(field_name, default)
    if type(value) is not bool:
        raise ValueError(f"expected boolean field {field_name!r}")
    return value


def _turn_count(row: Mapping[str, Any]) -> int:
    turns = row.get("turns")
    if turns is None:
        return 1
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("turns must be a sequence of strings")
    return len(turns)


def _prompt_token_ids(row: Mapping[str, Any]) -> list[int]:
    for field_name in ("prompt_token_ids", "input_ids", "token_ids"):
        value = row.get(field_name)
        if value is None:
            continue
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or not value
            or any(type(token_id) is not int for token_id in value)
        ):
            raise ValueError(f"{field_name} must be a non-empty integer sequence")
        return list(value)
    raise ValueError("prepared SPEED-Bench row must include preserved token IDs")


def overlay_prompt_from_prepared_row(
    row: Mapping[str, Any],
    *,
    max_prompt_tokens: int | None = None,
) -> OverlayPrompt:
    del max_prompt_tokens
    token_ids = _prompt_token_ids(row)
    return OverlayPrompt(
        prompt_id=_string_field(row, "question_id"),
        prompt_token_ids=token_ids,
        prompt_sha256=token_hash(token_ids),
        source_prompt_sha256=_optional_string_field(row, "canonical_hash"),
        category=_string_field(row, "category"),
        dataset_config=_string_field(row, "dataset_config"),
        turn_count=_turn_count(row),
        multiturn=_bool_field(row, "multiturn", _turn_count(row) > 1),
    )


def _record_to_overlay_prompt(
    record: SpeedBenchRecord,
    token_ids: Sequence[int],
) -> OverlayPrompt:
    return overlay_prompt_from_prepared_row(
        {
            **asdict(record),
            "prompt_token_ids": list(token_ids),
        }
    )


def build_overlay_prompt_batches(
    records: Sequence[SpeedBenchRecord],
    *,
    prepared_token_ids_by_question_id: Mapping[str, Sequence[int]],
    seed: int,
) -> tuple[tuple[OverlayPrompt, ...], ...]:
    selected = select_sync_overlay_rows(records, seed=seed)
    return tuple(
        tuple(
            _record_to_overlay_prompt(
                record,
                prepared_token_ids_by_question_id[record.question_id],
            )
            for record in batch
        )
        for batch in selected
    )


def _prompt_by_id(prompts: Sequence[OverlayPrompt]) -> dict[str, OverlayPrompt]:
    return {prompt.prompt_id: prompt for prompt in prompts}


def prepare_overlay_requests(
    prompts: Sequence[OverlayPrompt],
    *,
    request_plan: RequestPlan,
    samples_per_prompt: int,
    seed_start: int,
    rollout_batch_index: int,
    max_model_len: int,
) -> list[OverlayRequest]:
    by_prompt = _prompt_by_id(prompts)
    resolved = resolve_request_plan(
        request_plan,
        prompt_ids=[prompt.prompt_id for prompt in prompts],
        samples_per_prompt=samples_per_prompt,
        seed_start=seed_start,
        prompt_token_lengths=[len(prompt.prompt_token_ids) for prompt in prompts],
        rollout_batch_index=rollout_batch_index,
        max_model_len=max_model_len,
    )
    requests: list[OverlayRequest] = []
    for item in resolved:
        prompt = by_prompt[item.prompt_id]
        requests.append(
            OverlayRequest(
                request_id=(
                    f"speedbench-{rollout_batch_index}-"
                    f"{item.prompt_id}-{item.sample_index}"
                ),
                prompt_id=item.prompt_id,
                prompt_sha256=prompt.prompt_sha256,
                source_prompt_sha256=prompt.source_prompt_sha256,
                category=prompt.category,
                sample_index=item.sample_index,
                seed=item.seed,
                prompt_token_ids=list(prompt.prompt_token_ids),
                max_tokens=item.max_tokens,
                min_tokens=item.min_tokens,
                ignore_eos=item.ignore_eos,
            )
        )
    return requests


def validate_request_plan_exact_work(
    requests: Sequence[OverlayRequest],
    prompts: Sequence[OverlayPrompt],
    *,
    samples_per_prompt: int,
) -> dict[str, int]:
    expected_prompt_ids = [prompt.prompt_id for prompt in prompts]
    expected = {
        (prompt_id, sample_index)
        for prompt_id in expected_prompt_ids
        for sample_index in range(samples_per_prompt)
    }
    actual = {
        (request.prompt_id, request.sample_index)
        for request in requests
    }
    if actual != expected or len(requests) != len(expected):
        raise ValueError(
            "request-plan exact-work mismatch: "
            f"expected={len(expected)} actual={len(requests)}"
        )
    return {
        "expected_requests": len(expected),
        "actual_requests": len(requests),
        "unique_prompts": len(expected_prompt_ids),
    }


def build_prompt_shape_warmup_requests(
    prompts: Sequence[OverlayPrompt],
    *,
    samples_per_prompt: int,
    seed_start: int,
    max_tokens: int,
) -> list[OverlayRequest]:
    requests: list[OverlayRequest] = []
    seed = seed_start
    for prompt in prompts:
        for sample_index in range(samples_per_prompt):
            requests.append(
                OverlayRequest(
                    request_id=f"warmup-{prompt.prompt_id}-{sample_index}",
                    prompt_id=prompt.prompt_id,
                    prompt_sha256=prompt.prompt_sha256,
                    source_prompt_sha256=prompt.source_prompt_sha256,
                    category=prompt.category,
                    sample_index=sample_index,
                    seed=seed,
                    prompt_token_ids=list(prompt.prompt_token_ids),
                    max_tokens=max_tokens,
                    min_tokens=0,
                    ignore_eos=False,
                )
            )
            seed += 1
    return requests


def first_candidate(output: Any) -> Any:
    return output.outputs[0]


def candidate_token_ids(output: Any) -> list[int]:
    return list(first_candidate(output).token_ids)


def finish_reason(output: Any) -> str:
    return str(getattr(first_candidate(output), "finish_reason", "unknown"))


async def run_one_request_async(
    engine: Any,
    request: OverlayRequest,
    sampling_params: Any,
    *,
    batch_started_at_s: float,
    clock: Callable[[], float],
) -> CompletedRequest:
    first_output_at_s: float | None = None
    final_output: Any | None = None
    final_token_ids: list[int] = []
    final_time_s = batch_started_at_s
    async for output in engine.generate(
        prompt={"prompt_token_ids": request.prompt_token_ids},
        sampling_params=sampling_params,
        request_id=request.request_id,
    ):
        now_s = clock()
        if first_output_at_s is None:
            first_output_at_s = now_s
        final_output = output
        final_token_ids = candidate_token_ids(output)
        final_time_s = now_s
        if bool(getattr(output, "finished", False)):
            break
    if first_output_at_s is None or final_output is None:
        raise RuntimeError(f"AsyncLLM request produced no output: {request.request_id}")
    return CompletedRequest(
        request=request,
        output=final_output,
        output_token_ids=final_token_ids,
        ttft_s=round(first_output_at_s - batch_started_at_s, 6),
        finished_at_s=final_time_s,
        completion_time_s=round(final_time_s - batch_started_at_s, 6),
        finish_reason=finish_reason(final_output),
    )


async def run_overlay_batch_async(
    engine: Any,
    requests: Sequence[OverlayRequest],
    *,
    sampling_params_by_request: Mapping[str, Any],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    batch_started_at_s = clock()
    completed = await asyncio.gather(
        *(
            run_one_request_async(
                engine,
                request,
                sampling_params_by_request[request.request_id],
                batch_started_at_s=batch_started_at_s,
                clock=clock,
            )
            for request in requests
        )
    )
    barrier_finished_at_s = clock()
    max_request_finished_at_s = max(
        (item.finished_at_s for item in completed),
        default=batch_started_at_s,
    )
    barrier_finished_at_s = max(barrier_finished_at_s, max_request_finished_at_s)
    output_token_ids = [item.output_token_ids for item in completed]
    return {
        "sync_barrier": "AsyncLLM.gather",
        "request_count": len(completed),
        "batch_started_at_s": batch_started_at_s,
        "barrier_finished_at_s": barrier_finished_at_s,
        "barrier_time_s": round(barrier_finished_at_s - batch_started_at_s, 6),
        "ttft_s": [item.ttft_s for item in completed],
        "completion_time_s": [item.completion_time_s for item in completed],
        "output_token_ids": output_token_ids,
        "output_token_hashes": [token_hash(token_ids) for token_ids in output_token_ids],
        "prompt_token_ids": [
            list(item.request.prompt_token_ids) for item in completed
        ],
        "finish_reasons": {
            reason: [item.finish_reason for item in completed].count(reason)
            for reason in sorted({item.finish_reason for item in completed})
        },
        "requests": [request_provenance(item.request) for item in completed],
    }


def output_position_acceptance_windows(
    *,
    output_token_ids: Sequence[Sequence[int]],
    accepted_tokens_per_pos: Sequence[float],
    window_size: int,
) -> list[dict[str, float | int]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    max_length = max((len(tokens) for tokens in output_token_ids), default=0)
    windows: list[dict[str, float | int]] = []
    for start in range(0, max_length, window_size):
        end = min(start + window_size, max_length) - 1
        contributor_count = sum(
            1
            for position in range(start, end + 1)
            for tokens in output_token_ids
            if len(tokens) > position
        )
        accepted = sum(
            float(accepted_tokens_per_pos[position])
            if position < len(accepted_tokens_per_pos)
            else 0.0
            for position in range(start, end + 1)
        )
        windows.append(
            {
                "start_pos": start,
                "end_pos": end,
                "contributor_count": contributor_count,
                "accepted_tokens": accepted,
                "acceptance_rate": (
                    round(accepted / contributor_count, 6)
                    if contributor_count
                    else 0.0
                ),
            }
        )
    return windows


def _schedule_k_for_concurrency(
    concurrency: int,
    schedule: Sequence[Sequence[int]],
) -> int:
    current_k = 0
    for start, end, k_value in schedule:
        if concurrency < start:
            return current_k
        current_k = k_value
        if start <= concurrency <= end:
            return k_value
    return current_k


def active_concurrency_k_tier_reachability(
    concurrencies: Iterable[int],
    dynamic_schedule: str,
) -> list[dict[str, int | bool]]:
    schedule = parse_dynamic_schedule(dynamic_schedule)
    return [
        {
            "concurrency": int(concurrency),
            "k": _schedule_k_for_concurrency(int(concurrency), schedule),
            "reachable": True,
        }
        for concurrency in concurrencies
    ]


def require_k_tier_reachability(
    concurrencies: Iterable[int],
    dynamic_schedule: str,
) -> None:
    schedule = parse_dynamic_schedule(dynamic_schedule)
    required = {int(item[2]) for item in schedule if int(item[2]) > 0}
    reached = {
        int(item["k"])
        for item in active_concurrency_k_tier_reachability(
            concurrencies,
            dynamic_schedule,
        )
        if int(item["k"]) > 0
    }
    missing = sorted(required - reached)
    if missing:
        raise ValueError(f"K-tier not reached by active concurrency plan: {missing}")


def build_official_speedbench_command(
    *,
    model: str,
    modelopt_root: Path,
    prepared_root: Path,
    dataset_config: str,
    output_dir: Path,
    variant: str,
    tensor_parallel_size: int,
    max_model_len: int,
    draft_model: str = "",
    static_k: int = 0,
    dynamic_schedule: str = "",
) -> list[str]:
    command = [
        "python3",
        str(modelopt_root / "examples/specdec_bench/benchmark.py"),
        "--dataset",
        "speed",
        "--config",
        dataset_config,
        "--prepared-root",
        str(prepared_root),
        "--model",
        model,
        "--output-dir",
        str(output_dir),
        "--variant",
        variant,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
    ]
    if draft_model:
        command.extend(["--draft-model", draft_model])
    if static_k:
        command.extend(["--num-speculative-tokens", str(static_k)])
    if dynamic_schedule:
        command.extend(["--dynamic-schedule", dynamic_schedule])
    return command


def request_provenance(request: OverlayRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "prompt_id": request.prompt_id,
        "prompt_sha256": request.prompt_sha256,
        "source_prompt_sha256": request.source_prompt_sha256,
        "category": request.category,
        "sample_index": request.sample_index,
        "seed": request.seed,
        "prompt_tokens": len(request.prompt_token_ids),
        "max_tokens": request.max_tokens,
        "min_tokens": request.min_tokens,
        "ignore_eos": request.ignore_eos,
    }


def length_statistics(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    ordered = sorted(lengths)

    def percentile(quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def build_sampling_params(
    sampling_params_cls: Any,
    requests: Sequence[OverlayRequest],
    *,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    return {
        request.request_id: sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
            seed=request.seed,
            logprobs=0,
        )
        for request in requests
    }


def load_overlay_prompt_jsonl(path: Path) -> tuple[OverlayPrompt, ...]:
    prompts: list[OverlayPrompt] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                prompts.append(overlay_prompt_from_prepared_row(json.loads(line)))
    return tuple(prompts)


def chunk_prompts(
    prompts: Sequence[OverlayPrompt],
    *,
    batch_size: int,
    batches: int,
) -> list[list[OverlayPrompt]]:
    required = batch_size * batches
    if len(prompts) < required:
        raise ValueError(f"need {required} overlay prompts, found {len(prompts)}")
    return [
        list(prompts[start : start + batch_size])
        for start in range(0, required, batch_size)
    ]


async def run_overlay(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import SamplingParams  # pyright: ignore[reportMissingImports]
    from vllm.engine.arg_utils import AsyncEngineArgs  # pyright: ignore[reportMissingImports]
    from vllm.v1.engine.async_llm import AsyncLLM  # pyright: ignore[reportMissingImports]

    prompts = load_overlay_prompt_jsonl(args.prepared_jsonl)
    prompt_batches = chunk_prompts(
        prompts,
        batch_size=args.active_concurrency,
        batches=args.rollout_batches,
    )
    request_plan = load_request_plan(args.request_plan)
    dynamic_schedule = parse_dynamic_schedule(args.dynamic_schedule)
    speculative_config = build_speculative_config(
        mode=args.mode,
        draft_model=args.draft_model,
        static_k=args.static_k,
        dynamic_schedule=dynamic_schedule,
    )
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.active_concurrency,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        speculative_config=copy.deepcopy(speculative_config),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    rows: list[dict[str, Any]] = []
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    try:
        warmup_requests = build_prompt_shape_warmup_requests(
            prompt_batches[0],
            samples_per_prompt=args.samples_per_prompt,
            seed_start=args.seed - (args.active_concurrency * args.samples_per_prompt),
            max_tokens=args.warmup_max_tokens,
        )
        warmup_params = build_sampling_params(
            SamplingParams,
            warmup_requests,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        await run_overlay_batch_async(
            engine,
            warmup_requests,
            sampling_params_by_request=warmup_params,
        )
        for batch_index, prompt_batch in enumerate(prompt_batches):
            requests = prepare_overlay_requests(
                prompt_batch,
                request_plan=request_plan,
                samples_per_prompt=args.samples_per_prompt,
                seed_start=args.seed,
                rollout_batch_index=batch_index,
                max_model_len=args.max_model_len,
            )
            if args.request_plan_exact_work:
                validate_request_plan_exact_work(
                    requests,
                    prompt_batch,
                    samples_per_prompt=args.samples_per_prompt,
                )
            before = read_spec_decode_counters(engine)
            params = build_sampling_params(
                SamplingParams,
                requests,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = await run_overlay_batch_async(
                engine,
                requests,
                sampling_params_by_request=params,
            )
            metrics = diff_spec_decode_counters(read_spec_decode_counters(engine), before)
            output_lengths = [len(tokens) for tokens in row["output_token_ids"]]
            row.update(
                {
                    "batch_index": batch_index,
                    "output_tokens": sum(output_lengths),
                    "output_tok_s": sum(output_lengths) / row["barrier_time_s"],
                    "output_tok_s_per_gpu": (
                        sum(output_lengths) / row["barrier_time_s"] / total_gpus
                    ),
                    "completion_length": length_statistics(output_lengths),
                    "spec_decode_metrics": metrics,
                    "acceptance_windows": output_position_acceptance_windows(
                        output_token_ids=row["output_token_ids"],
                        accepted_tokens_per_pos=metrics.get(
                            "num_accepted_tokens_per_pos", []
                        ),
                        window_size=args.acceptance_window_size,
                    ),
                }
            )
            rows.append(row)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    total_time_s = sum(float(row["barrier_time_s"]) for row in rows)
    total_output_tokens = sum(int(row["output_tokens"]) for row in rows)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "runtime": runtime_metadata(),
        "config": {
            "cohort": "overlay",
            "scenario": "speedbench_sync_overlay",
            "sync_barrier": "AsyncLLM.gather",
            "mode": args.mode,
            "model": args.model,
            "draft_model": args.draft_model,
            "speculative_config": speculative_config,
            "request_plan": str(args.request_plan),
            "request_plan_hash": request_plan.plan_hash,
            "prepared_jsonl": str(args.prepared_jsonl),
            "active_concurrency": args.active_concurrency,
            "samples_per_prompt": args.samples_per_prompt,
            "rollout_batches": args.rollout_batches,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "runtime_image_sha256": args.runtime_image_sha256 or None,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "cudagraph_mode": args.cudagraph_mode,
        },
        "rollout_batches": rows,
        "summary": {
            "total_rollout_time_s": total_time_s,
            "total_output_tokens": total_output_tokens,
            "output_tok_s": total_output_tokens / total_time_s if total_time_s else 0.0,
            "output_tok_s_per_gpu": (
                total_output_tokens / total_time_s / total_gpus if total_time_s else 0.0
            ),
            "spec_decode_metrics": sum_spec_decode_counters(
                [row["spec_decode_metrics"] for row in rows]
            ),
        },
    }
    write_json_atomic(args.output, payload)
    return payload


def run_official(args: argparse.Namespace) -> int:
    command = build_official_speedbench_command(
        model=args.model,
        modelopt_root=args.modelopt_root,
        prepared_root=args.prepared_root,
        dataset_config=args.dataset_config,
        output_dir=args.output.parent,
        variant=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        draft_model=args.draft_model,
        static_k=args.static_k,
        dynamic_schedule=args.dynamic_schedule if args.mode == "dynamic" else "",
    )
    if args.print_official_command:
        print(json.dumps(command))
        return 0
    return subprocess.run(command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("official", "overlay"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", default="")
    parser.add_argument(
        "--mode",
        choices=("baseline", "static", "dynamic", "mtp_static", "mtp_dynamic"),
        required=True,
    )
    parser.add_argument("--static-k", type=int, default=5)
    parser.add_argument("--dynamic-schedule", default=DEFAULT_DYNAMIC_SCHEDULE)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--active-concurrency", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--rollout-batches", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--acceptance-window-size", type=int, default=16)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE")
    parser.add_argument("--prepared-jsonl", type=Path)
    parser.add_argument("--request-plan", type=Path)
    parser.add_argument("--request-plan-exact-work", action="store_true")
    parser.add_argument("--runtime-image-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modelopt-root", type=Path, default=Path("/workspace/modelopt"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/workspace/speedbench/prepared/speed"))
    parser.add_argument("--dataset-config", default="throughput_1k")
    parser.add_argument("--print-official-command", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cohort == "official":
        raise SystemExit(run_official(args))
    if args.prepared_jsonl is None:
        raise ValueError("--prepared-jsonl is required for overlay cohort")
    if args.request_plan is None:
        raise ValueError("--request-plan is required for overlay cohort")
    asyncio.run(run_overlay(args))


if __name__ == "__main__":
    main()
