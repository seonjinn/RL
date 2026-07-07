#!/usr/bin/env python3
"""Model synchronous RL rollout batches with vLLM offline generation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, NamedTuple

from benchmark import (
    DEFAULT_DYNAMIC_SCHEDULE,
    build_speculative_config,
    diff_spec_decode_counters,
    extract_prompt_text,
    parse_dynamic_schedule,
    read_spec_decode_counters,
    runtime_metadata,
    sum_spec_decode_counters,
    write_json_atomic,
)
from sync_rollout_core import RequestPlan, load_request_plan, resolve_request_plan


BUILTIN_PROMPTS = (
    "Find all real roots of x^2 - 5x + 6 = 0 and explain the steps.",
    "A rectangle has perimeter 54 and length 3 more than width. Find its area.",
    "Evaluate the sum of the first 40 positive odd integers.",
    "Prove that the square root of 2 is irrational.",
    "Solve 2^(x+1) = 32 and verify the answer.",
    "A fair die is rolled four times. Find the probability of exactly two sixes.",
    "Compute the derivative of x^3 sin(x) and simplify it.",
    "Find the minimum value of x + 9/x for positive real x.",
    "Determine the remainder when 7^100 is divided by 13.",
    "A triangle has sides 13, 14, and 15. Compute its area.",
    "Evaluate the integral of 2x/(x^2+1) from 0 to 1.",
    "How many onto functions are there from a four-element set to a two-element set?",
    "Solve the recurrence a_n = 3a_(n-1) with a_1 = 2.",
    "Find the coefficient of x^5 in (1+x)^9.",
    "Show that every integer squared is congruent to 0 or 1 modulo 4.",
    "Find the equation of the circle through (0,0), (4,0), and (0,6).",
)


class PromptRecord(NamedTuple):
    prompt_id: str
    token_ids: list[int]
    prompt_sha256: str
    source_prompt_sha256: str | None


class RolloutRequest(NamedTuple):
    prompt_id: str
    prompt_sha256: str
    source_prompt_sha256: str | None
    sample_index: int
    seed: int
    prompt_token_ids: list[int]
    max_tokens: int
    min_tokens: int
    ignore_eos: bool


def expand_prompt_samples(
    prompt_token_ids: list[list[int]],
    *,
    samples_per_prompt: int,
    seed_start: int,
) -> list[tuple[list[int], int]]:
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    requests: list[tuple[list[int], int]] = []
    seed = seed_start
    for token_ids in prompt_token_ids:
        for _ in range(samples_per_prompt):
            requests.append((list(token_ids), seed))
            seed += 1
    return requests


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def length_statistics(lengths: list[int]) -> dict[str, float | int]:
    return {
        "min": min(lengths, default=0),
        "mean": statistics.fmean(lengths) if lengths else 0.0,
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths, default=0),
    }


def tokenize_prompt(
    tokenizer: Any,
    text: str,
    max_prompt_tokens: int,
    *,
    allow_truncation: bool = True,
) -> list[int]:
    messages = [{"role": "user", "content": text}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    result = list(token_ids)
    if any(not isinstance(token_id, int) for token_id in result):
        raise TypeError("tokenizer returned non-integer prompt token IDs")
    if len(result) > max_prompt_tokens:
        if not allow_truncation:
            raise ValueError(
                f"prompt exceeds max_prompt_tokens: "
                f"prompt={len(result)} max={max_prompt_tokens}"
            )
        result = result[-max_prompt_tokens:]
    if not result:
        raise ValueError("tokenized prompt is empty")
    return result


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_row_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("id", "prompt_id", "source_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int):
            return str(value)
    return fallback


def source_prompt_hash(row: dict[str, Any]) -> str | None:
    value = row.get("prompt_sha256")
    return value if isinstance(value, str) and value else None


def load_prompt_batches(
    tokenizer: Any,
    *,
    prompt_jsonl: Path | None,
    prompt_offset: int,
    num_prompts: int,
    rollout_batches: int,
    max_prompt_tokens: int,
) -> list[list[PromptRecord]]:
    required = num_prompts * rollout_batches
    prompt_rows: list[tuple[str, str | None, str]] = []
    if prompt_jsonl is not None:
        with prompt_jsonl.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if line_number < prompt_offset or not line.strip():
                    continue
                row = json.loads(line)
                text = extract_prompt_text(row)
                prompt_rows.append(
                    (
                        prompt_row_id(row, f"jsonl-{line_number}"),
                        source_prompt_hash(row),
                        text,
                    )
                )
                if len(prompt_rows) == required:
                    break
        if len(prompt_rows) != required:
            raise ValueError(
                f"loaded {len(prompt_rows)} prompts from {prompt_jsonl}, need {required}"
            )
    else:
        prompt_rows = [
            (
                f"builtin-{index}",
                None,
                f"{BUILTIN_PROMPTS[index % len(BUILTIN_PROMPTS)]}\nPrompt id: {index}.",
            )
            for index in range(required)
        ]

    tokenized: list[PromptRecord] = []
    for prompt_id, source_hash, text in prompt_rows:
        token_ids = tokenize_prompt(
            tokenizer,
            text,
            max_prompt_tokens,
            allow_truncation=False,
        )
        tokenized.append(
            PromptRecord(
                prompt_id=prompt_id,
                prompt_sha256=token_hash(token_ids),
                source_prompt_sha256=source_hash,
                token_ids=token_ids,
            ),
        )
    return [
        tokenized[start : start + num_prompts]
        for start in range(0, required, num_prompts)
    ]


def token_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def prompt_batch_hash(prompt_token_ids: list[list[int]]) -> str:
    payload = json.dumps(prompt_token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def prompt_set_hash(prompt_batch_hashes: list[str]) -> str:
    payload = json.dumps(prompt_batch_hashes, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config_hash(model: str) -> str | None:
    return file_sha256(Path(model) / "config.json")


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_file_hash(path: Path, *, label: str) -> str:
    value = file_sha256(path)
    if value is None:
        raise ValueError(f"missing {label}: {path}")
    return value


def _view_marker_hash(model: Path) -> str:
    marker = model / ".long_context_view.json"
    if marker.is_file():
        return _required_file_hash(marker, label="long-context view marker")
    return _canonical_hash({"kind": "native_checkpoint", "path": str(model.resolve())})


def _checkpoint_identity_hash(model: Path) -> str:
    indexes = []
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = model / name
        if path.is_file():
            indexes.append({"name": name, "sha256": _required_file_hash(path, label=name)})
    return _canonical_hash(
        {
            "path": str(model.resolve()),
            "config_sha256": _required_file_hash(
                model / "config.json",
                label="model config",
            ),
            "indexes": indexes,
            "view_marker_sha256": _view_marker_hash(model),
        }
    )


def _rope_config(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"model config must be an object: {config_path}")
    return {
        key: config.get(key)
        for key in (
            "max_position_embeddings",
            "original_max_position_embeddings",
            "rope_parameters",
            "rope_scaling",
            "rope_theta",
        )
    }


def _artifact_provenance(model_value: str, *, role: str) -> dict[str, str]:
    if not model_value:
        absent_hash = _canonical_hash({"kind": "absent", "role": role})
        return {
            "config_hash": absent_hash,
            "checkpoint_hash": absent_hash,
            "view_marker_hash": absent_hash,
        }
    model = Path(model_value)
    return {
        "config_hash": _required_file_hash(
            model / "config.json",
            label=f"{role} config",
        ),
        "checkpoint_hash": _checkpoint_identity_hash(model),
        "view_marker_hash": _view_marker_hash(model),
    }


def build_execution_provenance(
    args: argparse.Namespace,
    *,
    compilation_config: dict[str, Any],
) -> dict[str, Any]:
    runtime_sha = str(args.runtime_image_sha256 or "")
    if not runtime_sha or runtime_sha.lower() in {"unknown", "none"}:
        raise ValueError("runtime_image_sha256 is required and must not be unknown")
    if args.node_count <= 0:
        raise ValueError("node_count must be positive")
    backend = str(args.distributed_executor_backend or "")
    if not backend or backend.lower() in {"unknown", "none", "auto"}:
        raise ValueError(
            "distributed_executor_backend is required and must not be unknown"
        )
    context_profile = str(args.context_profile or "")
    if not context_profile or context_profile.lower() in {"unknown", "none"}:
        raise ValueError("context_profile is required and must not be unknown")
    if not compilation_config:
        raise ValueError("compilation_config is required and must not be empty")

    model = _artifact_provenance(args.model, role="model")
    drafter = _artifact_provenance(args.draft_model, role="drafter")
    rope_payload: dict[str, Any] = {"model": _rope_config(Path(args.model))}
    if args.draft_model:
        rope_payload["drafter"] = _rope_config(Path(args.draft_model))
    else:
        rope_payload["drafter"] = {"kind": "absent"}
    topology = {
        "nodes": args.node_count,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "distributed_executor_backend": backend,
    }
    return {
        "runtime_image_sha256": runtime_sha,
        "node_count": args.node_count,
        "distributed_executor_backend": backend,
        "compilation_config": copy.deepcopy(compilation_config),
        "model_config_hash": model["config_hash"],
        "model_checkpoint_hash": model["checkpoint_hash"],
        "model_view_marker_hash": model["view_marker_hash"],
        "drafter_config_hash": drafter["config_hash"],
        "drafter_checkpoint_hash": drafter["checkpoint_hash"],
        "drafter_view_marker_hash": drafter["view_marker_hash"],
        "context_profile": context_profile,
        "rope_config_hash": _canonical_hash(rope_payload),
        "topology": topology,
    }


def prompt_tokens(batch: list[PromptRecord]) -> list[list[int]]:
    return [record.token_ids for record in batch]


def prepare_rollout_requests(
    prompt_records: list[PromptRecord],
    *,
    request_plan: RequestPlan,
    samples_per_prompt: int,
    seed_start: int,
    rollout_batch_index: int,
    max_model_len: int,
) -> list[RolloutRequest]:
    by_prompt_id = {record.prompt_id: record for record in prompt_records}
    resolved = resolve_request_plan(
        request_plan,
        prompt_ids=[record.prompt_id for record in prompt_records],
        samples_per_prompt=samples_per_prompt,
        seed_start=seed_start,
        prompt_token_lengths=[len(record.token_ids) for record in prompt_records],
        rollout_batch_index=rollout_batch_index,
        max_model_len=max_model_len,
    )
    return [
        RolloutRequest(
            prompt_id=request.prompt_id,
            prompt_sha256=by_prompt_id[request.prompt_id].prompt_sha256,
            source_prompt_sha256=by_prompt_id[request.prompt_id].source_prompt_sha256,
            sample_index=request.sample_index,
            seed=request.seed,
            prompt_token_ids=list(by_prompt_id[request.prompt_id].token_ids),
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
        )
        for request in resolved
    ]


def expand_rollout_requests(
    prompt_records: list[PromptRecord],
    *,
    samples_per_prompt: int,
    seed_start: int,
    max_tokens: int,
) -> list[RolloutRequest]:
    requests: list[RolloutRequest] = []
    seed = seed_start
    for record in prompt_records:
        for sample_index in range(samples_per_prompt):
            requests.append(
                RolloutRequest(
                    prompt_id=record.prompt_id,
                    prompt_sha256=record.prompt_sha256,
                    source_prompt_sha256=record.source_prompt_sha256,
                    sample_index=sample_index,
                    seed=seed,
                    prompt_token_ids=list(record.token_ids),
                    max_tokens=max_tokens,
                    min_tokens=0,
                    ignore_eos=False,
                )
            )
            seed += 1
    return requests


def build_sampling_params(
    sampling_params_cls: Any,
    requests: list[RolloutRequest],
    *,
    temperature: float,
    top_p: float,
) -> list[Any]:
    return [
        sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
            seed=request.seed,
            logprobs=0,
        )
        for request in requests
    ]


def request_provenance(request: RolloutRequest) -> dict[str, Any]:
    return {
        "prompt_id": request.prompt_id,
        "prompt_sha256": request.prompt_sha256,
        "source_prompt_sha256": request.source_prompt_sha256,
        "sample_index": request.sample_index,
        "seed": request.seed,
        "prompt_tokens": len(request.prompt_token_ids),
        "max_tokens": request.max_tokens,
        "min_tokens": request.min_tokens,
        "ignore_eos": request.ignore_eos,
    }


def bucket_statistics(
    requests: list[RolloutRequest],
    output_token_ids: list[list[int]],
) -> list[dict[str, Any]]:
    lengths_by_cap: dict[int, list[int]] = {}
    for request, token_ids in zip(requests, output_token_ids, strict=True):
        lengths_by_cap.setdefault(request.max_tokens, []).append(len(token_ids))
    return [
        {
            "max_tokens": max_tokens,
            "request_count": len(lengths),
            "output_tokens": sum(lengths),
            "completion_length": length_statistics(lengths),
        }
        for max_tokens, lengths in sorted(lengths_by_cap.items())
    ]


def exact_output_work(
    requests: list[RolloutRequest],
    output_token_ids: list[list[int]],
) -> dict[str, list[int] | list[bool]]:
    planned = [request.max_tokens for request in requests]
    actual = [len(token_ids) for token_ids in output_token_ids]
    forced = [
        request.ignore_eos or request.min_tokens == request.max_tokens
        for request in requests
    ]
    for index, (request, actual_tokens, is_forced) in enumerate(
        zip(requests, actual, forced, strict=True)
    ):
        if is_forced and actual_tokens != request.max_tokens:
            raise ValueError(
                f"forced output length mismatch at request {index}: "
                f"prompt_id={request.prompt_id} sample_index={request.sample_index} "
                f"planned={request.max_tokens} actual={actual_tokens}"
            )
    return {
        "planned_output_tokens": planned,
        "actual_output_tokens": actual,
        "forced_output_mask": forced,
    }


def first_candidate(output: Any) -> Any:
    return output.outputs[0]


def candidate_token_ids(output: Any) -> list[int]:
    return list(first_candidate(output).token_ids)


def write_response_jsonl(
    path: Path,
    *,
    batch_index: int,
    requests: list[RolloutRequest],
    outputs: list[Any],
    append: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as stream:
        for request, output in zip(requests, outputs, strict=True):
            candidate = first_candidate(output)
            token_ids = candidate_token_ids(output)
            row = {
                "batch_index": batch_index,
                "prompt_id": request.prompt_id,
                "prompt_sha256": request.prompt_sha256,
                "source_prompt_sha256": request.source_prompt_sha256,
                "sample_index": request.sample_index,
                "seed": request.seed,
                "max_tokens": request.max_tokens,
                "min_tokens": request.min_tokens,
                "ignore_eos": request.ignore_eos,
                "finish_reason": str(candidate.finish_reason),
                "output_tokens": len(token_ids),
                "output_token_hash": token_hash(token_ids),
                "text": str(getattr(candidate, "text", "")),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--max-model-len", type=int, default=8448)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--engine-max-num-seqs", type=int, default=64)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--moe-backend", default="")
    parser.add_argument("--distributed-executor-backend", required=True)
    parser.add_argument("--distributed-timeout-seconds", type=int)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--context-profile", required=True)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--model-loader-num-threads", type=int, default=0)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE")
    parser.add_argument("--disable-fuse-allreduce-rms", action="store_true")
    parser.add_argument("--mamba-ssm-cache-dtype", default="")
    parser.add_argument("--mamba-backend", default="")
    parser.add_argument(
        "--enable-mamba-cache-stochastic-rounding", action="store_true"
    )
    parser.add_argument("--mamba-cache-philox-rounds", type=int)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--rollout-batches", type=int, default=3)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--request-plan", type=Path)
    parser.add_argument("--resolved-request-plan-output", type=Path)
    parser.add_argument("--response-output", type=Path)
    parser.add_argument("--runtime-image-sha256", default="")
    parser.add_argument("--source-recipe", default="")
    parser.add_argument("--global-num-prompts", type=int)
    parser.add_argument("--global-generation-replicas", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_prompts <= 0 or args.rollout_batches <= 0:
        raise ValueError("num_prompts and rollout_batches must be positive")
    request_plan = load_request_plan(args.request_plan) if args.request_plan else None
    if request_plan is not None and args.max_model_len != request_plan.max_model_len:
        raise ValueError(
            f"--max-model-len must match request plan max_model_len: "
            f"got {args.max_model_len}, expected {request_plan.max_model_len}"
        )
    request_count = args.num_prompts * args.samples_per_prompt
    dynamic_schedule = parse_dynamic_schedule(args.dynamic_schedule)
    speculative_config = build_speculative_config(
        mode=args.mode,
        draft_model=args.draft_model,
        static_k=args.static_k,
        dynamic_schedule=dynamic_schedule,
    )

    from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]

    compilation_config: dict[str, Any] = {
        "cudagraph_mode": args.cudagraph_mode,
    }
    if args.disable_fuse_allreduce_rms:
        compilation_config["pass_config"] = {"fuse_allreduce_rms": False}
    execution_provenance = build_execution_provenance(
        args,
        compilation_config=compilation_config,
    )

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": min(args.engine_max_num_seqs, request_count),
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "enable_expert_parallel": args.enable_expert_parallel,
        "seed": args.seed,
        "disable_log_stats": False,
        "compilation_config": compilation_config,
    }
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = copy.deepcopy(speculative_config)
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.moe_backend:
        llm_kwargs["kernel_config"] = {"moe_backend": args.moe_backend}
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = (
            args.distributed_executor_backend
        )
    if args.distributed_timeout_seconds is not None:
        llm_kwargs["distributed_timeout_seconds"] = args.distributed_timeout_seconds
    if args.model_loader_num_threads > 0:
        llm_kwargs["model_loader_extra_config"] = {
            "enable_multithread_load": True,
            "num_threads": args.model_loader_num_threads,
        }
    if args.mamba_ssm_cache_dtype:
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.mamba_backend:
        llm_kwargs["mamba_backend"] = args.mamba_backend
    if args.enable_mamba_cache_stochastic_rounding:
        llm_kwargs["enable_mamba_cache_stochastic_rounding"] = True
    if args.mamba_cache_philox_rounds is not None:
        llm_kwargs["mamba_cache_philox_rounds"] = args.mamba_cache_philox_rounds

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompt_batches = load_prompt_batches(
        tokenizer,
        prompt_jsonl=args.prompt_jsonl,
        prompt_offset=args.prompt_offset,
        num_prompts=args.num_prompts,
        rollout_batches=args.rollout_batches,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    prompt_batch_hashes = [
        prompt_batch_hash(prompt_tokens(batch)) for batch in prompt_batches
    ]

    warmup_prompt_ids = [
        tokenize_prompt(
            tokenizer,
            f"Warm up the rollout engine with request shape {index}.",
            args.max_prompt_tokens,
            allow_truncation=False,
        )
        for index in range(args.num_prompts)
    ]
    warmup_requests = expand_prompt_samples(
        warmup_prompt_ids,
        samples_per_prompt=args.samples_per_prompt,
        seed_start=args.seed - request_count,
    )
    warmup_prompts = [
        {"prompt_token_ids": token_ids} for token_ids, _ in warmup_requests
    ]
    warmup_params = [
        SamplingParams(
            max_tokens=args.warmup_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=seed,
            logprobs=0,
        )
        for _, seed in warmup_requests
    ]
    llm.generate(warmup_prompts, warmup_params, use_tqdm=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    resolved_plan_output = args.resolved_request_plan_output
    if resolved_plan_output is None and request_plan is not None:
        resolved_plan_output = args.output.parent / "resolved_request_plan.json"
    resolved_plan_payload: dict[str, Any] | None = None
    if request_plan is not None:
        resolved_plan_payload = {
            "schema_version": 1,
            "request_plan": {
                "name": request_plan.name,
                "path": str(args.request_plan),
                "plan_hash": request_plan.plan_hash,
                "max_model_len": request_plan.max_model_len,
                "buckets": [
                    {
                        "max_tokens": bucket.max_tokens,
                        "min_tokens": bucket.min_tokens,
                        "weight": bucket.weight,
                        "ignore_eos": bucket.ignore_eos,
                    }
                    for bucket in request_plan.buckets
                ],
            },
            "rollout_batches": [],
        }

    def flush_resolved_plan() -> None:
        if resolved_plan_output is not None and resolved_plan_payload is not None:
            write_json_atomic(resolved_plan_output, resolved_plan_payload)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "runtime": runtime_metadata(),
        "config": {
            "tag": args.tag,
            "scenario": "synchronous_rl_rollout",
            "sync_barrier": "LLM.generate_return",
            "model": args.model,
            "draft_model": args.draft_model,
            **execution_provenance,
            "mode": args.mode,
            "speculative_config": speculative_config,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "total_gpus": total_gpus,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": min(args.engine_max_num_seqs, request_count),
            "global_requests_per_rollout_batch": request_count,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enable_expert_parallel": args.enable_expert_parallel,
            "attention_backend": args.attention_backend or "auto",
            "moe_backend": args.moe_backend or "auto",
            "distributed_timeout_seconds": args.distributed_timeout_seconds,
            "model_loader_extra_config": llm_kwargs.get(
                "model_loader_extra_config"
            ),
            "cudagraph_mode": args.cudagraph_mode,
            "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype or "auto",
            "mamba_backend": args.mamba_backend or "auto",
            "enable_mamba_cache_stochastic_rounding": (
                args.enable_mamba_cache_stochastic_rounding
            ),
            "mamba_cache_philox_rounds": args.mamba_cache_philox_rounds,
            "num_prompts": args.num_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "requests_per_rollout_batch": request_count,
            "rollout_batches": args.rollout_batches,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "request_plan": str(args.request_plan) if args.request_plan else None,
            "request_plan_name": request_plan.name if request_plan else None,
            "request_plan_hash": request_plan.plan_hash if request_plan else None,
            "resolved_request_plan_output": (
                str(resolved_plan_output) if resolved_plan_output else None
            ),
            "response_output": str(args.response_output) if args.response_output else None,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "logprobs": 0,
            "seed": args.seed,
            "warmup_max_tokens": args.warmup_max_tokens,
            "prompt_jsonl": str(args.prompt_jsonl) if args.prompt_jsonl else None,
            "prompt_offset": args.prompt_offset,
            "source_recipe": args.source_recipe or None,
            "global_num_prompts": args.global_num_prompts,
            "global_generation_replicas": args.global_generation_replicas,
            "prompt_batch_hashes": prompt_batch_hashes,
            "prompt_set_hash": prompt_set_hash(prompt_batch_hashes),
        },
        "rollout_batches": rows,
        "summary": {},
    }

    def flush() -> None:
        write_json_atomic(args.output, payload)

    for batch_index, prompt_token_ids in enumerate(prompt_batches):
        if request_plan is None:
            rollout_requests = expand_rollout_requests(
                prompt_token_ids,
                samples_per_prompt=args.samples_per_prompt,
                seed_start=args.seed + batch_index * request_count,
                max_tokens=args.max_new_tokens,
            )
        else:
            rollout_requests = prepare_rollout_requests(
                prompt_token_ids,
                request_plan=request_plan,
                samples_per_prompt=args.samples_per_prompt,
                seed_start=args.seed,
                rollout_batch_index=batch_index,
                max_model_len=args.max_model_len,
            )
        prompts = [
            {"prompt_token_ids": request.prompt_token_ids}
            for request in rollout_requests
        ]
        sampling_params = build_sampling_params(
            SamplingParams,
            rollout_requests,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if resolved_plan_payload is not None:
            resolved_plan_payload["rollout_batches"].append(
                {
                    "batch_index": batch_index,
                    "requests": [
                        request_provenance(request) for request in rollout_requests
                    ],
                }
            )
            flush_resolved_plan()
        before = read_spec_decode_counters(llm)
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        rollout_time_s = time.perf_counter() - started
        metrics = diff_spec_decode_counters(read_spec_decode_counters(llm), before)
        if speculative_config is not None and (
            not metrics.get("metrics_available") or not metrics.get("active")
        ):
            raise RuntimeError(
                f"SpecDec counters are unavailable or inactive for mode={args.mode}, "
                f"rollout_batch={batch_index}"
            )
        output_token_ids = [candidate_token_ids(output) for output in outputs]
        output_work = exact_output_work(rollout_requests, output_token_ids)
        if args.response_output is not None:
            write_response_jsonl(
                args.response_output,
                batch_index=batch_index,
                requests=rollout_requests,
                outputs=outputs,
                append=batch_index > 0,
            )
        lengths = [len(token_ids) for token_ids in output_token_ids]
        output_tokens = sum(lengths)
        finish_reasons: dict[str, int] = {}
        for output in outputs:
            reason = str(first_candidate(output).finish_reason)
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        row = {
            "batch_index": batch_index,
            "request_count": request_count,
            "rollout_time_s": rollout_time_s,
            "output_tokens": output_tokens,
            "output_tok_s": output_tokens / rollout_time_s,
            "output_tok_s_per_gpu": output_tokens / rollout_time_s / total_gpus,
            "requests_per_s": request_count / rollout_time_s,
            "completion_length": length_statistics(lengths),
            "bucket_statistics": bucket_statistics(
                rollout_requests,
                output_token_ids,
            ),
            "finish_reasons": finish_reasons,
            "output_token_hashes": [token_hash(token_ids) for token_ids in output_token_ids],
            "planned_output_tokens": output_work["planned_output_tokens"],
            "actual_output_tokens": output_work["actual_output_tokens"],
            "forced_output_mask": output_work["forced_output_mask"],
            "requests": [
                request_provenance(request) for request in rollout_requests
            ],
            "spec_decode_metrics": metrics,
        }
        rows.append(row)
        flush()
        metrics_text = ""
        if metrics:
            metrics_text = (
                f" acceptance={metrics['acceptance_rate']:.2%}"
                f" mean_accept_len={metrics['mean_acceptance_length']:.2f}"
            )
        print(
            f"rollout_batch={batch_index} time={rollout_time_s:.3f}s "
            f"tok/s/GPU={row['output_tok_s_per_gpu']:.2f} "
            f"length_p99={row['completion_length']['p99']:.1f}{metrics_text}",
            flush=True,
        )

    total_time_s = sum(row["rollout_time_s"] for row in rows)
    total_output_tokens = sum(row["output_tokens"] for row in rows)
    all_metrics = sum_spec_decode_counters(
        [row["spec_decode_metrics"] for row in rows]
    )
    payload["summary"] = {
        "total_rollout_time_s": total_time_s,
        "offline_generation_makespan_s": total_time_s,
        "mean_rollout_time_s": statistics.fmean(
            row["rollout_time_s"] for row in rows
        ),
        "median_rollout_time_s": statistics.median(
            row["rollout_time_s"] for row in rows
        ),
        "total_output_tokens": total_output_tokens,
        "output_tok_s": total_output_tokens / total_time_s,
        "output_tok_s_per_gpu": total_output_tokens / total_time_s / total_gpus,
        "requests_per_s": request_count * len(rows) / total_time_s,
        "spec_decode_metrics": all_metrics,
    }
    payload["status"] = "complete"
    flush()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
