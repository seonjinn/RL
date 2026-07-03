#!/usr/bin/env python3
"""Model synchronous RL rollout batches with vLLM offline generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

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


def tokenize_prompt(tokenizer: Any, text: str, max_prompt_tokens: int) -> list[int]:
    messages = [{"role": "user", "content": text}]
    if hasattr(tokenizer, "apply_chat_template"):
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    result = list(token_ids)
    if len(result) > max_prompt_tokens:
        result = result[-max_prompt_tokens:]
    if not result:
        raise ValueError("tokenized prompt is empty")
    return result


def load_prompt_batches(
    tokenizer: Any,
    *,
    prompt_jsonl: Path | None,
    prompt_offset: int,
    num_prompts: int,
    rollout_batches: int,
    max_prompt_tokens: int,
) -> list[list[list[int]]]:
    required = num_prompts * rollout_batches
    texts: list[str] = []
    if prompt_jsonl is not None:
        with prompt_jsonl.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if line_number < prompt_offset or not line.strip():
                    continue
                texts.append(extract_prompt_text(json.loads(line)))
                if len(texts) == required:
                    break
        if len(texts) != required:
            raise ValueError(
                f"loaded {len(texts)} prompts from {prompt_jsonl}, need {required}"
            )
    else:
        texts = [
            f"{BUILTIN_PROMPTS[index % len(BUILTIN_PROMPTS)]}\nPrompt id: {index}."
            for index in range(required)
        ]

    tokenized = [
        tokenize_prompt(tokenizer, text, max_prompt_tokens) for text in texts
    ]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--mode", choices=("baseline", "static", "dynamic"), required=True)
    parser.add_argument("--static-k", type=int, default=5)
    parser.add_argument("--dynamic-schedule", default=DEFAULT_DYNAMIC_SCHEDULE)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8448)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--engine-max-num-seqs", type=int, default=64)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--cudagraph-mode", default="PIECEWISE")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_prompts <= 0 or args.rollout_batches <= 0:
        raise ValueError("num_prompts and rollout_batches must be positive")
    request_count = args.num_prompts * args.samples_per_prompt
    dynamic_schedule = parse_dynamic_schedule(args.dynamic_schedule)
    speculative_config = build_speculative_config(
        mode=args.mode,
        draft_model=args.draft_model,
        static_k=args.static_k,
        dynamic_schedule=dynamic_schedule,
    )

    from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]

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
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "seed": args.seed,
        "disable_log_stats": False,
        "compilation_config": {"cudagraph_mode": args.cudagraph_mode},
    }
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend

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

    warmup_prompt_ids = [
        tokenize_prompt(
            tokenizer,
            f"Warm up the rollout engine with request shape {index}.",
            args.max_prompt_tokens,
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
            "attention_backend": args.attention_backend or "auto",
            "cudagraph_mode": args.cudagraph_mode,
            "num_prompts": args.num_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "requests_per_rollout_batch": request_count,
            "rollout_batches": args.rollout_batches,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "logprobs": 0,
            "seed": args.seed,
            "warmup_max_tokens": args.warmup_max_tokens,
            "prompt_jsonl": str(args.prompt_jsonl) if args.prompt_jsonl else None,
            "prompt_offset": args.prompt_offset,
            "prompt_batch_hashes": [
                prompt_batch_hash(batch) for batch in prompt_batches
            ],
        },
        "rollout_batches": rows,
        "summary": {},
    }

    def flush() -> None:
        write_json_atomic(args.output, payload)

    for batch_index, prompt_token_ids in enumerate(prompt_batches):
        requests = expand_prompt_samples(
            prompt_token_ids,
            samples_per_prompt=args.samples_per_prompt,
            seed_start=args.seed + batch_index * request_count,
        )
        prompts = [{"prompt_token_ids": ids} for ids, _ in requests]
        sampling_params = [
            SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
                logprobs=0,
            )
            for _, seed in requests
        ]
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
        output_token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        lengths = [len(token_ids) for token_ids in output_token_ids]
        output_tokens = sum(lengths)
        finish_reasons: dict[str, int] = {}
        for output in outputs:
            reason = str(output.outputs[0].finish_reason)
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
            "finish_reasons": finish_reasons,
            "output_token_hashes": [token_hash(token_ids) for token_ids in output_token_ids],
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
