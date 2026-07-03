#!/usr/bin/env python3
"""Benchmark static and dynamic speculative decoding with vLLM 0.24."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Literal


Mode = Literal["baseline", "static", "dynamic"]
DEFAULT_DYNAMIC_SCHEDULE = "1:16:5,17:32:4,33:64:3,65:128:1,129:512:0"


def parse_dynamic_schedule(value: str) -> list[list[int]]:
    """Parse non-overlapping inclusive ranges in ``start:end:k`` form."""
    rows: list[list[int]] = []
    previous_end = 0
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            start, end, k = (int(part) for part in entry.split(":"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid DynamicSD entry {entry!r}; expected start:end:k"
            ) from exc
        if not rows and start != 1:
            raise ValueError(
                f"the first DynamicSD range must start at BS=1, got {start}"
            )
        if rows and start <= previous_end:
            raise ValueError(
                f"DynamicSD ranges must not overlap; previous end={previous_end}, "
                f"got start={start}"
            )
        if end < start:
            raise ValueError(f"DynamicSD range end must be >= start: {entry!r}")
        if k < 0:
            raise ValueError(f"DynamicSD K must be >= 0: {entry!r}")
        rows.append([start, end, k])
        previous_end = end
    if not rows:
        raise ValueError("DynamicSD schedule must not be empty")
    return rows


def build_speculative_config(
    *,
    mode: Mode,
    draft_model: str,
    static_k: int,
    dynamic_schedule: list[list[int]],
) -> dict[str, Any] | None:
    if mode == "baseline":
        return None
    global_k = static_k
    if mode == "dynamic":
        global_k = max(row[2] for row in dynamic_schedule)
    if global_k <= 0:
        raise ValueError("the global speculative-token count must be > 0")
    config: dict[str, Any] = {
        "method": "eagle3",
        "model": draft_model,
        "num_speculative_tokens": global_k,
        "draft_tensor_parallel_size": 1,
    }
    if mode == "dynamic":
        config["num_speculative_tokens_per_batch_size"] = dynamic_schedule
    return config


def dynamic_k_for_batch_size(schedule: list[list[int]], batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    current_k = schedule[0][2]
    for start, end, k in schedule:
        if batch_size < start:
            return current_k
        current_k = k
        if batch_size <= end:
            return current_k
    return current_k


def extract_prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
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
    raise ValueError(f"could not extract prompt from keys={sorted(row)}")


def fit_token_ids(token_ids: list[int], length: int) -> list[int]:
    if length <= 0:
        raise ValueError("ISL must be positive")
    if not token_ids:
        raise ValueError("tokenizer returned an empty prompt")
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def load_prompt_token_ids(
    tokenizer: Any,
    *,
    count: int,
    isl: int,
    prompt_jsonl: Path | None,
    prompt_offset: int,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    if prompt_jsonl is not None:
        with prompt_jsonl.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if line_number < prompt_offset or not line.strip():
                    continue
                text = extract_prompt_text(json.loads(line))
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                prompts.append(fit_token_ids(list(token_ids), isl))
                if len(prompts) == count:
                    break
        if len(prompts) != count:
            raise ValueError(
                f"loaded {len(prompts)} prompts from {prompt_jsonl}, need {count}"
            )
        return prompts

    seed_text = (
        "Solve the problem carefully, show the reasoning, and return the final "
        "answer in a concise form. "
    )
    seed_ids = list(tokenizer.encode(seed_text, add_special_tokens=False))
    base = fit_token_ids(seed_ids, isl)
    return [list(base) for _ in range(count)]


def get_metrics_snapshot(llm: Any) -> list[Any]:
    try:
        return list(llm.get_metrics())
    except Exception:
        pass
    try:
        from vllm.v1.metrics.reader import (  # pyright: ignore[reportMissingImports]
            get_metrics_snapshot,
        )

        return list(get_metrics_snapshot())
    except Exception:
        return []


def read_spec_decode_counters(llm: Any) -> dict[str, Any]:
    counters: dict[str, Any] = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
        "num_accepted_tokens_per_pos": [],
    }
    matched = False
    names = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
    }
    for metric in get_metrics_snapshot(llm):
        raw_name = str(getattr(metric, "name", ""))
        name = raw_name.removesuffix("_total")
        if name in names:
            counters[names[name]] += float(getattr(metric, "value", 0.0))
            matched = True
            continue
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = list(getattr(metric, "values", []) or [])
            if not values and hasattr(metric, "value"):
                values = [getattr(metric, "value", 0.0)]
            current = counters["num_accepted_tokens_per_pos"]
            if len(current) < len(values):
                current.extend([0.0] * (len(values) - len(current)))
            for index, value in enumerate(values):
                current[index] += float(value)
            matched = True
    return counters if matched else {}


def diff_spec_decode_counters(
    after: dict[str, Any], before: dict[str, Any]
) -> dict[str, Any]:
    if not after:
        return {}
    result: dict[str, Any] = {}
    for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        result[key] = max(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 0.0)
    after_pos = list(after.get("num_accepted_tokens_per_pos", []))
    before_pos = list(before.get("num_accepted_tokens_per_pos", []))
    result["num_accepted_tokens_per_pos"] = [
        max(
            float(after_pos[index] if index < len(after_pos) else 0.0)
            - float(before_pos[index] if index < len(before_pos) else 0.0),
            0.0,
        )
        for index in range(max(len(after_pos), len(before_pos)))
    ]
    drafts = result["num_drafts"]
    draft_tokens = result["num_draft_tokens"]
    accepted_tokens = result["num_accepted_tokens"]
    result["active"] = draft_tokens > 0
    result["acceptance_rate"] = accepted_tokens / draft_tokens if draft_tokens else 0.0
    result["mean_acceptance_length"] = (
        1.0 + accepted_tokens / drafts if drafts else 0.0
    )
    result["accepted_tokens_per_draft"] = (
        accepted_tokens / drafts if drafts else 0.0
    )
    result["metrics_available"] = True
    result["acceptance_rate_per_pos"] = [
        value / drafts if drafts else 0.0
        for value in result["num_accepted_tokens_per_pos"]
    ]
    return result


def sum_spec_decode_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or not any(row for row in rows):
        return {}
    total: dict[str, Any] = {
        "num_drafts": sum(float(row.get("num_drafts", 0.0)) for row in rows),
        "num_draft_tokens": sum(
            float(row.get("num_draft_tokens", 0.0)) for row in rows
        ),
        "num_accepted_tokens": sum(
            float(row.get("num_accepted_tokens", 0.0)) for row in rows
        ),
    }
    max_positions = max(
        (len(row.get("num_accepted_tokens_per_pos", [])) for row in rows),
        default=0,
    )
    total["num_accepted_tokens_per_pos"] = [
        sum(
            float(row.get("num_accepted_tokens_per_pos", [])[index])
            if index < len(row.get("num_accepted_tokens_per_pos", []))
            else 0.0
            for row in rows
        )
        for index in range(max_positions)
    ]
    drafts = total["num_drafts"]
    draft_tokens = total["num_draft_tokens"]
    accepted_tokens = total["num_accepted_tokens"]
    total["active"] = draft_tokens > 0
    total["acceptance_rate"] = accepted_tokens / draft_tokens if draft_tokens else 0.0
    total["mean_acceptance_length"] = (
        1.0 + accepted_tokens / drafts if drafts else 0.0
    )
    total["accepted_tokens_per_draft"] = (
        accepted_tokens / drafts if drafts else 0.0
    )
    total["metrics_available"] = True
    total["acceptance_rate_per_pos"] = [
        value / drafts if drafts else 0.0
        for value in total["num_accepted_tokens_per_pos"]
    ]
    return total


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def runtime_metadata() -> dict[str, Any]:
    import torch  # pyright: ignore[reportMissingImports]

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm_version": package_version("vllm"),
        "torch_version": package_version("torch"),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "environment": {
            name: os.environ.get(name)
            for name in (
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_ATTENTION_BACKEND",
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--mode", choices=("baseline", "static", "dynamic"), required=True)
    parser.add_argument("--static-k", type=int, default=5)
    parser.add_argument("--dynamic-schedule", default=DEFAULT_DYNAMIC_SCHEDULE)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", default="")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=1792)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--cudagraph-mode", default="PIECEWISE")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=512)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--measure-repeats", type=int, default=3)
    parser.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        "max_num_seqs": max(args.batch_sizes),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "seed": args.seed,
        "disable_log_stats": False,
        "compilation_config": {"cudagraph_mode": args.cudagraph_mode},
    }
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = load_prompt_token_ids(
        tokenizer,
        count=max(args.batch_sizes),
        isl=args.isl,
        prompt_jsonl=args.prompt_jsonl,
        prompt_offset=args.prompt_offset,
    )
    sampling_params = SamplingParams(
        min_tokens=args.osl,
        max_tokens=args.osl,
        ignore_eos=True,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "runtime": runtime_metadata(),
        "config": {
            "tag": args.tag,
            "model": args.model,
            "draft_model": args.draft_model,
            "mode": args.mode,
            "speculative_config": speculative_config,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "tp": args.tensor_parallel_size,
            "pp": args.pipeline_parallel_size,
            "total_gpus": args.tensor_parallel_size * args.pipeline_parallel_size,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": max(args.batch_sizes),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "attention_backend": args.attention_backend or "auto",
            "cudagraph_mode": args.cudagraph_mode,
            "isl": args.isl,
            "osl": args.osl,
            "batch_sizes": sorted(set(args.batch_sizes)),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "warmup_repeats": args.warmup_repeats,
            "measure_repeats": args.measure_repeats,
            "prompt_jsonl": str(args.prompt_jsonl) if args.prompt_jsonl else None,
            "prompt_offset": args.prompt_offset,
            "prompt_count_loaded": len(prompt_token_ids),
            "cuda_profiler_range": args.cuda_profiler_range,
        },
        "results": rows,
    }

    def flush() -> None:
        write_json_atomic(args.output, payload)

    import torch  # pyright: ignore[reportMissingImports]

    for batch_size in sorted(set(args.batch_sizes)):
        prompts = [
            {"prompt_token_ids": prompt_token_ids[index]}
            for index in range(batch_size)
        ]
        for _ in range(args.warmup_repeats):
            llm.generate(prompts, sampling_params, use_tqdm=False)

        repeats: list[dict[str, Any]] = []
        for repeat in range(args.measure_repeats):
            before = read_spec_decode_counters(llm)
            torch.cuda.synchronize()
            if args.cuda_profiler_range:
                if repeat != 0 or args.measure_repeats != 1:
                    raise ValueError(
                        "--cuda-profiler-range requires --measure-repeats 1"
                    )
                torch.cuda.profiler.start()
            range_name = f"vllm024.{args.mode}.bs{batch_size}.repeat{repeat}"
            torch.cuda.nvtx.range_push(range_name)
            started = time.perf_counter()
            try:
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                torch.cuda.synchronize()
                latency_s = time.perf_counter() - started
            finally:
                torch.cuda.nvtx.range_pop()
                if args.cuda_profiler_range:
                    torch.cuda.profiler.stop()
            output_tokens = sum(
                len(request.outputs[0].token_ids) for request in outputs
            )
            counters = diff_spec_decode_counters(
                read_spec_decode_counters(llm), before
            )
            repeats.append(
                {
                    "repeat": repeat,
                    "latency_s": latency_s,
                    "output_tokens": output_tokens,
                    "output_tok_s": output_tokens / latency_s,
                    "spec_decode_metrics": counters,
                }
            )

        total_output_tokens = sum(row["output_tokens"] for row in repeats)
        total_latency_s = sum(row["latency_s"] for row in repeats)
        total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
        metrics = sum_spec_decode_counters(
            [row["spec_decode_metrics"] for row in repeats]
        )
        expected_k = 0
        if args.mode == "static":
            expected_k = args.static_k
        elif args.mode == "dynamic":
            expected_k = dynamic_k_for_batch_size(dynamic_schedule, batch_size)
        if expected_k > 0 and (
            not metrics.get("metrics_available") or not metrics.get("active")
        ):
            raise RuntimeError(
                f"SpecDec counters are unavailable or inactive for mode={args.mode}, "
                f"bs={batch_size}, expected_k={expected_k}"
            )
        row = {
            "bs": batch_size,
            "latency_s_mean": statistics.fmean(
                repeat["latency_s"] for repeat in repeats
            ),
            "latency_s_median": statistics.median(
                repeat["latency_s"] for repeat in repeats
            ),
            "output_tokens": total_output_tokens,
            "output_tok_s": total_output_tokens / total_latency_s,
            "output_tok_s_per_gpu": total_output_tokens / total_latency_s / total_gpus,
            "spec_decode_metrics": metrics,
            "repeats": repeats,
        }
        row["latency_s"] = row["latency_s_mean"]
        row["mean_latency_s"] = row["latency_s_mean"]
        row["prompt_count_used"] = batch_size
        row["num_batches"] = args.measure_repeats
        rows.append(row)
        flush()
        metrics_text = ""
        if metrics:
            metrics_text = (
                f" acceptance={metrics['acceptance_rate']:.2%}"
                f" mean_accept_len={metrics['mean_acceptance_length']:.2f}"
            )
        print(
            f"bs={batch_size} tok/s/GPU={row['output_tok_s_per_gpu']:.2f}"
            f" latency={row['latency_s_mean']:.3f}s{metrics_text}",
            flush=True,
        )

    payload["status"] = "complete"
    flush()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
