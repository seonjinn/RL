#!/usr/bin/env python3
"""Produce token and chosen-logprob artifacts for baseline/SpecDec parity."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str


@dataclass(frozen=True)
class SampleRequest:
    prompt: PromptRecord
    sample_id: str


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    tokenizer: str
    draft_model: str | None
    method: str
    num_speculative_tokens: int
    target_tp: int
    draft_tp: int
    max_model_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    num_nodes: int = 1
    gpus_per_node: int = 4
    gpu_memory_utilization: float = 0.8
    draft_sample_method: str = "greedy"
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 0


def load_prompt_records(path: Path, *, limit: int) -> list[PromptRecord]:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    prompts: list[PromptRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as prompt_file:
        for line_number, line in enumerate(prompt_file, start=1):
            if len(prompts) >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt_id = str(payload["id"])
            text = str(payload["prompt"])
            if not prompt_id or not text:
                raise ValueError(f"{path}:{line_number} has an empty id or prompt")
            if prompt_id in seen:
                raise ValueError(
                    f"{path}:{line_number} duplicates prompt id {prompt_id!r}"
                )
            seen.add(prompt_id)
            prompts.append(PromptRecord(prompt_id=prompt_id, text=text))

    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


def expand_prompt_records(
    prompts: Sequence[PromptRecord], *, samples_per_prompt: int
) -> list[SampleRequest]:
    if samples_per_prompt <= 0:
        raise ValueError(
            f"samples_per_prompt must be positive, got {samples_per_prompt}"
        )
    return [
        SampleRequest(prompt=prompt, sample_id=f"{sample_index:04d}")
        for prompt in prompts
        for sample_index in range(samples_per_prompt)
    ]


def extract_generated_sample(
    *,
    prompt_id: str,
    sample_id: str,
    output_ids: Sequence[int],
    token_logprobs: Sequence[float],
    input_length: int,
    generation_length: int,
) -> dict[str, Any]:
    if input_length < 0 or generation_length <= 0:
        raise ValueError(
            "input_length must be nonnegative and generation_length must be positive"
        )
    required_length = input_length + generation_length
    if len(output_ids) < required_length:
        raise ValueError(
            f"output_ids is shorter than required sequence length {required_length}"
        )
    if len(token_logprobs) < required_length:
        raise ValueError(
            f"token_logprobs is shorter than required sequence length {required_length}"
        )

    generated_ids = [int(token) for token in output_ids[input_length:required_length]]
    generated_logprobs = [
        float(value) for value in token_logprobs[input_length:required_length]
    ]
    if not all(math.isfinite(value) for value in generated_logprobs):
        raise ValueError("generated token_logprobs contains a non-finite value")

    return {
        "prompt_id": prompt_id,
        "sample_id": sample_id,
        "token_ids": generated_ids,
        "token_logprobs": generated_logprobs,
    }


def extract_batch_samples(
    *,
    requests: Sequence[SampleRequest],
    input_lengths: Sequence[int],
    generation_lengths: Sequence[int],
    output_ids: Sequence[Sequence[int]],
    token_logprobs: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    row_counts = {
        len(requests),
        len(input_lengths),
        len(generation_lengths),
        len(output_ids),
        len(token_logprobs),
    }
    if len(row_counts) != 1:
        raise ValueError(
            "request, length, token, and logprob batches must have equal row counts"
        )

    return [
        extract_generated_sample(
            prompt_id=request.prompt.prompt_id,
            sample_id=request.sample_id,
            output_ids=output_ids[index],
            token_logprobs=token_logprobs[index],
            input_length=int(input_lengths[index]),
            generation_length=int(generation_lengths[index]),
        )
        for index, request in enumerate(requests)
    ]


def run_generation_batches(
    policy: Any,
    requests: Sequence[SampleRequest],
    *,
    batch_size: int,
    greedy: bool,
    build_batch: Callable[[Sequence[SampleRequest]], Any],
    output_file: TextIO,
) -> dict[str, int | float]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    started = time.perf_counter()
    completed_samples = 0
    generated_tokens = 0
    for offset in range(0, len(requests), batch_size):
        request_batch = requests[offset : offset + batch_size]
        input_batch = build_batch(request_batch)
        generated = policy.generate(input_batch, greedy=greedy)
        samples = extract_batch_samples(
            requests=request_batch,
            input_lengths=input_batch["input_lengths"],
            generation_lengths=generated["generation_lengths"],
            output_ids=generated["output_ids"],
            token_logprobs=generated["logprobs"],
        )
        for sample in samples:
            output_file.write(json.dumps(sample, sort_keys=True) + "\n")
            completed_samples += 1
            generated_tokens += len(sample["token_ids"])
        output_file.flush()

    elapsed_seconds = time.perf_counter() - started
    return {
        "completed_samples": completed_samples,
        "generated_tokens": generated_tokens,
        "generation_elapsed_seconds": elapsed_seconds,
        "generation_throughput_tokens_per_second": (
            generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }


def cleanup_runtime(policy: Any, cluster: Any, ray_module: Any) -> list[str]:
    errors: list[str] = []
    if policy is not None:
        try:
            if not policy.shutdown():
                errors.append("VllmGeneration shutdown returned false")
        except BaseException as error:
            errors.append(f"policy shutdown: {type(error).__name__}: {error}")
    if cluster is not None:
        try:
            cluster.shutdown()
        except BaseException as error:
            errors.append(f"cluster shutdown: {type(error).__name__}: {error}")
    if ray_module is not None:
        try:
            ray_module.shutdown()
        except BaseException as error:
            errors.append(f"ray shutdown: {type(error).__name__}: {error}")
    return errors


def build_generation_config(settings: GenerationSettings) -> dict[str, Any]:
    vllm_kwargs: dict[str, Any] = {
        "compilation_config": {"cudagraph_mode": "PIECEWISE"},
        "enable_chunked_prefill": settings.enable_chunked_prefill,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
    }
    if settings.max_num_seqs > 0:
        vllm_kwargs["max_num_seqs"] = settings.max_num_seqs
    if settings.draft_model is not None:
        vllm_kwargs["speculative_config"] = {
            "method": settings.method,
            "model": settings.draft_model,
            "num_speculative_tokens": settings.num_speculative_tokens,
            "draft_tensor_parallel_size": settings.draft_tp,
            "rejection_sample_method": "standard",
            "draft_sample_method": settings.draft_sample_method,
        }

    return {
        "backend": "vllm",
        "model_name": settings.model,
        "tokenizer": {"name": settings.tokenizer},
        "dtype": "bfloat16",
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "ignore_eos": False,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": settings.target_tp,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "max_model_len": settings.max_model_len,
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": False,
            "kv_cache_dtype": "auto",
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 0.5,
            "use_deep_gemm": False,
            "enable_prefix_caching": settings.enable_prefix_caching,
        },
        "colocated": {
            "enabled": False,
            "resources": {
                "num_nodes": settings.num_nodes,
                "gpus_per_node": settings.gpus_per_node,
            },
        },
        "vllm_kwargs": vllm_kwargs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--draft-model")
    parser.add_argument("--method", default="eagle3")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--draft-sample-method", choices=("greedy", "probabilistic"), default="greedy"
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("greedy", "sampled"), required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--ray-log-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tokenize_prompt(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _build_token_batch(tokenizer: Any, requests: Sequence[SampleRequest]) -> Any:
    # Torch and NeMo-RL are intentionally deferred so host-only contract tests stay light.
    import torch

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    tokenized = [
        _tokenize_prompt(tokenizer, request.prompt.text) for request in requests
    ]
    max_length = max(len(token_ids) for token_ids in tokenized)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    padded = [
        token_ids + [int(pad_token_id)] * (max_length - len(token_ids))
        for token_ids in tokenized
    ]
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "input_lengths": torch.tensor(
                [len(token_ids) for token_ids in tokenized], dtype=torch.int32
            ),
        }
    )


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata_json or args.output_jsonl.with_suffix(
        args.output_jsonl.suffix + ".metadata.json"
    )
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output_jsonl}; pass --overwrite to replace it"
        )

    settings = GenerationSettings(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        draft_model=args.draft_model,
        method=args.method,
        num_speculative_tokens=args.num_speculative_tokens,
        target_tp=args.target_tp,
        draft_tp=args.draft_tp,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_utilization=args.gpu_memory_utilization,
        draft_sample_method=args.draft_sample_method,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    prompts = load_prompt_records(args.prompt_data, limit=args.prompt_limit)
    requests = expand_prompt_records(
        prompts, samples_per_prompt=args.samples_per_prompt
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.ray_log_dir is not None:
        args.ray_log_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "status": "starting",
        "git_commit": _git_commit(),
        "mode": args.mode,
        "prompt_data": str(args.prompt_data),
        "prompt_count": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "requested_samples": len(requests),
        "batch_size": args.batch_size,
        "settings": settings.__dict__,
        "output_jsonl": str(args.output_jsonl),
        "started_at_unix": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_metadata(metadata_path, metadata)

    ray_module: Any = None
    cluster: Any = None
    policy: Any = None
    try:
        # Ray, Transformers, and NeMo-RL are compute-node dependencies.
        import ray
        from transformers import AutoTokenizer

        from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
        from nemo_rl.models.generation import configure_generation_config
        from nemo_rl.models.generation.vllm import VllmGeneration

        ray_module = ray
        tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generation_config = configure_generation_config(
            build_generation_config(settings), tokenizer, is_eval=True
        )
        metadata["resolved_generation_config"] = generation_config
        _write_metadata(metadata_path, metadata)

        init_ray(log_dir=str(args.ray_log_dir) if args.ray_log_dir else None)
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[settings.gpus_per_node] * settings.num_nodes,
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=settings.gpus_per_node,
            name="vllm024_generation_parity",
        )
        policy = VllmGeneration(
            cluster, generation_config, name_prefix="vllm024_generation_parity"
        )
        policy.snapshot_step_metrics()
        mode = "w" if args.overwrite else "x"
        with args.output_jsonl.open(mode, encoding="utf-8") as output_file:
            generation_summary = run_generation_batches(
                policy,
                requests,
                batch_size=args.batch_size,
                greedy=args.mode == "greedy",
                build_batch=lambda batch_requests: _build_token_batch(
                    tokenizer, batch_requests
                ),
                output_file=output_file,
            )
        specdec_metrics = policy.get_step_metrics()
        allocated_gpus = settings.num_nodes * settings.gpus_per_node
        metadata.update(generation_summary)
        metadata.update(
            {
                "status": "passed",
                "specdec_metrics": specdec_metrics,
                "allocated_gpus": allocated_gpus,
                "generation_throughput_tokens_per_second_per_gpu": (
                    generation_summary["generation_throughput_tokens_per_second"]
                    / allocated_gpus
                ),
            }
        )
        return 0
    except BaseException as error:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        metadata["finished_at_unix"] = time.time()
        metadata["elapsed_seconds"] = (
            metadata["finished_at_unix"] - metadata["started_at_unix"]
        )
        cleanup_errors = cleanup_runtime(policy, cluster, ray_module)
        if cleanup_errors:
            metadata["cleanup_errors"] = cleanup_errors
            for cleanup_error in cleanup_errors:
                print(f"WARNING: {cleanup_error}", flush=True)
        try:
            _write_metadata(metadata_path, metadata)
        except Exception as metadata_error:
            print(f"ERROR: metadata write failed: {metadata_error}", flush=True)
            if metadata["status"] != "failed":
                raise


if __name__ == "__main__":
    raise SystemExit(main())
