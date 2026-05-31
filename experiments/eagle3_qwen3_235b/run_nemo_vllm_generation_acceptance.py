#!/usr/bin/env python3
"""Run a NeMo-RL VllmGeneration-only speculative decoding acceptance check.

This avoids Megatron policy import/checkpoint setup and exercises the same
NeMo-RL vLLM generation backend and metric counters used by GRPO.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = env(name, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "y", "on"}


def env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def env_int_list(name: str) -> list[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [int(item) for item in raw.replace(",", " ").split()]


def normalize_optional_path(value: str) -> str:
    if value.strip().lower() in {"", "none", "null", "baseline"}:
        return ""
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=env("MODEL_PATH", "Qwen/Qwen3-235B-A22B-Thinking-2507"))
    parser.add_argument("--tokenizer", default=env("TOKENIZER_PATH", ""))
    parser.add_argument("--draft-model", default=env("DRAFT_MODEL", ""))
    parser.add_argument("--prompt-data", type=Path, default=Path(env("PROMPT_DATA", "")))
    parser.add_argument("--output-json", type=Path, default=Path(env("OUTPUT_JSON", "nemo_vllm_acceptance.json")))
    parser.add_argument("--prompt-limit", type=int, default=env_int("PROMPT_LIMIT", 8))
    parser.add_argument("--prompt-offset", type=int, default=env_int("PROMPT_OFFSET", 0))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("MAX_NEW_TOKENS", 512))
    parser.add_argument("--max-model-len", type=int, default=env_int("MAX_MODEL_LEN", 4096))
    parser.add_argument("--temperature", type=float, default=env_float("TEMPERATURE", 1.0))
    parser.add_argument("--top-p", type=float, default=env_float("TOP_P", 1.0))
    parser.add_argument("--num-speculative-tokens", type=int, default=env_int("NUM_SPECULATIVE_TOKENS", 1))
    parser.add_argument("--draft-tp", type=int, default=env_int("DRAFT_TP", 1))
    parser.add_argument("--vllm-tp", type=int, default=env_int("VLLM_TP", 8))
    parser.add_argument("--vllm-pp", type=int, default=env_int("VLLM_PP", 1))
    parser.add_argument("--num-nodes", type=int, default=env_int("NUM_NODES", 2))
    parser.add_argument("--gpus-per-node", type=int, default=env_int("GPUS_PER_NODE", 4))
    parser.add_argument("--gpu-memory-utilization", type=float, default=env_float("VLLM_GPU_UTIL", 0.8))
    parser.add_argument("--attention-backend", default=env("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"))
    parser.add_argument("--max-num-seqs", type=int, default=env_int("VLLM_MAX_NUM_SEQS", 0))
    parser.add_argument("--max-cudagraph-capture-size", type=int, default=env_int("VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE", 0))
    parser.add_argument("--async-engine", action=argparse.BooleanOptionalAction, default=env_bool("ASYNC_ENGINE", True))
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=env_bool("VLLM_ENFORCE_EAGER", False))
    ray_log_dir = env("RAY_LOG_DIR", "")
    parser.add_argument("--ray-log-dir", type=Path, default=Path(ray_log_dir) if ray_log_dir else None)
    return parser.parse_args()


def load_prompts(path: Path, limit: int, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"prompt data not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            if line_no < offset:
                continue
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "problem" in item:
                text = item["problem"]
            elif "prompt" in item:
                text = item["prompt"]
            elif "messages" in item:
                text = "\n".join(msg.get("content", "") for msg in item["messages"] if msg.get("role") == "user")
            else:
                raise ValueError(f"unsupported prompt row keys at source line {line_no + 1}: {sorted(item)}")
            rows.append({"id": item.get("id") or item.get("conversation_id") or f"row-{line_no}", "text": text})
    if not rows:
        raise ValueError(f"no prompts loaded from {path}")
    return rows


def build_batch(tokenizer: Any, prompt: str) -> BatchedDataDict:
    messages = [{"role": "user", "content": prompt}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    input_lengths = torch.tensor([len(token_ids)], dtype=torch.int32)
    return BatchedDataDict({"input_ids": input_ids, "input_lengths": input_lengths})


def decode_generated_text(tokenizer: Any, batch: BatchedDataDict, generated: BatchedDataDict) -> str:
    input_len = int(batch["input_lengths"][0].item())
    gen_len = int(generated["generation_lengths"][0].item())
    if "output_ids" not in generated:
        return ""
    token_ids = generated["output_ids"][0, input_len : input_len + gen_len]
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def extract_spec_decode_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    spec_decode = {}
    if isinstance(metrics, dict):
        vllm_logger_metrics = metrics.get("vllm_logger_metrics", metrics)
        if isinstance(vllm_logger_metrics, dict):
            spec_decode = vllm_logger_metrics.get("spec_decode", {}) or {}
        if not spec_decode and "vllm/spec_acceptance_rate" in metrics:
            spec_decode = {
                "metrics_available": True,
                "active": metrics.get("vllm/spec_num_draft_tokens", 0) > 0,
                "acceptance_rate": metrics.get("vllm/spec_acceptance_rate"),
                "num_accepted_tokens": metrics.get("vllm/spec_num_accepted_tokens"),
                "num_draft_tokens": metrics.get("vllm/spec_num_draft_tokens"),
                "num_drafts": metrics.get("vllm/spec_num_drafts"),
                "mean_acceptance_length": metrics.get("vllm/spec_acceptance_length"),
            }
    return spec_decode if isinstance(spec_decode, dict) else {}


def snapshot_generation_metrics(policy: VllmGeneration) -> str:
    if hasattr(policy, "snapshot_step_metrics"):
        policy.snapshot_step_metrics()
        return "step_metrics"
    policy.clear_vllm_logger_metrics()
    return "vllm_logger_metrics"


def collect_generation_metrics(policy: VllmGeneration, source: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if source == "step_metrics" and hasattr(policy, "get_step_metrics"):
        metrics.update(policy.get_step_metrics())
    try:
        metrics["vllm_logger_metrics"] = policy.get_vllm_logger_metrics()
    except Exception as exc:
        metrics["vllm_logger_metrics_error"] = f"{type(exc).__name__}: {exc}"
    return metrics


def generation_config(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    tokenizer_name = args.tokenizer or args.model
    compilation_config: dict[str, Any] = {
        "pass_config": {"fuse_allreduce_rms": False},
    }
    compilation_level = env_int_optional("VLLM_COMPILATION_LEVEL")
    if compilation_level is not None:
        compilation_config["level"] = compilation_level
    cudagraph_mode = env("VLLM_CUDAGRAPH_MODE", "").strip()
    if cudagraph_mode:
        compilation_config["cudagraph_mode"] = cudagraph_mode
    cudagraph_sizes = env_int_list("VLLM_CUDAGRAPH_CAPTURE_SIZES")
    if cudagraph_sizes is not None:
        compilation_config["cudagraph_capture_sizes"] = cudagraph_sizes

    vllm_kwargs: dict[str, Any] = {
        "compilation_config": compilation_config,
    }
    if args.draft_model:
        vllm_kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": args.draft_tp,
        }
    if args.max_num_seqs > 0:
        vllm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_cudagraph_capture_size > 0:
        vllm_kwargs["max_cudagraph_capture_size"] = args.max_cudagraph_capture_size

    cfg: dict[str, Any] = {
        "backend": "vllm",
        "model_name": args.model,
        "tokenizer": {"name": tokenizer_name},
        "dtype": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": args.vllm_tp,
            "pipeline_parallel_size": args.vllm_pp,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "async_engine": args.async_engine,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": args.enforce_eager,
            "kv_cache_dtype": "auto",
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 0.5,
            "use_deep_gemm": False,
        },
        "colocated": {
            "enabled": False,
            "resources": {
                "num_nodes": args.num_nodes,
                "gpus_per_node": args.gpus_per_node,
            },
        },
        "vllm_kwargs": vllm_kwargs,
    }
    return configure_generation_config(cfg, tokenizer, is_eval=True)


async def run_async_generation(policy: VllmGeneration, tokenizer: Any, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for prompt in prompts:
        batch = build_batch(tokenizer, prompt["text"])
        start = time.time()
        generated = None
        async for _idx, result in policy.generate_async(batch, greedy=False):
            generated = result
        elapsed = time.time() - start
        if generated is None:
            raise RuntimeError(f"no generation result for prompt {prompt['id']}")
        outputs.append(
            {
                "id": prompt["id"],
                "prompt_text": prompt["text"],
                "input_tokens": int(batch["input_lengths"][0].item()),
                "generated_tokens": int(generated["generation_lengths"][0].item()),
                "generated_text": decode_generated_text(tokenizer, batch, generated),
                "elapsed_sec": elapsed,
            }
        )
        print(json.dumps(outputs[-1], sort_keys=True), flush=True)
    return outputs


def run_sync_generation(policy: VllmGeneration, tokenizer: Any, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for prompt in prompts:
        batch = build_batch(tokenizer, prompt["text"])
        start = time.time()
        generated = policy.generate(batch, greedy=False)
        elapsed = time.time() - start
        outputs.append(
            {
                "id": prompt["id"],
                "prompt_text": prompt["text"],
                "input_tokens": int(batch["input_lengths"][0].item()),
                "generated_tokens": int(generated["generation_lengths"][0].item()),
                "generated_text": decode_generated_text(tokenizer, batch, generated),
                "elapsed_sec": elapsed,
            }
        )
        print(json.dumps(outputs[-1], sort_keys=True), flush=True)
    return outputs


def main() -> None:
    args = parse_args()
    args.draft_model = normalize_optional_path(args.draft_model)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.ray_log_dir is not None:
        args.ray_log_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_data, args.prompt_limit, args.prompt_offset)
    tokenizer = get_tokenizer({"name": args.tokenizer or args.model})
    cfg = generation_config(args, tokenizer)

    init_ray(log_dir=str(args.ray_log_dir) if args.ray_log_dir is not None else None)
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[args.gpus_per_node for _ in range(args.num_nodes)],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=args.gpus_per_node,
        name="nemo-vllm-eagle3-acceptance",
    )

    started = time.time()
    policy: VllmGeneration | None = None
    try:
        policy = VllmGeneration(cluster, cfg, name_prefix="nemo_vllm_eagle3_acceptance")
        metrics_source = snapshot_generation_metrics(policy)
        if args.async_engine:
            generations = asyncio.run(run_async_generation(policy, tokenizer, prompts))
        else:
            generations = run_sync_generation(policy, tokenizer, prompts)
        time.sleep(2.0)
        metrics = collect_generation_metrics(policy, metrics_source)
        spec_decode = extract_spec_decode_metrics(metrics)
        generation_tokens = sum(item["generated_tokens"] for item in generations)
        generation_elapsed_sec = sum(item["elapsed_sec"] for item in generations)
        summary = {
            "status": "pass",
            "elapsed_sec": time.time() - started,
            "model": args.model,
            "draft_model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "prompt_data": str(args.prompt_data),
            "prompt_count": len(prompts),
            "generation_tokens": generation_tokens,
            "generation_elapsed_sec": generation_elapsed_sec,
            "generation_throughput_tok_s": (
                generation_tokens / generation_elapsed_sec if generation_elapsed_sec > 0 else None
            ),
            "generations": generations,
            "metrics": metrics,
            "spec_decode_metrics_available": spec_decode.get("metrics_available"),
            "spec_decode_active": spec_decode.get("active"),
            "acceptance_rate": spec_decode.get("acceptance_rate"),
            "accepted_tokens": spec_decode.get("num_accepted_tokens"),
            "draft_tokens": spec_decode.get("num_draft_tokens"),
            "num_drafts": spec_decode.get("num_drafts"),
            "acceptance_length": spec_decode.get("mean_acceptance_length"),
        }
    except Exception as exc:
        summary = {
            "status": "fail",
            "elapsed_sec": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model": args.model,
            "draft_model": args.draft_model,
            "prompt_data": str(args.prompt_data),
        }
        raise
    finally:
        if policy is not None:
            try:
                policy.finish_generation()
            except Exception as exc:
                print(f"finish_generation failed: {exc}", flush=True)
        try:
            cluster.shutdown()
        except Exception as exc:
            print(f"cluster shutdown failed: {exc}", flush=True)
        ray.shutdown()
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
