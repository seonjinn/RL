#!/usr/bin/env python3
"""Run one TP1 vLLM worker for the matched 2,048-sample barrier study."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from experiments.dynamic_sd_sync_rollout import sync_rollout_dynamic_sd as harness

from .contract import (
    Arm,
    ExperimentContract,
    TargetAttentionBackend,
    build_arms,
    worker_assignment,
)


def build_llm_kwargs(
    contract: ExperimentContract,
    arm: Arm,
    *,
    target_path: str,
    drafter_path: str,
) -> dict[str, Any]:
    """Build the version-pinned LLM arguments shared by every arm."""
    kwargs: dict[str, Any] = {
        "model": target_path,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": 0.85,
        "seed": contract.base_seed,
        "disable_log_stats": False,
        "max_model_len": contract.max_model_len,
        "max_num_seqs": contract.max_num_seqs,
        "max_num_batched_tokens": contract.max_num_batched_tokens,
        "attention_backend": contract.target_attention_backend,
        "kernel_config": {
            "moe_backend": "flashinfer_trtllm",
            "enable_flashinfer_autotune": False,
        },
        "compilation_config": {
            "cudagraph_mode": contract.cuda_graph_mode,
            "max_cudagraph_capture_size": contract.max_cudagraph_capture_size,
        },
    }
    speculative_config = arm.speculative_config(drafter_path)
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config
    return kwargs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=[arm.key for arm in build_arms()])
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--drafter-path", required=True)
    parser.add_argument("--prompt-jsonl", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target-attention-backend",
        choices=("FLEX_ATTENTION", "TRITON_ATTN"),
        default="FLEX_ATTENTION",
    )
    return parser.parse_args()


def _validate_worker_output(
    path: Path,
    *,
    contract: ExperimentContract,
    arm: Arm,
    worker_index: int,
    prompt_offset: int,
    seed: int,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if payload.get("partial") is not False or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("worker result is incomplete")
    row = rows[0]
    lengths = row.get("output_lengths") or {}
    request_timing = row.get("request_timing") or []
    actual_tokens = sum(int(item["output_tokens"]) for item in request_timing)
    if row.get("num_sequences") != contract.samples_per_worker:
        raise ValueError("worker result has the wrong sequence count")
    if lengths.get("count") != contract.samples_per_worker:
        raise ValueError("worker length summary has the wrong sequence count")
    if actual_tokens != lengths.get("total"):
        raise ValueError("worker token accounting mismatch")
    payload["runtime_contract"] = {
        "worker_index": worker_index,
        "prompt_offset": prompt_offset,
        "seed": seed,
        "vllm_version": contract.vllm_version,
        "vllm_commit": contract.vllm_commit,
        "arm": arm.key,
        "max_num_seqs": contract.max_num_seqs,
        "max_num_batched_tokens": contract.max_num_batched_tokens,
        "cuda_graph_mode": contract.cuda_graph_mode,
        "max_cudagraph_capture_size": contract.max_cudagraph_capture_size,
        "tokens_ok": True,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parsed = _parse_args()
    contract = replace(
        ExperimentContract(),
        target_attention_backend=cast(
            TargetAttentionBackend, parsed.target_attention_backend
        ),
    )
    assignment = worker_assignment(contract, parsed.worker_index)
    arm = next(arm for arm in build_arms(contract) if arm.key == parsed.arm)

    from vllm import LLM

    llm = LLM(
        **build_llm_kwargs(
            contract,
            arm,
            target_path=parsed.target_path,
            drafter_path=parsed.drafter_path,
        )
    )
    run_args = argparse.Namespace(
        mode="rollout",
        model=parsed.target_path,
        speculative_config=(
            None
            if arm.speculative_config(parsed.drafter_path) is None
            else json.dumps(arm.speculative_config(parsed.drafter_path), sort_keys=True)
        ),
        tp=1,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        disable_custom_all_reduce=False,
        disable_flashinfer_autotune=True,
        attention_backend=contract.target_attention_backend,
        moe_backend="flashinfer_trtllm",
        max_model_len=contract.max_model_len,
        max_num_seqs=contract.max_num_seqs,
        max_num_batched_tokens=contract.max_num_batched_tokens,
        cudagraph_capture_sizes=[],
        seed=assignment.seed,
        prompt_jsonl=parsed.prompt_jsonl,
        prompt_offset=assignment.prompt_offset,
        isl_cap=4_096,
        temperature=contract.temperature,
        top_p=contract.top_p,
        top_k=-1,
        batch_sizes=[],
        osl=contract.max_output_tokens,
        repeats=1,
        num_prompts_per_step=contract.prompts_per_worker,
        num_generations_per_prompt=contract.generations_per_prompt,
        num_steps=1,
        max_tokens=contract.max_output_tokens,
        per_request_seed=True,
        replay_jsonl=None,
        replay_trajectories=0,
        replay_max_turns=0,
        replay_copies=0,
        replay_turn_max_tokens=0,
        save_token_ids=False,
        output=str(parsed.output),
        tag=f"{arm.key}-worker-{parsed.worker_index:02d}",
    )
    harness.run_rollout(run_args, llm)
    _validate_worker_output(
        parsed.output,
        contract=contract,
        arm=arm,
        worker_index=parsed.worker_index,
        prompt_offset=assignment.prompt_offset,
        seed=assignment.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
