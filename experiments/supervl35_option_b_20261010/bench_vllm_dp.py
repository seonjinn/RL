"""Benchmark two TP2 vLLM data-parallel ranks on one four-GPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path
from typing import Any

if __package__:
    from .bench_vllm_generation import _prompt_ids, benchmark_cases, build_engine_kwargs
else:
    from bench_vllm_generation import _prompt_ids, benchmark_cases, build_engine_kwargs


def requests_for_rank(total: int, rank: int, dp_size: int) -> int:
    if total < dp_size:
        raise ValueError("each DP rank needs at least one request")
    if not 0 <= rank < dp_size:
        raise ValueError("invalid DP rank")
    return total // dp_size + int(rank < total % dp_size)


def trials_for_batch(batch_size: int, trials: int, high_concurrency_trials: int) -> int:
    return high_concurrency_trials if batch_size > 128 else trials


def aggregate_rank_runs(rows: list[dict[str, Any]], gpus: int) -> dict[str, Any]:
    if gpus <= 0 or len({row["rank"] for row in rows}) != len(rows):
        raise ValueError("invalid GPU count or duplicate rank")
    if not rows or any(row["finished_s"] <= row["started_s"] for row in rows):
        raise ValueError("invalid rank timing")
    elapsed = max(row["finished_s"] for row in rows) - min(
        row["started_s"] for row in rows
    )
    tokens = sum(row["output_tokens"] for row in rows)
    return {
        "requests": sum(row["requests"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "output_tokens": tokens,
        "elapsed_s": elapsed,
        "output_tokens_per_second": tokens / elapsed,
        "output_tokens_per_second_per_gpu": tokens / elapsed / gpus,
        "rank_elapsed_s": [row["finished_s"] - row["started_s"] for row in rows],
    }


def _open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(
    rank: int,
    config: str,
    prompt_tokens: int,
    cases: list[tuple[int, int]],
    trials: int,
    high_concurrency_trials: int,
    max_num_seqs: int,
    port: int,
    barrier: Any,
    queue: Any,
) -> None:
    os.environ.update(
        VLLM_DP_RANK=str(rank),
        VLLM_DP_RANK_LOCAL=str(rank),
        VLLM_DP_SIZE="2",
        VLLM_DP_MASTER_IP="127.0.0.1",
        VLLM_DP_MASTER_PORT=str(port),
    )

    from omegaconf import OmegaConf
    from vllm import LLM, SamplingParams, __version__ as vllm_version

    full = OmegaConf.load(config)
    policy = full.policy
    generation = policy.generation
    recipe = {
        "policy": {
            "precision": policy.precision,
            "model_name": policy.model_name,
            "generation": {
                "model_name": generation.model_name,
                "vllm_cfg": OmegaConf.to_container(generation.vllm_cfg, resolve=True),
                "vllm_kwargs": OmegaConf.to_container(
                    generation.vllm_kwargs, resolve=True
                ),
            },
        }
    }
    cfg = recipe["policy"]["generation"]["vllm_cfg"]
    cfg["tensor_parallel_size"] = 2
    cfg["expert_parallel_size"] = 4
    fp8_kwargs = None
    if cfg["precision"] == "fp8":
        from nemo_rl.models.generation.vllm.quantization.fp8 import init_fp8

        fp8_kwargs = init_fp8(
            {**cfg, "async_engine": False},
            recipe["policy"]["generation"]["model_name"]
            or recipe["policy"]["model_name"],
            cfg["tensor_parallel_size"] * cfg["pipeline_parallel_size"],
        )
    kwargs = build_engine_kwargs(recipe, fp8_kwargs)
    kwargs["max_num_seqs"] = max_num_seqs
    kwargs["max_num_batched_tokens"] = min(
        kwargs["max_num_batched_tokens"], max_num_seqs * 512
    )
    llm = LLM(**kwargs)
    prompt_ids = _prompt_ids(llm.get_tokenizer(), prompt_tokens)
    prompt_hash = hashlib.sha256(
        json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    print(f"DP rank={rank} vLLM={vllm_version} prompt_sha256={prompt_hash}", flush=True)
    llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(max_tokens=32, ignore_eos=True, temperature=0),
        use_tqdm=False,
    )
    rank_runs = []
    for total, output_length in cases:
        for trial in range(trials_for_batch(total, trials, high_concurrency_trials)):
            count = requests_for_rank(total, rank, 2)
            prompts = [{"prompt_token_ids": prompt_ids} for _ in range(count)]
            params = SamplingParams(
                max_tokens=output_length, ignore_eos=True, temperature=0
            )
            barrier.wait()
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            finished = time.perf_counter()
            lengths = [len(item.outputs[0].token_ids) for item in outputs]
            if len(lengths) != count or any(
                length != output_length for length in lengths
            ):
                raise ValueError(f"DP rank {rank} output length mismatch")
            rank_runs.append(
                {
                    "rank": rank,
                    "batch_size": total,
                    "output_length": output_length,
                    "trial": trial,
                    "requests": count,
                    "prompt_tokens": sum(
                        len(item.prompt_token_ids) for item in outputs
                    ),
                    "output_tokens": sum(lengths),
                    "started_s": started,
                    "finished_s": finished,
                }
            )
            print(f"DP rank={rank} {json.dumps(rank_runs[-1], sort_keys=True)}", flush=True)
    queue.put(
        {
            "rank": rank,
            "prompt_sha256": prompt_hash,
            "vllm_version": vllm_version,
            "runs": rank_runs,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrencies", default="2,4,8")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--high-concurrency-trials", type=int, default=1)
    parser.add_argument("--max-num-seqs-per-rank", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if (
        min(
            args.prompt_tokens,
            args.trials,
            args.high_concurrency_trials,
            args.max_num_seqs_per_rank,
            args.timeout_seconds,
        )
        <= 0
    ):
        parser.error("length, trial, sequence cap and timeout must be positive")
    cases = benchmark_cases(args.concurrencies, args.output_tokens)
    if cases[0][0] < 2:
        parser.error("DP2 MoE requires at least one request on each rank")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2, timeout=min(args.timeout_seconds, 600))
    queue = ctx.Queue()
    port = _open_port()
    workers = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                str(args.config),
                args.prompt_tokens,
                cases,
                args.trials,
                args.high_concurrency_trials,
                args.max_num_seqs_per_rank,
                port,
                barrier,
                queue,
            ),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + args.timeout_seconds
    for worker in workers:
        worker.join(max(0, deadline - time.monotonic()))
    if any(worker.exitcode is None for worker in workers):
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()
        raise TimeoutError("TP2/DP2 benchmark exceeded its deadline")
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError(f"DP rank exit codes: {[w.exitcode for w in workers]}")
    ranks = [queue.get(timeout=10) for _ in workers]
    if {row["rank"] for row in ranks} != {0, 1}:
        raise ValueError("missing DP rank results")
    if len({row["prompt_sha256"] for row in ranks}) != 1:
        raise ValueError("DP rank prompts differ")
    if len({row["vllm_version"] for row in ranks}) != 1:
        raise ValueError("DP rank vLLM versions differ")
    runs = []
    offset = 0
    for batch_size, output_length in cases:
        for trial in range(
            trials_for_batch(batch_size, args.trials, args.high_concurrency_trials)
        ):
            rows = [rank["runs"][offset] for rank in ranks]
            if any(
                row["batch_size"] != batch_size or row["trial"] != trial for row in rows
            ):
                raise ValueError("DP rank case ordering differs")
            metrics = aggregate_rank_runs(rows, gpus=4)
            metrics.update(
                batch_size=batch_size, output_length=output_length, trial=trial
            )
            runs.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            offset += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "vllm_version": ranks[0]["vllm_version"],
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "prompt_sha256": ranks[0]["prompt_sha256"],
                "topology": {"tp": 2, "dp": 2, "ep": 4, "gpus": 4},
                "max_num_seqs_per_rank": args.max_num_seqs_per_rank,
                "trials": args.trials,
                "high_concurrency_trials": args.high_concurrency_trials,
                "runs": runs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
