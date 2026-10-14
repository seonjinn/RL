"""Run a matched SuperVL3.5 vLLM 0.31 fixed-token generation trial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path
from typing import Any

import yaml

from bench_vllm_dp import aggregate_rank_runs, requests_for_rank
from bench_vllm_generation import _prompt_ids
from vllm031_smoke import engine_kwargs


def build_benchmark_kwargs(
    recipe: dict[str, Any], model: str, mode: str, dp_size: int, concurrency: int
) -> dict[str, Any]:
    if dp_size not in (1, 2) or concurrency < dp_size:
        raise ValueError("supported DP sizes are 1 and 2, with one request per rank")
    generation = recipe["policy"]["generation"]
    kwargs = copy.deepcopy(generation["vllm_kwargs"])
    kwargs["allowed_local_media_path"] = "/lustre"
    kwargs.update(engine_kwargs(recipe, model, mode))
    kwargs["tensor_parallel_size"] = 4 // dp_size
    kwargs["pipeline_parallel_size"] = 1
    kwargs["max_num_seqs"] = (concurrency + dp_size - 1) // dp_size
    kwargs["max_num_batched_tokens"] //= dp_size
    kwargs["load_format"] = "auto"
    kwargs["enforce_eager"] = generation["vllm_cfg"]["enforce_eager"]
    kwargs["disable_log_stats"] = False
    return kwargs


def _open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_rank(
    rank: int,
    dp_size: int,
    port: int,
    recipe_path: str,
    model: str,
    mode: str,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    multiplier: int,
    barrier: Any,
) -> dict[str, Any]:
    if dp_size == 2:
        os.environ.update(
            VLLM_DP_RANK=str(rank),
            VLLM_DP_RANK_LOCAL=str(rank),
            VLLM_DP_SIZE="2",
            VLLM_DP_MASTER_IP="127.0.0.1",
            VLLM_DP_MASTER_PORT=str(port),
        )
    from vllm import LLM, SamplingParams, __version__

    if __version__ != "0.31.0":
        raise RuntimeError(f"expected vLLM 0.31.0, got {__version__}")
    recipe = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8"))
    kwargs = build_benchmark_kwargs(recipe, model, mode, dp_size, concurrency)
    llm = LLM(**kwargs)
    prompt_ids = _prompt_ids(llm.get_tokenizer(), prompt_tokens)
    prompt_hash = hashlib.sha256(
        json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(max_tokens=32, ignore_eos=True, temperature=0),
        use_tqdm=False,
    )
    count = requests_for_rank(concurrency, rank, dp_size) * multiplier
    prompts = [{"prompt_token_ids": prompt_ids} for _ in range(count)]
    if barrier is not None:
        barrier.wait()
    started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=output_tokens, ignore_eos=True, temperature=0),
        use_tqdm=False,
    )
    finished = time.perf_counter()
    lengths = [len(item.outputs[0].token_ids) for item in outputs]
    if len(lengths) != count or any(length != output_tokens for length in lengths):
        raise ValueError(f"rank {rank} output length mismatch")
    row = {
        "rank": rank,
        "requests": count,
        "prompt_tokens": sum(len(item.prompt_token_ids) for item in outputs),
        "output_tokens": sum(lengths),
        "started_s": started,
        "finished_s": finished,
        "prompt_sha256": prompt_hash,
        "vllm_version": __version__,
    }
    print(f"rank={rank} {json.dumps(row, sort_keys=True)}", flush=True)
    return row


def _worker(queue: Any, *args: Any) -> None:
    queue.put(_run_rank(*args))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dp_size == 1:
        rows = [
            _run_rank(
                0,
                1,
                0,
                str(args.recipe),
                args.model,
                args.mode,
                args.prompt_tokens,
                args.output_tokens,
                args.concurrency,
                args.request_multiplier,
                None,
            )
        ]
    else:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        barrier = ctx.Barrier(2, timeout=300)
        port = _open_port()
        processes = [
            ctx.Process(
                target=_worker,
                args=(
                    queue,
                    rank,
                    2,
                    port,
                    str(args.recipe),
                    args.model,
                    args.mode,
                    args.prompt_tokens,
                    args.output_tokens,
                    args.concurrency,
                    args.request_multiplier,
                    barrier,
                ),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + args.timeout_seconds
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        if any(process.exitcode is None for process in processes):
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join()
            raise TimeoutError("DP2 benchmark exceeded its deadline")
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"DP rank exit codes: {[p.exitcode for p in processes]}")
        rows = [queue.get(timeout=10) for _ in processes]
    if len({row["prompt_sha256"] for row in rows}) != 1:
        raise ValueError("DP rank prompts differ")
    metrics = aggregate_rank_runs(rows, gpus=4)
    expected_requests = args.concurrency * args.request_multiplier
    if metrics["requests"] != expected_requests:
        raise ValueError("aggregate request count mismatch")
    return {
        "vllm_version": rows[0]["vllm_version"],
        "mode": args.mode,
        "model": args.model,
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "prompt_sha256": rows[0]["prompt_sha256"],
        "topology": {"tp": 4 // args.dp_size, "dp": args.dp_size, "ep": 4, "gpus": 4},
        "prompt_tokens_per_request": args.prompt_tokens,
        "output_tokens_per_request": args.output_tokens,
        "concurrency": args.concurrency,
        "request_multiplier": args.request_multiplier,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("bf16", "mxfp8_modelopt"), required=True)
    parser.add_argument("--dp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--request-multiplier", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.prompt_tokens,
            args.output_tokens,
            args.concurrency,
            args.request_multiplier,
        )
        <= 0
    ):
        parser.error("lengths, concurrency and multiplier must be positive")
    if args.concurrency < args.dp_size:
        parser.error("each DP rank needs at least one request")
    if args.dry_run:
        recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
        print(
            json.dumps(
                build_benchmark_kwargs(
                    recipe, args.model, args.mode, args.dp_size, args.concurrency
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
