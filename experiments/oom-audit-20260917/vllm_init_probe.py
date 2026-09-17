"""Initialization-only BF16 probe with dummy weights; NOT an accuracy benchmark."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-mode", choices=("true", "false"), required=True)
    parser.add_argument("--profile-before-graphs", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument("--inspect-cumem", action="store_true")
    parser.add_argument("--sleep-cycles", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--moe-backend", choices=("auto", "triton", "flashinfer_cutlass"), default="auto"
    )
    args = parser.parse_args()
    if args.patch_only and not args.profile_before_graphs:
        parser.error("--patch-only requires --profile-before-graphs")

    from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches

    _apply_vllm_patches(sys.executable)
    if args.profile_before_graphs:
        source = (
            Path(importlib.util.find_spec("vllm").origin).parent
            / "v1/worker/gpu_worker.py"
        )
        patch = Path(__file__).with_name("vllm-profile-before-graphs.patch")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        original_sha = "7e00284da7b453154af47300630483ed7ea5a5d79e724c5ee61d4a24edaf930e"
        patched_sha = "d255de12f2a9f60cb94030349287ca06af1ef464e8cae52beee110bda4228f5b"
        if before not in (original_sha, patched_sha):
            raise RuntimeError(f"Unexpected vLLM source hash: {before}")
        if before == original_sha:
            subprocess.run(
                ["patch", "--batch", "--fuzz=0", str(source), str(patch)], check=True
            )
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        if after != patched_sha:
            raise RuntimeError(f"Unexpected patched vLLM hash: {after}")
        print(
            "PROFILE_BOUNDARY_PATCH",
            before,
            hashlib.sha256(source.read_bytes()).hexdigest(),
            flush=True,
        )
    if args.patch_only:
        return
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    config = {
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "gpu_memory_utilization": 0.7,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_model_len": 8192,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 256,
        "seed": 3072,
        "enable_sleep_mode": args.sleep_mode == "true",
    }
    if args.moe_backend != "auto":
        config["moe_backend"] = args.moe_backend
    print("INIT_PROBE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    if args.inspect_cumem:
        config["worker_cls"] = "cumem_probe_worker.ProbeWorker"
    if args.sleep_cycles:
        if args.sleep_mode != "true":
            parser.error("sleep cycles require sleep mode")

        async def exercise_sleep() -> None:
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config), stat_loggers=[])
            try:
                for cycle in range(args.sleep_cycles):
                    await engine.sleep(level=1)
                    if not await engine.is_sleeping():
                        raise RuntimeError("Engine did not enter sleep")
                    await engine.wake_up()
                    if await engine.is_sleeping():
                        raise RuntimeError("Engine did not wake")
                    print(f"SLEEP_CYCLE_PASS {cycle + 1}", flush=True)
                print("INIT_PROBE_PASS", flush=True)
            finally:
                engine.shutdown()

        asyncio.run(exercise_sleep())
        return
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config), stat_loggers=[])
    print("INIT_PROBE_PASS", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
