"""Initialization-only BF16 probe with dummy weights; NOT an accuracy benchmark."""

import argparse
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
    args = parser.parse_args()

    from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches

    _apply_vllm_patches(sys.executable)
    if args.profile_before_graphs:
        source = (
            Path(importlib.util.find_spec("vllm").origin).parent
            / "v1/worker/gpu_worker.py"
        )
        patch = Path(__file__).with_name("vllm-profile-before-graphs.patch")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        subprocess.run(
            ["patch", "--batch", "--fuzz=0", str(source), str(patch)], check=True
        )
        print(
            "PROFILE_BOUNDARY_PATCH",
            before,
            hashlib.sha256(source.read_bytes()).hexdigest(),
            flush=True,
        )
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
    print("INIT_PROBE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config), stat_loggers=[])
    print("INIT_PROBE_PASS", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
