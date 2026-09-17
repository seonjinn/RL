"""Initialization-only BF16 probe with dummy weights; NOT an accuracy benchmark."""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-mode", choices=("true", "false"), required=True)
    args = parser.parse_args()

    from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches

    _apply_vllm_patches(sys.executable)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    config = dict(
        model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        load_format="dummy",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.7,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        seed=3072,
        enable_sleep_mode=args.sleep_mode == "true",
    )
    print("INIT_PROBE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config), stat_loggers=[])
    print("INIT_PROBE_PASS", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
