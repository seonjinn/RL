"""Isolate Super generation initialization; not a GRPO performance run."""

import asyncio
import json
import os
import sys

import ray

from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches


async def main() -> None:
    _apply_vllm_patches(sys.executable, extra_env_vars=["VLLM_LOGGING_LEVEL"])
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    context = ray.init(
        num_gpus=4, _temp_dir=os.environ["RAY_TMPDIR"], include_dashboard=False
    )
    os.environ["RAY_ADDRESS"] = context.address_info["gcs_address"]
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    args = AsyncEngineArgs(
        model=model,
        served_model_name=model,
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        enable_sleep_mode=True,
        enforce_eager=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        disable_log_stats=False,
        logprobs_mode="processed_logprobs",
        moe_backend="flashinfer_trtllm",
        expert_placement_strategy="linear",
        hf_overrides={},
        additional_config={"nemo_rl_refit_cache_loader_routes": True},
        worker_extension_cls=(
            "nemo_rl.models.generation.vllm.vllm_backend.VllmInternalWorkerExtension"
        ),
        seed=0,
    )
    engine = None
    try:
        engine = AsyncLLM.from_engine_args(args, stat_loggers=[])
        config = engine.vllm_config
        print(
            json.dumps(
                {
                    "initialization_completed": True,
                    "scheduler": str(config.scheduler_config),
                    "cache": str(config.cache_config),
                    "compilation": str(config.compilation_config),
                }
            ),
            flush=True,
        )
    finally:
        if engine is not None:
            engine.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
