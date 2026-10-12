"""Measure one SuperVL vLLM replica without training, Gym, or refit timing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def build_engine_kwargs(
    recipe: dict[str, Any], fp8_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    generation = recipe["policy"]["generation"]
    cfg = generation["vllm_cfg"]
    kwargs = copy.deepcopy(generation["vllm_kwargs"])
    if cfg["precision"] == "fp8":
        if not cfg.get("is_mx") or fp8_kwargs is None:
            raise ValueError("MXFP8 arm requires NeMo-RL FP8 engine kwargs")
        fp8_kwargs = copy.deepcopy(fp8_kwargs)
        fp8_hf_overrides = fp8_kwargs.pop("hf_overrides", {})
        existing_hf_overrides = kwargs.get("hf_overrides", {})
        if "quantization_config" in existing_hf_overrides:
            raise ValueError("unexpected pre-existing quantization_config")
        kwargs.update(fp8_kwargs)
        kwargs["hf_overrides"] = {**existing_hf_overrides, **fp8_hf_overrides}
    elif cfg["precision"] == "bfloat16":
        if fp8_kwargs is not None:
            raise ValueError("BF16 arm must not have FP8 engine kwargs")
    else:
        raise ValueError(f"unsupported rollout precision: {cfg['precision']}")

    kwargs.update(
        model=generation.get("model_name") or recipe["policy"]["model_name"],
        load_format="auto",
        dtype=recipe["policy"]["precision"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        pipeline_parallel_size=cfg["pipeline_parallel_size"],
        enable_expert_parallel=cfg["expert_parallel_size"] > 1,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=cfg["enable_prefix_caching"],
        enforce_eager=cfg["enforce_eager"],
        max_model_len=cfg["max_model_len"],
        trust_remote_code=True,
        disable_log_stats=False,
    )
    return kwargs


def summarize_outputs(
    outputs: list[dict[str, int]],
    elapsed_s: float,
    expected_output_tokens: int,
    gpus: int,
) -> dict[str, float | int]:
    if elapsed_s <= 0 or gpus <= 0:
        raise ValueError("elapsed time and GPU count must be positive")
    if any(row["output_tokens"] != expected_output_tokens for row in outputs):
        raise ValueError("output length differs from the fixed-token benchmark")
    prompt_tokens = sum(row["prompt_tokens"] for row in outputs)
    output_tokens = sum(row["output_tokens"] for row in outputs)
    throughput = output_tokens / elapsed_s
    return {
        "requests": len(outputs),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "output_tokens_per_second": throughput,
        "output_tokens_per_second_per_gpu": throughput / gpus,
    }


def _prompt_ids(tokenizer: Any, target_length: int) -> list[int]:
    body = "Summarize the sequence and explain the reasoning in detail. " * 1000
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if len(chat) < target_length:
        raise ValueError("synthetic prompt is shorter than the requested length")
    return chat[:target_length]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=10500)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    if args.prompt_tokens <= 0 or args.trials <= 0:
        parser.error("prompt-tokens and trials must be positive")

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from omegaconf import OmegaConf

    full_recipe = OmegaConf.load(args.config)
    policy = full_recipe.policy
    generation_cfg = policy.generation
    recipe = {
        "policy": {
            "precision": policy.precision,
            "model_name": policy.model_name,
            "generation": {
                "model_name": generation_cfg.model_name,
                "vllm_cfg": OmegaConf.to_container(
                    generation_cfg.vllm_cfg, resolve=True
                ),
                "vllm_kwargs": OmegaConf.to_container(
                    generation_cfg.vllm_kwargs, resolve=True
                ),
            },
        }
    }
    generation = recipe["policy"]["generation"]
    cfg = generation["vllm_cfg"]
    fp8_kwargs = None
    if cfg["precision"] == "fp8":
        from nemo_rl.models.generation.vllm.quantization.fp8 import init_fp8

        standalone_cfg = {**cfg, "async_engine": False}
        fp8_kwargs = init_fp8(
            standalone_cfg,
            generation.get("model_name") or recipe["policy"]["model_name"],
            cfg["tensor_parallel_size"] * cfg["pipeline_parallel_size"],
        )

    engine_kwargs = build_engine_kwargs(recipe, fp8_kwargs)
    llm = LLM(**engine_kwargs)
    prompt_ids = _prompt_ids(llm.get_tokenizer(), args.prompt_tokens)
    prompt_hash = hashlib.sha256(
        json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    print(
        f"vLLM={vllm_version} precision={cfg['precision']} prompt_sha256={prompt_hash}",
        flush=True,
    )

    llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(max_tokens=32, ignore_eos=True, temperature=0),
        use_tqdm=False,
    )

    runs = []
    for batch_size, output_length in ((1, 1), (16, 512), (64, 512), (64, 2048)):
        for trial in range(args.trials):
            prompts = [{"prompt_token_ids": prompt_ids} for _ in range(batch_size)]
            params = SamplingParams(
                max_tokens=output_length,
                ignore_eos=True,
                temperature=0,
            )
            started = time.perf_counter()
            generated = llm.generate(prompts, params, use_tqdm=False)
            elapsed_s = time.perf_counter() - started
            rows = [
                {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                }
                for output in generated
            ]
            metrics = summarize_outputs(
                rows,
                elapsed_s=elapsed_s,
                expected_output_tokens=output_length,
                gpus=cfg["tensor_parallel_size"] * cfg["pipeline_parallel_size"],
            )
            metrics.update(
                batch_size=batch_size, output_length=output_length, trial=trial
            )
            runs.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "vllm_version": vllm_version,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "model": engine_kwargs["model"],
                "load_format": "auto",
                "prompt_sha256": prompt_hash,
                "prompt_tokens": len(prompt_ids),
                "scoped_mxfp8_ignore": cfg.get("quantization_ignore_patterns", []),
                "moe_backend": engine_kwargs["kernel_config"]["moe_backend"],
                "linear_backend": engine_kwargs["kernel_config"]["linear_backend"],
                "runs": runs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
