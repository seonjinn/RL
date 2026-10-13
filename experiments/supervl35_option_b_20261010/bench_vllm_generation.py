"""Measure one SuperVL vLLM replica without training, Gym, or refit timing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_engine_kwargs(
    recipe: dict[str, Any],
    fp8_kwargs: dict[str, Any] | None = None,
    attention_backend: str | None = None,
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
    if attention_backend is not None:
        kwargs["attention_backend"] = attention_backend
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


def benchmark_cases(
    concurrencies: str | None, output_tokens: int
) -> list[tuple[int, int]]:
    if concurrencies is None:
        return [(1, 1), (16, 512), (64, 512), (64, 2048)]
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    try:
        sizes = [int(value) for value in concurrencies.split(",")]
    except ValueError as exc:
        raise ValueError("concurrencies must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("concurrencies must be positive")
    if any(left >= right for left, right in zip(sizes, sizes[1:])):
        raise ValueError("concurrencies must be strictly increasing")
    return [(size, output_tokens) for size in sizes]


def build_profile_config(
    trace_dir: Path, delay_iterations: int, max_iterations: int
) -> dict[str, Any]:
    if not trace_dir.is_absolute():
        raise ValueError("profile trace directory must be absolute")
    if delay_iterations < 0 or max_iterations <= 0:
        raise ValueError("profile delay must be nonnegative and max iterations positive")
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_use_gzip": True,
        "torch_profiler_with_stack": False,
        "ignore_frontend": True,
        "delay_iterations": delay_iterations,
        "max_iterations": max_iterations,
    }


def apply_ntrace_profile(
    engine_kwargs: dict[str, Any],
    trace_dir: Path,
    delay_iterations: int,
    max_iterations: int,
) -> None:
    # vLLM uses the profiler config to enable start/stop RPCs; the custom
    # worker handles them with ntrace instead of instantiating TorchProfiler.
    engine_kwargs["profiler_config"] = build_profile_config(
        trace_dir, delay_iterations, max_iterations
    )
    engine_kwargs["worker_cls"] = (
        "ntrace_vllm_worker.NtraceWorker"
    )


def generate_case(
    llm: Any,
    prompts: list[dict[str, Any]],
    params: Any,
    profile_prefix: str | None = None,
) -> tuple[list[Any], float]:
    if profile_prefix is not None:
        llm.start_profile(profile_prefix)
    try:
        started = time.perf_counter()
        generated = llm.generate(prompts, params, use_tqdm=False)
        elapsed_s = time.perf_counter() - started
    finally:
        if profile_prefix is not None:
            llm.stop_profile()
    return generated, elapsed_s


def _prompt_ids(tokenizer: Any, target_length: int) -> list[int]:
    phrase = "Summarize the sequence and explain the reasoning in detail. "
    repetitions = 1000
    for _ in range(16):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": phrase * repetitions}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if (
            isinstance(encoded, (list, tuple))
            and len(encoded) == 1
            and isinstance(encoded[0], (list, tuple))
        ):
            encoded = encoded[0]
        if not isinstance(encoded, (list, tuple)) or not all(
            type(token) is int for token in encoded
        ):
            raise TypeError("chat template must return a flat sequence of token IDs")
        if len(encoded) >= target_length:
            return list(encoded[:target_length])
        repetitions *= 2
    raise ValueError("tokenizer could not produce the requested prompt length")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=10500)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--concurrencies", help="comma-separated outstanding request counts")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--high-concurrency-trials", type=int, default=1)
    parser.add_argument(
        "--attention-backend", choices=("FLASHINFER", "TRITON_ATTN")
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-kind", choices=("torch", "ntrace"), default="torch")
    parser.add_argument("--profile-delay-iterations", type=int, default=0)
    parser.add_argument("--profile-max-iterations", type=int, default=32)
    args = parser.parse_args()
    if args.prompt_tokens <= 0 or args.trials <= 0 or args.high_concurrency_trials <= 0:
        parser.error("prompt-tokens and trial counts must be positive")
    cases = benchmark_cases(args.concurrencies, args.output_tokens)
    if args.profile_dir is not None and (
        len(cases) != 1 or args.trials != 1 or args.high_concurrency_trials != 1
    ):
        parser.error("profiling requires one concurrency and one trial")

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

    engine_kwargs = build_engine_kwargs(recipe, fp8_kwargs, args.attention_backend)
    if args.profile_dir is not None:
        if args.profile_kind == "ntrace":
            apply_ntrace_profile(
                engine_kwargs,
                args.profile_dir,
                args.profile_delay_iterations,
                args.profile_max_iterations,
            )
        else:
            engine_kwargs["profiler_config"] = build_profile_config(
                args.profile_dir,
                args.profile_delay_iterations,
                args.profile_max_iterations,
            )
        args.profile_dir.mkdir(parents=True, exist_ok=True)
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
    for batch_size, output_length in cases:
        trials = args.high_concurrency_trials if batch_size > 128 else args.trials
        for trial in range(trials):
            prompts = [{"prompt_token_ids": prompt_ids} for _ in range(batch_size)]
            params = SamplingParams(
                max_tokens=output_length,
                ignore_eos=True,
                temperature=0,
            )
            profile_prefix = (
                f"isl{args.prompt_tokens}_osl{output_length}_c{batch_size}"
                if args.profile_dir is not None
                else None
            )
            generated, elapsed_s = generate_case(
                llm, prompts, params, profile_prefix
            )
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
                "concurrencies": [size for size, _ in cases],
                "max_num_seqs": engine_kwargs["max_num_seqs"],
                "trials": args.trials,
                "high_concurrency_trials": args.high_concurrency_trials,
                "scoped_mxfp8_ignore": cfg.get("quantization_ignore_patterns", []),
                "moe_backend": engine_kwargs["kernel_config"]["moe_backend"],
                "linear_backend": engine_kwargs["kernel_config"]["linear_backend"],
                "attention_backend": engine_kwargs.get("attention_backend"),
                "profile": (
                    {
                        "trace_dir": str(args.profile_dir),
                        "kind": args.profile_kind,
                        "delay_iterations": args.profile_delay_iterations,
                        "max_iterations": args.profile_max_iterations,
                    }
                    if args.profile_dir is not None
                    else None
                ),
                "runs": runs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
