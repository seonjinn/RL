"""Check SuperVL3.5 text generation with the vLLM 0.31 container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def engine_kwargs(recipe: dict[str, Any], model: str, mode: str) -> dict[str, Any]:
    generation = recipe["policy"]["generation"]
    cfg = generation["vllm_cfg"]
    vllm_kwargs = generation["vllm_kwargs"]
    kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": cfg["tensor_parallel_size"],
        "enable_expert_parallel": cfg["expert_parallel_size"] > 1,
        "max_model_len": cfg["max_model_len"],
        "gpu_memory_utilization": cfg["gpu_memory_utilization"],
        "attention_backend": vllm_kwargs["attention_backend"],
        "moe_backend": vllm_kwargs["moe_backend"],
        "kernel_config": vllm_kwargs["kernel_config"],
        "enable_chunked_prefill": vllm_kwargs["enable_chunked_prefill"],
        "enable_prefix_caching": True,
        "max_num_seqs": 16,
        "max_num_batched_tokens": vllm_kwargs["max_num_batched_tokens"],
        "skip_mm_profiling": vllm_kwargs["skip_mm_profiling"],
        "hf_overrides": vllm_kwargs["hf_overrides"].copy(),
    }
    if mode == "mxfp8_modelopt":
        ignore = cfg["quantization_ignore_patterns"]
        if not ignore or cfg["precision"] != "fp8" or not cfg["is_mx"]:
            raise ValueError("MXFP8 mode requires the scoped FP8 recipe")
        kwargs["quantization"] = "fp8"
        kwargs["hf_overrides"]["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignored_layers": ["lm_head"],
            "ignore": list(dict.fromkeys([*ignore, "lm_head"])),
        }
    elif mode != "bf16":
        raise ValueError(f"unsupported mode: {mode}")
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("bf16", "mxfp8_modelopt"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    kwargs = engine_kwargs(recipe, args.model, args.mode)
    if args.dry_run:
        print(json.dumps(kwargs, indent=2, sort_keys=True))
        return

    from vllm import LLM, SamplingParams, __version__

    if __version__ != "0.31.0":
        raise RuntimeError(f"expected vLLM 0.31.0, got {__version__}")
    model = LLM(**kwargs)
    outputs = model.generate(
        [{"prompt": "Hello"}],
        SamplingParams(max_tokens=16, ignore_eos=True, temperature=0),
        use_tqdm=False,
    )
    lengths = [len(output.outputs[0].token_ids) for output in outputs]
    if lengths != [16]:
        raise RuntimeError(f"unexpected generated token counts: {lengths}")
    result = {
        "vllm_version": __version__,
        "mode": args.mode,
        "model": args.model,
        "generated_tokens": lengths,
        "tensor_parallel_size": kwargs["tensor_parallel_size"],
        "moe_backend": kwargs["moe_backend"],
        "attention_backend": kwargs["attention_backend"],
        "linear_backend": kwargs["kernel_config"]["linear_backend"],
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
