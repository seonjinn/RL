"""Build matched SuperVL rollout-precision recipes from the existing pair."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml


_PROJECTION_SUFFIXES = ("in_proj", "out_proj", "qkv_proj", "o_proj")
_OTHER_LINEAR_SUFFIXES = (
    "up_proj",
    "down_proj",
    "shared_experts.*",
    "fc1_latent_proj",
    "fc2_latent_proj",
)
_BOUNDARY_PATTERN = re.compile(r"\*model\.layers\.\d+\.\*")


def _generation(config: dict[str, Any]) -> dict[str, Any]:
    return config["policy"]["generation"]


def build_configs(
    bf16_base: dict[str, Any], moe_base: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    bf16 = copy.deepcopy(bf16_base)
    moe = copy.deepcopy(moe_base)
    bf16_cfg = _generation(bf16)["vllm_cfg"]
    moe_cfg = _generation(moe)["vllm_cfg"]
    if (
        bf16_cfg["precision"] != "bfloat16"
        or bf16_cfg["is_mx"]
        or bf16_cfg["quantization_ignore_patterns"]
    ):
        raise ValueError("BF16 baseline must have no quantization exclusions")
    if moe_cfg["precision"] != "fp8" or not moe_cfg["is_mx"]:
        raise ValueError("MoE baseline must use MXFP8")

    original_ignore = moe_cfg["quantization_ignore_patterns"]
    if len(original_ignore) != len(set(original_ignore)):
        raise ValueError("duplicate quantization exclusions in MoE baseline")
    projection_patterns = {
        f"*model.layers.*.mixer.{suffix}" for suffix in _PROJECTION_SUFFIXES
    }
    other_linear_patterns = {
        f"*model.layers.*.mixer.{suffix}" for suffix in _OTHER_LINEAR_SUFFIXES
    }
    required = projection_patterns | other_linear_patterns
    missing = required - set(original_ignore)
    if missing:
        raise ValueError(f"missing required exclusions: {sorted(missing)}")
    if not any(_BOUNDARY_PATTERN.fullmatch(pattern) for pattern in original_ignore):
        raise ValueError("missing required exclusions: BF16 boundary layers")

    # The two reference recipes must differ only in rollout quantization.
    normalized_moe = copy.deepcopy(moe)
    normalized_cfg = _generation(normalized_moe)["vllm_cfg"]
    normalized_cfg["precision"] = "bfloat16"
    normalized_cfg["is_mx"] = False
    normalized_cfg["quantization_ignore_patterns"] = []
    if normalized_moe != bf16:
        raise ValueError("reference recipes differ outside rollout quantization")

    configs = {"bf16": bf16, "moe": moe}
    for name, removed in (
        ("qkvo_mamba", projection_patterns),
        ("all_decoder", projection_patterns | other_linear_patterns),
    ):
        config = copy.deepcopy(moe)
        ignore = _generation(config)["vllm_cfg"]["quantization_ignore_patterns"]
        _generation(config)["vllm_cfg"]["quantization_ignore_patterns"] = [
            pattern
            for pattern in ignore
            if pattern not in removed
            and not (name == "all_decoder" and _BOUNDARY_PATTERN.fullmatch(pattern))
        ]
        configs[name] = config

    for config in configs.values():
        kwargs = _generation(config)["vllm_kwargs"]
        kwargs["moe_backend"] = "flashinfer_trtllm"
        kwargs["kernel_config"]["moe_backend"] = "flashinfer_trtllm"
        kwargs["kernel_config"]["linear_backend"] = "flashinfer_cutedsl"
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--moe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.bf16.open(encoding="utf-8") as handle:
        bf16 = yaml.safe_load(handle)
    with args.moe.open(encoding="utf-8") as handle:
        moe = yaml.safe_load(handle)
    configs = build_configs(bf16, moe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        destination = args.output_dir / f"{name}.yaml"
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(destination)


if __name__ == "__main__":
    main()
