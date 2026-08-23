#!/usr/bin/env python3

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
from pathlib import Path
from typing import Any

import yaml


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic defaults chain at {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"expected mapping in {path}")
    default = config.get("defaults")
    if not isinstance(default, str):
        return config
    parent = load_config(path.parent / default, seen | {path})
    return merge(parent, config)


def excluded(patterns: list[str], name: str) -> bool:
    return any(
        pattern == name or pattern in name or fnmatch(name, pattern)
        for pattern in patterns
    )


def record(
    patterns: list[str], names: dict[str, str], quantized: list[str], ignored: list[str]
) -> None:
    for name in names.values():
        (ignored if excluded(patterns, name) else quantized).append(name)


def audit_qwen(patterns: list[str], arm: str, layers: int) -> dict[str, list[str]]:
    quantized: list[str] = []
    ignored: list[str] = []
    expect_qkvo = arm == "qkvo"
    for index in range(layers):
        names = {
            "qkv": f"model.layers.{index}.self_attn.qkv_proj",
            "o": f"model.layers.{index}.self_attn.o_proj",
            "router": f"model.layers.{index}.mlp.gate",
            "experts": f"model.layers.{index}.mlp.experts",
        }
        for family in ("qkv", "o"):
            if excluded(patterns, names[family]) == expect_qkvo:
                raise AssertionError(f"unexpected QKVO scope: {names[family]}")
        if not excluded(patterns, names["router"]):
            raise AssertionError(f"router must stay BF16: {names['router']}")
        if excluded(patterns, names["experts"]):
            raise AssertionError(f"routed experts must be MXFP8: {names['experts']}")
        record(patterns, names, quantized, ignored)
    return {"quantized": quantized, "excluded": ignored}


def audit_nano(patterns: list[str], arm: str, pattern: str) -> dict[str, list[str]]:
    quantized: list[str] = []
    ignored: list[str] = []
    expect_qkvo = arm in ("qkvo", "qkvo_mamba")
    expect_mamba = arm == "qkvo_mamba"
    for index, layer_type in enumerate(pattern):
        if layer_type == "*":
            names = {
                "qkv": f"model.layers.{index}.mixer.qkv_proj",
                "o": f"model.layers.{index}.mixer.o_proj",
            }
            for name in names.values():
                if excluded(patterns, name) == expect_qkvo:
                    raise AssertionError(f"unexpected Nano QKVO scope: {name}")
        elif layer_type == "M":
            names = {
                "in": f"model.layers.{index}.mixer.in_proj",
                "out": f"model.layers.{index}.mixer.out_proj",
            }
            for name in names.values():
                if excluded(patterns, name) == expect_mamba:
                    raise AssertionError(f"unexpected Nano Mamba scope: {name}")
        else:
            names = {
                "router": f"model.layers.{index}.mixer.gate",
                "experts": f"model.layers.{index}.mixer.experts.0.up_proj",
                "shared": f"model.layers.{index}.mixer.shared_experts.up_proj",
            }
            if not excluded(patterns, names["router"]):
                raise AssertionError(f"router must stay BF16: {names['router']}")
            if excluded(patterns, names["experts"]):
                raise AssertionError(f"routed experts must be MXFP8: {names['experts']}")
            if not excluded(patterns, names["shared"]):
                raise AssertionError(f"shared experts must stay BF16: {names['shared']}")
        record(patterns, names, quantized, ignored)
    return {"quantized": quantized, "excluded": ignored}


def main() -> None:
    from transformers import AutoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen30", "nano"), required=True)
    parser.add_argument(
        "--arm", choices=("moe_only", "qkvo", "qkvo_mamba"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.model == "qwen30" and args.arm == "qkvo_mamba":
        raise ValueError("qkvo_mamba is only valid for Nano")

    config = load_config(args.config)
    patterns = config["policy"]["generation"]["vllm_cfg"][
        "quantization_ignore_patterns"
    ]
    if not excluded(patterns, "lm_head"):
        raise AssertionError("lm_head must stay outside MXFP8 scope")

    model_name = config["policy"]["model_name"]
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if args.model == "qwen30":
        details = audit_qwen(patterns, args.arm, int(model_config.num_hidden_layers))
        attention_layers = int(model_config.num_hidden_layers)
        mamba_layers = 0
    else:
        layer_pattern = str(model_config.hybrid_override_pattern)
        details = audit_nano(patterns, args.arm, layer_pattern)
        attention_layers = layer_pattern.count("*")
        mamba_layers = layer_pattern.count("M")

    report = {
        "status": "pass",
        "model": args.model,
        "arm": args.arm,
        "model_name": model_name,
        "attention_layers": attention_layers,
        "mamba_layers": mamba_layers,
        "ignore_patterns": patterns,
        "quantized_family_count": len(details["quantized"]),
        "excluded_family_count": len(details["excluded"]),
        "quantized_families": details["quantized"],
        "excluded_families": details["excluded"],
        "non_linear_exclusions": ["embedding", "conv1d", "state_parameters"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
