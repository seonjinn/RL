#!/usr/bin/env python3

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
from pathlib import Path
from typing import Any

import yaml


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


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def excluded(patterns: list[str], name: str) -> bool:
    return any(
        pattern == name or pattern in name or fnmatch(name, pattern)
        for pattern in patterns
    )


def audit_qwen(patterns: list[str], qkvo: bool, layers: int) -> dict[str, Any]:
    quantized: list[str] = []
    excluded_names: list[str] = []
    for index in range(layers):
        names = {
            "qkv": f"model.layers.{index}.self_attn.qkv_proj",
            "o": f"model.layers.{index}.self_attn.o_proj",
            "router": f"model.layers.{index}.mlp.gate",
            "experts": f"model.layers.{index}.mlp.experts",
        }
        for family in ("qkv", "o"):
            if excluded(patterns, names[family]) == qkvo:
                raise AssertionError(f"unexpected QKVO scope: {names[family]}")
        if not excluded(patterns, names["router"]):
            raise AssertionError(f"router must stay BF16: {names['router']}")
        if excluded(patterns, names["experts"]):
            raise AssertionError(f"routed experts must be MXFP8: {names['experts']}")
        quantized.extend(
            name for name in names.values() if not excluded(patterns, name)
        )
        excluded_names.extend(
            name for name in names.values() if excluded(patterns, name)
        )
    return {"quantized": quantized, "excluded": excluded_names}


def audit_nano(
    patterns: list[str], qkvo: bool, pattern: str
) -> dict[str, Any]:
    quantized: list[str] = []
    excluded_names: list[str] = []
    for index, layer_type in enumerate(pattern):
        names: dict[str, str]
        if layer_type == "*":
            names = {
                "qkv": f"model.layers.{index}.mixer.qkv_proj",
                "o": f"model.layers.{index}.mixer.o_proj",
            }
            for name in names.values():
                if excluded(patterns, name) == qkvo:
                    raise AssertionError(f"unexpected Nano QKVO scope: {name}")
        elif layer_type == "M":
            names = {
                family: f"model.layers.{index}.mixer.{family}"
                for family in ("in_proj", "out_proj", "up_proj", "down_proj")
            }
            if any(not excluded(patterns, name) for name in names.values()):
                raise AssertionError(f"Mamba projection entered MXFP8 scope: {names}")
        else:
            names = {
                "router": f"model.layers.{index}.mixer.gate",
                "experts": f"model.layers.{index}.mixer.experts",
            }
            if not excluded(patterns, names["router"]):
                raise AssertionError(f"router must stay BF16: {names['router']}")
            if excluded(patterns, names["experts"]):
                raise AssertionError(f"routed experts must be MXFP8: {names['experts']}")
        quantized.extend(
            name for name in names.values() if not excluded(patterns, name)
        )
        excluded_names.extend(
            name for name in names.values() if excluded(patterns, name)
        )
    return {"quantized": quantized, "excluded": excluded_names}


def main() -> None:
    from transformers import AutoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen30", "nano"), required=True)
    parser.add_argument("--arm", choices=("moe_only", "qkvo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    vllm_cfg = config["policy"]["generation"]["vllm_cfg"]
    patterns = vllm_cfg["quantization_ignore_patterns"]
    if not excluded(patterns, "lm_head"):
        raise AssertionError("lm_head must stay outside MXFP8 scope")

    model_name = config["policy"]["model_name"]
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if args.model == "qwen30":
        details = audit_qwen(
            patterns, args.arm == "qkvo", int(model_config.num_hidden_layers)
        )
        attention_layers = int(model_config.num_hidden_layers)
    else:
        pattern = str(model_config.hybrid_override_pattern)
        details = audit_nano(patterns, args.arm == "qkvo", pattern)
        attention_layers = pattern.count("*")

    report = {
        "status": "pass",
        "model": args.model,
        "arm": args.arm,
        "model_name": model_name,
        "attention_layers": attention_layers,
        "ignore_patterns": patterns,
        "quantized_family_count": len(details["quantized"]),
        "excluded_family_count": len(details["excluded"]),
        "quantized_families": details["quantized"],
        "excluded_families": details["excluded"],
        "non_linear_exclusions": ["embedding"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
