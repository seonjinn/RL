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
            "q": f"model.layers.{index}.self_attn.q_proj",
            "k": f"model.layers.{index}.self_attn.k_proj",
            "v": f"model.layers.{index}.self_attn.v_proj",
            "o": f"model.layers.{index}.self_attn.o_proj",
            "router": f"model.layers.{index}.mlp.gate",
            "experts": f"model.layers.{index}.mlp.experts.0.up_proj",
        }
        for family in ("q", "k", "v", "o"):
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
                "q": f"model.layers.{index}.mixer.q_proj",
                "k": f"model.layers.{index}.mixer.k_proj",
                "v": f"model.layers.{index}.mixer.v_proj",
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
                raise AssertionError(
                    f"routed experts must be MXFP8: {names['experts']}"
                )
            if not excluded(patterns, names["shared"]):
                raise AssertionError(
                    f"shared experts must stay BF16: {names['shared']}"
                )
        record(patterns, names, quantized, ignored)
    return {"quantized": quantized, "excluded": ignored}


def allowed_runtime_module(model: str, arm: str, name: str) -> bool:
    if model == "qwen30":
        return ".mlp.experts." in name or (
            arm == "qkvo"
            and any(
                f".self_attn.{projection}" in name
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        )

    if ".mixer.experts." in name:
        return True
    if arm in ("qkvo", "qkvo_mamba") and any(
        f".mixer.{projection}" in name
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ):
        return True
    return arm == "qkvo_mamba" and any(
        f".mixer.{projection}" in name for projection in ("in_proj", "out_proj")
    )


def audit_runtime_linear_modules(
    model_name: str, model: str, arm: str, patterns: list[str]
) -> dict[str, Any]:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        hf_model = AutoModel.from_config(model_config, trust_remote_code=True)

    linear_names = [
        f"model.{name}".replace("model.backbone.", "backbone.")
        for name, module in hf_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    quantized = [name for name in linear_names if not excluded(patterns, name)]
    unexpected = [
        name for name in quantized if not allowed_runtime_module(model, arm, name)
    ]
    if unexpected:
        raise AssertionError(
            "unexpected Linear modules entered MXFP8 scope: "
            + ", ".join(unexpected[:20])
        )
    return {
        "linear_module_count": len(linear_names),
        "quantized_linear_count": len(quantized),
        "quantized_linear_modules": quantized,
    }


def main() -> None:
    from transformers import AutoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen30", "nano"), required=True)
    parser.add_argument(
        "--arm", choices=("moe_only", "qkvo", "qkvo_mamba"), required=True
    )
    parser.add_argument("--runtime-model-audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.model == "qwen30" and args.arm == "qkvo_mamba":
        raise ValueError("qkvo_mamba is only valid for Nano")

    config = load_config(args.config)
    patterns = config["policy"]["generation"]["vllm_cfg"][
        "quantization_ignored_layer_kws"
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
    if args.runtime_model_audit:
        try:
            report["runtime_model_audit"] = audit_runtime_linear_modules(
                model_name, args.model, args.arm, patterns
            )
        except ImportError as error:
            if args.model != "nano" or "mamba" not in str(error).lower():
                raise
            report["runtime_model_audit"] = {
                "status": "skipped",
                "reason": str(error),
                "scope_gate": "hybrid layer-pattern audit passed",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
