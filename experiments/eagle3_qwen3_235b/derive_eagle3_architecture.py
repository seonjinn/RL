#!/usr/bin/env python3
"""Derive ModelOpt Eagle3 architecture overrides from a verifier config.json.

This is the reusable step for moving from the Qwen3-235B-specific wrapper to an
arbitrary LLaMA-like verifier model. It mirrors ModelOpt's EAGLE-3 default aux
layer rule:

    sorted({1, max(0, num_layers // 2 - 1), max(0, num_layers - 4)})

The generated env file can be sourced before the existing train wrappers.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


ARCH_KEYS_FOR_DOTLIST = [
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling.rope_type",
    "rope_scaling.rope_theta",
    "use_aux_hidden_state",
    "eagle_aux_hidden_state_layer_ids",
    "use_input_layernorm_in_first_layer",
    "use_last_layernorm",
    "use_mtp_layernorm",
    "has_lm_head",
]

ENV_KEY_MAP = {
    "num_attention_heads": "NUM_ATTENTION_HEADS",
    "num_key_value_heads": "NUM_KEY_VALUE_HEADS",
    "intermediate_size": "INTERMEDIATE_SIZE",
    "head_dim": "HEAD_DIM",
    "rms_norm_eps": "RMS_NORM_EPS",
    "rope_theta": "ROPE_THETA",
    "eagle_aux_hidden_state_layer_ids": "AUX_LAYERS",
    "use_aux_hidden_state": "USE_AUX_HIDDEN_STATE",
    "use_input_layernorm_in_first_layer": "USE_INPUT_LAYERNORM_IN_FIRST_LAYER",
    "use_last_layernorm": "USE_LAST_LAYERNORM",
}

LLAMA_LIKE_MODEL_TYPES = {
    "baichuan",
    "deepseek_v2",
    "deepseek_v3",
    "gemma",
    "gemma2",
    "gemma3",
    "granite",
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "yi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verifier-config",
        type=Path,
        required=True,
        help="Path to verifier config.json or to a directory containing config.json.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--env-out", type=Path, default=None)
    parser.add_argument("--dotlist-out", type=Path, default=None)
    parser.add_argument(
        "--aux-layers",
        default=None,
        help="Comma-separated 0-based aux layer ids. Defaults to ModelOpt Eagle3 rule.",
    )
    parser.add_argument(
        "--eagle-decoder-type",
        choices=("auto", "llama", "kimik2"),
        default="auto",
        help="ModelOpt Eagle decoder type. Auto maps Kimi/K2 model types to kimik2, otherwise llama.",
    )
    parser.add_argument(
        "--copy-rope-scaling",
        action="store_true",
        help="Copy verifier rope_scaling object instead of using ModelOpt's default rope_type=default form.",
    )
    parser.add_argument(
        "--include-hidden-size",
        action="store_true",
        help="Include hidden_size/vocab_size/max_position_embeddings in the JSON reference for auditing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def default_aux_layers(num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_layers}")
    return sorted({1, max(0, num_layers // 2 - 1), max(0, num_layers - 4)})


def parse_aux_layers(text: str | None, num_layers: int) -> list[int]:
    if text is None:
        return default_aux_layers(num_layers)
    layers = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not layers:
        raise ValueError("--aux-layers produced an empty list")
    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"aux layers out of range for {num_layers} layers: {invalid}")
    return layers


def require_int(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"verifier config missing required field: {key}")
    return int(value)


def infer_decoder_type(model_type: str, requested: str) -> str:
    if requested != "auto":
        return requested
    normalized = model_type.lower()
    if "kimi" in normalized or "k2" in normalized:
        return "kimik2"
    return "llama"


def get_rope_theta(cfg: dict[str, Any]) -> int | float:
    if cfg.get("rope_theta") is not None:
        return cfg["rope_theta"]
    rope_scaling = cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_theta") is not None:
        return rope_scaling["rope_theta"]
    return 10000


def get_rope_type(cfg: dict[str, Any]) -> str:
    rope_scaling = cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        return str(rope_scaling.get("rope_type") or rope_scaling.get("type") or "default")
    return "default"


def derive_architecture(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    num_layers = require_int(cfg, "num_hidden_layers")
    num_attention_heads = require_int(cfg, "num_attention_heads")
    hidden_size = require_int(cfg, "hidden_size")
    head_dim = int(cfg.get("head_dim") or hidden_size // num_attention_heads)
    rope_theta = get_rope_theta(cfg)

    rope_scaling: dict[str, Any]
    if args.copy_rope_scaling and isinstance(cfg.get("rope_scaling"), dict):
        rope_scaling = dict(cfg["rope_scaling"])
        rope_scaling.setdefault("rope_theta", rope_theta)
    else:
        rope_scaling = {"rope_type": "default", "rope_theta": rope_theta}

    arch: dict[str, Any] = {
        "num_hidden_layers": 1,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": int(cfg.get("num_key_value_heads") or num_attention_heads),
        "intermediate_size": require_int(cfg, "intermediate_size"),
        "head_dim": head_dim,
        "rms_norm_eps": cfg.get("rms_norm_eps", cfg.get("layer_norm_epsilon", 1e-5)),
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "use_aux_hidden_state": True,
        "eagle_aux_hidden_state_layer_ids": parse_aux_layers(args.aux_layers, num_layers),
        "use_input_layernorm_in_first_layer": True,
        "use_last_layernorm": True,
        "use_mtp_layernorm": False,
        "has_lm_head": False,
    }
    if cfg.get("hidden_act") is not None:
        arch["hidden_act"] = cfg["hidden_act"]
    if cfg.get("attention_bias") is not None:
        arch["attention_bias"] = cfg["attention_bias"]
    if cfg.get("mlp_bias") is not None:
        arch["mlp_bias"] = cfg["mlp_bias"]
    if cfg.get("attention_dropout") is not None:
        arch["attention_dropout"] = cfg["attention_dropout"]
    return arch


def nested_get(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def scalar_for_dotlist(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def render_dotlist(arch: dict[str, Any], decoder_type: str) -> str:
    lines = [f"eagle.eagle_decoder_type={decoder_type}"]
    for key in ARCH_KEYS_FOR_DOTLIST:
        value = nested_get(arch, key)
        if value is None:
            continue
        lines.append(f"eagle.eagle_architecture_config.{key}={scalar_for_dotlist(value)}")
    return "\n".join(lines) + "\n"


def render_env(arch: dict[str, Any], decoder_type: str, verifier_cfg: dict[str, Any]) -> str:
    train_aux_layers = scalar_for_dotlist(arch["eagle_aux_hidden_state_layer_ids"])
    dump_aux_layers = ",".join(str(item) for item in arch["eagle_aux_hidden_state_layer_ids"])
    lines = [
        "# shellcheck shell=bash",
        "# Source this before modelopt_qwen3_235b_offline_train.sh or online_train.sh.",
        f"EAGLE_DECODER_TYPE={shlex.quote(decoder_type)}",
        f"EAGLE_TRAIN_AUX_LAYERS={shlex.quote(train_aux_layers)}",
        f"EAGLE_DUMP_AUX_LAYERS={shlex.quote(dump_aux_layers)}",
        f"EAGLE_AUX_COUNT={len(arch['eagle_aux_hidden_state_layer_ids'])}",
    ]
    if verifier_cfg.get("hidden_size") is not None:
        lines.append(f"EAGLE_VERIFIER_HIDDEN_SIZE={int(verifier_cfg['hidden_size'])}")
        lines.append(f"EXPECTED_HIDDEN_SIZE={int(verifier_cfg['hidden_size'])}")
    lines.append(f"EXPECTED_AUX_COUNT={len(arch['eagle_aux_hidden_state_layer_ids'])}")
    for arch_key, env_key in ENV_KEY_MAP.items():
        value = arch[arch_key]
        lines.append(f"{env_key}={shlex.quote(scalar_for_dotlist(value))}")
    return "\n".join(lines) + "\n"


def write_text(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_json(args.verifier_config)
    model_type = str(cfg.get("model_type") or "")
    decoder_type = infer_decoder_type(model_type, args.eagle_decoder_type)
    arch = derive_architecture(cfg, args)

    warnings: list[str] = []
    if model_type and decoder_type == "llama" and model_type.lower() not in LLAMA_LIKE_MODEL_TYPES:
        warnings.append(
            f"model_type={model_type!r} is not in the known LLaMA-like allowlist; "
            "verify ModelOpt supports this verifier before training."
        )
    if get_rope_type(cfg) != "default" and not args.copy_rope_scaling:
        warnings.append(
            "verifier has non-default rope_scaling; generated training arch keeps "
            "rope_type=default and carries only rope_theta, matching the Qwen3 path."
        )

    payload: dict[str, Any] = {
        "source": str(args.verifier_config),
        "model_type": model_type or None,
        "eagle_decoder_type": decoder_type,
        "aux_layer_rule": "sorted({1, max(0, num_hidden_layers // 2 - 1), max(0, num_hidden_layers - 4)})",
        "aux_layer_indexing": "0-based transformer layer ids",
        "verifier_summary": {
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "hidden_size": cfg.get("hidden_size"),
            "num_attention_heads": cfg.get("num_attention_heads"),
            "num_key_value_heads": cfg.get("num_key_value_heads"),
            "intermediate_size": cfg.get("intermediate_size"),
            "rope_theta": get_rope_theta(cfg),
            "rope_scaling_type": get_rope_type(cfg),
        },
        "eagle_architecture_config": arch,
        "warnings": warnings,
    }
    if args.include_hidden_size:
        payload["verifier_identity_fields"] = {
            "hidden_size": cfg.get("hidden_size"),
            "vocab_size": cfg.get("vocab_size"),
            "max_position_embeddings": cfg.get("max_position_embeddings"),
        }

    write_json(args.json_out, payload)
    write_text(args.env_out, render_env(arch, decoder_type, cfg))
    write_text(args.dotlist_out, render_dotlist(arch, decoder_type))

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.env_out:
        print(f"Wrote env overrides: {args.env_out}")
    if args.dotlist_out:
        print(f"Wrote dotlist overrides: {args.dotlist_out}")
    if args.json_out:
        print(f"Wrote architecture reference: {args.json_out}")
    if warnings:
        for warning in warnings:
            print(f"WARN {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
