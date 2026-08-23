"""Fail closed when a DFlash or DSpark checkpoint schema is not exact."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def state_dict_keys(path: Path) -> set[str]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return set(header) - {"__metadata__"}


def expected_keys(variant: str) -> set[str]:
    keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    for layer in range(5):
        keys.update(
            {
                f"layers.{layer}.input_layernorm.weight",
                f"layers.{layer}.post_attention_layernorm.weight",
                f"layers.{layer}.self_attn.q_proj.weight",
                f"layers.{layer}.self_attn.k_proj.weight",
                f"layers.{layer}.self_attn.v_proj.weight",
                f"layers.{layer}.self_attn.o_proj.weight",
                f"layers.{layer}.self_attn.q_norm.weight",
                f"layers.{layer}.self_attn.k_norm.weight",
                f"layers.{layer}.mlp.gate_proj.weight",
                f"layers.{layer}.mlp.up_proj.weight",
                f"layers.{layer}.mlp.down_proj.weight",
            }
        )
    if variant == "dspark":
        keys.update(
            {
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
                "confidence_head.proj.bias",
            }
        )
    return keys


def config_mismatches(variant: str, config: dict[str, object]) -> list[str]:
    expected_architecture = {
        "dflash": "DFlashDraftModel",
        "dspark": "Qwen3DSparkModel",
    }[variant]
    expected: dict[str, object] = {
        "architectures": [expected_architecture],
        "block_size": 8,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "head_dim": 128,
        "num_hidden_layers": 5,
    }
    mismatches = [
        f"{key}={config.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    dflash = config.get("dflash_config")
    if not isinstance(dflash, dict):
        return [*mismatches, "dflash_config is missing or not an object"]
    expected_dflash: dict[str, object] = {
        "mask_token_id": 151669,
        "target_layer_ids": [1, 12, 23, 34, 45],
    }
    if variant == "dspark":
        expected_dflash.update(
            {
                "markov_head_type": "vanilla",
                "markov_rank": 256,
                "projector_type": "dspark",
                "shift_label": True,
                "use_confidence_head": True,
            }
        )
    mismatches.extend(
        f"dflash_config.{key}={dflash.get(key)!r} expected={value!r}"
        for key, value in expected_dflash.items()
        if dflash.get(key) != value
    )
    return mismatches


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("dflash", "dspark"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--identity-file", type=Path, required=True)
parser.add_argument("--verify-content-sha", action="store_true")
args = parser.parse_args()

identity = json.loads(args.identity_file.read_text())[args.variant]
identity_errors: list[str] = []
for filename in ("config.json", "model.safetensors"):
    path = args.checkpoint / filename
    expected = identity[filename]
    actual_size = path.stat().st_size
    if actual_size != expected["size"]:
        identity_errors.append(
            f"{filename}.size={actual_size} expected={expected['size']}"
        )
    if args.verify_content_sha:
        actual_sha = sha256(path)
        if actual_sha != expected["sha256"]:
            identity_errors.append(
                f"{filename}.sha256={actual_sha} expected={expected['sha256']}"
            )
if identity_errors:
    raise SystemExit(f"checkpoint identity mismatch: {identity_errors}")
config = json.loads((args.checkpoint / "config.json").read_text())
config_errors = config_mismatches(args.variant, config)
if config_errors:
    raise SystemExit(f"checkpoint config mismatch: {config_errors}")
actual = state_dict_keys(args.checkpoint / "model.safetensors")
expected = expected_keys(args.variant)
missing = sorted(expected - actual)
unexpected = sorted(actual - expected)
print(f"STATE_DICT_GATE missing={len(missing)} unexpected={len(unexpected)}")
if missing or unexpected:
    raise SystemExit(f"state-dict mismatch: missing={missing} unexpected={unexpected}")
