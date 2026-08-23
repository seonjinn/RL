"""Fail closed unless a DFlash/DSpark checkpoint has the pinned identity and schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def assert_config(variant: str, config: dict[str, Any]) -> None:
    expected = {
        "architectures": [{"dflash": "DFlashDraftModel", "dspark": "Qwen3DSparkModel"}[variant]],
        "block_size": 8,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "head_dim": 128,
        "num_hidden_layers": 5,
    }
    errors = [f"{key}={config.get(key)!r} expected={value!r}" for key, value in expected.items() if config.get(key) != value]
    draft = config.get("dflash_config")
    if not isinstance(draft, dict):
        errors.append("dflash_config missing")
    else:
        expected_draft: dict[str, Any] = {"mask_token_id": 151669, "target_layer_ids": [1, 12, 23, 34, 45]}
        if variant == "dspark":
            expected_draft.update({"markov_head_type": "vanilla", "markov_rank": 256, "projector_type": "dspark", "shift_label": True, "use_confidence_head": True})
        errors.extend(f"dflash_config.{key}={draft.get(key)!r} expected={value!r}" for key, value in expected_draft.items() if draft.get(key) != value)
    if errors:
        raise SystemExit(f"checkpoint config mismatch: {errors}")


parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("dflash", "dspark"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--identity-file", type=Path, required=True)
parser.add_argument("--verify-content-sha", action="store_true")
args = parser.parse_args()

identity = json.loads(args.identity_file.read_text())[args.variant]
for filename in ("config.json", "model.safetensors"):
    path = args.checkpoint / filename
    if not path.is_file():
        raise SystemExit(f"missing checkpoint file: {path}")
    if path.stat().st_size != identity[filename]["size"]:
        raise SystemExit(f"checkpoint size mismatch: {path}")
    if args.verify_content_sha and sha256(path) != identity[filename]["sha256"]:
        raise SystemExit(f"checkpoint sha256 mismatch: {path}")
assert_config(args.variant, json.loads((args.checkpoint / "config.json").read_text()))
actual = state_dict_keys(args.checkpoint / "model.safetensors")
expected = expected_keys(args.variant)
missing = sorted(expected - actual)
unexpected = sorted(actual - expected)
if missing or unexpected:
    raise SystemExit(f"state-dict mismatch: missing={missing} unexpected={unexpected}")
print(f"STATE_DICT_GATE_PASS variant={args.variant} keys={len(actual)} missing=0 unexpected=0")
