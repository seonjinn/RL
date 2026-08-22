"""Fail closed when a DFlash or DSpark checkpoint schema is not exact."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def _state_dict_keys(path: Path) -> set[str]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return set(header) - {"__metadata__"}


def _expected_keys(variant: str) -> set[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dflash", "dspark"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    actual = _state_dict_keys(args.checkpoint / "model.safetensors")
    expected = _expected_keys(args.variant)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    print(f"STATE_DICT_GATE missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise SystemExit(f"state-dict mismatch: missing={missing} unexpected={unexpected}")


if __name__ == "__main__":
    main()
