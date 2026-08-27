"""Fail closed when a DFlash or DSpark checkpoint schema is not exact."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def state_dict_keys(path: Path) -> set[str]:
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dflash", "dspark"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--print-expected", choices=("dflash", "dspark"))
    args = parser.parse_args()

    if args.print_expected is not None:
        if args.variant is not None or args.checkpoint is not None:
            parser.error(
                "--print-expected cannot be combined with checkpoint validation"
            )
        print(json.dumps(sorted(expected_keys(args.print_expected))))
        return
    if args.variant is None or args.checkpoint is None:
        parser.error("--variant and --checkpoint are required for validation")

    actual = state_dict_keys(args.checkpoint / "model.safetensors")
    expected = expected_keys(args.variant)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    print(f"STATE_DICT_GATE missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise SystemExit(
            f"state-dict mismatch: missing={missing} unexpected={unexpected}"
        )


if __name__ == "__main__":
    main()
