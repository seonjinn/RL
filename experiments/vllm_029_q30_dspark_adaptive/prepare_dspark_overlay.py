#!/usr/bin/env python3
"""Validate a node-local DSpark copy and expose its confidence-head layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    return header


def prepare_overlay(checkpoint: Path) -> dict[str, object]:
    """Mutate only a node-local checkpoint copy after validating head width."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3DSparkModel"]:
        raise ValueError("checkpoint is not a Qwen3DSparkModel")
    draft = config.get("dflash_config") or {}
    if draft.get("use_confidence_head") is not True:
        raise ValueError("checkpoint does not declare a trained confidence head")

    header = _safetensors_header(weights_path)
    weight = header.get("confidence_head.proj.weight")
    if not isinstance(weight, dict) or not isinstance(weight.get("shape"), list):
        raise ValueError("checkpoint does not contain confidence_head.proj.weight")
    shape = weight["shape"]
    if len(shape) != 2 or shape[0] != 1:
        raise ValueError("confidence head has an invalid tensor shape")
    width = int(shape[1])
    hidden_size = int(config["hidden_size"])
    markov_rank = int(config["markov_rank"])
    if width == hidden_size:
        with_markov = False
    elif width == hidden_size + markov_rank:
        with_markov = True
    else:
        raise ValueError(
            "confidence head width matches neither hidden_size nor hidden_size + markov_rank"
        )

    config["enable_confidence_head"] = True
    config["confidence_head_with_markov"] = with_markov
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    receipt = {
        "checkpoint": str(checkpoint),
        "confidence_head_width": width,
        "hidden_size": hidden_size,
        "markov_rank": markov_rank,
        "confidence_head_with_markov": with_markov,
    }
    (checkpoint / "adaptive-overlay-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parsed = parser.parse_args()
    print(json.dumps(prepare_overlay(parsed.checkpoint), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

