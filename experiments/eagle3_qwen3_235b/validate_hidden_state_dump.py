#!/usr/bin/env python3
"""Validate ModelOpt offline Eagle3 hidden-state dump files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hidden_states_dir", type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--require-loss-mask", action="store_true")
    parser.add_argument("--require-positive-loss-mask", action="store_true")
    parser.add_argument("--expected-hidden-size", type=int, default=None)
    parser.add_argument("--expected-aux-count", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--validate-modelopt-loader", action="store_true")
    parser.add_argument("--modelopt-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"object has no shape: {type(value).__name__}")
    return tuple(int(dim) for dim in shape)


def check_aux_shape(
    path: Path,
    aux_shape: tuple[int, ...],
    expected_hidden_size: int | None,
    expected_aux_count: int | None,
) -> None:
    if expected_hidden_size is None and expected_aux_count is None:
        return

    if len(aux_shape) == 2:
        expected_width = None
        if expected_hidden_size is not None and expected_aux_count is not None:
            expected_width = expected_hidden_size * expected_aux_count
        if expected_width is not None and aux_shape[-1] != expected_width:
            raise SystemExit(
                f"{path} aux_hidden_states width mismatch: {aux_shape[-1]} != {expected_width}"
            )
        return

    if len(aux_shape) == 3:
        if expected_aux_count is not None and aux_shape[1] != expected_aux_count:
            raise SystemExit(
                f"{path} aux_hidden_states aux-count mismatch: {aux_shape[1]} != {expected_aux_count}"
            )
        if expected_hidden_size is not None and aux_shape[-1] != expected_hidden_size:
            raise SystemExit(
                f"{path} aux_hidden_states hidden-size mismatch: {aux_shape[-1]} != {expected_hidden_size}"
            )
        return

    raise SystemExit(f"{path} aux_hidden_states rank must be 2 or 3, got shape {aux_shape}")


def validate_modelopt_loader(args: argparse.Namespace, files: list[Path], checked: int) -> dict[str, Any]:
    modelopt_dir = args.modelopt_dir or Path(__file__).resolve().parents[2] / "Model-Optimizer"
    sys.path.insert(0, str(modelopt_dir))
    try:
        from modelopt.torch.speculative.eagle.utils import (  # type: ignore
            EagleOfflineDataCollator,
            OfflineSupervisedDataset,
        )
    except Exception as exc:
        raise SystemExit(f"failed to import ModelOpt offline dataset from {modelopt_dir}: {exc}") from exc

    dataset = OfflineSupervisedDataset(
        [str(path) for path in files[:checked]],
        answer_only_loss=args.require_loss_mask,
    )
    collator = EagleOfflineDataCollator(args.max_seq_len or max(1, checked))
    items = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        missing = {
            "input_ids",
            "base_model_hidden_states",
            "aux_hidden_states",
            "attention_mask",
            "loss_mask",
            "labels",
        } - set(item)
        if missing:
            raise SystemExit(f"ModelOpt dataset item {idx} missing keys: {sorted(missing)}")
        items.append(item)
    if items:
        batch = collator(items[:1])
        if "base_model_outputs" not in batch:
            raise SystemExit("ModelOpt collator output lacks base_model_outputs")
    return {
        "modelopt_dir": str(modelopt_dir),
        "dataset_items_checked": len(items),
        "collator_checked": bool(items),
    }


def main() -> None:
    args = parse_args()
    import torch

    files = sorted(args.hidden_states_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found under {args.hidden_states_dir}")

    checked = 0
    token_count = 0
    max_seq_len_seen = 0
    positive_loss_mask_files = 0
    for path in files[: args.limit]:
        obj = torch.load(path, map_location="cpu")
        missing = {"input_ids", "hidden_states", "aux_hidden_states"} - set(obj)
        if missing:
            raise SystemExit(f"{path} missing required keys: {sorted(missing)}")
        if args.require_loss_mask and "loss_mask" not in obj:
            raise SystemExit(f"{path} missing loss_mask")

        input_ids = obj["input_ids"]
        hidden_states = obj["hidden_states"]
        aux_hidden_states = obj["aux_hidden_states"]
        input_shape = shape_of(input_ids)
        hidden_shape = shape_of(hidden_states)
        aux_shape = shape_of(aux_hidden_states)
        if len(input_shape) != 1:
            raise SystemExit(f"{path} input_ids rank must be 1, got shape {input_shape}")
        if len(hidden_shape) != 2:
            raise SystemExit(f"{path} hidden_states rank must be 2, got shape {hidden_shape}")

        seq_len = input_shape[0]
        if args.max_seq_len is not None and seq_len > args.max_seq_len:
            raise SystemExit(f"{path} seq_len {seq_len} exceeds max_seq_len {args.max_seq_len}")
        if hidden_shape[0] != seq_len:
            raise SystemExit(
                f"{path} hidden_states seq mismatch: {hidden_shape[0]} != {seq_len}"
            )
        if args.expected_hidden_size is not None and hidden_shape[-1] != args.expected_hidden_size:
            raise SystemExit(
                f"{path} hidden_states hidden-size mismatch: {hidden_shape[-1]} != {args.expected_hidden_size}"
            )
        if aux_shape[0] != seq_len:
            raise SystemExit(f"{path} aux_hidden_states seq mismatch: {aux_shape[0]} != {seq_len}")
        check_aux_shape(path, aux_shape, args.expected_hidden_size, args.expected_aux_count)
        if args.require_loss_mask:
            loss_mask_shape = shape_of(obj["loss_mask"])
            if loss_mask_shape[0] != seq_len:
                raise SystemExit(f"{path} loss_mask seq mismatch: {loss_mask_shape[0]} != {seq_len}")
            if args.require_positive_loss_mask:
                loss_mask_sum = float(obj["loss_mask"].sum().item())
                if loss_mask_sum <= 0:
                    raise SystemExit(f"{path} loss_mask has no positive tokens")
                positive_loss_mask_files += 1
        checked += 1
        token_count += seq_len
        max_seq_len_seen = max(max_seq_len_seen, seq_len)

    summary = {
        "hidden_states_dir": str(args.hidden_states_dir),
        "total_files": len(files),
        "checked_files": checked,
        "checked_tokens": token_count,
        "max_seq_len_seen": max_seq_len_seen,
        "require_loss_mask": args.require_loss_mask,
        "positive_loss_mask_files": positive_loss_mask_files,
        "expected_hidden_size": args.expected_hidden_size,
        "expected_aux_count": args.expected_aux_count,
        "modelopt_loader_validation": None,
    }
    if args.validate_modelopt_loader:
        summary["modelopt_loader_validation"] = validate_modelopt_loader(args, files, checked)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"validated {checked} files under {args.hidden_states_dir} "
        f"({token_count} tokens, max seq {max_seq_len_seen})"
    )


if __name__ == "__main__":
    main()
