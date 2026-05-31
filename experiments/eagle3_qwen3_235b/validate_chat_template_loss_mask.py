#!/usr/bin/env python3
"""Validate that a chat template produces assistant-token loss masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-or-tokenizer",
        default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-235B-A22B-Thinking-2507"),
        help="HF model id or local tokenizer directory.",
    )
    parser.add_argument("--chat-template", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-missing-transformers",
        action="store_true",
        help="Return success with a warning if Transformers is unavailable.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def flatten_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"expected token id list, got {type(value).__name__}")
    return [int(x) for x in value]


def flatten_mask(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"expected mask list, got {type(value).__name__}")
    return [int(x) for x in value]


def mask_from_output(output: Any) -> tuple[list[int], list[int]]:
    if not isinstance(output, Mapping):
        raise TypeError(f"apply_chat_template did not return a dict: {type(output).__name__}")
    input_ids = flatten_ids(output.get("input_ids"))
    for key in ("assistant_masks", "assistant_tokens_mask"):
        if key in output:
            return input_ids, flatten_mask(output[key])
    raise KeyError("assistant mask key not found in tokenizer output")


def write_summary(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        payload = {
            "status": "warning",
            "reason": "transformers_unavailable",
            "error": str(exc),
        }
        write_summary(args.json_out, payload)
        if args.allow_missing_transformers:
            print(f"WARN Transformers unavailable: {exc}")
            return 0
        print(f"FAIL Transformers unavailable: {exc}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_or_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.chat_template:
        tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")

    template = tokenizer.chat_template
    if not template or "generation" not in template or "endgeneration" not in template:
        payload = {
            "status": "failed",
            "reason": "missing_generation_tags",
            "model_or_tokenizer": args.model_or_tokenizer,
            "chat_template": str(args.chat_template) if args.chat_template else None,
        }
        write_summary(args.json_out, payload)
        print("FAIL chat template lacks generation/endgeneration tags", file=sys.stderr)
        return 1

    messages = [
        {"role": "user", "content": "Solve this simple test task."},
        {
            "role": "assistant",
            "content": "<think>\nWe need answer directly.\n</think>\n\nThe answer is 42.",
        },
    ]
    try:
        output = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids, loss_mask = mask_from_output(output)
    except Exception as exc:
        payload = {
            "status": "failed",
            "reason": "apply_chat_template_failed",
            "error": str(exc),
            "model_or_tokenizer": args.model_or_tokenizer,
            "chat_template": str(args.chat_template) if args.chat_template else None,
        }
        write_summary(args.json_out, payload)
        print(f"FAIL apply_chat_template failed: {exc}", file=sys.stderr)
        return 1

    mask_sum = sum(loss_mask)
    if len(input_ids) != len(loss_mask):
        payload = {
            "status": "failed",
            "reason": "mask_length_mismatch",
            "input_tokens": len(input_ids),
            "mask_tokens": len(loss_mask),
        }
        write_summary(args.json_out, payload)
        print(
            f"FAIL assistant mask length {len(loss_mask)} != input length {len(input_ids)}",
            file=sys.stderr,
        )
        return 1
    if mask_sum <= 0:
        payload = {
            "status": "failed",
            "reason": "empty_assistant_mask",
            "input_tokens": len(input_ids),
            "mask_sum": mask_sum,
        }
        write_summary(args.json_out, payload)
        print("FAIL assistant mask has no positive tokens", file=sys.stderr)
        return 1

    payload = {
        "status": "passed",
        "model_or_tokenizer": args.model_or_tokenizer,
        "chat_template": str(args.chat_template) if args.chat_template else None,
        "input_tokens": len(input_ids),
        "assistant_mask_tokens": mask_sum,
        "assistant_mask_ratio": mask_sum / len(input_ids),
    }
    write_summary(args.json_out, payload)
    print(
        "validated assistant loss mask: "
        f"tokens={len(input_ids)} assistant_mask_tokens={mask_sum}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
