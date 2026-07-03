#!/usr/bin/env python3
"""Materialize pinned RL math datasets into a normalized prompt JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal


Source = Literal["dapo_math_17k", "openmathinstruct2"]
SOURCE_SPECS: dict[Source, dict[str, str]] = {
    "dapo_math_17k": {
        "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "split": "train",
    },
    "openmathinstruct2": {
        "dataset": "nvidia/OpenMathInstruct-2",
        "revision": "469216e3f46f4dacf476b382e192485ea51a143e",
        "split": "train_1M",
    },
}


def prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("prompt must be a list of role/content messages")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "assistant":
            break
        if isinstance(role, str) and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("prompt contains no usable pre-assistant messages")
    return messages


def normalize_row(source: Source, row: dict[str, Any], *, source_row: int) -> dict[str, Any]:
    if source == "dapo_math_17k":
        messages = normalize_messages(row.get("prompt"))
        reward_model = row.get("reward_model")
        reward = reward_model if isinstance(reward_model, dict) else {}
        extra_info = row.get("extra_info")
        extra = extra_info if isinstance(extra_info, dict) else {}
        expected_answer = reward.get("ground_truth")
        source_id = extra.get("index")
        metadata = {
            "ability": row.get("ability"),
            "data_source": row.get("data_source"),
            "reward_style": reward.get("style"),
        }
    elif source == "openmathinstruct2":
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("OpenMathInstruct-2 row has no problem")
        messages = [{"role": "user", "content": problem}]
        expected_answer = row.get("expected_answer")
        source_id = None
        metadata = {"problem_source": row.get("problem_source")}
    else:
        raise ValueError(f"unsupported source: {source}")

    digest = prompt_hash(messages)
    return {
        "id": str(source_id or f"{source}-{source_row}-{digest[:16]}"),
        "source": source,
        "source_row": source_row,
        "prompt_sha256": digest,
        "messages": messages,
        "expected_answer": expected_answer,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=tuple(SOURCE_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit <= 0 or args.offset < 0:
        raise ValueError("limit must be positive and offset must be non-negative")
    spec = SOURCE_SPECS[args.source]

    from datasets import load_dataset  # pyright: ignore[reportMissingImports]

    dataset = load_dataset(
        spec["dataset"],
        split=spec["split"],
        revision=spec["revision"],
        streaming=args.streaming,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    seen_prompts: set[str] = set()
    written = 0
    examined = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for source_row, raw_row in enumerate(dataset):
            if source_row < args.offset:
                continue
            examined += 1
            normalized = normalize_row(args.source, dict(raw_row), source_row=source_row)
            digest = normalized["prompt_sha256"]
            if digest in seen_prompts:
                continue
            seen_prompts.add(digest)
            stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1
            if written == args.limit:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if written != args.limit:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"materialized {written} unique prompts, expected {args.limit}"
        )
    temporary.replace(args.output)

    metadata = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": args.source,
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "split": spec["split"],
        "streaming": args.streaming,
        "offset": args.offset,
        "examined_rows": examined,
        "unique_prompts": written,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
