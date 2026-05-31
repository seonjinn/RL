#!/usr/bin/env python3
"""Write OpenMathInstruct-2 prompts as JSONL for direct vLLM generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE = (
    "Think step-by-step to solve the following problem. "
    "Output your answer inside of \\\\boxed{{}} tags.:\n{}\n\nLet's think step-by-step"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="nvidia/OpenMathInstruct-2")
    parser.add_argument("--split", default="train_1M")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--id-key", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompt-template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def prompt_id(row: dict[str, Any], index: int, id_key: str | None) -> str:
    keys = [id_key] if id_key else []
    keys.extend(("id", "problem_id", "uuid"))
    for key in keys:
        if key and row.get(key) not in (None, ""):
            return str(row[key])
    return f"openmath-{index:08d}"


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    rows = []
    seen = 0
    for index, row in enumerate(dataset):
        if index < args.offset:
            continue
        problem = row.get(args.problem_key)
        if not isinstance(problem, str) or not problem.strip():
            continue
        sid = prompt_id(row, index, args.id_key)
        rows.append(
            {
                "id": sid,
                "problem": args.prompt_template.format(problem),
                "source_dataset": args.dataset,
                "source_split": args.split,
                "source_index": index,
            }
        )
        seen += 1
        if seen >= args.limit:
            break

    if args.inspect_only:
        print(json.dumps({"rows": len(rows), "sample": rows[:2]}, indent=2, ensure_ascii=False))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"rows_written": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
