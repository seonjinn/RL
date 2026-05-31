#!/usr/bin/env python3
"""Materialize OpenMath-style rows as ModelOpt conversation JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_TEMPLATE = (
    "Think step-by-step to solve the following problem. "
    "Output your answer inside of \\\\boxed{{}} tags.:\n{}\n\nLet's think step-by-step"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="nvidia/OpenMathInstruct-2")
    parser.add_argument("--split", default="train_1M")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-scan", type=int, default=200000)
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument(
        "--response-keys",
        default="generated_solution,solution,answer,output,response",
        help="Comma-separated assistant response fields to try in order.",
    )
    parser.add_argument("--id-key", default="")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--exclude-prompts-from",
        type=Path,
        action="append",
        default=[],
        help="Conversation JSONL files whose user prompts must be excluded.",
    )
    parser.add_argument(
        "--allow-duplicate-prompts",
        action="store_true",
        help="Allow duplicate user prompts within the materialized output.",
    )
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def row_id(row: dict[str, Any], source_index: int, id_key: str) -> str:
    keys = [id_key] if id_key else []
    keys.extend(("id", "problem_id", "uuid"))
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"openmath-{source_index:08d}"


def first_text(row: dict[str, Any], keys: list[str]) -> tuple[str, str] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    return None


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def user_prompt(record: dict[str, Any]) -> str | None:
    messages = record.get("messages", record.get("conversations"))
    if not isinstance(messages, list):
        messages = record.get("prompt")
    if isinstance(messages, str) and messages.strip():
        return messages
    if not isinstance(messages, list):
        for key in ("problem", "question", "input"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", ""))).lower()
        role = {"human": "user"}.get(role, role)
        content = message.get("content", message.get("value", message.get("text")))
        if role == "user" and isinstance(content, str) and content.strip():
            return content
    return None


def load_excluded_prompts(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"--exclude-prompts-from does not exist: {path}")
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                prompt = user_prompt(record)
                if prompt:
                    excluded.add(normalize_prompt(prompt))
    return excluded


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    from datasets import load_dataset  # type: ignore

    response_keys = [key.strip() for key in args.response_keys.split(",") if key.strip()]
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    excluded_prompts = load_excluded_prompts(args.exclude_prompts_from)
    seen_prompts: set[str] = set()

    sample_rows: list[dict[str, Any]] = []
    inspected_keys: dict[str, int] = {}
    skipped_missing_problem = 0
    skipped_missing_response = 0
    skipped_excluded_prompt = 0
    skipped_duplicate_prompt = 0

    scanned = 0
    rows_written = 0
    output_tmp: Path | None = None
    output_fh = None
    if not args.inspect_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_tmp = args.output.with_name(f"{args.output.name}.tmp")
        output_fh = output_tmp.open("w", encoding="utf-8")

    try:
        for source_index, row in enumerate(dataset):
            if source_index < args.offset:
                continue
            scanned += 1
            if args.max_scan > 0 and scanned > args.max_scan:
                break
            if rows_written >= args.limit:
                break
            for key in row:
                inspected_keys[key] = inspected_keys.get(key, 0) + 1
            problem = row.get(args.problem_key)
            if not isinstance(problem, str) or not problem.strip():
                skipped_missing_problem += 1
                continue
            response_pair = first_text(row, response_keys)
            if response_pair is None:
                skipped_missing_response += 1
                continue
            response_key, response = response_pair
            formatted_prompt = args.prompt_template.format(problem)
            prompt_key = normalize_prompt(formatted_prompt)
            if prompt_key in excluded_prompts:
                skipped_excluded_prompt += 1
                continue
            if not args.allow_duplicate_prompts and prompt_key in seen_prompts:
                skipped_duplicate_prompt += 1
                continue
            seen_prompts.add(prompt_key)
            sid = row_id(row, source_index, args.id_key)
            record = {
                "conversation_id": f"{sid}-r00",
                "messages": [
                    {"role": "user", "content": formatted_prompt},
                    {"role": "assistant", "content": response},
                ],
                "source_id": sid,
                "source_dataset": args.dataset,
                "source_split": args.split,
                "source_index": source_index,
                "response_key": response_key,
                "response_index": 0,
                "model": args.model,
            }
            if len(sample_rows) < 2:
                sample_rows.append(record)
            if output_fh is not None:
                output_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
    finally:
        if output_fh is not None:
            output_fh.close()

    summary = {
        "rows_written": rows_written,
        "dataset": args.dataset,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "max_scan": args.max_scan,
        "scanned": scanned,
        "output": str(args.output),
        "inspected_keys": inspected_keys,
        "skipped_missing_problem": skipped_missing_problem,
        "skipped_missing_response": skipped_missing_response,
        "skipped_excluded_prompt": skipped_excluded_prompt,
        "skipped_duplicate_prompt": skipped_duplicate_prompt,
        "excluded_prompt_count": len(excluded_prompts),
        "unique_output_prompts": len(seen_prompts),
        "sample": sample_rows,
    }
    if args.inspect_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if rows_written == 0:
        if output_tmp is not None:
            output_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"No rows materialized from {args.dataset}/{args.split}; "
            f"available sampled keys: {sorted(inspected_keys)}"
        )
    assert output_tmp is not None
    output_tmp.replace(args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, indent=2))


if __name__ == "__main__":
    main()
