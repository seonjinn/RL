#!/usr/bin/env python3
"""Find RL rollout JSONL files that can become Eagle3 training conversations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from normalize_rl_rollouts_to_conversations import extract_from_record


DEFAULT_PATTERNS = (
    "train_data_step*.jsonl",
    "*rollout*.jsonl",
    "*trajectory*.jsonl",
    "*trajectories*.jsonl",
    "*generation*.jsonl",
    "*sample*.jsonl",
)


@dataclass
class Candidate:
    path: Path
    rows_sampled: int = 0
    json_rows: int = 0
    extracted_conversations: int = 0
    invalid_json: int = 0
    key_counts: dict[str, int] = field(default_factory=dict)
    sample_error: str | None = None

    @property
    def score(self) -> tuple[int, float, int]:
        ratio = self.extracted_conversations / self.json_rows if self.json_rows else 0.0
        return (self.extracted_conversations, ratio, -self.invalid_json)

    def to_json(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = None
            mtime = None
        return {
            "path": str(self.path),
            "size_bytes": size,
            "mtime": mtime,
            "rows_sampled": self.rows_sampled,
            "json_rows": self.json_rows,
            "extracted_conversations": self.extracted_conversations,
            "invalid_json": self.invalid_json,
            "key_counts": dict(sorted(self.key_counts.items(), key=lambda item: (-item[1], item[0]))[:24]),
            "sample_error": self.sample_error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Files or directories to scan")
    parser.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS))
    parser.add_argument("--include-all-jsonl", action="store_true")
    parser.add_argument("--include-json", action="store_true", help="Also scan .json files such as Codex llm_completions records.")
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--sample-lines", type=int, default=40)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--id-key", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument(
        "--output-schema",
        choices=("modelopt", "specforge"),
        default="modelopt",
        help="Schema for --prepare-output. SpecForge uses id/conversations.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument(
        "--prepare-output",
        type=Path,
        default=None,
        help="Normalize top candidate files into this ModelOpt conversation JSONL.",
    )
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--include-reasoning-content", action="store_true")
    parser.add_argument("--infer-flat-content-roles", action="store_true")
    parser.add_argument(
        "--compact-current-turn",
        action="store_true",
        help=(
            "When scanning SWE/Codex traces, count/prepare only system/developer "
            "context plus the final user turn and assistant response."
        ),
    )
    parser.add_argument("--reasoning-open-tag", default="<think>\n")
    parser.add_argument("--reasoning-close-tag", default="\n</think>\n\n")
    parser.add_argument("--limit", type=int, default=None, help="Max input rows when preparing output")
    return parser.parse_args()


def collect_files(
    roots: Iterable[Path],
    patterns: list[str],
    include_all_jsonl: bool,
    include_json: bool,
    max_files: int,
) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    active_patterns = list(patterns)
    if include_all_jsonl and "*.jsonl" not in active_patterns:
        active_patterns.append("*.jsonl")
    if include_json and "*.json" not in active_patterns:
        active_patterns.append("*.json")

    def add_file(path: Path) -> None:
        if len(files) >= max_files:
            return
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        files.append(path)

    for root in roots:
        if root.is_file():
            add_file(root)
            continue
        if not root.is_dir():
            continue
        for pattern in active_patterns:
            for path in sorted(root.rglob(pattern)):
                add_file(path)
                if len(files) >= max_files:
                    return files
    return files


def inspect_file(
    path: Path,
    sample_lines: int,
    min_assistant_chars: int,
    id_key: str | None,
    include_reasoning_content: bool,
    reasoning_open_tag: str,
    reasoning_close_tag: str,
    infer_flat_content_roles: bool,
    compact_current_turn: bool,
) -> Candidate:
    candidate = Candidate(path=path)
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            records = value if isinstance(value, list) else [value]
            for line_num, record in enumerate(records[:sample_lines], 1):
                candidate.rows_sampled += 1
                if not isinstance(record, dict):
                    continue
                candidate.json_rows += 1
                for key in record:
                    candidate.key_counts[key] = candidate.key_counts.get(key, 0) + 1
                candidate.extracted_conversations += len(
                    extract_from_record(
                        record,
                        path,
                        line_num,
                        id_key,
                        min_assistant_chars,
                        include_reasoning_content,
                        reasoning_open_tag,
                        reasoning_close_tag,
                        infer_flat_content_roles,
                        compact_current_turn=compact_current_turn,
                    )
                )
            return candidate
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                candidate.rows_sampled += 1
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    candidate.invalid_json += 1
                    if candidate.rows_sampled >= sample_lines:
                        break
                    continue
                if isinstance(record, dict):
                    candidate.json_rows += 1
                    for key in record:
                        candidate.key_counts[key] = candidate.key_counts.get(key, 0) + 1
                    candidate.extracted_conversations += len(
                        extract_from_record(
                            record,
                            path,
                            line_num,
                            id_key,
                            min_assistant_chars,
                            include_reasoning_content,
                            reasoning_open_tag,
                            reasoning_close_tag,
                            infer_flat_content_roles,
                            compact_current_turn=compact_current_turn,
                        )
                    )
                if candidate.rows_sampled >= sample_lines:
                    break
    except Exception as exc:
        candidate.sample_error = str(exc)
    return candidate


def select_candidates(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    positives = [c for c in candidates if c.extracted_conversations > 0]
    return sorted(positives, key=lambda c: c.score, reverse=True)[:top_k]


def write_payload(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RL Rollout Conversation Source Discovery",
        "",
        f"Files scanned: **{payload['files_scanned']}**",
        f"Positive candidates: **{payload['positive_candidates']}**",
        "",
        "| rank | extracted/sample | rows | invalid | file |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, item in enumerate(payload["selected_candidates"], 1):
        lines.append(
            f"| {idx} | {item['extracted_conversations']} | {item['rows_sampled']} | "
            f"{item['invalid_json']} | `{item['path']}` |"
        )
    if payload.get("prepared_output"):
        lines.extend(["", f"Prepared output: `{payload['prepared_output']}`"])
    return "\n".join(lines) + "\n"


def run_prepare(selected: list[Candidate], args: argparse.Namespace) -> dict[str, Any]:
    assert args.prepare_output is not None
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("normalize_rl_rollouts_to_conversations.py")),
        "--input",
        *[str(candidate.path) for candidate in selected],
        "--output",
        str(args.prepare_output),
        "--model",
        args.model,
        "--output-schema",
        args.output_schema,
    ]
    if args.include_metadata:
        cmd.append("--include-metadata")
    if args.include_reasoning_content:
        cmd.extend(
            [
                "--include-reasoning-content",
                "--reasoning-open-tag",
                args.reasoning_open_tag,
                "--reasoning-close-tag",
                args.reasoning_close_tag,
            ]
        )
    if args.infer_flat_content_roles:
        cmd.append("--infer-flat-content-roles")
    if args.compact_current_turn:
        cmd.append("--compact-current-turn")
    if args.id_key:
        cmd.extend(["--id-key", args.id_key])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

    validate_cmd = [
        sys.executable,
        str(Path(__file__).with_name("validate_training_conversations.py")),
        str(args.prepare_output),
        "--max-seq-len",
        "16384",
    ]
    validate = subprocess.run(
        validate_cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "output": str(args.prepare_output),
        "normalize_returncode": result.returncode,
        "normalize_output": result.stdout[-4000:],
        "validate_returncode": validate.returncode,
        "validate_output": validate.stdout[-4000:],
    }


def main() -> int:
    args = parse_args()
    patterns = [part.strip() for part in args.patterns.split(",") if part.strip()]
    files = collect_files(args.roots, patterns, args.include_all_jsonl, args.include_json, args.max_files)
    candidates = [
        inspect_file(
            path,
            args.sample_lines,
            args.min_assistant_chars,
            args.id_key,
            args.include_reasoning_content,
            args.reasoning_open_tag,
            args.reasoning_close_tag,
            args.infer_flat_content_roles,
            args.compact_current_turn,
        )
        for path in files
    ]
    selected = select_candidates(candidates, args.top_k)
    prepare_result = None
    if args.prepare_output is not None:
        if not selected:
            prepare_result = {
                "output": str(args.prepare_output),
                "normalize_returncode": 1,
                "normalize_output": "no positive candidates selected",
                "validate_returncode": None,
                "validate_output": "",
            }
        else:
            prepare_result = run_prepare(selected, args)

    payload = {
        "roots": [str(root) for root in args.roots],
        "patterns": patterns,
        "files_scanned": len(files),
        "positive_candidates": sum(1 for c in candidates if c.extracted_conversations > 0),
        "selected_candidates": [candidate.to_json() for candidate in selected],
        "all_candidates": [candidate.to_json() for candidate in sorted(candidates, key=lambda c: c.score, reverse=True)[:50]],
        "prepared_output": str(args.prepare_output) if args.prepare_output else None,
        "prepare_result": prepare_result,
        "compact_current_turn": args.compact_current_turn,
    }

    write_payload(args.json_out, payload)
    markdown = render_markdown(payload)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")

    if prepare_result and (prepare_result["normalize_returncode"] != 0 or prepare_result["validate_returncode"] != 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
