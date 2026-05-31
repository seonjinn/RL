#!/usr/bin/env python3
"""Discover SWE/R2E NemoGym JSONL datasets for Qwen3 rollout capture.

This scans bounded roots for JSONL files whose first sampled records match the
NemoGym SWE-agent format consumed by `NemoGymDataset`:

  {"responses_create_params": {"input": ..., "metadata": {...}}}

It is intentionally read-only and conservative; it does not treat conversation
logs or already-normalized ModelOpt rows as rollout input datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_PATHS = (
    Path(
        "/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/"
        "dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
    ),
    Path(
        "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/"
        "dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
    ),
)

DEFAULT_ROOTS = (
    Path(
        "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/ultra/"
        "tk-nemo-gym"
    ),
    Path(
        "/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/ultra/"
        "tk-nemo-gym"
    ),
    Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data"),
)

DEFAULT_NAME_HINTS = (
    "*swe*.jsonl",
    "*r2e*.jsonl",
    "*gym*.jsonl",
    "example.jsonl",
)
REQUIRED_SWE_METADATA_KEYS = (
    "problem_statement",
    "instance_id",
    "base_commit",
    "dataset_name",
    "split",
    "instance_dict",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--expected-path", action="append", type=Path, default=list(DEFAULT_EXPECTED_PATHS))
    parser.add_argument("--name-hint", action="append", default=list(DEFAULT_NAME_HINTS))
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=2000)
    parser.add_argument("--sample-lines", type=int, default=8)
    parser.add_argument("--min-full-lines", type=int, default=100, help="Line count threshold for a usable non-smoke SWE dataset.")
    parser.add_argument("--count-lines", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def depth_from(root: Path, path: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return 10**9


def collect_files(roots: list[Path], hints: list[str], max_depth: int, max_files: int) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []

    def add(path: Path) -> None:
        if len(files) >= max_files or not path.is_file():
            return
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(path)

    for root in roots:
        if root.is_file():
            add(root)
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if depth_from(root, current) >= max_depth:
                dirnames[:] = []
            for name in filenames:
                path = current / name
                if any(path.match(hint) for hint in hints):
                    add(path)
                    if len(files) >= max_files:
                        return files
    return files


def line_count(path: Path) -> int | None:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def inspect_jsonl(path: Path, sample_lines: int, count_lines: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": None,
        "line_count": None,
        "sampled": 0,
        "json_records": 0,
        "nemogym_records": 0,
        "swe_like_records": 0,
        "valid_swe_records": 0,
        "invalid_swe_metadata": 0,
        "invalid_json": 0,
        "dataset_names": {},
        "splits": {},
        "instance_ids": [],
        "top_level_keys": {},
        "status": "missing",
        "detail": "",
    }
    if not path.exists():
        result["detail"] = "not visible"
        return result
    if not path.is_file():
        result["status"] = "not_file"
        result["detail"] = "path is visible but not a file"
        return result
    try:
        result["size_bytes"] = path.stat().st_size
    except OSError:
        pass
    if count_lines:
        result["line_count"] = line_count(path)

    dataset_names: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    keys: Counter[str] = Counter()
    instance_ids: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if result["sampled"] >= sample_lines:
                    break
                text = line.strip()
                if not text:
                    continue
                result["sampled"] += 1
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    result["invalid_json"] += 1
                    continue
                if not isinstance(record, dict):
                    continue
                result["json_records"] += 1
                keys.update(record.keys())
                params = record.get("responses_create_params")
                if not isinstance(params, dict):
                    continue
                result["nemogym_records"] += 1
                metadata = params.get("metadata")
                if isinstance(metadata, dict):
                    instance_id = metadata.get("instance_id")
                    if isinstance(instance_id, str) and instance_id and len(instance_ids) < 5:
                        instance_ids.append(instance_id)
                    dataset_name = metadata.get("dataset_name")
                    if isinstance(dataset_name, str):
                        dataset_names[dataset_name] += 1
                    split = metadata.get("split")
                    if isinstance(split, str):
                        splits[split] += 1
                    if any(key in metadata for key in ("problem_statement", "base_commit", "repo", "golden_patch")):
                        result["swe_like_records"] += 1
                        missing = [key for key in REQUIRED_SWE_METADATA_KEYS if metadata.get(key) in (None, "")]
                        instance_raw = metadata.get("instance_dict")
                        if not missing and isinstance(instance_raw, str):
                            try:
                                if isinstance(json.loads(instance_raw), dict):
                                    result["valid_swe_records"] += 1
                                else:
                                    result["invalid_swe_metadata"] += 1
                            except json.JSONDecodeError:
                                result["invalid_swe_metadata"] += 1
                        else:
                            result["invalid_swe_metadata"] += 1
    except Exception as exc:
        result["status"] = "error"
        result["detail"] = str(exc)
        return result

    result["dataset_names"] = dict(dataset_names)
    result["splits"] = dict(splits)
    result["instance_ids"] = instance_ids
    result["top_level_keys"] = dict(keys)
    is_swe_dataset = any("swe" in name.lower() for name in dataset_names)
    if result["nemogym_records"] and result["valid_swe_records"]:
        result["status"] = "swe_candidate"
        result["detail"] = "sampled records match complete SWE-agent NemoGym input"
    elif result["nemogym_records"] and (result["swe_like_records"] or is_swe_dataset):
        result["status"] = "swe_candidate_incomplete"
        result["detail"] = "sampled records are SWE-like but miss required SWE-agent metadata"
    elif result["nemogym_records"]:
        result["status"] = "generic_nemogym"
        result["detail"] = "sampled records match NemoGym format but not SWE/R2E metadata"
    elif result["json_records"]:
        result["status"] = "jsonl_not_nemogym"
        result["detail"] = "JSONL is visible but not NemoGym SWE-agent input"
    else:
        result["status"] = "invalid_or_empty"
        result["detail"] = "no JSON object records found in sample"
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SWE NemoGym Dataset Discovery",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Roots scanned: **{len(payload['roots'])}**",
        f"Files scanned: **{payload['files_scanned']}**",
        f"SWE/R2E candidates: **{payload['candidate_count']}**",
        f"Incomplete SWE/R2E candidates: **{payload['incomplete_candidate_count']}**",
        f"Generic NemoGym files: **{payload['generic_nemogym_count']}**",
        "",
        "## Expected Paths",
        "",
        "| status | path | detail |",
        "| --- | --- | --- |",
    ]
    for item in payload["expected_paths"]:
        lines.append(f"| `{item['status']}` | `{item['path']}` | {item['detail']} |")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| rank | status | lines | size | swe-like/sample | dataset | split | path |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for idx, item in enumerate(payload["candidates"], 1):
        dataset = ", ".join(item.get("dataset_names", {}).keys()) or "-"
        split = ", ".join(item.get("splits", {}).keys()) or "-"
        lines.append(
            f"| {idx} | `{item['status']}` | {item.get('line_count') or ''} | {item.get('size_bytes') or ''} | "
            f"{item.get('swe_like_records', 0)}/{item.get('sampled', 0)} | {dataset} | {split} | `{item['path']}` |"
        )
    if not payload["candidates"]:
        lines.append("| - | - | - | - | - | - | - | no candidates found |")
    lines.extend(
        [
            "",
            "## Incomplete SWE-Like Inputs",
            "",
            "| rank | status | lines | size | valid/sample | dataset | split | path |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for idx, item in enumerate(payload.get("incomplete_candidates", []), 1):
        dataset = ", ".join(item.get("dataset_names", {}).keys()) or "-"
        split = ", ".join(item.get("splits", {}).keys()) or "-"
        lines.append(
            f"| {idx} | `{item['status']}` | {item.get('line_count') or ''} | {item.get('size_bytes') or ''} | "
            f"{item.get('valid_swe_records', 0)}/{item.get('sampled', 0)} | {dataset} | {split} | `{item['path']}` |"
        )
    if not payload.get("incomplete_candidates"):
        lines.append("| - | - | - | - | - | - | - | no incomplete candidates found |")
    lines.extend(
        [
            "",
            "## Generic NemoGym Sample",
            "",
            "| status | lines | size | path |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for item in payload.get("generic_nemogym_sample", []):
        lines.append(
            f"| `{item['status']}` | {item.get('line_count') or ''} | {item.get('size_bytes') or ''} | `{item['path']}` |"
        )
    if not payload.get("generic_nemogym_sample"):
        lines.append("| - | - | - | no generic NemoGym files found |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    expected = [inspect_jsonl(path, args.sample_lines, args.count_lines) for path in args.expected_path]
    files = collect_files(args.roots, args.name_hint, args.max_depth, args.max_files)
    inspected = [inspect_jsonl(path, args.sample_lines, args.count_lines) for path in files]
    candidates = [item for item in inspected if item["status"] == "swe_candidate"]
    incomplete_candidates = [item for item in inspected if item["status"] == "swe_candidate_incomplete"]
    generic_nemogym = [item for item in inspected if item["status"] == "generic_nemogym"]
    candidates.sort(
        key=lambda item: (
            item.get("swe_like_records", 0),
            item.get("nemogym_records", 0),
            item.get("size_bytes") or 0,
        ),
        reverse=True,
    )
    full_candidates = [
        item
        for item in candidates
        if isinstance(item.get("line_count"), int) and item["line_count"] >= args.min_full_lines
    ]
    if full_candidates:
        overall = "full_candidate_found"
    elif candidates:
        overall = "smoke_only"
    else:
        overall = "missing"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "roots": [str(path) for path in args.roots],
        "name_hints": args.name_hint,
        "max_depth": args.max_depth,
        "min_full_lines": args.min_full_lines,
        "files_scanned": len(files),
        "candidate_count": len(candidates),
        "incomplete_candidate_count": len(incomplete_candidates),
        "full_candidate_count": len(full_candidates),
        "generic_nemogym_count": len(generic_nemogym),
        "expected_paths": expected,
        "candidates": candidates[:50],
        "incomplete_candidates": incomplete_candidates[:50],
        "generic_nemogym_sample": generic_nemogym[:25],
        "non_candidates_sample": [
            item
            for item in inspected
            if item["status"] not in {"swe_candidate", "generic_nemogym"}
        ][:25],
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
