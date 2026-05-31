#!/usr/bin/env python3
"""Validate pipeline-submit preflight row-count gates.

This is a lightweight no-submit contract test. It imports
preflight_eagle3_pipeline_submit.py and exercises only the training-conversation
validation gate, proving that a 5-row smoke corpus cannot make RUN_PILOT=true
submit-ready while an 8-row pilot corpus satisfies the row-count guard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def load_preflight_module() -> Any:
    spec = importlib.util.spec_from_file_location("preflight_eagle3_pipeline_submit_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_conversations(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for idx in range(rows):
            record = {
                "conversation_id": f"row-{idx}",
                "messages": [
                    {"role": "user", "content": f"solve item {idx}"},
                    {"role": "assistant", "content": f"answer {idx}"},
                ],
            }
            fh.write(json.dumps(record) + "\n")


def run_case(module: Any, root: Path, name: str, rows: int, min_rows: int = 8) -> dict[str, Any]:
    input_data = root / f"{name}.jsonl"
    write_conversations(input_data, rows)
    checks: list[dict[str, Any]] = []
    args = SimpleNamespace(input_data=input_data, run_pilot="true", min_pilot_rows=min_rows)
    result = module.check_training_conversations(checks, args)
    pilot_checks = [check for check in checks if check.get("name") == "pilot minimum rows"]
    return {
        "name": name,
        "rows": rows,
        "min_rows": min_rows,
        "returncode": result.get("returncode"),
        "checks": checks,
        "pilot_check": pilot_checks[-1] if pilot_checks else None,
    }


def check_case(item: dict[str, Any], expected_status: str) -> list[str]:
    problems: list[str] = []
    pilot_check = item.get("pilot_check") or {}
    if pilot_check.get("status") != expected_status:
        problems.append(f"pilot minimum rows status {pilot_check.get('status')!r} != {expected_status!r}")
    if item.get("returncode") != 0:
        problems.append(f"conversation validation returncode {item.get('returncode')!r} != 0")
    return problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Pipeline Submit Preflight Contract",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in payload["scenarios"]:
        detail = "; ".join(item["problems"]) if item["problems"] else "-"
        lines.append(f"| {item['name']} | {item['status']} | {detail.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="pipeline_submit_preflight_contract_"))
    try:
        module = load_preflight_module()
        raw_cases = [
            run_case(module, temp_root, "smoke_five_rows_rejected", rows=5),
            run_case(module, temp_root, "pilot_eight_rows_allowed", rows=8),
        ]
        checks = [
            check_case(raw_cases[0], "fail"),
            check_case(raw_cases[1], "pass"),
        ]
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    rendered: list[dict[str, Any]] = []
    problems: list[str] = []
    for item, item_problems in zip(raw_cases, checks, strict=True):
        rendered.append(
            {
                "name": item["name"],
                "status": "pass" if not item_problems else "fail",
                "problems": item_problems,
            }
        )
        problems.extend(f"{item['name']}: {problem}" for problem in item_problems)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "pass" if not problems else "fail",
        "scenarios": rendered,
        "problems": problems,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if payload["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
