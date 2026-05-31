#!/usr/bin/env python3
"""Validate rollout-submit preflight safety contracts.

This is a no-submit unit-style validator for preflight_rollout_capture_submit.py.
It proves that source-built vLLM rollout names cannot pass without the runtime
passthrough env that keeps the NeMo-RL rollout on the source-built vLLM site.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/preflight_rollout_capture_submit.py"
SOURCE_ENV_KEYS = (
    "SHARED_VLLM_SITE",
    "VLLM_PIP_SPEC",
    "VLLM_ENFORCE_EAGER",
    "VLLM_COMPILATION_LEVEL",
    "VLLM_USE_INDUCTOR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def load_preflight_module() -> Any:
    spec = importlib.util.spec_from_file_location("preflight_rollout_capture_submit_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_case(
    module: Any,
    temp_root: Path,
    name: str,
    wandb_name: str,
    env_updates: dict[str, str],
    expected_check_status: str,
    require_source: bool = False,
) -> dict[str, Any]:
    snapshot = {key: os.environ.get(key) for key in (*SOURCE_ENV_KEYS, "REQUIRE_SOURCE_VLLM_ENV")}
    for key in (*SOURCE_ENV_KEYS, "REQUIRE_SOURCE_VLLM_ENV"):
        os.environ.pop(key, None)
    os.environ.update(env_updates)
    try:
        args = SimpleNamespace(wandb_name=wandb_name, require_source_vllm_env=require_source)
        rollout_log_dir = temp_root / name / "logs"
        output_conversations = temp_root / name / "data/out.jsonl"
        checks: list[dict[str, Any]] = []
        result = module.check_source_vllm_env(checks, args, rollout_log_dir, output_conversations)
    finally:
        restore_env(snapshot)

    source_checks = [check for check in checks if check.get("name") == "source-built vLLM env"]
    check = source_checks[-1] if source_checks else {}
    ok = check.get("status") == expected_check_status
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "expected_check_status": expected_check_status,
        "actual_check_status": check.get("status"),
        "result": result,
        "check": check,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Submit Preflight Contract",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | status | expected | actual | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload["scenarios"]:
        detail = (item.get("check") or {}).get("detail") or "-"
        lines.append(
            f"| {item['name']} | {item['status'].upper()} | `{item['expected_check_status']}` | "
            f"`{item.get('actual_check_status')}` | {str(detail).replace('|', '/')} |"
        )
    if payload["problems"]:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in payload["problems"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="rollout_submit_preflight_contract_"))
    try:
        module = load_preflight_module()
        source_site = temp_root / "vllm_site"
        source_site.mkdir(parents=True)
        good_env = {
            "SHARED_VLLM_SITE": str(source_site),
            "VLLM_PIP_SPEC": "https://example.invalid/vllm-0.10.2.tar.gz",
            "VLLM_ENFORCE_EAGER": "True",
            "VLLM_COMPILATION_LEVEL": "0",
            "VLLM_USE_INDUCTOR": "False",
        }
        scenarios = [
            run_case(
                module,
                temp_root,
                "generic_name_no_env_allowed",
                "qwen3-235b-swe-rollout-smoke",
                {},
                "pass",
            ),
            run_case(
                module,
                temp_root,
                "source_name_missing_env_fails",
                "qwen3-235b-swe-rollout-vllm0102src-swegym-full",
                {},
                "fail",
            ),
            run_case(
                module,
                temp_root,
                "source_name_bad_compile_env_fails",
                "qwen3-235b-swe-rollout-vllm0102src-swegym-full",
                {**good_env, "VLLM_USE_INDUCTOR": "True"},
                "fail",
            ),
            run_case(
                module,
                temp_root,
                "source_name_good_env_passes",
                "qwen3-235b-swe-rollout-vllm0102src-swegym-full",
                good_env,
                "pass",
            ),
            run_case(
                module,
                temp_root,
                "explicit_require_missing_env_fails",
                "qwen3-235b-swe-rollout-smoke",
                {},
                "fail",
                require_source=True,
            ),
        ]
    finally:
        if args.keep_temp:
            print(f"Kept temp root: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    problems = [
        f"{item['name']}: expected {item['expected_check_status']} got {item.get('actual_check_status')}"
        for item in scenarios
        if item["status"] != "pass"
    ]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "pass" if not problems else "fail",
        "scenarios": scenarios,
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
