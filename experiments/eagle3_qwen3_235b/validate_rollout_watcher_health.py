#!/usr/bin/env python3
"""Validate rollout watcher health behavior with synthetic reports.

This is a no-submit test for summarize_rollout_watcher_health.py. It verifies
that an active generic rollout requires both the generic materialize watcher and
pipeline ready-submit watcher, and that the dynamic active rollout state report
is included in freshness checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/summarize_rollout_watcher_health.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pid(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_active_generic_fixture(root: Path, *, stale_state: bool, extension_launcher: bool = False) -> str:
    reports = root / "reports"
    job_id = "23456"
    prefix = f"rollout_capture_vllm0102_{job_id}_swegym"
    write_json(
        reports / "rollout_queue_wait_summary.json",
        {
            "overall_status": "waiting",
            "jobs": [
                {
                    "job_id": job_id,
                    "current_squeue": {
                        "job_id": job_id,
                        "name": "qwen3-235b-swe-rollout-vllm0102-balanced24n4g",
                        "state": "PENDING",
                        "elapsed": "0:00",
                        "nodes": "24",
                        "reason": "(Priority)",
                        "start": "2026-05-22T09:00:00",
                    },
                    "watcher_timeout": {"risk": "ok"},
                }
            ],
        },
    )
    state_path = reports / f"{prefix}_state_advance.json"
    write_json(
        state_path,
        {
            "artifact_root": str(root),
            "repo_root": "/tmp/specdec-rl",
            "rollout_log_dir": str(root / "rl_rollout_capture_logs/smoke"),
            "output_data": str(root / "data/qwen3_235b_swe_rollout_conversations_23456.jsonl"),
            "decision": {"overall_status": "running", "next_step": "poll"},
            "job": {"job_id": job_id, "slurm": {"job_id": job_id, "state": "PENDING"}},
        },
    )
    if stale_state:
        old = time.time() - 1800
        os.utime(state_path, (old, old))
    write_pid(reports / f"{prefix}_watch.pid")
    write_pid(reports / "eagle3_pipeline_ready_submit_watch.pid")
    if extension_launcher:
        write_pid(reports / f"{prefix}_watch_extension_launcher.pid")
    return state_path.stem.replace("_state_advance", " state")


def run_health(root: Path) -> tuple[dict[str, Any], str]:
    out = root / "reports/rollout_watcher_health.json"
    md = out.with_suffix(".md")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-root",
            str(root),
            "--json-out",
            str(out),
            "--markdown-out",
            str(md),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"health helper crashed:\n{result.stdout}")
    return read_json(out), result.stdout


def scenario(temp_root: Path, name: str, *, stale_state: bool, extension_launcher: bool = False) -> dict[str, Any]:
    root = temp_root / name / "qwen3_235b_eagle3"
    expected_label = write_active_generic_fixture(root, stale_state=stale_state, extension_launcher=extension_launcher)
    payload, output = run_health(root)
    return {
        "name": name,
        "expected_label": expected_label,
        "payload": payload,
        "output_tail": output[-4000:],
    }


def check_fresh_generic_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    problems: list[str] = []
    if payload.get("overall_status") != "pass":
        problems.append(f"overall {payload.get('overall_status')!r} != 'pass'")
    labels = {watcher.get("label"): watcher for watcher in payload.get("watchers") or []}
    for label in ["generic materialize watcher 23456", "pipeline ready-submit watcher"]:
        watcher = labels.get(label)
        if not watcher:
            problems.append(f"missing watcher row {label!r}")
        elif watcher.get("required_now") is not True or watcher.get("status") != "alive":
            problems.append(f"watcher {label!r} not required/alive: {watcher}")
    if payload.get("stale_reports"):
        problems.append(f"unexpected stale reports: {payload.get('stale_reports')}")
    return problems


def check_stale_generic_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    problems: list[str] = []
    expected_label = item["expected_label"]
    if payload.get("overall_status") != "warn":
        problems.append(f"overall {payload.get('overall_status')!r} != 'warn'")
    stale = payload.get("stale_reports") or []
    if expected_label not in stale:
        problems.append(f"stale reports {stale!r} missing {expected_label!r}")
    if payload.get("dead_or_missing_required_watchers"):
        problems.append(f"unexpected watcher liveness issues: {payload.get('dead_or_missing_required_watchers')}")
    return problems


def check_extension_launcher_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    problems = check_fresh_generic_state(item)
    labels = {watcher.get("label"): watcher for watcher in payload.get("watchers") or []}
    label = "generic current-code extension launcher 23456"
    watcher = labels.get(label)
    if not watcher:
        problems.append(f"missing watcher row {label!r}")
    elif watcher.get("required_now") is not True or watcher.get("status") != "alive":
        problems.append(f"watcher {label!r} not required/alive: {watcher}")
    return problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Watcher Health Validation",
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
    temp_root = Path(tempfile.mkdtemp(prefix="rollout_watcher_health_validation_"))
    try:
        scenarios = [
            scenario(temp_root, "fresh_active_generic_state", stale_state=False),
            scenario(temp_root, "stale_active_generic_state", stale_state=True),
            scenario(temp_root, "active_generic_extension_launcher", stale_state=False, extension_launcher=True),
        ]
        checks = [
            check_fresh_generic_state(scenarios[0]),
            check_stale_generic_state(scenarios[1]),
            check_extension_launcher_state(scenarios[2]),
        ]
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    rendered: list[dict[str, Any]] = []
    problems: list[str] = []
    for item, item_problems in zip(scenarios, checks, strict=True):
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
