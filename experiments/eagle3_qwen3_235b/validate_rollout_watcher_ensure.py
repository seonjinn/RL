#!/usr/bin/env python3
"""Validate rollout watcher ensure behavior with synthetic reports.

This is a no-submit test for ensure_rollout_watchers.py. It verifies the
operator guard for three states:

- active rollout watchers are alive and no action is needed.
- a required watcher is missing and a restart command is emitted.
- a queue timeout risk exists and lock-waiting extension commands are emitted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/ensure_rollout_watchers.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(root: Path) -> None:
    write_json(
        root / "reports/rollout_capture_state_advance.json",
        {
            "artifact_root": str(root),
            "repo_root": "/tmp/specdec-rl",
            "rollout_log_dir": str(root / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke"),
            "output_data": str(root / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
        },
    )


def write_generic_state(root: Path, job_id: str) -> None:
    write_json(
        root / f"reports/rollout_capture_vllm0102_{job_id}_swegym_state_advance.json",
        {
            "artifact_root": str(root),
            "repo_root": "/tmp/specdec-rl",
            "rollout_log_dir": str(root / f"rl_rollout_capture_logs/qwen3_235b_swe_capture_{job_id}"),
            "output_data": str(root / f"data/qwen3_235b_swe_rollout_conversations_{job_id}.jsonl"),
        },
    )


def write_queue(root: Path, risk: str) -> None:
    write_json(
        root / "reports/rollout_queue_wait_summary.json",
        {
            "overall_status": "warn" if risk == "risk" else "waiting",
            "jobs": [
                {
                    "job_id": "12345",
                    "current_squeue": {
                        "job_id": "12345",
                        "name": "qwen3-235b-swe-rollout-capture-smoke",
                        "state": "PENDING",
                        "elapsed": "0:00",
                        "nodes": "32",
                        "reason": "(Priority)",
                        "start": "2026-05-22T09:00:00",
                    },
                    "watcher_timeout": {
                        "risk": risk,
                        "terminal_buffer_minutes": 180,
                    },
                }
            ],
        },
    )


def write_queue_multiple(root: Path) -> None:
    write_json(
        root / "reports/rollout_queue_wait_summary.json",
        {
            "overall_status": "waiting",
            "jobs": [
                {
                    "job_id": "12345",
                    "current_squeue": {
                        "job_id": "12345",
                        "name": "qwen3-235b-swe-rollout-capture-smoke",
                        "state": "PENDING",
                        "elapsed": "0:00",
                        "nodes": "32",
                        "reason": "(Resources)",
                        "start": "2026-05-22T09:00:00",
                    },
                    "watcher_timeout": {"risk": "ok"},
                },
                {
                    "job_id": "23456",
                    "current_squeue": {
                        "job_id": "23456",
                        "name": "qwen3-235b-swe-rollout-vllm0102-balanced24n4g",
                        "state": "PENDING",
                        "elapsed": "0:00",
                        "nodes": "24",
                        "reason": "(Priority)",
                        "start": "2026-05-22T08:30:00",
                    },
                    "watcher_timeout": {"risk": "ok"},
                },
            ],
        },
    )


def write_health(
    root: Path,
    *,
    materialize_alive: bool,
    pending_alive: bool,
    generic_alive: bool = False,
    arbitration_alive: bool = False,
    pipeline_ready_alive: bool = True,
) -> None:
    watchers = []
    if materialize_alive:
        watchers.append(
            {
                "label": "official materialize watcher",
                "required_now": True,
                "status": "alive",
                "pid": 111,
            }
        )
    else:
        watchers.append(
            {
                "label": "official materialize watcher",
                "required_now": True,
                "status": "missing",
            }
        )
    if pending_alive:
        watchers.append(
            {
                "label": "official pending-state watcher",
                "required_now": True,
                "status": "alive",
                "pid": 222,
            }
        )
    else:
        watchers.append(
            {
                "label": "official pending-state watcher",
                "required_now": True,
                "status": "missing",
            }
        )
    if generic_alive:
        watchers.append(
            {
                "label": "generic materialize watcher 23456",
                "required_now": True,
                "status": "alive",
                "pid": 333,
            }
        )
    if arbitration_alive:
        watchers.append(
            {
                "label": "rollout job arbitration watcher",
                "required_now": True,
                "status": "alive",
                "pid": 444,
            }
        )
    if pipeline_ready_alive:
        watchers.append(
            {
                "label": "pipeline ready-submit watcher",
                "required_now": True,
                "status": "alive",
                "pid": 555,
            }
        )
    write_json(
        root / "reports/rollout_watcher_health.json",
        {
            "overall_status": "pass",
            "watchers": watchers,
        },
    )


def run_ensure(root: Path) -> tuple[dict[str, Any], str]:
    out = root / "reports/rollout_watcher_ensure.json"
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
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"ensure helper crashed:\n{result.stdout}")
    return read_json(out), result.stdout


def scenario(root: Path, name: str, *, risk: str, materialize_alive: bool, pending_alive: bool) -> dict[str, Any]:
    scenario_root = root / name / "qwen3_235b_eagle3"
    write_state(scenario_root)
    write_queue(scenario_root, risk)
    write_health(scenario_root, materialize_alive=materialize_alive, pending_alive=pending_alive)
    payload, output = run_ensure(scenario_root)
    rows = payload.get("rows") or []
    return {
        "name": name,
        "payload": payload,
        "output_tail": output[-4000:],
        "rows": rows,
    }


def arbitration_scenario(root: Path) -> dict[str, Any]:
    scenario_root = root / "missing_arbitration_restart" / "qwen3_235b_eagle3"
    write_state(scenario_root)
    write_generic_state(scenario_root, "23456")
    write_queue_multiple(scenario_root)
    write_health(scenario_root, materialize_alive=True, pending_alive=True, generic_alive=True)
    payload, output = run_ensure(scenario_root)
    rows = payload.get("rows") or []
    return {
        "name": "missing_arbitration_restart",
        "payload": payload,
        "output_tail": output[-4000:],
        "rows": rows,
    }


def pipeline_ready_scenario(root: Path) -> dict[str, Any]:
    scenario_root = root / "missing_pipeline_ready_submit_restart" / "qwen3_235b_eagle3"
    write_state(scenario_root)
    write_queue(scenario_root, "ok")
    write_health(scenario_root, materialize_alive=True, pending_alive=True, pipeline_ready_alive=False)
    payload, output = run_ensure(scenario_root)
    rows = payload.get("rows") or []
    return {
        "name": "missing_pipeline_ready_submit_restart",
        "payload": payload,
        "output_tail": output[-4000:],
        "rows": rows,
    }


def check_pass_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    problems: list[str] = []
    if payload.get("overall_status") != "pass":
        problems.append(f"overall {payload.get('overall_status')!r} != 'pass'")
    for key in ["restart_needed_count", "extension_needed_count", "action_needed_count", "started_count"]:
        if payload.get(key) != 0:
            problems.append(f"{key} {payload.get(key)!r} != 0")
    return problems


def check_pipeline_ready_restart_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    rows = item["rows"]
    problems: list[str] = []
    if payload.get("overall_status") != "restart_recommended":
        problems.append(f"overall {payload.get('overall_status')!r} != 'restart_recommended'")
    if payload.get("restart_needed_count") != 1:
        problems.append(f"restart_needed_count {payload.get('restart_needed_count')!r} != 1")
    commands = [row.get("command") or "" for row in rows if row.get("restart_needed")]
    if len(commands) != 1:
        problems.append(f"expected 1 restart command, saw {len(commands)}")
    elif "watch_eagle3_pipeline_ready_submit.sh" not in commands[0]:
        problems.append("restart command does not start pipeline-ready submit watcher")
    return problems


def check_arbitration_restart_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    rows = item["rows"]
    problems: list[str] = []
    if payload.get("overall_status") != "restart_recommended":
        problems.append(f"overall {payload.get('overall_status')!r} != 'restart_recommended'")
    if payload.get("restart_needed_count") != 1:
        problems.append(f"restart_needed_count {payload.get('restart_needed_count')!r} != 1")
    commands = [row.get("command") or "" for row in rows if row.get("restart_needed")]
    if len(commands) != 1:
        problems.append(f"expected 1 restart command, saw {len(commands)}")
    elif "watch_rollout_job_arbitration.sh" not in commands[0]:
        problems.append("restart command does not start arbitration watcher")
    return problems


def check_restart_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    rows = item["rows"]
    problems: list[str] = []
    if payload.get("overall_status") != "restart_recommended":
        problems.append(f"overall {payload.get('overall_status')!r} != 'restart_recommended'")
    if payload.get("restart_needed_count") != 1:
        problems.append(f"restart_needed_count {payload.get('restart_needed_count')!r} != 1")
    commands = [row.get("command") or "" for row in rows if row.get("restart_needed")]
    if not commands:
        problems.append("missing restart command")
    elif any("WAIT_FOR_LOCK=true" in command for command in commands):
        problems.append("restart command unexpectedly waits for lock")
    return problems


def check_extension_state(item: dict[str, Any]) -> list[str]:
    payload = item["payload"]
    rows = item["rows"]
    problems: list[str] = []
    if payload.get("overall_status") != "watch_deadline_risk":
        problems.append(f"overall {payload.get('overall_status')!r} != 'watch_deadline_risk'")
    if payload.get("extension_needed_count") != 2:
        problems.append(f"extension_needed_count {payload.get('extension_needed_count')!r} != 2")
    if payload.get("action_needed_count") != 2:
        problems.append(f"action_needed_count {payload.get('action_needed_count')!r} != 2")
    commands = [row.get("command") or "" for row in rows if row.get("extension_needed")]
    if len(commands) != 2:
        problems.append(f"expected 2 extension commands, saw {len(commands)}")
    if any("WAIT_FOR_LOCK=true" not in command for command in commands):
        problems.append("extension command missing WAIT_FOR_LOCK=true")
    if not any("watch_rollout_capture_materialize.sh" in command for command in commands):
        problems.append("missing materialize extension command")
    if not any("watch_rollout_pending_state_refresh.sh" in command for command in commands):
        problems.append("missing pending-state extension command")
    return problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Watcher Ensure Validation",
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
    temp_root = Path(tempfile.mkdtemp(prefix="rollout_watcher_ensure_validation_"))
    try:
        scenarios = [
            scenario(temp_root, "all_alive_ok", risk="ok", materialize_alive=True, pending_alive=True),
            scenario(temp_root, "missing_materialize_restart", risk="ok", materialize_alive=False, pending_alive=True),
            scenario(temp_root, "deadline_risk_extension", risk="risk", materialize_alive=True, pending_alive=True),
            arbitration_scenario(temp_root),
            pipeline_ready_scenario(temp_root),
        ]
        checks = [
            check_pass_state(scenarios[0]),
            check_restart_state(scenarios[1]),
            check_extension_state(scenarios[2]),
            check_arbitration_restart_state(scenarios[3]),
            check_pipeline_ready_restart_state(scenarios[4]),
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
