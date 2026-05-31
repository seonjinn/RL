#!/usr/bin/env python3
"""Safely run or print one action from eagle3_next_actions.json.

Default behavior is print-only. To execute a Slurm-submit action, the caller
must pass both --execute and --allow-slurm. Actions marked heavy_gpu also
require --allow-heavy-gpu.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
    )
    parser.add_argument("--action-id", help="Action id from next_actions. Defaults to the first ready action with a command.")
    parser.add_argument("--index", type=int, help="1-based action index from next_actions.")
    parser.add_argument("--list", action="store_true", help="List available actions and exit.")
    parser.add_argument("--execute", action="store_true", help="Actually run the selected command.")
    parser.add_argument("--allow-slurm", action="store_true", help="Allow actions marked submits_slurm=true.")
    parser.add_argument("--allow-heavy-gpu", action="store_true", help="Allow actions marked heavy_gpu=true.")
    parser.add_argument("--allow-non-ready", action="store_true", help="Allow executing an action whose status is not ready_for_operator.")
    parser.add_argument(
        "--run-after",
        action="store_true",
        help="After a successful non-Slurm action execution, run its after_commands.",
    )
    parser.add_argument(
        "--allow-run-after-for-slurm",
        action="store_true",
        help="Override the guard that prevents running analyzers immediately after Slurm submission.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional execution record JSON path.")
    args = parser.parse_args()
    if args.plan_json is None:
        args.plan_json = Path(
            os.environ.get("NEXT_ACTION_PLAN_JSON", args.artifact_root / "reports/eagle3_next_actions.json")
        )
    return args


def load_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"next-action plan is not visible: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot parse next-action plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"next-action plan is not a JSON object: {path}")
    return payload


def actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("next_actions") or []
    return [item for item in raw if isinstance(item, dict)]


def select_action(items: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if args.action_id:
        for item in items:
            if item.get("id") == args.action_id:
                return item
        raise SystemExit(f"action id not found: {args.action_id}")
    if args.index is not None:
        if args.index < 1 or args.index > len(items):
            raise SystemExit(f"action index out of range: {args.index}")
        return items[args.index - 1]
    for item in items:
        if item.get("status") == "ready_for_operator" and item.get("command"):
            return item
    raise SystemExit("no ready action with a command is available")


def print_actions(plan: dict[str, Any], items: list[dict[str, Any]]) -> None:
    print(f"Plan: {plan.get('overall_status')}  artifact_root={plan.get('artifact_root')}")
    if not items:
        print("No next actions.")
        return
    for idx, item in enumerate(items, 1):
        print(
            f"{idx}. {item.get('id')} | status={item.get('status')} | "
            f"slurm={item.get('submits_slurm')} | heavy_gpu={item.get('heavy_gpu')}"
        )
        print(f"   {item.get('title')}")
        if item.get("command"):
            print(f"   command: {item.get('command')}")
        after = item.get("after_commands") or []
        for command in after:
            print(f"   after: {command}")


def validate_action(item: dict[str, Any], args: argparse.Namespace) -> None:
    if not item.get("command"):
        raise SystemExit(f"selected action has no command: {item.get('id')}")
    if item.get("status") != "ready_for_operator" and not args.allow_non_ready:
        raise SystemExit(
            f"selected action status is {item.get('status')!r}; pass --allow-non-ready to override"
        )
    if item.get("submits_slurm") and not args.allow_slurm:
        raise SystemExit("selected action submits Slurm; pass --allow-slurm to execute it")
    if item.get("heavy_gpu") and not args.allow_heavy_gpu:
        raise SystemExit("selected action may spend heavy GPU time; pass --allow-heavy-gpu to execute it")
    if item.get("submits_slurm") and args.run_after and not args.allow_run_after_for_slurm:
        raise SystemExit(
            "selected action submits Slurm; do not use --run-after until the Slurm job reaches a terminal state. "
            "Run the printed after_commands manually after checking the job state, or pass "
            "--allow-run-after-for-slurm to override."
        )


def run_shell(command: str) -> int:
    result = subprocess.run(command, cwd=ROOT, shell=True, check=False)
    return result.returncode


def write_record(path: Path | None, record: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def default_execution_record_path(args: argparse.Namespace, item: dict[str, Any]) -> Path | None:
    if args.json_out:
        return args.json_out
    if not args.execute:
        return None
    action_id = str(item.get("id") or "selected_action")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", action_id).strip("_") or "selected_action"
    return args.artifact_root / "reports" / "operator_execution" / f"{safe_id}.json"


def main() -> int:
    args = parse_args()
    plan = load_plan(args.plan_json)
    items = actions(plan)
    if args.list:
        print_actions(plan, items)
        return 0

    item = select_action(items, args)
    print_actions(plan, [item])
    json_out = default_execution_record_path(args, item)
    started = time.time()

    record: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(started)),
        "started_at_epoch": started,
        "completed_at": None,
        "completed_at_epoch": None,
        "duration_seconds": None,
        "host": socket.gethostname(),
        "cwd": str(ROOT),
        "plan_json": str(args.plan_json),
        "artifact_root": str(args.artifact_root),
        "json_out": str(json_out) if json_out else None,
        "action": item,
        "mode": "execute" if args.execute else "print_only",
        "returncode": None,
        "after_returncodes": [],
        "after_policy": "after_slurm_terminal_state" if item.get("submits_slurm") else "after_command_success",
    }

    if not args.execute:
        print("\nPrint-only mode. Re-run with --execute plus required allow flags to run the selected action.")
        completed = time.time()
        record["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(completed))
        record["completed_at_epoch"] = completed
        record["duration_seconds"] = round(completed - started, 3)
        write_record(json_out, record)
        return 0

    validate_action(item, args)
    rc = run_shell(str(item["command"]))
    record["returncode"] = rc
    if rc == 0 and args.run_after:
        for command in item.get("after_commands") or []:
            after_rc = run_shell(str(command))
            record["after_returncodes"].append({"command": command, "returncode": after_rc})
            if after_rc != 0:
                rc = after_rc
                break
    elif item.get("after_commands"):
        if item.get("submits_slurm"):
            print("\nAfter the submitted Slurm job reaches a terminal state, run:")
        else:
            print("\nAfter the action completes, run:")
        for command in item.get("after_commands") or []:
            print(command)

    completed = time.time()
    record["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(completed))
    record["completed_at_epoch"] = completed
    record["duration_seconds"] = round(completed - started, 3)
    write_record(json_out, record)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
