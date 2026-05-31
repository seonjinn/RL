#!/usr/bin/env python3
"""Validate Eagle3 operator queue transitions with synthetic reports.

This is a no-submit test for summarize_eagle3_operator_queue.py. It builds
temporary operator sheets, execution summaries, follow-up guard reports, and
ready-submit preflight reports, then checks that the queue emits the expected
next step for each operator state.
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
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/summarize_eagle3_operator_queue.py"


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


def action_record(
    root: Path,
    action_id: str,
    order: int,
    heavy_gpu: bool = False,
    *,
    submits_slurm: bool = True,
    stage: str | None = None,
) -> dict[str, Any]:
    followup = root / "reports/operator_followups" / f"{order:02d}_{action_id}.json"
    execute_flags = "--execute"
    if submits_slurm:
        execute_flags += " --allow-slurm"
    if heavy_gpu:
        execute_flags += " --allow-heavy-gpu"
    return {
        "order": order,
        "id": action_id,
        "title": action_id.replace("_", " "),
        "stage": stage or ("rollout_capture" if heavy_gpu else "container_gate"),
        "status": "ready_for_operator",
        "submits_slurm": submits_slurm,
        "heavy_gpu": heavy_gpu,
        "execute_command": (
            f"python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py "
            f"--artifact-root {root} --plan-json {root / 'reports/eagle3_next_actions.json'} "
            f"--action-id {action_id} {execute_flags} "
            f"--json-out {root / 'reports/operator_execution' / f'{order:02d}_{action_id}.json'}"
        ),
        "followup_status_command": (
            f"python3 experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py "
            f"--artifact-root {root} --plan-json {root / 'reports/eagle3_next_actions.json'} "
            f"--operator-sheet-json {root / 'reports/eagle3_operator_sheet.json'} "
            f"--action-id {action_id} --execution-record "
            f"{root / 'reports/operator_execution' / f'{order:02d}_{action_id}.json'} "
            f"--json-out {followup} --markdown-out {followup.with_suffix('.md')}"
        ),
        "execute_followup_command": (
            f"python3 experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py "
            f"--artifact-root {root} --plan-json {root / 'reports/eagle3_next_actions.json'} "
            f"--operator-sheet-json {root / 'reports/eagle3_operator_sheet.json'} "
            f"--action-id {action_id} --execution-record "
            f"{root / 'reports/operator_execution' / f'{order:02d}_{action_id}.json'} "
            f"--json-out {followup} --markdown-out {followup.with_suffix('.md')} --execute-after"
        ),
        "execution_record": str(root / "reports/operator_execution" / f"{order:02d}_{action_id}.json"),
        "followup_record": str(followup) if submits_slurm else None,
    }


def base_reports(root: Path) -> dict[str, Path]:
    reports = root / "reports"
    paths = {
        "plan": reports / "eagle3_next_actions.json",
        "sheet": reports / "eagle3_operator_sheet.json",
        "execution": reports / "eagle3_operator_execution.json",
        "followup_validation": reports / "eagle3_operator_followups_validation.json",
        "ready_preflight": reports / "eagle3_operator_ready_submit_preflight.json",
        "queue": reports / "eagle3_operator_queue.json",
    }
    actions = [
        action_record(root, "submit_container_preflight", 1, heavy_gpu=False),
        action_record(root, "submit_rollout_capture", 2, heavy_gpu=True),
    ]
    write_json(
        paths["plan"],
        {
            "overall_status": "ready_for_operator_submit",
            "artifact_root": str(root),
            "next_actions": [
                {
                    "id": action["id"],
                    "status": "ready_for_operator",
                    "stage": action["stage"],
                    "submits_slurm": action["submits_slurm"],
                    "heavy_gpu": action["heavy_gpu"],
                    "command": action["execute_command"],
                }
                for action in actions
            ],
        },
    )
    write_json(
        paths["sheet"],
        {
            "overall_status": "ready_for_operator",
            "artifact_root": str(root),
            "ready_actions": actions,
        },
    )
    write_json(
        paths["ready_preflight"],
        {
            "overall_status": "pass",
            "submit_ready": True,
            "counts": {"pass": 2},
            "ready_actions": [
                {"id": "submit_container_preflight", "submits_slurm": True, "heavy_gpu": False},
                {"id": "submit_rollout_capture", "submits_slurm": True, "heavy_gpu": True},
            ],
        },
    )
    write_json(
        paths["followup_validation"],
        {
            "overall_status": "pass",
            "followup_state_counts": {"not_submitted": 2},
        },
    )
    write_json(
        paths["execution"],
        {
            "overall_status": "not_started",
            "latest_by_action": {},
        },
    )
    return paths


def write_execution(paths: dict[str, Path], action_id: str, returncode: int) -> None:
    write_json(
        paths["execution"],
        {
            "overall_status": "pass" if returncode == 0 else "fail",
            "latest_by_action": {
                action_id: {
                    "path": str(paths["execution"].parent / "operator_execution" / f"01_{action_id}.json"),
                    "returncode": returncode,
                    "completed_at": "2026-05-22 00:00:00 PDT",
                    "after_returncodes": [],
                }
            },
        },
    )


def write_followup(path: Path, action_id: str, status: str) -> None:
    jobs: list[dict[str, Any]]
    after_rows: list[dict[str, Any]] = []
    mode = "inspect_only"
    detail = "synthetic follow-up state"
    if status == "not_submitted":
        jobs = []
    elif status == "waiting":
        jobs = [{"job_id": "12345", "status": "active", "state": "RUNNING", "terminal": False}]
    elif status == "ready_for_followup":
        jobs = [{"job_id": "12345", "status": "terminal", "state": "COMPLETED", "terminal": True}]
    elif status == "pass":
        jobs = [{"job_id": "12345", "status": "terminal", "state": "COMPLETED", "terminal": True}]
        after_rows = [{"command": "python3 analyzer.py", "returncode": 0}]
        mode = "execute_after"
    elif status == "fail":
        jobs = [{"job_id": "12345", "status": "terminal", "state": "FAILED", "terminal": True}]
        detail = "synthetic failure"
    else:
        raise ValueError(f"unknown follow-up status: {status}")
    write_json(
        path,
        {
            "generated_at": "2026-05-22 00:00:00 PDT",
            "artifact_root": str(path.parents[2]),
            "plan_json": str(path.parents[1] / "eagle3_next_actions.json"),
            "operator_sheet_json": str(path.parents[1] / "eagle3_operator_sheet.json"),
            "action_id": action_id,
            "overall_status": status,
            "detail": detail,
            "mode": mode,
            "execution_record": {"status": "pass"},
            "job_files": [],
            "jobs": jobs,
            "after_commands": ["python3 analyzer.py"],
            "after_returncodes": after_rows,
        },
    )


def set_ready_actions(paths: dict[str, Path], root: Path, actions: list[dict[str, Any]]) -> None:
    write_json(
        paths["plan"],
        {
            "overall_status": "ready_for_operator_submit" if actions else "incomplete",
            "artifact_root": str(root),
            "next_actions": [
                {
                    "id": action["id"],
                    "status": "ready_for_operator",
                    "stage": action["stage"],
                    "submits_slurm": action["submits_slurm"],
                    "heavy_gpu": action["heavy_gpu"],
                    "command": action["execute_command"],
                }
                for action in actions
            ],
        },
    )
    write_json(
        paths["sheet"],
        {
            "overall_status": "ready_for_operator" if actions else "no_ready_actions",
            "artifact_root": str(root),
            "ready_actions": actions,
        },
    )
    write_json(
        paths["ready_preflight"],
        {
            "overall_status": "pass",
            "submit_ready": True,
            "counts": {"pass": len(actions)},
            "ready_actions": [
                {"id": action["id"], "submits_slurm": action["submits_slurm"], "heavy_gpu": action["heavy_gpu"]}
                for action in actions
            ],
        },
    )


def run_queue(paths: dict[str, Path], root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--artifact-root",
        str(root),
        "--plan-json",
        str(paths["plan"]),
        "--operator-sheet-json",
        str(paths["sheet"]),
        "--operator-execution-json",
        str(paths["execution"]),
        "--operator-followup-validation-json",
        str(paths["followup_validation"]),
        "--operator-ready-submit-preflight-json",
        str(paths["ready_preflight"]),
        "--json-out",
        str(paths["queue"]),
        "--markdown-out",
        str(paths["queue"].with_suffix(".md")),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"queue summary failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return read_json(paths["queue"])


def row_steps(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("id")): str(row.get("next_step"))
        for row in payload.get("queue", [])
        if isinstance(row, dict) and row.get("id")
    }


def run_scenario(root: Path, name: str, setup: str, expected: dict[str, str], expected_overall: str) -> dict[str, Any]:
    paths = base_reports(root)
    container_followup = root / "reports/operator_followups/01_submit_container_preflight.json"
    if setup == "no_execution":
        pass
    elif setup == "submitted_not_submitted":
        write_execution(paths, "submit_container_preflight", 0)
        write_followup(container_followup, "submit_container_preflight", "not_submitted")
    elif setup == "waiting":
        write_execution(paths, "submit_container_preflight", 0)
        write_followup(container_followup, "submit_container_preflight", "waiting")
    elif setup == "waiting_ready_preflight_failed":
        write_execution(paths, "submit_container_preflight", 0)
        write_followup(container_followup, "submit_container_preflight", "waiting")
        sheet = read_json(paths["sheet"])
        sheet["ready_actions"] = sheet.get("ready_actions", [])[:1]
        write_json(paths["sheet"], sheet)
        plan = read_json(paths["plan"])
        plan["next_actions"] = plan.get("next_actions", [])[:1]
        write_json(paths["plan"], plan)
        write_json(paths["ready_preflight"], {"overall_status": "warn", "submit_ready": False, "ready_actions": []})
    elif setup == "ready_for_followup":
        write_execution(paths, "submit_container_preflight", 0)
        write_followup(container_followup, "submit_container_preflight", "ready_for_followup")
    elif setup == "followup_passed":
        write_execution(paths, "submit_container_preflight", 0)
        write_followup(container_followup, "submit_container_preflight", "pass")
    elif setup == "execution_failed":
        write_execution(paths, "submit_container_preflight", 1)
    elif setup == "ready_preflight_failed":
        write_json(paths["ready_preflight"], {"overall_status": "fail", "submit_ready": False, "ready_actions": []})
    elif setup == "non_slurm_reference_gate":
        set_ready_actions(
            paths,
            root,
            [
                action_record(
                    root,
                    "probe_remote_hosts",
                    1,
                    submits_slurm=False,
                    stage="reference_gate",
                )
            ],
        )
    else:
        raise ValueError(f"unknown setup: {setup}")

    payload = run_queue(paths, root)
    observed = row_steps(payload)
    problems: list[str] = []
    if payload.get("overall_status") != expected_overall:
        problems.append(f"overall {payload.get('overall_status')!r} != expected {expected_overall!r}")
    for action_id, step in expected.items():
        if observed.get(action_id) != step:
            problems.append(f"{action_id} next_step {observed.get(action_id)!r} != expected {step!r}")
    return {
        "name": name,
        "status": "pass" if not problems else "fail",
        "expected_overall": expected_overall,
        "observed_overall": payload.get("overall_status"),
        "expected_steps": expected,
        "observed_steps": observed,
        "problems": problems,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Queue Transition Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | status | expected overall | observed overall | observed steps |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scenario in payload["scenarios"]:
        steps = ", ".join(f"{key}={value}" for key, value in sorted(scenario["observed_steps"].items()))
        lines.append(
            f"| {scenario['name']} | {scenario['status']} | `{scenario['expected_overall']}` | "
            f"`{scenario['observed_overall']}` | `{steps}` |"
        )
    if payload["problems"]:
        lines += ["", "## Problems", ""]
        lines.extend(f"- {problem}" for problem in payload["problems"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="eagle3_queue_transition_validation_"))
    scenario_defs = [
        (
            "no_execution",
            "no_execution",
            {"submit_container_preflight": "execute_submit", "submit_rollout_capture": "execute_submit"},
            "ready_for_operator_submit",
        ),
        (
            "submitted_not_submitted",
            "submitted_not_submitted",
            {"submit_container_preflight": "poll_slurm", "submit_rollout_capture": "execute_submit"},
            "ready_for_operator_submit",
        ),
        (
            "waiting",
            "waiting",
            {"submit_container_preflight": "keep_polling", "submit_rollout_capture": "execute_submit"},
            "ready_for_operator_submit",
        ),
        (
            "waiting_ready_preflight_failed",
            "waiting_ready_preflight_failed",
            {"submit_container_preflight": "keep_polling"},
            "waiting_for_slurm",
        ),
        (
            "ready_for_followup",
            "ready_for_followup",
            {"submit_container_preflight": "execute_followup", "submit_rollout_capture": "execute_submit"},
            "ready_for_followup",
        ),
        (
            "followup_passed",
            "followup_passed",
            {"submit_container_preflight": "refresh_state", "submit_rollout_capture": "execute_submit"},
            "ready_for_operator_submit",
        ),
        (
            "execution_failed",
            "execution_failed",
            {"submit_container_preflight": "inspect_execution_failure", "submit_rollout_capture": "execute_submit"},
            "blocked",
        ),
        (
            "ready_preflight_failed",
            "ready_preflight_failed",
            {"submit_container_preflight": "blocked", "submit_rollout_capture": "blocked"},
            "blocked",
        ),
        (
            "non_slurm_reference_gate",
            "non_slurm_reference_gate",
            {"probe_remote_hosts": "execute_submit"},
            "ready_for_operator_submit",
        ),
    ]
    scenarios: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        for name, setup, expected_steps, expected_overall in scenario_defs:
            root = temp_root / name / "qwen3_235b_eagle3"
            result = run_scenario(root, name, setup, expected_steps, expected_overall)
            scenarios.append(result)
            problems.extend(f"{name}: {problem}" for problem in result["problems"])
    except Exception as exc:
        problems.append(str(exc))
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    overall = "pass" if not problems and all(item["status"] == "pass" for item in scenarios) else "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
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
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
