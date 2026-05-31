#!/usr/bin/env python3
"""Run next-action follow-up commands only after Slurm jobs are terminal.

This is the guarded companion to run_eagle3_next_action.py. It is print-only by
default. With --execute-after, it runs an action's after_commands only when the
submitted Slurm job ids are no longer visible in squeue and sacct reports a
terminal state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}

ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RUNNING",
    "RESIZING",
    "SIGNALING",
    "SUSPENDED",
}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path(os.environ.get("NEXT_ACTION_PLAN_JSON", artifact_root / "reports/eagle3_next_actions.json")),
    )
    parser.add_argument(
        "--operator-sheet-json",
        type=Path,
        default=Path(os.environ.get("OPERATOR_SHEET_JSON", artifact_root / "reports/eagle3_operator_sheet.json")),
    )
    parser.add_argument("--action-id", help="Action id. Defaults to the first Slurm action with follow-up commands.")
    parser.add_argument("--execution-record", type=Path, help="Execution record written by run_eagle3_next_action.py.")
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=Path(os.environ.get("OPERATOR_EXECUTION_DIR", artifact_root / "reports/operator_execution")),
    )
    parser.add_argument("--job-file", type=Path, action="append", default=[], help="Job-file to inspect. Repeatable.")
    parser.add_argument("--job-id", action="append", default=[], help="Slurm job id to inspect. Repeatable.")
    parser.add_argument("--execute-after", action="store_true", help="Run after_commands after all jobs are terminal.")
    parser.add_argument(
        "--allow-missing-execution-record",
        action="store_true",
        help="Allow --execute-after without a successful run_eagle3_next_action.py execution record.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional follow-up record JSON path.")
    parser.add_argument("--markdown-out", type=Path, help="Optional follow-up record Markdown path.")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def load_json(path: Path | None) -> tuple[Any | None, str | None]:
    if path is None:
        return None, "not provided"
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid JSON: {exc}"


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in as_list(plan.get("next_actions")) if isinstance(item, dict)]


def select_action(items: list[dict[str, Any]], action_id: str | None) -> dict[str, Any]:
    if action_id:
        for item in items:
            if item.get("id") == action_id:
                return item
        raise SystemExit(f"action id not found in next-action plan: {action_id}")
    for item in items:
        if item.get("submits_slurm") and item.get("after_commands"):
            return item
    raise SystemExit("no Slurm action with after_commands is available")


def operator_action(sheet: dict[str, Any], action_id: str) -> dict[str, Any]:
    for item in as_list(sheet.get("ready_actions")):
        if isinstance(item, dict) and item.get("id") == action_id:
            return item
    return {}


def latest_execution_record(execution_dir: Path, action_id: str) -> Path | None:
    if not execution_dir.exists():
        return None
    latest: tuple[float, Path] | None = None
    for path in sorted(execution_dir.glob("*.json")):
        raw, error = load_json(path)
        if error:
            continue
        record = as_dict(raw)
        record_action = as_dict(record.get("action"))
        if record_action.get("id") != action_id:
            continue
        completed = float(record.get("completed_at_epoch") or path.stat().st_mtime)
        if latest is None or completed >= latest[0]:
            latest = (completed, path)
    return latest[1] if latest else None


def resolve_execution_record(args: argparse.Namespace, sheet_item: dict[str, Any], action_id: str) -> Path | None:
    if args.execution_record:
        return args.execution_record
    if sheet_item.get("execution_record"):
        return Path(str(sheet_item["execution_record"]))
    return latest_execution_record(args.execution_dir, action_id)


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def command_option(tokens: list[str], name: str) -> str | None:
    for idx, token in enumerate(tokens):
        if token == name and idx + 1 < len(tokens):
            return tokens[idx + 1]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def resolve_relative(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def command_env_assignments(command: str) -> dict[str, str]:
    tokens = split_command(command)
    if not tokens:
        return {}
    idx = 1 if tokens[0] == "env" else 0
    env: dict[str, str] = {}
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            idx += 1
            continue
        if "=" not in token or token.startswith("-"):
            break
        key, value = token.split("=", 1)
        if not key.replace("_", "").isalnum() or not key[:1].isalpha():
            break
        env[key] = value
        idx += 1
    return env


def infer_job_files(action_id: str, action_command: str | None, after_commands: list[str]) -> list[Path]:
    paths: list[Path] = []
    repo_roots: list[Path] = []
    artifact_roots: list[Path] = []
    for command in [action_command or "", *after_commands]:
        if not command:
            continue
        env = command_env_assignments(command)
        for key in ("SWE_REPO_ROOT", "REPO_ROOT"):
            if env.get(key):
                repo_roots.append(resolve_relative(env[key]))
        if env.get("ARTIFACT_ROOT"):
            artifact_roots.append(resolve_relative(env["ARTIFACT_ROOT"]))
        tokens = split_command(command)
        artifact_root = command_option(tokens, "--artifact-root")
        if artifact_root:
            artifact_roots.append(resolve_relative(artifact_root))
        job_file = command_option(tokens, "--job-file")
        if job_file:
            paths.append(resolve_relative(job_file))
        repo = command_option(tokens, "--repo-root")
        if repo:
            repo_roots.append(resolve_relative(repo))

    if action_id == "submit_container_preflight":
        paths.append(ROOT / "latest_eagle3_container_preflight_job.txt")
    elif action_id in {"submit_rollout_capture", "submit_rollout_fallback"}:
        for repo_root in repo_roots:
            paths.append(repo_root / "latest_235b_swe_job_id.txt")
        paths.append(ROOT / "latest_235b_swe_job_id.txt")
    elif action_id == "submit_eagle3_pilot_pipeline":
        paths.append(ROOT / "latest_eagle3_pipeline_jobs.txt")
        for artifact_root in artifact_roots:
            paths.append(artifact_root / "reports/eagle3_pipeline_jobs.env")
    elif action_id == "submit_trained_draft_spec_tokens_sweep":
        paths.append(ROOT / "latest_trained_draft_spec_tokens_sweep_jobs.txt")

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def extract_job_id(value: str) -> str | None:
    match = re.search(r"\b(\d{4,})(?:[.;]\w+)?\b", value)
    return match.group(1) if match else None


def parse_job_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "jobs": []}
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key = "line"
        value = line
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
        key_l = key.lower()
        job_id = extract_job_id(value) if key_l == "line" or "job" in key_l else None
        rows.append({"key": key, "value": value, "job_id": job_id or ""})
        if job_id and job_id not in seen:
            seen.add(job_id)
            result["jobs"].append({"key": key, "job_id": job_id})
    result["rows"] = rows
    return result


def run_command(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return {"available": False, "command": cmd[0], "output": ""}
    return {
        "available": True,
        "command": " ".join(shlex.quote(part) for part in cmd),
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def normalize_state(state: str | None) -> str:
    if not state:
        return ""
    clean = state.strip().upper().split()[0]
    if "+" in clean:
        clean = clean.split("+", 1)[0]
    return clean


def sacct_primary_state(rows: list[str], job_id: str) -> tuple[str, str | None]:
    fallback_state = ""
    fallback_exit: str | None = None
    for row in rows:
        cols = row.split("|")
        if len(cols) < 2:
            continue
        row_job = cols[0].split(".", 1)[0]
        state = cols[1]
        exit_code = cols[2] if len(cols) >= 3 else None
        if not fallback_state:
            fallback_state = state
            fallback_exit = exit_code
        if row_job == job_id:
            return state, exit_code
    return fallback_state, fallback_exit


def slurm_state(job_id: str) -> dict[str, Any]:
    squeue = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"])
    queue_states = [line.strip() for line in (squeue.get("output") or "").splitlines() if line.strip()]
    if queue_states:
        state = normalize_state(queue_states[0])
        return {
            "job_id": job_id,
            "status": "active",
            "state": state,
            "terminal": False,
            "squeue": squeue,
        }

    sacct = run_command(["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "-P", "-n"])
    rows = [line for line in (sacct.get("output") or "").splitlines() if line.strip()]
    raw_state, exit_code = sacct_primary_state(rows, job_id)
    state = normalize_state(raw_state)
    if state in TERMINAL_STATES:
        status = "terminal"
        terminal = True
    elif state in ACTIVE_STATES:
        status = "active"
        terminal = False
    elif state:
        status = "unknown_state"
        terminal = False
    else:
        status = "unknown"
        terminal = False
    return {
        "job_id": job_id,
        "status": status,
        "state": state or None,
        "exit_code": exit_code,
        "terminal": terminal,
        "squeue": squeue,
        "sacct": sacct,
        "sacct_rows": rows[:20],
    }


def collect_jobs(
    args: argparse.Namespace,
    action_id: str,
    action_command: str | None,
    after_commands: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    job_files = list(args.job_file) + infer_job_files(action_id, action_command, after_commands)
    seen_files: set[str] = set()
    file_payloads: list[dict[str, Any]] = []
    job_ids: list[str] = []
    seen_jobs: set[str] = set()

    for path in job_files:
        resolved = resolve_relative(path)
        key = str(resolved)
        if key in seen_files:
            continue
        seen_files.add(key)
        parsed = parse_job_file(resolved)
        file_payloads.append(parsed)
        for job in parsed.get("jobs") or []:
            job_id = str(job.get("job_id") or "")
            if job_id and job_id not in seen_jobs:
                seen_jobs.add(job_id)
                job_ids.append(job_id)

    for value in args.job_id:
        job_id = extract_job_id(value) or value
        if job_id and job_id not in seen_jobs:
            seen_jobs.add(job_id)
            job_ids.append(job_id)

    states = [slurm_state(job_id) for job_id in job_ids]
    return file_payloads, states


def execution_record_status(path: Path | None) -> dict[str, Any]:
    raw, error = load_json(path)
    if error:
        return {"path": str(path) if path else None, "visible": False, "status": "missing", "error": error}
    record = as_dict(raw)
    returncode = record.get("returncode")
    return {
        "path": str(path),
        "visible": True,
        "status": "pass" if record.get("mode") == "execute" and returncode == 0 else "not_successful",
        "mode": record.get("mode"),
        "returncode": returncode,
        "completed_at": record.get("completed_at"),
        "after_policy": record.get("after_policy"),
    }


def run_after_commands(commands: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, shell=True, text=True, check=False)
        rows.append({"command": command, "returncode": result.returncode})
        if result.returncode != 0:
            break
    return rows


def decide_status(
    action: dict[str, Any],
    execution: dict[str, Any],
    jobs: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    execute_after: bool,
    allow_missing_execution_record: bool,
) -> tuple[str, str]:
    if not action.get("after_commands"):
        return "no_followups", "selected action has no after_commands"
    if not jobs:
        return "not_submitted", "no concrete Slurm job id was found"
    if any(job.get("status") == "active" for job in jobs):
        active = [job.get("job_id") for job in jobs if job.get("status") == "active"]
        return "waiting", f"Slurm jobs are still active: {', '.join(active)}"
    if not all(job.get("terminal") for job in jobs):
        unknown = [job.get("job_id") for job in jobs if not job.get("terminal")]
        return "unknown", f"Slurm terminal state is not proven for: {', '.join(unknown)}"
    if execution.get("status") != "pass" and not allow_missing_execution_record:
        return "blocked", "successful run_eagle3_next_action.py execution record is required before running follow-ups"
    if not execute_after:
        return "ready_for_followup", "all visible Slurm job ids are terminal; pass --execute-after to run after_commands"
    if all(row.get("returncode") == 0 for row in after_rows) and len(after_rows) == len(action.get("after_commands") or []):
        return "pass", "all follow-up commands returned zero"
    return "fail", "one or more follow-up commands failed or did not run"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Slurm Follow-Up Guard",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Action: `{payload['action_id']}`",
        f"Detail: {payload['detail']}",
        "",
        "| job id | status | state | terminal | exit code |",
        "| --- | --- | --- | --- | --- |",
    ]
    for job in payload["jobs"]:
        lines.append(
            f"| `{job.get('job_id')}` | `{job.get('status')}` | `{job.get('state') or '-'}` | "
            f"{str(job.get('terminal')).lower()} | `{job.get('exit_code') or '-'}` |"
        )
    if not payload["jobs"]:
        lines.append("| - | `not_submitted` | - | false | - |")
    lines += ["", "## Follow-Up Commands", ""]
    for command in payload["after_commands"]:
        lines += ["```bash", command, "```", ""]
    if payload["after_returncodes"]:
        lines += ["## Follow-Up Return Codes", "", "| returncode | command |", "| ---: | --- |"]
        for row in payload["after_returncodes"]:
            command = str(row.get("command") or "").replace("|", "/")
            lines.append(f"| {row.get('returncode')} | `{command}` |")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(args: argparse.Namespace, payload: dict[str, Any], markdown: str) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")


def main() -> int:
    args = parse_args()
    plan_raw, plan_error = load_json(args.plan_json)
    if plan_error:
        raise SystemExit(f"cannot load next-action plan: {plan_error}")
    plan = as_dict(plan_raw)
    action = select_action(actions(plan), args.action_id)
    action_id = str(action.get("id") or "")
    sheet_raw, _ = load_json(args.operator_sheet_json)
    sheet_item = operator_action(as_dict(sheet_raw), action_id)
    execution_path = resolve_execution_record(args, sheet_item, action_id)
    execution = execution_record_status(execution_path)
    after_commands = [str(command) for command in as_list(action.get("after_commands"))]
    action_command = str(action.get("command") or "")
    job_files, jobs = collect_jobs(args, action_id, action_command, after_commands)

    after_rows: list[dict[str, Any]] = []
    can_run = (
        args.execute_after
        and after_commands
        and jobs
        and all(job.get("terminal") for job in jobs)
        and (execution.get("status") == "pass" or args.allow_missing_execution_record)
    )
    if can_run:
        after_rows = run_after_commands(after_commands)

    overall, detail = decide_status(
        action,
        execution,
        jobs,
        after_rows,
        args.execute_after,
        args.allow_missing_execution_record,
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": socket.gethostname(),
        "cwd": str(ROOT),
        "artifact_root": str(args.artifact_root),
        "plan_json": str(args.plan_json),
        "operator_sheet_json": str(args.operator_sheet_json),
        "action_id": action_id,
        "overall_status": overall,
        "detail": detail,
        "mode": "execute_after" if args.execute_after else "inspect_only",
        "execution_record": execution,
        "job_files": job_files,
        "jobs": jobs,
        "after_commands": after_commands,
        "after_returncodes": after_rows,
    }
    markdown = render_markdown(payload)
    write_outputs(args, payload, markdown)
    print(markdown, end="")

    if overall == "fail":
        return 1
    if args.fail_on_not_ready and overall not in {"pass", "ready_for_followup"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
