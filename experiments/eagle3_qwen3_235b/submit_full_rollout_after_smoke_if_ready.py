#!/usr/bin/env python3
"""Gate full SWE-Gym rollout submission on a passing smoke rollout.

This script is intentionally no-submit by default. It reads the smoke rollout
state and the full-rollout submit preflight report, then emits the exact next
action. With ``--execute --allow-heavy-gpu`` it can submit the full rollout only
when all gates are satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_REPO_ROOT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("SWE_REPO_ROOT") or os.environ.get("REPO_ROOT") or DEFAULT_REPO_ROOT),
    )
    parser.add_argument("--smoke-job-id", default=os.environ.get("SMOKE_JOB_ID"))
    parser.add_argument(
        "--smoke-report-prefix",
        default=os.environ.get(
            "SMOKE_REPORT_PREFIX",
            "rollout_capture_vllm0102src_swegym_fixed_instancedict_2861605_swegym",
        ),
    )
    parser.add_argument(
        "--smoke-state-json",
        type=Path,
        default=None,
        help="Existing advance_rollout_capture_state JSON. Defaults from --smoke-report-prefix.",
    )
    parser.add_argument(
        "--full-preflight-json",
        type=Path,
        default=None,
        help="Full rollout submit preflight JSON. Defaults to reports/rollout_capture_swegym_full_submit_preflight.json.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-heavy-gpu", action="store_true")
    parser.add_argument(
        "--start-watcher",
        action="store_true",
        help="After a successful full rollout submit, start the rollout materialization watcher.",
    )
    parser.add_argument(
        "--allow-background",
        action="store_true",
        help="Required with --start-watcher because it starts a background process.",
    )
    parser.add_argument(
        "--full-report-prefix-template",
        default=os.environ.get("FULL_REPORT_PREFIX_TEMPLATE", "rollout_capture_vllm0102src_swegym_full_{job_id}_swegym"),
        help="Python format string used for the full rollout watcher report prefix.",
    )
    parser.add_argument("--watcher-poll-seconds", default=os.environ.get("FULL_ROLLOUT_WATCHER_POLL_SECONDS", "120"))
    parser.add_argument("--watcher-max-polls", default=os.environ.get("FULL_ROLLOUT_WATCHER_MAX_POLLS", "720"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def json_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return str(payload.get("overall_status") or payload.get("status") or decision.get("overall_status") or "unknown")


def active_rollout_job_ids(root: Path) -> set[str]:
    payload = try_load_json(root / "reports" / "rollout_queue_wait_summary.json")
    if not payload:
        return set()
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
    ids: set[str] = set()
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
        state = str(snapshot.get("state") or "").upper()
        if state in active_states:
            job_id = str(job.get("job_id") or snapshot.get("job_id") or "")
            if job_id:
                ids.add(job_id)
    return ids


def rollout_state_job_id(payload: dict[str, Any]) -> str:
    for key in ("job_id", "rollout_job_id"):
        value = payload.get(key)
        if value:
            return str(value)
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    if job.get("job_id"):
        return str(job["job_id"])
    return ""


def select_smoke_state_report(root: Path, fallback: Path) -> Path:
    reports = root / "reports"
    active_ids = active_rollout_job_ids(root)
    candidates: list[tuple[int, float, Path]] = []
    for path in reports.glob("rollout_capture*_state_advance.json"):
        if path.name == "rollout_capture_compact16n4g_state_advance.json":
            continue
        payload = try_load_json(path)
        if not payload:
            continue
        job_id = rollout_state_job_id(payload)
        status = json_status(payload).lower()
        priority = 0
        if active_ids and (job_id in active_ids or any(active_id in path.name for active_id in active_ids)):
            priority = 3
        elif status in {"running", "pass"}:
            priority = 2
        elif path == fallback:
            priority = 1
        candidates.append((priority, path.stat().st_mtime, path))
    if not candidates:
        return fallback
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def run_command(command: str) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "output_tail": result.stdout[-6000:],
    }


def start_background(env: dict[str, str], command: list[str], log_path: Path, pid_path: Path) -> dict[str, Any]:
    merged = os.environ.copy()
    merged.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("ab")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=merged,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    return {"pid": proc.pid, "pid_path": str(pid_path), "log_path": str(log_path)}


def shell(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False, "command": command[0], "output": ""}
    return {
        "available": True,
        "command": " ".join(shlex.quote(part) for part in command),
        "returncode": result.returncode,
        "output": result.stdout,
    }


def default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    reports = args.artifact_root / "reports"
    fallback_smoke = reports / f"{args.smoke_report_prefix}_state_advance.json"
    smoke = args.smoke_state_json or select_smoke_state_report(args.artifact_root, fallback_smoke)
    full = args.full_preflight_json or reports / "rollout_capture_swegym_full_submit_preflight.json"
    return smoke, full


def parse_active_job_rows(output: str, name: str) -> list[dict[str, str]]:
    rows = []
    for line in output.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[2] == name:
            rows.append({"job_id": parts[0], "state": parts[1], "name": parts[2]})
    return rows


def active_job_with_name(name: str) -> dict[str, Any]:
    if not name:
        return {"checked": False, "active": False, "reason": "missing name"}
    result = shell(["squeue", "-h", "-o", "%i|%T|%.240j"])
    if not result.get("available") or result.get("returncode") != 0:
        return {"checked": False, "active": False, "reason": "squeue unavailable", "result": result}
    rows = parse_active_job_rows(result.get("output", ""), name)
    return {"checked": True, "active": bool(rows), "rows": rows}


def extract_job_id(output: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    if match:
        return match.group(1)
    match = re.search(r"\bjob(?:id)?[=:\s]+(\d{5,})\b", output, re.IGNORECASE)
    return match.group(1) if match else None


def decide(
    smoke_state: dict[str, Any] | None,
    full_preflight: dict[str, Any] | None,
    active_full: dict[str, Any],
    execute: bool,
    allow_heavy_gpu: bool,
    start_watcher: bool,
    allow_background: bool,
) -> dict[str, Any]:
    if smoke_state is None:
        return {"overall_status": "waiting", "next_step": "refresh_smoke_state", "detail": "smoke state report is missing"}
    smoke_decision = smoke_state.get("decision") or {}
    smoke_status = smoke_decision.get("overall_status")
    if smoke_status in {"running", "not_submitted", "missing_capture"}:
        return {"overall_status": "waiting", "next_step": "poll_smoke", "detail": f"smoke rollout is not proven yet: {smoke_status}"}
    if smoke_status == "needs_materialize":
        return {"overall_status": "waiting", "next_step": "materialize_smoke", "detail": "smoke train_data exists but conversation corpus is not materialized"}
    if smoke_status != "pass":
        return {"overall_status": "fail", "next_step": "inspect_smoke", "detail": f"smoke rollout did not pass: {smoke_status}"}

    if full_preflight is None:
        return {"overall_status": "waiting", "next_step": "run_full_preflight", "detail": "full rollout preflight report is missing"}
    if not full_preflight.get("submit_ready"):
        return {"overall_status": "fail", "next_step": "inspect_full_preflight", "detail": "full rollout submit preflight is not ready"}
    wandb_name = str(full_preflight.get("wandb_name") or "")
    if "dryrun" in wandb_name.lower().replace("_", "-"):
        return {
            "overall_status": "fail",
            "next_step": "inspect_full_preflight",
            "detail": f"full rollout preflight uses a dryrun experiment name: {wandb_name}",
        }

    output_conversations = Path(str(full_preflight.get("output_conversations") or ""))
    if output_conversations.exists():
        return {"overall_status": "pass", "next_step": "full_already_materialized", "detail": "full rollout conversations already exist"}

    rollout_log_dir = Path(str(full_preflight.get("rollout_log_dir") or ""))
    train_data = sorted(rollout_log_dir.glob("train_data_step*.jsonl")) if rollout_log_dir.exists() else []
    if train_data:
        return {"overall_status": "needs_materialize", "next_step": "materialize_full", "detail": "full rollout train_data exists but conversations are not materialized"}

    if active_full.get("active"):
        return {"overall_status": "running", "next_step": "poll_full", "detail": "full rollout job is already active"}

    if execute and not allow_heavy_gpu:
        return {"overall_status": "fail", "next_step": "rerun_with_allow_heavy_gpu", "detail": "--allow-heavy-gpu is required with --execute"}
    if execute and start_watcher and not allow_background:
        return {"overall_status": "fail", "next_step": "rerun_with_allow_background", "detail": "--allow-background is required with --start-watcher"}
    return {"overall_status": "ready", "next_step": "submit_full_rollout", "detail": "smoke passed and full rollout preflight is ready"}


def full_report_prefix(template: str, job_id: str) -> str:
    try:
        return template.format(job_id=job_id)
    except Exception:
        return f"rollout_capture_swegym_full_{job_id}"


def watcher_command(
    args: argparse.Namespace,
    full_preflight: dict[str, Any],
    job_id: str,
) -> tuple[dict[str, str], list[str], Path, Path, str]:
    reports = args.artifact_root / "reports"
    prefix = full_report_prefix(args.full_report_prefix_template, job_id)
    log_path = reports / f"watch_full_swegym_rollout_{job_id}.log"
    pid_path = reports / f"{prefix}_watch.pid"
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "SWE_REPO_ROOT": str(args.repo_root),
        "JOB_ID": job_id,
        "ROLLOUT_LOG_DIR": str(full_preflight.get("rollout_log_dir") or ""),
        "OUTPUT_CONVERSATIONS": str(full_preflight.get("output_conversations") or ""),
        "REPORT_PREFIX": prefix,
        "POLL_SECONDS": str(args.watcher_poll_seconds),
        "MAX_POLLS": str(args.watcher_max_polls),
        "PROMOTE_TO_CANONICAL": "true",
        "RUN_PIPELINE_PREFLIGHT": "true",
        "AUTO_SUBMIT_PIPELINE": "false",
        "RUN_FULL_ROLLOUT_GATE": "false",
        "WATCH_PID_FILE": str(pid_path),
    }
    command = ["bash", "experiments/eagle3_qwen3_235b/watch_rollout_capture_materialize.sh"]
    return env, command, log_path, pid_path, prefix


def shell_join(env: dict[str, str], command: list[str]) -> str:
    return " ".join([*(f"{key}={shlex.quote(value)}" for key, value in env.items()), *(shlex.quote(part) for part in command)])


def render_markdown(data: dict[str, Any]) -> str:
    decision = data["decision"]
    command = data.get("submit_command") or "# no submit command available"
    lines = [
        "# Full SWE-Gym Rollout Gate",
        "",
        f"Overall: **{decision['overall_status'].upper()}**",
        f"Next step: `{decision['next_step']}`",
        "",
        decision["detail"],
        "",
        "| gate | status | detail |",
        "| --- | --- | --- |",
        f"| smoke | {(data.get('smoke_decision') or {}).get('overall_status', 'missing')} | {data.get('smoke_detail', '')} |",
        f"| full preflight | {data.get('full_preflight_status', 'missing')} | submit_ready={data.get('full_submit_ready')} |",
        f"| full active job | {str((data.get('active_full') or {}).get('active')).lower()} | {(data.get('active_full') or {}).get('rows', [])} |",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    if data.get("execute_result") is not None:
        lines.extend(
            [
                "## Execute Result",
                "",
                f"Return code: `{data['execute_result'].get('returncode')}`",
                f"Submitted job id: `{data.get('submitted_job_id') or ''}`",
                "",
                "```text",
                data["execute_result"].get("output_tail", ""),
                "```",
            ]
        )
    if data.get("watcher_command"):
        lines.extend(["", "## Full Rollout Watcher", "", "```bash", data["watcher_command"], "```"])
        watcher_result = data.get("watcher_result")
        if watcher_result:
            lines.extend(
                [
                    "",
                    f"Watcher started: **{str(bool(watcher_result.get('pid'))).lower()}**",
                    f"PID: `{watcher_result.get('pid')}`",
                    f"Log: `{watcher_result.get('log_path')}`",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    smoke_path, full_path = default_paths(args)
    smoke_state = load_json(smoke_path) if smoke_path.exists() else None
    full_preflight = load_json(full_path) if full_path.exists() else None
    full_name = str((full_preflight or {}).get("wandb_name") or "")
    active_full = active_job_with_name(full_name)
    decision = decide(
        smoke_state,
        full_preflight,
        active_full,
        args.execute,
        args.allow_heavy_gpu,
        args.start_watcher,
        args.allow_background,
    )
    submit_command = ((full_preflight or {}).get("commands") or {}).get("submit")

    execute_result = None
    submitted_job_id = None
    watcher_cmd_text = None
    watcher_result = None
    watcher_report_prefix = None
    if args.execute and decision["overall_status"] == "ready" and submit_command:
        execute_result = run_command(submit_command)
        submitted_job_id = extract_job_id(execute_result.get("output_tail", ""))
        if execute_result["returncode"] == 0 and submitted_job_id:
            if args.start_watcher and full_preflight:
                env, command, log_path, pid_path, watcher_report_prefix = watcher_command(args, full_preflight, submitted_job_id)
                watcher_cmd_text = shell_join(env, command)
                watcher_result = start_background(env, command, log_path, pid_path)
            decision = {
                "overall_status": "submitted",
                "next_step": "watch_full_rollout",
                "detail": f"submitted full rollout job {submitted_job_id}"
                + (" and started materialization watcher" if watcher_result else ""),
            }
        else:
            decision = {
                "overall_status": "fail",
                "next_step": "inspect_submit_output",
                "detail": "full rollout submit command failed or no job id was detected",
            }

    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "repo_root": str(args.repo_root),
        "smoke_state_json": str(smoke_path),
        "full_preflight_json": str(full_path),
        "smoke_decision": (smoke_state or {}).get("decision"),
        "smoke_detail": ((smoke_state or {}).get("decision") or {}).get("detail"),
        "full_preflight_status": (full_preflight or {}).get("overall_status"),
        "full_submit_ready": (full_preflight or {}).get("submit_ready"),
        "active_full": active_full,
        "decision": decision,
        "submit_command": submit_command,
        "execute_result": execute_result,
        "submitted_job_id": submitted_job_id,
        "watcher_command": watcher_cmd_text,
        "watcher_result": watcher_result,
        "watcher_report_prefix": watcher_report_prefix,
    }

    text = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text + "\n")
    print(text)
    return 1 if decision["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
