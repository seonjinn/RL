#!/usr/bin/env python3
"""Plan or start rollout watcher restarts for active Qwen3 rollout jobs.

This helper does not submit Slurm work. By default it only writes a report with
the exact commands needed to restart rollout materialization and pending-state
watchers if they are missing, dead, or near timeout. With --execute and
--allow-background it starts only the watchers that are currently required and
not alive.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--queue-json", type=Path)
    parser.add_argument("--health-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-background", action="store_true", help="Required with --execute because this starts background watcher processes.")
    parser.add_argument("--poll-seconds", default=os.environ.get("ROLLOUT_WATCHER_POLL_SECONDS", "120"))
    parser.add_argument("--max-polls", default=os.environ.get("ROLLOUT_WATCHER_MAX_POLLS", "720"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    reports = args.artifact_root / "reports"
    if args.queue_json is None:
        args.queue_json = reports / "rollout_queue_wait_summary.json"
    if args.health_json is None:
        args.health_json = reports / "rollout_watcher_health.json"
    return args


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "top-level JSON is not an object"
    return payload, None


def shell_join(env: dict[str, str], argv: list[str | Path]) -> str:
    return " ".join([*(f"{key}={shlex.quote(value)}" for key, value in env.items()), *(shlex.quote(str(part)) for part in argv)])


def watcher_alive(health: dict[str, Any], label: str) -> bool:
    for item in health.get("watchers") or []:
        if isinstance(item, dict) and item.get("label") == label:
            return item.get("status") == "alive"
    return False


def pid_file_alive(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        raw_pid = path.read_text(encoding="utf-8").strip()
        pid = int(raw_pid)
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_jobs(queue: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for item in queue.get("jobs") or []:
        if not isinstance(item, dict):
            continue
        snapshot = item.get("current_squeue") or {}
        state = str(snapshot.get("state") or "").upper()
        if state in ACTIVE_STATES:
            jobs.append(item)
    return jobs


def job_kind(job: dict[str, Any]) -> str:
    snapshot = job.get("current_squeue") or {}
    name = str(snapshot.get("name") or "").lower()
    if "compact" in name:
        return "compact"
    if any(
        marker in name
            for marker in ("fixedcontainer", "systemvenv", "systemvllm", "sharedvllm", "aarchvllm", "vllm0130", "vllm0112", "vllm0102")
    ):
        return "generic"
    return "official"


def dynamic_prefix(job: dict[str, Any], job_id: str) -> str:
    snapshot = job.get("current_squeue") or {}
    name = str(snapshot.get("name") or "").lower()
    if "systemvenv" in name:
        return f"rollout_capture_systemvenv_{job_id}"
    if "systemvllm" in name:
        return f"rollout_capture_systemvllm_{job_id}"
    if "sharedvllm" in name:
        return f"rollout_capture_sharedvllm_{job_id}"
    if "aarchvllm" in name:
        return f"rollout_capture_aarchvllm_{job_id}"
    if "vllm0112" in name:
        if "swegym" in name:
            return f"rollout_capture_vllm0112_{job_id}_swegym"
        return f"rollout_capture_vllm0112_{job_id}"
    if "vllm0130" in name:
        if "swegym" in name:
            return f"rollout_capture_vllm0130src_{job_id}_swegym"
        return f"rollout_capture_vllm0130src_{job_id}"
    if "vllm0102" in name:
        if "swegym" in name:
            return f"rollout_capture_vllm0102_{job_id}_swegym"
        return f"rollout_capture_vllm0102_{job_id}"
    if "fixedcontainer" in name:
        return f"rollout_capture_fixedcontainer_{job_id}"
    return f"rollout_capture_{job_id}"


def dynamic_state_file(reports: Path, job_id: str) -> Path | None:
    candidates = sorted(reports.glob(f"rollout_capture_*{job_id}*_state_advance.json"))
    return candidates[0] if candidates else None


def dynamic_pid_file(reports: Path, job_id: str) -> Path:
    candidates = sorted(reports.glob(f"*{job_id}*watch.pid"))
    return candidates[0] if candidates else reports / f"rollout_capture_{job_id}_watch.pid"


def dynamic_log_file(reports: Path, job_id: str) -> Path:
    candidates = sorted(reports.glob(f"*{job_id}*.log"))
    return candidates[0] if candidates else reports / f"rollout_capture_{job_id}_watch.log"


def kind_config(root: Path, kind: str, job: dict[str, Any] | None = None) -> dict[str, Any]:
    reports = root / "reports"
    snapshot = (job or {}).get("current_squeue") or {}
    job_id = str((job or {}).get("job_id") or snapshot.get("job_id") or "")
    if kind == "compact":
        return {
            "kind": "compact",
            "prefix": "rollout_capture_compact16n4g",
            "state_json": reports / "rollout_capture_compact16n4g_state_advance.json",
            "state_md": reports / "rollout_capture_compact16n4g_state_advance.md",
            "materialize_label": "compact materialize watcher",
            "pending_label": "compact pending-state watcher",
            "materialize_pid": reports / "rollout_capture_compact16n4g_watch.pid",
            "materialize_log": reports / "rollout_capture_compact16n4g_watch.log",
            "materialize_extension_pid": reports / "rollout_capture_compact16n4g_watch_extension.pid",
            "materialize_extension_log": reports / "rollout_capture_compact16n4g_watch_extension.log",
            "pending_pid": reports / "rollout_capture_compact16n4g_pending_state_watch.pid",
            "pending_log": reports / "rollout_capture_compact16n4g_pending_state_watch.log",
            "pending_extension_pid": reports / "rollout_capture_compact16n4g_pending_state_watch_extension.pid",
            "pending_extension_log": reports / "rollout_capture_compact16n4g_pending_state_watch_extension.log",
        }
    if kind == "generic":
        state_json = dynamic_state_file(reports, job_id)
        if state_json:
            prefix = state_json.name.removesuffix("_state_advance.json")
            state_md = state_json.with_suffix(".md")
        else:
            prefix = dynamic_prefix(job or {}, job_id)
            state_json = reports / f"{prefix}_state_advance.json"
            state_md = reports / f"{prefix}_state_advance.md"
        if not state_json.exists():
            state_json = reports / "rollout_capture_state_advance.json"
            state_md = reports / "rollout_capture_state_advance.md"
        return {
            "kind": "generic",
            "prefix": prefix,
            "state_json": state_json,
            "state_md": state_md,
            "materialize_label": f"generic materialize watcher {job_id}",
            "pending_label": None,
            "materialize_pid": dynamic_pid_file(reports, job_id),
            "materialize_log": dynamic_log_file(reports, job_id),
            "materialize_extension_pid": reports / f"{prefix}_watch_extension.pid",
            "materialize_extension_log": reports / f"{prefix}_watch_extension.log",
        }
    return {
        "kind": "official",
        "prefix": "rollout_capture_official32n4g",
        "state_json": reports / "rollout_capture_state_advance.json",
        "state_md": reports / "rollout_capture_state_advance.md",
        "materialize_label": "official materialize watcher",
        "pending_label": "official pending-state watcher",
        "materialize_pid": reports / "rollout_capture_official32n4g_watch.pid",
        "materialize_log": reports / "rollout_capture_official32n4g_watch.log",
        "materialize_extension_pid": reports / "rollout_capture_official32n4g_watch_extension.pid",
        "materialize_extension_log": reports / "rollout_capture_official32n4g_watch_extension.log",
        "pending_pid": reports / "rollout_capture_official32n4g_pending_state_watch.pid",
        "pending_log": reports / "rollout_capture_official32n4g_pending_state_watch.log",
        "pending_extension_pid": reports / "rollout_capture_official32n4g_pending_state_watch_extension.pid",
        "pending_extension_log": reports / "rollout_capture_official32n4g_pending_state_watch_extension.log",
    }


def state_inputs(config: dict[str, Any], fallback_root: Path) -> tuple[dict[str, Any], str | None]:
    state, error = load_json(config["state_json"])
    if error:
        return {}, error
    assert state is not None
    required = ["repo_root", "rollout_log_dir", "output_data"]
    missing = [key for key in required if not state.get(key)]
    if missing:
        return state, f"state report missing fields: {', '.join(missing)}"
    if str(state.get("artifact_root") or fallback_root) != str(fallback_root):
        state["artifact_root"] = str(fallback_root)
    return state, None


def watcher_command(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    state: dict[str, Any],
    job_id: str,
    watcher_type: str,
    wait_for_lock: bool = False,
) -> tuple[dict[str, str], list[str | Path], Path, Path]:
    is_pending = watcher_type == "pending"
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "SWE_REPO_ROOT": str(state["repo_root"]),
        "JOB_ID": job_id,
        "ROLLOUT_LOG_DIR": str(state["rollout_log_dir"]),
        "OUTPUT_CONVERSATIONS": str(state["output_data"]),
        "REPORT_PREFIX": str(config["prefix"]),
        "POLL_SECONDS": str(args.poll_seconds),
        "MAX_POLLS": str(args.max_polls),
    }
    if wait_for_lock:
        env["WAIT_FOR_LOCK"] = "true"
    if is_pending:
        env["STATE_JSON"] = str(config["state_json"])
        env["STATE_MD"] = str(config["state_md"])
        argv: list[str | Path] = ["bash", "experiments/eagle3_qwen3_235b/watch_rollout_pending_state_refresh.sh"]
        log_key = "pending_extension_log" if wait_for_lock else "pending_log"
        pid_key = "pending_extension_pid" if wait_for_lock else "pending_pid"
        env["WATCH_PID_FILE"] = str(config[pid_key])
        return env, argv, config[log_key], config[pid_key]
    argv = ["bash", "experiments/eagle3_qwen3_235b/watch_rollout_capture_materialize.sh"]
    log_key = "materialize_extension_log" if wait_for_lock else "materialize_log"
    pid_key = "materialize_extension_pid" if wait_for_lock else "materialize_pid"
    env["WATCH_PID_FILE"] = str(config[pid_key])
    return env, argv, config[log_key], config[pid_key]


def arbitration_command(args: argparse.Namespace) -> tuple[dict[str, str], list[str | Path], Path, Path]:
    reports = args.artifact_root / "reports"
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "POLL_SECONDS": str(args.poll_seconds),
        "MAX_POLLS": str(args.max_polls),
        "AUTO_CANCEL_PENDING_DUPLICATES": "true",
    }
    argv: list[str | Path] = ["bash", "experiments/eagle3_qwen3_235b/watch_rollout_job_arbitration.sh"]
    return env, argv, reports / "rollout_job_arbitration_watch.log", reports / "rollout_job_arbitration_watch.pid"


def pipeline_ready_submit_command(args: argparse.Namespace) -> tuple[dict[str, str], list[str | Path], Path, Path]:
    reports = args.artifact_root / "reports"
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "POLL_SECONDS": str(args.poll_seconds),
        "MAX_POLLS": str(args.max_polls),
        "WATCH_PID_FILE": str(reports / "eagle3_pipeline_ready_submit_watch.pid"),
    }
    argv: list[str | Path] = ["bash", "experiments/eagle3_qwen3_235b/watch_eagle3_pipeline_ready_submit.sh"]
    return env, argv, reports / "eagle3_pipeline_ready_submit_watch.log", reports / "eagle3_pipeline_ready_submit_watch.pid"


def pipeline_already_submitted(root: Path) -> bool:
    payload, _ = load_json(root / "reports/eagle3_pipeline_gated_submit.json")
    if not payload:
        return False
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), dict) else {}
    return (
        payload.get("overall_status") == "pass"
        and payload.get("executed") is True
        and {"dump_job", "train_job", "export_job"}.issubset(jobs)
    )


def start_background(env: dict[str, str], argv: list[str | Path], log_path: Path, pid_path: Path) -> dict[str, Any]:
    merged = os.environ.copy()
    merged.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("ab")
    proc = subprocess.Popen(
        [str(part) for part in argv],
        cwd=ROOT,
        env=merged,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    return {"pid": proc.pid, "pid_path": str(pid_path), "log_path": str(log_path)}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    queue, queue_error = load_json(args.queue_json)
    health, health_error = load_json(args.health_json)
    queue = queue or {}
    health = health or {}
    rows: list[dict[str, Any]] = []
    errors = {"queue": queue_error, "health": health_error}

    if not queue_error:
        active = active_jobs(queue)
        if active and not pipeline_already_submitted(args.artifact_root):
            pipeline_ready_alive = watcher_alive(health, "pipeline ready-submit watcher")
            pipeline_ready_restart_needed = not pipeline_ready_alive
            env, argv, log_path, pid_path = pipeline_ready_submit_command(args)
            command = shell_join(env, argv)
            run = None
            if args.execute and args.allow_background and pipeline_ready_restart_needed:
                run = start_background(env, argv, log_path, pid_path)
            rows.append(
                {
                    "job_id": ",".join(
                        str((job.get("current_squeue") or {}).get("job_id") or job.get("job_id") or "")
                        for job in active
                    ),
                    "kind": "pipeline_ready_submit",
                    "watcher_type": "gated_pipeline_submit",
                    "state": "waiting_for_submit_ready",
                    "start": "",
                    "alive": pipeline_ready_alive,
                    "extension_alive": False,
                    "restart_needed": pipeline_ready_restart_needed,
                    "extension_needed": False,
                    "action_needed": pipeline_ready_restart_needed,
                    "timeout_risk": None,
                    "state_error": None,
                    "command": command,
                    "run": run,
                }
            )

        if len(active) > 1:
            arbitration_alive = watcher_alive(health, "rollout job arbitration watcher")
            arbitration_restart_needed = not arbitration_alive
            env, argv, log_path, pid_path = arbitration_command(args)
            command = shell_join(env, argv)
            run = None
            if args.execute and args.allow_background and arbitration_restart_needed:
                run = start_background(env, argv, log_path, pid_path)
            rows.append(
                {
                    "job_id": ",".join(
                        str((job.get("current_squeue") or {}).get("job_id") or job.get("job_id") or "")
                        for job in active
                    ),
                    "kind": "arbitration",
                    "watcher_type": "pending_duplicate_cleanup",
                    "state": "multiple_active",
                    "start": "",
                    "alive": arbitration_alive,
                    "extension_alive": False,
                    "restart_needed": arbitration_restart_needed,
                    "extension_needed": False,
                    "action_needed": arbitration_restart_needed,
                    "timeout_risk": None,
                    "state_error": None,
                    "command": command,
                    "run": run,
                }
            )

        for job in active:
            snapshot = job.get("current_squeue") or {}
            job_id = str(job.get("job_id") or snapshot.get("job_id") or "")
            if dynamic_state_file(args.artifact_root / "reports", job_id):
                kind = "generic"
            else:
                kind = job_kind(job)
            config = kind_config(args.artifact_root, kind, job)
            state, state_error = state_inputs(config, args.artifact_root)
            timeout_risk = (job.get("watcher_timeout") or {}).get("risk")
            materialize_alive = watcher_alive(health, config["materialize_label"])
            pending_alive = watcher_alive(health, config["pending_label"]) if config.get("pending_label") else True
            needs_materialize = not materialize_alive
            needs_pending = bool(config.get("pending_label")) and not pending_alive
            needs_extension = timeout_risk == "risk"

            watcher_rows = [("materialize", materialize_alive, needs_materialize)]
            if config.get("pending_label"):
                watcher_rows.append(("pending", pending_alive, needs_pending))

            for watcher_type, alive, restart_needed in watcher_rows:
                extension_pid_key = f"{watcher_type}_extension_pid"
                extension_alive = pid_file_alive(config[extension_pid_key]) if config.get(extension_pid_key) else False
                extension_needed = bool(needs_extension and alive and not extension_alive)
                action_needed = bool(restart_needed or extension_needed)
                if state_error:
                    command = None
                    run = None
                else:
                    env, argv, log_path, pid_path = watcher_command(
                        args=args,
                        config=config,
                        state=state,
                        job_id=job_id,
                        watcher_type=watcher_type,
                        wait_for_lock=extension_needed,
                    )
                    command = shell_join(env, argv)
                    run = None
                    if args.execute and args.allow_background and action_needed:
                        run = start_background(env, argv, log_path, pid_path)
                rows.append(
                    {
                        "job_id": job_id,
                        "kind": kind,
                        "watcher_type": watcher_type,
                        "state": snapshot.get("state"),
                        "start": snapshot.get("start"),
                        "alive": alive,
                        "extension_alive": extension_alive,
                        "restart_needed": restart_needed,
                        "extension_needed": extension_needed,
                        "action_needed": action_needed,
                        "timeout_risk": timeout_risk,
                        "state_error": state_error,
                        "command": command,
                        "run": run,
                    }
                )

    if args.execute and not args.allow_background:
        errors["execute"] = "--allow-background is required with --execute"

    restart_needed = [row for row in rows if row["restart_needed"]]
    extension_needed = [row for row in rows if row.get("extension_needed")]
    action_needed = [row for row in rows if row.get("action_needed")]
    started = [row for row in rows if row.get("run")]
    if any(errors.values()):
        overall = "warn"
    elif action_needed and not args.execute:
        overall = "restart_recommended" if restart_needed else "watch_deadline_risk"
    elif action_needed and len(started) != len(action_needed):
        overall = "restart_incomplete"
    elif extension_needed:
        overall = "watch_deadline_risk"
    else:
        overall = "pass"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "overall_status": overall,
        "executed": bool(args.execute and args.allow_background),
        "queue_json": str(args.queue_json),
        "health_json": str(args.health_json),
        "errors": {key: value for key, value in errors.items() if value},
        "rows": rows,
        "restart_needed_count": len(restart_needed),
        "extension_needed_count": len(extension_needed),
        "action_needed_count": len(action_needed),
        "started_count": len(started),
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Rollout Watcher Ensure Report",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Generated: `{data['generated_at']}`",
        f"Executed: **{str(data['executed']).lower()}**",
        "",
        "| job | kind | watcher | state | alive | extension alive | restart needed | extension needed | timeout risk |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["rows"]:
        lines.append(
            f"| {row['job_id']} | {row['kind']} | {row['watcher_type']} | {row.get('state') or '-'} | "
            f"{str(row['alive']).lower()} | {str(row.get('extension_alive')).lower()} | {str(row['restart_needed']).lower()} | "
            f"{str(row.get('extension_needed')).lower()} | {row.get('timeout_risk') or '-'} |"
        )
    commands = [row for row in data["rows"] if row.get("action_needed") and row.get("command")]
    if commands:
        lines += ["", "## Restart Or Extension Commands", ""]
        for row in commands:
            lines += [f"### {row['job_id']} {row['watcher_type']}", "", "```bash", str(row["command"]), "```", ""]
    if data["errors"]:
        lines += ["", "## Errors", "", "```json", json.dumps(data["errors"], indent=2, sort_keys=True), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    data = build_payload(args)
    markdown = render_markdown(data)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if data["overall_status"] in {"pass", "restart_recommended", "watch_deadline_risk"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
