#!/usr/bin/env python3
"""Arbitrate concurrent rollout-capture jobs for the Qwen3 Eagle3 gate.

This helper is intentionally conservative. By default it never mutates Slurm
state; it writes a report that says whether duplicate pending rollout jobs can
be cancelled after another rollout has already started or after canonical
promotion has been claimed. With --execute-cancel and --allow-scancel it only
cancels jobs still in PENDING state.
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


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
CANCELABLE_STATES = {"PENDING"}
WINNER_STATES = {"RUNNING", "CONFIGURING", "COMPLETING"}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--queue-json", type=Path)
    parser.add_argument("--promotion-marker", type=Path)
    parser.add_argument("--canonical-output", type=Path)
    parser.add_argument(
        "--experimental-cancel-grace-minutes",
        type=float,
        default=float(os.environ.get("EXPERIMENTAL_CANCEL_GRACE_MINUTES", "20")),
        help=(
            "When an experimental rollout is the first RUNNING job, keep safer "
            "pending fallbacks alive until this many elapsed minutes have passed."
        ),
    )
    parser.add_argument("--execute-cancel", action="store_true")
    parser.add_argument("--allow-scancel", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    reports = args.artifact_root / "reports"
    if args.queue_json is None:
        args.queue_json = reports / "rollout_queue_wait_summary.json"
    if args.promotion_marker is None:
        args.promotion_marker = reports / "rollout_capture_canonical_promotion.json"
    if args.canonical_output is None:
        args.canonical_output = args.artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl"
    if args.json_out is None:
        args.json_out = reports / "rollout_job_arbitration.json"
    if args.markdown_out is None:
        args.markdown_out = reports / "rollout_job_arbitration.md"
    return args


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def active_jobs(queue: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for item in queue.get("jobs") or []:
        if not isinstance(item, dict):
            continue
        snapshot = item.get("current_squeue")
        if not isinstance(snapshot, dict):
            continue
        state = str(snapshot.get("state") or "").upper()
        if state in ACTIVE_STATES:
            job_id = str(item.get("job_id") or snapshot.get("job_id") or "")
            if not job_id:
                continue
            copied = dict(item)
            copied["job_id"] = job_id
            copied["current_squeue"] = dict(snapshot)
            jobs.append(copied)
    return sorted(jobs, key=job_sort_key)


def job_sort_key(job: dict[str, Any]) -> tuple[int, int, str]:
    snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
    state = str(snapshot.get("state") or "").upper()
    name = str(snapshot.get("name") or "").lower()
    nodes = parse_int(snapshot.get("nodes"))
    state_rank = {"RUNNING": 0, "CONFIGURING": 1, "COMPLETING": 2, "PENDING": 3}.get(state, 4)
    # Prefer smaller valid fallback jobs once they actually start, because the
    # purpose of this arbiter is to reduce duplicate GPU allocation.
    role_rank = 0 if is_fallback_name(name) else 1
    return (state_rank, role_rank + nodes, str(job.get("job_id") or ""))


def parse_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def is_fallback_name(name: str) -> bool:
    markers = ("balanced24n4g", "balanced_24n4g", "experimental", "20n4g", "18n4g", "16n4g", "compact")
    return any(marker in name.lower() for marker in markers)


def is_experimental_name(name: str) -> bool:
    lowered = name.lower()
    return "experimental" in lowered or "20n4g" in lowered or "18n4g" in lowered


def role_label(name: str) -> str:
    lowered = name.lower()
    if is_experimental_name(lowered):
        return "experimental fallback"
    if is_fallback_name(lowered):
        return "balanced fallback"
    return "official/generic"


def refresh_squeue(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    job_ids = [str(job.get("job_id") or "") for job in jobs if job.get("job_id")]
    if not job_ids:
        return [], None
    command = ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i|%j|%T|%M|%D|%R|%S"]
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return jobs, {"command": shell_join(command), "returncode": "command_not_found", "output": str(exc)}
    if result.returncode != 0:
        return jobs, {"command": shell_join(command), "returncode": result.returncode, "output": result.stdout.strip()}
    rows: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        job_id, name, state, elapsed, nodes, reason, start = parts
        rows[job_id] = {
            "job_id": job_id,
            "name": name,
            "state": state,
            "elapsed": elapsed,
            "nodes": nodes,
            "reason": reason,
            "start": start,
        }
    refreshed: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        snapshot = rows.get(job_id)
        if not snapshot:
            continue
        copied = dict(job)
        copied["current_squeue"] = snapshot
        refreshed.append(copied)
    return sorted(refreshed, key=job_sort_key), None


def job_summary(job: dict[str, Any]) -> dict[str, Any]:
    snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
    return {
        "job_id": str(job.get("job_id") or snapshot.get("job_id") or ""),
        "name": str(snapshot.get("name") or ""),
        "state": str(snapshot.get("state") or ""),
        "nodes": str(snapshot.get("nodes") or ""),
        "reason": str(snapshot.get("reason") or ""),
        "start": str(snapshot.get("start") or ""),
        "elapsed": str(snapshot.get("elapsed") or ""),
    }


def elapsed_minutes(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"N/A", "Unknown", "None", "-"}:
        return None
    parts = text.split("-")
    days = 0
    clock = text
    if len(parts) == 2:
        try:
            days = int(parts[0])
        except ValueError:
            return None
        clock = parts[1]
    fields = clock.split(":")
    try:
        if len(fields) == 3:
            hours, minutes, seconds = (int(item) for item in fields)
        elif len(fields) == 2:
            hours = 0
            minutes, seconds = (int(item) for item in fields)
        else:
            return None
    except ValueError:
        return None
    return days * 24 * 60 + hours * 60 + minutes + seconds / 60


def cancel_gate(
    *,
    winner: dict[str, Any] | None,
    promotion: dict[str, Any],
    canonical_output: Path,
    experimental_cancel_grace_minutes: float,
) -> dict[str, Any]:
    promoted_job = str(promotion.get("job_id") or "")
    canonical_ready = canonical_output.exists() and canonical_output.stat().st_size > 0
    if promoted_job or canonical_ready:
        return {
            "ready": True,
            "reason": "canonical promotion/output is already present",
            "promoted_job": promoted_job,
            "canonical_output_exists": canonical_ready,
        }
    if not winner:
        return {"ready": False, "reason": "no running rollout winner is visible"}
    snapshot = winner.get("current_squeue") if isinstance(winner.get("current_squeue"), dict) else {}
    name = str(snapshot.get("name") or "")
    state = str(snapshot.get("state") or "").upper()
    elapsed = elapsed_minutes(snapshot.get("elapsed"))
    if state not in WINNER_STATES:
        return {"ready": False, "reason": f"winner state {state or 'unknown'} is not cancellable evidence"}
    if is_experimental_name(name) and (elapsed is None or elapsed < experimental_cancel_grace_minutes):
        return {
            "ready": False,
            "reason": (
                "experimental rollout winner is still inside warm-up grace; "
                "keep safer pending fallbacks alive"
            ),
            "elapsed_minutes": elapsed,
            "experimental_cancel_grace_minutes": experimental_cancel_grace_minutes,
        }
    return {
        "ready": True,
        "reason": "running rollout winner has enough evidence to cancel pending duplicates",
        "elapsed_minutes": elapsed,
        "experimental_cancel_grace_minutes": experimental_cancel_grace_minutes,
    }


def choose_winner(jobs: list[dict[str, Any]], promotion: dict[str, Any], canonical_output: Path) -> tuple[dict[str, Any] | None, str]:
    promoted_job = str(promotion.get("job_id") or "")
    if promoted_job:
        for job in jobs:
            if str(job.get("job_id") or "") == promoted_job:
                return job, f"canonical promotion already claimed by job {promoted_job}"
        return None, f"canonical promotion already claimed by terminal job {promoted_job}"
    if canonical_output.exists() and canonical_output.stat().st_size > 0:
        return None, f"canonical output already exists at {canonical_output}"
    runnable = [
        job
        for job in jobs
        if str((job.get("current_squeue") or {}).get("state") or "").upper() in WINNER_STATES
    ]
    if runnable:
        return sorted(runnable, key=job_sort_key)[0], "a rollout job has started; pending duplicates can be cancelled"
    return None, "no started rollout job or canonical promotion is visible yet"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    queue = load_json(args.queue_json)
    promotion = load_json(args.promotion_marker)
    jobs, squeue_error = refresh_squeue(active_jobs(queue))
    winner, winner_reason = choose_winner(jobs, promotion, args.canonical_output)
    winner_id = str((winner or {}).get("job_id") or "")
    gate = cancel_gate(
        winner=winner,
        promotion=promotion,
        canonical_output=args.canonical_output,
        experimental_cancel_grace_minutes=args.experimental_cancel_grace_minutes,
    )
    cancel_candidates: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        state = str((job.get("current_squeue") or {}).get("state") or "").upper()
        if winner_id and job_id == winner_id:
            continue
        if state in CANCELABLE_STATES and gate["ready"]:
            cancel_candidates.append(job)

    commands = [["scancel", str(job.get("job_id"))] for job in cancel_candidates]
    executions = []
    if args.execute_cancel:
        if not args.allow_scancel:
            executions.append(
                {
                    "returncode": 1,
                    "command": "",
                    "output": "--execute-cancel requires --allow-scancel",
                }
            )
        else:
            for command in commands:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                executions.append(
                    {
                        "command": shell_join(command),
                        "returncode": result.returncode,
                        "output": result.stdout.strip(),
                    }
                )

    if cancel_candidates and args.execute_cancel and all(item.get("returncode") == 0 for item in executions):
        overall = "executed"
        recommendation = "cancelled_pending_duplicates"
    elif cancel_candidates:
        overall = "action_recommended"
        recommendation = "cancel_pending_duplicates"
    elif winner_id and not gate["ready"]:
        overall = "watching"
        recommendation = "wait_for_cancel_gate"
    elif len(jobs) > 1 and not winner_id:
        overall = "waiting"
        recommendation = "wait_for_first_started_rollout"
    elif jobs:
        overall = "watching"
        recommendation = "no_cancel_needed"
    else:
        overall = "idle"
        recommendation = "no_active_rollout_jobs"
    if squeue_error:
        overall = "warn"
        recommendation = "inspect_squeue_error"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "overall_status": overall,
        "recommendation": recommendation,
        "winner_reason": winner_reason,
        "cancel_gate": gate,
        "winner": job_summary(winner) if winner else None,
        "active_jobs": [job_summary(job) for job in jobs],
        "cancel_candidates": [job_summary(job) for job in cancel_candidates],
        "cancel_commands": [shell_join(command) for command in commands],
        "executed": bool(args.execute_cancel),
        "executions": executions,
        "squeue_error": squeue_error,
        "promotion_marker": str(args.promotion_marker),
        "canonical_output": str(args.canonical_output),
        "canonical_output_exists": args.canonical_output.exists(),
        "canonical_output_size": args.canonical_output.stat().st_size if args.canonical_output.exists() else 0,
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Rollout Job Arbitration",
        "",
        f"Overall: **{str(data['overall_status']).upper()}**",
        f"Generated: `{data['generated_at']}`",
        f"Recommendation: `{data['recommendation']}`",
        "",
        data.get("winner_reason") or "",
        "",
        "| job | state | nodes | reason | start | role |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for job in data.get("active_jobs") or []:
        role = role_label(str(job.get("name") or ""))
        lines.append(
            f"| {job.get('job_id')} | {job.get('state')} | {job.get('nodes')} | "
            f"{str(job.get('reason') or '').replace('|', '/')} | {job.get('start')} | {role} |"
        )
    if data.get("winner"):
        lines += ["", f"Winner: `{data['winner'].get('job_id')}`"]
    if data.get("cancel_gate"):
        gate = data["cancel_gate"]
        lines += [
            "",
            f"Cancel gate: `{'ready' if gate.get('ready') else 'waiting'}` - {gate.get('reason', '-')}",
        ]
    if data.get("cancel_candidates"):
        lines += ["", "Pending duplicate cancel candidates:"]
        for job in data["cancel_candidates"]:
            lines.append(f"- `{job.get('job_id')}` {job.get('name')} ({job.get('state')})")
        lines += ["", "Commands:", "", "```bash"]
        lines.extend(data.get("cancel_commands") or [])
        lines.append("```")
    if data.get("executions"):
        lines += ["", "Executions:", "", "```json", json.dumps(data["executions"], indent=2), "```"]
    if data.get("squeue_error"):
        lines += ["", "squeue error:", "", "```json", json.dumps(data["squeue_error"], indent=2), "```"]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    data = build_payload(args)
    markdown = render_markdown(data)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
