#!/usr/bin/env python3
"""Decide whether to keep waiting or use a smaller rollout-capture fallback.

This is a no-submit guard. It inspects the current Slurm job and the
prevalidated rollout resource profiles, then emits the exact fallback command
only when the official job is still active and its estimated start is too far
out.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--job-id", default=os.environ.get("ROLLOUT_JOB_ID"))
    parser.add_argument(
        "--queue-json",
        type=Path,
        default=artifact_root / "reports/rollout_queue_wait_summary.json",
        help="Used to infer the current active rollout job when --job-id is omitted.",
    )
    parser.add_argument(
        "--resource-profiles-json",
        type=Path,
        default=artifact_root / "reports/rollout_resource_profiles_preflight.json",
    )
    parser.add_argument(
        "--max-start-delay-minutes",
        type=int,
        default=int(os.environ.get("ROLLOUT_FALLBACK_MAX_START_DELAY_MINUTES", "120")),
    )
    parser.add_argument(
        "--compact-output",
        type=Path,
        default=artifact_root / "data/qwen3_235b_swe_rollout_conversations_compact16n4g.jsonl",
    )
    parser.add_argument(
        "--fallback-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the selected fallback profile.",
    )
    parser.add_argument(
        "--official-output",
        type=Path,
        default=artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def run(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": cmd, "returncode": result.returncode, "output": result.stdout.strip()}


def parse_start_time(value: str) -> datetime | None:
    if not value or value in {"N/A", "Unknown", "None"}:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return {}, f"invalid JSON: {exc}"


def fallback_profiles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for profile in payload.get("profiles", []):
        if not isinstance(profile, dict):
            continue
        profile_id = str(profile.get("id") or "")
        if profile_id == "official_32n4g_async":
            continue
        if profile.get("experimental") is True:
            continue
        if profile.get("status") == "pass" and profile.get("submit_command"):
            profiles.append(profile)
    profiles.sort(key=profile_sort_key)
    return profiles


def profile_sort_key(profile: dict[str, Any]) -> tuple[int, int, str]:
    env = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    try:
        nodes = int(env.get("NUM_NODES") or 10**9)
    except (TypeError, ValueError):
        nodes = 10**9
    try:
        gen_nodes = int(env.get("NUM_GEN_NODES") or 10**9)
    except (TypeError, ValueError):
        gen_nodes = 10**9
    return nodes, gen_nodes, str(profile.get("id") or "")


def profile_output_path(profile: dict[str, Any], artifact_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    submit_env = profile.get("submit_env") if isinstance(profile.get("submit_env"), dict) else {}
    if submit_env.get("OUTPUT_CONVERSATIONS"):
        return Path(str(submit_env["OUTPUT_CONVERSATIONS"]))
    output_name = profile.get("output_name")
    if output_name:
        return artifact_root / "data" / str(output_name)
    return artifact_root / "data" / f"qwen3_235b_swe_rollout_conversations_{profile.get('id', 'fallback')}.jsonl"


def active_fallback_jobs(active_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers = ("compact", "balanced", "experimental", "24n4g", "20n4g", "18n4g", "16n4g")
    jobs: list[dict[str, Any]] = []
    for job in active_jobs:
        name = str(((job.get("current_squeue") or {}).get("name") or "")).lower()
        if any(marker in name for marker in markers):
            jobs.append(job)
    return jobs


def compact_profile(payload: dict[str, Any]) -> dict[str, Any] | None:
    for profile in payload.get("profiles", []):
        if isinstance(profile, dict) and profile.get("id") == "compact_16n4g_smoke":
            return profile
    return None


def infer_active_job_id(queue_json: Path) -> str | None:
    payload, error = load_json(queue_json)
    if error:
        return None
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
    active: list[dict[str, Any]] = []
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
        state = str(snapshot.get("state") or "").upper()
        if state in active_states:
            active.append(job)
    if not active:
        return None
    active.sort(
        key=lambda job: (
            0
            if any(
                marker in str((job.get("current_squeue") or {}).get("name") or "").lower()
                for marker in (
                    "fixedcontainer",
                    "systemvenv",
                    "systemvllm",
                    "sharedvllm",
                    "aarchvllm",
                    "vllm0112",
                    "vllm0102",
                )
            )
            else 1,
            str(job.get("job_id") or (job.get("current_squeue") or {}).get("job_id") or ""),
        )
    )
    snapshot = active[0].get("current_squeue") if isinstance(active[0].get("current_squeue"), dict) else {}
    return str(active[0].get("job_id") or snapshot.get("job_id") or "") or None


def active_queue_jobs(queue_json: Path) -> list[dict[str, Any]]:
    payload, error = load_json(queue_json)
    if error:
        return []
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
    active: list[dict[str, Any]] = []
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
        state = str(snapshot.get("state") or "").upper()
        if state in active_states:
            active.append(job)
    return active


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    active_jobs = active_queue_jobs(args.queue_json)
    active_smaller_jobs = active_fallback_jobs(active_jobs)
    job_id = args.job_id or infer_active_job_id(args.queue_json)
    squeue = run(["squeue", "-j", str(job_id or ""), "-h", "-o", "%T|%S|%R"]) if job_id else {
        "command": [],
        "returncode": 1,
        "output": "",
    }
    active = squeue["returncode"] == 0 and bool(squeue["output"])
    state = reason = start_raw = None
    estimated_start_delay_minutes: float | None = None
    if active:
        parts = str(squeue["output"]).split("|", 2)
        state = parts[0] if len(parts) > 0 else None
        start_raw = parts[1] if len(parts) > 1 else None
        reason = parts[2] if len(parts) > 2 else None
        start = parse_start_time(start_raw or "")
        if start:
            estimated_start_delay_minutes = round((start - datetime.now()).total_seconds() / 60.0, 1)
            if estimated_start_delay_minutes < 0:
                estimated_start_delay_minutes = 0.0

    profiles_payload, profiles_error = load_json(args.resource_profiles_json)
    compact = compact_profile(profiles_payload)
    compact_ready = bool(compact and compact.get("status") == "pass" and compact.get("submit_command"))
    fallback_candidates = fallback_profiles(profiles_payload)
    selected_fallback = fallback_candidates[0] if fallback_candidates else None
    selected_fallback_output = profile_output_path(selected_fallback, args.artifact_root, args.fallback_output) if selected_fallback else None
    official_output_exists = args.official_output.exists()
    compact_output_exists = args.compact_output.exists()
    fallback_output_exists = selected_fallback_output.exists() if selected_fallback_output else False
    smaller_already_active = bool(active_smaller_jobs)

    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "official rollout output",
            "status": "pass" if official_output_exists else "missing",
            "detail": str(args.official_output),
        }
    )
    checks.append(
        {
            "name": "fallback profile",
            "status": "pass" if selected_fallback else "fail",
            "detail": (
                f"{selected_fallback.get('id')} is selected"
                if selected_fallback
                else "no non-official fallback profile is prevalidated"
            ),
        }
    )
    checks.append(
        {
            "name": "compact fallback profile",
            "status": "pass" if compact_ready else "fail",
            "detail": "compact_16n4g_smoke is prevalidated" if compact_ready else "compact_16n4g_smoke is not prevalidated",
        }
    )
    checks.append(
        {
            "name": "compact output collision",
            "status": "warn" if compact_output_exists else "pass",
            "detail": str(args.compact_output),
        }
    )
    if selected_fallback_output:
        checks.append(
            {
                "name": "selected fallback output collision",
                "status": "warn" if fallback_output_exists else "pass",
                "detail": str(selected_fallback_output),
            }
        )

    if official_output_exists:
        overall = "official_output_ready"
        recommendation = "use_official_output"
        next_command = None
        detail = "official rollout conversations already exist"
    elif not active:
        overall = "official_job_terminal_or_missing"
        recommendation = "run_guarded_followup"
        next_command = None
        detail = "official job is not active in squeue; run guarded follow-up before deciding fallback"
    elif smaller_already_active:
        overall = "keep_waiting"
        recommendation = "fallback_already_active"
        next_command = None
        detail = (
            "a smaller fallback rollout is already active; do not submit another fallback "
            f"({', '.join(str(job.get('job_id') or (job.get('current_squeue') or {}).get('job_id')) for job in active_smaller_jobs)})"
        )
    elif state in {"RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}:
        overall = "keep_waiting"
        recommendation = "keep_waiting"
        next_command = None
        detail = "official job is already running; do not submit fallback"
    elif not selected_fallback:
        overall = "keep_waiting"
        recommendation = "keep_waiting"
        next_command = None
        detail = "no smaller fallback profile is prevalidated"
    elif fallback_output_exists:
        overall = "fallback_output_ready"
        recommendation = f"inspect_{selected_fallback.get('id')}_output"
        next_command = None
        detail = f"{selected_fallback.get('id')} output already exists"
    elif estimated_start_delay_minutes is not None and estimated_start_delay_minutes > args.max_start_delay_minutes:
        overall = "fallback_ready"
        recommendation = f"submit_{selected_fallback.get('id')}"
        next_command = str(selected_fallback.get("submit_command"))
        detail = (
            f"official job estimated start delay {estimated_start_delay_minutes} min exceeds "
            f"{args.max_start_delay_minutes} min; selected fallback {selected_fallback.get('id')}"
        )
    else:
        overall = "keep_waiting"
        recommendation = "keep_waiting"
        next_command = None
        detail = "official job is active and fallback threshold is not met"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "recommendation": recommendation,
        "detail": detail,
        "artifact_root": str(args.artifact_root),
        "job_id": str(job_id or ""),
        "job": {
            "active": active,
            "state": state,
            "reason": reason,
            "start_time": start_raw,
            "estimated_start_delay_minutes": estimated_start_delay_minutes,
            "squeue": squeue,
        },
        "active_compact_jobs": [
            {
                "job_id": str(job.get("job_id") or (job.get("current_squeue") or {}).get("job_id") or ""),
                "name": str((job.get("current_squeue") or {}).get("name") or ""),
                "state": str((job.get("current_squeue") or {}).get("state") or ""),
                "start": str((job.get("current_squeue") or {}).get("start") or ""),
            }
            for job in active_smaller_jobs
        ],
        "fallback_candidates": [
            {
                "id": str(profile.get("id") or ""),
                "status": str(profile.get("status") or ""),
                "env": profile.get("env") if isinstance(profile.get("env"), dict) else {},
                "output": str(profile_output_path(profile, args.artifact_root)),
            }
            for profile in fallback_candidates
        ],
        "selected_fallback": {
            "id": str(selected_fallback.get("id") or ""),
            "output": str(selected_fallback_output or ""),
        }
        if selected_fallback
        else None,
        "queue_json": str(args.queue_json),
        "thresholds": {"max_start_delay_minutes": args.max_start_delay_minutes},
        "resource_profiles_json": str(args.resource_profiles_json),
        "resource_profiles_error": profiles_error,
        "official_output": str(args.official_output),
        "compact_output": str(args.compact_output),
        "fallback_output": str(selected_fallback_output or ""),
        "checks": checks,
        "next_command": next_command,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Fallback Decision",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Recommendation: `{payload['recommendation']}`",
        f"Detail: {payload['detail']}",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| job id | `{payload['job_id']}` |",
        f"| job state | `{payload['job'].get('state') or '-'}` |",
        f"| start time | `{payload['job'].get('start_time') or '-'}` |",
        f"| estimated delay minutes | `{payload['job'].get('estimated_start_delay_minutes')}` |",
        f"| max delay minutes | `{payload['thresholds']['max_start_delay_minutes']}` |",
        f"| selected fallback | `{(payload.get('selected_fallback') or {}).get('id') or '-'}` |",
        "",
        "## Next Command",
        "",
    ]
    if payload.get("next_command"):
        lines += ["```bash", str(payload["next_command"]), "```"]
    else:
        lines.append("No fallback submit command is recommended right now.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
