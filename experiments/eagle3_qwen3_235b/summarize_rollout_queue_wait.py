#!/usr/bin/env python3
"""Summarize rollout-capture Slurm queue wait and start-estimate drift.

This is a no-submit helper. It reads rollout watcher logs plus current `squeue`
state and writes a compact report so long queue waits do not hide whether the
rollout jobs are merely pending or have started producing artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_WATCHER_POLL_SECONDS = 120
DEFAULT_WATCHER_MAX_POLLS = 240
DEFAULT_TERMINAL_BUFFER_MINUTES = 180
ACTIVE_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+job=(?P<job_id>\d+)\s+active state=(?P<state>\S+)\s+"
    r"start=(?P<start>\S+)\s+reason=(?P<reason>.*?)(?:;|$)"
)
START_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+.*watcher start job=(?P<job_id>\d+)"
    r"(?:.*poll_seconds=(?P<poll_seconds>\d+)\s+max_polls=(?P<max_polls>\d+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--log", type=Path, action="append", help="Watcher log to parse. Defaults to rollout watcher logs under reports.")
    parser.add_argument("--watcher-poll-seconds", type=int, default=DEFAULT_WATCHER_POLL_SECONDS)
    parser.add_argument("--watcher-max-polls", type=int, default=DEFAULT_WATCHER_MAX_POLLS)
    parser.add_argument("--terminal-buffer-minutes", type=int, default=DEFAULT_TERMINAL_BUFFER_MINUTES)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def default_logs(artifact_root: Path) -> list[Path]:
    reports = artifact_root / "reports"
    patterns = [
        "rollout_capture*_watch.log",
        "rollout_capture*_pending_state_watch.log",
        "watch_rollout_capture*.log",
        "*rollout_capture*watch*.log",
    ]
    logs: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(reports.glob(pattern)):
            if path.exists() and path not in seen:
                logs.append(path)
                seen.add(path)
    return logs


def pid_file_status(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists(), "pid": None, "alive": False}
    if not path.exists():
        return item
    try:
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (OSError, ValueError):
        item["error"] = "invalid_pid_file"
        return item
    item["pid"] = pid
    if pid <= 0:
        item["error"] = "non_positive_pid"
        return item
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        item["error"] = "not_running"
    except PermissionError:
        item["alive"] = True
        item["permission_limited"] = True
    else:
        item["alive"] = True
    return item


def extension_coverage(artifact_root: Path, job_id: str) -> dict[str, Any]:
    reports = artifact_root / "reports"
    pid_files = sorted(reports.glob(f"*{job_id}*watch_extension.pid"))
    statuses = [pid_file_status(path) for path in pid_files]
    alive = [item for item in statuses if item.get("alive")]
    return {
        "covered": bool(alive),
        "alive": bool(alive),
        "alive_count": len(alive),
        "pid_file_count": len(statuses),
        "pid_files": statuses,
    }


def parse_logs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.replace("\x00", "").strip()
            start_match = START_RE.search(stripped)
            if start_match:
                job_id = start_match.group("job_id")
                job = jobs.setdefault(
                    job_id,
                    {
                        "job_id": job_id,
                        "samples": [],
                        "watcher_starts": [],
                        "log_paths": set(),
                    },
                )
                job["watcher_starts"].append(
                    {
                        "timestamp": start_match.group("timestamp"),
                        "poll_seconds": int(start_match.group("poll_seconds") or 0) or None,
                        "max_polls": int(start_match.group("max_polls") or 0) or None,
                        "log_path": str(path),
                    }
                )
                job["log_paths"].add(str(path))
                continue
            match = ACTIVE_RE.search(stripped)
            if not match:
                continue
            job_id = match.group("job_id")
            job = jobs.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "samples": [],
                    "watcher_starts": [],
                    "log_paths": set(),
                },
            )
            sample = {
                "timestamp": match.group("timestamp"),
                "state": match.group("state"),
                "start": match.group("start"),
                "reason": match.group("reason"),
                "log_path": str(path),
            }
            job["samples"].append(sample)
            job["log_paths"].add(str(path))
    for job in jobs.values():
        samples = job["samples"]
        starts = [sample["start"] for sample in samples]
        distinct_starts = []
        for value in starts:
            if value not in distinct_starts:
                distinct_starts.append(value)
        job["sample_count"] = len(samples)
        job["first_seen"] = samples[0]["timestamp"] if samples else None
        job["last_seen"] = samples[-1]["timestamp"] if samples else None
        job["latest_log_state"] = samples[-1]["state"] if samples else None
        job["latest_log_start"] = samples[-1]["start"] if samples else None
        job["latest_log_reason"] = samples[-1]["reason"] if samples else None
        job["distinct_start_estimates"] = distinct_starts
        job["start_estimate_changes"] = max(0, len(distinct_starts) - 1)
        job["recent_start_estimates"] = starts[-8:]
        job["log_paths"] = sorted(job["log_paths"])
    return jobs


def parse_local_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    parts = value.split()
    if len(parts) >= 6:
        # Drop the timezone token and interpret the timestamp in the local
        # timezone of the cluster process that produced the log.
        normalized = " ".join([parts[0], parts[1], parts[2], parts[3], parts[-1]])
        try:
            return time.mktime(time.strptime(normalized, "%a %b %d %H:%M:%S %Y"))
        except ValueError:
            return None
    return None


def parse_slurm_start(value: str | None) -> float | None:
    if not value or value in {"N/A", "Unknown", "None", "-"}:
        return None
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def format_local_time(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(epoch))


def enrich_timeout_risk(job: dict[str, Any], snapshot: dict[str, str] | None, args: argparse.Namespace) -> dict[str, Any]:
    starts = job.get("watcher_starts") or []
    first_start = starts[0] if starts else None
    first_seen_epoch = parse_local_timestamp((first_start or {}).get("timestamp")) or parse_local_timestamp(job.get("first_seen"))
    poll_seconds = (first_start or {}).get("poll_seconds") or args.watcher_poll_seconds
    max_polls = (first_start or {}).get("max_polls") or args.watcher_max_polls
    deadline_epoch = first_seen_epoch + poll_seconds * max_polls if first_seen_epoch is not None else None
    snapshot_start = (snapshot or {}).get("start")
    start_source = "squeue"
    start_epoch = parse_slurm_start(snapshot_start)
    if start_epoch is None and job.get("latest_log_start"):
        start_source = "watcher_log"
        start_epoch = parse_slurm_start(job.get("latest_log_start"))
    now = time.time()
    terminal_buffer_seconds = args.terminal_buffer_minutes * 60
    risk = "unknown"
    if deadline_epoch is not None:
        if start_epoch is not None:
            risk = "risk" if start_epoch + terminal_buffer_seconds > deadline_epoch else "ok"
        elif deadline_epoch <= now + terminal_buffer_seconds:
            risk = "risk"
        else:
            risk = "unknown_start"
    return {
        "risk": risk,
        "poll_seconds": poll_seconds,
        "max_polls": max_polls,
        "first_seen_epoch": first_seen_epoch,
        "watcher_deadline": format_local_time(deadline_epoch),
        "minutes_until_watcher_deadline": None if deadline_epoch is None else round((deadline_epoch - now) / 60, 1),
        "minutes_until_start_estimate": None if start_epoch is None else round((start_epoch - now) / 60, 1),
        "start_estimate_source": start_source if start_epoch is not None else None,
        "terminal_buffer_minutes": args.terminal_buffer_minutes,
    }


def squeue_snapshot(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    command = ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i|%j|%T|%M|%D|%R|%S"]
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return {"_error": {"output": str(exc), "returncode": "command_not_found"}}
    rows: dict[str, dict[str, str]] = {}
    if result.returncode != 0:
        return {"_error": {"output": result.stdout.strip(), "returncode": str(result.returncode)}}
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
    return rows


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    logs = args.log or default_logs(args.artifact_root)
    jobs = parse_logs(logs)
    current = squeue_snapshot(sorted(jobs))
    current_error = current.pop("_error", None)

    status_counts: dict[str, int] = defaultdict(int)
    timeout_risks: list[str] = []
    timeout_covered: list[str] = []
    for job_id, job in jobs.items():
        snapshot = current.get(job_id)
        if snapshot:
            job["current_squeue"] = snapshot
            status_counts[snapshot["state"].lower()] += 1
        else:
            job["current_squeue"] = None
            status_counts["not_in_squeue"] += 1
        job["watcher_timeout"] = enrich_timeout_risk(job, snapshot, args)
        job["watcher_extension_coverage"] = extension_coverage(args.artifact_root, job_id)
        if snapshot and job["watcher_timeout"].get("risk") == "risk" and job["watcher_extension_coverage"].get("covered"):
            job["watcher_timeout"]["raw_risk"] = "risk"
            job["watcher_timeout"]["risk"] = "covered_by_extension"
            job["watcher_timeout"]["covered_by_extension"] = True
            timeout_covered.append(job_id)
        elif snapshot and job["watcher_timeout"].get("risk") == "risk":
            timeout_risks.append(job_id)

    if current_error:
        overall = "warn"
    elif timeout_risks:
        overall = "warn"
    elif any((job.get("current_squeue") or {}).get("state") in {"PENDING", "RUNNING", "CONFIGURING"} for job in jobs.values()):
        overall = "waiting"
    elif jobs:
        overall = "terminal_or_unknown"
    else:
        overall = "idle"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "overall_status": overall,
        "logs": [str(path) for path in logs],
        "counts": dict(status_counts),
        "squeue_error": current_error,
        "watcher_timeout_risk_jobs": timeout_risks,
        "watcher_timeout_covered_jobs": timeout_covered,
        "watcher_timeout_defaults": {
            "poll_seconds": args.watcher_poll_seconds,
            "max_polls": args.watcher_max_polls,
            "terminal_buffer_minutes": args.terminal_buffer_minutes,
        },
        "jobs": [jobs[job_id] for job_id in sorted(jobs)],
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Rollout Queue Wait Summary",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Generated: `{data['generated_at']}`",
        f"Artifact root: `{data['artifact_root']}`",
        "",
        "| job | current | nodes | reason | current start | watcher deadline | timeout risk | extension | log samples | estimate changes | latest log start |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for job in data["jobs"]:
        current = job.get("current_squeue") or {}
        timeout = job.get("watcher_timeout") or {}
        coverage = job.get("watcher_extension_coverage") or {}
        alive_pids = [
            str(item.get("pid"))
            for item in coverage.get("pid_files") or []
            if item.get("alive") and item.get("pid") is not None
        ]
        extension = "alive:" + ",".join(alive_pids) if alive_pids else ("none" if not coverage.get("pid_file_count") else "not_alive")
        lines.append(
            "| {job_id} | {state} | {nodes} | {reason} | {start} | {deadline} | {risk} | {extension} | {samples} | {changes} | {latest} |".format(
                job_id=job["job_id"],
                state=current.get("state", "not_in_squeue"),
                nodes=current.get("nodes", "-"),
                reason=str(current.get("reason", "-")).replace("|", "/"),
                start=current.get("start", "-"),
                deadline=timeout.get("watcher_deadline") or "-",
                risk=timeout.get("risk") or "-",
                extension=extension,
                samples=job.get("sample_count", 0),
                changes=job.get("start_estimate_changes", 0),
                latest=job.get("latest_log_start") or "-",
            )
        )
    if not data["jobs"]:
        lines.append("| - | idle | - | no rollout watcher logs/jobs observed | - | - | - | - | 0 | 0 | - |")
    if data.get("watcher_timeout_risk_jobs"):
        lines += ["", "Watcher timeout risks:"]
        for job_id in data["watcher_timeout_risk_jobs"]:
            lines.append(f"- `{job_id}` may need a restarted watcher before terminal rollout handling.")
    if data.get("watcher_timeout_covered_jobs"):
        lines += ["", "Watcher timeout risks covered by extension watchers:"]
        for job_id in data["watcher_timeout_covered_jobs"]:
            lines.append(f"- `{job_id}` has an alive extension watcher for terminal rollout handling.")
    lines += ["", "## Recent Start Estimates", ""]
    for job in data["jobs"]:
        recent = ", ".join(job.get("recent_start_estimates") or []) or "-"
        lines.append(f"- `{job['job_id']}`: {recent}")
    if data.get("squeue_error"):
        lines += ["", "## squeue Error", "", "```text", json.dumps(data["squeue_error"], indent=2), "```"]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
