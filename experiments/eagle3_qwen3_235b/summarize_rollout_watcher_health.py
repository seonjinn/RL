#!/usr/bin/env python3
"""Summarize rollout watcher process health and report freshness.

This helper is no-submit. It checks the watcher PID files used by the rollout
capture path and records whether the expected current-run monitors are alive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
ACTIVE_SLURM_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid_file(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "pid": None,
        "alive": False,
        "status": "missing",
    }
    if not path.exists():
        return item
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    item["raw"] = raw
    try:
        pid = int(raw)
    except ValueError:
        item["status"] = "invalid"
        return item
    item["pid"] = pid
    item["alive"] = pid_alive(pid)
    item["status"] = "alive" if item["alive"] else "dead"
    return item


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "status": "missing"}
    stat = path.stat()
    age = max(0.0, time.time() - stat.st_mtime)
    return {
        "path": str(path),
        "exists": True,
        "status": "present",
        "size_bytes": stat.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(stat.st_mtime)),
        "age_seconds": round(age, 1),
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def squeue_snapshot(job_ids: list[str]) -> dict[str, dict[str, str]]:
    ids = sorted({job_id for job_id in job_ids if job_id})
    if not ids:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-j", ",".join(ids), "-h", "-o", "%i|%j|%T"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return {}
    snapshots: dict[str, dict[str, str]] = {}
    if result.returncode != 0:
        return snapshots
    for line in result.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id, name, state = (part.strip() for part in parts)
        if job_id:
            snapshots[job_id] = {"job_id": job_id, "name": name, "state": state.upper()}
    return snapshots


def state_report_jobs(reports: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in sorted(reports.glob("rollout_capture_*_state_advance.json")):
        state = load_json(path)
        if not state:
            continue
        decision = state.get("decision") if isinstance(state.get("decision"), dict) else {}
        job = state.get("job") if isinstance(state.get("job"), dict) else {}
        slurm = job.get("slurm") if isinstance(job.get("slurm"), dict) else {}
        job_id = str(job.get("job_id") or slurm.get("job_id") or "").strip()
        slurm_state = str(slurm.get("state") or "").upper()
        if decision.get("overall_status") != "running" and slurm_state not in ACTIVE_SLURM_STATES:
            continue
        if not job_id:
            continue
        jobs.append(
            {
                "job_id": job_id,
                "name": "",
                "state": slurm_state or "UNKNOWN",
                "state_report": str(path),
                "rollout_log_dir": str(state.get("rollout_log_dir") or ""),
                "output_data": str(state.get("output_data") or ""),
                "dynamic_state_report": True,
            }
        )
    snapshots = squeue_snapshot([job["job_id"] for job in jobs])
    active_jobs: list[dict[str, Any]] = []
    for job in jobs:
        snapshot = snapshots.get(job["job_id"]) or {}
        if not snapshot:
            continue
        job["name"] = snapshot.get("name") or job["name"]
        job["state"] = snapshot.get("state") or job["state"]
        if str(job.get("state") or "").upper() in ACTIVE_SLURM_STATES:
            active_jobs.append(job)
    return active_jobs


def dynamic_state_file(reports: Path, job_id: str) -> Path | None:
    candidates = sorted(reports.glob(f"rollout_capture_*{job_id}*_state_advance.json"))
    return candidates[0] if candidates else None


def queue_context(root: Path) -> dict[str, Any]:
    reports = root / "reports"
    queue = load_json(reports / "rollout_queue_wait_summary.json") or {}
    gated = load_json(reports / "eagle3_pipeline_gated_submit.json") or {}
    active_jobs_by_id: dict[str, dict[str, Any]] = {}
    for job in queue.get("jobs") or []:
        snapshot = job.get("current_squeue") or {}
        state = str(snapshot.get("state") or "").upper()
        if state in ACTIVE_SLURM_STATES:
            job_id = str(job.get("job_id") or snapshot.get("job_id") or "")
            if job_id:
                active_job: dict[str, Any] = {
                    "job_id": job_id,
                    "name": str(snapshot.get("name") or ""),
                    "state": state,
                    "dynamic_state_report": False,
                }
                state_report = dynamic_state_file(reports, job_id)
                if state_report:
                    state_payload = load_json(state_report) or {}
                    active_job.update(
                        {
                            "dynamic_state_report": True,
                            "state_report": str(state_report),
                            "rollout_log_dir": str(state_payload.get("rollout_log_dir") or ""),
                            "output_data": str(state_payload.get("output_data") or ""),
                        }
                    )
                active_jobs_by_id[job_id] = active_job

    for job in state_report_jobs(reports):
        active_jobs_by_id[job["job_id"]] = job

    active_jobs = list(active_jobs_by_id.values())

    active_official = False
    active_compact = False
    active_generic_jobs = []
    for job in active_jobs:
        if job.get("dynamic_state_report"):
            active_generic_jobs.append(job)
            continue
        name = job["name"].lower()
        if "compact" in name:
            active_compact = True
        elif any(
            marker in name
            for marker in ("fixedcontainer", "systemvenv", "systemvllm", "sharedvllm", "aarchvllm", "vllm0130", "vllm0112", "vllm0102")
        ):
            active_generic_jobs.append(job)
        else:
            active_official = True

    official_state = load_json(reports / "rollout_capture_state_advance.json") or {}
    compact_state = load_json(reports / "rollout_capture_compact16n4g_state_advance.json") or {}
    state_reports = [official_state, compact_state]
    rollout_ready = False
    for state in state_reports:
        decision = state.get("decision") if isinstance(state.get("decision"), dict) else {}
        output = Path(str(state.get("output_data") or ""))
        if decision.get("overall_status") == "pass" and decision.get("next_step") == "pipeline_dry_run" and output.exists():
            rollout_ready = True

    gated_jobs = gated.get("jobs") if isinstance(gated.get("jobs"), dict) else {}
    pipeline_submitted = (
        gated.get("overall_status") == "pass"
        and gated.get("executed") is True
        and {"dump_job", "train_job", "export_job"}.issubset(gated_jobs)
    )

    return {
        "queue_status": queue.get("overall_status") or "missing",
        "active_jobs": active_jobs,
        "active_official": active_official,
        "active_compact": active_compact,
        "active_generic_jobs": active_generic_jobs,
        "rollout_ready": rollout_ready,
        "pipeline_submitted": pipeline_submitted,
    }


def dynamic_pid_file(reports: Path, job_id: str) -> Path:
    candidates = sorted(reports.glob(f"*{job_id}*watch.pid"))
    return candidates[0] if candidates else reports / f"rollout_capture_{job_id}_watch.pid"


def optional_autosubmit_pid_file(reports: Path, job_id: str) -> Path:
    candidates = sorted(reports.glob(f"*{job_id}*autosubmit.pid"))
    return candidates[0] if candidates else reports / f"rollout_capture_{job_id}_watch_autosubmit.pid"


def optional_extension_launcher_pid_file(reports: Path, job_id: str) -> Path:
    candidates = sorted(reports.glob(f"*{job_id}*watch_extension_launcher.pid"))
    return candidates[0] if candidates else reports / f"rollout_capture_{job_id}_watch_extension_launcher.pid"


def expected_pid_files(root: Path, context: dict[str, Any]) -> list[tuple[str, Path, bool, str]]:
    reports = root / "reports"
    active_jobs = bool(context["active_jobs"])
    active_job_count = len(context.get("active_jobs") or [])
    active_official = bool(context["active_official"])
    active_compact = bool(context["active_compact"])
    rollout_ready = bool(context["rollout_ready"])
    pipeline_submitted = bool(context.get("pipeline_submitted"))
    needs_rollout_monitoring = (active_official or active_compact) and not rollout_ready
    expected = [
        (
            "official materialize watcher",
            reports / "rollout_capture_official32n4g_watch.pid",
            active_official and not rollout_ready,
            "official rollout job is still active",
        ),
        (
            "compact materialize watcher",
            reports / "rollout_capture_compact16n4g_watch.pid",
            active_compact and not rollout_ready,
            "compact rollout job is still active",
        ),
        (
            "official pending-state watcher",
            reports / "rollout_capture_official32n4g_pending_state_watch.pid",
            active_official and not rollout_ready,
            "official rollout job is still active",
        ),
        (
            "compact pending-state watcher",
            reports / "rollout_capture_compact16n4g_pending_state_watch.pid",
            active_compact and not rollout_ready,
            "compact rollout job is still active",
        ),
        (
            "operator rollout follow-up watcher",
            reports / "operator_followups/01_submit_rollout_capture_watch.pid",
            needs_rollout_monitoring,
            "at least one rollout job is still active",
        ),
        (
            "rollout job arbitration watcher",
            reports / "rollout_job_arbitration_watch.pid",
            active_job_count > 1 and not rollout_ready,
            "multiple rollout jobs are active and pending duplicate cleanup must be monitored",
        ),
        (
            "pipeline ready-submit watcher",
            reports / "eagle3_pipeline_ready_submit_watch.pid",
            active_jobs and not pipeline_submitted,
            "rollout is active or pending and Eagle3 pipeline should submit once gated preflight is ready",
        ),
        ("pipeline watcher", reports / "eagle3_pipeline_watch.pid", False, "pipeline has not been submitted yet"),
    ]
    for job in context.get("active_generic_jobs") or []:
        job_id = str(job.get("job_id") or "")
        expected.append(
            (
                f"generic materialize watcher {job_id}",
                dynamic_pid_file(reports, job_id),
                bool(job_id) and not rollout_ready,
                "generic rollout materialize watcher also refreshes pending state",
            )
        )
        autosubmit_pid = optional_autosubmit_pid_file(reports, job_id)
        expected.append(
            (
                f"generic gated auto-submit watcher {job_id}",
                autosubmit_pid,
                bool(job_id) and autosubmit_pid.exists(),
                "optional gated Eagle3 pilot submit watcher was started for this rollout",
            )
        )
        extension_launcher_pid = optional_extension_launcher_pid_file(reports, job_id)
        expected.append(
            (
                f"generic current-code extension launcher {job_id}",
                extension_launcher_pid,
                bool(job_id) and extension_launcher_pid.exists() and not rollout_ready,
                "optional lock-waiting current-code materialize watcher was started for this rollout",
            )
        )
    return expected


def report_files(root: Path) -> list[tuple[str, Path]]:
    reports = root / "reports"
    items = [
        ("official rollout state", reports / "rollout_capture_state_advance.json"),
        ("compact rollout state", reports / "rollout_capture_compact16n4g_state_advance.json"),
        ("queue wait summary", reports / "rollout_queue_wait_summary.json"),
        ("pipeline submit preflight", reports / "eagle3_pipeline_submit_preflight.json"),
        ("gated pipeline submit", reports / "eagle3_pipeline_gated_submit.json"),
        ("operator state refresh", reports / "eagle3_operator_state_refresh.json"),
    ]
    for path in sorted(reports.glob("rollout_capture_*_state_advance.json")):
        if path.name in {"rollout_capture_state_advance.json", "rollout_capture_compact16n4g_state_advance.json"}:
            continue
        items.append((path.stem.replace("_state_advance", " state"), path))
    return items


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    context = queue_context(args.artifact_root)
    watchers = []
    for label, path, required_now, required_reason in expected_pid_files(args.artifact_root, context):
        item = read_pid_file(path)
        item["label"] = label
        item["required_now"] = required_now
        item["required_reason"] = required_reason if required_now else "not required in current queue state"
        watchers.append(item)

    reports = []
    for label, path in report_files(args.artifact_root):
        item = file_info(path)
        item["label"] = label
        reports.append(item)

    required = [item for item in watchers if item["required_now"]]
    dead_or_missing = [item for item in required if item["status"] != "alive"]
    freshness_required = {"queue wait summary"}
    if context.get("active_official"):
        freshness_required.add("official rollout state")
    if context.get("active_compact"):
        freshness_required.add("compact rollout state")
    for job in context.get("active_generic_jobs") or []:
        state_report = job.get("state_report")
        if state_report:
            freshness_required.add(Path(str(state_report)).stem.replace("_state_advance", " state"))
    stale_reports = [
        item
        for item in reports
        if item["label"] in freshness_required
        and (context["active_jobs"] or not context["rollout_ready"])
        and (not item.get("exists") or float(item.get("age_seconds", 10**9)) > 900)
    ]
    overall = "pass" if not dead_or_missing and not stale_reports else "warn"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "overall_status": overall,
        "queue_context": context,
        "watchers": watchers,
        "reports": reports,
        "dead_or_missing_required_watchers": [item["label"] for item in dead_or_missing],
        "stale_reports": [item["label"] for item in stale_reports],
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Rollout Watcher Health",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Generated: `{data['generated_at']}`",
        "",
        "Queue context:",
        "",
        "```json",
        json.dumps(data.get("queue_context") or {}, indent=2, sort_keys=True),
        "```",
        "",
        "| watcher | required now | status | pid |",
        "| --- | --- | --- | ---: |",
    ]
    for item in data["watchers"]:
        lines.append(
            f"| {item['label']} | {str(item['required_now']).lower()} | {item['status']} | {item.get('pid') or '-'} |"
        )
    lines += ["", "| report | status | age seconds | mtime |", "| --- | --- | ---: | --- |"]
    for item in data["reports"]:
        lines.append(
            f"| {item['label']} | {item['status']} | {item.get('age_seconds', '-')} | {item.get('mtime', '-')} |"
        )
    if data["dead_or_missing_required_watchers"]:
        lines += ["", "Required watcher issues:"]
        for label in data["dead_or_missing_required_watchers"]:
            lines.append(f"- {label}")
    if data["stale_reports"]:
        lines += ["", "Stale report issues:"]
        for label in data["stale_reports"]:
            lines.append(f"- {label}")
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
