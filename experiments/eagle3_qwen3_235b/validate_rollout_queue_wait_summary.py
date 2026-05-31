#!/usr/bin/env python3
"""Validate rollout queue-wait timeout coverage handling.

This is a no-submit synthetic test for summarize_rollout_queue_wait.py. It
checks that a pending rollout whose original watcher deadline is too early is a
WARN only until a live extension watcher pid file is present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/summarize_rollout_queue_wait.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("queue_wait_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_log(root: Path, *, job_id: str, base_epoch: float) -> str:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%a %b %d %H:%M:%S PDT %Y", time.localtime(base_epoch))
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(base_epoch + 5 * 3600))
    log = reports / f"rollout_capture_{job_id}_watch.log"
    log.write_text(
        "\n".join(
            [
                f"[{timestamp}] watcher start job={job_id} poll_seconds=120 max_polls=1",
                f"[{timestamp}] job={job_id} active state=PENDING start={start} reason=Priority;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return start


def build_payload(module: ModuleType, root: Path, *, job_id: str, base_epoch: float, slurm_start: str) -> dict[str, Any]:
    args = SimpleNamespace(
        artifact_root=root,
        log=None,
        watcher_poll_seconds=120,
        watcher_max_polls=1,
        terminal_buffer_minutes=180,
        json_out=None,
        markdown_out=None,
    )
    original_squeue = module.squeue_snapshot
    original_time = module.time.time
    module.squeue_snapshot = lambda job_ids: {
        job_id: {
            "job_id": job_id,
            "name": "qwen3-235b-swe-rollout-vllm0102src-swegym-fixed-instancedict-smoke1step",
            "state": "PENDING",
            "elapsed": "0:00",
            "nodes": "16",
            "reason": "Priority",
            "start": slurm_start,
        }
    }
    module.time.time = lambda: base_epoch + 10 * 60
    try:
        return module.build_payload(args)
    finally:
        module.squeue_snapshot = original_squeue
        module.time.time = original_time


def run_scenarios(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    module = load_module()
    job_id = "12345"
    base_epoch = time.mktime(time.strptime("2026-05-22 08:00:00", "%Y-%m-%d %H:%M:%S"))
    slurm_start = write_log(root, job_id=job_id, base_epoch=base_epoch)
    problems: list[str] = []
    scenarios: list[dict[str, Any]] = []

    idle_root = root.parent / "idle_qwen3_235b_eagle3"
    idle_args = SimpleNamespace(
        artifact_root=idle_root,
        log=None,
        watcher_poll_seconds=120,
        watcher_max_polls=1,
        terminal_buffer_minutes=180,
        json_out=None,
        markdown_out=None,
    )
    idle_payload = module.build_payload(idle_args)
    scenarios.append({"name": "idle_without_rollout_jobs", "payload": idle_payload})
    if idle_payload.get("overall_status") != "idle":
        problems.append(f"idle_without_rollout_jobs overall {idle_payload.get('overall_status')!r} != 'idle'")
    if idle_payload.get("jobs") != []:
        problems.append("idle_without_rollout_jobs should not report synthetic jobs")

    risk_payload = build_payload(module, root, job_id=job_id, base_epoch=base_epoch, slurm_start=slurm_start)
    risk_job = risk_payload["jobs"][0]
    scenarios.append({"name": "risk_without_extension", "payload": risk_payload})
    if risk_payload.get("overall_status") != "warn":
        problems.append(f"risk_without_extension overall {risk_payload.get('overall_status')!r} != 'warn'")
    if risk_payload.get("watcher_timeout_risk_jobs") != [job_id]:
        problems.append(f"risk jobs {risk_payload.get('watcher_timeout_risk_jobs')!r} != [{job_id!r}]")
    if (risk_job.get("watcher_timeout") or {}).get("risk") != "risk":
        problems.append(f"risk_without_extension timeout {(risk_job.get('watcher_timeout') or {}).get('risk')!r} != 'risk'")

    extension_pid = root / "reports" / f"rollout_capture_{job_id}_watch_extension.pid"
    extension_pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    covered_payload = build_payload(module, root, job_id=job_id, base_epoch=base_epoch, slurm_start=slurm_start)
    covered_job = covered_payload["jobs"][0]
    scenarios.append({"name": "risk_covered_by_extension", "payload": covered_payload})
    if covered_payload.get("overall_status") != "waiting":
        problems.append(f"risk_covered_by_extension overall {covered_payload.get('overall_status')!r} != 'waiting'")
    if covered_payload.get("watcher_timeout_risk_jobs") != []:
        problems.append(f"covered risk jobs {covered_payload.get('watcher_timeout_risk_jobs')!r} != []")
    if covered_payload.get("watcher_timeout_covered_jobs") != [job_id]:
        problems.append(f"covered jobs {covered_payload.get('watcher_timeout_covered_jobs')!r} != [{job_id!r}]")
    timeout = covered_job.get("watcher_timeout") or {}
    if timeout.get("risk") != "covered_by_extension":
        problems.append(f"covered timeout risk {timeout.get('risk')!r} != 'covered_by_extension'")
    if timeout.get("raw_risk") != "risk":
        problems.append(f"covered raw risk {timeout.get('raw_risk')!r} != 'risk'")
    coverage = covered_job.get("watcher_extension_coverage") or {}
    if not coverage.get("alive"):
        problems.append("covered scenario did not report alive extension coverage")

    markdown = module.render_markdown(covered_payload)
    if "covered_by_extension" not in markdown or "alive:" not in markdown:
        problems.append("covered markdown does not mention extension coverage")
    idle_markdown = module.render_markdown(idle_payload)
    if "idle" not in idle_markdown or "no rollout watcher logs/jobs observed" not in idle_markdown:
        problems.append("idle markdown does not record idle queue state")
    return scenarios, problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Queue-Wait Summary Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | observed overall | timeout risk | covered jobs |",
        "| --- | --- | --- | --- |",
    ]
    for scenario in payload["scenarios"]:
        scenario_payload = scenario["payload"]
        jobs = scenario_payload.get("jobs") or [{}]
        timeout = (jobs[0].get("watcher_timeout") if jobs else {}) or {}
        lines.append(
            "| {name} | {overall} | {risk} | {covered} |".format(
                name=scenario["name"],
                overall=scenario_payload.get("overall_status"),
                risk=timeout.get("risk"),
                covered=",".join(scenario_payload.get("watcher_timeout_covered_jobs") or []) or "-",
            )
        )
    if payload.get("problems"):
        lines += ["", "## Problems", ""]
        lines.extend(f"- {problem}" for problem in payload["problems"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp = Path(tempfile.mkdtemp(prefix="qwen3_queue_wait_"))
    try:
        scenarios, problems = run_scenarios(temp / "qwen3_235b_eagle3")
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "overall_status": "pass" if not problems else "fail",
            "script": str(SCRIPT),
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
        return 0 if not problems else 1
    finally:
        if args.keep_temp:
            print(f"kept temp: {temp}", file=sys.stderr)
        else:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
