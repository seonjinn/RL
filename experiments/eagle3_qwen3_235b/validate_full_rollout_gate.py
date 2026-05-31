#!/usr/bin/env python3
"""Validate the full SWE-Gym after-smoke rollout gate.

The validator uses synthetic reports and never submits a Slurm job. It checks
that the gate waits while smoke is unproven, becomes ready only after smoke
PASS and full preflight readiness, requires ``--allow-heavy-gpu`` for execute,
and recognizes already-materialized / needs-materialize full rollout states.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/submit_full_rollout_after_smoke_if_ready.py"


def load_gate_module() -> Any:
    spec = importlib.util.spec_from_file_location("submit_full_rollout_after_smoke_if_ready", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gate module: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_preflight(root: Path, submit_ready: bool = True) -> dict[str, Any]:
    return {
        "overall_status": "pass" if submit_ready else "fail",
        "submit_ready": submit_ready,
        "wandb_name": "full-rollout-gate-validator-not-active",
        "output_conversations": str(root / "data/full.jsonl"),
        "rollout_log_dir": str(root / "logs/full"),
        "commands": {"submit": "echo Submitted batch job 123456"},
    }


def run_gate(root: Path, smoke: dict[str, Any] | None, preflight: dict[str, Any] | None, *extra: str) -> dict[str, Any]:
    reports = root / "reports"
    smoke_path = reports / "smoke_state.json"
    preflight_path = reports / "full_preflight.json"
    out_path = reports / "gate.json"
    if smoke is not None:
        write_json(smoke_path, smoke)
    if preflight is not None:
        write_json(preflight_path, preflight)
    cmd = [
        "python3",
        str(SCRIPT),
        "--artifact-root",
        str(root),
        "--smoke-state-json",
        str(smoke_path),
        "--full-preflight-json",
        str(preflight_path),
        "--json-out",
        str(out_path),
        *extra,
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    return {"returncode": result.returncode, "output_tail": result.stdout[-4000:], "payload": payload}


def decision(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    item = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return item.get("overall_status"), item.get("next_step")


def smoke(status: str) -> dict[str, Any]:
    return {"decision": {"overall_status": status, "detail": f"synthetic {status}"}}


def check_case(
    cases: list[dict[str, Any]],
    name: str,
    result: dict[str, Any],
    expected_status: str,
    expected_next_step: str,
    expected_returncode: int = 0,
    submitted_job_id: str | None = None,
    watcher_started: bool | None = None,
) -> None:
    status, next_step = decision(result["payload"])
    watcher_result = result["payload"].get("watcher_result") if isinstance(result["payload"].get("watcher_result"), dict) else {}
    actual_watcher_started = bool(watcher_result.get("pid"))
    ok = (
        result["returncode"] == expected_returncode
        and status == expected_status
        and next_step == expected_next_step
        and (submitted_job_id is None or result["payload"].get("submitted_job_id") == submitted_job_id)
        and (watcher_started is None or actual_watcher_started == watcher_started)
    )
    cases.append(
        {
            "name": name,
            "status": "pass" if ok else "fail",
            "returncode": result["returncode"],
            "expected_returncode": expected_returncode,
            "actual_overall": status,
            "expected_overall": expected_status,
            "actual_next_step": next_step,
            "expected_next_step": expected_next_step,
            "submitted_job_id": result["payload"].get("submitted_job_id"),
            "watcher_started": actual_watcher_started,
            "output_tail": result["output_tail"][-1000:],
        }
    )


def run_cases() -> list[dict[str, Any]]:
    root = Path(tempfile.mkdtemp(prefix="full_rollout_gate_validator_"))
    cases: list[dict[str, Any]] = []
    try:
        gate = load_gate_module()
        parsed = gate.parse_active_job_rows(
            "123|PENDING|qwen3-235b-swe-rollout-vllm0102src-swegym-full-dryrun\n"
            "124|PENDING|unrelated\n",
            "qwen3-235b-swe-rollout-vllm0102src-swegym-full-dryrun",
        )
        cases.append(
            {
                "name": "active_job_name_parser",
                "status": "pass" if parsed == [
                    {
                        "job_id": "123",
                        "state": "PENDING",
                        "name": "qwen3-235b-swe-rollout-vllm0102src-swegym-full-dryrun",
                    }
                ] else "fail",
                "returncode": 0,
                "expected_returncode": 0,
                "actual_overall": "parser_match" if parsed else "parser_miss",
                "expected_overall": "parser_match",
                "actual_next_step": "active",
                "expected_next_step": "active",
                "submitted_job_id": None,
                "watcher_started": False,
                "output_tail": json.dumps(parsed, sort_keys=True),
            }
        )
        default_root = root / "default_active_smoke_state"
        default_reports = default_root / "reports"
        fallback_prefix = "rollout_capture_old_2861605"
        fallback_path = default_reports / f"{fallback_prefix}_state_advance.json"
        active_path = default_reports / "rollout_capture_active_2863716_state_advance.json"
        write_json(
            default_reports / "rollout_queue_wait_summary.json",
            {
                "jobs": [
                    {
                        "job_id": "2863716",
                        "current_squeue": {"job_id": "2863716", "state": "PENDING"},
                    }
                ]
            },
        )
        write_json(fallback_path, {"job_id": "2861605", "decision": {"overall_status": "running"}})
        write_json(active_path, {"job_id": "2863716", "decision": {"overall_status": "running"}})
        selected_smoke, _ = gate.default_paths(
            argparse.Namespace(
                artifact_root=default_root,
                smoke_state_json=None,
                smoke_report_prefix=fallback_prefix,
                full_preflight_json=None,
            )
        )
        cases.append(
            {
                "name": "default_smoke_state_selects_active_rollout",
                "status": "pass" if selected_smoke == active_path else "fail",
                "returncode": 0,
                "expected_returncode": 0,
                "actual_overall": selected_smoke.name,
                "expected_overall": active_path.name,
                "actual_next_step": "selected",
                "expected_next_step": "selected",
                "submitted_job_id": None,
                "watcher_started": False,
                "output_tail": str(selected_smoke),
            }
        )
        check_case(
            cases,
            "missing_smoke_waits",
            run_gate(root / "missing_smoke", None, base_preflight(root / "missing_smoke")),
            "waiting",
            "refresh_smoke_state",
        )
        check_case(
            cases,
            "running_smoke_waits",
            run_gate(root / "running", smoke("running"), base_preflight(root / "running")),
            "waiting",
            "poll_smoke",
        )
        check_case(
            cases,
            "needs_materialize_smoke_waits",
            run_gate(root / "needs_materialize_smoke", smoke("needs_materialize"), base_preflight(root / "needs_materialize_smoke")),
            "waiting",
            "materialize_smoke",
        )
        check_case(
            cases,
            "failed_smoke_fails",
            run_gate(root / "failed_smoke", smoke("fail"), base_preflight(root / "failed_smoke")),
            "fail",
            "inspect_smoke",
            expected_returncode=1,
        )
        check_case(
            cases,
            "pass_smoke_bad_preflight_fails",
            run_gate(root / "bad_preflight", smoke("pass"), base_preflight(root / "bad_preflight", submit_ready=False)),
            "fail",
            "inspect_full_preflight",
            expected_returncode=1,
        )
        dryrun_name_root = root / "dryrun_name"
        dryrun_name_preflight = base_preflight(dryrun_name_root)
        dryrun_name_preflight["wandb_name"] = "qwen3-235b-swe-rollout-full-dryrun"
        check_case(
            cases,
            "pass_smoke_dryrun_full_name_fails",
            run_gate(dryrun_name_root, smoke("pass"), dryrun_name_preflight),
            "fail",
            "inspect_full_preflight",
            expected_returncode=1,
        )
        ready_root = root / "ready"
        check_case(
            cases,
            "pass_smoke_ready",
            run_gate(ready_root, smoke("pass"), base_preflight(ready_root)),
            "ready",
            "submit_full_rollout",
        )
        output_root = root / "already_materialized"
        output_path = output_root / "data/full.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"id":"x","conversations":[]}\n', encoding="utf-8")
        check_case(
            cases,
            "full_output_exists_pass",
            run_gate(output_root, smoke("pass"), base_preflight(output_root)),
            "pass",
            "full_already_materialized",
        )
        train_root = root / "needs_materialize_full"
        train_dir = train_root / "logs/full"
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "train_data_step0.jsonl").write_text('{"x":1}\n', encoding="utf-8")
        check_case(
            cases,
            "full_train_data_needs_materialize",
            run_gate(train_root, smoke("pass"), base_preflight(train_root)),
            "needs_materialize",
            "materialize_full",
        )
        check_case(
            cases,
            "execute_requires_allow",
            run_gate(root / "execute_no_allow", smoke("pass"), base_preflight(root / "execute_no_allow"), "--execute"),
            "fail",
            "rerun_with_allow_heavy_gpu",
            expected_returncode=1,
        )
        check_case(
            cases,
            "execute_with_allow_submits",
            run_gate(root / "execute_allow", smoke("pass"), base_preflight(root / "execute_allow"), "--execute", "--allow-heavy-gpu"),
            "submitted",
            "watch_full_rollout",
            submitted_job_id="123456",
        )
        check_case(
            cases,
            "execute_start_watcher_requires_allow_background",
            run_gate(
                root / "execute_watcher_no_allow",
                smoke("pass"),
                base_preflight(root / "execute_watcher_no_allow"),
                "--execute",
                "--allow-heavy-gpu",
                "--start-watcher",
            ),
            "fail",
            "rerun_with_allow_background",
            expected_returncode=1,
            watcher_started=False,
        )
        check_case(
            cases,
            "execute_with_allow_starts_watcher",
            run_gate(
                root / "execute_watcher_allow",
                smoke("pass"),
                base_preflight(root / "execute_watcher_allow"),
                "--execute",
                "--allow-heavy-gpu",
                "--start-watcher",
                "--allow-background",
                "--watcher-max-polls",
                "0",
                "--watcher-poll-seconds",
                "1",
            ),
            "submitted",
            "watch_full_rollout",
            submitted_job_id="123456",
            watcher_started=True,
        )
        return cases
    finally:
        shutil.rmtree(root, ignore_errors=True)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Full Rollout Gate Validator",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        "",
        "| case | status | expected | actual |",
        "| --- | --- | --- | --- |",
    ]
    for case in payload["cases"]:
        expected = f"{case['expected_overall']}/{case['expected_next_step']}"
        actual = f"{case['actual_overall']}/{case['actual_next_step']}"
        lines.append(f"| {case['name']} | {case['status'].upper()} | `{expected}` | `{actual}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    cases = run_cases()
    overall = "pass" if all(case["status"] == "pass" for case in cases) else "fail"
    payload = {"overall_status": overall, "cases": cases}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
