#!/usr/bin/env python3
"""Summarize the current Eagle3 operator queue without submitting jobs.

This report joins the next-action plan, operator sheet, execution records,
follow-up guard reports, and ready-submit preflight so an operator can see the
next concrete step for each ready action.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    reports = artifact_root / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path(os.environ.get("NEXT_ACTION_PLAN_JSON", reports / "eagle3_next_actions.json")),
    )
    parser.add_argument(
        "--operator-sheet-json",
        type=Path,
        default=Path(os.environ.get("OPERATOR_SHEET_JSON", reports / "eagle3_operator_sheet.json")),
    )
    parser.add_argument(
        "--operator-execution-json",
        type=Path,
        default=Path(os.environ.get("OPERATOR_EXECUTION_JSON", reports / "eagle3_operator_execution.json")),
    )
    parser.add_argument(
        "--operator-followup-validation-json",
        type=Path,
        default=Path(os.environ.get("OPERATOR_FOLLOWUP_VALIDATION_JSON", reports / "eagle3_operator_followups_validation.json")),
    )
    parser.add_argument(
        "--operator-ready-submit-preflight-json",
        type=Path,
        default=Path(os.environ.get("OPERATOR_READY_SUBMIT_PREFLIGHT_JSON", reports / "eagle3_operator_ready_submit_preflight.json")),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(os.environ.get("OPERATOR_QUEUE_JSON", reports / "eagle3_operator_queue.json")),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path(os.environ.get("OPERATOR_QUEUE_MARKDOWN", reports / "eagle3_operator_queue.md")),
    )
    parser.add_argument("--fail-on-error", action="store_true")
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


def report_status(payload: Any, error: str | None) -> str:
    if error:
        return "missing"
    data = as_dict(payload)
    return str(data.get("overall_status") or data.get("status") or "unknown")


def plan_action_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in as_list(plan.get("next_actions")):
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def ready_submit_action_ids(payload: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in as_list(payload.get("ready_actions")):
        if isinstance(item, dict) and item.get("id"):
            result.add(str(item["id"]))
        elif isinstance(item, str):
            result.add(item)
    return result


def read_followup_report(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value or ""))
    if not path_value:
        return {"status": "missing", "path": None, "detail": "operator sheet has no followup_record"}
    payload, error = load_json(path)
    if error:
        return {"status": "missing", "path": str(path), "detail": error}
    data = as_dict(payload)
    return {
        "status": str(data.get("overall_status") or "unknown"),
        "path": str(path),
        "mode": data.get("mode"),
        "detail": data.get("detail"),
        "job_count": len(as_list(data.get("jobs"))),
        "terminal_job_count": sum(1 for job in as_list(data.get("jobs")) if isinstance(job, dict) and job.get("terminal")),
        "after_returncode_count": len(as_list(data.get("after_returncodes"))),
    }


def decide_next_step(
    sheet_item: dict[str, Any],
    execution: dict[str, Any] | None,
    followup: dict[str, Any],
    ready_preflight_pass: bool,
) -> tuple[str, str, str | None]:
    action_id = str(sheet_item.get("id") or "")
    if sheet_item.get("status") != "ready_for_operator":
        return "blocked", "action is not ready_for_operator", None

    if execution is None:
        if not ready_preflight_pass:
            return "blocked", "operator ready-submit preflight is not PASS for this action", None
        return "execute_submit", "execute the guarded ready action command", sheet_item.get("execute_command")
    if execution.get("returncode") != 0:
        return "inspect_execution_failure", "operator execution record has nonzero or missing returncode", None

    if not sheet_item.get("submits_slurm"):
        return "refresh_state", "non-Slurm action executed; refresh operator state", None

    status = str(followup.get("status") or "missing")
    if status in {"missing", "not_submitted"}:
        return "poll_slurm", "inspect Slurm job state with the follow-up guard", sheet_item.get("followup_status_command")
    if status in {"waiting", "unknown", "blocked"}:
        return "keep_polling", str(followup.get("detail") or "Slurm terminal state is not proven yet"), sheet_item.get("followup_status_command")
    if status == "ready_for_followup":
        return "execute_followup", "Slurm job is terminal; run guarded follow-up analyzers", sheet_item.get("execute_followup_command")
    if status == "pass":
        return "refresh_state", f"{action_id} follow-up analyzers have passed", None
    if status == "fail":
        return "inspect_followup_failure", "follow-up guard or analyzer reported failure", None
    return "inspect_followup_state", f"unclassified follow-up status: {status}", sheet_item.get("followup_status_command")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    plan_raw, plan_error = load_json(args.plan_json)
    sheet_raw, sheet_error = load_json(args.operator_sheet_json)
    execution_raw, execution_error = load_json(args.operator_execution_json)
    followup_validation_raw, followup_validation_error = load_json(args.operator_followup_validation_json)
    ready_preflight_raw, ready_preflight_error = load_json(args.operator_ready_submit_preflight_json)

    plan = as_dict(plan_raw)
    sheet = as_dict(sheet_raw)
    execution = as_dict(execution_raw)
    followup_validation = as_dict(followup_validation_raw)
    ready_preflight = as_dict(ready_preflight_raw)
    plan_actions = plan_action_map(plan)
    latest_execution = as_dict(execution.get("latest_by_action"))
    ready_preflight_ids = ready_submit_action_ids(ready_preflight)
    ready_preflight_status = report_status(ready_preflight, ready_preflight_error)
    ready_preflight_submit_ready = ready_preflight.get("submit_ready") is True

    rows: list[dict[str, Any]] = []
    for item in as_list(sheet.get("ready_actions")):
        if not isinstance(item, dict):
            continue
        action_id = str(item.get("id") or "")
        exec_row = as_dict(latest_execution.get(action_id)) if action_id in latest_execution else None
        followup = read_followup_report(item.get("followup_record")) if item.get("submits_slurm") else {}
        action_ready_preflight_pass = (
            ready_preflight_status == "pass"
            and ready_preflight_submit_ready
            and (not ready_preflight_ids or action_id in ready_preflight_ids)
        )
        next_step, detail, command = decide_next_step(item, exec_row, followup, action_ready_preflight_pass)
        rows.append(
            {
                "order": item.get("order"),
                "id": action_id,
                "stage": item.get("stage") or as_dict(plan_actions.get(action_id)).get("stage"),
                "status": item.get("status"),
                "submits_slurm": bool(item.get("submits_slurm")),
                "heavy_gpu": bool(item.get("heavy_gpu")),
                "execution_record": item.get("execution_record"),
                "execution_status": "not_started" if exec_row is None else ("pass" if exec_row.get("returncode") == 0 else "fail"),
                "execution_returncode": None if exec_row is None else exec_row.get("returncode"),
                "followup_record": item.get("followup_record"),
                "followup_status": followup.get("status") if followup else None,
                "followup_detail": followup.get("detail") if followup else None,
                "followup_job_count": followup.get("job_count") if followup else None,
                "ready_submit_preflight": "pass" if action_ready_preflight_pass else ready_preflight_status,
                "next_step": next_step,
                "next_step_detail": detail,
                "next_command": command,
            }
        )

    step_counts: dict[str, int] = {}
    for row in rows:
        step = str(row["next_step"])
        step_counts[step] = step_counts.get(step, 0) + 1

    input_statuses = {
        "plan": report_status(plan, plan_error),
        "operator_sheet": report_status(sheet, sheet_error),
        "operator_execution": report_status(execution, execution_error),
        "operator_followup_validation": report_status(followup_validation, followup_validation_error),
        "operator_ready_submit_preflight": ready_preflight_status,
    }
    errors = {
        key: value
        for key, value in {
            "plan": plan_error,
            "operator_sheet": sheet_error,
            "operator_execution": execution_error,
            "operator_followup_validation": followup_validation_error,
            "operator_ready_submit_preflight": ready_preflight_error,
        }.items()
        if value
    }

    if errors:
        overall = "warn"
    elif any(row["next_step"].startswith("inspect_") or row["next_step"] == "blocked" for row in rows):
        overall = "blocked"
    elif any(row["next_step"] == "execute_followup" for row in rows):
        overall = "ready_for_followup"
    elif any(row["next_step"] == "execute_submit" for row in rows):
        overall = "ready_for_operator_submit"
    elif any(row["next_step"] in {"poll_slurm", "keep_polling"} for row in rows):
        overall = "waiting_for_slurm"
    elif rows:
        overall = "current_ready_set_processed"
    else:
        overall = "no_ready_actions"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "artifact_root": str(args.artifact_root),
        "inputs": {
            "plan_json": str(args.plan_json),
            "operator_sheet_json": str(args.operator_sheet_json),
            "operator_execution_json": str(args.operator_execution_json),
            "operator_followup_validation_json": str(args.operator_followup_validation_json),
            "operator_ready_submit_preflight_json": str(args.operator_ready_submit_preflight_json),
        },
        "input_statuses": input_statuses,
        "input_errors": errors,
        "counts": {
            "ready_actions": len(rows),
            "slurm_actions": sum(1 for row in rows if row["submits_slurm"]),
            "heavy_gpu_actions": sum(1 for row in rows if row["heavy_gpu"]),
            "next_steps": step_counts,
        },
        "ready_submit_preflight": {
            "overall_status": ready_preflight_status,
            "submit_ready": ready_preflight_submit_ready,
            "counts": ready_preflight.get("counts"),
        },
        "followup_state_counts": followup_validation.get("followup_state_counts"),
        "queue": rows,
        "next_command": next((row.get("next_command") for row in rows if row.get("next_command")), None),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Queue",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Artifact root: `{payload['artifact_root']}`",
        "",
        "| action | stage | slurm | heavy GPU | execution | follow-up | next step |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["queue"]:
        lines.append(
            f"| `{row['id']}` | `{row.get('stage') or '-'}` | {str(row['submits_slurm']).lower()} | "
            f"{str(row['heavy_gpu']).lower()} | `{row['execution_status']}` | "
            f"`{row.get('followup_status') or '-'}` | `{row['next_step']}` |"
        )
    if not payload["queue"]:
        lines.append("| - | - | - | - | - | - | no ready actions |")

    lines += ["", "## Next Command", ""]
    if payload.get("next_command"):
        lines += ["```bash", str(payload["next_command"]), "```"]
    else:
        lines.append("No immediate command is available from the current queue state.")

    lines += ["", "## Details", "", "| action | detail |", "| --- | --- |"]
    for row in payload["queue"]:
        detail = str(row.get("next_step_detail") or "").replace("|", "/")
        lines.append(f"| `{row['id']}` | {detail} |")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    markdown = render_markdown(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 1 if args.fail_on_error and payload["overall_status"] in {"blocked", "warn"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
