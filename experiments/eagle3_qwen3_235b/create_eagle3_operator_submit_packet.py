#!/usr/bin/env python3
"""Create a concise no-submit packet for the current Eagle3 operator actions.

The operator sheet is the full source of truth. This packet is a smaller view
for the next human action: which command to run now, which guard to run after
submission, and which reports prove the handoff is safe.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--operator-sheet-validation-json", type=Path)
    parser.add_argument("--operator-followup-validation-json", type=Path)
    parser.add_argument("--operator-execution-json", type=Path)
    parser.add_argument("--goal-evidence-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    defaults = {
        "operator_sheet_json": root / "reports/eagle3_operator_sheet.json",
        "operator_sheet_validation_json": root / "reports/eagle3_operator_sheet_validation.json",
        "operator_followup_validation_json": root / "reports/eagle3_operator_followups_validation.json",
        "operator_execution_json": root / "reports/eagle3_operator_execution.json",
        "goal_evidence_json": root / "reports/eagle3_goal_evidence.json",
        "json_out": root / "reports/eagle3_operator_submit_packet.json",
        "markdown_out": root / "reports/eagle3_operator_submit_packet.md",
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def load_json(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "not provided"
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"top-level JSON is not an object: {path}"
    return payload, None


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def report_status(payload: dict[str, Any] | None, error: str | None) -> str:
    if error:
        return "missing"
    return str((payload or {}).get("overall_status") or (payload or {}).get("status") or "unknown")


def execution_latest(payload: dict[str, Any] | None) -> dict[str, Any]:
    latest = (payload or {}).get("latest_by_action")
    return latest if isinstance(latest, dict) else {}


def packet_status(
    sheet: dict[str, Any] | None,
    sheet_error: str | None,
    sheet_validation: dict[str, Any] | None,
    sheet_validation_error: str | None,
    followup_validation: dict[str, Any] | None,
    followup_validation_error: str | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if sheet_error:
        blockers.append(f"operator sheet is not readable: {sheet_error}")
    if report_status(sheet_validation, sheet_validation_error) != "pass":
        blockers.append("operator sheet validation is not PASS")
    if report_status(followup_validation, followup_validation_error) != "pass":
        blockers.append("operator follow-up validation is not PASS")
    ready = [item for item in as_list((sheet or {}).get("ready_actions")) if isinstance(item, dict)]
    if not ready:
        blockers.append("operator sheet has no ready actions")
    return ("ready_for_operator_submit" if not blockers else "blocked", blockers)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    sheet, sheet_error = load_json(args.operator_sheet_json)
    sheet_validation, sheet_validation_error = load_json(args.operator_sheet_validation_json)
    followup_validation, followup_validation_error = load_json(args.operator_followup_validation_json)
    execution, execution_error = load_json(args.operator_execution_json)
    goal, goal_error = load_json(args.goal_evidence_json)
    status, blockers = packet_status(
        sheet,
        sheet_error,
        sheet_validation,
        sheet_validation_error,
        followup_validation,
        followup_validation_error,
    )
    latest = execution_latest(execution)

    ready_actions: list[dict[str, Any]] = []
    for item in as_list((sheet or {}).get("ready_actions")):
        if not isinstance(item, dict):
            continue
        action_id = str(item.get("id") or "")
        ready_actions.append(
            {
                "order": item.get("order"),
                "id": action_id,
                "title": item.get("title"),
                "status": item.get("status"),
                "submits_slurm": bool(item.get("submits_slurm")),
                "heavy_gpu": bool(item.get("heavy_gpu")),
                "after_policy": item.get("after_policy"),
                "execute_command": item.get("execute_command"),
                "planner_command": item.get("raw_command"),
                "post_submit_status_command": item.get("followup_status_command"),
                "execute_followup_after_terminal_command": item.get("execute_followup_command"),
                "execution_record": item.get("execution_record"),
                "followup_record": item.get("followup_record"),
                "already_executed": action_id in latest,
                "latest_execution": latest.get(action_id),
            }
        )

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "artifact_root": str(args.artifact_root),
        "blockers": blockers,
        "policy": {
            "mode": "no_submit_packet",
            "run_execute_command_only_after_review": True,
            "slurm_followup_requires_terminal_state": True,
            "do_not_use_run_after_for_slurm": True,
        },
        "report_statuses": {
            "operator_sheet": report_status(sheet, sheet_error),
            "operator_sheet_validation": report_status(sheet_validation, sheet_validation_error),
            "operator_followup_validation": report_status(followup_validation, followup_validation_error),
            "operator_execution": report_status(execution, execution_error),
            "goal_evidence": report_status(goal, goal_error),
        },
        "report_paths": {
            "operator_sheet": str(args.operator_sheet_json),
            "operator_sheet_validation": str(args.operator_sheet_validation_json),
            "operator_followup_validation": str(args.operator_followup_validation_json),
            "operator_execution": str(args.operator_execution_json),
            "goal_evidence": str(args.goal_evidence_json),
        },
        "counts": {
            "ready_actions": len(ready_actions),
            "slurm_ready_actions": sum(1 for item in ready_actions if item["submits_slurm"]),
            "heavy_gpu_ready_actions": sum(1 for item in ready_actions if item["heavy_gpu"]),
            "already_executed_actions": sum(1 for item in ready_actions if item["already_executed"]),
        },
        "followup_state_counts": (followup_validation or {}).get("followup_state_counts") or {},
        "goal_open_requirements": (goal or {}).get("open_requirements") or [],
        "ready_actions": ready_actions,
    }


def md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "/").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Submit Packet",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Artifact root: `{payload['artifact_root']}`",
        "",
        "This packet is no-submit. It extracts the current ready actions from the operator sheet.",
        "",
        "## Report Gates",
        "",
        "| report | status | path |",
        "| --- | --- | --- |",
    ]
    for key, status in payload["report_statuses"].items():
        lines.append(f"| {key} | `{status}` | `{payload['report_paths'].get(key)}` |")

    if payload["blockers"]:
        lines += ["", "## Blockers", ""]
        lines.extend(f"- {blocker}" for blocker in payload["blockers"])

    lines += [
        "",
        "## Ready Actions",
        "",
        "| order | action | Slurm | heavy GPU | already executed |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for item in payload["ready_actions"]:
        lines.append(
            f"| {item.get('order')} | `{md_escape(item.get('id'))}` | "
            f"{str(item.get('submits_slurm')).lower()} | {str(item.get('heavy_gpu')).lower()} | "
            f"{str(item.get('already_executed')).lower()} |"
        )
    if not payload["ready_actions"]:
        lines.append("| - | - | - | - | - |")

    for item in payload["ready_actions"]:
        lines += [
            "",
            f"### {item.get('order')}. {md_escape(item.get('title'))}",
            "",
            "Execute after review:",
            "",
            "```bash",
            str(item.get("execute_command") or ""),
            "```",
        ]
        if item.get("planner_command"):
            lines += [
                "",
                "Planner command executed by the runner:",
                "",
                "```bash",
                str(item.get("planner_command") or ""),
                "```",
            ]
        if item.get("post_submit_status_command"):
            lines += [
                "",
                "Check Slurm terminal state before analyzers:",
                "",
                "```bash",
                str(item.get("post_submit_status_command") or ""),
                "```",
                "",
                "Only after the guard reports READY_FOR_FOLLOWUP or PASS:",
                "",
                "```bash",
                str(item.get("execute_followup_after_terminal_command") or ""),
                "```",
            ]

    open_items = payload.get("goal_open_requirements") or []
    lines += ["", "## Goal Still Open", ""]
    if open_items:
        for item in open_items[:12]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = with_defaults(parse_args())
    payload = build_payload(args)
    markdown = render_markdown(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0 if payload["overall_status"] == "ready_for_operator_submit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
