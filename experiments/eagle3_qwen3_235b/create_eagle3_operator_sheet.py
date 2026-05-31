#!/usr/bin/env python3
"""Create an operator command sheet from eagle3_next_actions.json.

The sheet is intentionally no-submit: it does not run Slurm, GPU, or analyzer
commands. It turns the planner output into an ordered review/execution page
that makes the required allow flags and follow-up analyzer commands explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")

RUNNER = "experiments/eagle3_qwen3_235b/run_eagle3_next_action.py"
FOLLOWUP_RUNNER = "experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py"
PLANNER = "experiments/eagle3_qwen3_235b/plan_eagle3_next_actions.py"
PLAN_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_next_action_plan.py"
TRANSITION_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_next_action_transitions.py"
QUEUE_TRANSITION_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_operator_queue_transitions.py"
COMPLETION_CONTRACT = "experiments/eagle3_qwen3_235b/validate_eagle3_completion_contract.py"
SLURM_CAPACITY = "experiments/eagle3_qwen3_235b/probe_eagle3_slurm_capacity.py"
RESOURCE_PROFILE_APPLICATION = "experiments/eagle3_qwen3_235b/validate_eagle3_resource_profile_application.py"
SHEET_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_operator_sheet.py"
EXECUTION_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_operator_execution.py"
FOLLOWUP_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_operator_followups.py"
SUBMIT_PACKET = "experiments/eagle3_qwen3_235b/create_eagle3_operator_submit_packet.py"
SUBMIT_PACKET_VALIDATOR = "experiments/eagle3_qwen3_235b/validate_eagle3_operator_submit_packet.py"
READY_SUBMIT_PREFLIGHT = "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py"
REFRESH_STATE = "experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py"
QUEUE_SUMMARY = "experiments/eagle3_qwen3_235b/summarize_eagle3_operator_queue.py"
OPERATOR_SHEET = "experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py"
GOAL_EVIDENCE = "experiments/eagle3_qwen3_235b/audit_eagle3_goal_evidence.py"

def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path(os.environ.get("NEXT_ACTION_PLAN_JSON", artifact_root / "reports/eagle3_next_actions.json")),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(os.environ.get("OPERATOR_SHEET_JSON", artifact_root / "reports/eagle3_operator_sheet.json")),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path(os.environ.get("OPERATOR_SHEET_MARKDOWN", artifact_root / "reports/eagle3_operator_sheet.md")),
    )
    parser.add_argument(
        "--blocked-limit",
        type=int,
        default=8,
        help="Maximum number of non-ready actions to include in the sheet.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"next-action plan is not visible: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot parse next-action plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"next-action plan is not a JSON object: {path}")
    return payload


def actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("next_actions") or []
    return [item for item in raw if isinstance(item, dict)]


def sort_actions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # plan_eagle3_next_actions.py already emits next_actions in priority order.
    # Preserve that exact order so every downstream operator artifact agrees.
    return list(items)


def shell_join(command: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def runner_base(args: argparse.Namespace, action_id: str) -> list[str | Path]:
    return [
        "python3",
        RUNNER,
        "--artifact-root",
        args.artifact_root,
        "--plan-json",
        args.plan_json,
        "--action-id",
        action_id,
    ]


def execution_flags(action: dict[str, Any]) -> list[str]:
    flags = ["--execute"]
    if action.get("submits_slurm"):
        flags.append("--allow-slurm")
    if action.get("heavy_gpu"):
        flags.append("--allow-heavy-gpu")
    return flags


def safe_action_filename(action_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", action_id).strip("_") or "selected_action"


def execution_record_path(args: argparse.Namespace, action_id: str, order: int) -> Path:
    return args.artifact_root / "reports" / "operator_execution" / f"{order:02d}_{safe_action_filename(action_id)}.json"


def followup_record_path(args: argparse.Namespace, action_id: str, order: int) -> Path:
    return args.artifact_root / "reports" / "operator_followups" / f"{order:02d}_{safe_action_filename(action_id)}.json"


def followup_markdown_path(args: argparse.Namespace, action_id: str, order: int) -> Path:
    return args.artifact_root / "reports" / "operator_followups" / f"{order:02d}_{safe_action_filename(action_id)}.md"


def command_record(args: argparse.Namespace, action: dict[str, Any], order: int) -> dict[str, Any]:
    action_id = str(action.get("id") or f"action_{order}")
    base = runner_base(args, action_id)
    record_path = execution_record_path(args, action_id, order)
    record_flags = ["--json-out", record_path]
    execute = base + execution_flags(action) + record_flags
    execute_with_after = None if action.get("submits_slurm") else base + execution_flags(action) + ["--run-after"] + record_flags
    followup_json = followup_record_path(args, action_id, order)
    followup_markdown = followup_markdown_path(args, action_id, order)
    followup_base = [
        "python3",
        FOLLOWUP_RUNNER,
        "--artifact-root",
        args.artifact_root,
        "--plan-json",
        args.plan_json,
        "--operator-sheet-json",
        args.json_out,
        "--action-id",
        action_id,
        "--execution-record",
        record_path,
        "--json-out",
        followup_json,
        "--markdown-out",
        followup_markdown,
    ]
    followup_status = followup_base if action.get("submits_slurm") else None
    execute_followup = followup_base + ["--execute-after"] if action.get("submits_slurm") else None
    after_policy_detail = (
        "Run after_commands only after the submitted Slurm job reaches a terminal state; sbatch submission returns before the job runs."
        if action.get("submits_slurm")
        else "after_commands can be run immediately after the action command returns zero."
    )
    return {
        "order": order,
        "id": action_id,
        "title": action.get("title"),
        "stage": action.get("stage"),
        "status": action.get("status"),
        "reason": action.get("reason"),
        "report": action.get("report"),
        "submits_slurm": bool(action.get("submits_slurm")),
        "heavy_gpu": bool(action.get("heavy_gpu")),
        "print_command": shell_join(base),
        "execute_command": shell_join(execute),
        "execute_with_after_commands": shell_join(execute_with_after) if execute_with_after else None,
        "followup_status_command": shell_join(followup_status) if followup_status else None,
        "execute_followup_command": shell_join(execute_followup) if execute_followup else None,
        "after_policy": "after_slurm_terminal_state" if action.get("submits_slurm") else "after_command_success",
        "after_policy_detail": after_policy_detail,
        "execution_record": str(record_path),
        "followup_record": str(followup_json) if action.get("submits_slurm") else None,
        "raw_command": action.get("command"),
        "after_commands": action.get("after_commands") or [],
    }


def refresh_commands(args: argparse.Namespace) -> list[str]:
    return [
        shell_join(
            [
                "python3",
                PLANNER,
                "--artifact-root",
                args.artifact_root,
                "--json-out",
                args.artifact_root / "reports/eagle3_next_actions.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_next_actions.md",
            ]
        ),
        shell_join(
            [
                "python3",
                PLAN_VALIDATOR,
                "--plan-json",
                args.plan_json,
                "--json-out",
                args.artifact_root / "reports/eagle3_next_actions_validation.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_next_actions_validation.md",
            ]
        ),
        shell_join(
            [
                "python3",
                TRANSITION_VALIDATOR,
                "--json-out",
                args.artifact_root / "reports/eagle3_next_action_transitions.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_next_action_transitions.md",
            ]
        ),
        shell_join(
            [
                "python3",
                QUEUE_TRANSITION_VALIDATOR,
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_queue_transitions.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_queue_transitions.md",
            ]
        ),
        shell_join(
            [
                "python3",
                COMPLETION_CONTRACT,
                "--json-out",
                args.artifact_root / "reports/eagle3_completion_contract.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_completion_contract.md",
            ]
        ),
        shell_join(
            [
                "python3",
                SLURM_CAPACITY,
                "--artifact-root",
                args.artifact_root,
                "--json-out",
                args.artifact_root / "reports/eagle3_slurm_capacity.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_slurm_capacity.md",
                "--env-out",
                args.artifact_root / "reports/eagle3_resource_profile.env",
            ]
        ),
        shell_join(
            [
                "python3",
                RESOURCE_PROFILE_APPLICATION,
                "--artifact-root",
                args.artifact_root,
                "--resource-profile-env",
                args.artifact_root / "reports/eagle3_resource_profile.env",
                "--json-out",
                args.artifact_root / "reports/eagle3_resource_profile_application.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_resource_profile_application.md",
            ]
        ),
        shell_join(
            [
                "python3",
                OPERATOR_SHEET,
                "--artifact-root",
                args.artifact_root,
                "--plan-json",
                args.plan_json,
                "--json-out",
                args.json_out,
                "--markdown-out",
                args.markdown_out,
            ]
        ),
        shell_join(
            [
                "python3",
                SHEET_VALIDATOR,
                "--artifact-root",
                args.artifact_root,
                "--plan-json",
                args.plan_json,
                "--operator-sheet-json",
                args.json_out,
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_sheet_validation.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_sheet_validation.md",
            ]
        ),
        shell_join(
            [
                "python3",
                EXECUTION_VALIDATOR,
                "--artifact-root",
                args.artifact_root,
                "--plan-json",
                args.plan_json,
                "--operator-sheet-json",
                args.json_out,
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_execution.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_execution.md",
            ]
        ),
        shell_join(
            [
                "python3",
                FOLLOWUP_VALIDATOR,
                "--artifact-root",
                args.artifact_root,
                "--plan-json",
                args.plan_json,
                "--operator-sheet-json",
                args.json_out,
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_followups_validation.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_followups_validation.md",
            ]
        ),
        shell_join(
            [
                "python3",
                GOAL_EVIDENCE,
                "--artifact-root",
                args.artifact_root,
                "--json-out",
                args.artifact_root / "reports/eagle3_goal_evidence.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_goal_evidence.md",
            ]
        ),
        shell_join(
            [
                "python3",
                SUBMIT_PACKET,
                "--artifact-root",
                args.artifact_root,
                "--operator-sheet-json",
                args.json_out,
                "--operator-sheet-validation-json",
                args.artifact_root / "reports/eagle3_operator_sheet_validation.json",
                "--operator-followup-validation-json",
                args.artifact_root / "reports/eagle3_operator_followups_validation.json",
                "--operator-execution-json",
                args.artifact_root / "reports/eagle3_operator_execution.json",
                "--goal-evidence-json",
                args.artifact_root / "reports/eagle3_goal_evidence.json",
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_submit_packet.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_submit_packet.md",
            ]
        ),
        shell_join(
            [
                "python3",
                SUBMIT_PACKET_VALIDATOR,
                "--artifact-root",
                args.artifact_root,
                "--operator-submit-packet-json",
                args.artifact_root / "reports/eagle3_operator_submit_packet.json",
                "--operator-sheet-json",
                args.json_out,
                "--operator-sheet-validation-json",
                args.artifact_root / "reports/eagle3_operator_sheet_validation.json",
                "--operator-followup-validation-json",
                args.artifact_root / "reports/eagle3_operator_followups_validation.json",
                "--operator-execution-json",
                args.artifact_root / "reports/eagle3_operator_execution.json",
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_submit_packet_validation.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_submit_packet_validation.md",
            ]
        ),
        shell_join(
            [
                "python3",
                READY_SUBMIT_PREFLIGHT,
                "--artifact-root",
                args.artifact_root,
                "--operator-sheet-json",
                args.json_out,
                "--operator-submit-packet-validation-json",
                args.artifact_root / "reports/eagle3_operator_submit_packet_validation.json",
                "--rollout-submit-preflight-json",
                args.artifact_root / "reports/rollout_capture_submit_preflight.json",
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_ready_submit_preflight.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_ready_submit_preflight.md",
            ]
        ),
        shell_join(
            [
                "python3",
                QUEUE_SUMMARY,
                "--artifact-root",
                args.artifact_root,
                "--plan-json",
                args.plan_json,
                "--operator-sheet-json",
                args.json_out,
                "--operator-execution-json",
                args.artifact_root / "reports/eagle3_operator_execution.json",
                "--operator-followup-validation-json",
                args.artifact_root / "reports/eagle3_operator_followups_validation.json",
                "--operator-ready-submit-preflight-json",
                args.artifact_root / "reports/eagle3_operator_ready_submit_preflight.json",
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_queue.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_queue.md",
            ]
        ),
        shell_join(
            [
                "python3",
                REFRESH_STATE,
                "--artifact-root",
                args.artifact_root,
                "--json-out",
                args.artifact_root / "reports/eagle3_operator_state_refresh.json",
                "--markdown-out",
                args.artifact_root / "reports/eagle3_operator_state_refresh.md",
            ]
        ),
    ]


def operator_status(ready: list[dict[str, Any]], waiting: list[dict[str, Any]], plan: dict[str, Any]) -> str:
    if ready:
        return "ready_for_operator"
    plan_status = str(plan.get("overall_status") or "unknown")
    if plan_status == "fail":
        return "blocked_by_failed_plan"
    if waiting:
        return "waiting_on_prerequisites"
    return "no_ready_actions"


def action_counts(ready: list[dict[str, Any]], waiting: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "ready": len(ready),
        "waiting_or_blocked": len(waiting),
        "slurm_ready": sum(1 for item in ready if item.get("submits_slurm")),
        "heavy_gpu_ready": sum(1 for item in ready if item.get("heavy_gpu")),
    }


def build_payload(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    items = actions(plan)
    ready_items = [item for item in items if item.get("status") == "ready_for_operator" and item.get("command")]
    waiting_items = [item for item in items if item not in ready_items]
    ordered_ready = sort_actions(ready_items)
    blocked = sort_actions(waiting_items)[: max(args.blocked_limit, 0)]

    records = [command_record(args, action, idx) for idx, action in enumerate(ordered_ready, 1)]
    blockers = plan.get("blockers") or []
    if not isinstance(blockers, list):
        blockers = []

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": operator_status(ordered_ready, blocked, plan),
        "artifact_root": str(args.artifact_root),
        "plan_json": str(args.plan_json),
        "plan_overall_status": plan.get("overall_status"),
        "counts": action_counts(ordered_ready, blocked),
        "current_ready_actions": [item["id"] for item in records],
        "execution_policy": {
            "default_mode": "print_only",
            "no_submit": True,
            "slurm_requires": ["--execute", "--allow-slurm"],
            "heavy_gpu_requires": ["--execute", "--allow-slurm", "--allow-heavy-gpu"],
            "preferred_order": "Execute ready actions in listed order. For Slurm actions, run after_commands only after the job reaches a terminal state.",
        },
        "ready_actions": records,
        "waiting_actions": [
            {
                "id": action.get("id"),
                "title": action.get("title"),
                "stage": action.get("stage"),
                "status": action.get("status"),
                "reason": action.get("reason"),
                "submits_slurm": bool(action.get("submits_slurm")),
                "heavy_gpu": bool(action.get("heavy_gpu")),
                "report": action.get("report"),
            }
            for action in blocked
        ],
        "blockers": [
            {
                "id": item.get("id"),
                "severity": item.get("severity"),
                "summary": item.get("summary"),
                "report": item.get("report"),
            }
            for item in blockers
            if isinstance(item, dict)
        ],
        "refresh_and_validation_commands": refresh_commands(args),
    }


def md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "/").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Sheet",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Overall: **{str(payload.get('overall_status') or 'unknown').upper()}**",
        f"Plan status: **{str(payload.get('plan_overall_status') or 'unknown').upper()}**",
        f"Plan JSON: `{payload['plan_json']}`",
        f"Artifact root: `{payload['artifact_root']}`",
        "",
        "This sheet is no-submit. The commands below are reviewable wrappers around `run_eagle3_next_action.py`; Slurm and heavy-GPU actions require explicit allow flags.",
        "",
        "## Ready Actions",
        "",
    ]
    ready = payload.get("ready_actions") or []
    if not ready:
        lines += [
            "No ready operator action with a command is present. Inspect blockers and regenerate the planner report.",
            "",
        ]
    else:
        lines += ["| order | action | status | Slurm | heavy GPU | reason |", "| ---: | --- | --- | --- | --- | --- |"]
        for item in ready:
            lines.append(
                f"| {item['order']} | `{md_escape(item['id'])}` | {md_escape(item['status'])} | "
                f"{str(item['submits_slurm']).lower()} | {str(item['heavy_gpu']).lower()} | "
                f"{md_escape(item.get('reason'))} |"
            )
        lines.append("")
        for item in ready:
            lines += [
                f"### {item['order']}. {md_escape(item.get('title'))}",
                "",
                f"Stage: `{md_escape(item.get('stage'))}`",
                f"Report: `{md_escape(item.get('report'))}`",
                f"Execution record: `{md_escape(item.get('execution_record'))}`",
                "",
                "Review only:",
                "",
                "```bash",
                str(item["print_command"]),
                "```",
                "",
                "Execute selected action only after review:",
                "",
                "```bash",
                str(item["execute_command"]),
                "```",
                "",
                f"Follow-up timing: {md_escape(item.get('after_policy_detail'))}",
                "",
                "Raw command from planner:",
                "",
                "```bash",
                str(item.get("raw_command") or ""),
                "```",
                "",
            ]
            if item.get("execute_with_after_commands"):
                lines += [
                    "Execute and immediately run recorded follow-up commands:",
                    "",
                    "```bash",
                    str(item["execute_with_after_commands"]),
                    "```",
                    "",
                ]
            if item.get("followup_status_command"):
                lines += [
                    "Check submitted Slurm job state before follow-up analyzers:",
                    "",
                    "```bash",
                    str(item["followup_status_command"]),
                    "```",
                    "",
                    "Run follow-up analyzers only after the guard reports terminal jobs:",
                    "",
                    "```bash",
                    str(item["execute_followup_command"]),
                    "```",
                    "",
                ]
            after = item.get("after_commands") or []
            if after:
                lines += ["Follow-up commands:", ""]
                for command in after:
                    lines += ["```bash", str(command), "```", ""]

    lines += ["## Waiting Or Blocked Actions", ""]
    waiting = payload.get("waiting_actions") or []
    if waiting:
        lines += ["| action | status | Slurm | heavy GPU | reason |", "| --- | --- | --- | --- | --- |"]
        for item in waiting:
            lines.append(
                f"| `{md_escape(item.get('id'))}` | {md_escape(item.get('status'))} | "
                f"{str(item.get('submits_slurm')).lower()} | {str(item.get('heavy_gpu')).lower()} | "
                f"{md_escape(item.get('reason'))} |"
            )
        lines.append("")
    else:
        lines += ["No waiting actions were included.", ""]

    blockers = payload.get("blockers") or []
    lines += ["## Blockers And Warnings", ""]
    if blockers:
        lines += ["| severity | id | summary | report |", "| --- | --- | --- | --- |"]
        for item in blockers:
            lines.append(
                f"| {md_escape(item.get('severity'))} | `{md_escape(item.get('id'))}` | "
                f"{md_escape(item.get('summary'))} | `{md_escape(item.get('report'))}` |"
            )
        lines.append("")
    else:
        lines += ["No blockers are present in the planner report.", ""]

    lines += [
        "## Refresh And Validate",
        "",
        "Run these after action/analyzer results change, then reopen this sheet.",
        "",
    ]
    for command in payload.get("refresh_and_validation_commands") or []:
        lines += ["```bash", str(command), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan_json)
    payload = build_payload(plan, args)
    markdown = render_markdown(payload)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
