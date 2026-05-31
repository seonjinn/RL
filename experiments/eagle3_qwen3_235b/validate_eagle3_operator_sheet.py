#!/usr/bin/env python3
"""Validate the no-submit operator sheet for Qwen3 Eagle3 gates.

This validator checks the command sheet produced by
create_eagle3_operator_sheet.py. It does not submit jobs. Its purpose is to
catch handoff mistakes before an operator copies an execution command: missing
allow flags, missing execution-record paths, stale action ids, unsafe
print-only commands, or refresh commands that no longer regenerate the proof
reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")

REQUIRED_REFRESH_SNIPPETS = [
    "plan_eagle3_next_actions.py",
    "validate_eagle3_next_action_plan.py",
    "validate_eagle3_next_action_transitions.py",
    "validate_eagle3_operator_queue_transitions.py",
    "validate_eagle3_completion_contract.py",
    "probe_eagle3_slurm_capacity.py",
    "validate_eagle3_resource_profile_application.py",
    "create_eagle3_operator_sheet.py",
    "validate_eagle3_operator_sheet.py",
    "validate_eagle3_operator_execution.py",
    "validate_eagle3_operator_followups.py",
    "create_eagle3_operator_submit_packet.py",
    "validate_eagle3_operator_submit_packet.py",
    "preflight_eagle3_operator_ready_submit.py",
    "summarize_eagle3_operator_queue.py",
    "refresh_eagle3_operator_state.py",
    "audit_eagle3_goal_evidence.py",
]


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--expect-ready-action",
        action="append",
        default=[],
        help="Require this ready action id to be present in the sheet. Repeatable.",
    )
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    defaults = {
        "plan_json": Path(os.environ.get("NEXT_ACTION_PLAN_JSON", root / "reports/eagle3_next_actions.json")),
        "operator_sheet_json": Path(os.environ.get("OPERATOR_SHEET_JSON", root / "reports/eagle3_operator_sheet.json")),
        "json_out": Path(os.environ.get("OPERATOR_SHEET_VALIDATION_JSON", root / "reports/eagle3_operator_sheet_validation.json")),
        "markdown_out": Path(
            os.environ.get("OPERATOR_SHEET_VALIDATION_MARKDOWN", root / "reports/eagle3_operator_sheet_validation.md")
        ),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


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


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def action_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in as_list(plan.get("next_actions")):
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def ready_plan_actions(plan: dict[str, Any]) -> set[str]:
    return set(ready_plan_action_order(plan))


def ready_plan_action_order(plan: dict[str, Any]) -> list[str]:
    return [
        str(item["id"])
        for item in as_list(plan.get("next_actions"))
        if isinstance(item, dict) and item.get("id") and item.get("status") == "ready_for_operator" and item.get("command")
    ]


def split_command(command: Any) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []


def command_has(command: str, *snippets: str) -> bool:
    return all(snippet in command for snippet in snippets)


def validate_execution_command(
    checks: list[dict[str, Any]],
    item: dict[str, Any],
    command_key: str,
    require_run_after: bool,
) -> None:
    action_id = str(item.get("id") or "")
    command = str(item.get(command_key) or "")
    tokens = split_command(command)
    label = f"{action_id} {command_key}"
    if not command:
        add(checks, "command", label, "fail", "command is missing")
        return
    if not tokens:
        add(checks, "command", label, "fail", "command cannot be shell-split")
        return
    required = [
        "run_eagle3_next_action.py",
        "--artifact-root",
        "--plan-json",
        "--action-id",
        action_id,
        "--execute",
        "--json-out",
        str(item.get("execution_record") or ""),
    ]
    missing = [snippet for snippet in required if snippet and snippet not in command]
    if item.get("submits_slurm") and "--allow-slurm" not in tokens:
        missing.append("--allow-slurm")
    if item.get("heavy_gpu") and "--allow-heavy-gpu" not in tokens:
        missing.append("--allow-heavy-gpu")
    if require_run_after and "--run-after" not in tokens:
        missing.append("--run-after")
    if not require_run_after and "--run-after" in tokens:
        add(checks, "command", label, "warn", "plain execute command unexpectedly contains --run-after")
        return
    if missing:
        add(checks, "command", label, "fail", "execution command is missing required snippets", missing=missing)
    else:
        add(checks, "command", label, "pass", "execution command has required runner flags")


def validate_followup_command(
    checks: list[dict[str, Any]],
    item: dict[str, Any],
    command_key: str,
    require_execute_after: bool,
) -> None:
    action_id = str(item.get("id") or "")
    command = str(item.get(command_key) or "")
    tokens = split_command(command)
    label = f"{action_id} {command_key}"
    if not command:
        add(checks, "followup", label, "fail", "follow-up guard command is missing")
        return
    if not tokens:
        add(checks, "followup", label, "fail", "follow-up guard command cannot be shell-split")
        return
    required = [
        "run_eagle3_slurm_followups.py",
        "--artifact-root",
        "--plan-json",
        "--operator-sheet-json",
        "--action-id",
        action_id,
        "--execution-record",
        str(item.get("execution_record") or ""),
        "--json-out",
        str(item.get("followup_record") or ""),
        "--markdown-out",
    ]
    missing = [snippet for snippet in required if snippet and snippet not in command]
    if require_execute_after and "--execute-after" not in tokens:
        missing.append("--execute-after")
    if not require_execute_after and "--execute-after" in tokens:
        add(checks, "followup", label, "fail", "inspect command must not contain --execute-after")
        return
    if missing:
        add(checks, "followup", label, "fail", "follow-up guard command is missing required snippets", missing=missing)
    else:
        add(checks, "followup", label, "pass", "follow-up guard command has required runner flags")


def validate_ready_action(
    checks: list[dict[str, Any]],
    item: dict[str, Any],
    order: int,
    plan_actions: dict[str, dict[str, Any]],
    artifact_root: Path,
) -> None:
    action_id = str(item.get("id") or "")
    if not action_id:
        add(checks, "action", f"ready[{order}]", "fail", "ready action is missing an id")
        return

    plan_item = plan_actions.get(action_id)
    if not plan_item:
        add(checks, "action", action_id, "fail", "ready action is not present in the next-action plan")
    elif plan_item.get("status") != item.get("status"):
        add(
            checks,
            "action",
            action_id,
            "fail",
            "ready action status does not match the next-action plan",
            sheet_status=item.get("status"),
            plan_status=plan_item.get("status"),
        )
    else:
        add(checks, "action", action_id, "pass", "ready action is linked to the current next-action plan")

    if item.get("order") != order:
        add(checks, "action", action_id, "warn", "ready action order field does not match list order", order=item.get("order"))
    else:
        add(checks, "action", action_id, "pass", "ready action order is stable")

    record = item.get("execution_record")
    if not record:
        add(checks, "record", action_id, "fail", "execution_record path is missing")
    else:
        record_path = Path(str(record))
        if artifact_root not in record_path.parents:
            add(checks, "record", action_id, "warn", "execution_record is outside artifact_root", record=str(record))
        elif "operator_execution" not in record_path.parts:
            add(checks, "record", action_id, "fail", "execution_record is not under reports/operator_execution", record=str(record))
        elif not record_path.name.startswith(f"{order:02d}_"):
            add(checks, "record", action_id, "warn", "execution_record filename does not preserve ready-action order", record=str(record))
        else:
            add(checks, "record", action_id, "pass", "execution_record path is ordered under reports/operator_execution")

    print_command = str(item.get("print_command") or "")
    print_tokens = split_command(print_command)
    if not print_command:
        add(checks, "command", f"{action_id} print", "fail", "print command is missing")
    elif not print_tokens:
        add(checks, "command", f"{action_id} print", "fail", "print command cannot be shell-split")
    elif "--execute" in print_tokens or "--allow-slurm" in print_tokens or "--allow-heavy-gpu" in print_tokens:
        add(checks, "command", f"{action_id} print", "fail", "print command contains execution allow flags")
    elif not command_has(print_command, "run_eagle3_next_action.py", "--artifact-root", "--plan-json", "--action-id", action_id):
        add(checks, "command", f"{action_id} print", "fail", "print command does not target the expected runner/action")
    else:
        add(checks, "command", f"{action_id} print", "pass", "print command is no-submit and targets the expected action")

    validate_execution_command(checks, item, "execute_command", require_run_after=False)
    after_policy = item.get("after_policy")
    if item.get("submits_slurm"):
        if item.get("execute_with_after_commands"):
            add(
                checks,
                "command",
                f"{action_id} execute_with_after_commands",
                "fail",
                "Slurm actions must not advertise immediate --run-after execution",
            )
        elif after_policy == "after_slurm_terminal_state":
            add(checks, "policy", f"{action_id} after_policy", "pass", "Slurm follow-up policy waits for terminal job state")
            validate_followup_command(checks, item, "followup_status_command", require_execute_after=False)
            validate_followup_command(checks, item, "execute_followup_command", require_execute_after=True)
        else:
            add(
                checks,
                "policy",
                f"{action_id} after_policy",
                "fail",
                "Slurm action is missing after_slurm_terminal_state policy",
                after_policy=after_policy,
            )
    else:
        validate_execution_command(checks, item, "execute_with_after_commands", require_run_after=True)

    raw_command = str(item.get("raw_command") or "")
    if item.get("status") == "ready_for_operator" and not raw_command:
        add(checks, "command", f"{action_id} raw", "fail", "raw planner command is missing")
    elif action_id == "submit_eagle3_pilot_pipeline":
        required = [
            "submit_eagle3_pipeline_if_ready.py",
            "--preflight-json",
            "eagle3_pipeline_submit_preflight.json",
            "--json-out",
            "eagle3_pipeline_gated_submit.json",
            "--execute",
            "--allow-heavy-gpu",
        ]
        missing = [snippet for snippet in required if snippet not in raw_command]
        if missing:
            add(
                checks,
                "command",
                f"{action_id} raw",
                "fail",
                "pipeline raw command must use the gated submit helper",
                missing=missing,
            )
        else:
            add(checks, "command", f"{action_id} raw", "pass", "pipeline raw command uses the gated submit helper")
    elif item.get("submits_slurm") and "SUBMIT=true" not in raw_command and "DRY_RUN=false" not in raw_command:
        add(checks, "command", f"{action_id} raw", "warn", "Slurm action raw command lacks expected submit marker")
    else:
        add(checks, "command", f"{action_id} raw", "pass", "raw planner command is present")

    after = as_list(item.get("after_commands"))
    if item.get("submits_slurm") and not after:
        add(checks, "after", action_id, "fail", "Slurm action has no follow-up analyzer commands")
    elif after and not any("plan_eagle3_next_actions.py" in str(command) for command in after):
        add(checks, "after", action_id, "fail", "after_commands do not refresh next-action plan")
    else:
        add(checks, "after", action_id, "pass", "after_commands are present or not required", count=len(after))


def validate_refresh_commands(checks: list[dict[str, Any]], sheet: dict[str, Any]) -> None:
    commands = [str(command) for command in as_list(sheet.get("refresh_and_validation_commands"))]
    text = "\n".join(commands)
    if not commands:
        add(checks, "refresh", "commands", "fail", "refresh_and_validation_commands is empty")
        return
    missing = [snippet for snippet in REQUIRED_REFRESH_SNIPPETS if snippet not in text]
    if missing:
        add(checks, "refresh", "commands", "fail", "refresh commands are missing required validators", missing=missing)
    else:
        add(checks, "refresh", "commands", "pass", "refresh commands regenerate planner, sheet, execution, and evidence reports")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    plan_raw, plan_error = load_json(args.plan_json)
    sheet_raw, sheet_error = load_json(args.operator_sheet_json)
    plan = as_dict(plan_raw)
    sheet = as_dict(sheet_raw)
    plan_actions = action_map(plan)

    checks: list[dict[str, Any]] = []
    if plan_error:
        add(checks, "input", "next-action plan", "fail", plan_error, path=str(args.plan_json))
    else:
        add(checks, "input", "next-action plan", "pass", "next-action plan is readable", actions=len(plan_actions))
    if sheet_error:
        add(checks, "input", "operator sheet", "fail", sheet_error, path=str(args.operator_sheet_json))
    else:
        add(checks, "input", "operator sheet", "pass", "operator sheet is readable")

    ready = [item for item in as_list(sheet.get("ready_actions")) if isinstance(item, dict)]
    sheet_ready_ids = [str(item.get("id")) for item in ready if item.get("id")]
    plan_ready_order = ready_plan_action_order(plan)
    plan_ready_ids = set(plan_ready_order)
    current_ready = [str(item) for item in as_list(sheet.get("current_ready_actions"))]

    if sheet and sheet.get("artifact_root") != str(args.artifact_root):
        add(
            checks,
            "schema",
            "artifact_root",
            "warn",
            "sheet artifact_root differs from CLI artifact_root",
            sheet_artifact_root=sheet.get("artifact_root"),
            cli_artifact_root=str(args.artifact_root),
        )
    else:
        add(checks, "schema", "artifact_root", "pass", "sheet artifact_root matches CLI artifact_root")

    if current_ready != sheet_ready_ids:
        add(
            checks,
            "schema",
            "current_ready_actions",
            "fail",
            "current_ready_actions does not match ready_actions order",
            current_ready_actions=current_ready,
            ready_actions=sheet_ready_ids,
        )
    else:
        add(checks, "schema", "current_ready_actions", "pass", "current_ready_actions matches ready_actions")

    unexpected_ready = sorted(set(sheet_ready_ids) - plan_ready_ids)
    missing_ready = sorted(plan_ready_ids - set(sheet_ready_ids))
    if unexpected_ready or missing_ready:
        add(
            checks,
            "schema",
            "ready action set",
            "fail",
            "operator sheet ready action set differs from next-action plan",
            unexpected_ready=unexpected_ready,
            missing_ready=missing_ready,
        )
    else:
        add(checks, "schema", "ready action set", "pass", "operator sheet ready actions match next-action plan")

    if sheet_ready_ids != plan_ready_order:
        add(
            checks,
            "schema",
            "ready action order",
            "fail",
            "operator sheet ready action order differs from next-action plan",
            sheet_ready_actions=sheet_ready_ids,
            plan_ready_actions=plan_ready_order,
        )
    else:
        add(checks, "schema", "ready action order", "pass", "operator sheet preserves next-action plan order")

    duplicates = sorted({item for item in sheet_ready_ids if sheet_ready_ids.count(item) > 1})
    if duplicates:
        add(checks, "schema", "unique ready actions", "fail", "duplicate ready actions found", duplicates=duplicates)
    else:
        add(checks, "schema", "unique ready actions", "pass", "ready action ids are unique")

    for action_id in args.expect_ready_action:
        if action_id in sheet_ready_ids:
            add(checks, "expectation", action_id, "pass", "expected ready action is present in the operator sheet")
        else:
            add(checks, "expectation", action_id, "fail", "expected ready action is missing from the operator sheet")

    policy = as_dict(sheet.get("execution_policy"))
    heavy_requires = set(as_list(policy.get("heavy_gpu_requires")))
    slurm_requires = set(as_list(policy.get("slurm_requires")))
    preferred_order = str(policy.get("preferred_order") or "")
    if (
        "--execute" in slurm_requires
        and "--allow-slurm" in slurm_requires
        and "--allow-heavy-gpu" in heavy_requires
        and "terminal state" in preferred_order
    ):
        add(checks, "policy", "allow flags", "pass", "execution policy records Slurm and heavy-GPU allow flags")
    else:
        add(checks, "policy", "allow flags", "fail", "execution policy is missing required allow flags", policy=policy)

    seen_records: set[str] = set()
    for order, item in enumerate(ready, 1):
        validate_ready_action(checks, item, order, plan_actions, args.artifact_root)
        record = str(item.get("execution_record") or "")
        if record:
            if record in seen_records:
                add(checks, "record", item.get("id") or f"ready[{order}]", "fail", "duplicate execution_record path", record=record)
            seen_records.add(record)

    validate_refresh_commands(checks, sheet)

    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1

    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif any(check["status"] == "warn" for check in checks):
        status = "warn"
    else:
        status = "pass"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "artifact_root": str(args.artifact_root),
        "plan_json": str(args.plan_json),
        "operator_sheet_json": str(args.operator_sheet_json),
        "ready_actions": sheet_ready_ids,
        "counts": counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Sheet Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Plan JSON: `{payload['plan_json']}`",
        f"Operator sheet JSON: `{payload['operator_sheet_json']}`",
        f"Ready actions: `{', '.join(payload['ready_actions']) or '-'}`",
        "",
        "This report is no-submit. It validates the operator handoff commands before any Slurm execution.",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['area']} | {check['name']} | {check['status'].upper()} | {detail} |")
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
    if payload["overall_status"] == "fail":
        return 1
    if args.fail_on_warn and payload["overall_status"] == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
