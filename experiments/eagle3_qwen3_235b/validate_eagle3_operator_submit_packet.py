#!/usr/bin/env python3
"""Validate the no-submit Eagle3 operator submit packet.

The submit packet is intentionally a compact view of the operator sheet. This
validator checks that the packet is current, preserves Slurm safety guards, and
does not collapse terminal-state follow-up handling into immediate execution.
It does not submit jobs or run analyzers.
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

ACCEPTED_OPERATOR_EXECUTION_STATUSES = {"not_started", "incomplete", "warn", "pass"}
READY_PACKET_STATUS = "ready_for_operator_submit"


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--operator-submit-packet-json", type=Path)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--operator-sheet-validation-json", type=Path)
    parser.add_argument("--operator-followup-validation-json", type=Path)
    parser.add_argument("--operator-execution-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--expect-ready-action",
        action="append",
        default=[],
        help="Require this ready action id to be present in the packet. Repeatable.",
    )
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    defaults = {
        "operator_submit_packet_json": Path(
            os.environ.get("OPERATOR_SUBMIT_PACKET_JSON", root / "reports/eagle3_operator_submit_packet.json")
        ),
        "operator_sheet_json": Path(os.environ.get("OPERATOR_SHEET_JSON", root / "reports/eagle3_operator_sheet.json")),
        "operator_sheet_validation_json": Path(
            os.environ.get("OPERATOR_SHEET_VALIDATION_JSON", root / "reports/eagle3_operator_sheet_validation.json")
        ),
        "operator_followup_validation_json": Path(
            os.environ.get(
                "OPERATOR_FOLLOWUP_VALIDATION_JSON",
                root / "reports/eagle3_operator_followups_validation.json",
            )
        ),
        "operator_execution_json": Path(
            os.environ.get("OPERATOR_EXECUTION_JSON", root / "reports/eagle3_operator_execution.json")
        ),
        "json_out": Path(
            os.environ.get(
                "OPERATOR_SUBMIT_PACKET_VALIDATION_JSON",
                root / "reports/eagle3_operator_submit_packet_validation.json",
            )
        ),
        "markdown_out": Path(
            os.environ.get(
                "OPERATOR_SUBMIT_PACKET_VALIDATION_MARKDOWN",
                root / "reports/eagle3_operator_submit_packet_validation.md",
            )
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


def report_status(payload: dict[str, Any], error: str | None) -> str:
    if error:
        return "missing"
    return str(payload.get("overall_status") or payload.get("status") or "unknown")


def packet_ready_actions(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in as_list(packet.get("ready_actions")) if isinstance(item, dict)]


def sheet_ready_actions(sheet: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in as_list(sheet.get("ready_actions")) if isinstance(item, dict)]


def by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        action_id = str(item.get("id") or "")
        if action_id:
            result[action_id] = item
    return result


def split_command(command: Any) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []


def command_has_all(command: str, snippets: list[Any]) -> list[str]:
    return [str(snippet) for snippet in snippets if snippet is not None and str(snippet) not in command]


def latest_execution_map(execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latest = execution.get("latest_by_action")
    if not isinstance(latest, dict):
        return {}
    return {str(key): value for key, value in latest.items() if isinstance(value, dict)}


def validate_report_gates(
    checks: list[dict[str, Any]],
    packet: dict[str, Any],
    sheet: dict[str, Any],
    sheet_error: str | None,
    sheet_validation: dict[str, Any],
    sheet_validation_error: str | None,
    followup_validation: dict[str, Any],
    followup_validation_error: str | None,
    execution: dict[str, Any],
    execution_error: str | None,
) -> None:
    packet_statuses = as_dict(packet.get("report_statuses"))
    actual_statuses = {
        "operator_sheet": report_status(sheet, sheet_error),
        "operator_sheet_validation": report_status(sheet_validation, sheet_validation_error),
        "operator_followup_validation": report_status(followup_validation, followup_validation_error),
        "operator_execution": report_status(execution, execution_error),
    }
    for key, actual in actual_statuses.items():
        recorded = str(packet_statuses.get(key) or "missing")
        if recorded != actual:
            add(
                checks,
                "reports",
                key,
                "fail",
                "packet report status does not match the current source report",
                packet_status=recorded,
                actual_status=actual,
            )
        else:
            add(checks, "reports", key, "pass", "packet report status matches source report", observed_status=actual)

    if actual_statuses["operator_sheet_validation"] == "pass":
        add(checks, "gates", "operator sheet validation", "pass", "operator sheet validation is PASS")
    else:
        add(
            checks,
            "gates",
            "operator sheet validation",
            "fail",
            "operator sheet validation must be PASS before submit packet use",
            observed_status=actual_statuses["operator_sheet_validation"],
        )

    if actual_statuses["operator_followup_validation"] == "pass":
        add(checks, "gates", "operator follow-up validation", "pass", "operator follow-up validation is PASS")
    else:
        add(
            checks,
            "gates",
            "operator follow-up validation",
            "fail",
            "operator follow-up validation must be PASS before submit packet use",
            observed_status=actual_statuses["operator_followup_validation"],
        )

    execution_status = actual_statuses["operator_execution"]
    if execution_status in ACCEPTED_OPERATOR_EXECUTION_STATUSES:
        add(
            checks,
            "gates",
            "operator execution",
            "pass",
            "operator execution status is safe for a no-submit packet",
            observed_status=execution_status,
        )
    else:
        add(
            checks,
            "gates",
            "operator execution",
            "fail",
            "operator execution status is not safe for a submit packet",
            observed_status=execution_status,
        )


def validate_packet_schema(checks: list[dict[str, Any]], packet: dict[str, Any], artifact_root: Path) -> None:
    required = [
        "generated_at",
        "overall_status",
        "artifact_root",
        "policy",
        "report_statuses",
        "report_paths",
        "counts",
        "ready_actions",
    ]
    missing = [key for key in required if key not in packet]
    if missing:
        add(checks, "schema", "required keys", "fail", "packet is missing required keys", missing=missing)
    else:
        add(checks, "schema", "required keys", "pass", "packet has required keys")

    if packet.get("artifact_root") != str(artifact_root):
        add(
            checks,
            "schema",
            "artifact_root",
            "warn",
            "packet artifact_root differs from CLI artifact_root",
            packet_artifact_root=packet.get("artifact_root"),
            cli_artifact_root=str(artifact_root),
        )
    else:
        add(checks, "schema", "artifact_root", "pass", "packet artifact_root matches CLI artifact_root")

    blockers = as_list(packet.get("blockers"))
    status = str(packet.get("overall_status") or "unknown")
    ready = packet_ready_actions(packet)
    if status == READY_PACKET_STATUS and blockers:
        add(checks, "schema", "blockers", "fail", "ready packet still records blockers", blockers=blockers)
    elif status == READY_PACKET_STATUS and not ready:
        add(checks, "schema", "ready_actions", "fail", "ready packet has no ready actions")
    elif status == READY_PACKET_STATUS:
        add(checks, "schema", "overall_status", "pass", "packet status is ready for operator submit")
    else:
        add(checks, "schema", "overall_status", "fail", "packet is not ready for operator submit", observed_status=status)

    policy = as_dict(packet.get("policy"))
    if (
        policy.get("mode") == "no_submit_packet"
        and policy.get("run_execute_command_only_after_review") is True
        and policy.get("slurm_followup_requires_terminal_state") is True
        and policy.get("do_not_use_run_after_for_slurm") is True
    ):
        add(checks, "policy", "no-submit", "pass", "packet records the no-submit and Slurm terminal-state policy")
    else:
        add(checks, "policy", "no-submit", "fail", "packet policy is missing required safety flags", policy=policy)


def validate_command_shape(checks: list[dict[str, Any]], action: dict[str, Any]) -> None:
    action_id = str(action.get("id") or "")
    execute_command = str(action.get("execute_command") or "")
    execute_tokens = split_command(execute_command)
    if not execute_command:
        add(checks, "command", f"{action_id} execute", "fail", "execute command is missing")
        return
    if not execute_tokens:
        add(checks, "command", f"{action_id} execute", "fail", "execute command cannot be shell-split")
        return

    missing = command_has_all(
        execute_command,
        [
            "run_eagle3_next_action.py",
            "--artifact-root",
            "--plan-json",
            "--action-id",
            action_id,
            "--execute",
            "--json-out",
            action.get("execution_record"),
        ],
    )
    if action.get("submits_slurm") and "--allow-slurm" not in execute_tokens:
        missing.append("--allow-slurm")
    if action.get("heavy_gpu") and "--allow-heavy-gpu" not in execute_tokens:
        missing.append("--allow-heavy-gpu")
    if "--run-after" in execute_tokens:
        add(checks, "command", f"{action_id} execute", "fail", "execute command must not include --run-after")
    elif missing:
        add(checks, "command", f"{action_id} execute", "fail", "execute command is missing required snippets", missing=missing)
    else:
        add(checks, "command", f"{action_id} execute", "pass", "execute command has required no-immediate-followup flags")

    status_command = str(action.get("post_submit_status_command") or "")
    followup_command = str(action.get("execute_followup_after_terminal_command") or "")
    if action.get("submits_slurm"):
        if action.get("after_policy") != "after_slurm_terminal_state":
            add(checks, "policy", f"{action_id} after_policy", "fail", "Slurm action must wait for terminal state")
        else:
            add(checks, "policy", f"{action_id} after_policy", "pass", "Slurm action waits for terminal state")

        validate_followup_command(checks, action, "post_submit_status_command", require_execute_after=False)
        validate_followup_command(checks, action, "execute_followup_after_terminal_command", require_execute_after=True)
    elif status_command or followup_command:
        add(checks, "followup", action_id, "warn", "non-Slurm action unexpectedly has Slurm follow-up commands")
    else:
        add(checks, "followup", action_id, "pass", "non-Slurm action has no Slurm follow-up commands")


def validate_followup_command(
    checks: list[dict[str, Any]],
    action: dict[str, Any],
    key: str,
    require_execute_after: bool,
) -> None:
    action_id = str(action.get("id") or "")
    command = str(action.get(key) or "")
    tokens = split_command(command)
    label = f"{action_id} {key}"
    if not command:
        add(checks, "followup", label, "fail", "Slurm follow-up command is missing")
        return
    if not tokens:
        add(checks, "followup", label, "fail", "Slurm follow-up command cannot be shell-split")
        return
    required = [
        "run_eagle3_slurm_followups.py",
        "--artifact-root",
        "--plan-json",
        "--operator-sheet-json",
        "--action-id",
        action_id,
        "--execution-record",
        action.get("execution_record"),
        "--json-out",
        action.get("followup_record"),
        "--markdown-out",
    ]
    missing = command_has_all(command, required)
    has_execute_after = "--execute-after" in tokens
    if require_execute_after and not has_execute_after:
        missing.append("--execute-after")
    if not require_execute_after and has_execute_after:
        add(checks, "followup", label, "fail", "status guard command must not contain --execute-after")
        return
    if missing:
        add(checks, "followup", label, "fail", "follow-up command is missing required snippets", missing=missing)
    else:
        add(checks, "followup", label, "pass", "follow-up command has required guard flags")


def validate_ready_actions(
    checks: list[dict[str, Any]],
    packet: dict[str, Any],
    sheet: dict[str, Any],
    execution: dict[str, Any],
    expect_ready_actions: list[str],
) -> None:
    packet_actions = packet_ready_actions(packet)
    sheet_actions = sheet_ready_actions(sheet)
    packet_ids = [str(item.get("id") or "") for item in packet_actions if item.get("id")]
    sheet_ids = [str(item.get("id") or "") for item in sheet_actions if item.get("id")]
    sheet_map = by_id(sheet_actions)

    if packet_ids != sheet_ids:
        add(
            checks,
            "actions",
            "ready action order",
            "fail",
            "packet ready action list does not match operator sheet order",
            packet_ready_actions=packet_ids,
            sheet_ready_actions=sheet_ids,
        )
    else:
        add(checks, "actions", "ready action order", "pass", "packet ready actions match operator sheet order")

    duplicates = sorted({item for item in packet_ids if packet_ids.count(item) > 1})
    if duplicates:
        add(checks, "actions", "unique ready actions", "fail", "duplicate packet ready actions found", duplicates=duplicates)
    else:
        add(checks, "actions", "unique ready actions", "pass", "packet ready action ids are unique")

    for action_id in expect_ready_actions:
        if action_id in packet_ids:
            add(checks, "expectation", action_id, "pass", "expected ready action is present in the packet")
        else:
            add(checks, "expectation", action_id, "fail", "expected ready action is missing from the packet")

    latest = latest_execution_map(execution)
    for order, action in enumerate(packet_actions, 1):
        action_id = str(action.get("id") or "")
        sheet_action = sheet_map.get(action_id)
        if not sheet_action:
            continue

        field_pairs = {
            "order": "order",
            "title": "title",
            "status": "status",
            "submits_slurm": "submits_slurm",
            "heavy_gpu": "heavy_gpu",
            "after_policy": "after_policy",
            "execute_command": "execute_command",
            "planner_command": "raw_command",
            "post_submit_status_command": "followup_status_command",
            "execute_followup_after_terminal_command": "execute_followup_command",
            "execution_record": "execution_record",
            "followup_record": "followup_record",
        }
        mismatched = [
            packet_key
            for packet_key, sheet_key in field_pairs.items()
            if action.get(packet_key) != sheet_action.get(sheet_key)
        ]
        if mismatched:
            add(checks, "actions", action_id, "fail", "packet action differs from operator sheet", mismatched_fields=mismatched)
        elif action.get("order") != order:
            add(checks, "actions", action_id, "warn", "packet action order field differs from list order", order=action.get("order"))
        else:
            add(checks, "actions", action_id, "pass", "packet action mirrors operator sheet")

        already_executed = bool(action.get("already_executed"))
        latest_execution = action.get("latest_execution")
        if already_executed and not isinstance(latest_execution, dict):
            add(checks, "execution", action_id, "fail", "already_executed action lacks latest_execution")
        elif not already_executed and latest_execution:
            add(checks, "execution", action_id, "fail", "not-executed action unexpectedly has latest_execution")
        elif already_executed and action_id not in latest:
            add(checks, "execution", action_id, "fail", "already_executed does not match operator execution report")
        elif not already_executed and action_id in latest:
            add(checks, "execution", action_id, "fail", "packet missed an operator execution record for this action")
        else:
            add(checks, "execution", action_id, "pass", "packet execution marker matches operator execution report")

        planner_command = str(action.get("planner_command") or "")
        if action_id == "submit_eagle3_pilot_pipeline":
            required = [
                "submit_eagle3_pipeline_if_ready.py",
                "--preflight-json",
                "eagle3_pipeline_submit_preflight.json",
                "--json-out",
                "eagle3_pipeline_gated_submit.json",
                "--execute",
                "--allow-heavy-gpu",
            ]
            missing = command_has_all(planner_command, required)
            if missing:
                add(
                    checks,
                    "command",
                    f"{action_id} planner",
                    "fail",
                    "packet planner command must use the gated pipeline submit helper",
                    missing=missing,
                )
            else:
                add(
                    checks,
                    "command",
                    f"{action_id} planner",
                    "pass",
                    "packet planner command uses the gated pipeline submit helper",
                )

        validate_command_shape(checks, action)


def validate_counts(checks: list[dict[str, Any]], packet: dict[str, Any]) -> None:
    actions = packet_ready_actions(packet)
    counts = as_dict(packet.get("counts"))
    expected = {
        "ready_actions": len(actions),
        "slurm_ready_actions": sum(1 for item in actions if item.get("submits_slurm")),
        "heavy_gpu_ready_actions": sum(1 for item in actions if item.get("heavy_gpu")),
        "already_executed_actions": sum(1 for item in actions if item.get("already_executed")),
    }
    mismatched = {key: {"packet": counts.get(key), "expected": value} for key, value in expected.items() if counts.get(key) != value}
    if mismatched:
        add(checks, "counts", "ready actions", "fail", "packet counts do not match ready_actions", mismatched=mismatched)
    else:
        add(checks, "counts", "ready actions", "pass", "packet counts match ready_actions")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    packet_raw, packet_error = load_json(args.operator_submit_packet_json)
    sheet_raw, sheet_error = load_json(args.operator_sheet_json)
    sheet_validation_raw, sheet_validation_error = load_json(args.operator_sheet_validation_json)
    followup_validation_raw, followup_validation_error = load_json(args.operator_followup_validation_json)
    execution_raw, execution_error = load_json(args.operator_execution_json)

    packet = as_dict(packet_raw)
    sheet = as_dict(sheet_raw)
    sheet_validation = as_dict(sheet_validation_raw)
    followup_validation = as_dict(followup_validation_raw)
    execution = as_dict(execution_raw)

    checks: list[dict[str, Any]] = []
    if packet_error:
        add(checks, "input", "operator submit packet", "fail", packet_error, path=str(args.operator_submit_packet_json))
    else:
        add(checks, "input", "operator submit packet", "pass", "operator submit packet is readable")
    if sheet_error:
        add(checks, "input", "operator sheet", "fail", sheet_error, path=str(args.operator_sheet_json))
    else:
        add(checks, "input", "operator sheet", "pass", "operator sheet is readable")
    if sheet_validation_error:
        add(checks, "input", "operator sheet validation", "fail", sheet_validation_error, path=str(args.operator_sheet_validation_json))
    else:
        add(checks, "input", "operator sheet validation", "pass", "operator sheet validation report is readable")
    if followup_validation_error:
        add(
            checks,
            "input",
            "operator follow-up validation",
            "fail",
            followup_validation_error,
            path=str(args.operator_followup_validation_json),
        )
    else:
        add(checks, "input", "operator follow-up validation", "pass", "operator follow-up validation report is readable")
    if execution_error:
        add(checks, "input", "operator execution", "fail", execution_error, path=str(args.operator_execution_json))
    else:
        add(checks, "input", "operator execution", "pass", "operator execution report is readable")

    if not packet_error:
        validate_packet_schema(checks, packet, args.artifact_root)
        validate_counts(checks, packet)
    if not packet_error and not sheet_error:
        validate_ready_actions(checks, packet, sheet, execution, args.expect_ready_action)
    if not packet_error:
        validate_report_gates(
            checks,
            packet,
            sheet,
            sheet_error,
            sheet_validation,
            sheet_validation_error,
            followup_validation,
            followup_validation_error,
            execution,
            execution_error,
        )

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
        "packet_status": str(packet.get("overall_status") or "unknown"),
        "artifact_root": str(args.artifact_root),
        "operator_submit_packet_json": str(args.operator_submit_packet_json),
        "operator_sheet_json": str(args.operator_sheet_json),
        "ready_actions": [str(item.get("id")) for item in packet_ready_actions(packet) if item.get("id")],
        "counts": counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Submit Packet Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Packet status: **{payload['packet_status'].upper()}**",
        f"Packet JSON: `{payload['operator_submit_packet_json']}`",
        f"Operator sheet JSON: `{payload['operator_sheet_json']}`",
        f"Ready actions: `{', '.join(payload['ready_actions']) or '-'}`",
        "",
        "This report is no-submit. It validates that the compact submit packet mirrors the operator sheet and keeps Slurm follow-up execution behind a terminal-state guard.",
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
