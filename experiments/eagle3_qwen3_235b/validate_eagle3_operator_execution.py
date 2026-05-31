#!/usr/bin/env python3
"""Validate operator execution records produced by run_eagle3_next_action.py.

This is a no-submit validator. It does not infer that a Slurm job finished
successfully; it only checks whether operator-side executions are recorded,
well-formed, linked to known next actions, and have successful shell return
codes. Stage success still belongs to the container/rollout/pipeline analyzers.
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
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--execution-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--require-record",
        action="append",
        default=[],
        help="Require a successful execution record for the given action id. Repeatable.",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    defaults = {
        "plan_json": Path(os.environ.get("NEXT_ACTION_PLAN_JSON", root / "reports/eagle3_next_actions.json")),
        "operator_sheet_json": Path(os.environ.get("OPERATOR_SHEET_JSON", root / "reports/eagle3_operator_sheet.json")),
        "execution_dir": Path(os.environ.get("OPERATOR_EXECUTION_DIR", root / "reports/operator_execution")),
        "json_out": Path(os.environ.get("OPERATOR_EXECUTION_JSON", root / "reports/eagle3_operator_execution.json")),
        "markdown_out": Path(os.environ.get("OPERATOR_EXECUTION_MARKDOWN", root / "reports/eagle3_operator_execution.md")),
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


def action_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in plan.get("next_actions") or []:
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def sheet_record_map(sheet: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in sheet.get("ready_actions") or []:
        if isinstance(item, dict) and item.get("id") and item.get("execution_record"):
            result[str(item["id"])] = str(item["execution_record"])
    return result


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def validate_record(
    path: Path,
    record: dict[str, Any],
    plan_actions: dict[str, dict[str, Any]],
    sheet_records: dict[str, str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    action = as_dict(record.get("action"))
    action_id = str(action.get("id") or "")
    mode = record.get("mode")
    expected_path = sheet_records.get(action_id)
    current_action = action_id in plan_actions
    current_record = False

    missing = [
        key
        for key in (
            "generated_at",
            "completed_at",
            "duration_seconds",
            "host",
            "cwd",
            "plan_json",
            "artifact_root",
            "action",
            "mode",
            "returncode",
            "after_returncodes",
        )
        if key not in record
    ]
    if missing:
        add(checks, "schema", path.name, "fail", "execution record is missing required keys", missing=missing)
    else:
        add(checks, "schema", path.name, "pass", "execution record has required keys")

    if not action_id:
        add(checks, "action", path.name, "fail", "execution record has no action id")
    elif action_id in plan_actions:
        add(checks, "action", action_id, "pass", "action id is present in the current next-action plan")
    else:
        add(
            checks,
            "action",
            action_id,
            "info",
            "historical action id is not present in the current next-action plan",
        )

    if expected_path:
        if Path(expected_path).expanduser().resolve(strict=False) == path.expanduser().resolve(strict=False):
            current_record = True
            add(checks, "path", action_id, "pass", "record path matches the current operator sheet")
        else:
            add(
                checks,
                "path",
                action_id,
                "info",
                "historical record path does not match the current operator sheet path",
                expected_path=expected_path,
                actual_path=str(path),
            )

    requires_current_result = current_record or (current_action and not expected_path)

    recorded_json_out = record.get("json_out")
    if recorded_json_out:
        if Path(str(recorded_json_out)).expanduser().resolve(strict=False) == path.expanduser().resolve(strict=False):
            add(checks, "path", f"{action_id} json_out", "pass", "recorded json_out points to this file")
        else:
            add(
                checks,
                "path",
                f"{action_id} json_out",
                "warn",
                "recorded json_out points elsewhere",
                recorded_json_out=recorded_json_out,
                actual_path=str(path),
            )

    if mode != "execute":
        status = "incomplete" if requires_current_result else "info"
        detail = f"record mode is {mode!r}, not execute"
        if status == "info":
            detail = f"historical record mode is {mode!r}, not execute"
        add(checks, "execution", action_id or path.name, status, detail)
        return checks

    returncode = record.get("returncode")
    if isinstance(returncode, int) and returncode == 0:
        add(checks, "execution", action_id, "pass", "action shell command returned zero")
    elif requires_current_result:
        add(checks, "execution", action_id, "fail", "action shell command returned nonzero or missing", returncode=returncode)
    else:
        add(checks, "execution", action_id, "info", "historical action shell command returned nonzero or missing", returncode=returncode)

    after_rows = record.get("after_returncodes") or []
    if not isinstance(after_rows, list):
        status = "fail" if requires_current_result else "info"
        add(checks, "after", action_id, status, "after_returncodes is not a list")
    elif not after_rows and action.get("after_commands"):
        status = "warn" if requires_current_result else "info"
        detail = "action has recorded after_commands, but this execution did not run them"
        if status == "info":
            detail = "historical action has recorded after_commands, but this execution did not run them"
        add(
            checks,
            "after",
            action_id,
            status,
            detail,
            after_command_count=len(action.get("after_commands") or []),
        )
    elif all(isinstance(row, dict) and row.get("returncode") == 0 for row in after_rows):
        add(checks, "after", action_id, "pass", "all recorded after_commands returned zero", count=len(after_rows))
    else:
        add(
            checks,
            "after",
            action_id,
            "fail",
            "one or more recorded after_commands returned nonzero",
            after_returncodes=after_rows,
        )
    return checks


def latest_by_action(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        action_id = str((item.get("record") or {}).get("action", {}).get("id") or "")
        if not action_id:
            continue
        previous = result.get(action_id)
        item_current = bool(item.get("current_record"))
        previous_current = bool((previous or {}).get("current_record"))
        item_time = float((item.get("record") or {}).get("completed_at_epoch") or 0)
        previous_time = float(((previous or {}).get("record") or {}).get("completed_at_epoch") or 0)
        if (
            previous is None
            or (item_current and not previous_current)
            or (item_current == previous_current and item_time >= previous_time)
        ):
            result[action_id] = item
    return result


def overall_status(checks: list[dict[str, Any]], records: list[dict[str, Any]], required_actions: list[str]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    latest = latest_by_action(records)
    missing_required = [action for action in required_actions if action not in latest]
    if missing_required:
        return "incomplete"
    if not records:
        return "not_started"
    if any(check["status"] in {"warn", "incomplete"} for check in checks):
        return "warn"
    return "pass"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    plan_raw, plan_error = load_json(args.plan_json)
    sheet_raw, sheet_error = load_json(args.operator_sheet_json)
    plan = as_dict(plan_raw)
    sheet = as_dict(sheet_raw)
    plan_actions = action_map(plan)
    sheet_records = sheet_record_map(sheet)

    checks: list[dict[str, Any]] = []
    if plan_error:
        add(checks, "input", "next-action plan", "warn", plan_error, path=str(args.plan_json))
    else:
        add(checks, "input", "next-action plan", "pass", "next-action plan is readable", actions=len(plan_actions))
    if sheet_error:
        add(checks, "input", "operator sheet", "warn", sheet_error, path=str(args.operator_sheet_json))
    else:
        add(checks, "input", "operator sheet", "pass", "operator sheet is readable", expected_records=len(sheet_records))

    record_items: list[dict[str, Any]] = []
    if not args.execution_dir.exists():
        add(checks, "records", "execution directory", "incomplete", "operator execution directory does not exist yet", path=str(args.execution_dir))
    else:
        paths = sorted(args.execution_dir.glob("*.json"))
        if not paths:
            add(checks, "records", "execution records", "incomplete", "no operator execution records are present yet", path=str(args.execution_dir))
        for path in paths:
            raw, error = load_json(path)
            if error:
                add(checks, "records", path.name, "fail", error, path=str(path))
                continue
            record = as_dict(raw)
            record_checks = validate_record(path, record, plan_actions, sheet_records)
            action_id = str((record.get("action") or {}).get("id") or "")
            expected_path = sheet_records.get(action_id)
            current_record = bool(
                expected_path
                and Path(expected_path).expanduser().resolve(strict=False) == path.expanduser().resolve(strict=False)
            )
            record_items.append(
                {
                    "path": str(path),
                    "current_record": current_record,
                    "record": record,
                    "checks": record_checks,
                }
            )
            checks.extend(record_checks)

    latest = latest_by_action(record_items)
    for action_id in args.require_record:
        if action_id in latest:
            add(checks, "expectation", action_id, "pass", "required action has an execution record")
        else:
            add(checks, "expectation", action_id, "fail", "required action is missing an execution record")

    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status(checks, record_items, args.require_record),
        "artifact_root": str(args.artifact_root),
        "plan_json": str(args.plan_json),
        "operator_sheet_json": str(args.operator_sheet_json),
        "execution_dir": str(args.execution_dir),
        "required_actions": args.require_record,
        "counts": counts,
        "records": [
            {
                "path": item["path"],
                "current_record": item.get("current_record"),
                "action_id": (item["record"].get("action") or {}).get("id"),
                "mode": item["record"].get("mode"),
                "returncode": item["record"].get("returncode"),
                "completed_at": item["record"].get("completed_at"),
                "duration_seconds": item["record"].get("duration_seconds"),
                "after_returncodes": item["record"].get("after_returncodes") or [],
            }
            for item in record_items
        ],
        "latest_by_action": {
            action_id: {
                "path": item["path"],
                "current_record": item.get("current_record"),
                "returncode": item["record"].get("returncode"),
                "completed_at": item["record"].get("completed_at"),
                "after_returncodes": item["record"].get("after_returncodes") or [],
            }
            for action_id, item in sorted(latest.items())
        },
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Execution Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Execution dir: `{payload['execution_dir']}`",
        f"Plan JSON: `{payload['plan_json']}`",
        f"Operator sheet JSON: `{payload['operator_sheet_json']}`",
        "",
        "This report validates operator-side execution records only. Slurm job success still comes from the stage analyzers.",
        "",
        "## Records",
        "",
        "| action | current | mode | return code | completed | record |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in payload["records"]:
        lines.append(
            f"| {item.get('action_id') or '-'} | {str(item.get('current_record')).lower()} | {item.get('mode') or '-'} | "
            f"{item.get('returncode') if item.get('returncode') is not None else '-'} | "
            f"{item.get('completed_at') or '-'} | `{item.get('path')}` |"
        )
    if not payload["records"]:
        lines.append("| - | - | - | - | - | no execution records found |")

    lines += [
        "",
        "## Checks",
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
    if args.fail_on_incomplete and payload["overall_status"] in {"incomplete", "not_started"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
