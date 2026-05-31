#!/usr/bin/env python3
"""Summarize remote-access diagnostics for the Eagle3 operator path.

This is a no-submit helper. It turns the raw SSH probe into an operator-facing
diagnosis so an unreachable local workstation does not get confused with proof
that remote Hayate/ModelOpt paths are absent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host-probe-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"top-level JSON is not an object: {path}"
    return payload, None


def edit_distance(a: str, b: str, limit: int = 3) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def ssh_config_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for row in rows:
        host = str(row.get("host") or "")
        configured = str(row.get("configured_hostname") or "")
        if not host or not configured or host == configured:
            continue
        distance = edit_distance(host, configured, limit=2)
        if distance <= 2 and "." not in host and "." not in configured:
            findings.append(
                {
                    "host": host,
                    "configured_hostname": configured,
                    "severity": "warning",
                    "finding": "possible_ssh_config_hostname_typo",
                    "detail": (
                        "effective ssh HostName is very close to the Host alias but differs; "
                        "verify ~/.ssh/config before treating DNS failure as a network-only issue"
                    ),
                }
            )
    return findings


def host_rows(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for host in (probe or {}).get("hosts") or []:
        if not isinstance(host, dict):
            continue
        ssh_config = host.get("ssh_config") if isinstance(host.get("ssh_config"), dict) else {}
        resolution = host.get("local_resolution") if isinstance(host.get("local_resolution"), dict) else {}
        rows.append(
            {
                "host": host.get("host"),
                "configured_hostname": ssh_config.get("hostname") or host.get("host"),
                "resolved": bool(resolution.get("resolved")),
                "resolution_query": resolution.get("query"),
                "resolution_error": resolution.get("error"),
                "reachable": bool(host.get("reachable")),
                "returncode": host.get("returncode"),
                "stderr": host.get("stderr"),
            }
        )
    return rows


def build_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    probe, error = load_json(args.remote_host_probe_json)
    rows = host_rows(probe)
    config_findings = ssh_config_findings(rows)
    reachable = [row for row in rows if row["reachable"]]
    resolved = [row for row in rows if row["resolved"]]
    unresolved = [row for row in rows if not row["resolved"]]
    status = (probe or {}).get("overall_status")
    if error:
        overall = "missing_probe"
        diagnosis = "remote host probe report is missing or unreadable"
    elif reachable:
        overall = "pass"
        diagnosis = "at least one remote host is reachable from this environment"
    elif rows and not resolved and len(unresolved) == len(rows):
        overall = "blocked_local_dns"
        diagnosis = "all probed host aliases failed local DNS resolution before SSH could test remote paths"
    elif rows:
        overall = "unreachable"
        diagnosis = "no probed remote host is reachable from this environment"
    else:
        overall = "missing_hosts"
        diagnosis = "remote host probe contains no host rows"

    discovery = (probe or {}).get("host_discovery") if isinstance((probe or {}).get("host_discovery"), dict) else {}
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "diagnosis": diagnosis,
        "remote_host_probe_json": str(args.remote_host_probe_json),
        "remote_probe_status": status if not error else "missing",
        "error": error,
        "counts": {
            "hosts": len(rows),
            "resolved_hosts": len(resolved),
            "unresolved_hosts": len(unresolved),
            "reachable_hosts": len(reachable),
            "ssh_config_hosts": len(discovery.get("ssh_config_hosts") or []),
            "ssh_config_hostname_warnings": len(config_findings),
        },
        "host_discovery": discovery,
        "configuration_findings": config_findings,
        "host_diagnostics": rows,
        "gate_interpretation": {
            "remote_hayate_reference_probe": "open",
            "remote_path_absence_proven": False,
            "reason": (
                "The current local environment cannot reach a remote host, so Hayate ModelOpt and SpecForge "
                "remote paths remain unproven rather than disproven."
            ),
        },
        "next_actions": [
            "Connect to the network/VPN or resolver context that can resolve the selected cluster aliases.",
            "Review configuration_findings and fix any SSH HostName typo before rerunning the probe.",
            "Rerun probe_eagle3_remote_host.py with --include-ssh-config-hosts from that environment.",
            "If a different reachable alias is known, set REMOTE_HOSTS or pass --hosts explicitly.",
            "After a reachable host is recorded, refresh the operator state so the training-path manifest can close remote_hayate_reference_probe.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Remote Access Diagnostics",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Diagnosis: {payload['diagnosis']}",
        "",
        "## Counts",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    for key, value in (payload.get("counts") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Host Diagnostics",
        "",
        "| host | configured hostname | resolved | reachable | returncode |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("host_diagnostics") or []:
        lines.append(
            "| `{host}` | `{configured}` | {resolved} | {reachable} | `{returncode}` |".format(
                host=row.get("host"),
                configured=row.get("configured_hostname"),
                resolved=str(row.get("resolved")).lower(),
                reachable=str(row.get("reachable")).lower(),
                returncode=row.get("returncode"),
            )
        )
    findings = payload.get("configuration_findings") or []
    if findings:
        lines += [
            "",
            "## Configuration Findings",
            "",
            "| host | configured hostname | finding | detail |",
            "| --- | --- | --- | --- |",
        ]
        for item in findings:
            lines.append(
                f"| `{item.get('host')}` | `{item.get('configured_hostname')}` | "
                f"{item.get('finding')} | {str(item.get('detail') or '').replace('|', '/')} |"
            )
    lines += ["", "## Next Actions", ""]
    for item in payload.get("next_actions") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = build_diagnostics(args)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
