#!/usr/bin/env python3
"""Probe remote hosts for the Qwen3-235B Eagle3 cluster handoff.

This is intentionally read-only. It checks SSH reachability, Slurm command
visibility, Lustre paths, Hayate/Hiso reference paths, and git state without
submitting jobs or modifying remote files.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_HOSTS = [
    "oci-hsg-cs-001-vscode-02",
    "oci-hsg-cs-001-vscode-01",
    "oci-hsg-cs-001-vscode-03",
    "oci-hsg-cs-001-login-01.nvidia.com",
    "oci-hsg",
]

DEFAULT_SSH_CONFIG_HOST_PATTERNS = [
    "oci-*",
    "*hsg*",
    "*-cs-*",
]

DEFAULT_PATHS = [
    "/lustre",
    "/lustre/fs1",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer-worktrees",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer-worktrees/eagle3",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/Model-Optimizer",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec/models",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge/configs",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge/outputs",
    "/lustre/fsw/portfolios/coreai/users/sna",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=os.environ.get("REMOTE_HOSTS", "").split() or DEFAULT_HOSTS,
        help="Remote host aliases to probe.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Additional remote path to check. Can be repeated.",
    )
    parser.add_argument("--remote-workdir", default=os.environ.get("REMOTE_WORKDIR"))
    parser.add_argument("--artifact-root", default=os.environ.get("REMOTE_ARTIFACT_ROOT") or os.environ.get("ARTIFACT_ROOT"))
    parser.add_argument("--connect-timeout", type=int, default=int(os.environ.get("SSH_CONNECT_TIMEOUT", "8")))
    parser.add_argument("--command-timeout", type=int, default=30)
    parser.add_argument("--ssh-option", action="append", default=[], help="Extra ssh option, for example '-J jump-host'.")
    parser.add_argument(
        "--include-ssh-config-hosts",
        action="store_true",
        default=os.environ.get("INCLUDE_SSH_CONFIG_HOSTS", "").lower() in {"1", "true", "yes"},
        help="Add matching Host aliases from ~/.ssh/config to the probe host list.",
    )
    parser.add_argument(
        "--ssh-config-host-pattern",
        action="append",
        default=os.environ.get("SSH_CONFIG_HOST_PATTERNS", "").split() or [],
        help="fnmatch pattern for Host aliases to include when --include-ssh-config-hosts is set. Repeatable.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="Return nonzero if no host is reachable.")
    return parser.parse_args()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def ssh_config_hosts(patterns: list[str]) -> list[str]:
    config = Path.home() / ".ssh" / "config"
    if not config.exists():
        return []
    matched: list[str] = []
    try:
        lines = config.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0].lower() != "host":
            continue
        for alias in parts[1:]:
            if alias.startswith("!") or any(char in alias for char in "*?"):
                continue
            if any(fnmatch.fnmatchcase(alias, pattern) for pattern in patterns):
                matched.append(alias)
    return unique(matched)


def effective_ssh_config(host: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ssh", "-G", host],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"error": str(exc)}
    if proc.returncode:
        return {"returncode": proc.returncode, "error": proc.stderr.strip()}
    parsed: dict[str, Any] = {}
    for raw in proc.stdout.splitlines():
        parts = raw.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0].lower(), parts[1]
        if key in {"hostname", "port", "proxyjump"}:
            parsed[key] = value
        elif key == "proxycommand":
            parsed["has_proxycommand"] = value.lower() != "none"
    return parsed


def local_resolution(host: str, ssh_config: dict[str, Any]) -> dict[str, Any]:
    hostname = ssh_config.get("hostname") if isinstance(ssh_config.get("hostname"), str) else host
    result: dict[str, Any] = {"query": hostname, "resolved": False, "addresses": []}
    try:
        infos = socket.getaddrinfo(hostname, int(str(ssh_config.get("port") or "22")))
    except Exception as exc:
        result["error"] = str(exc)
        return result
    addresses = unique([info[4][0] for info in infos if info and info[4]])
    result["resolved"] = bool(addresses)
    result["addresses"] = addresses[:4]
    return result


def shell_array(values: list[str]) -> str:
    return " ".join(shlex.quote(v) for v in values)


def remote_probe_script(paths: list[str], artifact_root: str | None) -> str:
    path_array = shell_array(paths)
    artifact = shlex.quote(artifact_root or "/lustre")
    return f"""
set +e
tab="$(printf '\\t')"
emit() {{
  printf '%s\\t%s\\t%s\\n' "$1" "$2" "$3"
}}
emit FIELD probe_time "$(date -Is 2>/dev/null || date)"
emit FIELD hostname "$(hostname 2>/dev/null || true)"
emit FIELD pwd "$PWD"
for cmd in sbatch srun squeue sinfo sacct git python3 nvidia-smi; do
  found="$(command -v "$cmd" 2>/dev/null || true)"
  emit CMD "$cmd" "$found"
done
nvidia_out="$(nvidia-smi -L 2>&1 | head -5 | tr '\\t\\n' '  ')"
emit FIELD nvidia_smi "$nvidia_out"
df_out="$(df -h {artifact} 2>&1 | tail -1 | tr '\\t' ' ')"
emit FIELD artifact_df "$df_out"
for p in {path_array}; do
  exists=false
  readable=false
  executable=false
  listing=""
  git_branch=""
  git_head=""
  if [ -e "$p" ]; then
    exists=true
    [ -r "$p" ] && readable=true
    [ -x "$p" ] && executable=true
    listing="$(ls -ld "$p" 2>&1 | tr '\\t' ' ')"
    git_branch="$(git -C "$p" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    git_head="$(git -C "$p" rev-parse HEAD 2>/dev/null || true)"
  else
    listing="missing"
  fi
  printf 'PATH\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "$p" "$exists" "$readable" "$executable" "$listing" "$git_branch" "$git_head"
done
"""


def run_ssh(host: str, args: argparse.Namespace, paths: list[str]) -> dict[str, Any]:
    ssh_config = effective_ssh_config(host)
    ssh_cmd = [
        "ssh",
        "-S",
        "none",
        "-o",
        "ControlMaster=no",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-o",
        "LogLevel=ERROR",
    ]
    ssh_cmd.extend(args.ssh_option)
    script = remote_probe_script(paths, args.artifact_root)
    ssh_cmd.extend([host, script])

    started = time.time()
    result: dict[str, Any] = {
        "host": host,
        "reachable": False,
        "returncode": None,
        "duration_s": None,
        "fields": {},
        "commands": {},
        "paths": [],
        "stderr": "",
        "ssh_config": ssh_config,
        "local_resolution": local_resolution(host, ssh_config),
        "ssh_command": " ".join(shlex.quote(part) for part in ssh_cmd[:8] + ["...", host]),
    }
    try:
        proc = subprocess.run(
            ssh_cmd,
            text=True,
            capture_output=True,
            timeout=args.command_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result["returncode"] = "timeout"
        result["duration_s"] = round(time.time() - started, 3)
        result["stderr"] = str(exc)
        return result

    result["returncode"] = proc.returncode
    result["duration_s"] = round(time.time() - started, 3)
    result["stderr"] = proc.stderr.strip()
    result["reachable"] = proc.returncode == 0

    for raw in proc.stdout.splitlines():
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        kind = parts[0]
        if kind == "FIELD" and len(parts) >= 3:
            result["fields"][parts[1]] = parts[2]
        elif kind == "CMD" and len(parts) >= 3:
            result["commands"][parts[1]] = parts[2]
        elif kind == "PATH" and len(parts) >= 8:
            result["paths"].append(
                {
                    "path": parts[1],
                    "exists": parts[2] == "true",
                    "readable": parts[3] == "true",
                    "executable": parts[4] == "true",
                    "listing": parts[5],
                    "git_branch": parts[6] or None,
                    "git_head": parts[7] or None,
                }
            )
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Remote Host Probe",
        "",
        f"Overall: **{str(payload.get('overall_status', 'unknown')).upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Reachable hosts: `{len(payload.get('reachable_hosts', []))}` / `{len(payload.get('hosts_requested', []))}`",
        "",
        "## Summary",
        "",
        "| host | configured hostname | resolves | reachable | sbatch | remote hostname | nvidia-smi |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for host in payload["hosts"]:
        fields = host.get("fields", {})
        ssh_config = host.get("ssh_config", {}) if isinstance(host.get("ssh_config"), dict) else {}
        resolution = host.get("local_resolution", {}) if isinstance(host.get("local_resolution"), dict) else {}
        nvidia = fields.get("nvidia_smi", "")
        if len(nvidia) > 80:
            nvidia = nvidia[:77] + "..."
        lines.append(
            "| {host} | `{configured}` | {resolves} | {reachable} | `{sbatch}` | `{hostname}` | `{nvidia}` |".format(
                host=host["host"],
                configured=ssh_config.get("hostname") or host["host"],
                resolves=resolution.get("resolved"),
                reachable=host["reachable"],
                sbatch=host.get("commands", {}).get("sbatch", ""),
                hostname=fields.get("hostname", ""),
                nvidia=nvidia.replace("|", "\\|"),
            )
        )

    lines += ["", "## Paths", ""]
    for host in payload["hosts"]:
        lines += [
            f"### {host['host']}",
            "",
            "| path | exists | readable | executable | git |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in host.get("paths", []):
            git = ""
            if item.get("git_head"):
                git = f"{item.get('git_branch') or '?'}@{item['git_head'][:12]}"
            lines.append(
                "| `{path}` | {exists} | {readable} | {executable} | `{git}` |".format(
                    path=item["path"],
                    exists=item["exists"],
                    readable=item["readable"],
                    executable=item["executable"],
                    git=git,
                )
            )
        if host.get("stderr"):
            lines += ["", f"stderr: `{host['stderr']}`"]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    patterns = args.ssh_config_host_pattern or DEFAULT_SSH_CONFIG_HOST_PATTERNS
    configured_hosts = ssh_config_hosts(patterns) if args.include_ssh_config_hosts else []
    hosts = unique(list(args.hosts) + configured_hosts)
    paths = list(dict.fromkeys(DEFAULT_PATHS + args.path))
    if args.remote_workdir:
        paths.append(args.remote_workdir)
    if args.artifact_root:
        paths.append(args.artifact_root)
    paths = list(dict.fromkeys(paths))

    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hosts_requested": hosts,
        "host_discovery": {
            "include_ssh_config_hosts": bool(args.include_ssh_config_hosts),
            "ssh_config_host_patterns": patterns,
            "ssh_config_hosts": configured_hosts,
        },
        "paths_requested": paths,
        "artifact_root": args.artifact_root,
        "remote_workdir": args.remote_workdir,
        "hosts": [],
    }

    for host in hosts:
        print(f"Probing {host}...", file=sys.stderr)
        payload["hosts"].append(run_ssh(host, args, paths))

    reachable = [h for h in payload["hosts"] if h["reachable"]]
    payload["reachable_hosts"] = [h["host"] for h in reachable]
    payload["overall_status"] = "pass" if reachable else "unreachable"
    payload["counts"] = {
        "reachable": len(reachable),
        "unreachable": len(payload["hosts"]) - len(reachable),
        "requested": len(payload["hosts"]),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(payload) + "\n")

    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "reachable_hosts": payload["reachable_hosts"],
                "counts": payload["counts"],
            },
            indent=2,
        )
    )
    if args.strict and not reachable:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
