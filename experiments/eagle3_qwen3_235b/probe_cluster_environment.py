#!/usr/bin/env python3
"""Probe cluster substrate needed by the Qwen3 Eagle3 Slurm pilot.

This script is intentionally lightweight: it checks command availability,
Slurm visibility, container/mount paths, artifact disk space, and a few Python
imports without submitting jobs or loading model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT / "outputs" / "qwen3_235b_eagle3"


@dataclass
class Check:
    area: str
    name: str
    status: str
    required: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument("--modelopt-dir", type=Path, default=Path(os.environ.get("MODELOPT_DIR", ROOT / "Model-Optimizer")))
    parser.add_argument("--verifier-config-dir", type=Path, default=env_path("VERIFIER_CONFIG_DIR"))
    parser.add_argument("--input-data", type=Path, default=env_path("INPUT_DATA"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", ""))
    parser.add_argument("--mounts", default=os.environ.get("MOUNTS", ""))
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "dummy"))
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--min-artifact-free-gib", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true", help="Return nonzero if required cluster checks fail.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def run(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:],
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": str(exc)}


def add(
    checks: list[Check],
    area: str,
    name: str,
    status: str,
    detail: str,
    *,
    required: bool = False,
    **evidence: Any,
) -> None:
    checks.append(Check(area=area, name=name, status=status, required=required, detail=detail, evidence=evidence))


def command_check(checks: list[Check], command: str, *, required: bool) -> str | None:
    path = shutil.which(command)
    if path:
        version = run([command, "--version"], timeout=6)
        add(
            checks,
            "commands",
            f"{command} available",
            "pass",
            f"{command} found",
            required=required,
            path=path,
            version=version.get("stdout", "").splitlines()[:3],
        )
        return path
    add(checks, "commands", f"{command} available", "fail" if required else "warn", f"{command} not found", required=required)
    return None


def check_slurm_associations(args: argparse.Namespace, checks: list[Check]) -> None:
    if not command_check(checks, "sacctmgr", required=False):
        return

    user = os.environ.get("USER") or run(["whoami"], timeout=3).get("stdout", "").strip()
    if not user:
        add(checks, "slurm", "account associations", "warn", "could not determine current user", required=False)
        return

    result = run(
        [
            "sacctmgr",
            "-nP",
            "show",
            "assoc",
            f"user={user}",
            "format=Account,Partition,QOS,DefaultQOS",
        ],
        timeout=8,
    )
    rows = [line for line in result.get("stdout", "").splitlines() if line.strip()]
    accounts = sorted({row.split("|", 1)[0] for row in rows if row.split("|", 1)[0]})
    if accounts:
        add(
            checks,
            "slurm",
            "account associations",
            "pass",
            f"{len(accounts)} Slurm accounts visible for {user}",
            required=False,
            user=user,
            accounts=accounts,
            rows=rows[:20],
        )
    else:
        add(
            checks,
            "slurm",
            "account associations",
            "warn",
            f"no sacctmgr associations visible for {user}",
            required=False,
            user=user,
            returncode=result.get("returncode"),
            output=result.get("stdout", "").splitlines()[:20],
        )
        return

    if args.sbatch_account and args.sbatch_account != "dummy":
        valid = args.sbatch_account in accounts
        status = "pass" if valid else ("fail" if args.strict else "warn")
        add(
            checks,
            "slurm",
            "SBATCH_ACCOUNT association",
            status,
            "SBATCH_ACCOUNT is visible in sacctmgr associations" if valid else "SBATCH_ACCOUNT was not found in sacctmgr associations",
            required=args.strict,
            value=args.sbatch_account,
            accounts=accounts,
        )


def check_slurm(args: argparse.Namespace, checks: list[Check]) -> None:
    sbatch = command_check(checks, "sbatch", required=True)
    srun = command_check(checks, "srun", required=True)
    command_check(checks, "squeue", required=False)
    sinfo = command_check(checks, "sinfo", required=False)
    if args.sbatch_account == "dummy":
        add(checks, "slurm", "SBATCH_ACCOUNT", "fail", "SBATCH_ACCOUNT is still dummy", required=True, value=args.sbatch_account)
    else:
        add(checks, "slurm", "SBATCH_ACCOUNT", "pass", "SBATCH_ACCOUNT is set", required=True, value=args.sbatch_account)
    check_slurm_associations(args, checks)
    if args.sbatch_partition:
        add(checks, "slurm", "SBATCH_PARTITION", "pass", "SBATCH_PARTITION is set", required=False, value=args.sbatch_partition)
    if sinfo:
        result = run(["sinfo", "-h", "-o", "%P|%D|%G|%m", "-p", args.sbatch_partition], timeout=8)
        status = "pass" if result.get("returncode") == 0 and result.get("stdout", "").strip() else "warn"
        add(
            checks,
            "slurm",
            "partition visibility",
            status,
            f"sinfo probe for partition {args.sbatch_partition}",
            required=False,
            output=result.get("stdout", "").splitlines()[:10],
            returncode=result.get("returncode"),
        )
    if sbatch and srun:
        add(checks, "slurm", "submitter commands", "pass", "sbatch and srun are both available", required=True)


def parse_mount_paths(mounts: str) -> list[str]:
    paths: list[str] = []
    for item in mounts.split(","):
        item = item.strip()
        if not item:
            continue
        host_path = item.split(":", 1)[0]
        paths.append(host_path)
    return paths


def check_paths(args: argparse.Namespace, checks: list[Check]) -> None:
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    writable = os.access(args.artifact_root, os.W_OK)
    add(
        checks,
        "paths",
        "artifact root writable",
        "pass" if writable else "fail",
        "artifact root exists and write permission was checked",
        required=True,
        path=str(args.artifact_root),
    )
    usage = shutil.disk_usage(args.artifact_root)
    free_gib = usage.free / (1024**3)
    disk_ok = free_gib >= args.min_artifact_free_gib
    add(
        checks,
        "paths",
        "artifact root free space",
        "pass" if disk_ok else "fail",
        f"{free_gib:.1f} GiB free under artifact root",
        required=args.min_artifact_free_gib > 0,
        free_gib=round(free_gib, 2),
        min_required_gib=args.min_artifact_free_gib,
    )
    modelopt_ok = (args.modelopt_dir / "examples/speculative_decoding/launch_train.sh").exists()
    add(
        checks,
        "paths",
        "ModelOpt checkout",
        "pass" if modelopt_ok else "fail",
        "ModelOpt speculative decoding entrypoint visibility",
        required=True,
        path=str(args.modelopt_dir),
    )
    if args.verifier_config_dir:
        visible = (args.verifier_config_dir / "config.json").exists()
        add(
            checks,
            "paths",
            "verifier config",
            "pass" if visible else "warn",
            "VERIFIER_CONFIG_DIR/config.json visibility",
            required=False,
            path=str(args.verifier_config_dir),
        )
    if args.input_data:
        visible = args.input_data.exists() and args.input_data.stat().st_size > 0
        add(
            checks,
            "paths",
            "input data",
            "pass" if visible else "warn",
            "INPUT_DATA visibility",
            required=False,
            path=str(args.input_data),
        )
    if args.container:
        exists = Path(args.container).exists()
        add(
            checks,
            "container",
            "container image path",
            "pass" if exists else "warn",
            "CONTAINER path visibility",
            required=False,
            path=args.container,
        )
    if args.mounts:
        for host_path in parse_mount_paths(args.mounts):
            exists = Path(host_path).exists()
            add(
                checks,
                "container",
                f"mount host path {host_path}",
                "pass" if exists else "warn",
                "host-side mount path visibility",
                required=False,
                path=host_path,
            )


def check_gpu(checks: list[Check]) -> None:
    nvidia_smi = command_check(checks, "nvidia-smi", required=False)
    if not nvidia_smi:
        return
    result = run(["nvidia-smi", "-L"], timeout=8)
    output = result.get("stdout", "").strip()
    status = "pass" if result.get("returncode") == 0 and output else "warn"
    add(
        checks,
        "gpu",
        "nvidia-smi GPU list",
        status,
        "GPU visibility on this host",
        required=False,
        output=output.splitlines()[:16],
        returncode=result.get("returncode"),
    )


def check_python(args: argparse.Namespace, checks: list[Check]) -> None:
    py = shutil.which(args.python_bin) or args.python_bin
    version = run([py, "--version"], timeout=6)
    add(
        checks,
        "python",
        "python executable",
        "pass" if version.get("returncode") == 0 else "fail",
        "Python executable availability",
        required=True,
        python=py,
        output=version.get("stdout", "").strip(),
    )
    import_probe = (
        "import importlib.util, json; "
        "mods=['torch','transformers','datasets']; "
        "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
    )
    result = run([py, "-c", import_probe], timeout=10)
    status = "pass" if result.get("returncode") == 0 else "warn"
    add(
        checks,
        "python",
        "core python packages",
        status,
        "torch/transformers/datasets import-spec probe",
        required=False,
        output=result.get("stdout", "").strip(),
        returncode=result.get("returncode"),
    )


def overall_status(checks: list[Check]) -> str:
    if any(item.required and item.status == "fail" for item in checks):
        return "fail"
    if any(item.status in {"fail", "warn"} for item in checks):
        return "warn"
    return "pass"


def payload(args: argparse.Namespace, checks: list[Check]) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status(checks),
        "host": {
            "hostname": run(["hostname"], timeout=3).get("stdout", "").strip(),
            "cwd": str(Path.cwd()),
        },
        "inputs": {
            "artifact_root": str(args.artifact_root),
            "modelopt_dir": str(args.modelopt_dir),
            "verifier_config_dir": str(args.verifier_config_dir) if args.verifier_config_dir else None,
            "input_data": str(args.input_data) if args.input_data else None,
            "container": args.container,
            "mounts": args.mounts,
            "sbatch_account": args.sbatch_account,
            "sbatch_partition": args.sbatch_partition,
            "min_artifact_free_gib": args.min_artifact_free_gib,
        },
        "checks": [
            {
                "area": item.area,
                "name": item.name,
                "status": item.status,
                "required": item.required,
                "detail": item.detail,
                "evidence": item.evidence,
            }
            for item in checks
        ],
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Eagle3 Cluster Environment Probe",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Generated: `{data['generated_at']}`",
        f"Host: `{data['host'].get('hostname')}`",
        "",
        "| area | check | required | status | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in data["checks"]:
        lines.append(
            f"| {item['area']} | {item['name']} | {item['required']} | "
            f"{item['status'].upper()} | {item['detail'].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(data: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(data)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")


def main() -> int:
    args = parse_args()
    checks: list[Check] = []
    check_slurm(args, checks)
    check_paths(args, checks)
    check_gpu(checks)
    check_python(args, checks)
    data = payload(args, checks)
    write_outputs(data, args)
    return 1 if args.strict and data["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
