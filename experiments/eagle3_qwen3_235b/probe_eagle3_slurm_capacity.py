#!/usr/bin/env python3
"""Probe Slurm GPU shape for the Qwen3-235B Eagle3 pipeline.

This is read-only. It does not submit jobs. It compares the visible Slurm
partition GRES shape with the hidden-state dump/train/export resource defaults
used by submit_eagle3_pipeline.sh, so the operator can catch a 4-GPU-node vs
8-GPU-request mismatch before sbatch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
GPU_RE = re.compile(r"gpu(?::[^:(),]+)?:(\d+)")


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=root)
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--dump-nodes", type=int, default=int(os.environ.get("DUMP_NODES", "1")))
    parser.add_argument("--dump-gpus-per-node", type=int, default=int(os.environ.get("DUMP_GPUS_PER_NODE", "8")))
    parser.add_argument("--dump-ntasks-per-node", type=int, default=int(os.environ.get("DUMP_NTASKS_PER_NODE", "1")))
    parser.add_argument("--train-nodes", type=int, default=int(os.environ.get("TRAIN_NODES", "1")))
    parser.add_argument("--train-gpus-per-node", type=int, default=int(os.environ.get("TRAIN_GPUS_PER_NODE", "8")))
    parser.add_argument("--export-gpus-per-node", type=int, default=int(os.environ.get("EXPORT_GPUS_PER_NODE", "1")))
    parser.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--env-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def run(command: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "lines": result.stdout.splitlines(),
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "error": str(exc), "stdout": "", "lines": []}


def parse_gpu_count(gres: str) -> int | None:
    matches = [int(item) for item in GPU_RE.findall(gres or "")]
    if matches:
        return max(matches)
    if "gpu" in (gres or "").lower():
        bare = re.search(r"gpu:(\d+)", gres)
        if bare:
            return int(bare.group(1))
    return None


def parse_partition_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        partition, nodes, gres, memory = parts[:4]
        features = parts[4] if len(parts) > 4 else ""
        try:
            node_count = int(str(nodes).strip())
        except ValueError:
            node_count = None
        try:
            memory_mb = int(str(memory).strip())
        except ValueError:
            memory_mb = None
        rows.append(
            {
                "partition": partition,
                "nodes": node_count,
                "gres": gres,
                "gpu_per_node": parse_gpu_count(gres),
                "memory_mb": memory_mb,
                "features": features,
            }
        )
    return rows


def parse_node_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        node, gres, memory, features = parts[:4]
        try:
            memory_mb = int(str(memory).strip())
        except ValueError:
            memory_mb = None
        rows.append(
            {
                "node": node,
                "gres": gres,
                "gpu_per_node": parse_gpu_count(gres),
                "memory_mb": memory_mb,
                "features": features,
            }
        )
    return rows


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def request_check(
    checks: list[dict[str, Any]],
    label: str,
    requested: int,
    max_gpu_per_node: int | None,
    *,
    nodes: int = 1,
) -> None:
    if max_gpu_per_node is None:
        add(checks, "capacity", label, "warn", "cannot determine visible GPUs per node from sinfo", requested=requested)
    elif requested <= max_gpu_per_node:
        add(
            checks,
            "capacity",
            label,
            "pass",
            "requested GPUs per node fit the visible partition shape",
            requested_gpus_per_node=requested,
            max_visible_gpus_per_node=max_gpu_per_node,
            nodes=nodes,
        )
    else:
        add(
            checks,
            "capacity",
            label,
            "fail",
            "requested GPUs per node exceed the visible partition shape",
            requested_gpus_per_node=requested,
            max_visible_gpus_per_node=max_gpu_per_node,
            nodes=nodes,
        )


def recommendations(args: argparse.Namespace, max_gpu_per_node: int | None, checks: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    has_gpu_mismatch = any(item["status"] == "fail" and item["area"] == "capacity" for item in checks)
    if has_gpu_mismatch and max_gpu_per_node:
        recs.append(
            "Either use a partition with at least the requested GPUs per node, or set "
            f"DUMP_GPUS_PER_NODE={max_gpu_per_node} TRAIN_GPUS_PER_NODE={max_gpu_per_node} TP={max_gpu_per_node} "
            "and rerun preflight. Memory fit still needs the container/runtime preflight."
        )
    elif max_gpu_per_node:
        recs.append("Visible Slurm GPU count is compatible with the current per-node requests.")
    else:
        recs.append("sinfo did not expose GPU counts; verify partition GPU shape with the cluster operator before submit.")
    recs.append(
        "GB200/B200 4-GPU Slurm nodes may be viable for the offline path if TRT-LLM fits the verifier with TP=4. "
        "H100 usually needs an 8x80GB node or a separately validated multi-node/quantized verifier setup for Qwen3-235B."
    )
    return recs


def build_resource_profile(args: argparse.Namespace, max_gpu_per_node: int | None, checks: list[dict[str, Any]]) -> dict[str, Any]:
    has_gpu_mismatch = any(item["status"] == "fail" and item["area"] == "capacity" for item in checks)
    if max_gpu_per_node is None:
        return {
            "status": "requested_unverified",
            "name": "requested_profile_slurm_shape_unproven",
            "detail": "sinfo did not expose GPU counts; preserve the requested profile for downstream dry-run propagation checks, but do not treat capacity as proven",
            "env": {
                "EAGLE3_RESOURCE_PROFILE": "requested_profile_slurm_shape_unproven",
                "EAGLE3_RESOURCE_PROFILE_STATUS": "slurm_shape_unproven",
                "DUMP_GPUS_PER_NODE": str(args.dump_gpus_per_node),
                "TRAIN_GPUS_PER_NODE": str(args.train_gpus_per_node),
                "EXPORT_GPUS_PER_NODE": str(args.export_gpus_per_node),
                "TP": str(args.tp),
            },
            "memory_fit": "unproven",
            "source": "current request; Slurm shape unavailable",
        }

    if has_gpu_mismatch:
        selected = max_gpu_per_node
        return {
            "status": "candidate",
            "name": f"visible_{selected}gpu_per_node",
            "detail": "resource request fits visible Slurm GPU shape, but Qwen3-235B runtime memory fit is still unproven",
            "env": {
                "EAGLE3_RESOURCE_PROFILE": f"visible_{selected}gpu_per_node",
                "EAGLE3_RESOURCE_PROFILE_STATUS": "shape_pass_memory_unproven",
                "DUMP_GPUS_PER_NODE": str(selected),
                "TRAIN_GPUS_PER_NODE": str(selected),
                "EXPORT_GPUS_PER_NODE": str(min(args.export_gpus_per_node, selected)),
                "TP": str(selected),
            },
            "memory_fit": "unproven",
            "source": "visible Slurm max GPU-per-node",
        }

    return {
        "status": "current",
        "name": "current_requests_fit_partition",
        "detail": "current resource request already fits visible Slurm GPU shape",
            "env": {
                "EAGLE3_RESOURCE_PROFILE": "current_requests_fit_partition",
                "EAGLE3_RESOURCE_PROFILE_STATUS": "shape_pass_memory_unproven",
                "DUMP_GPUS_PER_NODE": str(args.dump_gpus_per_node),
                "TRAIN_GPUS_PER_NODE": str(args.train_gpus_per_node),
                "EXPORT_GPUS_PER_NODE": str(args.export_gpus_per_node),
                "TP": str(args.tp),
            },
        "memory_fit": "unproven",
        "source": "current request",
    }


def render_env(profile: dict[str, Any]) -> str:
    env = profile.get("env") or {}
    lines = [
        "# Generated by probe_eagle3_slurm_capacity.py",
        f"# profile: {profile.get('name')}",
        f"# status: {profile.get('status')}",
        f"# memory_fit: {profile.get('memory_fit')}",
    ]
    for key in sorted(env):
        lines.append(f"export {key}={shlex.quote(str(env[key]))}")
    return "\n".join(lines).rstrip() + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    sinfo = shutil.which("sinfo")
    if not sinfo:
        add(checks, "slurm", "sinfo", "warn", "sinfo is not on PATH")
        partition_rows: list[dict[str, Any]] = []
        node_rows: list[dict[str, Any]] = []
        partition_result = {"returncode": None, "stdout": ""}
        node_result = {"returncode": None, "stdout": ""}
    else:
        add(checks, "slurm", "sinfo", "pass", f"sinfo visible: {sinfo}", path=sinfo)
        partition_result = run(["sinfo", "-h", "-o", "%P|%D|%G|%m|%f", "-p", args.sbatch_partition])
        node_result = run(["sinfo", "-h", "-N", "-o", "%N|%G|%m|%f", "-p", args.sbatch_partition])
        partition_rows = parse_partition_rows(partition_result.get("lines") or [])
        node_rows = parse_node_rows(node_result.get("lines") or [])

    if partition_rows:
        add(
            checks,
            "slurm",
            "partition visibility",
            "pass",
            f"partition {args.sbatch_partition} is visible",
            partition_rows=partition_rows[:8],
        )
    else:
        add(
            checks,
            "slurm",
            "partition visibility",
            "warn",
            f"no sinfo rows for partition {args.sbatch_partition}",
            returncode=partition_result.get("returncode"),
            output=(partition_result.get("stdout") or "").splitlines()[:8],
        )

    visible_gpu_counts = [row["gpu_per_node"] for row in node_rows + partition_rows if row.get("gpu_per_node")]
    max_gpu_per_node = max(visible_gpu_counts) if visible_gpu_counts else None
    unique_gres = sorted({str(row.get("gres")) for row in node_rows + partition_rows if row.get("gres")})
    memory_values = [int(row["memory_mb"]) for row in node_rows + partition_rows if row.get("memory_mb")]
    max_memory_mb = max(memory_values) if memory_values else None

    request_check(checks, "hidden-state dump GPU request", args.dump_gpus_per_node, max_gpu_per_node, nodes=args.dump_nodes)
    request_check(checks, "offline train GPU request", args.train_gpus_per_node, max_gpu_per_node, nodes=args.train_nodes)
    request_check(checks, "export GPU request", args.export_gpus_per_node, max_gpu_per_node, nodes=1)
    if args.tp <= args.dump_gpus_per_node:
        add(
            checks,
            "capacity",
            "TRT-LLM TP vs dump GPUs",
            "pass",
            "TP does not exceed hidden-state dump GPUs per node",
            tp=args.tp,
            dump_gpus_per_node=args.dump_gpus_per_node,
        )
    else:
        add(
            checks,
            "capacity",
            "TRT-LLM TP vs dump GPUs",
            "fail",
            "TP exceeds hidden-state dump GPUs per node",
            tp=args.tp,
            dump_gpus_per_node=args.dump_gpus_per_node,
        )

    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    if counts.get("fail"):
        overall = "fail"
    elif counts.get("warn"):
        overall = "warn"
    else:
        overall = "pass"

    profile = build_resource_profile(args, max_gpu_per_node, checks)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "artifact_root": str(args.artifact_root),
        "sbatch_partition": args.sbatch_partition,
        "requests": {
            "dump_nodes": args.dump_nodes,
            "dump_gpus_per_node": args.dump_gpus_per_node,
            "dump_ntasks_per_node": args.dump_ntasks_per_node,
            "train_nodes": args.train_nodes,
            "train_gpus_per_node": args.train_gpus_per_node,
            "export_gpus_per_node": args.export_gpus_per_node,
            "tp": args.tp,
        },
        "visible_capacity": {
            "max_gpu_per_node": max_gpu_per_node,
            "max_memory_mb": max_memory_mb,
            "unique_gres": unique_gres,
            "partition_rows": partition_rows[:16],
            "node_rows_sample": node_rows[:16],
        },
        "resource_profile": profile,
        "recommendations": recommendations(args, max_gpu_per_node, checks),
        "counts": counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Slurm Capacity Probe",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Partition: `{payload['sbatch_partition']}`",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Visible Capacity",
        "",
        f"- max GPU per node: `{payload['visible_capacity'].get('max_gpu_per_node')}`",
        f"- unique GRES: `{', '.join(payload['visible_capacity'].get('unique_gres') or []) or '-'}`",
        f"- max node memory MB: `{payload['visible_capacity'].get('max_memory_mb')}`",
        "",
        "## Requests",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    for key, value in payload["requests"].items():
        lines.append(f"| {key} | `{value}` |")
    lines += ["", "## Checks", "", "| area | check | status | detail |", "| --- | --- | --- | --- |"]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['area']} | {check['name']} | {check['status'].upper()} | {detail} |")
    lines += ["", "## Recommendations", ""]
    lines.extend(f"- {item}" for item in payload["recommendations"])
    profile = payload.get("resource_profile") or {}
    lines += [
        "",
        "## Resource Profile",
        "",
        f"- name: `{profile.get('name')}`",
        f"- status: `{profile.get('status')}`",
        f"- memory fit: `{profile.get('memory_fit')}`",
        f"- detail: {profile.get('detail')}",
    ]
    env = profile.get("env") or {}
    if env:
        lines += ["", "```bash"]
        lines.extend(f"export {key}={shlex.quote(str(env[key]))}" for key in sorted(env))
        lines.append("```")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.env_out:
        args.env_out.parent.mkdir(parents=True, exist_ok=True)
        args.env_out.write_text(render_env(payload.get("resource_profile") or {}), encoding="utf-8")
    return 1 if args.strict and payload["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
