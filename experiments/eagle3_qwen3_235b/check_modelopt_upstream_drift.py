#!/usr/bin/env python3
"""Report ModelOpt upstream/Hayate drift for the Qwen3 Eagle3 path.

This is a lightweight provenance check. It does not fetch or mutate git refs.
When network is available, it probes the official NVIDIA/Model-Optimizer main
branch with `git ls-remote`; when Hayate's Lustre worktree is mounted, it also
compares key speculative-decoding files by hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_MODELOPT = ROOT / "Model-Optimizer"
DEFAULT_UPSTREAM_URL = "https://github.com/NVIDIA/Model-Optimizer.git"
DEFAULT_UPSTREAM_REF = "refs/heads/main"
DEFAULT_HAYATE_MODELOPT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer"
)

FOCUS_PATHS = [
    "examples/speculative_decoding/README.md",
    "examples/speculative_decoding/launch_train.sh",
    "examples/speculative_decoding/main.py",
    "examples/speculative_decoding/fsdp_config.json",
    "examples/speculative_decoding/collect_hidden_states/common.py",
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py",
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
    "examples/speculative_decoding/scripts/export_hf_checkpoint.py",
    "examples/speculative_decoding/scripts/convert_to_vllm_ckpt.py",
    "examples/speculative_decoding/prepare_input_conversations/add_dapo17k.py",
    "examples/speculative_decoding/prepare_input_conversations/generate_responses.py",
    "examples/speculative_decoding/slurm/generate_responses.sbatch",
    "examples/speculative_decoding/slurm/train_eagle3.sbatch",
    "examples/speculative_decoding/slurm/submit_all.sh",
    "examples/speculative_decoding/eagle_config_qwen3_8b.json",
    "examples/speculative_decoding/eagle_config_qwen3_30b_moe.json",
    "examples/speculative_decoding/eagle_config_qwen3_32b.json",
    "modelopt_recipes/general/speculative_decoding/eagle3.yaml",
    "modelopt/torch/speculative/eagle/config.py",
    "modelopt/torch/speculative/eagle/eagle_model.py",
    "modelopt/torch/speculative/eagle/utils.py",
]

ALLOWED_QWEN3_FOCUS_DIFFS = {
    # Local/remote training path patch: preserve answer-only loss_mask through
    # the TRT-LLM hidden-state dumper used before offline Eagle3 training.
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelopt-dir", type=Path, default=DEFAULT_MODELOPT)
    parser.add_argument("--hayate-modelopt-dir", type=Path, default=DEFAULT_HAYATE_MODELOPT)
    parser.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL)
    parser.add_argument("--upstream-ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument("--probe-upstream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "cmd": cmd,
        }
    except Exception as exc:
        return {"returncode": -1, "stdout": str(exc), "cmd": cmd}


def git(repo: Path, *args: str) -> dict[str, Any]:
    return run(["git", "-C", str(repo), *args])


def first_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def file_hash(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "exists": True,
        "sha256": h.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def summarize_repo(path: Path, label: str, focus_paths: list[str]) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": exists,
    }
    if not exists:
        return payload

    inside = git(path, "rev-parse", "--is-inside-work-tree")
    payload["is_git_repo"] = inside["returncode"] == 0 and inside["stdout"] == "true"
    if not payload["is_git_repo"]:
        return payload

    branch = git(path, "branch", "--show-current")
    head = git(path, "rev-parse", "HEAD")
    status = git(path, "status", "--porcelain")
    remotes = git(path, "remote", "-v")
    payload.update(
        {
            "branch": first_line(branch["stdout"]),
            "head": first_line(head["stdout"]),
            "short_head": (first_line(head["stdout"]) or "")[:12],
            "dirty_files": status["stdout"].splitlines() if status["stdout"] else [],
            "remote_lines": remotes["stdout"].splitlines() if remotes["stdout"] else [],
        }
    )

    diff_stat = git(path, "diff", "--stat", "--", *focus_paths)
    diff_names = git(path, "diff", "--name-status", "--", *focus_paths)
    payload["focus_diff_stat"] = diff_stat["stdout"]
    payload["focus_diff_files"] = diff_names["stdout"].splitlines() if diff_names["stdout"] else []
    payload["focus_files"] = {rel: file_hash(path / rel) for rel in focus_paths}
    return payload


def compare_focus_files(local: dict[str, Any], hayate: dict[str, Any]) -> list[dict[str, Any]]:
    local_files = local.get("focus_files") or {}
    hayate_files = hayate.get("focus_files") or {}
    rows: list[dict[str, Any]] = []
    for rel in FOCUS_PATHS:
        lhs = local_files.get(rel, {})
        rhs = hayate_files.get(rel, {})
        rows.append(
            {
                "path": rel,
                "local_exists": lhs.get("exists", False),
                "hayate_exists": rhs.get("exists", False),
                "same_sha256": bool(lhs.get("sha256") and lhs.get("sha256") == rhs.get("sha256")),
                "local_sha256": lhs.get("sha256"),
                "hayate_sha256": rhs.get("sha256"),
            }
        )
    return rows


def dirty_path(status_line: str) -> str:
    parts = status_line.split(maxsplit=1)
    if len(parts) == 2:
        path = parts[1]
    else:
        path = status_line[3:] if len(status_line) > 3 else status_line
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip()


def dirty_paths(repo: dict[str, Any]) -> list[str]:
    return [dirty_path(str(item)) for item in repo.get("dirty_files") or []]


def focus_dirty_paths(repo: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for item in repo.get("focus_diff_files") or []:
        path = dirty_path(str(item))
        if path:
            result.append(path)
    return result


def upstream_focus_diff(local: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Compare local HEAD to the probed upstream HEAD for Eagle3 focus paths.

    This is intentionally non-fetching. If the upstream commit object is not
    already present locally, the result is marked unavailable and the caller can
    treat the global upstream drift conservatively.
    """
    path = Path(str(local.get("path") or ""))
    upstream_head = upstream.get("head")
    local_head = local.get("head")
    if not path.exists() or not local.get("is_git_repo"):
        return {"status": "unavailable", "reason": "local ModelOpt is not a git repo"}
    if not upstream_head:
        return {"status": "unavailable", "reason": "upstream head was not probed"}
    if local_head == upstream_head:
        return {"status": "same_head", "files": []}
    present = git(path, "cat-file", "-e", f"{upstream_head}^{{commit}}")
    if present["returncode"] != 0:
        return {
            "status": "unavailable",
            "reason": "upstream commit object is not present locally; fetch before comparing focus paths",
            "upstream_head": upstream_head,
        }
    diff = git(path, "diff", "--name-status", f"HEAD..{upstream_head}", "--", *FOCUS_PATHS)
    files = diff["stdout"].splitlines() if diff["stdout"] else []
    return {
        "status": "ok" if diff["returncode"] == 0 else "error",
        "files": files,
        "file_count": len(files),
        "upstream_head": upstream_head,
    }


def build_training_source_decision(
    local: dict[str, Any],
    upstream: dict[str, Any],
    hayate: dict[str, Any],
    upstream_focus: dict[str, Any],
) -> dict[str, Any]:
    local_head = local.get("head")
    upstream_head = upstream.get("head")
    upstream_probe_ok = upstream.get("status") == "ok"
    upstream_head_matches = bool(upstream_probe_ok and local_head and upstream_head and local_head == upstream_head)
    dirty = dirty_paths(local)
    focus_dirty = focus_dirty_paths(local)
    disallowed_focus_dirty = sorted(set(focus_dirty) - ALLOWED_QWEN3_FOCUS_DIFFS)
    allowed_focus_dirty = sorted(set(focus_dirty) & ALLOWED_QWEN3_FOCUS_DIFFS)
    unrelated_dirty = sorted(set(dirty) - set(focus_dirty))

    upstream_focus_status = upstream_focus.get("status")
    upstream_focus_files = upstream_focus.get("files") or []

    if not local.get("exists") or not local.get("is_git_repo"):
        status = "fail"
        summary = "local ModelOpt checkout is not usable as the training source"
    elif not upstream_probe_ok:
        status = "warn"
        summary = "official upstream could not be probed; use local focus-file checks only"
    elif not upstream_head_matches and upstream_focus_status == "ok" and not upstream_focus_files:
        status = "pass"
        summary = "official main advanced, but Eagle3/speculative-decoding focus paths are unchanged"
    elif not upstream_head_matches:
        status = "warn"
        summary = "local ModelOpt HEAD is not at official upstream main"
    elif disallowed_focus_dirty:
        status = "warn"
        summary = "speculative-decoding focus files include unclassified local edits"
    else:
        status = "pass"
        summary = "ModelOpt training source is official-upstream current with only the allowed Qwen3 loss-mask focus patch"

    return {
        "overall_status": status,
        "summary": summary,
        "source_of_truth": "local_modelopt",
        "upstream_probe_ok": upstream_probe_ok,
        "upstream_head_matches": upstream_head_matches,
        "upstream_focus_status": upstream_focus_status,
        "upstream_focus_diff_files": upstream_focus_files,
        "upstream_focus_diff_file_count": len(upstream_focus_files),
        "local_head": local_head,
        "upstream_head": upstream_head,
        "allowed_focus_diffs": allowed_focus_dirty,
        "disallowed_focus_diffs": disallowed_focus_dirty,
        "unrelated_dirty_files": unrelated_dirty,
        "unrelated_dirty_file_count": len(unrelated_dirty),
        "hayate_reference_only": True,
        "hayate_head": hayate.get("head"),
        "hayate_dirty_file_count": len(hayate.get("dirty_files") or []),
        "recommendation": (
            "Use the local/remote ModelOpt checkout as the training source. Treat Hayate's checkout as workflow "
            "reference only; port only intentional workflow ideas, not the older checkout wholesale."
        ),
    }


def probe_upstream(url: str, ref: str, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "skipped"}
    result = run(["git", "ls-remote", url, ref], timeout=30)
    if result["returncode"] != 0:
        return {
            "enabled": True,
            "status": "unavailable",
            "detail": result["stdout"],
            "url": url,
            "ref": ref,
        }
    line = first_line(result["stdout"]) or ""
    sha = line.split()[0] if line else None
    return {
        "enabled": True,
        "status": "ok" if sha else "missing_ref",
        "head": sha,
        "short_head": (sha or "")[:12],
        "url": url,
        "ref": ref,
    }


def determine_status(
    local: dict[str, Any],
    upstream: dict[str, Any],
    hayate: dict[str, Any],
    upstream_focus: dict[str, Any],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    status = "pass"
    if not local.get("exists") or not local.get("is_git_repo"):
        return "fail", ["local ModelOpt checkout is missing or not a git repository"]
    if local.get("dirty_files"):
        status = "warn"
        notes.append("local ModelOpt worktree has uncommitted changes")
    if upstream.get("status") == "ok":
        local_head = local.get("head")
        if local_head and local_head != upstream.get("head"):
            status = "warn"
            if upstream_focus.get("status") == "ok" and not (upstream_focus.get("files") or []):
                notes.append("local ModelOpt HEAD is behind official upstream main, but Eagle3 focus paths are unchanged")
            else:
                notes.append("local ModelOpt HEAD does not match probed official upstream main")
    elif upstream.get("enabled"):
        status = "warn"
        notes.append("official upstream main could not be probed from this host")
    if not hayate.get("exists"):
        notes.append("Hayate ModelOpt path is not visible on this host")
    elif hayate.get("dirty_files"):
        status = "warn"
        notes.append("Hayate/Hiso ModelOpt checkout has uncommitted or untracked files")
    return status, notes


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    local = summarize_repo(args.modelopt_dir, "local_modelopt", FOCUS_PATHS)
    hayate = summarize_repo(args.hayate_modelopt_dir, "hayate_modelopt", FOCUS_PATHS)
    upstream = probe_upstream(args.upstream_url, args.upstream_ref, args.probe_upstream)
    upstream_focus = upstream_focus_diff(local, upstream)
    status, notes = determine_status(local, upstream, hayate, upstream_focus)
    decision = build_training_source_decision(local, upstream, hayate, upstream_focus)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "decision": decision,
        "notes": notes,
        "official_example_url": "https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/speculative_decoding",
        "focus_paths": FOCUS_PATHS,
        "local": local,
        "upstream_probe": upstream,
        "upstream_focus_diff": upstream_focus,
        "hayate": hayate,
        "local_vs_hayate_focus_files": compare_focus_files(local, hayate) if hayate.get("exists") else [],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    local = payload["local"]
    upstream = payload["upstream_probe"]
    hayate = payload["hayate"]
    lines = [
        "# ModelOpt Upstream Drift Report",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Official example: {payload['official_example_url']}",
        "",
        "## Summary",
        "",
        f"- Local ModelOpt: `{local.get('path')}`",
        f"- Local branch/head: `{local.get('branch')}` / `{local.get('short_head')}`",
        f"- Dirty files: `{len(local.get('dirty_files') or [])}`",
        f"- Upstream probe: `{upstream.get('status')}` / `{upstream.get('short_head')}`",
        f"- Training source decision: `{(payload.get('decision') or {}).get('overall_status')}`",
        f"- Training source summary: {(payload.get('decision') or {}).get('summary')}",
        f"- Hayate ModelOpt visible: `{hayate.get('exists')}`",
        f"- Hayate branch/head: `{hayate.get('branch')}` / `{hayate.get('short_head')}`",
        f"- Hayate dirty/untracked files: `{len(hayate.get('dirty_files') or [])}`",
    ]
    decision = payload.get("decision") or {}
    if decision:
        lines += [
            "",
            "## Training Source Decision",
            "",
            f"- Status: `{decision.get('overall_status')}`",
            f"- Upstream HEAD matches local: `{decision.get('upstream_head_matches')}`",
            f"- Allowed focus diffs: `{', '.join(decision.get('allowed_focus_diffs') or []) or '-'}`",
            f"- Disallowed focus diffs: `{', '.join(decision.get('disallowed_focus_diffs') or []) or '-'}`",
            f"- Unrelated dirty files: `{decision.get('unrelated_dirty_file_count')}`",
            f"- Hayate reference only: `{decision.get('hayate_reference_only')}`",
            f"- Recommendation: {decision.get('recommendation')}",
        ]
    if payload.get("notes"):
        lines += ["", "## Notes"]
        lines.extend(f"- {note}" for note in payload["notes"])
    if local.get("dirty_files"):
        lines += ["", "## Local Dirty Files", ""]
        lines.extend(f"- `{item}`" for item in local["dirty_files"])
    if local.get("focus_diff_stat"):
        lines += ["", "## Local Focus Diff Stat", "", "```text", local["focus_diff_stat"], "```"]
    if hayate.get("dirty_files"):
        lines += ["", "## Hayate Dirty/Untracked Files", ""]
        lines.extend(f"- `{item}`" for item in hayate["dirty_files"][:40])
    comparisons = payload.get("local_vs_hayate_focus_files") or []
    if comparisons:
        lines += [
            "",
            "## Local vs Hayate Focus Files",
            "",
            "| path | local | hayate | same sha |",
            "| --- | --- | --- | --- |",
        ]
        for row in comparisons:
            lines.append(
                f"| `{row['path']}` | {row['local_exists']} | {row['hayate_exists']} | {row['same_sha256']} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(payload)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    write_outputs(payload, args)
    return 1 if payload["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
