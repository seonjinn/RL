#!/usr/bin/env python3
"""Export the local ModelOpt Eagle3 patch as a handoff bundle.

The Qwen3 RL path currently depends on a small local ModelOpt patch that makes
the TRT-LLM hidden-state dumper save `loss_mask` for answer-only offline
training. This script turns that dirty worktree delta into explicit artifacts
that can be carried to a cluster checkout or re-applied after updating
ModelOpt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELOPT = ROOT / "Model-Optimizer"
DEFAULT_OUT_DIR = ROOT / "outputs/qwen3_235b_eagle3/patches/modelopt_eagle3_qwen3"

PATCH_PATHS = [
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
]
SNAPSHOT_PATHS = [
    "examples/speculative_decoding/collect_hidden_states/common.py",
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
]
COMMON_REQUIRED_SYMBOLS = [
    "add_answer_only_loss_args",
    "load_chat_template",
    "verify_generation_tags",
    "tokenize_with_loss_mask",
]
TRTLLM_REQUIRED_SNIPPETS = [
    "add_answer_only_loss_args(parser)",
    "verify_generation_tags(tokenizer.chat_template)",
    'trtllm_dumped["loss_mask"] = loss_mask.cpu()',
    "tokenize_with_loss_mask(",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelopt-dir", type=Path, default=DEFAULT_MODELOPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--compat-modelopt-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional target ModelOpt checkout to test with this patch. "
            "Can be passed multiple times; the target worktree is not modified."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, input_text: str | None = None) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "cmd": cmd}


def git(repo: Path, *args: str, input_text: str | None = None) -> dict[str, Any]:
    return run(["git", "-C", str(repo), *args], input_text=input_text)


def first_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def symbol_check(path: Path, symbols: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "missing": symbols, "status": "fail"}
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [symbol for symbol in symbols if symbol not in text]
    return {
        "exists": True,
        "missing": missing,
        "status": "pass" if not missing else "fail",
    }


def copy_snapshots(modelopt: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshot_root = out_dir / "snapshots"
    for rel in SNAPSHOT_PATHS:
        src = modelopt / rel
        dest = snapshot_root / rel
        row = {"path": rel, "source_exists": src.exists()}
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            row["bundle_path"] = str(dest)
            row["sha256"] = file_sha256(dest)
        rows.append(row)
    return rows


def compat_check(target: Path, patch_text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
    }
    if not target.exists():
        payload["status"] = "missing"
        return payload
    inside = git(target, "rev-parse", "--is-inside-work-tree")
    payload["is_git_repo"] = inside["returncode"] == 0 and inside["stdout"].strip() == "true"
    if not payload["is_git_repo"]:
        payload["status"] = "not_git_repo"
        return payload
    branch = git(target, "branch", "--show-current")
    head = git(target, "rev-parse", "HEAD")
    status = git(target, "status", "--porcelain")
    apply_check = git(target, "apply", "--check", "-", input_text=patch_text) if patch_text else {"returncode": 1, "stdout": "empty patch"}
    reverse_check = git(target, "apply", "--reverse", "--check", "-", input_text=patch_text) if patch_text else {"returncode": 1, "stdout": "empty patch"}
    common_check = symbol_check(target / SNAPSHOT_PATHS[0], COMMON_REQUIRED_SYMBOLS)
    trtllm_check = symbol_check(target / PATCH_PATHS[0], TRTLLM_REQUIRED_SNIPPETS)
    payload.update(
        {
            "branch": first_line(branch["stdout"]),
            "head": first_line(head["stdout"]),
            "short_head": (first_line(head["stdout"]) or "")[:12],
            "dirty_files": status["stdout"].splitlines() if status["stdout"] else [],
            "apply_check": {
                "status": "pass" if apply_check["returncode"] == 0 else "fail",
                "stdout": apply_check["stdout"],
            },
            "reverse_apply_check": {
                "status": "pass" if reverse_check["returncode"] == 0 else "fail",
                "stdout": reverse_check["stdout"],
            },
            "common_symbol_check": common_check,
            "trtllm_snippet_check": trtllm_check,
        }
    )
    if apply_check["returncode"] == 0 and common_check["status"] == "pass":
        payload["status"] = "compatible_clean"
    elif reverse_check["returncode"] == 0 and trtllm_check["status"] == "pass":
        payload["status"] = "already_applied"
    elif apply_check["returncode"] == 0:
        payload["status"] = "patch_applies_but_helpers_missing"
    else:
        payload["status"] = "incompatible"
    return payload


def render_readme(payload: dict[str, Any]) -> str:
    patch_name = Path(payload["patch_path"]).name
    return "\n".join(
        [
            "# ModelOpt Eagle3 Qwen3 Patch Bundle",
            "",
            "This bundle captures the local ModelOpt patch required for Qwen3-235B",
            "Eagle3 offline hidden-state dumping with answer-only `loss_mask`.",
            "",
            "Apply to a compatible clean ModelOpt checkout with:",
            "",
            "```bash",
            f"git -C /path/to/Model-Optimizer apply {patch_name}",
            "```",
            "",
            "Then rerun the Qwen3 Eagle3 preflight before submitting Slurm jobs.",
            "",
            f"Overall status: `{payload['overall_status']}`",
            f"Patch SHA256: `{payload.get('patch_sha256')}`",
            f"Local ModelOpt head: `{payload.get('local_head_short')}`",
            "",
        ]
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ModelOpt Eagle3 Patch Bundle",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"ModelOpt: `{payload['modelopt_dir']}`",
        f"Head: `{payload.get('local_head_short')}`",
        f"Patch: `{payload['patch_path']}`",
        f"Patch SHA256: `{payload.get('patch_sha256')}`",
        "",
        "## Checks",
        "",
        f"- Dirty patch present: `{payload['patch_nonempty']}`",
        f"- `git diff --check`: `{payload['diff_check']['status']}`",
        f"- reverse apply check: `{payload['reverse_apply_check']['status']}`",
        f"- common helpers: `{payload['common_symbol_check']['status']}`",
        f"- TRT-LLM snippets: `{payload['trtllm_snippet_check']['status']}`",
    ]
    if payload.get("compatibility_checks"):
        lines += ["", "## Compatibility Checks", "", "| target | head | status |", "| --- | --- | --- |"]
        for row in payload["compatibility_checks"]:
            lines.append(f"| `{row.get('path')}` | `{row.get('short_head')}` | `{row.get('status')}` |")
    if payload.get("dirty_files"):
        lines += ["", "## Dirty Files"]
        lines.extend(f"- `{item}`" for item in payload["dirty_files"])
    if payload.get("diff_stat"):
        lines += ["", "## Diff Stat", "", "```text", payload["diff_stat"], "```"]
    if payload["common_symbol_check"].get("missing"):
        lines += ["", "Missing common helpers:"]
        lines.extend(f"- `{item}`" for item in payload["common_symbol_check"]["missing"])
    return "\n".join(lines) + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    modelopt = args.modelopt_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    branch = git(modelopt, "branch", "--show-current")
    head = git(modelopt, "rev-parse", "HEAD")
    status = git(modelopt, "status", "--porcelain")
    diff = git(modelopt, "diff", "--", *PATCH_PATHS)
    diff_stat = git(modelopt, "diff", "--stat", "--", *PATCH_PATHS)
    diff_check = git(modelopt, "diff", "--check", "--", *PATCH_PATHS)
    patch_text = diff["stdout"]

    patch_path = out_dir / "modelopt_eagle3_qwen3.patch"
    patch_path.write_text(patch_text + ("\n" if patch_text and not patch_text.endswith("\n") else ""))

    reverse_apply = {"status": "skipped", "stdout": ""}
    if patch_text:
        reverse = git(modelopt, "apply", "--reverse", "--check", "-", input_text=patch_text)
        reverse_apply = {
            "status": "pass" if reverse["returncode"] == 0 else "fail",
            "stdout": reverse["stdout"],
        }

    common_check = symbol_check(modelopt / SNAPSHOT_PATHS[0], COMMON_REQUIRED_SYMBOLS)
    trtllm_check = symbol_check(modelopt / PATCH_PATHS[0], TRTLLM_REQUIRED_SNIPPETS)
    snapshots = copy_snapshots(modelopt, out_dir)
    compatibility_checks = [compat_check(path, patch_text) for path in args.compat_modelopt_dir]

    checks_pass = (
        bool(patch_text)
        and diff_check["returncode"] == 0
        and reverse_apply["status"] == "pass"
        and common_check["status"] == "pass"
        and trtllm_check["status"] == "pass"
        and all(
            item.get("status") in {"compatible_clean", "already_applied"}
            for item in compatibility_checks
        )
    )
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "pass" if checks_pass else "warn",
        "modelopt_dir": str(modelopt),
        "local_branch": first_line(branch["stdout"]),
        "local_head": first_line(head["stdout"]),
        "local_head_short": (first_line(head["stdout"]) or "")[:12],
        "dirty_files": status["stdout"].splitlines() if status["stdout"] else [],
        "patch_paths": PATCH_PATHS,
        "snapshot_paths": SNAPSHOT_PATHS,
        "patch_path": str(patch_path),
        "patch_nonempty": bool(patch_text),
        "patch_sha256": sha256_text(patch_text) if patch_text else None,
        "patch_file_sha256": file_sha256(patch_path),
        "diff_stat": diff_stat["stdout"],
        "diff_check": {
            "status": "pass" if diff_check["returncode"] == 0 else "fail",
            "stdout": diff_check["stdout"],
        },
        "reverse_apply_check": reverse_apply,
        "common_symbol_check": common_check,
        "trtllm_snippet_check": trtllm_check,
        "compatibility_checks": compatibility_checks,
        "snapshots": snapshots,
    }

    (out_dir / "README.md").write_text(render_readme(payload))
    return payload


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    json_out = args.json_out or (args.out_dir / "manifest.json")
    markdown_out = args.markdown_out or (args.out_dir / "patch_report.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(payload)
    markdown_out.write_text(markdown)
    print(markdown, end="")


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    write_outputs(payload, args)
    return 0 if payload["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
