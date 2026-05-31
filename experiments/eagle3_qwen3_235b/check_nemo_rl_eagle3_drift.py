#!/usr/bin/env python3
"""Report NeMo-RL Eagle3 support and upstream drift for the Qwen3 SWE path.

This check is deliberately read-only. It answers two questions:

1. Does the mounted SpecDec-RL/NeMo-RL checkout support fixed Eagle3 rollout
   acceleration through vLLM speculative_config?
2. Does it appear to support NeMo-RL online draft training, where the trainer
   owns and updates the draft model during RL?

The fixed-draft path is enough for the first Qwen3-235B Eagle3 validation. The
online path is a later stage and should not be attempted unless this report can
prove the required source markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEMO_RL_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)
DEFAULT_UPSTREAM_URL = "https://github.com/NVIDIA-NeMo/RL.git"
DEFAULT_UPSTREAM_REF = "refs/heads/main"
DEFAULT_RAW_BASE = "https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main"
DOC_URL = "https://docs.nvidia.com/nemo/rl/0.6.0/guides/eagle3-speculative-decoding.html"

FOCUS_PATHS = [
    "nemo_rl/models/generation/__init__.py",
    "nemo_rl/models/generation/vllm/utils.py",
    "nemo_rl/models/generation/vllm/vllm_worker.py",
    "nemo_rl/models/generation/vllm/vllm_generation.py",
    "nemo_rl/models/policy/__init__.py",
    "nemo_rl/models/policy/lm_policy.py",
    "nemo_rl/algorithms/grpo.py",
    "examples/configs/recipes/llm/grpo-qwen3-30ba3b-8n8g-megatron.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-rl-dir", type=Path, default=Path(os.environ.get("SPECDEC_RL_DIR", DEFAULT_NEMO_RL_DIR)))
    parser.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL)
    parser.add_argument("--upstream-ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument("--raw-base", default=DEFAULT_RAW_BASE)
    parser.add_argument("--probe-upstream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fetch-raw", action=argparse.BooleanOptionalAction, default=True)
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
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "cmd": cmd}
    except Exception as exc:
        return {"returncode": -1, "stdout": str(exc), "cmd": cmd}


def git(repo: Path, *args: str) -> dict[str, Any]:
    return run(["git", "-C", str(repo), *args])


def first_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def read_local(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        return "", {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, {"exists": True, "sha256": sha256_text(text), "size_bytes": path.stat().st_size}


def fetch_raw(raw_base: str, rel: str, enabled: bool) -> tuple[str, dict[str, Any]]:
    url = raw_base.rstrip("/") + "/" + rel
    if not enabled:
        return "", {"status": "skipped", "url": url}
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            text = response.read().decode("utf-8", errors="replace")
        return text, {"status": "fetched", "url": url, "sha256": sha256_text(text), "size_bytes": len(text)}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "", {"status": "fetch_failed", "url": url, "error": str(exc)}


def markers(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {
        "speculative_config": "speculative_config" in text,
        "load_format_auto": "load_format" in text and '"auto"' in text,
        "spec_decode_metrics": "aggregate_spec_decode_counters" in text and "spec_acceptance_rate" in text,
        "post_step_patch": "post_step" in text and "speculative" in lower and "vllm" in lower,
        "policy_draft_config": "policy.draft" in text or '["draft"]' in text or "['draft']" in text,
        "draft_loss_weight": "loss_weight" in text and "draft" in lower,
        "draft_model_name": "model_name" in text and "draft" in lower,
        "sequence_packing_constraint": "sequence_packing" in text and "draft" in lower,
    }


def summarize_files(root: Path, raw_base: str, fetch_raw_enabled: bool) -> dict[str, Any]:
    local_files: dict[str, Any] = {}
    upstream_files: dict[str, Any] = {}
    local_combined = ""
    upstream_combined = ""
    for rel in FOCUS_PATHS:
        text, meta = read_local(root / rel)
        meta["markers"] = markers(text)
        local_files[rel] = meta
        local_combined += "\n" + text

        raw_text, raw_meta = fetch_raw(raw_base, rel, fetch_raw_enabled)
        raw_meta["markers"] = markers(raw_text)
        upstream_files[rel] = raw_meta
        upstream_combined += "\n" + raw_text
    return {
        "local_files": local_files,
        "upstream_files": upstream_files,
        "local_markers": markers(local_combined),
        "upstream_markers": markers(upstream_combined),
    }


def repo_summary(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return payload
    inside = git(path, "rev-parse", "--is-inside-work-tree")
    payload["is_git_repo"] = inside["returncode"] == 0 and inside["stdout"] == "true"
    if not payload["is_git_repo"]:
        return payload
    head = git(path, "rev-parse", "HEAD")
    branch = git(path, "branch", "--show-current")
    status = git(path, "status", "--porcelain")
    remotes = git(path, "remote", "-v")
    payload.update(
        {
            "head": first_line(head["stdout"]),
            "short_head": (first_line(head["stdout"]) or "")[:12],
            "branch": first_line(branch["stdout"]),
            "dirty_files": status["stdout"].splitlines() if status["stdout"] else [],
            "remote_lines": remotes["stdout"].splitlines() if remotes["stdout"] else [],
        }
    )
    return payload


def probe_upstream(url: str, ref: str, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "skipped", "url": url, "ref": ref}
    result = run(["git", "ls-remote", url, ref], timeout=30)
    if result["returncode"] != 0:
        return {"enabled": True, "status": "unavailable", "url": url, "ref": ref, "detail": result["stdout"]}
    line = first_line(result["stdout"]) or ""
    head = line.split()[0] if line else None
    return {
        "enabled": True,
        "status": "ok" if head else "missing_ref",
        "url": url,
        "ref": ref,
        "head": head,
        "short_head": (head or "")[:12],
    }


def support(local_markers: dict[str, bool]) -> dict[str, Any]:
    fixed_checks = {
        "load_format_auto": local_markers["speculative_config"] and local_markers["load_format_auto"],
        "spec_decode_metrics": local_markers["spec_decode_metrics"],
        "post_step_patch": local_markers["post_step_patch"],
    }
    online_checks = {
        "policy_draft_config": local_markers["policy_draft_config"],
        "draft_loss_weight": local_markers["draft_loss_weight"],
        "draft_model_name": local_markers["draft_model_name"],
    }
    fixed_ok = all(fixed_checks.values())
    online_ok = all(online_checks.values())
    return {
        "fixed_generation": {
            "status": "pass" if fixed_ok else "incomplete",
            "checks": fixed_checks,
        },
        "online_draft_training": {
            "status": "pass" if online_ok else "missing_source_support",
            "checks": online_checks,
            "required_config_overrides": [
                "policy.megatron_cfg.enabled=true",
                "policy.dtensor_cfg.enabled=false",
                "policy.sequence_packing.enabled=false",
                "++policy.draft.enabled=true",
                "++policy.draft.model_name=<draft>",
                "++policy.draft.loss_weight=<weight>",
            ],
        },
    }


def decide(repo: dict[str, Any], upstream: dict[str, Any], file_summary: dict[str, Any], support_summary: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    status = "pass"
    if not repo.get("exists") or not repo.get("is_git_repo"):
        return "incomplete", ["NeMo-RL checkout is missing or not a git repository; fixed-draft-first route is recorded, but local source support is unproven"]
    if repo.get("dirty_files"):
        status = "warn"
        notes.append("NeMo-RL checkout has dirty or untracked files")
    if upstream.get("status") == "ok" and repo.get("head") and repo.get("head") != upstream.get("head"):
        status = "warn"
        notes.append("NeMo-RL checkout HEAD differs from official upstream main")
    elif upstream.get("enabled") and upstream.get("status") != "ok":
        status = "warn"
        notes.append("official NeMo-RL upstream could not be probed")

    if support_summary["fixed_generation"]["status"] != "pass":
        status = "fail"
        notes.append("fixed Eagle3 rollout support markers are incomplete")
    if support_summary["online_draft_training"]["status"] != "pass":
        if status == "pass":
            status = "warn"
        notes.append("online draft training support markers are missing in this checkout")

    upstream_online = file_summary["upstream_markers"]["policy_draft_config"] and file_summary["upstream_markers"]["draft_loss_weight"]
    local_online = support_summary["online_draft_training"]["status"] == "pass"
    if upstream_online and not local_online:
        status = "warn" if status != "fail" else status
        notes.append("official upstream raw files appear to contain online-draft markers that local checkout lacks")
    return status, notes


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    repo = repo_summary(args.nemo_rl_dir)
    upstream = probe_upstream(args.upstream_url, args.upstream_ref, args.probe_upstream)
    file_summary = summarize_files(args.nemo_rl_dir, args.raw_base, args.fetch_raw)
    support_summary = support(file_summary["local_markers"])
    status, notes = decide(repo, upstream, file_summary, support_summary)
    primary_route = "fixed_exported_draft_generation_first"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "notes": notes,
        "docs_url": DOC_URL,
        "primary_route": primary_route,
        "repo": repo,
        "upstream": upstream,
        "support": support_summary,
        "markers": {
            "local": file_summary["local_markers"],
            "upstream_raw": file_summary["upstream_markers"],
        },
        "files": {
            "local": file_summary["local_files"],
            "upstream_raw": file_summary["upstream_files"],
        },
        "recommendation": recommendation(support_summary, repo),
    }


def recommendation(support_summary: dict[str, Any], repo: dict[str, Any]) -> str:
    if not repo.get("exists") or not repo.get("is_git_repo"):
        return "Use fixed exported Eagle3 draft generation first after the SpecDec-RL checkout is visible; do not attempt NeMo-RL online draft training until local source support is proven."
    if support_summary["fixed_generation"]["status"] != "pass":
        return "Keep the fixed exported Eagle3 draft route first, but update or patch NeMo-RL before attempting RL rollout validation."
    if support_summary["online_draft_training"]["status"] != "pass":
        return "Use fixed exported Eagle3 draft generation first; do not attempt NeMo-RL online draft training in this checkout."
    return "Fixed draft and online draft markers are present; still validate fixed draft speedup before enabling online draft training."


def render_markdown(data: dict[str, Any]) -> str:
    repo = data["repo"]
    upstream = data["upstream"]
    fixed = data["support"]["fixed_generation"]
    online = data["support"]["online_draft_training"]
    lines = [
        "# NeMo-RL Eagle3 Drift / Support",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        "",
        f"Checkout: `{repo.get('path')}`",
        f"Branch/head: `{repo.get('branch')}` / `{repo.get('short_head')}`",
        f"Official upstream: `{upstream.get('short_head')}` ({upstream.get('status')})",
        f"Docs: {data['docs_url']}",
        "",
        f"Recommendation: {data['recommendation']}",
        "",
        "| capability | status | checks |",
        "| --- | --- | --- |",
        f"| fixed Eagle3 rollout | {fixed['status']} | {json.dumps(fixed['checks'], sort_keys=True)} |",
        f"| online draft training | {online['status']} | {json.dumps(online['checks'], sort_keys=True)} |",
        "",
    ]
    if data["notes"]:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in data["notes"])
        lines.append("")
    lines.extend(
        [
            "## Local Dirty Files",
            "",
        ]
    )
    dirty = repo.get("dirty_files") or []
    if dirty:
        lines.extend(f"- `{item}`" for item in dirty[:80])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_outputs(data: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(data)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown)


def main() -> int:
    args = parse_args()
    data = build_payload(args)
    write_outputs(data, args)
    return 1 if data["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
