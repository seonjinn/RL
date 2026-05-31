#!/usr/bin/env python3
"""Analyze accessible Hayate/Hiso ModelOpt Eagle3 workflow files.

The user-visible Hayate worktree path can move or be inaccessible from a given
host. This report focuses on the accessible TensorRT-Model-Optimizer checkout
and classifies whether its EAGLE3 additions are reusable for the Qwen3-235B
SWE/RL path or only useful as a reference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_HAYATE_MODELOPT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer"
)
FALLBACK_CANDIDATES = [
    DEFAULT_HAYATE_MODELOPT,
    Path("/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/Model-Optimizer"),
    Path(
        "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer-worktrees/eagle3"
    ),
]
EXP_DIR = Path(__file__).resolve().parent
BUNDLED_WORKFLOW_SNAPSHOT = EXP_DIR / "modelopt_upstream_drift_remote.json"

QWEN_CONFIG_GLOB = "examples/speculative_decoding/eagle_config_qwen3*.json"
KEY_FILES = [
    "examples/speculative_decoding/prepare_input_conversations/add_dapo17k.py",
    "examples/speculative_decoding/prepare_input_conversations/generate_responses.py",
    "examples/speculative_decoding/slurm/generate_responses.sbatch",
    "examples/speculative_decoding/slurm/train_eagle3.sbatch",
    "examples/speculative_decoding/slurm/submit_all.sh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hayate-modelopt-dir",
        type=Path,
        default=Path(os.environ.get("HAYATE_MODEL_OPT_DIR", DEFAULT_HAYATE_MODELOPT)),
    )
    parser.add_argument("--workflow-snapshot", type=Path, default=BUNDLED_WORKFLOW_SNAPSHOT)
    parser.add_argument(
        "--disable-bundled-fallback",
        action="store_true",
        help="Report missing_reference instead of using the repo-captured remote Hayate drift snapshot.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "cmd": cmd}
    except Exception as exc:
        return {"returncode": -1, "stdout": str(exc), "cmd": cmd}


def first_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def choose_path(requested: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = [requested]
    for candidate in FALLBACK_CANDIDATES:
        if candidate not in candidates:
            candidates.append(candidate)
    inspected = []
    for candidate in candidates:
        item = {"path": str(candidate), "exists": candidate.exists()}
        inspected.append(item)
        if candidate.exists():
            return candidate, inspected
    return requested, inspected


def read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def slurm_value(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(?:\$\{{[^:]+:-)?[\"']?([^\"'\n}}]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def sbatch_directive(text: str, key: str) -> str | None:
    match = re.search(rf"^#SBATCH\s+--{re.escape(key)}[=\s]+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def argparse_default(text: str, flag: str) -> str | int | float | None:
    match = re.search(rf"parser\.add_argument\(\s*[\"']{re.escape(flag)}[\"'].*?default=([^,\n)]+)", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip().strip("\"'")
    if raw == "None":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def inspect_qwen_configs(root: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for path in sorted(root.glob(QWEN_CONFIG_GLOB)):
        item: dict[str, Any] = {"path": str(path.relative_to(root))}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            item["status"] = "parsed"
            item["fields"] = {
                key: payload.get(key)
                for key in [
                    "num_hidden_layers",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "rope_theta",
                    "use_aux_hidden_state",
                    "has_lm_head",
                ]
            }
        except Exception as exc:
            item["status"] = "invalid"
            item["error"] = str(exc)
        configs.append(item)
    return configs


def inspect_slurm(root: Path) -> dict[str, Any]:
    generate = read_text(root / "examples/speculative_decoding/slurm/generate_responses.sbatch")
    train = read_text(root / "examples/speculative_decoding/slurm/train_eagle3.sbatch")
    submit_all = read_text(root / "examples/speculative_decoding/slurm/submit_all.sh")
    models = sorted(set(re.findall(r"Qwen/Qwen3-[A-Za-z0-9.\-]+", submit_all)))
    return {
        "generate_responses": {
            "exists": bool(generate),
            "account": sbatch_directive(generate, "account"),
            "partition": sbatch_directive(generate, "partition"),
            "gpus": sbatch_directive(generate, "gres"),
            "container": slurm_value(generate, "CONTAINER"),
            "default_model": slurm_value(generate, "MODEL"),
            "tp_size": slurm_value(generate, "TP_SIZE"),
            "max_new_tokens": slurm_value(generate, "MAX_NEW_TOKENS"),
            "uses_vllm": "--use-vllm" in generate,
        },
        "train_eagle3": {
            "exists": bool(train),
            "account": sbatch_directive(train, "account"),
            "partition": sbatch_directive(train, "partition"),
            "gpus": sbatch_directive(train, "gres"),
            "container": slurm_value(train, "CONTAINER"),
            "default_model": slurm_value(train, "MODEL"),
            "data_file": slurm_value(train, "DATA_FILE"),
            "num_epochs": slurm_value(train, "NUM_EPOCHS"),
            "training_seq_len": slurm_value(train, "TRAINING_SEQ_LEN"),
            "train_bs": slurm_value(train, "TRAIN_BS"),
            "lr": slurm_value(train, "LR"),
            "uses_legacy_main_cli": "main.py \\" in train and "--mode eagle3" in train,
            "uses_recipe_launch_train": "launch_train.sh" in train and "--config" in train,
            "runs_ar_validate": "scripts/ar_validate.py" in train,
            "exports_hf": "scripts/export_hf_checkpoint.py" in train,
        },
        "submit_all": {
            "exists": bool(submit_all),
            "models": models,
            "prepares_dapo17k": "add_dapo17k.py" in submit_all,
            "chains_generation_to_training": "--dependency=afterok:$GEN_JOB" in submit_all,
        },
    }


def inspect_data_scripts(root: Path) -> dict[str, Any]:
    add_dapo = read_text(root / KEY_FILES[0])
    generate = read_text(root / KEY_FILES[1])
    return {
        "add_dapo17k": {
            "exists": bool(add_dapo),
            "dataset": "BytedTsinghua-SIA/DAPO-Math-17k" if "BytedTsinghua-SIA/DAPO-Math-17k" in add_dapo else None,
            "outputs_prompts_only": "ground_truth" in add_dapo and "generate_responses.py" in add_dapo,
        },
        "generate_responses": {
            "exists": bool(generate),
            "uses_hf": "AutoModelForCausalLM" in generate,
            "uses_vllm": "from vllm import LLM" in generate,
            "default_max_model_len": int(re.search(r"max_model_len\s*=\s*(\d+)", generate).group(1))
            if re.search(r"max_model_len\s*=\s*(\d+)", generate)
            else None,
            "default_max_new_tokens": argparse_default(generate, "--max-new-tokens"),
            "default_temperature": argparse_default(generate, "--temperature"),
            "default_top_p": argparse_default(generate, "--top-p"),
            "default_batch_size": argparse_default(generate, "--batch-size"),
            "writes_conversations": '"conversations"' in generate or "'conversations'" in generate,
        },
    }


def inspect_logs(root: Path) -> dict[str, Any]:
    log_dir = root / "examples/speculative_decoding/logs"
    files = sorted(log_dir.glob("*")) if log_dir.exists() else []
    latest = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    return {
        "exists": log_dir.exists(),
        "file_count": len(files),
        "latest": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(path.stat().st_mtime)),
            }
            for path in latest
        ],
    }


def classify(payload: dict[str, Any]) -> dict[str, str]:
    if not payload["path"]["exists"]:
        return {
            "overall_status": "missing_reference",
            "classification": "not_accessible",
            "detail": "Hayate/Hiso ModelOpt checkout is not visible from this host",
        }

    slurm = payload["slurm"]["train_eagle3"]
    submit_models = payload["slurm"]["submit_all"]["models"]
    has_235b = any("235B" in model for model in submit_models)
    dapo = payload["data_scripts"]["add_dapo17k"].get("dataset")
    legacy = slurm.get("uses_legacy_main_cli") and not slurm.get("uses_recipe_launch_train")

    if dapo and not has_235b and legacy:
        return {
            "overall_status": "reference_only",
            "classification": "math_dapo_legacy_modelopt_workflow",
            "detail": (
                "Accessible Hayate/Hiso files describe DAPO-Math response generation and legacy "
                "Eagle3 training for Qwen3 8B/30B/32B, not Qwen3-235B SWE/RL rollout training."
            ),
        }
    if dapo:
        return {
            "overall_status": "reference_only",
            "classification": "math_bootstrap_workflow",
            "detail": "Workflow is math-data oriented and should be supplemental for SWE/RL.",
        }
    return {
        "overall_status": "needs_review",
        "classification": "unclassified_hayate_workflow",
        "detail": "Accessible files do not match the expected DAPO/Qwen3 legacy pattern; inspect manually.",
    }


def present_focus_files(snapshot: dict[str, Any], paths: list[str]) -> list[dict[str, Any]]:
    hayate = snapshot.get("hayate") if isinstance(snapshot.get("hayate"), dict) else {}
    focus = hayate.get("focus_files") if isinstance(hayate.get("focus_files"), dict) else {}
    result = []
    for rel in paths:
        record = focus.get(rel) if isinstance(focus.get(rel), dict) else {}
        result.append(
            {
                "path": rel,
                "exists": bool(record.get("exists")),
                "size_bytes": record.get("size_bytes"),
                "sha256": record.get("sha256"),
            }
        )
    return result


def build_snapshot_payload(args: argparse.Namespace, inspected: list[dict[str, Any]], snapshot: dict[str, Any]) -> dict[str, Any]:
    hayate = snapshot.get("hayate") if isinstance(snapshot.get("hayate"), dict) else {}
    focus = hayate.get("focus_files") if isinstance(hayate.get("focus_files"), dict) else {}
    qwen_config_paths = [
        "examples/speculative_decoding/eagle_config_qwen3_8b.json",
        "examples/speculative_decoding/eagle_config_qwen3_30b_moe.json",
        "examples/speculative_decoding/eagle_config_qwen3_32b.json",
    ]
    qwen_configs = [
        {
            "path": rel,
            "status": "present_in_snapshot",
            "fields": {},
            "size_bytes": focus.get(rel, {}).get("size_bytes") if isinstance(focus.get(rel), dict) else None,
            "sha256": focus.get(rel, {}).get("sha256") if isinstance(focus.get(rel), dict) else None,
        }
        for rel in qwen_config_paths
        if isinstance(focus.get(rel), dict) and focus[rel].get("exists")
    ]
    workflow_files = present_focus_files(snapshot, KEY_FILES)
    path_payload: dict[str, Any] = {
        "requested": str(args.hayate_modelopt_dir),
        "chosen": str(hayate.get("path") or args.hayate_modelopt_dir),
        "exists": False,
        "candidates": inspected,
    }
    data_scripts = {
        "add_dapo17k": {
            "exists": any(item["path"].endswith("add_dapo17k.py") and item["exists"] for item in workflow_files),
            "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
            "outputs_prompts_only": None,
            "source": "bundled drift snapshot proves file presence; dataset classification is from the repo README's captured Hayate findings",
        },
        "generate_responses": {
            "exists": any(item["path"].endswith("generate_responses.py") and item["exists"] for item in workflow_files),
            "uses_hf": None,
            "uses_vllm": None,
            "default_max_model_len": None,
            "default_max_new_tokens": None,
            "default_temperature": None,
            "default_top_p": None,
            "default_batch_size": None,
            "writes_conversations": None,
            "source": "bundled drift snapshot proves file presence; content-level defaults require live Hayate path",
        },
    }
    slurm = {
        "generate_responses": {
            "exists": any(item["path"].endswith("generate_responses.sbatch") and item["exists"] for item in workflow_files),
            "account": None,
            "partition": None,
            "gpus": None,
            "container": None,
            "default_model": None,
            "tp_size": None,
            "max_new_tokens": None,
            "uses_vllm": None,
        },
        "train_eagle3": {
            "exists": any(item["path"].endswith("train_eagle3.sbatch") and item["exists"] for item in workflow_files),
            "account": None,
            "partition": None,
            "gpus": None,
            "container": None,
            "default_model": None,
            "data_file": None,
            "num_epochs": None,
            "training_seq_len": None,
            "train_bs": None,
            "lr": None,
            "uses_legacy_main_cli": "documented_in_readme_snapshot",
            "uses_recipe_launch_train": None,
            "runs_ar_validate": "documented_in_readme_snapshot",
            "exports_hf": "documented_in_readme_snapshot",
            "source": "bundled drift snapshot proves slurm wrapper presence; CLI defaults require live Hayate path",
        },
        "submit_all": {
            "exists": any(item["path"].endswith("submit_all.sh") and item["exists"] for item in workflow_files),
            "models": ["Qwen3-8B", "Qwen3-30B-A3B", "Qwen3-32B"],
            "prepares_dapo17k": data_scripts["add_dapo17k"]["exists"],
            "chains_generation_to_training": None,
        },
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "reference_only",
        "classification": "math_dapo_legacy_modelopt_workflow_snapshot",
        "detail": (
            "Live Hayate/Hiso ModelOpt checkout is not visible from this host. "
            "Using the bundled remote drift snapshot plus README-captured classification: "
            "DAPO-Math bootstrap files and Qwen3 8B/30B/32B Eagle3 workflow artifacts were visible, "
            "so this is reference evidence rather than a drop-in Qwen3-235B SWE/RL recipe."
        ),
        "source": "bundled_remote_drift_snapshot",
        "live_hayate_visible": False,
        "snapshot": {
            "path": str(args.workflow_snapshot),
            "generated_at": snapshot.get("generated_at"),
            "hayate_path": hayate.get("path"),
            "hayate_head": hayate.get("head"),
            "hayate_short_head": hayate.get("short_head"),
            "hayate_exists_at_snapshot": hayate.get("exists"),
        },
        "path": path_payload,
        "inspected_paths": inspected,
        "selected_path": str(hayate.get("path") or args.hayate_modelopt_dir),
        "git": {
            "is_git_repo": hayate.get("is_git_repo"),
            "dirty_files": hayate.get("dirty_files") or [],
            "head": hayate.get("head"),
            "date": None,
            "author": None,
            "subject": None,
        },
        "qwen_configs": qwen_configs,
        "data_scripts": data_scripts,
        "slurm": slurm,
        "logs": {"exists": None, "file_count": None, "latest": []},
        "key_files": workflow_files + present_focus_files(snapshot, qwen_config_paths),
        "classification_basis": {
            "workflow_files_present": [item["path"] for item in workflow_files if item["exists"]],
            "qwen_config_paths_present": [item["path"] for item in qwen_configs],
            "hayate_dirty_files": hayate.get("dirty_files") or [],
            "readme_capture": "README Hayate/Hiso Findings classify this as DAPO-Math response generation plus legacy Qwen3 8B/30B/32B Eagle3 training.",
        },
        "warnings": [
            "This fallback does not prove current live access to Hayate/Hiso ModelOpt; remote path evidence remains a separate gate.",
            "Content-level Slurm defaults should be refreshed from the live checkout once remote access recovers.",
        ],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    chosen, inspected = choose_path(args.hayate_modelopt_dir)
    path_payload: dict[str, Any] = {
        "requested": str(args.hayate_modelopt_dir),
        "chosen": str(chosen),
        "exists": chosen.exists(),
        "candidates": inspected,
    }
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "path": path_payload,
        "inspected_paths": inspected,
        "selected_path": str(chosen),
        "git": {},
        "qwen_configs": [],
        "data_scripts": {},
        "slurm": {},
        "logs": {},
        "key_files": [],
    }
    if not chosen.exists():
        if not args.disable_bundled_fallback:
            snapshot = load_json(args.workflow_snapshot)
            hayate = snapshot.get("hayate") if isinstance((snapshot or {}).get("hayate"), dict) else {}
            if snapshot and hayate.get("exists"):
                return build_snapshot_payload(args, inspected, snapshot)
        decision = classify(payload)
        payload["source"] = "missing_reference"
        payload["live_hayate_visible"] = False
        payload.update(decision)
        return payload

    inside = run(["git", "-C", str(chosen), "rev-parse", "--is-inside-work-tree"])
    status = run(["git", "-C", str(chosen), "status", "--short"])
    head = run(["git", "-C", str(chosen), "log", "-1", "--date=iso", "--pretty=%H%n%ad%n%an%n%s"])
    head_lines = head["stdout"].splitlines()
    payload["git"] = {
        "is_git_repo": inside["returncode"] == 0 and inside["stdout"] == "true",
        "dirty_files": status["stdout"].splitlines() if status["stdout"] else [],
        "head": head_lines[0] if len(head_lines) > 0 else None,
        "date": head_lines[1] if len(head_lines) > 1 else None,
        "author": head_lines[2] if len(head_lines) > 2 else None,
        "subject": head_lines[3] if len(head_lines) > 3 else None,
    }
    payload["qwen_configs"] = inspect_qwen_configs(chosen)
    payload["data_scripts"] = inspect_data_scripts(chosen)
    payload["slurm"] = inspect_slurm(chosen)
    payload["logs"] = inspect_logs(chosen)
    payload["key_files"] = [
        {"path": rel, "exists": (chosen / rel).exists(), "size_bytes": (chosen / rel).stat().st_size if (chosen / rel).exists() else None}
        for rel in [*KEY_FILES, *[str(p.relative_to(chosen)) for p in sorted(chosen.glob(QWEN_CONFIG_GLOB))]]
    ]
    payload["source"] = "live_hayate_modelopt_path"
    payload["live_hayate_visible"] = True
    payload.update(classify(payload))
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hayate/Hiso ModelOpt Workflow Report",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Classification: `{payload['classification']}`",
        f"Source: `{payload.get('source', 'unknown')}`",
        f"Live Hayate visible: `{payload.get('live_hayate_visible')}`",
        f"Generated: `{payload['generated_at']}`",
        "",
        payload["detail"],
        "",
        "## Path",
        "",
        f"- requested: `{payload['path']['requested']}`",
        f"- chosen: `{payload['path']['chosen']}`",
        f"- exists: `{payload['path']['exists']}`",
    ]
    if payload.get("snapshot"):
        snap = payload["snapshot"]
        lines += [
            f"- snapshot: `{snap.get('path')}`",
            f"- snapshot generated: `{snap.get('generated_at')}`",
            f"- Hayate existed at snapshot: `{snap.get('hayate_exists_at_snapshot')}`",
        ]
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in payload["warnings"])
    if not payload["path"]["exists"] and not payload.get("qwen_configs") and not payload.get("data_scripts"):
        return "\n".join(lines) + "\n"

    git = payload["git"]
    lines += [
        "",
        "## Git",
        "",
        f"- head: `{str(git.get('head') or '')[:12]}`",
        f"- date: `{git.get('date')}`",
        f"- subject: `{git.get('subject')}`",
        f"- dirty/untracked files: `{len(git.get('dirty_files') or [])}`",
    ]
    if git.get("dirty_files"):
        lines += ["", "Dirty/untracked highlights:"]
        lines.extend(f"- `{item}`" for item in git["dirty_files"][:20])

    lines += [
        "",
        "## Workflow Shape",
        "",
        f"- data source: `{payload['data_scripts'].get('add_dapo17k', {}).get('dataset')}`",
        f"- response generation uses vLLM: `{payload['data_scripts'].get('generate_responses', {}).get('uses_vllm')}`",
        f"- training CLI is legacy `main.py --mode eagle3`: `{payload['slurm'].get('train_eagle3', {}).get('uses_legacy_main_cli')}`",
        f"- training CLI uses current recipe launcher: `{payload['slurm'].get('train_eagle3', {}).get('uses_recipe_launch_train')}`",
        f"- submit_all models: `{', '.join(payload['slurm'].get('submit_all', {}).get('models') or [])}`",
        f"- log files visible: `{payload['logs'].get('file_count')}`",
        "",
        "## Operational Defaults",
        "",
        "| area | value |",
        "| --- | --- |",
        f"| generation account/partition | `{payload['slurm'].get('generate_responses', {}).get('account')}` / `{payload['slurm'].get('generate_responses', {}).get('partition')}` |",
        f"| generation container | `{payload['slurm'].get('generate_responses', {}).get('container')}` |",
        f"| generation TP / max new tokens | `{payload['slurm'].get('generate_responses', {}).get('tp_size')}` / `{payload['slurm'].get('generate_responses', {}).get('max_new_tokens')}` |",
        f"| generation max model len | `{payload['data_scripts'].get('generate_responses', {}).get('default_max_model_len')}` |",
        f"| generation sampling | `temperature={payload['data_scripts'].get('generate_responses', {}).get('default_temperature')}`, `top_p={payload['data_scripts'].get('generate_responses', {}).get('default_top_p')}` |",
        f"| training account/partition | `{payload['slurm'].get('train_eagle3', {}).get('account')}` / `{payload['slurm'].get('train_eagle3', {}).get('partition')}` |",
        f"| training container | `{payload['slurm'].get('train_eagle3', {}).get('container')}` |",
        f"| training epochs / seq len / batch / lr | `{payload['slurm'].get('train_eagle3', {}).get('num_epochs')}` / `{payload['slurm'].get('train_eagle3', {}).get('training_seq_len')}` / `{payload['slurm'].get('train_eagle3', {}).get('train_bs')}` / `{payload['slurm'].get('train_eagle3', {}).get('lr')}` |",
        f"| post-train validation/export | `ar_validate={payload['slurm'].get('train_eagle3', {}).get('runs_ar_validate')}`, `export_hf={payload['slurm'].get('train_eagle3', {}).get('exports_hf')}` |",
        "",
        "## Qwen Configs",
        "",
        "| file | heads | kv heads | ffn | rope theta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cfg in payload["qwen_configs"]:
        fields = cfg.get("fields") or {}
        lines.append(
            f"| `{cfg['path']}` | {fields.get('num_attention_heads')} | {fields.get('num_key_value_heads')} | "
            f"{fields.get('intermediate_size')} | {fields.get('rope_theta')} |"
        )

    lines += [
        "",
        "## Implication For Qwen3-235B SWE/RL",
        "",
        "- Treat this as evidence of the other team's math-target bootstrap flow, not as a drop-in Qwen3-235B recipe.",
        "- The reusable part is the high-level sequence: prepare prompts, generate target-model responses, train Eagle3, export, validate acceptance rate.",
        "- For this workstream, replace DAPO prompts with actual NeMo-RL rollout conversations and use the current ModelOpt recipe API plus the Qwen3-235B derived architecture.",
        "- Do not copy the hard-coded Hiso containers/accounts directly; use the proven `nemo_25.07.01.sqsh` runtime and `coreai_dlalgo_nemorl` account on oci-hsg.",
        "- The Hayate generation/training lengths (`max_model_len=4096`, `TRAINING_SEQ_LEN=2048`) are too short for the current SWE/RL 16k-token target, so they are reference defaults only.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(payload)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
