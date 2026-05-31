#!/usr/bin/env python3
"""Collect provenance for the Qwen3-235B Eagle3 workstream.

Run this on the host where the repo, ModelOpt checkout, Hayate/Hiso paths, and
artifact directories are visible. It records git state, key file hashes, visible
model/artifact paths, and existing Eagle3 draft configs without submitting jobs
or loading model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"

DEFAULT_HAYATE_MODEL_OPT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer"
)
DEFAULT_HAYATE_NEMO_RL = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/"
    "nemo-rl-internal-worktrees/feat-eagle3-online-specdec"
)
DEFAULT_HAYATE_DRAFT_MODELS = DEFAULT_HAYATE_NEMO_RL / "models"

CRITICAL_MODELOPT_FILES = [
    "modelopt_recipes/general/speculative_decoding/eagle3.yaml",
    "examples/speculative_decoding/launch_train.sh",
    "examples/speculative_decoding/main.py",
    "examples/speculative_decoding/eagle_utils.py",
    "examples/speculative_decoding/collect_hidden_states/common.py",
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py",
    "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
    "modelopt/recipe/config.py",
    "modelopt/torch/speculative/config.py",
    "modelopt/torch/speculative/utils.py",
    "modelopt/torch/speculative/plugins/hf_training_args.py",
    "modelopt/torch/speculative/plugins/hf_eagle.py",
    "modelopt/torch/speculative/plugins/modeling_fakebase.py",
]


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "outputs/qwen3_235b_eagle3")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--modelopt-dir", type=Path, default=Path(os.environ.get("MODELOPT_DIR", ROOT / "Model-Optimizer")))
    parser.add_argument("--hayate-modelopt-dir", type=Path, default=DEFAULT_HAYATE_MODEL_OPT)
    parser.add_argument("--hayate-nemo-rl-dir", type=Path, default=DEFAULT_HAYATE_NEMO_RL)
    parser.add_argument("--hayate-draft-models-dir", type=Path, default=DEFAULT_HAYATE_DRAFT_MODELS)
    parser.add_argument("--verifier-config-dir", type=Path, default=env_path("VERIFIER_CONFIG_DIR"))
    parser.add_argument("--input-data", type=Path, default=env_path("INPUT_DATA"))
    parser.add_argument("--hidden-states-dir", type=Path, default=env_path("HIDDEN_STATES_DIR"))
    parser.add_argument("--output-dir", type=Path, default=env_path("OUTPUT_DIR"))
    parser.add_argument("--export-dir", type=Path, default=env_path("EXPORT_DIR"))
    parser.add_argument("--vllm-draft-dir", type=Path, default=env_path("VLLM_DRAFT_DIR"))
    parser.add_argument("--extra-path", action="append", default=[], type=Path)
    parser.add_argument("--max-draft-configs", type=int, default=80)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 12) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": cmd,
            "returncode": result.returncode,
            "output": result.stdout[-12000:],
        }
    except Exception as exc:
        return {"command": cmd, "returncode": None, "error": str(exc)}


def sha256(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            total += len(chunk)
            h.update(chunk)
    return {"path": str(path), "bytes": total, "sha256": h.hexdigest()}


def safe_exists(path: Path) -> tuple[bool, str | None]:
    try:
        return path.exists(), None
    except OSError as exc:
        return False, str(exc)


def safe_is_file(path: Path) -> tuple[bool, str | None]:
    try:
        return path.is_file(), None
    except OSError as exc:
        return False, str(exc)


def safe_is_dir(path: Path) -> tuple[bool, str | None]:
    try:
        return path.is_dir(), None
    except OSError as exc:
        return False, str(exc)


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    label = str(path)
    if root is not None:
        try:
            label = str(path.resolve().relative_to(root.resolve()))
        except Exception:
            label = str(path)
    exists, exists_error = safe_exists(path)
    if not exists:
        record = {"path": label, "exists": False}
        if exists_error:
            record["error"] = exists_error
        return record
    is_file, file_error = safe_is_file(path)
    if not is_file:
        record = {"path": label, "exists": True, "type": "dir"}
        if file_error:
            record["error"] = file_error
        return record
    try:
        record = sha256(path)
    except OSError as exc:
        return {"path": label, "exists": True, "type": "file", "error": str(exc)}
    record["path"] = label
    record["exists"] = True
    return record


def git_info(path: Path, label: str) -> dict[str, Any]:
    exists, exists_error = safe_exists(path)
    info: dict[str, Any] = {"label": label, "path": str(path), "exists": exists}
    if exists_error:
        info["error"] = exists_error
    if not exists:
        return info
    is_dir, dir_error = safe_is_dir(path)
    if dir_error:
        info["error"] = dir_error
    if not is_dir:
        info["is_git_worktree"] = False
        return info
    inside = run(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"])
    info["is_git_worktree"] = inside.get("returncode") == 0 and inside.get("output", "").strip() == "true"
    if not info["is_git_worktree"]:
        return info
    for key, cmd in {
        "branch": ["git", "-C", str(path), "branch", "--show-current"],
        "head": ["git", "-C", str(path), "rev-parse", "HEAD"],
        "head_summary": ["git", "-C", str(path), "log", "-1", "--oneline", "--decorate"],
        "recent_log": ["git", "-C", str(path), "log", "-5", "--oneline", "--decorate"],
        "status_short": ["git", "-C", str(path), "status", "--short"],
        "diff_name_status": ["git", "-C", str(path), "diff", "--name-status", "HEAD", "--"],
        "diff_stat": ["git", "-C", str(path), "diff", "--stat", "HEAD", "--"],
    }.items():
        result = run(cmd)
        info[key] = result.get("output", "").strip() if result.get("returncode") == 0 else None
    return info


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def nested(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def config_summary(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    if cfg is None:
        return {"path": str(path), "status": "invalid_json"}
    transformer_cfg = cfg.get("transformer_layer_config") if isinstance(cfg.get("transformer_layer_config"), dict) else {}
    return {
        "path": str(path),
        "parent": str(path.parent),
        "status": "ok",
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "speculators_model_type": cfg.get("speculators_model_type"),
        "hidden_size": cfg.get("hidden_size", transformer_cfg.get("hidden_size")),
        "vocab_size": cfg.get("vocab_size", transformer_cfg.get("vocab_size")),
        "num_hidden_layers": cfg.get("num_hidden_layers", transformer_cfg.get("num_hidden_layers")),
        "num_attention_heads": cfg.get("num_attention_heads", transformer_cfg.get("num_attention_heads")),
        "num_key_value_heads": cfg.get("num_key_value_heads", transformer_cfg.get("num_key_value_heads")),
        "intermediate_size": cfg.get("intermediate_size", transformer_cfg.get("intermediate_size")),
        "rope_theta": cfg.get("rope_theta", transformer_cfg.get("rope_theta")),
        "aux_layers": cfg.get("eagle_aux_hidden_state_layer_ids")
        or nested(cfg, "eagle_config.eagle_aux_hidden_state_layer_ids"),
    }


def iter_configs(root: Path, max_items: int) -> list[Path]:
    exists, _ = safe_exists(root)
    if not exists:
        return []
    is_file, _ = safe_is_file(root)
    if is_file and root.name == "config.json":
        return [root]
    configs: list[Path] = []
    try:
        candidates = sorted(root.rglob("config.json"))
    except OSError:
        return []
    for path in candidates:
        is_config, _ = safe_is_file(path)
        if is_config:
            configs.append(path)
            if len(configs) >= max_items:
                break
    return configs


def path_snapshot(path: Path | None, label: str, sample_limit: int = 2000) -> dict[str, Any]:
    if path is None:
        return {"label": label, "path": None, "status": "not_provided"}
    exists, exists_error = safe_exists(path)
    item: dict[str, Any] = {"label": label, "path": str(path), "exists": exists}
    if exists_error:
        item["error"] = exists_error
    if not exists:
        return item
    is_file, file_error = safe_is_file(path)
    if file_error:
        item["error"] = file_error
    if is_file:
        try:
            item.update({"type": "file", "bytes": path.stat().st_size})
        except OSError as exc:
            item.update({"type": "file", "error": str(exc)})
        return item
    file_count_sampled = 0
    total_bytes_sampled = 0
    pt_count_sampled = 0
    truncated = False
    try:
        for item_path in path.rglob("*"):
            is_child_file, _ = safe_is_file(item_path)
            if not is_child_file:
                continue
            if file_count_sampled >= sample_limit:
                truncated = True
                break
            file_count_sampled += 1
            try:
                total_bytes_sampled += item_path.stat().st_size
            except OSError:
                pass
            if item_path.suffix == ".pt":
                pt_count_sampled += 1
    except OSError as exc:
        item["error"] = str(exc)
    try:
        safetensors_count = len(list(path.glob("*.safetensors")))
    except OSError:
        safetensors_count = 0
    config_exists, _ = safe_exists(path / "config.json")
    item.update(
        {
            "type": "dir",
            "file_count_sampled": file_count_sampled,
            "file_count_truncated": truncated,
            "total_bytes_sampled": total_bytes_sampled,
            "config_json": config_exists,
            "safetensors_count": safetensors_count,
            "pt_count_sampled": pt_count_sampled,
        }
    )
    return item


def critical_file_hashes(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(EXP.iterdir()):
        if path.name == "__pycache__" or not path.is_file():
            continue
        if path.suffix in {".py", ".sh", ".sbatch", ".yaml", ".json", ".html", ".md"}:
            records.append(file_record(path, ROOT))
    for rel in CRITICAL_MODELOPT_FILES:
        records.append(file_record(args.modelopt_dir / rel, args.modelopt_dir))
    return records


def collect(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root
    repos = [
        git_info(args.repo_root, "local_repo"),
        git_info(args.modelopt_dir, "local_modelopt"),
        git_info(args.hayate_modelopt_dir, "hayate_modelopt"),
        git_info(args.hayate_nemo_rl_dir, "hayate_nemo_rl"),
    ]
    paths = [
        path_snapshot(artifact_root, "artifact_root"),
        path_snapshot(args.verifier_config_dir, "verifier_config_dir"),
        path_snapshot(args.input_data, "input_data"),
        path_snapshot(args.hidden_states_dir, "hidden_states_dir"),
        path_snapshot(args.output_dir, "modelopt_output_dir"),
        path_snapshot(args.export_dir, "export_dir"),
        path_snapshot(args.vllm_draft_dir, "vllm_draft_dir"),
        path_snapshot(args.hayate_draft_models_dir, "hayate_draft_models_dir"),
    ]
    for idx, extra in enumerate(args.extra_path, 1):
        paths.append(path_snapshot(extra, f"extra_path_{idx}"))

    visible_configs = []
    for label, root in {
        "verifier": args.verifier_config_dir,
        "export": args.export_dir,
        "vllm_draft": args.vllm_draft_dir,
        "hayate_drafts": args.hayate_draft_models_dir,
    }.items():
        if root is None:
            continue
        for config in iter_configs(root, args.max_draft_configs):
            item = config_summary(config)
            item["bucket"] = label
            visible_configs.append(item)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": str(Path.cwd()),
            "user": os.environ.get("USER"),
        },
        "artifact_root": str(artifact_root),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "CUDA_VISIBLE_DEVICES",
                "CONTAINER",
                "MOUNTS",
                "SBATCH_ACCOUNT",
                "SBATCH_PARTITION",
            )
            if os.environ.get(key) is not None
        },
        "repos": repos,
        "paths": paths,
        "critical_files": critical_file_hashes(args),
        "visible_configs": visible_configs,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Eagle3 Provenance",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Host: `{payload['host']['hostname']}`",
        f"Artifact root: `{payload['artifact_root']}`",
        "",
        "## Git Worktrees",
        "",
        "| label | exists | git | branch | head | dirty files | path |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for repo in payload["repos"]:
        dirty = len([line for line in str(repo.get("status_short") or "").splitlines() if line.strip()])
        head = str(repo.get("head") or "")[:12] or "-"
        lines.append(
            f"| {repo['label']} | {repo.get('exists')} | {repo.get('is_git_worktree', False)} | "
            f"{repo.get('branch') or '-'} | `{head}` | {dirty} | `{repo['path']}` |"
        )
    lines += ["", "## Artifact Paths", "", "| label | exists | type | files | configs | path |", "| --- | --- | --- | ---: | --- | --- |"]
    for item in payload["paths"]:
        lines.append(
            f"| {item['label']} | {item.get('exists')} | {item.get('type', '-')} | "
            f"{item.get('file_count_sampled', item.get('file_count', '-'))} | {item.get('config_json', '-')} | `{item.get('path')}` |"
        )
    lines += ["", "## Visible Configs", "", "| bucket | layers | hidden | heads | kv | rope | aux | config |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for item in payload["visible_configs"][:80]:
        lines.append(
            f"| {item.get('bucket')} | {item.get('num_hidden_layers')} | {item.get('hidden_size')} | "
            f"{item.get('num_attention_heads')} | {item.get('num_key_value_heads')} | "
            f"{item.get('rope_theta')} | `{item.get('aux_layers')}` | `{item.get('path')}` |"
        )
    lines += ["", "## Critical File Hashes", "", "| exists | bytes | sha256 | file |", "| --- | ---: | --- | --- |"]
    for item in payload["critical_files"]:
        digest = item.get("sha256", "-")
        lines.append(f"| {item.get('exists')} | {item.get('bytes', '-')} | `{digest}` | `{item['path']}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = collect(args)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
