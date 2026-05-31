#!/usr/bin/env python3
"""Validate the SpecDec-RL remote patch overlay for Qwen3 Eagle3 rollout.

This is a no-submit, mostly local check. It proves that the handoff contains the
runtime compatibility files needed for the current Qwen3-235B SWE rollout path
without claiming that a remote SpecDec-RL checkout has already been patched.
When a target checkout is visible, it also reports whether the overlay appears
to be applied there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_PATCH_ROOT = EXP / "remote_patches" / "SpecDec-RL"
DEFAULT_TARGET = Path("/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL")

REQUIRED_FILES: dict[str, list[str]] = {
    "ray.sub": [
        "maybe_gres_arg()",
        'partition_gres="$(sinfo -p "$SLURM_JOB_PARTITION"',
        'COMMON_SRUN_ARGS="$GRES_ARG"',
        "CPUS_PER_WORKER=${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}",
        "RAY_VERSION=${RAY_VERSION:-2.49.2}",
    ],
    "nemo_rl/models/generation/vllm/vllm_worker_async.py": [
        "model_config = self.llm_async_engine_args.create_model_config()",
        'openai_serving_models_kwargs["model_config"] = model_config',
        "spec_decode_enabled = bool(llm_kwargs.get(\"speculative_config\"))",
        "or spec_decode_enabled",
        "request.logprobs = False",
        "self._vllm_spec_decode_metrics_baseline",
    ],
    "nemo_rl/models/generation/vllm/vllm_worker.py": [
        "def _patch_vllm_speculative_decoding_post_step(required: bool)",
        "def _patch_vllm_batch_gated_speculative_decoding()",
        "NRL_SPECDEC_BATCH_GATE_PATCH_V2",
        "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD",
        "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD",
        "_nrl_specdec_batch_gate_threshold",
        "_nrl_specdec_batch_gate_token_threshold",
        "specdec_batch_gate_num_requests",
        "partial or outdated vLLM SpecDec batch-gate",
        "NRL_ALLOW_SPECDEC_DISABLE_BY_BATCH_SIZE",
        "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH",
        "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS",
        "max_new_tokens=allowed_new_tokens_per_sample[idx]",
        "stop_strings=self._merge_stop_strings(",
    ],
    "nemo_rl/models/generation/vllm/vllm_generation.py": [
        "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS",
        "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH",
        "spec_decode_totals[\"metrics_available\"] = True",
        "spec_decode_totals[\"metrics_complete\"]",
    ],
    "nemo_rl/algorithms/grpo.py": [
        "def _uses_vllm_specdec_without_generation_logprobs(",
        "def _repair_specdec_generation_logprobs_if_safe(",
        "ray.get(trajectory_collector.set_weight_version.remote(weight_version))",
        "collection_task = trajectory_collector.start_collection.remote(dataloader)",
    ],
    "nemo_rl/distributed/virtual_cluster.py": [
        "NRL_RAY_NODE_IP_ADDRESS",
        "RAY_NODE_IP_ADDRESS",
        'ray_init_kwargs["_node_ip_address"] = ray_node_ip_address',
    ],
    "nemo_rl/models/megatron/community_import.py": [
        "def _patch_distcp_writer_for_ray_import()",
        "NRL_MEGATRON_IMPORT_INLINE_WRITER",
        "write_preloaded_data_inline",
        "def _save_megatron_model_with_checkpointing(",
        "def _save_megatron_model(",
        "from megatron.bridge.training.model_load_save import save_megatron_model as save",
        "from megatron.bridge.training.checkpointing import (",
        "MockGPTDatasetConfig",
        "reset_position_ids=False",
        "reset_attention_mask=False",
        "eod_mask_loss=False",
        "model_provider.initialize_model_parallel(seed=0, **model_parallel_kwargs)",
        "provide_distributed_model = getattr(",
        "model_provider.provide_models(wrap_with_ddp=False)",
        "_save_megatron_model(bridge, megatron_model, output_path, hf_model_name, model_provider)",
    ],
    "nemo_rl/models/megatron/setup.py": [
        "from inspect import signature",
        "def calculate_padded_vocab_size(",
        "ProcessGroupCollection = None",
        "def _call_accepts_kwarg(",
        "def _add_pg_collection_kwarg(",
        "_use_mpu_process_groups()",
    ],
    "nemo_rl/models/policy/workers/megatron_policy_worker.py": [
        "try:",
        "from megatron.bridge.training.utils.pg_utils import get_pg_collection",
        "def get_pg_collection(model):",
        "return SimpleNamespace(mp=parallel_state.get_model_parallel_group())",
        "custom_FSDP = ()",
    ],
    "nemo_rl/models/policy/workers/patches.py": [
        "def apply_torch_aten_alias_tensor_patch():",
        "from torch.distributed.tensor._ops._tensor_ops",
        "propagate_single_input_strategy",
    ],
}

IGNORED_DIRS = {"__pycache__", ".git"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument(
        "--target-specdec-rl-dir",
        type=Path,
        default=Path(os.environ.get("SPECDEC_RL_DIR", DEFAULT_TARGET)),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--require-target-applied",
        action="store_true",
        help="Return FAIL unless the visible target checkout already contains the required snippets.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def is_ignored(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    if any(part in IGNORED_DIRS for part in rel.parts):
        return True
    return path.suffix in IGNORED_SUFFIXES


def iter_overlay_files(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    ignored: list[Path] = []
    if not root.exists():
        return files, ignored
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_ignored(path, root):
            ignored.append(path)
        else:
            files.append(path)
    return files, ignored


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def required_file_rows(patch_root: Path, target_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_visible = target_root.exists()
    for rel, snippets in REQUIRED_FILES.items():
        source = patch_root / rel
        row: dict[str, Any] = {
            "path": rel,
            "source_exists": source.exists(),
            "required_snippet_count": len(snippets),
        }
        if source.exists():
            source_text = read_text(source)
            missing = [snippet for snippet in snippets if snippet not in source_text]
            row.update(
                {
                    "source_sha256": sha256_file(source),
                    "source_size_bytes": source.stat().st_size,
                    "source_missing_snippets": missing,
                    "source_status": "pass" if not missing else "fail",
                }
            )
        else:
            row.update({"source_missing_snippets": snippets, "source_status": "fail"})

        target = target_root / rel
        if not target_visible:
            row["target_status"] = "not_visible"
        elif not target.exists():
            row["target_status"] = "missing"
        else:
            target_text = read_text(target)
            target_missing = [snippet for snippet in snippets if snippet not in target_text]
            target_sha = sha256_file(target)
            row.update(
                {
                    "target_sha256": target_sha,
                    "target_size_bytes": target.stat().st_size,
                    "target_missing_snippets": target_missing,
                    "target_matches_overlay": bool(row.get("source_sha256") and row.get("source_sha256") == target_sha),
                    "target_status": "applied" if not target_missing else "needs_overlay",
                }
            )
        rows.append(row)
    return rows


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    patch_root = args.patch_root
    target_root = args.target_specdec_rl_dir
    files, ignored = iter_overlay_files(patch_root)
    rows = required_file_rows(patch_root, target_root)

    if patch_root.exists():
        add(checks, "source", "patch root", "pass", f"visible: {patch_root}")
    else:
        add(checks, "source", "patch root", "fail", f"not visible: {patch_root}")

    for row in rows:
        rel = row["path"]
        if row["source_status"] == "pass":
            add(checks, "source", rel, "pass", "required overlay file contains expected markers")
        else:
            add(
                checks,
                "source",
                rel,
                "fail",
                "required overlay file is missing or lacks expected markers",
                missing_snippets=row.get("source_missing_snippets"),
            )

    if ignored:
        add(
            checks,
            "source",
            "ignored generated files",
            "warn",
            "generated or binary files are present but excluded from the handoff overlay",
            count=len(ignored),
            examples=[str(path.relative_to(patch_root)) for path in ignored[:8]],
        )
    else:
        add(checks, "source", "ignored generated files", "pass", "no generated files found in the overlay tree")

    if target_root.exists():
        add(checks, "target", "SpecDec-RL checkout", "pass", f"visible: {target_root}")
        target_needs = [row for row in rows if row.get("target_status") != "applied"]
        if target_needs:
            status = "fail" if args.require_target_applied else "warn"
            add(
                checks,
                "target",
                "overlay applied",
                status,
                "target checkout is visible but one or more required files do not contain the overlay markers",
                missing=[row["path"] for row in target_needs],
            )
        else:
            add(checks, "target", "overlay applied", "pass", "target checkout contains all required overlay markers")
    else:
        status = "fail" if args.require_target_applied else "warn"
        add(checks, "target", "SpecDec-RL checkout", status, f"not visible: {target_root}")

    status_counts: dict[str, int] = {}
    for check in checks:
        status_counts[check["status"]] = status_counts.get(check["status"], 0) + 1

    source_ok = all(row.get("source_status") == "pass" for row in rows) and patch_root.exists()
    target_ok = target_root.exists() and all(row.get("target_status") == "applied" for row in rows)
    if not source_ok:
        overall = "fail"
    elif args.require_target_applied and not target_ok:
        overall = "fail"
    else:
        overall = "pass"

    copy_command = (
        f'rsync -a --exclude "__pycache__" --exclude "*.pyc" '
        f'{patch_root.as_posix().rstrip("/")}/ {target_root.as_posix().rstrip("/")}/'
    )
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "patch_root": str(patch_root),
        "target_specdec_rl_dir": str(target_root),
        "target_status": "applied" if target_ok else ("not_visible" if not target_root.exists() else "needs_overlay"),
        "file_count": len(files),
        "ignored_file_count": len(ignored),
        "required_files": rows,
        "copy_command": copy_command,
        "counts": status_counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SpecDec-RL Remote Patch Bundle Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Patch root: `{payload['patch_root']}`",
        f"Target checkout: `{payload['target_specdec_rl_dir']}`",
        f"Target status: `{payload['target_status']}`",
        "",
        "## Required Files",
        "",
        "| path | source | target |",
        "| --- | --- | --- |",
    ]
    for row in payload["required_files"]:
        lines.append(f"| `{row['path']}` | {row.get('source_status')} | {row.get('target_status')} |")
    lines += [
        "",
        "## Stage Command",
        "",
        "```bash",
        payload["copy_command"],
        "```",
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
    args = parse_args()
    payload = build_payload(args)
    markdown = render_markdown(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 1 if payload["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
