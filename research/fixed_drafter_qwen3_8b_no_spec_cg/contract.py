#!/usr/bin/env python3
"""Validate the matched no-SpecDec CUDA Graph baseline and its resumes."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).parent
DFLASH_DIR = EXPERIMENT_DIR.parent / "fixed_drafter_qwen3_8b_dflash"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CONTAINER_SHA256 = "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"
TRAINING_HORIZON_STEPS = 1000
WANDB_PROJECT = "sna-nemo-rl-fixed-drafter"
WANDB_GROUP = "qwen3-8b-dflash-fixed-drafter-k-sweep"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DFLASH_CONTRACT = _load_module(
    "dflash_contract_for_baseline", DFLASH_DIR / "contract.py"
)
DFLASH_RESUME = _load_module(
    "dflash_resume_for_baseline", DFLASH_DIR / "resume_contract.py"
)


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}; got {actual!r}")


def validate_config(
    config_path: Path,
    *,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Prove that the baseline differs from K5 only by SpecDec and provenance."""
    reference_path = reference_path or DFLASH_DIR / "config_k5.yaml"
    reference = DFLASH_CONTRACT.load_config(reference_path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    default = raw.pop("defaults", None)
    if not isinstance(default, str) or not default.endswith(
        "fixed_drafter_qwen3_8b_dflash/config_k5.yaml"
    ):
        raise ValueError("baseline must inherit the exact DFlash K5 config")
    config = DFLASH_CONTRACT._merge_config(reference, raw)

    experiment = config["experiment"]
    generation = config["policy"]["generation"]
    vllm_cfg = generation["vllm_cfg"]
    vllm_kwargs = generation["vllm_kwargs"]
    logger = config["logger"]
    wandb = logger["wandb"]
    wandb_config = wandb["config"]

    _require_equal(vllm_kwargs["speculative_config"], None, name="speculative_config")
    _require_equal(vllm_cfg["enforce_eager"], False, name="enforce_eager")
    _require_equal(
        experiment["target_revision"],
        reference["experiment"]["target_revision"],
        name="target_revision",
    )
    for key in ("drafter_repo", "drafter_revision", "drafter_config_sha256"):
        _require_equal(experiment[key], None, name=f"experiment.{key}")
    _require_equal(wandb["project"], WANDB_PROJECT, name="logger.wandb.project")
    _require_equal(wandb["group"], WANDB_GROUP, name="logger.wandb.group")
    _require_equal(
        wandb["name"],
        "qwen3-8b-no-specdec-cudagraph-step001-seed42",
        name="logger.wandb.name",
    )
    _require_equal(wandb_config["method"], "no-specdec", name="wandb method")
    _require_equal(wandb_config["k"], None, name="wandb k")
    _require_equal(wandb_config["draft_tp"], None, name="wandb draft_tp")
    _require_equal(
        wandb_config["per_position_acceptance_positions"],
        [],
        name="wandb per-position acceptance",
    )

    expected_wandb = copy.deepcopy(reference["logger"]["wandb"])
    expected_wandb.update(
        {
            "name": "qwen3-8b-no-specdec-cudagraph-step001-seed42",
            "tags": [
                "baseline",
                "no-specdec",
                "qwen3-8b",
                "cudagraph",
                "target-only-grpo",
                "seed42",
                "step001",
            ],
        }
    )
    expected_wandb["config"].update(
        {
            "experiment": "fixed-drafter-qwen3-8b-no-spec-cg",
            "method": "no-specdec",
            "drafter_repo": None,
            "drafter_revision": None,
            "drafter_config_sha256": None,
            "k": None,
            "max_dflash_decode_query_tokens": None,
            "per_position_acceptance_positions": [],
            "draft_tp": None,
        }
    )
    _require_equal(wandb, expected_wandb, name="logger.wandb")

    normalized = copy.deepcopy(config)
    for key in ("drafter_repo", "drafter_revision", "drafter_config_sha256"):
        normalized["experiment"][key] = reference["experiment"][key]
    normalized["policy"]["generation"]["vllm_kwargs"]["speculative_config"] = (
        copy.deepcopy(
            reference["policy"]["generation"]["vllm_kwargs"]["speculative_config"]
        )
    )
    normalized["logger"]["log_dir"] = reference["logger"]["log_dir"]
    normalized["logger"]["wandb"] = copy.deepcopy(reference["logger"]["wandb"])
    if normalized != reference:
        raise ValueError(
            "baseline has an unmatched change outside SpecDec and provenance"
        )

    compilation = vllm_kwargs["compilation_config"]
    policy = config["policy"]
    grpo = config["grpo"]
    megatron = policy["megatron_cfg"]
    return {
        "method": "no-specdec",
        "target_revision": experiment["target_revision"],
        "speculative_config": vllm_kwargs["speculative_config"],
        "enforce_eager": vllm_cfg["enforce_eager"],
        "cudagraph_mode": compilation["cudagraph_mode"],
        "cudagraph_capture_sizes": compilation["cudagraph_capture_sizes"],
        "seed": grpo["seed"],
        "dataset": config["data"]["train"]["dataset_name"],
        "prompts_per_step": grpo["num_prompts_per_step"],
        "generations_per_prompt": grpo["num_generations_per_prompt"],
        "global_batch_size": policy["train_global_batch_size"],
        "micro_batch_size": policy["train_micro_batch_size"],
        "training_tp": megatron["tensor_model_parallel_size"],
        "training_pp": megatron["pipeline_model_parallel_size"],
        "training_cp": megatron["context_parallel_size"],
        "generation_tp": vllm_cfg["tensor_parallel_size"],
        "max_new_tokens": generation["max_new_tokens"],
        "max_total_sequence_length": policy["max_total_sequence_length"],
        "wandb_project": wandb["project"],
        "wandb_group": wandb["group"],
    }


def _checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    return sorted(
        int(path.name.removeprefix("step_"))
        for path in checkpoint_dir.glob("step_*")
        if path.is_dir() and path.name.removeprefix("step_").isdigit()
    )


def write_manifest(
    manifest_path: Path,
    *,
    git_sha: str,
    checkpoint_dir: Path,
    wandb_run_id: str,
) -> None:
    payload = {
        "schema_version": 1,
        "method": "no-specdec",
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_dir.resolve()),
        "wandb_run_id": wandb_run_id,
        "target_revision": TARGET_REVISION,
        "container_sha256": CONTAINER_SHA256,
        "training_horizon_steps": TRAINING_HORIZON_STEPS,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)


def validate_manifest(
    manifest_path: Path,
    *,
    git_sha: str,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": 1,
        "method": "no-specdec",
        "git_sha": git_sha,
        "checkpoint_root": str(checkpoint_dir.resolve()),
        "target_revision": TARGET_REVISION,
        "container_sha256": CONTAINER_SHA256,
        "training_horizon_steps": TRAINING_HORIZON_STEPS,
    }
    for key, value in expected.items():
        _require_equal(manifest.get(key), value, name=f"manifest {key}")
    if (
        not isinstance(manifest.get("wandb_run_id"), str)
        or not manifest["wandb_run_id"]
    ):
        raise ValueError("manifest wandb_run_id must be non-empty")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--previous-step", type=int)
    parser.add_argument("--current-step", type=int)
    parser.add_argument("--git-sha")
    parser.add_argument("--gate-manifest", type=Path)
    parser.add_argument("--wandb-log", type=Path)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--print-latest-step", action="store_true")
    parser.add_argument("--verify-nemo-resume-paths", action="store_true")
    parser.add_argument("--create-gate-manifest", action="store_true")
    parser.add_argument("--validate-gate-manifest", action="store_true")
    parser.add_argument("--print-manifest-wandb-run-id", action="store_true")
    args = parser.parse_args()

    if args.validate_config:
        if args.config is None:
            raise ValueError("--config is required")
        print(json.dumps(validate_config(args.config), indent=2, sort_keys=True))
    if args.print_latest_step:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required")
        steps = _checkpoint_steps(args.checkpoint_dir)
        if not steps:
            raise ValueError(f"no checkpoint under {args.checkpoint_dir}")
        print(steps[-1])
    if args.expected_step is not None:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required")
        step_dir = DFLASH_RESUME.validate_checkpoint(
            args.checkpoint_dir,
            expected_step=args.expected_step,
            expected_horizon_steps=TRAINING_HORIZON_STEPS,
        )
        if args.verify_nemo_resume_paths:
            DFLASH_RESUME.verify_nemo_resume_paths(step_dir)
    if args.previous_step is not None or args.current_step is not None:
        if args.previous_step is None or args.current_step is None:
            raise ValueError("--previous-step and --current-step are required together")
        DFLASH_RESUME.validate_progress(args.previous_step, args.current_step)
    if args.create_gate_manifest:
        if None in (
            args.gate_manifest,
            args.git_sha,
            args.checkpoint_dir,
            args.wandb_log,
        ):
            raise ValueError("all gate manifest arguments are required")
        write_manifest(
            args.gate_manifest,
            git_sha=args.git_sha,
            checkpoint_dir=args.checkpoint_dir,
            wandb_run_id=DFLASH_RESUME.extract_wandb_run_id(args.wandb_log),
        )
    if args.validate_gate_manifest or args.print_manifest_wandb_run_id:
        if None in (args.gate_manifest, args.git_sha, args.checkpoint_dir):
            raise ValueError("all gate manifest arguments are required")
        manifest = validate_manifest(
            args.gate_manifest,
            git_sha=args.git_sha,
            checkpoint_dir=args.checkpoint_dir,
        )
        if args.print_manifest_wandb_run_id:
            print(manifest["wandb_run_id"])


if __name__ == "__main__":
    main()
