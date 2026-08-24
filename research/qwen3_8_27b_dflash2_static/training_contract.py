#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import cast

import yaml


TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
TRAINING_STEPS = frozenset({1, 20})


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def load_training_recipe(path: Path) -> dict[str, object]:
    values = _mapping(yaml.safe_load(path.read_bytes()), name="recipe")
    grpo = _mapping(values.get("grpo"), name="grpo")
    policy = _mapping(values.get("policy"), name="policy")
    tokenizer = _mapping(policy.get("tokenizer"), name="policy.tokenizer")
    draft = _mapping(policy.get("draft"), name="policy.draft")
    generation = _mapping(policy.get("generation"), name="policy.generation")
    vllm_cfg = _mapping(generation.get("vllm_cfg"), name="vllm_cfg")
    env_vars = _mapping(vllm_cfg.get("env_vars"), name="vllm_cfg.env_vars")
    vllm_kwargs = _mapping(generation.get("vllm_kwargs"), name="vllm_kwargs")
    speculative = _mapping(
        vllm_kwargs.get("speculative_config"), name="speculative_config"
    )

    resolved = {
        "max_num_steps": grpo.get("max_num_steps"),
        "target_placeholder": policy.get("model_name"),
        "draft_placeholder": speculative.get("model"),
        "draft_refit": draft.get("enabled"),
        "language_model_only": vllm_kwargs.get("language_model_only"),
        "speculative_method": speculative.get("method"),
        "num_speculative_tokens": speculative.get("num_speculative_tokens"),
        "v2_model_runner": env_vars.get("VLLM_USE_V2_MODEL_RUNNER"),
    }
    expected = {
        "max_num_steps": 1,
        "target_placeholder": "__DFLASH2_TARGET_SNAPSHOT__",
        "draft_placeholder": "__DFLASH2_DRAFT_SNAPSHOT__",
        "draft_refit": False,
        "language_model_only": True,
        "speculative_method": "dflash",
        "num_speculative_tokens": 7,
        "v2_model_runner": "1",
    }
    if resolved != expected or tokenizer.get("name") != expected["target_placeholder"]:
        raise ValueError("GRPO recipe violates the static target-only DFlash2 contract")
    return resolved


def validate_training_steps(steps: object) -> int:
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps not in TRAINING_STEPS
    ):
        raise ValueError("training steps must be exactly 1 or 20")
    return steps


def _validate_root(path: Path, *, root: str, name: str) -> None:
    if not path.is_absolute() or path.parts[:2] != ("/", root):
        raise ValueError(f"{name} must be under /{root}")


def _validate_snapshot(path: Path, *, revision: str, name: str) -> None:
    _validate_root(path, root="lustre", name=name)
    if path.name != revision:
        raise ValueError(f"{name} must end with pinned revision {revision}")


def build_training_command(
    *,
    repo_root: Path,
    recipe: Path,
    target_snapshot: Path,
    draft_snapshot: Path,
    output_dir: Path,
    steps: object,
) -> list[str]:
    steps = validate_training_steps(steps)
    _validate_root(repo_root, root="home", name="repo root")
    _validate_snapshot(
        target_snapshot, revision=TARGET_REVISION, name="target snapshot"
    )
    _validate_snapshot(draft_snapshot, revision=DRAFT_REVISION, name="draft snapshot")
    _validate_root(output_dir, root="lustre", name="output directory")
    if recipe != repo_root / "research/qwen3_8_27b_dflash2_static/grpo.yaml":
        raise ValueError("recipe must be the committed DFlash2 GRPO recipe")

    return [
        sys.executable,
        str(repo_root / "examples/run_grpo.py"),
        "--config",
        str(recipe),
        f"grpo.max_num_steps={steps}",
        f"policy.model_name={target_snapshot}",
        f"policy.tokenizer.name={target_snapshot}",
        (f"policy.generation.vllm_kwargs.speculative_config.model={draft_snapshot}"),
        f"logger.log_dir={output_dir}/logs",
        f"checkpointing.checkpoint_dir={output_dir}/checkpoints",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--target-snapshot", type=Path, required=True)
    parser.add_argument("--draft-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_training_recipe(args.recipe)
    command = build_training_command(
        repo_root=args.repo_root,
        recipe=args.recipe,
        target_snapshot=args.target_snapshot,
        draft_snapshot=args.draft_snapshot,
        output_dir=args.output_dir,
        steps=args.steps,
    )
    if args.dry_run:
        print(shlex.join(command))
        return

    missing = [
        str(path)
        for path in (args.target_snapshot, args.draft_snapshot)
        if not path.is_dir()
    ]
    if missing:
        raise RuntimeError(
            "pinned snapshot directories do not exist: " + ", ".join(missing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "training_summary.json"
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "kind": "nemo_rl_grpo_training",
        "requested_training_steps": args.steps,
        "target_revision": TARGET_REVISION,
        "draft_revision": DRAFT_REVISION,
        "draft_refit": False,
        "command": command,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    result = subprocess.run(command, check=False)
    summary["returncode"] = result.returncode
    summary["status"] = "passed" if result.returncode == 0 else "failed"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
