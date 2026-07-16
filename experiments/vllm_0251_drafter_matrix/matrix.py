# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed resolution for the vLLM 0.25.1 Qwen drafter matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """Authoritative topology and recipe path for one controlled model."""

    key: str
    path: str
    target_repo_id: str
    target_revision: str
    nodes: int
    segment: int
    max_osl: int

    def target_ref_path(self, hf_home: Path) -> Path:
        """Return the Hugging Face ``main`` ref used by the recipe target."""
        return (
            hf_home
            / "hub"
            / f"models--{self.target_repo_id.replace('/', '--')}"
            / "refs"
            / "main"
        )


@dataclass(frozen=True, slots=True)
class ClusterSpec:
    """Scheduler metadata for a supported cluster."""

    key: str
    account: str
    partition: str
    gpus_per_node: int
    hf_home: Path


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """Bounded execution phase for a matrix run."""

    key: str
    max_steps: int
    time_limit: str


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """Immutable Hugging Face identity for one target-specific draft model."""

    model_key: str
    repo_id: str
    revision: str

    def snapshot_path(self, hf_home: Path) -> Path:
        """Return the immutable local snapshot path under ``HF_HOME``."""
        return (
            hf_home
            / "hub"
            / f"models--{self.repo_id.replace('/', '--')}"
            / "snapshots"
            / self.revision
        )


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """An official vLLM speculative-decoding configuration."""

    key: str
    method: str | None
    runner: str
    num_speculative_tokens: int | None
    compatible_models: frozenset[str]
    checkpoints: tuple[CheckpointSpec, ...] = ()
    uses_draft_model: bool = False
    draft_attention_backend: str | None = None
    parallel_drafting: bool = False
    suffix_tree_depth: int | None = None
    ngram_size: int | None = None

    def checkpoint_for(self, model_key: str) -> CheckpointSpec | None:
        """Return the exact drafter checkpoint for a compatible model."""
        for checkpoint in self.checkpoints:
            if checkpoint.model_key == model_key:
                return checkpoint
        return None


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """Fully validated matrix entry with scheduler and Hydra arguments."""

    recipe: RecipeSpec
    cluster: ClusterSpec
    phase: PhaseSpec
    variant: VariantSpec
    draft_checkpoint: CheckpointSpec | None
    hydra_overrides: tuple[str, ...]

    def command_parts(self) -> tuple[str, ...]:
        """Return the deterministic training command as an argument tuple."""
        model_runner_v2 = "1" if self.variant.runner == "mrv2" else "0"
        return (
            "env",
            f"VLLM_USE_V2_MODEL_RUNNER={model_runner_v2}",
            f"GPUS_PER_NODE={self.cluster.gpus_per_node}",
            "python3",
            "examples/run_grpo.py",
            "--config",
            self.recipe.path,
            *self.hydra_overrides,
        )

    def sbatch_parts(self) -> tuple[str, ...]:
        """Return Lyris scheduler arguments without GPU-resource flags."""
        return (
            "sbatch",
            "--dependency=",
            f"--export=ALL,GPUS_PER_NODE={self.cluster.gpus_per_node}",
            f"--account={self.cluster.account}",
            f"--partition={self.cluster.partition}",
            f"--nodes={self.recipe.nodes}",
            "--ntasks-per-node=1",
            "--exclusive",
            f"--time={self.phase.time_limit}",
            f"--segment={self.recipe.segment}",
            f"--job-name=nemorl-{self.recipe.key}-{self.variant.key}-{self.phase.key}",
            "ray.sub",
        )


@dataclass(frozen=True, slots=True)
class CheckoutState:
    """Validated Git identity for a submission checkout."""

    branch: str
    head: str
    fork_ref: str
    submodules: tuple[str, ...]


G_MODEL_KEYS = frozenset(("qwen30", "qwen32", "qwen235"))

G_RECIPES = (
    RecipeSpec(
        key="qwen30",
        path="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        target_repo_id="Qwen/Qwen3-30B-A3B",
        target_revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        nodes=4,
        segment=4,
        max_osl=4096,
    ),
    RecipeSpec(
        key="qwen32",
        path="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
        target_repo_id="Qwen/Qwen3-32B",
        target_revision="9216db5781bf21249d130ec9da846c4624c16137",
        nodes=4,
        segment=4,
        max_osl=4096,
    ),
    RecipeSpec(
        key="qwen235",
        path="examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml",
        target_repo_id="Qwen/Qwen3-235B-A22B",
        target_revision="8efa61729e24bd65b1d152b5ab5409052aa80e65",
        nodes=16,
        segment=16,
        max_osl=8192,
    ),
)

G_CLUSTERS = (
    ClusterSpec(
        key="lyris",
        account="coreai_dlalgo_llm",
        partition="gb200",
        gpus_per_node=4,
        hf_home=Path("/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"),
    ),
)

G_PHASES = (
    PhaseSpec(key="smoke2", max_steps=2, time_limit="01:00:00"),
    PhaseSpec(key="smoke5", max_steps=5, time_limit="02:00:00"),
    PhaseSpec(key="final20", max_steps=20, time_limit="05:00:00"),
)

G_EAGLE3_CHECKPOINTS = (
    CheckpointSpec(
        model_key="qwen30",
        repo_id="RedHatAI/Qwen3-30B-A3B-speculator.eagle3",
        revision="6afc5aa2477b923467fb9a8d906782b984a9a6ba",
    ),
    CheckpointSpec(
        model_key="qwen32",
        repo_id="RedHatAI/Qwen3-32B-speculator.eagle3",
        revision="dc84fe7ff1db31efa824776f49c141fc8195eb47",
    ),
    CheckpointSpec(
        model_key="qwen235",
        repo_id="nvidia/Qwen3-235B-A22B-Eagle3",
        revision="33f3c01ce807376d1171301b9a148b1b28f239ba",
    ),
)

G_EAGLE3_THINKING_CHECKPOINTS = (
    CheckpointSpec(
        model_key="qwen32",
        repo_id="RedHatAI/Qwen3-32B-Thinking-speculator.eagle3",
        revision="a1403e07b73a66fc9ef561463631c31864616933",
    ),
    CheckpointSpec(
        model_key="qwen235",
        repo_id="RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3",
        revision="3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87",
    ),
)

G_PARD_CHECKPOINTS = (
    CheckpointSpec(
        model_key="qwen30",
        repo_id="amd/PARD-Qwen3-0.6B",
        revision="f9f650fbab180c26498817718f0db5cae8f25136",
    ),
    CheckpointSpec(
        model_key="qwen32",
        repo_id="amd/PARD-Qwen3-0.6B",
        revision="f9f650fbab180c26498817718f0db5cae8f25136",
    ),
    CheckpointSpec(
        model_key="qwen235",
        repo_id="amd/PARD-Qwen3-0.6B",
        revision="f9f650fbab180c26498817718f0db5cae8f25136",
    ),
)

G_DFLASH_CHECKPOINTS = (
    CheckpointSpec(
        model_key="qwen30",
        repo_id="RedHatAI/Qwen3-30B-A3B-speculator.dflash",
        revision="edcff83783141eb9383e2bd6c33610d9a3104288",
    ),
    CheckpointSpec(
        model_key="qwen32",
        repo_id="AICP-Labs/qwen3-32b-dflash-en-zh",
        revision="68ccc7fd27b104271321b179a2959c759dce5eef",
    ),
)

G_VARIANTS = (
    VariantSpec(
        key="baseline",
        method=None,
        runner="mrv2",
        num_speculative_tokens=None,
        compatible_models=G_MODEL_KEYS,
    ),
    VariantSpec(
        key="baseline_mrv1",
        method=None,
        runner="mrv1",
        num_speculative_tokens=None,
        compatible_models=G_MODEL_KEYS,
    ),
    VariantSpec(
        key="eagle3_k1",
        method="eagle3",
        runner="mrv2",
        num_speculative_tokens=1,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_EAGLE3_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="eagle3_k3",
        method="eagle3",
        runner="mrv2",
        num_speculative_tokens=3,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_EAGLE3_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="eagle3_k5",
        method="eagle3",
        runner="mrv2",
        num_speculative_tokens=5,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_EAGLE3_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="eagle3_thinking_k1",
        method="eagle3",
        runner="mrv2",
        num_speculative_tokens=1,
        compatible_models=frozenset(("qwen32", "qwen235")),
        checkpoints=G_EAGLE3_THINKING_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="eagle3_thinking_k3",
        method="eagle3",
        runner="mrv2",
        num_speculative_tokens=3,
        compatible_models=frozenset(("qwen32", "qwen235")),
        checkpoints=G_EAGLE3_THINKING_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="eagle3_thinking_k5",
        method="eagle3",
        runner="mrv2",
        num_speculative_tokens=5,
        compatible_models=frozenset(("qwen32", "qwen235")),
        checkpoints=G_EAGLE3_THINKING_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="dflash_k3",
        method="dflash",
        runner="mrv2",
        num_speculative_tokens=3,
        compatible_models=frozenset(("qwen30", "qwen32")),
        checkpoints=G_DFLASH_CHECKPOINTS,
        uses_draft_model=True,
        draft_attention_backend="FLASH_ATTN",
    ),
    VariantSpec(
        key="dflash_k5",
        method="dflash",
        runner="mrv2",
        num_speculative_tokens=5,
        compatible_models=frozenset(("qwen30", "qwen32")),
        checkpoints=G_DFLASH_CHECKPOINTS,
        uses_draft_model=True,
        draft_attention_backend="FLASH_ATTN",
    ),
    VariantSpec(
        key="draft_k1",
        method="draft_model",
        runner="mrv1",
        num_speculative_tokens=1,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_PARD_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="draft_k5",
        method="draft_model",
        runner="mrv1",
        num_speculative_tokens=5,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_PARD_CHECKPOINTS,
        uses_draft_model=True,
    ),
    VariantSpec(
        key="pard_k5",
        method="draft_model",
        runner="mrv1",
        num_speculative_tokens=5,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_PARD_CHECKPOINTS,
        uses_draft_model=True,
        parallel_drafting=True,
    ),
    VariantSpec(
        key="pard_k16",
        method="draft_model",
        runner="mrv1",
        num_speculative_tokens=16,
        compatible_models=G_MODEL_KEYS,
        checkpoints=G_PARD_CHECKPOINTS,
        uses_draft_model=True,
        parallel_drafting=True,
    ),
    VariantSpec(
        key="suffix_k32",
        method="suffix",
        runner="mrv1",
        num_speculative_tokens=32,
        compatible_models=G_MODEL_KEYS,
        suffix_tree_depth=32,
    ),
    VariantSpec(
        key="ngram_k5",
        method="ngram",
        runner="mrv1",
        num_speculative_tokens=5,
        compatible_models=G_MODEL_KEYS,
        ngram_size=5,
    ),
    VariantSpec(
        key="ngram_gpu_k5",
        method="ngram_gpu",
        runner="mrv1",
        num_speculative_tokens=5,
        compatible_models=G_MODEL_KEYS,
        ngram_size=5,
    ),
)


def _find_recipe(key: str) -> RecipeSpec:
    """Return a controlled recipe by key."""
    for recipe in G_RECIPES:
        if recipe.key == key:
            return recipe
    raise ValueError(f"Unknown recipe key: {key}")


def _find_cluster(key: str) -> ClusterSpec:
    """Return a supported cluster by key."""
    for cluster in G_CLUSTERS:
        if cluster.key == key:
            return cluster
    raise ValueError(f"Unknown cluster key: {key}")


def _find_phase(key: str) -> PhaseSpec:
    """Return an execution phase by key."""
    for phase in G_PHASES:
        if phase.key == key:
            return phase
    raise ValueError(f"Unknown phase key: {key}")


def _find_variant(key: str) -> VariantSpec:
    """Return a configured variant by key."""
    for variant in G_VARIANTS:
        if variant.key == key:
            return variant
    raise ValueError(f"Unknown variant key: {key}")


def _speculative_overrides(
    variant: VariantSpec,
    draft_checkpoint: CheckpointSpec | None,
    hf_home: Path,
) -> tuple[str, ...]:
    """Return official vLLM speculative settings for a validated variant."""
    if variant.method is None:
        return ()

    prefix = "++policy.generation.vllm_kwargs.speculative_config"
    overrides = [f"{prefix}.method={variant.method}"]
    if variant.num_speculative_tokens is not None:
        overrides.append(
            f"{prefix}.num_speculative_tokens={variant.num_speculative_tokens}"
        )
    if variant.uses_draft_model:
        if draft_checkpoint is None:
            raise ValueError(f"Variant '{variant.key}' requires a draft checkpoint")
        overrides.extend(
            (
                f"{prefix}.model={draft_checkpoint.snapshot_path(hf_home)}",
                f"{prefix}.draft_tensor_parallel_size=1",
            )
        )
    if variant.draft_attention_backend is not None:
        overrides.append(
            f"{prefix}.attention_backend={variant.draft_attention_backend}"
        )
    if variant.parallel_drafting:
        overrides.append(f"{prefix}.parallel_drafting=true")
    if variant.suffix_tree_depth is not None:
        overrides.append(
            f"{prefix}.suffix_decoding_max_tree_depth={variant.suffix_tree_depth}"
        )
    if variant.ngram_size is not None:
        overrides.extend(
            (
                f"{prefix}.prompt_lookup_min={variant.ngram_size}",
                f"{prefix}.prompt_lookup_max={variant.ngram_size}",
            )
        )
    return tuple(overrides)


def resolve_run(
    model_key: str, variant_key: str, phase: str, cluster: str
) -> ResolvedRun:
    """Resolve an allowed model, variant, phase, and cluster into a run record."""
    recipe = _find_recipe(model_key)
    variant = _find_variant(variant_key)
    phase_spec = _find_phase(phase)
    cluster_spec = _find_cluster(cluster)
    if model_key not in variant.compatible_models:
        raise ValueError(
            f"Variant '{variant_key}' is not available for model '{model_key}'"
        )
    draft_checkpoint = variant.checkpoint_for(model_key)
    if variant.uses_draft_model and draft_checkpoint is None:
        raise ValueError(
            f"Variant '{variant_key}' has no exact checkpoint for model "
            f"'{model_key}'"
        )

    base_overrides = (
        f"grpo.max_num_steps={phase_spec.max_steps}",
        "checkpointing.enabled=false",
        "policy.generation.vllm_cfg.enforce_eager=false",
        "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode="
        "FULL_AND_PIECEWISE",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
    )
    return ResolvedRun(
        recipe=recipe,
        cluster=cluster_spec,
        phase=phase_spec,
        variant=variant,
        draft_checkpoint=draft_checkpoint,
        hydra_overrides=base_overrides
        + _speculative_overrides(
            variant,
            draft_checkpoint,
            cluster_spec.hf_home,
        ),
    )


G_DEFAULT_CONTAINER = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
    "nemo_rl_nightly_20260715.sqsh"
)
G_DEFAULT_EXPERIMENT_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/"
    "vllm0251_drafter_matrix"
)
G_WANDB_PROJECT = "nemo-rl-vllm0251-drafter-matrix"
G_FORK_URLS = frozenset(
    {
        "git@github-seonjinn:seonjinn/RL.git",
        "git@github.com:seonjinn/RL.git",
    }
)
G_LUSTRE_ROOT = Path("/lustre")
G_ALLOWED_ENVIRONMENT = frozenset(
    {
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NO_PROXY",
        "PATH",
        "SHELL",
        "SLURM_CONF",
        "TMPDIR",
        "USER",
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
    }
)
G_SECRET_ENVIRONMENT = frozenset({"WANDB_API_KEY"})
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def build_runtime_command(
    run: ResolvedRun,
    repo_dir: Path,
    run_dir: Path,
    run_tag: str,
    *,
    target_snapshot: Path | None = None,
    runtime_id: str | None = None,
) -> tuple[str, ...]:
    """Build the command executed by ``ray.sub`` inside the container."""
    runner_v2 = "1" if run.variant.runner == "mrv2" else "0"
    safe_tag = runtime_id or "".join(
        char if char.isalnum() or char in "-_" else "-" for char in run_tag
    )
    environment = (
        f"VLLM_USE_V2_MODEL_RUNNER={runner_v2}",
        f"GPUS_PER_NODE={run.cluster.gpus_per_node}",
        "WANDB_RUN_GROUP=vllm0251-drafter-matrix",
        "WANDB_RESUME=never",
        f"HF_HOME={run.cluster.hf_home}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        f"PYTHONPATH={repo_dir}",
        f"NEMO_RL_VENV_DIR=/tmp/nemorl-v0251-{safe_tag}",
        "NRL_FORCE_REBUILD_VENVS=true",
        f"TRITON_CACHE_DIR=/tmp/nemorl-v0251-triton-{safe_tag}",
        f"TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v0251-inductor-{safe_tag}",
        *(("NRL_DISABLE_VLLM_PORT_OVERRIDE=1",) if run.recipe.key == "qwen235" else ()),
        "PYTHONFAULTHANDLER=1",
        "RAY_DEDUP_LOGS=0",
    )
    output_overrides = (
        f"checkpointing.checkpoint_dir={run_dir / 'checkpoints'}",
        f"logger.log_dir={run_dir / 'nemo_logs'}",
        f"logger.wandb.project={G_WANDB_PROJECT}",
        f"logger.wandb.name={run_tag}",
    )
    target_overrides = (
        (
            f"policy.model_name={target_snapshot}",
            f"policy.tokenizer.name={target_snapshot}",
        )
        if target_snapshot is not None
        else ()
    )
    return (
        "env",
        *environment,
        "/opt/nemo_rl_venv/bin/python",
        "examples/run_grpo.py",
        "--config",
        run.recipe.path,
        *run.hydra_overrides,
        *target_overrides,
        *output_overrides,
    )


def build_scheduler_command(
    run: ResolvedRun,
    repo_dir: Path,
    run_dir: Path,
    mode: str,
) -> tuple[str, ...]:
    """Build an exact scheduler preflight or submission command."""
    if mode not in {"test-only", "submit"}:
        raise ValueError(f"Unsupported scheduler mode: {mode}")
    base = run.sbatch_parts()
    mode_flag = "--test-only" if mode == "test-only" else "--parsable"
    return (
        *base[:-1],
        f"--output={run_dir / 'slurm-%j.out'}",
        "--comment=metrics",
        mode_flag,
        str(repo_dir / base[-1]),
    )


def build_scheduler_sequence(
    run: ResolvedRun,
    repo_dir: Path,
    run_dir: Path,
    action: str,
) -> tuple[tuple[str, ...], ...]:
    """Return the exact preflight/submission sequence for one CLI action."""
    preflight = build_scheduler_command(run, repo_dir, run_dir, "test-only")
    if action == "test-only":
        return (preflight,)
    if action == "submit":
        return (preflight, build_scheduler_command(run, repo_dir, run_dir, "submit"))
    raise ValueError(f"Unsupported scheduler action: {action}")


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def validate_checkout(
    repo_dir: Path, expected_fork_url: str | None = None
) -> CheckoutState:
    """Require a clean branch whose exact HEAD exists on the user fork."""
    status = _git(repo_dir, "status", "--porcelain=v1", "--untracked-files=normal")
    if status:
        raise RuntimeError("Submission requires a clean tracked and untracked checkout")
    branch = _git(repo_dir, "symbolic-ref", "--quiet", "--short", "HEAD")
    head = _git(repo_dir, "rev-parse", "HEAD")
    push_url = _git(repo_dir, "remote", "get-url", "--push", "fork")
    allowed_fork_urls = (
        frozenset({expected_fork_url})
        if expected_fork_url is not None
        else G_FORK_URLS
    )
    if push_url not in allowed_fork_urls:
        raise RuntimeError(
            "fork push URL must identify the approved seonjinn/RL fork; "
            f"found {push_url!r}"
        )
    fork_ref = f"refs/remotes/fork/{branch}"
    remote_ref = f"refs/heads/{branch}"
    remote_result = subprocess.run(
        ("git", "ls-remote", "--exit-code", "fork", remote_ref),
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if remote_result.returncode == 2:
        raise RuntimeError(f"Branch has no pushed fork ref: {remote_ref}")
    if remote_result.returncode != 0:
        detail = remote_result.stderr.rstrip() or "no git diagnostic"
        raise RuntimeError(f"Could not query fork ref {remote_ref}: {detail}")
    remote_line = remote_result.stdout.rstrip()
    remote_parts = remote_line.split()
    if len(remote_parts) != 2 or remote_parts[1] != remote_ref:
        raise RuntimeError(f"Branch has no pushed fork ref: {remote_ref}")
    fork_head = remote_parts[0]
    if fork_head != head:
        raise RuntimeError(
            f"HEAD {head} is not pushed to the fork branch {fork_ref} ({fork_head})"
        )
    submodule_text = _git(repo_dir, "submodule", "status", "--recursive")
    submodules = tuple(line for line in submodule_text.splitlines() if line)
    invalid = tuple(line for line in submodules if not line.startswith(" "))
    if invalid:
        raise RuntimeError(
            "Recursive submodules are not initialized at recorded commits: "
            + "; ".join(invalid)
        )
    return CheckoutState(
        branch=branch,
        head=head,
        fork_ref=fork_ref,
        submodules=submodules,
    )


def validate_snapshot(snapshot: Path, revision: str, label: str) -> None:
    """Require a full immutable SHA, config, and at least one weight artifact."""
    if _FULL_SHA.fullmatch(revision) is None:
        raise RuntimeError(f"{label} revision must be a 40-character Git SHA")
    if snapshot.name != revision:
        raise RuntimeError(
            f"{label} snapshot directory does not match revision {revision}: {snapshot}"
        )
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"{label} snapshot is incomplete: {snapshot}")
    index_path = snapshot / "model.safetensors.index.json"
    has_index = index_path.is_file()
    has_weights = any(path.is_file() for path in snapshot.glob("*.safetensors"))
    if not has_index and not has_weights:
        raise FileNotFoundError(f"{label} snapshot has no model weights: {snapshot}")
    if has_index:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
            raise RuntimeError(f"{label} has an invalid weight index: {index_path}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"{label} has an invalid weight index: {index_path}")
        shard_values = tuple(weight_map.values())
        if not all(isinstance(name, str) for name in shard_values):
            raise RuntimeError(f"{label} has an invalid weight index: {index_path}")
        shard_names = set(shard_values)
        relative_shards = {PurePosixPath(name) for name in shard_names}
        if any(
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".safetensors"
            for path in relative_shards
        ):
            raise RuntimeError(f"{label} has an unsafe weight index: {index_path}")
        shards = {snapshot.joinpath(*path.parts) for path in relative_shards}
        missing_shards = tuple(sorted(str(path) for path in shards if not path.is_file()))
        if missing_shards:
            raise FileNotFoundError(
                f"{label} snapshot is missing indexed weight shards: "
                + ", ".join(missing_shards)
            )


def resolve_target_snapshot(recipe: RecipeSpec, hf_home: Path) -> Path:
    """Resolve and validate the immutable target snapshot behind ``refs/main``."""
    ref_path = recipe.target_ref_path(hf_home)
    if not ref_path.is_file():
        raise FileNotFoundError(f"Target model ref is missing: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    if _FULL_SHA.fullmatch(revision) is None:
        raise RuntimeError(
            f"Target model ref is not a full immutable revision: {ref_path}"
        )
    if revision != recipe.target_revision:
        raise RuntimeError(
            f"Target model ref moved: expected {recipe.target_revision}, got "
            f"{revision} in {ref_path}"
        )
    snapshot = ref_path.parent.parent / "snapshots" / revision
    validate_snapshot(snapshot, revision, "Target model")
    return snapshot


def validate_runtime_inputs(
    run: ResolvedRun,
    repo_dir: Path,
    container: Path,
) -> tuple[Path, Path | None]:
    """Validate immutable runtime artifacts before contacting SLURM."""
    if not container.is_file():
        raise FileNotFoundError(f"Container image is missing: {container}")
    if not (repo_dir / "ray.sub").is_file():
        raise FileNotFoundError(f"ray.sub is missing from checkout: {repo_dir}")
    target_snapshot = resolve_target_snapshot(run.recipe, run.cluster.hf_home)
    draft_snapshot = None
    if run.draft_checkpoint is not None:
        draft_snapshot = run.draft_checkpoint.snapshot_path(run.cluster.hf_home)
        validate_snapshot(
            draft_snapshot,
            run.draft_checkpoint.revision,
            "Draft model",
        )
    return target_snapshot, draft_snapshot


def validate_run_destination(
    repo_dir: Path,
    experiment_root: Path,
    run_tag: str,
    *,
    require_lustre: bool,
) -> Path:
    """Resolve a fresh run directory that cannot escape its external root."""
    if _RUN_TAG.fullmatch(run_tag) is None or run_tag in {".", ".."}:
        raise ValueError(
            "run tag must be one simple alphanumeric/dot/dash/underscore name"
        )
    root = experiment_root.resolve()
    run_dir = (root / run_tag).resolve()
    if require_lustre and not root.is_relative_to(G_LUSTRE_ROOT):
        raise ValueError(f"Experiment root must be on Lustre: {root}")
    if run_dir.parent != root:
        raise ValueError(f"Run tag escapes experiment root: {run_tag}")
    if run_dir.is_relative_to(repo_dir.resolve()):
        raise ValueError(f"Run directory must remain outside the checkout: {run_dir}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir


def build_submission_environment(
    public_environment: Mapping[str, str],
    command: str,
    source_environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Allowlist scheduler inputs while forwarding credentials by name only."""
    source = os.environ if source_environment is None else source_environment
    environment = {
        key: value for key, value in source.items() if key in G_ALLOWED_ENVIRONMENT
    }
    environment.update(public_environment)
    environment["COMMAND"] = command
    forwarded_secrets = tuple(
        sorted(key for key in G_SECRET_ENVIRONMENT if key in environment)
    )
    return environment, forwarded_secrets


def _flatten_provenance(
    value: Any, prefix: str = ""
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_provenance(value[key], next_prefix))
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, json.dumps(value, sort_keys=True)))
    else:
        rows.append((prefix, "" if value is None else str(value)))
    return rows


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_provenance(run_dir: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist machine- and human-readable run provenance."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if "WANDB_API_KEY" in serialized:
        raise ValueError("Secrets must not be written to provenance")
    text = "".join(
        f"{key}={value}\n" for key, value in _flatten_provenance(payload)
    )
    _atomic_write(run_dir / "provenance.json", serialized)
    _atomic_write(run_dir / "provenance.txt", text)


def _run_tag(model: str, variant: str, phase: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    return f"{model}-v0251-{variant}-{phase}-{timestamp}"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=tuple(r.key for r in G_RECIPES))
    parser.add_argument(
        "--variant", required=True, choices=tuple(v.key for v in G_VARIANTS)
    )
    parser.add_argument("--phase", default="smoke2", choices=tuple(p.key for p in G_PHASES))
    parser.add_argument("--cluster", default="lyris", choices=tuple(c.key for c in G_CLUSTERS))
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-root", type=Path, default=G_DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--container", type=Path, default=G_DEFAULT_CONTAINER)
    parser.add_argument("--mounts", default="/lustre:/lustre")
    parser.add_argument("--run-tag")


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic matrix command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("show", "test-only", "submit"):
        _add_common_arguments(subparsers.add_parser(action))
    return parser


def _show_record(
    run: ResolvedRun,
    repo_dir: Path,
    run_dir: Path,
    run_tag: str,
    container: Path,
    mounts: str,
) -> dict[str, Any]:
    scheduler = build_scheduler_command(run, repo_dir, run_dir, "test-only")
    return {
        "run_tag": run_tag,
        "model": run.recipe.key,
        "variant": run.variant.key,
        "phase": run.phase.key,
        "runner": run.variant.runner,
        "recipe": run.recipe.path,
        "target_repo_id": run.recipe.target_repo_id,
        "draft_repo_id": (
            run.draft_checkpoint.repo_id if run.draft_checkpoint is not None else None
        ),
        "draft_revision": (
            run.draft_checkpoint.revision if run.draft_checkpoint is not None else None
        ),
        "container": str(container),
        "mounts": mounts,
        "run_dir": str(run_dir),
        "runtime_command": list(build_runtime_command(run, repo_dir, run_dir, run_tag)),
        "scheduler_command": list(scheduler),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve, preflight, or submit one matrix entry."""
    args = build_parser().parse_args(argv)
    run = resolve_run(args.model, args.variant, args.phase, args.cluster)
    run_tag = args.run_tag or _run_tag(args.model, args.variant, args.phase)
    repo_dir = args.repo_dir.resolve()
    run_dir = validate_run_destination(
        repo_dir,
        args.experiment_root,
        run_tag,
        require_lustre=args.action != "show",
    )
    container = args.container.resolve()
    show_record = _show_record(
        run,
        repo_dir,
        run_dir,
        run_tag,
        container,
        args.mounts,
    )
    if args.action == "show":
        print(json.dumps(show_record, indent=2, sort_keys=True))
        return

    checkout = validate_checkout(repo_dir)
    target_snapshot, draft_snapshot = validate_runtime_inputs(
        run, repo_dir, container
    )
    runtime_id = uuid.uuid4().hex
    runtime_command = build_runtime_command(
        run,
        repo_dir,
        run_dir,
        run_tag,
        target_snapshot=target_snapshot,
        runtime_id=runtime_id,
    )
    scheduler_sequence = build_scheduler_sequence(
        run, repo_dir, run_dir, args.action
    )
    safe_environment = {
        "BASE_LOG_DIR": str(run_dir),
        "CONTAINER": str(container),
        "GPUS_PER_NODE": str(run.cluster.gpus_per_node),
        "HF_HOME": str(run.cluster.hf_home),
        "MOUNTS": args.mounts,
    }
    environment, forwarded_secret_names = build_submission_environment(
        safe_environment,
        shlex.join(runtime_command),
    )
    forwarded_environment_names = tuple(
        sorted(
            key
            for key in environment
            if key not in G_SECRET_ENVIRONMENT and key != "COMMAND"
        )
    )
    provenance: dict[str, Any] = {
        **show_record,
        "checkout": {
            "branch": checkout.branch,
            "head": checkout.head,
            "fork_ref": checkout.fork_ref,
            "submodules": checkout.submodules,
        },
        "target_snapshot": str(target_snapshot),
        "draft_snapshot": str(draft_snapshot) if draft_snapshot else None,
        "hydra_overrides": run.hydra_overrides,
        "runtime_id": runtime_id,
        "runtime_command": runtime_command,
        "scheduler_commands": scheduler_sequence,
        "environment": safe_environment,
        "forwarded_environment_names": forwarded_environment_names,
        "wandb_credential_forwarded": bool(forwarded_secret_names),
        "submission_state": "preflight",
    }
    write_provenance(run_dir, provenance)
    preflight_result = subprocess.run(
        scheduler_sequence[0],
        cwd=repo_dir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if preflight_result.stderr:
        print(preflight_result.stderr, end="", file=sys.stderr)
    provenance["submission_state"] = "preflight-passed"
    provenance["preflight_output"] = preflight_result.stdout.strip()
    write_provenance(run_dir, provenance)
    if args.action == "test-only":
        print(preflight_result.stdout, end="")
        return
    provenance["submission_state"] = "submitting"
    write_provenance(run_dir, provenance)
    result = subprocess.run(
        scheduler_sequence[1],
        cwd=repo_dir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse SLURM job ID: {result.stdout!r}")
    provenance["submission_state"] = "submitted"
    provenance["job_id"] = job_id
    write_provenance(run_dir, provenance)
    print(f"job_id={job_id}\nrun_dir={run_dir}")


if __name__ == "__main__":
    main()
