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

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """Authoritative topology and recipe path for one controlled model."""

    key: str
    path: str
    nodes: int
    segment: int
    max_osl: int


@dataclass(frozen=True, slots=True)
class ClusterSpec:
    """Scheduler metadata for a supported cluster."""

    key: str
    account: str
    partition: str
    gpus_per_node: int


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


G_MODEL_KEYS = frozenset(("qwen30", "qwen32", "qwen235"))

G_RECIPES = (
    RecipeSpec(
        key="qwen30",
        path="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        nodes=4,
        segment=4,
        max_osl=4096,
    ),
    RecipeSpec(
        key="qwen32",
        path="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
        nodes=4,
        segment=4,
        max_osl=4096,
    ),
    RecipeSpec(
        key="qwen235",
        path="examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml",
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
        repo_id="inference-optimization/Qwen3-30B-A3B-speculator.dflash",
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
    variant: VariantSpec, draft_checkpoint: CheckpointSpec | None
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
                f"{prefix}.model={draft_checkpoint.repo_id}",
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
        + _speculative_overrides(variant, draft_checkpoint),
    )
