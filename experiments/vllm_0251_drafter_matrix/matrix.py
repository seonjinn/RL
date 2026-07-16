"""Typed resolution for the vLLM 0.25.1 Qwen drafter matrix."""

from __future__ import annotations

from dataclasses import dataclass


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
class VariantSpec:
    """An official vLLM speculative-decoding configuration."""

    key: str
    method: str | None
    runner: str
    num_speculative_tokens: int | None
    compatible_models: frozenset[str]
    checkpoints: tuple[tuple[str, str], ...] = ()
    uses_draft_model: bool = False
    parallel_drafting: bool = False
    suffix_tree_depth: int | None = None
    ngram_size: int | None = None

    def checkpoint_for(self, model_key: str) -> str | None:
        """Return the exact drafter checkpoint for a compatible model."""
        for checkpoint_model_key, checkpoint in self.checkpoints:
            if checkpoint_model_key == model_key:
                return checkpoint
        return None


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """Fully validated matrix entry with scheduler and Hydra arguments."""

    recipe: RecipeSpec
    cluster: ClusterSpec
    phase: PhaseSpec
    variant: VariantSpec
    hydra_overrides: tuple[str, ...]

    def command_parts(self) -> tuple[str, ...]:
        """Return the deterministic training command as an argument tuple."""
        model_runner_v2 = "1" if self.variant.runner == "mrv2" else "0"
        return (
            "env",
            f"VLLM_USE_V2_MODEL_RUNNER={model_runner_v2}",
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
        partition="batch",
        gpus_per_node=4,
    ),
)

G_PHASES = (
    PhaseSpec(key="smoke2", max_steps=2, time_limit="01:00:00"),
    PhaseSpec(key="smoke5", max_steps=5, time_limit="02:00:00"),
    PhaseSpec(key="final20", max_steps=20, time_limit="05:00:00"),
)

G_EAGLE3_CHECKPOINTS = (
    (
        "qwen30",
        "RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3",
    ),
    ("qwen32", "RedHatAI/Qwen3-32B-speculator.eagle3"),
    ("qwen235", "nvidia/Qwen3-235B-A22B-Eagle3"),
)

G_PARD_CHECKPOINTS = (
    ("qwen30", "amd/PARD-Qwen3-0.6B"),
    ("qwen32", "amd/PARD-Qwen3-0.6B"),
    ("qwen235", "amd/PARD-Qwen3-0.6B"),
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
        compatible_models=frozenset(),
        uses_draft_model=True,
    ),
    VariantSpec(
        key="dflash_k5",
        method="dflash",
        runner="mrv2",
        num_speculative_tokens=5,
        compatible_models=frozenset(),
        uses_draft_model=True,
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
        num_speculative_tokens=None,
        compatible_models=G_MODEL_KEYS,
        ngram_size=5,
    ),
    VariantSpec(
        key="ngram_gpu_k5",
        method="ngram_gpu",
        runner="mrv1",
        num_speculative_tokens=None,
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


def _speculative_overrides(variant: VariantSpec, model_key: str) -> tuple[str, ...]:
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
        checkpoint = variant.checkpoint_for(model_key)
        if checkpoint is None:
            raise ValueError(
                f"Variant '{variant.key}' has no exact checkpoint for model "
                f"'{model_key}'"
            )
        overrides.extend(
            (
                f"{prefix}.model={checkpoint}",
                f"{prefix}.draft_tensor_parallel_size=1",
            )
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
        hydra_overrides=base_overrides + _speculative_overrides(variant, model_key),
    )
