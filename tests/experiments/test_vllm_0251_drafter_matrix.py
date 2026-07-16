from dataclasses import FrozenInstanceError

import pytest

from experiments.vllm_0251_drafter_matrix.matrix import resolve_run


@pytest.mark.parametrize(
    ("model_key", "nodes", "segment", "osl"),
    [
        ("qwen30", 4, 4, 4096),
        ("qwen32", 4, 4, 4096),
        ("qwen235", 16, 16, 8192),
    ],
)
def test_recipe_topology_is_authoritative(
    model_key: str, nodes: int, segment: int, osl: int
) -> None:
    run = resolve_run(model_key, "baseline", "smoke2", "lyris")

    assert (run.recipe.nodes, run.recipe.segment, run.recipe.max_osl) == (
        nodes,
        segment,
        osl,
    )


@pytest.mark.parametrize(
    ("phase", "max_steps"),
    [("smoke2", 2), ("smoke5", 5), ("final20", 20)],
)
def test_phase_controls_only_the_step_count(phase: str, max_steps: int) -> None:
    run = resolve_run("qwen30", "baseline", phase, "lyris")

    assert run.phase.max_steps == max_steps
    assert f"grpo.max_num_steps={max_steps}" in run.hydra_overrides


@pytest.mark.parametrize("model_key", ["qwen30", "qwen32", "qwen235"])
def test_baseline_uses_full_cuda_graphs_and_preserves_recipe_controls(
    model_key: str,
) -> None:
    run = resolve_run(model_key, "baseline", "smoke2", "lyris")

    assert run.variant.runner == "mrv2"
    assert "policy.generation.vllm_cfg.enforce_eager=false" in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode="
        "FULL_AND_PIECEWISE"
    ) in run.hydra_overrides
    assert "checkpointing.enabled=false" in run.hydra_overrides
    assert not any("speculative_config" in item for item in run.hydra_overrides)
    assert not any("cudagraph_capture_sizes" in item for item in run.hydra_overrides)
    assert not any("max_num_batched_tokens" in item for item in run.hydra_overrides)
    assert not any("policy.model_name" in item for item in run.hydra_overrides)
    assert not any("tensor_parallel_size" in item for item in run.hydra_overrides)


def test_pard_selects_mrv1_and_parallel_drafting() -> None:
    run = resolve_run("qwen32", "pard_k5", "smoke2", "lyris")

    assert run.variant.runner == "mrv1"
    assert (
        "++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true"
        in run.hydra_overrides
    )
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in run.command_parts()


def test_eagle_selects_mrv2_without_compact_capture_sizes() -> None:
    run = resolve_run("qwen30", "eagle3_k3", "smoke2", "lyris")

    assert run.variant.runner == "mrv2"
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in run.command_parts()
    assert all("cudagraph_capture_sizes" not in item for item in run.hydra_overrides)


@pytest.mark.parametrize(
    ("variant_key", "method", "runner", "tokens"),
    [
        ("eagle3_k1", "eagle3", "mrv2", 1),
        ("eagle3_k3", "eagle3", "mrv2", 3),
        ("eagle3_k5", "eagle3", "mrv2", 5),
        ("draft_k1", "draft_model", "mrv1", 1),
        ("draft_k5", "draft_model", "mrv1", 5),
        ("pard_k5", "draft_model", "mrv1", 5),
        ("pard_k16", "draft_model", "mrv1", 16),
        ("suffix_k32", "suffix", "mrv1", 32),
        ("ngram_k5", "ngram", "mrv1", None),
        ("ngram_gpu_k5", "ngram_gpu", "mrv1", None),
    ],
)
def test_supported_variants_emit_official_vllm_overrides(
    variant_key: str, method: str, runner: str, tokens: int | None
) -> None:
    run = resolve_run("qwen32", variant_key, "smoke2", "lyris")

    assert run.variant.method == method
    assert run.variant.runner == runner
    assert (
        f"++policy.generation.vllm_kwargs.speculative_config.method={method}"
        in run.hydra_overrides
    )
    if tokens is not None:
        assert (
            "++policy.generation.vllm_kwargs.speculative_config."
            f"num_speculative_tokens={tokens}"
        ) in run.hydra_overrides


@pytest.mark.parametrize("model_key", ["qwen30", "qwen32", "qwen235"])
def test_model_based_drafters_use_draft_tensor_parallelism_one(
    model_key: str,
) -> None:
    for variant_key in ("eagle3_k1", "draft_k1", "pard_k5"):
        run = resolve_run(model_key, variant_key, "smoke2", "lyris")
        assert (
            "++policy.generation.vllm_kwargs.speculative_config."
            "draft_tensor_parallel_size=1"
        ) in run.hydra_overrides


def test_suffix_and_ngram_variants_use_their_native_controls() -> None:
    suffix = resolve_run("qwen235", "suffix_k32", "smoke2", "lyris")
    ngram = resolve_run("qwen235", "ngram_k5", "smoke2", "lyris")
    ngram_gpu = resolve_run("qwen235", "ngram_gpu_k5", "smoke2", "lyris")

    assert (
        "++policy.generation.vllm_kwargs.speculative_config."
        "suffix_decoding_max_tree_depth=32"
    ) in suffix.hydra_overrides
    for run in (ngram, ngram_gpu):
        assert (
            "++policy.generation.vllm_kwargs.speculative_config.prompt_lookup_min=5"
            in run.hydra_overrides
        )
        assert (
            "++policy.generation.vllm_kwargs.speculative_config.prompt_lookup_max=5"
            in run.hydra_overrides
        )


@pytest.mark.parametrize("model_key", ["qwen30", "qwen32", "qwen235"])
@pytest.mark.parametrize("variant_key", ["dflash_k3", "dflash_k5"])
def test_dflash_rejects_models_without_an_exact_checkpoint(
    model_key: str, variant_key: str
) -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_run(model_key, variant_key, "smoke2", "lyris")


def test_lyris_scheduler_uses_recipe_topology_without_gres() -> None:
    run = resolve_run("qwen235", "baseline", "smoke2", "lyris")
    sbatch_parts = run.sbatch_parts()

    assert "--nodes=16" in sbatch_parts
    assert "--segment=16" in sbatch_parts
    assert run.cluster.gpus_per_node == 4
    assert not any(part.startswith("--gres") for part in sbatch_parts)


@pytest.mark.parametrize(
    ("model_key", "variant_key", "phase", "cluster"),
    [
        ("missing", "baseline", "smoke2", "lyris"),
        ("qwen30", "missing", "smoke2", "lyris"),
        ("qwen30", "baseline", "missing", "lyris"),
        ("qwen30", "baseline", "smoke2", "missing"),
    ],
)
def test_unknown_resolution_inputs_are_rejected(
    model_key: str, variant_key: str, phase: str, cluster: str
) -> None:
    with pytest.raises(ValueError):
        resolve_run(model_key, variant_key, phase, cluster)


def test_resolved_records_are_immutable() -> None:
    run = resolve_run("qwen30", "baseline", "smoke2", "lyris")

    with pytest.raises(FrozenInstanceError):
        run.recipe.nodes = 99  # type: ignore[misc]
