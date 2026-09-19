"""Contract tests for the matched Async-1off SpecDec launcher."""

from pathlib import Path

import pytest

from experiments.q30_q235_async1off_specdec_20260918.launch import (
    ARMS,
    MODELS,
    OVERLAY_BUILDER,
    SOURCE,
    capture_sizes,
    configuration,
    render,
    required_inputs,
    sbatch_arguments,
)


@pytest.mark.parametrize(
    ("model", "recipe", "nodes"),
    [
        ("q30", "grpo-qwen3-30ba3b-4n4g-async-1off.yaml", 4),
        ("q235", "grpo-qwen3-235b-32n4g-async-1off.yaml", 32),
    ],
)
def test_model_specs_use_official_async_1off_recipes(
    model: str, recipe: str, nodes: int
) -> None:
    spec = MODELS[model]

    assert spec.recipe.name == recipe
    assert spec.nodes == nodes
    assert spec.gpus_per_node == 4


@pytest.mark.parametrize("model", ["q30", "q235"])
def test_baseline_has_matched_s64_cuda_graph_contract(model: str) -> None:
    config = configuration(model=model, arm="baseline", steps=20)

    assert config["grpo.max_num_steps"] == "20"
    assert config["policy.generation.vllm_kwargs.max_num_seqs"] == "64"
    assert config["policy.generation.vllm_cfg.enforce_eager"] == "false"
    assert config["policy.generation.vllm_kwargs.moe_backend"] == ("flashinfer_trtllm")
    assert (
        config["policy.generation.vllm_kwargs.compilation_config.cudagraph_mode"]
        == "FULL_AND_PIECEWISE"
    )
    assert (
        config[
            "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
        ]
        == "[1,2,4,8,16,32,64]"
    )
    assert config["policy.generation.vllm_kwargs.speculative_config"] == "null"
    assert "policy.generation.refit_cfg.memory_lifecycle.mode" not in config


@pytest.mark.parametrize(
    ("model", "arm", "method", "k", "checkpoint_fragment"),
    [
        ("q30", "dflash_k5", "dflash", 5, "q30-base-ptv3swe-dflash"),
        ("q30", "dspark_k5", "dspark", 5, "q30-base-ptv3swe-dspark"),
        ("q235", "dflash_k7", "dflash", 7, "q235-base-ptv2en"),
        ("q235", "dspark_k7", "dspark", 7, "q235-base-ptv3rp25-dspark"),
    ],
)
def test_specdec_arms_add_only_runtime_speculation_contract(
    model: str,
    arm: str,
    method: str,
    k: int,
    checkpoint_fragment: str,
) -> None:
    config = configuration(model=model, arm=arm, steps=20)

    assert config["policy.generation.vllm_kwargs.max_num_seqs"] == "64"
    assert config["policy.generation.vllm_kwargs.speculative_config.method"] == method
    assert config[
        "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens"
    ] == str(k)
    assert (
        checkpoint_fragment
        in config["policy.generation.vllm_kwargs.speculative_config.model"]
    )
    assert (
        config[
            "policy.generation.vllm_kwargs.speculative_config."
            "draft_tensor_parallel_size"
        ]
        == "1"
    )
    assert (
        config["policy.generation.refit_cfg.memory_lifecycle.mode"]
        == "specdec_deep_refit"
    )
    assert "policy.generation.vllm_kwargs.speculative_config" not in config


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("baseline", "[1,2,4,8,16,32,64]"),
        ("dflash_k5", "[1,2,4,6,12,24,48,96,192,384]"),
        (
            "dspark_k5",
            "[1,2,4,5,6,12,18,24,36,48,78,96,156,192,318,320,384]",
        ),
        ("dflash_k7", "[1,2,4,8,16,32,64,128,256,512]"),
        (
            "dspark_k7",
            "[1,2,4,7,8,16,24,32,56,64,112,128,224,256,448,512]",
        ),
    ],
)
def test_capture_sizes_cover_request_and_method_widths(arm: str, expected: str) -> None:
    assert capture_sizes(ARMS[arm]) == expected


def test_model_rejects_an_arm_from_the_other_model() -> None:
    with pytest.raises(ValueError, match="not valid"):
        configuration(model="q30", arm="dflash_k7", steps=20)


def test_renderer_preserves_model_topology_and_node_local_caches() -> None:
    q30 = render(
        account="coreai_dlalgo_nemorl",
        run_name="Qwen3-30BA3B-Async1off-Baseline-S64-1step-test",
        model="q30",
        arm="baseline",
        steps=1,
        directory=Path("/lustre/test/q30"),
    )
    q235 = render(
        account="coreai_dlalgo_nemorl",
        run_name="Qwen3-235B-Async1off-DSparkK7-S64-1step-test",
        model="q235",
        arm="dspark_k7",
        steps=1,
        directory=Path("/lustre/test/q235"),
    )

    assert "#SBATCH --nodes=4" in q30
    assert "#SBATCH --nodes=32" in q235
    assert "grpo-qwen3-30ba3b-4n4g-async-1off.yaml" in q30
    assert "grpo-qwen3-235b-32n4g-async-1off.yaml" in q235
    assert "max_num_seqs=64" in q30
    assert "max_num_seqs=64" in q235
    assert "FULL_AND_PIECEWISE" in q30
    assert "FULL_AND_PIECEWISE" in q235
    assert "/raid/scratch/sna/async1off-${SLURM_JOB_ID}" in q30
    assert "/raid/scratch/sna/async1off-${SLURM_JOB_ID}" in q235
    assert "--dependency" not in q30
    assert "--dependency" not in q235


def test_sbatch_arguments_always_validate_before_independent_submission() -> None:
    job = Path("/tmp/async1off/job.sbatch")

    assert sbatch_arguments(job, test_only=True) == [
        "sbatch",
        "--test-only",
        str(job),
    ]
    assert sbatch_arguments(job, test_only=False) == [
        "sbatch",
        "--parsable",
        str(job),
    ]


def test_dspark_overlay_dependencies_are_versioned_with_the_launcher() -> None:
    inputs = required_inputs(model="q30", arm="dspark_k5")
    repository_root = Path(__file__).resolve().parents[3]
    overlay_inputs = [
        path
        for path in inputs
        if path == OVERLAY_BUILDER or path.parent.name == "patches"
    ]

    assert len(overlay_inputs) == 3
    assert all(
        (repository_root / path.relative_to(SOURCE)).is_file()
        for path in overlay_inputs
    )
