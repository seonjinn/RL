import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.matrix import (
    G_FORK_URLS,
    OptimizerOffloadMode,
    RecipeSpec,
    build_runtime_command,
    build_scheduler_command,
    build_scheduler_sequence,
    build_submission_environment,
    load_login_wandb_environment,
    resolve_run,
    resolve_target_snapshot,
    validate_megatron_checkpoint_cache,
    validate_run_destination,
    validate_snapshot,
    validate_checkout,
    write_provenance,
)


def test_matrix_module_has_nvidia_apache_header() -> None:
    matrix_path = (
        Path(__file__).parents[2] / "experiments/vllm_0251_drafter_matrix/matrix.py"
    )

    assert matrix_path.read_text().startswith(
        "# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.\n"
        "#\n"
        '# Licensed under the Apache License, Version 2.0 (the "License");\n'
    )


def test_approved_fork_urls_cover_local_alias_and_cluster_canonical_host() -> None:
    assert G_FORK_URLS == {
        "git@github-seonjinn:seonjinn/RL.git",
        "git@github.com:seonjinn/RL.git",
    }


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


def test_qwen235_long_context_override_updates_resolved_osl_and_hydra() -> None:
    run = resolve_run(
        "qwen235",
        "baseline",
        "smoke2",
        "lyris",
        max_osl=32768,
    )

    assert run.recipe.max_osl == 32768
    assert "policy.max_total_sequence_length=32768" in run.hydra_overrides


@pytest.mark.parametrize("max_osl", [0, 40961])
def test_qwen235_long_context_override_rejects_invalid_limits(max_osl: int) -> None:
    with pytest.raises(ValueError, match="max OSL"):
        resolve_run(
            "qwen235",
            "baseline",
            "smoke2",
            "lyris",
            max_osl=max_osl,
        )


def test_recipe_targets_use_expected_immutable_revisions() -> None:
    assert {
        model: resolve_run(model, "baseline", "smoke2", "lyris").recipe.target_revision
        for model in ("qwen30", "qwen32", "qwen235")
    } == {
        "qwen30": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "qwen32": "9216db5781bf21249d130ec9da846c4624c16137",
        "qwen235": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
    }


@pytest.mark.parametrize(
    ("phase", "max_steps"),
    [("smoke2", 2), ("smoke5", 5), ("final20", 20)],
)
def test_phase_controls_only_the_step_count(phase: str, max_steps: int) -> None:
    run = resolve_run("qwen30", "baseline", phase, "lyris")

    assert run.phase.max_steps == max_steps
    assert f"grpo.max_num_steps={max_steps}" in run.hydra_overrides


@pytest.mark.parametrize(
    ("offload_mode", "use_pinned", "use_coalesced"),
    [
        ("pageable", "false", "false"),
        ("coalesced-pinned", "true", "true"),
    ],
)
def test_optimizer_offload_ab_is_an_explicit_single_variable(
    offload_mode: OptimizerOffloadMode,
    use_pinned: str,
    use_coalesced: str,
) -> None:
    run = resolve_run(
        "qwen235",
        "eagle3_thinking_k3",
        "smoke5",
        "lyris",
        optimizer_offload_mode=offload_mode,
    )

    assert run.optimizer_offload_mode == offload_mode
    assert f"++policy.use_pinned_optimizer_offload={use_pinned}" in run.hydra_overrides
    assert (
        f"++policy.use_coalesced_optimizer_offload={use_coalesced}"
        in run.hydra_overrides
    )


def test_optimizer_offload_ab_enables_identical_rank_diagnostics(
    tmp_path: Path,
) -> None:
    commands = []
    for offload_mode in ("pageable", "coalesced-pinned"):
        run = resolve_run(
            "qwen235",
            "eagle3_thinking_k3",
            "smoke5",
            "lyris",
            optimizer_offload_mode=offload_mode,
        )
        commands.append(
            build_runtime_command(
                run,
                tmp_path / "repo",
                tmp_path / "runs" / offload_mode,
                f"unit-{offload_mode}",
            )
        )

    assert all("NRL_REFIT_OFFLOAD_DIAGNOSTICS=1" in command for command in commands)


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


def test_baselines_select_fair_runner_environments() -> None:
    baseline_mrv2 = resolve_run("qwen30", "baseline", "smoke2", "lyris")
    baseline_mrv1 = resolve_run("qwen30", "baseline_mrv1", "smoke2", "lyris")

    assert baseline_mrv2.variant.runner == "mrv2"
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in baseline_mrv2.command_parts()
    assert baseline_mrv1.variant.runner == "mrv1"
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in baseline_mrv1.command_parts()
    assert not any(
        "speculative_config" in item for item in baseline_mrv1.hydra_overrides
    )


def test_pard_selects_mrv1_and_parallel_drafting() -> None:
    run = resolve_run("qwen32", "pard_k5", "smoke2", "lyris")

    assert run.variant.runner == "mrv1"
    assert (
        "++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true"
        in run.hydra_overrides
    )
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in run.command_parts()


@pytest.mark.parametrize(
    ("variant_key", "k", "capture_endpoint"),
    [
        ("pard_k5_cg384", 5, 384),
        ("pard_k7_cg512", 7, 512),
        ("pard_k16_cg1088", 16, 1088),
    ],
)
def test_qwen235_pard_cuda_graph_variants_cover_verification_shapes(
    variant_key: str,
    k: int,
    capture_endpoint: int,
) -> None:
    run = resolve_run("qwen235", variant_key, "smoke2", "lyris")

    assert run.variant.runner == "mrv1"
    assert run.variant.num_speculative_tokens == k
    assert run.variant.cudagraph_capture_sizes[-1] == capture_endpoint
    assert (
        "++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true"
        in run.hydra_overrides
    )
    assert any(
        item.endswith(f",{capture_endpoint}]")
        for item in run.hydra_overrides
        if "cudagraph_capture_sizes" in item
    )
    assert (
        "++policy.generation.vllm_kwargs.speculative_config."
        "draft_tensor_parallel_size=8"
    ) in run.hydra_overrides
    assert "++policy.generation.vllm_kwargs.max_num_seqs=64" in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.max_num_scheduled_tokens=2048"
        in run.hydra_overrides
    )
    assert (
        f"++policy.generation.vllm_kwargs.max_num_batched_tokens={2048 + 64 * k}"
        in run.hydra_overrides
    )
    assert (
        "++policy.generation.vllm_kwargs.compilation_config.pass_config.enable_sp=false"
    ) in run.hydra_overrides


def test_qwen235_pard_matched_baseline_uses_the_same_scheduler_budget() -> None:
    run = resolve_run(
        "qwen235",
        "baseline_mrv1_sched64",
        "smoke2",
        "lyris",
    )

    assert run.variant.runner == "mrv1"
    assert "++policy.generation.vllm_kwargs.max_num_seqs=64" in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.max_num_scheduled_tokens=2048"
        in run.hydra_overrides
    )
    assert (
        "++policy.generation.vllm_kwargs.max_num_batched_tokens=2048"
        in run.hydra_overrides
    )
    assert not any("speculative_config" in item for item in run.hydra_overrides)


def test_eagle_selects_mrv2_without_compact_capture_sizes() -> None:
    run = resolve_run("qwen30", "eagle3_k3", "smoke2", "lyris")

    assert run.variant.runner == "mrv2"
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in run.command_parts()
    assert all("cudagraph_capture_sizes" not in item for item in run.hydra_overrides)


def test_qwen235_thinking_k3_cg256_captures_the_full_generation_batch() -> None:
    run = resolve_run("qwen235", "eagle3_thinking_k3_cg256", "smoke2", "lyris")

    assert run.draft_checkpoint is not None
    assert (
        run.draft_checkpoint.repo_id
        == "RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3"
    )
    assert run.variant.num_speculative_tokens == 3
    assert (
        "policy.generation.vllm_kwargs.compilation_config."
        "cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,192,256]"
    ) in run.hydra_overrides


def test_qwen235_thinking_k5_cg384_captures_the_full_generation_batch() -> None:
    run = resolve_run("qwen235", "eagle3_thinking_k5_cg384", "smoke2", "lyris")

    assert run.draft_checkpoint is not None
    assert (
        run.draft_checkpoint.repo_id
        == "RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3"
    )
    assert run.variant.num_speculative_tokens == 5
    assert (
        "policy.generation.vllm_kwargs.compilation_config."
        "cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,192,256,320,384]"
    ) in run.hydra_overrides


@pytest.mark.parametrize(
    ("variant_key", "method", "runner", "tokens"),
    [
        ("eagle3_k1", "eagle3", "mrv2", 1),
        ("eagle3_k3", "eagle3", "mrv2", 3),
        ("eagle3_k5", "eagle3", "mrv2", 5),
        ("eagle3_thinking_k1", "eagle3", "mrv2", 1),
        ("eagle3_thinking_k2", "eagle3", "mrv2", 2),
        ("eagle3_thinking_k3", "eagle3", "mrv2", 3),
        ("eagle3_thinking_k4", "eagle3", "mrv2", 4),
        ("eagle3_thinking_k5", "eagle3", "mrv2", 5),
        ("draft_k1", "draft_model", "mrv1", 1),
        ("draft_k5", "draft_model", "mrv1", 5),
        ("pard_k5", "draft_model", "mrv1", 5),
        ("pard_k16", "draft_model", "mrv1", 16),
        ("suffix_k32", "suffix", "mrv1", 32),
        ("ngram_k5", "ngram", "mrv1", 5),
        ("ngram_gpu_k5", "ngram_gpu", "mrv1", 5),
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


@pytest.mark.parametrize(
    ("model_key", "repo_id", "revision"),
    [
        (
            "qwen30",
            "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
            "edcff83783141eb9383e2bd6c33610d9a3104288",
        ),
        (
            "qwen32",
            "AICP-Labs/qwen3-32b-dflash-en-zh",
            "68ccc7fd27b104271321b179a2959c759dce5eef",
        ),
    ],
)
@pytest.mark.parametrize("variant_key", ["dflash_k3", "dflash_k5"])
def test_dflash_uses_exact_checkpoint_and_draft_flash_attention(
    model_key: str, repo_id: str, revision: str, variant_key: str
) -> None:
    run = resolve_run(model_key, variant_key, "smoke2", "lyris")

    assert run.draft_checkpoint is not None
    assert run.draft_checkpoint.repo_id == repo_id
    assert run.draft_checkpoint.revision == revision
    assert run.draft_checkpoint.snapshot_path(Path("/hf")) == (
        Path("/hf")
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    assert (
        "++policy.generation.vllm_kwargs.speculative_config."
        "attention_backend=FLASH_ATTN"
    ) in run.hydra_overrides
    assert not any(
        item.endswith("attention_backend=FLASH_ATTN")
        and "speculative_config" not in item
        for item in run.hydra_overrides
    )


@pytest.mark.parametrize(
    ("model_key", "variant_key"),
    [
        ("qwen30", "eagle3_k5"),
        ("qwen32", "dflash_k5"),
        ("qwen235", "draft_k5"),
    ],
)
def test_draft_model_override_uses_cluster_immutable_snapshot(
    model_key: str, variant_key: str
) -> None:
    run = resolve_run(model_key, variant_key, "smoke2", "lyris")

    assert run.draft_checkpoint is not None
    assert run.cluster.hf_home == Path(
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
    )
    snapshot = run.draft_checkpoint.snapshot_path(run.cluster.hf_home)
    assert (
        f"++policy.generation.vllm_kwargs.speculative_config.model={snapshot}"
    ) in run.hydra_overrides
    assert (
        "++policy.generation.vllm_kwargs.speculative_config.model="
        f"{run.draft_checkpoint.repo_id}"
    ) not in run.hydra_overrides


@pytest.mark.parametrize("variant_key", ["dflash_k3", "dflash_k5"])
def test_dflash_rejects_qwen235_without_an_exact_checkpoint(variant_key: str) -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_run("qwen235", variant_key, "smoke2", "lyris")


@pytest.mark.parametrize(
    ("model_key", "repo_id", "revision"),
    [
        (
            "qwen30",
            "RedHatAI/Qwen3-30B-A3B-speculator.eagle3",
            "6afc5aa2477b923467fb9a8d906782b984a9a6ba",
        ),
        (
            "qwen32",
            "RedHatAI/Qwen3-32B-speculator.eagle3",
            "dc84fe7ff1db31efa824776f49c141fc8195eb47",
        ),
        (
            "qwen235",
            "nvidia/Qwen3-235B-A22B-Eagle3",
            "33f3c01ce807376d1171301b9a148b1b28f239ba",
        ),
    ],
)
def test_eagle3_exposes_exact_base_checkpoint_identity(
    model_key: str, repo_id: str, revision: str
) -> None:
    run = resolve_run(model_key, "eagle3_k5", "smoke2", "lyris")

    assert run.draft_checkpoint is not None
    assert run.draft_checkpoint.repo_id == repo_id
    assert run.draft_checkpoint.revision == revision
    assert run.draft_checkpoint.snapshot_path(Path("/hf")) == (
        Path("/hf")
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


@pytest.mark.parametrize(
    ("model_key", "repo_id", "revision"),
    [
        (
            "qwen32",
            "RedHatAI/Qwen3-32B-Thinking-speculator.eagle3",
            "a1403e07b73a66fc9ef561463631c31864616933",
        ),
        (
            "qwen235",
            "RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3",
            "3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87",
        ),
    ],
)
@pytest.mark.parametrize(
    ("variant_key", "tokens"),
    [
        ("eagle3_thinking_k1", 1),
        ("eagle3_thinking_k2", 2),
        ("eagle3_thinking_k3", 3),
        ("eagle3_thinking_k4", 4),
        ("eagle3_thinking_k5", 5),
    ],
)
def test_eagle3_thinking_exposes_exact_reasoning_checkpoint_identity(
    model_key: str,
    repo_id: str,
    revision: str,
    variant_key: str,
    tokens: int,
) -> None:
    run = resolve_run(model_key, variant_key, "smoke2", "lyris")

    assert run.variant.runner == "mrv2"
    assert run.variant.num_speculative_tokens == tokens
    assert run.draft_checkpoint is not None
    assert run.draft_checkpoint.repo_id == repo_id
    assert run.draft_checkpoint.revision == revision


@pytest.mark.parametrize(
    "variant_key",
    (
        "eagle3_thinking_k1",
        "eagle3_thinking_k2",
        "eagle3_thinking_k3",
        "eagle3_thinking_k4",
        "eagle3_thinking_k5",
    ),
)
def test_qwen30_rejects_duplicate_thinking_alias(variant_key: str) -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_run("qwen30", variant_key, "smoke2", "lyris")


@pytest.mark.parametrize(
    "variant_key",
    [
        "baseline",
        "eagle3_k1",
        "eagle3_k3",
        "eagle3_k5",
        "eagle3_thinking_k1",
        "eagle3_thinking_k2",
        "eagle3_thinking_k3",
        "eagle3_thinking_k4",
        "eagle3_thinking_k5",
        "draft_k1",
        "draft_k5",
        "pard_k5",
        "pard_k16",
        "suffix_k32",
        "ngram_k5",
        "ngram_gpu_k5",
    ],
)
def test_qwen235_keeps_official_capture_sizes_without_matrix_override(
    variant_key: str,
) -> None:
    run = resolve_run("qwen235", variant_key, "smoke2", "lyris")

    assert not any("cudagraph_capture_sizes" in item for item in run.hydra_overrides)


def test_lyris_scheduler_uses_recipe_topology_without_gres() -> None:
    run = resolve_run("qwen235", "baseline", "smoke2", "lyris")
    sbatch_parts = run.sbatch_parts()

    assert "--nodes=16" in sbatch_parts
    assert "--segment=16" in sbatch_parts
    assert "--dependency=" in sbatch_parts
    assert "--partition=gb200" in sbatch_parts
    assert run.cluster.partition == "gb200"
    assert run.cluster.gpus_per_node == 4
    assert not any(part.startswith("--gres") for part in sbatch_parts)


def test_command_exposes_cluster_gpu_count_to_ray_sub() -> None:
    run = resolve_run("qwen30", "baseline", "smoke2", "lyris")

    assert "GPUS_PER_NODE=4" in run.command_parts()


def test_sbatch_exports_cluster_gpu_count_to_ray_sub() -> None:
    run = resolve_run("qwen30", "baseline", "smoke2", "lyris")

    assert "--export=ALL,GPUS_PER_NODE=4" in run.sbatch_parts()


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


def test_runtime_command_uses_container_python_and_external_run_directory(
    tmp_path: Path,
) -> None:
    run = resolve_run("qwen32", "eagle3_k5", "smoke2", "lyris")
    repo_dir = tmp_path / "checkout"
    run_dir = tmp_path / "results" / "unit-run"

    command = build_runtime_command(run, repo_dir, run_dir, "unit-run")

    assert "/opt/nemo_rl_venv/bin/python" in command
    assert "python3" not in command
    assert f"PYTHONPATH={repo_dir}" in command
    assert f"checkpointing.checkpoint_dir={run_dir / 'checkpoints'}" in command
    assert f"logger.log_dir={run_dir / 'nemo_logs'}" in command
    assert "logger.wandb.project=nemo-rl-vllm0251-drafter-matrix" in command
    assert "logger.wandb.name=unit-run" in command
    assert not run_dir.is_relative_to(repo_dir)


def test_runtime_command_binds_the_validated_target_snapshot(tmp_path: Path) -> None:
    run = resolve_run("qwen30", "eagle3_k3", "smoke2", "lyris")
    target_snapshot = tmp_path / "hf" / "snapshots" / ("a" * 40)

    command = build_runtime_command(
        run,
        tmp_path / "repo",
        tmp_path / "runs" / "unit",
        "unit",
        target_snapshot=target_snapshot,
        runtime_id="runtime-123",
    )

    assert f"policy.model_name={target_snapshot}" in command
    assert f"policy.tokenizer.name={target_snapshot}" in command
    assert "NEMO_RL_VENV_DIR=/tmp/nemorl-v0251-runtime-123" in command
    assert "TRITON_CACHE_DIR=/tmp/nemorl-v0251-triton-runtime-123" in command
    assert "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v0251-inductor-runtime-123" in command


@pytest.mark.parametrize(
    ("model_key", "uses_qwen235_startup_workarounds"),
    [("qwen30", False), ("qwen32", False), ("qwen235", True)],
)
def test_runtime_command_uses_qwen235_startup_workarounds(
    tmp_path: Path,
    model_key: str,
    uses_qwen235_startup_workarounds: bool,
) -> None:
    run = resolve_run(model_key, "baseline", "smoke2", "lyris")

    command = build_runtime_command(
        run,
        tmp_path / "repo",
        tmp_path / "runs" / "unit",
        "unit",
    )

    assert (
        "NRL_DISABLE_VLLM_PORT_OVERRIDE=1" in command
    ) is uses_qwen235_startup_workarounds
    assert ("NRL_DISABLE_NUMA_MEMBIND=1" in command) is uses_qwen235_startup_workarounds
    assert (
        any(part.startswith("NRL_MEGATRON_CHECKPOINT_DIR=") for part in command)
        is uses_qwen235_startup_workarounds
    )


def test_megatron_checkpoint_cache_requires_completion_marker(
    tmp_path: Path,
) -> None:
    target_snapshot = tmp_path / "hf" / "snapshots" / ("a" * 40)
    checkpoint_root = tmp_path / "megatron"
    model_dir = checkpoint_root / f"model_{str(target_snapshot).replace('/', '_')}"
    iteration_dir = model_dir / "iter_0000000"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    (iteration_dir / "run_config.yaml").write_text("model: unit\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_megatron_checkpoint_cache(checkpoint_root, target_snapshot)

    tracker = model_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text("0\n", encoding="utf-8")
    validate_megatron_checkpoint_cache(checkpoint_root, target_snapshot)

    tracker.write_text("1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected iteration"):
        validate_megatron_checkpoint_cache(checkpoint_root, target_snapshot)


@pytest.mark.parametrize(
    ("mode", "required_flag"),
    [("test-only", "--test-only"), ("submit", "--parsable")],
)
def test_scheduler_command_is_mode_specific_and_dependency_free(
    tmp_path: Path, mode: str, required_flag: str
) -> None:
    run = resolve_run("qwen235", "baseline", "smoke2", "lyris")
    repo_dir = tmp_path / "repo"
    run_dir = tmp_path / "runs" / "unit"

    command = build_scheduler_command(run, repo_dir, run_dir, mode)

    assert required_flag in command
    assert "--dependency=" in command
    assert "--segment=16" in command
    assert "--output=" + str(run_dir / "slurm-%j.out") in command
    assert command[-1] == str(repo_dir / "ray.sub")
    assert not any("singleton" in part for part in command)
    assert not any(part.startswith("--gres") for part in command)


def test_submit_scheduler_sequence_preflights_the_exact_submission(
    tmp_path: Path,
) -> None:
    run = resolve_run("qwen30", "baseline", "smoke2", "lyris")
    repo_dir = tmp_path / "repo"
    run_dir = Path("/lustre/unit/run")

    preflight, submission = build_scheduler_sequence(run, repo_dir, run_dir, "submit")

    assert "--test-only" in preflight
    assert "--parsable" in submission
    assert tuple(part for part in preflight if part != "--test-only") == tuple(
        part for part in submission if part != "--parsable"
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_pushed_checkout(tmp_path: Path) -> Path:
    remote = tmp_path / "fork.git"
    remote.mkdir()
    _git(remote, "init", "--bare")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "matrix")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "remote", "add", "fork", str(remote))
    _git(repo, "push", "-u", "fork", "matrix")
    return repo


def test_checkout_validation_requires_clean_fork_pushed_commit(tmp_path: Path) -> None:
    repo = _make_pushed_checkout(tmp_path)

    state = validate_checkout(repo, expected_fork_url=str(tmp_path / "fork.git"))

    assert state.branch == "matrix"
    assert state.head == _git(repo, "rev-parse", "HEAD")
    assert state.fork_ref == "refs/remotes/fork/matrix"


def test_checkout_validation_rejects_dirty_checkout(tmp_path: Path) -> None:
    repo = _make_pushed_checkout(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean"):
        validate_checkout(repo, expected_fork_url=str(tmp_path / "fork.git"))


def test_checkout_validation_rejects_unpushed_commit(tmp_path: Path) -> None:
    repo = _make_pushed_checkout(tmp_path)
    (repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "unpushed")

    with pytest.raises(RuntimeError, match="fork"):
        validate_checkout(repo, expected_fork_url=str(tmp_path / "fork.git"))


def test_checkout_validation_reports_a_missing_remote_branch(tmp_path: Path) -> None:
    repo = _make_pushed_checkout(tmp_path)
    _git(repo, "switch", "-c", "local-only")

    with pytest.raises(RuntimeError, match="has no pushed fork ref"):
        validate_checkout(repo, expected_fork_url=str(tmp_path / "fork.git"))


def test_checkout_validation_preserves_remote_transport_diagnostic(
    tmp_path: Path,
) -> None:
    repo = _make_pushed_checkout(tmp_path)
    missing_remote = tmp_path / "missing.git"
    _git(repo, "remote", "set-url", "fork", str(missing_remote))

    with pytest.raises(RuntimeError, match="Could not query fork ref") as error:
        validate_checkout(repo, expected_fork_url=str(missing_remote))

    assert "fatal:" in str(error.value)


def test_checkout_validation_rejects_an_unapproved_fork_url(tmp_path: Path) -> None:
    repo = _make_pushed_checkout(tmp_path)

    with pytest.raises(RuntimeError, match="push URL"):
        validate_checkout(repo, expected_fork_url="git@example.invalid:user/RL.git")


def test_checkout_validation_preserves_initialized_submodule_status(
    tmp_path: Path,
) -> None:
    submodule = tmp_path / "submodule"
    submodule.mkdir()
    _git(submodule, "init", "-b", "main")
    _git(submodule, "config", "user.email", "test@example.com")
    _git(submodule, "config", "user.name", "Test User")
    (submodule / "tracked.txt").write_text("submodule\n", encoding="utf-8")
    _git(submodule, "add", "tracked.txt")
    _git(submodule, "commit", "-m", "initial")
    repo = _make_pushed_checkout(tmp_path)
    subprocess.run(
        (
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(submodule),
            "vendor/submodule",
        ),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "commit", "-am", "add submodule")
    _git(repo, "push", "fork", "matrix")

    state = validate_checkout(repo, expected_fork_url=str(tmp_path / "fork.git"))

    assert state.submodules[0].startswith(" ")


@pytest.mark.parametrize("run_tag", ("../escape", "/absolute", "nested/name", "."))
def test_run_destination_rejects_escaping_tags(tmp_path: Path, run_tag: str) -> None:
    with pytest.raises(ValueError, match="run tag"):
        validate_run_destination(
            tmp_path / "repo",
            Path("/lustre/unit/experiments"),
            run_tag,
            require_lustre=True,
        )


def test_run_destination_requires_lustre_outside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Lustre"):
        validate_run_destination(
            tmp_path / "repo", tmp_path / "runs", "unit", require_lustre=True
        )
    with pytest.raises(ValueError, match="outside"):
        validate_run_destination(
            Path("/lustre/unit/repo"),
            Path("/lustre/unit/repo/runs"),
            "unit",
            require_lustre=True,
        )


def test_snapshot_validation_requires_full_sha_and_weights(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="weights"):
        validate_snapshot(snapshot, "a" * 40, "unit")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    validate_snapshot(snapshot, "a" * 40, "unit")
    with pytest.raises(RuntimeError, match="40-character"):
        validate_snapshot(snapshot, "a", "unit")


def test_snapshot_validation_requires_every_indexed_shard(tmp_path: Path) -> None:
    revision = "b" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="indexed weight shards"):
        validate_snapshot(snapshot, revision, "unit")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"weights")
    validate_snapshot(snapshot, revision, "unit")


@pytest.mark.parametrize(
    "weight_map",
    (
        {},
        {"layer.0": "config.json"},
        {"layer.0": "../outside.safetensors"},
        {"layer.0": "/absolute/model.safetensors"},
    ),
)
def test_snapshot_validation_rejects_empty_or_unsafe_weight_indices(
    tmp_path: Path, weight_map: dict[str, str]
) -> None:
    revision = "e" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="weight index"):
        validate_snapshot(snapshot, revision, "unit")


def test_target_snapshot_rejects_a_moved_main_ref(tmp_path: Path) -> None:
    expected_revision = "c" * 40
    recipe = RecipeSpec(
        key="unit",
        path="recipe.yaml",
        target_repo_id="org/model",
        target_revision=expected_revision,
        nodes=1,
        segment=1,
        max_osl=128,
    )
    ref = recipe.target_ref_path(tmp_path)
    ref.parent.mkdir(parents=True)
    ref.write_text("d" * 40, encoding="utf-8")

    with pytest.raises(RuntimeError, match="ref moved"):
        resolve_target_snapshot(recipe, tmp_path)


def test_submission_environment_drops_ambient_execution_controls() -> None:
    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/test",
        "USER": "test",
        "WANDB_API_KEY": "secret",
        "SETUP_COMMAND": "touch /tmp/unsafe",
        "UV_CACHE_DIR_OVERRIDE": "/tmp/unsafe-cache",
        "NODE_MANAGER_PORT": "9999",
    }
    public = {
        "CONTAINER": "/lustre/image.sqsh",
        "MOUNTS": "/lustre:/lustre",
        "BASE_LOG_DIR": "/lustre/run",
        "GPUS_PER_NODE": "4",
    }

    environment, forwarded_secret_names = build_submission_environment(
        public, "env python command", source
    )

    assert environment["WANDB_API_KEY"] == "secret"
    assert forwarded_secret_names == ("WANDB_API_KEY",)
    assert environment["COMMAND"] == "env python command"
    assert "SETUP_COMMAND" not in environment
    assert "UV_CACHE_DIR_OVERRIDE" not in environment
    assert "NODE_MANAGER_PORT" not in environment
    assert "secret" not in json.dumps(public)


def test_missing_wandb_key_is_loaded_from_login_shell_without_literal_secret() -> None:
    calls: list[tuple[str, ...]] = []

    def run_login_shell(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "shell startup output\n"
                "__NRL_WANDB_API_KEY_BEGIN__from-bashrc"
                "__NRL_WANDB_API_KEY_END__\n"
            ),
            stderr="",
        )

    environment = load_login_wandb_environment(
        {"HOME": "/home/test", "PATH": "/usr/bin"},
        run_login_shell=run_login_shell,
    )

    assert environment["WANDB_API_KEY"] == "from-bashrc"
    assert calls == [
        (
            "bash",
            "-ilc",
            'printf "__NRL_WANDB_API_KEY_BEGIN__%s'
            '__NRL_WANDB_API_KEY_END__\\n" "${WANDB_API_KEY:-}"',
        )
    ]
    assert "from-bashrc" not in " ".join(calls[0])


def test_existing_wandb_key_does_not_start_a_login_shell() -> None:
    def fail_if_called(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError((command, kwargs))

    environment = load_login_wandb_environment(
        {"WANDB_API_KEY": "ambient"},
        run_login_shell=fail_if_called,
    )

    assert environment["WANDB_API_KEY"] == "ambient"


def test_provenance_is_written_atomically_without_secrets(tmp_path: Path) -> None:
    run = resolve_run("qwen30", "eagle3_k3", "smoke2", "lyris")
    run_dir = tmp_path / "run"
    payload = {
        "run_tag": "unit",
        "recipe": run.recipe.path,
        "environment": {"HF_HOME": str(run.cluster.hf_home)},
    }

    write_provenance(run_dir, payload)

    assert json.loads((run_dir / "provenance.json").read_text()) == payload
    text = (run_dir / "provenance.txt").read_text()
    assert "run_tag=unit" in text
    assert "WANDB_API_KEY" not in text
    assert not list(run_dir.glob("*.tmp"))


def test_show_cli_emits_deterministic_json_without_cluster_access(
    tmp_path: Path,
) -> None:
    matrix_path = (
        Path(__file__).parents[2] / "experiments/vllm_0251_drafter_matrix/matrix.py"
    )
    result = subprocess.run(
        (
            sys.executable,
            str(matrix_path),
            "show",
            "--model",
            "qwen30",
            "--variant",
            "suffix_k32",
            "--phase",
            "smoke2",
            "--cluster",
            "lyris",
            "--repo-dir",
            str(tmp_path / "repo"),
            "--experiment-root",
            str(tmp_path / "runs"),
            "--container",
            str(tmp_path / "image.sqsh"),
            "--run-tag",
            "unit-show",
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["run_tag"] == "unit-show"
    assert payload["model"] == "qwen30"
    assert payload["variant"] == "suffix_k32"
    assert payload["runner"] == "mrv1"
    assert payload["container"] == str(tmp_path / "image.sqsh")
    assert payload["run_dir"] == str(tmp_path / "runs" / "unit-show")
    assert "--dependency=" in payload["scheduler_command"]


def test_submit_wrapper_has_repository_license_header() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "experiments/vllm_0251_drafter_matrix/submit_matrix.sh"
    )

    assert wrapper.read_text().startswith(
        "#!/usr/bin/env bash\n"
        "# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.\n"
    )
