"""Contract tests for the Qwen3-235B RP25 performance launcher."""

from pathlib import Path

from experiments.q235_rp25_perf_20260917.launch import (
    configuration,
    render,
    sbatch_arguments,
)


def test_baseline_preserves_official_performance_workload() -> None:
    config = configuration(steps=20)

    assert config["grpo.max_num_steps"] == "20"
    assert config["policy.model_name"].endswith(
        "models--Qwen--Qwen3-235B-A22B/snapshots/"
        "8efa61729e24bd65b1d152b5ab5409052aa80e65"
    )
    assert config["policy.precision"] == "bfloat16"
    assert config["policy.generation.vllm_cfg.enforce_eager"] == "false"
    assert "policy.generation.vllm_kwargs.moe_backend" not in config
    assert (
        config[
            "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode"
        ]
        == "FULL_AND_PIECEWISE"
    )
    assert config[
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
    ] == "[1,2,4,8,16,32,64]"

    assert "grpo.num_prompts_per_step" not in config
    assert "grpo.num_generations_per_prompt" not in config
    assert "policy.max_total_sequence_length" not in config
    assert "policy.generation.vllm_cfg.tensor_parallel_size" not in config
    assert "policy.generation.vllm_kwargs.max_num_seqs" not in config
    assert config["policy.generation.vllm_kwargs.speculative_config"] == "null"
    assert config["policy.draft.enabled"] == "false"


def test_render_uses_official_16n4g_recipe_and_bounded_runtime() -> None:
    script = render(
        account="coreai_dlalgo_nemorl",
        run_name="Qwen3-235B-Baseline-20step-test",
    )

    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --segment=16" in script
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "grpo-qwen3-235b-16n4g.yaml" in script
    assert "policy.generation.vllm_kwargs.max_num_seqs" not in script
    assert "NRL_MEGATRON_CHECKPOINT_DIR" in script
    assert "RAY_TMPDIR=/raid/scratch/sna/r${SLURM_JOB_ID}" in script


def test_render_scopes_shared_checkpoint_mount_markers_to_each_run() -> None:
    first = render(
        account="coreai_dlalgo_nemorl",
        run_name="Qwen3-235B-Baseline-20step-first",
    )
    second = render(
        account="coreai_dlalgo_nemorl",
        run_name="Qwen3-235B-Baseline-20step-second",
    )

    assert "NRL_MOUNT_CHECK_ID=Qwen3-235B-Baseline-20step-first" in first
    assert "NRL_MOUNT_CHECK_ID=Qwen3-235B-Baseline-20step-second" in second
    assert '.mount-check-${NRL_MOUNT_CHECK_ID}-$(hostname)' in first
    assert '.mount-check-{run_id}-*' in first
    assert 'p.glob(".mount-check-*")' not in first


def test_ptyche_baseline_uses_staged_target_and_september_16_container() -> None:
    config = configuration(steps=20, site="ptyche")
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Baseline-20step-ptyche-test",
        site="ptyche",
    )

    assert config["policy.model_name"] == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/models/"
        "Qwen3-235B-A22B-8efa61729e24bd65b1d152b5ab5409052aa80e65"
    )
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --time=05:00:00" in script
    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --gpus-per-node" not in script
    assert "#SBATCH --gres" not in script
    assert "nemo_rl_nightly_20260916_2837270.sqsh" in script
    assert "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/" in script
    assert "policy.generation.vllm_kwargs.max_num_seqs" not in script


def test_lyris_baseline_uses_gb200_partition_and_staged_inputs() -> None:
    config = configuration(steps=20, site="lyris")
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Baseline-20step-lyris-test",
        site="lyris",
    )

    assert config["policy.model_name"] == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--Qwen--Qwen3-235B-A22B/snapshots/"
        "8efa61729e24bd65b1d152b5ab5409052aa80e65"
    )
    assert "#SBATCH --partition=gb200" in script
    assert "#SBATCH --time=05:00:00" in script
    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --gpus-per-node" not in script
    assert "#SBATCH --gres" not in script
    assert "nemo_rl_nightly_20260916_3078480.sqsh" in script
    assert "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/" in script
    assert "policy.generation.vllm_kwargs.max_num_seqs" not in script


def test_q235_launcher_disables_nvls_like_official_performance_wrapper() -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Baseline-20step-nvls-test",
        site="lyris",
    )

    assert "export NCCL_NVLS_ENABLE=0" in script


def test_sbatch_arguments_can_wait_for_staging_job() -> None:
    job = Path("/tmp/q235/job.sbatch")

    assert sbatch_arguments(job, test_only=True, afterok=2842006) == [
        "sbatch",
        "--test-only",
        "--dependency=afterok:2842006",
        str(job),
    ]
    assert sbatch_arguments(job, test_only=False, afterok=2842006) == [
        "sbatch",
        "--parsable",
        "--dependency=afterok:2842006",
        str(job),
    ]
