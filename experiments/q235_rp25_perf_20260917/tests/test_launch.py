"""Contract tests for the Qwen3-235B RP25 performance launcher."""

from experiments.q235_rp25_perf_20260917.launch import configuration, render


def test_baseline_preserves_official_performance_workload() -> None:
    config = configuration(steps=20)

    assert config["grpo.max_num_steps"] == "20"
    assert config["policy.model_name"].endswith(
        "models--Qwen--Qwen3-235B-A22B/snapshots/"
        "8efa61729e24bd65b1d152b5ab5409052aa80e65"
    )
    assert config["policy.precision"] == "bfloat16"
    assert config["policy.generation.vllm_cfg.enforce_eager"] == "false"
    assert (
        config["policy.generation.vllm_kwargs.moe_backend"]
        == "flashinfer_trtllm"
    )
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
