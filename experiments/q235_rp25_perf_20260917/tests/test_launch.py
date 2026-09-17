"""Contract tests for the Qwen3-235B RP25 performance launcher."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from experiments.q235_rp25_perf_20260917.launch import (
    configuration,
    render,
    required_inputs,
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
        config["policy.generation.vllm_kwargs.compilation_config.cudagraph_mode"]
        == "FULL_AND_PIECEWISE"
    )
    assert (
        config[
            "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
        ]
        == "[1,2,4,8,16,32,64]"
    )

    assert "grpo.num_prompts_per_step" not in config
    assert "grpo.num_generations_per_prompt" not in config
    assert "policy.max_total_sequence_length" not in config
    assert "policy.generation.vllm_cfg.tensor_parallel_size" not in config
    assert "policy.generation.vllm_kwargs.max_num_seqs" not in config
    assert config["policy.generation.vllm_kwargs.speculative_config"] == "null"
    assert config["policy.draft.enabled"] == "false"
    assert config["policy.megatron_cfg.distributed_timeout_seconds"] == "2400"


@pytest.mark.parametrize(
    ("arm", "method", "k", "checkpoint_fragment"),
    [
        ("dflash_k5", "dflash", 5, "dflash-b8"),
        ("dflash_k7", "dflash", 7, "dflash-b8"),
        ("dspark_k5", "dspark", 5, "dspark-b8"),
        ("dspark_k7", "dspark", 7, "dspark-b8"),
        ("dflash_b16_k11", "dflash", 11, "dflash-b16"),
        ("dflash_b16_k13", "dflash", 13, "dflash-b16"),
        ("dspark_b16_k11", "dspark", 11, "dspark-b16"),
        ("dspark_b16_k13", "dspark", 13, "dspark-b16"),
        ("eagle3_k3", "eagle3", 3, "Qwen3-235B-A22B-speculator.eagle3"),
        ("eagle3_k5", "eagle3", 5, "Qwen3-235B-A22B-speculator.eagle3"),
    ],
)
def test_specdec_arm_changes_only_runtime_speculation_contract(
    arm: str,
    method: str,
    k: int,
    checkpoint_fragment: str,
) -> None:
    config = configuration(steps=20, site="lyris", arm=arm)

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
            "policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size"
        ]
        == "1"
    )
    assert (
        config["policy.generation.vllm_kwargs.speculative_config.attention_backend"]
        == "FLASH_ATTN"
    )
    assert "policy.generation.vllm_kwargs.speculative_config" not in config

    for inherited_key in (
        "grpo.num_prompts_per_step",
        "grpo.num_generations_per_prompt",
        "policy.max_total_sequence_length",
        "policy.generation.vllm_cfg.tensor_parallel_size",
        "policy.generation.vllm_kwargs.moe_backend",
    ):
        assert inherited_key not in config


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("dflash_k5", "[1,2,4,6,8,12,16,24,32,48,64,96,192,384]"),
        (
            "dspark_k5",
            "[1,2,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,160,192,320,384]",
        ),
        ("dflash_k7", "[1,2,4,8,16,32,64,128,256,512]"),
        (
            "dspark_k7",
            "[1,2,4,7,8,14,16,28,32,56,64,112,128,224,256,448,512]",
        ),
        ("eagle3_k3", "[1,2,4,8,16,32,64,128,256]"),
        ("eagle3_k5", "[1,2,4,6,8,12,16,24,32,48,64,96,192,384]"),
        ("dflash_b16_k11", "[1,2,4,8,12,16,24,32,48,64,96,192,384,768]"),
        (
            "dspark_b16_k11",
            "[1,2,4,8,11,12,16,22,24,32,44,48,64,88,96,176,192,352,384,704,768]",
        ),
        ("dflash_b16_k13", "[1,2,4,8,14,16,28,32,56,64,112,224,448,896]"),
        (
            "dspark_b16_k13",
            "[1,2,4,8,13,14,16,26,28,32,52,56,64,104,112,208,224,416,448,832,896]",
        ),
    ],
)
def test_specdec_cuda_graphs_cover_geometric_request_buckets(
    arm: str, expected: str
) -> None:
    config = configuration(steps=1, site="lyris", arm=arm)

    assert (
        config[
            "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes"
        ]
        == expected
    )


def test_unknown_specdec_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        configuration(steps=1, site="lyris", arm="not_a_method")


def test_dspark_render_stages_node_local_runtime_overlays() -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-DSparkK5-B8-1step-test",
        steps=1,
        site="lyris",
        arm="dspark_k5",
    )

    assert "speculative_config.method=dspark" in script
    assert "speculative_config.num_speculative_tokens=5" in script
    assert "max_num_seqs=64" in script
    assert "prepare_vllm_dspark_fap_overlay.py" in script
    assert "kernel_config.enable_flashinfer_autotune=false" in script
    assert "Q235_NODE_ROOT=/raid/scratch" in script
    assert "Q235_MCORE_OVERLAY=${Q235_NODE_ROOT}/mcore-overlay" in script
    assert "Q235_VLLM_OVERLAY=${Q235_NODE_ROOT}/vllm-overlay" in script
    assert "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH" in script


def test_isolated_matrix_reuses_initialized_mcore_without_recursive_clone() -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-DFlashK5-B8-1step-test",
        steps=1,
        site="lyris",
        arm="dflash_k5",
    )

    assert (
        "Q235_MCORE_SOURCE=/home/sna/nemorl-q235-rp25-perf-20260917/"
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
        in script
    )


def test_node_setup_copies_mcore_when_archive_metadata_is_unsupported(
    tmp_path: Path,
) -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Eagle3K3-1step-copy-test",
        steps=1,
        site="lyris",
        arm="eagle3_k3",
    )
    setup_assignment = script.split("export SETUP_COMMAND=", 1)[1].split(
        "\nexport COMMAND=", 1
    )[0]
    setup_command = shlex.split(f"value={setup_assignment}")[0].split("=", 1)[1]

    source = tmp_path / "source"
    helpers = source / "megatron/core/datasets/helpers.cpp"
    helpers.parent.mkdir(parents=True)
    helpers.write_text("// fixture\n")
    shared_checkpoint = tmp_path / "checkpoint"
    shared_checkpoint.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_cp = fake_bin / "cp"
    fake_cp.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == -a ]]; then\n"
        "  echo 'archive metadata is unsupported' >&2\n"
        "  exit 95\n"
        "fi\n"
        'exec /bin/cp "$@"\n'
    )
    fake_cp.chmod(0o755)

    node_root = tmp_path / "node"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "XDG_CACHE_HOME": str(node_root / "cache"),
        "TRITON_CACHE_DIR": str(node_root / "cache/triton"),
        "TORCH_EXTENSIONS_DIR": str(node_root / "cache/torch-extensions"),
        "WANDB_CACHE_DIR": str(node_root / "cache/wandb"),
        "WANDB_CONFIG_DIR": str(node_root / "cache/wandb-config"),
        "RAY_TMPDIR": str(node_root / "ray"),
        "NRL_NATIVE_TMP": str(node_root / "tmp"),
        "Q235_MCORE_OVERLAY": str(node_root / "mcore-overlay"),
        "Q235_MCORE_SOURCE": str(source),
        "NRL_MEGATRON_CHECKPOINT_DIR": str(shared_checkpoint),
        "NRL_MOUNT_CHECK_ID": "copy-test",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            "test() { "
            "if [[ ${1:-} == -x && "
            "( ${2:-} == /opt/* || ${2:-} == /usr/local/* ) ]]; "
            "then return 0; fi; "
            'builtin test "$@"; '
            "}; " + setup_command,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (node_root / "mcore-overlay/megatron/core/datasets/helpers.cpp").is_file()
    assert list(shared_checkpoint.glob(".mount-check-copy-test-*"))


def test_dflash_and_eagle_do_not_apply_dspark_runtime_patch() -> None:
    for arm in ("dflash_k5", "eagle3_k3"):
        script = render(
            account="coreai_dlalgo_llm",
            run_name=f"Qwen3-235B-{arm}-1step-test",
            steps=1,
            site="lyris",
            arm=arm,
        )

        assert "prepare_vllm_dspark_fap_overlay.py" not in script


def test_specdec_required_inputs_include_exact_export_files() -> None:
    inputs = required_inputs(site="lyris", arm="dflash_b16_k13")

    assert any(str(path).endswith("dflash-b16/config.json") for path in inputs)
    assert any(str(path).endswith("dflash-b16/model.safetensors") for path in inputs)
    assert any(
        str(path).endswith("megatron/core/datasets/helpers.cpp") for path in inputs
    )


def test_dspark_required_inputs_include_runtime_overlay_builder() -> None:
    inputs = required_inputs(site="lyris", arm="dspark_k5")

    assert any(
        str(path).endswith("prepare_vllm_dspark_fap_overlay.py") for path in inputs
    )
    assert any(
        str(path).endswith("vllm-0.25.1-pr48167-runtime.patch") for path in inputs
    )
    assert any(
        str(path).endswith("vllm-0.25.1-pr48167-group-causality-followup.patch")
        for path in inputs
    )


def test_cli_render_names_the_selected_arm() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/q235_rp25_perf_20260917/launch.py",
            "--site",
            "lyris",
            "--arm",
            "eagle3_k5",
            "--steps",
            "1",
            "--render",
        ],
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Qwen3-235B-Eagle3K5-1step-" in result.stdout
    assert "speculative_config.method=eagle3" in result.stdout


def test_render_uses_official_16n4g_recipe_and_bounded_runtime() -> None:
    script = render(
        account="coreai_dlalgo_nemorl",
        run_name="Qwen3-235B-Baseline-20step-test",
    )

    assert "#SBATCH --nodes=16" in script
    assert (
        "#SBATCH --job-name=coreai_dlalgo_nemorl-specdec."
        "Qwen3-235B-Baseline-20step-test" in script
    )
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --segment=16" in script
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "grpo-qwen3-235b-16n4g.yaml" in script
    assert "policy.generation.vllm_kwargs.max_num_seqs" not in script
    assert "NRL_MEGATRON_CHECKPOINT_DIR" in script
    assert "RAY_TMPDIR=/raid/scratch/sna/r${SLURM_JOB_ID}" in script


def test_one_step_gate_preserves_topology_with_one_hour_window() -> None:
    config = configuration(steps=1, site="lyris")
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Baseline-1step-gate",
        steps=1,
        site="lyris",
    )

    assert config["grpo.max_num_steps"] == "1"
    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --segment=16" in script
    assert "#SBATCH --time=01:00:00" in script


def test_render_executes_the_recorded_source_revision() -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Baseline-20step-source-test",
        site="lyris",
    )

    source = "/home/sna/nemorl-q235-specdec-matrix-20260917"
    assert f"cd {source}" in script
    assert "export PYTHONPATH=${Q235_VLLM_OVERLAY}:${Q235_MCORE_OVERLAY}:" in script
    assert f":{source}" in script
    assert "cd /opt/nemo-rl" not in script
    assert f'root = Path("{source}").resolve()' in script
    assert 'Path(os.environ["PYTHONPATH"])' not in script


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
    assert ".mount-check-${NRL_MOUNT_CHECK_ID}-$(hostname)" in first
    assert ".mount-check-{run_id}-*" in first
    assert 'p.glob(".mount-check-*")' not in first


def test_ptyche_baseline_uses_high_priority_a01r_partition_and_staged_inputs() -> None:
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
    assert "#SBATCH --partition=36x2-a01r" in script
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


def test_q235_launcher_keeps_cpu_affinity_without_hard_numa_membind() -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-Baseline-1step-numa-test",
        steps=1,
        site="lyris",
    )

    assert "export NRL_DISABLE_NUMA_MEMBIND=1" in script
    assert "NRL_DISABLE_NUMA_BINDING" not in script


def test_q235_launcher_uses_safe_ray_host_memory_headroom() -> None:
    script = render(
        account="coreai_dlalgo_llm",
        run_name="Qwen3-235B-DFlashK5-20step-ray-memory-test",
        steps=20,
        site="ptyche",
        arm="dflash_k5",
    )

    assert "export RAY_memory_usage_threshold=0.98" in script
    assert "RAY_memory_monitor_refresh_ms=0" not in script


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
