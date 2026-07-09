from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DYNAMICSD_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_eagle3_dynamicsd_step20.sh"
)
PARITY_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_generation_parity.sh"
)


def _run_script(path: Path, *args: str, **environment: str) -> str:
    result = subprocess.run(
        ["bash", str(path), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _dry_run_dynamicsd(model: str, variant: str) -> str:
    return _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        model,
        variant,
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )


def _dry_run_parity(variant: str, mode: str) -> str:
    return _run_script(
        PARITY_LAUNCHER,
        "dry-run",
        variant,
        mode,
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="parity-contract-test",
        TARGET_MODEL="/lustre/users/sna/qwen32",
        DRAFT_MODEL="/lustre/users/sna/qwen32-eagle3",
    )


def test_compat_smoke_uses_short_compute_node_tmpdir() -> None:
    output = _run_script(
        REPO_ROOT / "scripts" / "submit_vllm_024_compat_smoke.sh",
        REPO_DIR=str(REPO_ROOT),
        CONTAINER="/unused/nemo-rl.sqsh",
        DRY_RUN="true",
    )

    assert "TMPDIR=/tmp" in output


def test_performance_launcher_uses_short_compute_node_tmpdir() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
    )

    assert "TMPDIR=/tmp" in output


def test_performance_launcher_preserves_compute_visible_workdir() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "CONTAINER_WORKDIR=/lustre/users/sna/RL" in output


def test_performance_launcher_imports_nemo_rl_from_the_checkout() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "PYTHONPATH=/lustre/users/sna/RL" in output


def test_performance_launcher_uses_node_local_compiler_caches() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        RUN_TAG="cache-test",
    )

    assert "TRITON_CACHE_DIR=/tmp/nemorl-vllm024-triton-cache-test-qwen32b" in output
    assert (
        "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-vllm024-inductor-cache-test-qwen32b"
        in output
    )


def test_ray_launcher_accepts_an_explicit_container_workdir() -> None:
    source = (REPO_ROOT / "ray.sub").read_text(encoding="utf-8")

    assert "CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$SLURM_SUBMIT_DIR}" in source
    assert 'COMMON_SRUN_ARGS+=" --container-workdir=$CONTAINER_WORKDIR"' in source


def test_dynamicsd_launcher_preserves_matched_runtime_contract() -> None:
    output = _dry_run_dynamicsd("qwen32b", "dynamic")

    assert "grpo.max_num_steps=20" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "policy.generation.temperature=1.0" in output
    assert "policy.generation.top_p=1.0" in output
    assert "compilation_config.cudagraph_mode=PIECEWISE" in output
    assert "cluster.segment_size=4" in output
    assert "--nodes=4" in output
    assert "--segment=4" in output
    assert "--gres=gpu:4" in output
    assert "logger.wandb.entity=nvidia" in output
    assert "logger.tensorboard_enabled=false" in output
    assert "WANDB_RESUME=never" in output


def test_dynamicsd_launcher_rebuilds_the_vllm_024_worker_runtime() -> None:
    output = _dry_run_dynamicsd("qwen235b", "eagle3_k5")

    assert "/opt/nemo_rl_venv/bin/python" in output
    assert (
        "NEMO_RL_VENV_DIR=/tmp/nemorl-vllm024-venvs-contract-test-attempt-1-"
        "qwen235b-eagle3_k5"
    ) in output
    assert "NEMO_RL_VENV_DIR=/lustre" not in output
    assert "NRL_FORCE_REBUILD_VENVS=true" in output
    assert "uv run" not in output


def test_dynamicsd_launcher_renders_qwen30_long_context_topology() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "dynamic",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="long-context-contract-test",
        ATTEMPT_ID="attempt-1",
        QWEN30_RECIPE=(
            "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml"
        ),
        QWEN30_NODES="8",
        NUM_PROMPTS_PER_STEP="16",
        NUM_GENERATIONS_PER_PROMPT="16",
        TRAIN_GLOBAL_BATCH_SIZE="256",
        MAX_TOTAL_SEQUENCE_LENGTH="40960",
        MAX_NEW_TOKENS="32768",
        DYNAMIC_SCHEDULE="[[1,2,5],[3,4,4],[5,8,3],[9,16,1],[17,512,0]]",
    )

    assert "grpo-qwen3-30ba3b-4n8g-40K.yaml" in output
    assert "grpo.num_prompts_per_step=16" in output
    assert "grpo.num_generations_per_prompt=16" in output
    assert "policy.train_global_batch_size=256" in output
    assert "policy.max_total_sequence_length=40960" in output
    assert "policy.generation.max_new_tokens=32768" in output
    assert "cluster.gpus_per_node=4" in output
    assert "cluster.num_nodes=8" in output
    assert "cluster.segment_size=8" in output
    assert "--nodes=8" in output
    assert "--segment=8" in output
    assert "--gres=gpu:4" in output


def test_dynamicsd_launcher_renders_fixed_eagle3() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k5")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "num_speculative_tokens_per_batch_size" not in output
    assert "Qwen3-30B-A3B-Thinking-2507-speculator.eagle3" in output


def test_dynamicsd_launcher_renders_aggressive_fixed_k_values() -> None:
    k7_output = _dry_run_dynamicsd("qwen32b", "eagle3_k7")
    k9_output = _dry_run_dynamicsd("qwen32b", "eagle3_k9")

    assert "speculative_config.num_speculative_tokens=7" in k7_output
    assert "speculative_config.num_speculative_tokens=9" in k9_output
    assert "num_speculative_tokens_per_batch_size" not in k7_output
    assert "num_speculative_tokens_per_batch_size" not in k9_output


def test_dynamicsd_launcher_all_includes_k7_and_k9() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "all",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    assert "contract-test-attempt-1-qwen30ba3b-eagle3_k7" in output
    assert "contract-test-attempt-1-qwen30ba3b-eagle3_k9" in output


def test_dynamicsd_launcher_qwen30_comparison_includes_all_methods() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "compare",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    for variant in (
        "baseline",
        "eagle3_k5",
        "eagle3_k7",
        "eagle3_k9",
        "suffix_k32",
        "pard_k5",
        "pard_k16",
    ):
        assert f"contract-test-attempt-1-qwen30ba3b-{variant}" in output


def test_dynamicsd_launcher_renders_suffix_k32() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "suffix_k32",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        ARCTIC_OVERLAY="/lustre/users/sna/arctic-inference-0.1.1",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    assert "speculative_config.method=suffix" in output
    assert "speculative_config.num_speculative_tokens=32" in output
    assert "speculative_config.model=" not in output
    assert "speculative_config.draft_tensor_parallel_size" not in output
    assert (
        "PYTHONPATH=/lustre/users/sna/arctic-inference-0.1.1:/lustre/users/sna/RL"
        in output
    )


def test_dynamicsd_launcher_renders_pard_with_graph_patch() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "pard_k16")

    assert "speculative_config.method=draft_model" in output
    assert "speculative_config.num_speculative_tokens=16" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "speculative_config.parallel_drafting=true" in output
    assert "models--amd--PARD-Qwen3-0.6B" in output
    assert "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH=true" in output
    assert "policy.generation.vllm_kwargs.max_num_batched_tokens=32768" in output


def test_dynamicsd_launcher_renders_qwen235b_performance_topology() -> None:
    output = _dry_run_dynamicsd("qwen235b", "eagle3_k9")

    assert "grpo-qwen3-235b-16n4g.yaml" in output
    assert "Qwen3-235B-A22B-Eagle3" in output
    assert "speculative_config.num_speculative_tokens=9" in output
    assert "cluster.segment_size=16" in output
    assert "--nodes=16" in output
    assert "--segment=16" in output
    assert "--gres=gpu:4" in output
    assert "VLLM_PORT=" not in output
    assert "NRL_DISABLE_VLLM_PORT_OVERRIDE=1" in output


def test_dynamicsd_launcher_renders_dynamic_schedule() -> None:
    output = _dry_run_dynamicsd("qwen32b", "dynamic")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert (
        "speculative_config.num_speculative_tokens_per_batch_size="
        "\\[\\[1\\,16\\,5\\]\\,\\[17\\,32\\,4\\]\\,"
        "\\[33\\,64\\,3\\]\\,\\[65\\,128\\,1\\]\\,"
        "\\[129\\,512\\,0\\]\\]"
    ) in output


def test_dynamicsd_launcher_keeps_baseline_free_of_specdec() -> None:
    output = _dry_run_dynamicsd("qwen32b", "baseline")

    assert "compilation_config.cudagraph_mode=PIECEWISE" in output
    assert "speculative_config" not in output


def test_dynamicsd_launcher_defaults_to_aws_dynamically_staged_assets() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen32b",
        "dynamic",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    aws_root = "/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna"
    assert f"CONTAINER={aws_root}/containers/nemo_rl_nightly.sqsh" in output
    assert f"HF_HOME={aws_root}/hf_home" in output
    assert (
        "[DRY-RUN] wandb https://wandb.ai/nvidia/"
        "nemorl-vllm024-dynamicsd-aws-dfw/runs/"
        "contract-test-attempt-1-qwen32b-dynamic"
    ) in output


def test_dynamicsd_launcher_delegates_engine_ports_to_nemorl() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "baseline")

    assert "VLLM_PORT=" not in output


def test_dynamicsd_launcher_records_reproducibility_metadata() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    for field in (
        "container",
        "container_sha256",
        "max_steps",
        "static_k",
        "dynamic_schedule",
        "command",
    ):
        assert field in source


def test_dynamicsd_launcher_validates_selected_recipe_and_batch_geometry() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert 'git -C "${REPO_DIR}" ls-files --error-unmatch "${recipe}"' in source
    assert "total_trajectories % TRAIN_GLOBAL_BATCH_SIZE" in source
    assert "TRAIN_GLOBAL_BATCH_SIZE > total_trajectories" in source


def test_dynamicsd_launcher_rejects_partial_batch_geometry() -> None:
    result = subprocess.run(
        [
            "bash",
            str(DYNAMICSD_LAUNCHER),
            "dry-run",
            "qwen30ba3b",
            "baseline",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "TRAIN_GLOBAL_BATCH_SIZE": "256"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "require positive NUM_PROMPTS_PER_STEP" in result.stderr


def test_dynamicsd_launcher_ignores_generated_submodule_dirt() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert source.count("--ignore-submodules=dirty") == 2


def test_dynamicsd_launcher_preserves_logical_lustre_checkout_path() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert 'logical_pwd="$(pwd -L)"' in source
    assert 'repo_prefix="$(git rev-parse --show-prefix)"' in source


def test_generation_parity_launcher_matches_lyris_topology_and_cuda_graph() -> None:
    output = _dry_run_parity("eagle3_k5", "greedy")

    assert "--account=coreai_dlalgo_llm" in output
    assert "--partition=gb200" in output
    assert "--nodes=1" in output
    assert "--segment=1" in output
    assert "--gres" not in output
    assert "--target-tp 2" in output
    assert "--draft-tp 1" in output
    assert "--gpus-per-node 2" in output
    assert "--max-model-len 4096" in output
    assert "--max-num-batched-tokens 16384" in output
    assert "--mode greedy" in output
    assert "--samples-per-prompt 1" in output
    assert "/lustre/users/sna/qwen32-eagle3" in output


def test_generation_parity_launcher_uses_short_ray_socket_root() -> None:
    output = _dry_run_parity("eagle3_k5", "greedy")

    assert "--ray-log-dir /tmp/nrp-eagle3_k5-greedy" in output
    assert "parity-contract-test/eagle3_k5/greedy/ray_logs" not in output


def test_generation_parity_launcher_rebuilds_worker_venv_from_branch_lock() -> None:
    output = _dry_run_parity("eagle3_k5", "greedy")

    assert "NRL_FORCE_REBUILD_VENVS=true" in output
    assert (
        "NEMO_RL_VENV_DIR=/lustre/users/sna/experiments/"
        "vllm024-generation-parity/parity-contract-test/"
        "eagle3_k5/greedy/venvs"
    ) in output


def test_generation_parity_launcher_keeps_baseline_free_of_draft_model() -> None:
    output = _dry_run_parity("baseline", "sampled")

    assert "--mode sampled" in output
    assert "--samples-per-prompt 64" in output
    assert "--draft-model" not in output


def test_generation_parity_launcher_all_renders_four_independent_jobs() -> None:
    output = _dry_run_parity("all", "all")

    for label in (
        "baseline-greedy",
        "baseline-sampled",
        "eagle3_k5-greedy",
        "eagle3_k5-sampled",
    ):
        assert f"parity-contract-test-{label}" in output
