from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DYNAMICSD_LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_024_upgrade"
    / "submit_eagle3_dynamicsd_step20.sh"
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
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
    )

    assert "TMPDIR=/tmp" in output


def test_performance_launcher_preserves_compute_visible_workdir() -> None:
    output = _run_script(
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "CONTAINER_WORKDIR=/lustre/users/sna/RL" in output


def test_performance_launcher_imports_nemo_rl_from_the_checkout() -> None:
    output = _run_script(
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "PYTHONPATH=/lustre/users/sna/RL" in output


def test_performance_launcher_uses_node_local_compiler_caches() -> None:
    output = _run_script(
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
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

    assert 'CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$SLURM_SUBMIT_DIR}' in source
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
    assert "WANDB_RESUME=never" in output


def test_dynamicsd_launcher_renders_fixed_eagle3() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k5")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "num_speculative_tokens_per_batch_size" not in output
    assert "Qwen3-30B-A3B-Thinking-2507-speculator.eagle3" in output


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

    aws_root = (
        "/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna"
    )
    assert f"CONTAINER={aws_root}/containers/nemo_rl_nightly.sqsh" in output
    assert f"HF_HOME={aws_root}/hf_home" in output
    assert (
        "[DRY-RUN] wandb https://wandb.ai/nvidia/"
        "nemorl-vllm024-dynamicsd-aws-dfw/runs/"
        "contract-test-attempt-1-qwen32b-dynamic"
    ) in output


def test_dynamicsd_launcher_starts_outside_nemo_gym_port_range() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "baseline")

    assert "VLLM_PORT=20001" in output


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


def test_dynamicsd_launcher_ignores_generated_submodule_dirt() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert source.count("--ignore-submodules=dirty") == 2


def test_dynamicsd_launcher_preserves_logical_lustre_checkout_path() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert "logical_pwd=\"$(pwd -L)\"" in source
    assert "repo_prefix=\"$(git rev-parse --show-prefix)\"" in source
