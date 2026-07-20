from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import pytest

from experiments.nemogym_swe_full_rl.gym_openhands_tmux import (
    patch_gym_openhands_tmux_source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "experiments" / "nemogym_swe_full_rl" / "launch_lyris.py"
GRPO = REPO_ROOT / "nemo_rl" / "algorithms" / "grpo.py"
SWE_ENTRY_SMOKE = (
    REPO_ROOT / "experiments" / "nemogym_swe_full_rl" / "verify_openhands_swe_entry.py"
)
SWE_ENTRY_SMOKE_LYRIS = (
    REPO_ROOT
    / "experiments"
    / "nemogym_swe_full_rl"
    / "verify_openhands_swe_entry_lyris.sh"
)


class DryRunPayload(TypedDict):
    command: list[str]
    overrides: list[str]
    run_dir: str
    sbatch_args: list[str]
    submission_unset_environment: list[str]


BUGGY_GYM_TMUX_SOURCE = """
        agent_main_cmd = (
            "export TMUX_TMPDIR=/tmp && "
            "export TMUX=/tmp/tmux-$uid/default && "
            "mkdir -p /tmp/tmux-$uid && "
            "chown $uid:$uid /tmp/tmux-$uid || true && "
            "chmod 700 /tmp/tmux-$uid && "
            "tmux -S /tmp/tmux-$uid/default start-server || true && "
        )
"""


def _run_launcher(
    variant: str,
    *extra_args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WANDB_API_KEY"] = "must-not-be-serialized"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--mode",
            "dry-run",
            "--variant",
            variant,
            "--run-tag",
            "contract-test",
            "--repo-dir",
            str(REPO_ROOT),
            "--json",
            *extra_args,
        ],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result


def _dry_run(variant: str, *extra_args: str) -> DryRunPayload:
    result = _run_launcher(variant, *extra_args)
    return cast(DryRunPayload, json.loads(result.stdout))


def test_gym_openhands_tmux_patch_leaves_server_ownership_to_libtmux() -> None:
    patched = patch_gym_openhands_tmux_source(BUGGY_GYM_TMUX_SOURCE)

    assert '"export TMUX_TMPDIR=/tmp && "' in patched
    assert '"unset TMUX && "' in patched
    assert '"export TMUX=/tmp/tmux-$uid/default && "' not in patched
    assert "start-server" not in patched


def test_gym_openhands_tmux_patch_is_idempotent() -> None:
    patched = patch_gym_openhands_tmux_source(BUGGY_GYM_TMUX_SOURCE)

    assert patch_gym_openhands_tmux_source(patched) == patched


def test_gym_openhands_tmux_patch_rejects_unknown_upstream_source() -> None:
    with pytest.raises(ValueError, match="expected Gym tmux setup block"):
        patch_gym_openhands_tmux_source("agent_main_cmd = 'unknown upstream'\n")


def test_baseline_runs_full_async_swe_grpo_training() -> None:
    payload = _dry_run("baseline")
    command = payload["command"]
    overrides = payload["overrides"]

    assert command[-1] != "run_grpo_rollout_benchmark.py"
    assert any(part.endswith("run_grpo_nemo_gym.py") for part in command)
    assert any(
        part.endswith("grpo_qwen3_30ba3b_thinking_swe2_smoke.yaml") for part in command
    )
    for inherited_env in (
        "CONDA_PREFIX",
        "CONDA_PREFIX_1",
        "CONDA_DEFAULT_ENV",
        "CONDA_PYTHON_EXE",
        "CONDA_EXE",
        "_CONDA_EXE",
        "CONDA_ROOT",
        "_CONDA_ROOT",
        "CONDA_SHLVL",
        "CONDA_PROMPT_MODIFIER",
        "_CE_M",
        "_CE_CONDA",
        "VIRTUAL_ENV",
    ):
        position = command.index(inherited_env)
        assert command[position - 1 : position + 1] == ["-u", inherited_env]
    assert "grpo.async_grpo.enabled=true" in overrides
    assert "grpo.async_grpo.in_flight_weight_updates=true" in overrides
    assert "grpo.max_num_steps=2" in overrides
    assert "grpo.num_prompts_per_step=2" in overrides
    assert "grpo.num_generations_per_prompt=2" in overrides
    assert "policy.train_global_batch_size=4" in overrides
    assert "policy.generation_batch_size=4" in overrides
    assert "checkpointing.enabled=false" in overrides


def test_full_grpo_pins_only_gym_subprocess_openai_version() -> None:
    payload = _dry_run("baseline")
    command = payload["command"]
    overrides = payload["overrides"]

    assert any(part.endswith("run_grpo_nemo_gym.py") for part in command)
    assert "++env.nemo_gym.subprocess_openai_version=2.7.2" in overrides


def test_nemo_gym_server_venvs_use_shared_lustre_storage() -> None:
    payload = _dry_run("baseline")
    command = payload["command"]

    gym_venv_assignments = [
        part for part in command if part.startswith("NEMO_GYM_VENV_DIR=")
    ]
    assert len(gym_venv_assignments) == 1
    assert gym_venv_assignments[0].startswith(
        "NEMO_GYM_VENV_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/"
        "experiments/nemogym_swe_full_rl/gym_venvs/"
    )
    assert "/opt/gym_venvs" not in gym_venv_assignments[0]


def test_full_grpo_uses_shared_nemo_gym_actor_config_builder() -> None:
    source = GRPO.read_text()

    assert "build_nemo_gym_config(" in source


def test_training_and_generation_topology_fit_nine_lyris_nodes() -> None:
    payload = _dry_run("baseline")
    overrides = payload["overrides"]

    assert "cluster.num_nodes=9" in overrides
    assert "cluster.gpus_per_node=4" in overrides
    assert "++cluster.segment_size=8" in overrides
    assert "++env.nemo_gym.is_trajectory_collection=false" in overrides
    assert "policy.generation.colocated.enabled=false" in overrides
    assert "policy.generation.colocated.resources.num_nodes=1" in overrides
    assert "policy.generation.colocated.resources.gpus_per_node=4" in overrides
    assert "policy.megatron_cfg.tensor_model_parallel_size=4" in overrides
    assert "policy.megatron_cfg.pipeline_model_parallel_size=2" in overrides
    assert "policy.megatron_cfg.context_parallel_size=4" in overrides
    assert "policy.megatron_cfg.expert_model_parallel_size=8" in overrides


def test_dflash_variants_use_verified_checkpoint_and_k_aligned_graphs() -> None:
    expected = {
        "dflash_k7": (7, "[8,16,32,64,128]"),
        "dflash_k9": (9, "[10,20,40,80,160]"),
    }

    for variant, (k, capture_sizes) in expected.items():
        payload = _dry_run(variant)
        overrides = payload["overrides"]
        rendered = " ".join(overrides)

        assert "speculative_config.method=dflash" in rendered
        assert "models--RedHatAI--Qwen3-30B-A3B-speculator.dflash" in rendered
        assert "snapshots/edcff83783141eb9383e2bd6c33610d9a3104288" in rendered
        assert f"speculative_config.num_speculative_tokens={k}" in rendered
        assert "speculative_config.draft_tensor_parallel_size=1" in rendered
        assert "speculative_config.max_model_len=4096" in rendered
        assert "speculative_config.attention_backend=FLASH_ATTN" in rendered
        assert "compilation_config.cudagraph_mode=FULL" in rendered
        assert f"compilation_config.cudagraph_capture_sizes={capture_sizes}" in rendered


def test_grpo_rejects_single_generation_per_prompt() -> None:
    result = _run_launcher(
        "baseline",
        "--num-generations",
        "1",
        check=False,
    )

    assert result.returncode == 2
    assert "num-generations must be at least 2 for GRPO" in result.stderr


def test_lyris_submission_is_segmented_without_gres_or_real_dependency() -> None:
    payload = _dry_run("baseline")
    sbatch_args = payload["sbatch_args"]
    rendered = " ".join(sbatch_args)

    assert "--account=coreai_dlalgo_llm" in sbatch_args
    assert "--partition=gb200" in sbatch_args
    assert "--nodes=9" in sbatch_args
    assert "--segment=9" in sbatch_args
    assert "--time=05:00:00" in sbatch_args
    assert "--dependency=" in sbatch_args
    assert "singleton" not in rendered
    assert "--gres" not in rendered


def test_wandb_is_enabled_without_serializing_the_api_key() -> None:
    payload = _dry_run("baseline")
    rendered = json.dumps(payload, sort_keys=True)
    overrides = payload["overrides"]

    assert "logger.wandb_enabled=true" in overrides
    assert "logger.tensorboard_enabled=false" in overrides
    assert "logger.wandb.project=nemo-rl-vllm0251-swe-full-grpo" in overrides
    assert "logger.wandb.name=q30-swe-full-rl-baseline-contract-test" in overrides
    assert "must-not-be-serialized" not in rendered
    assert "WANDB_API_KEY" not in rendered


def test_twenty_step_mode_changes_only_the_requested_step_budget() -> None:
    payload = _dry_run("dflash_k7", "--steps", "20")
    overrides = payload["overrides"]

    assert "grpo.max_num_steps=20" in overrides
    assert "grpo.num_prompts_per_step=2" in overrides
    assert "grpo.num_generations_per_prompt=2" in overrides


def test_shell_wrapper_uses_portable_python3_entrypoint() -> None:
    wrapper = (
        REPO_ROOT / "experiments" / "nemogym_swe_full_rl" / "submit_lyris.sh"
    ).read_text()

    assert 'exec python3 "${SCRIPT_DIR}/launch_lyris.py" "$@"' in wrapper


def test_run_artifacts_live_outside_the_git_worktree() -> None:
    payload = _dry_run("baseline")

    assert payload["run_dir"].startswith(
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/nemogym_swe_full_rl/runs/"
    )
    assert not payload["run_dir"].startswith(str(REPO_ROOT))


def test_submit_requires_wandb_key_before_sbatch() -> None:
    env = os.environ.copy()
    env.pop("WANDB_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--mode",
            "submit",
            "--variant",
            "baseline",
            "--repo-dir",
            str(REPO_ROOT),
        ],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "WANDB_API_KEY must be set in the submission environment" in result.stderr
    assert "sbatch" not in result.stderr


def test_submission_sanitizes_host_python_environment_before_ray_starts() -> None:
    payload = _dry_run("baseline")

    assert payload["submission_unset_environment"] == [
        "CONDA_PREFIX",
        "CONDA_PREFIX_1",
        "CONDA_DEFAULT_ENV",
        "CONDA_PYTHON_EXE",
        "CONDA_EXE",
        "_CONDA_EXE",
        "CONDA_ROOT",
        "_CONDA_ROOT",
        "CONDA_SHLVL",
        "CONDA_PROMPT_MODIFIER",
        "_CE_M",
        "_CE_CONDA",
        "VIRTUAL_ENV",
    ]


def test_swe_entry_smoke_compares_direct_and_openhands_bash_execution() -> None:
    python_source = SWE_ENTRY_SMOKE.read_text()
    shell_source = SWE_ENTRY_SMOKE_LYRIS.read_text()

    assert "direct_source_elapsed_s=" in python_source
    assert "BashSession" in python_source
    assert "openhands_source_elapsed_s=" in python_source
    assert "PROMPT_COMMAND" in python_source
    assert "instance_swe_entry.sh" in python_source
    assert "#SBATCH --segment=1" in shell_source
    assert "--writable-tmpfs" in shell_source
    assert "--no-mount home,tmp,bind-paths" in shell_source
