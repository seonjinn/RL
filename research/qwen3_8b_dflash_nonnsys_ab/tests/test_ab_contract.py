import importlib.util
import os
from pathlib import Path
import subprocess
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_nonnsys_ab"
ONLINE_CONFIG = ROOT / "research/qwen3_8b_dflash_online_cp1/config.yaml"
FIXED_CONFIG = ROOT / "research/qwen3_8b_dflash_fixed_dense_control/config.yaml"
OPTIMIZED_SOURCE_SHA = "79e80af96a13522e6049658663a8c40ab21e8314"


def _module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm", "config_path", "draft_training_enabled"),
    [
        ("fixed", FIXED_CONFIG, False),
        ("online", ONLINE_CONFIG, True),
    ],
)
def test_arm_contract(
    arm: str,
    config_path: Path,
    draft_training_enabled: bool,
) -> None:
    contract = _module("runtime_contract.py")

    resolved = contract.resolve_arm(arm)

    assert ROOT / resolved.config_path == config_path
    assert resolved.draft_training_enabled is draft_training_enabled
    assert resolved.update_probe_enabled is False


def test_unknown_arm_fails_loudly() -> None:
    contract = _module("runtime_contract.py")

    with pytest.raises(ValueError, match="Unsupported A/B arm"):
        contract.resolve_arm("typo")


def test_fixed_and_online_configs_only_change_trainer_owned_draft_state() -> None:
    register_omegaconf_resolvers()
    online = OmegaConf.to_container(load_config(ONLINE_CONFIG), resolve=True)
    fixed = OmegaConf.to_container(load_config(FIXED_CONFIG), resolve=True)
    assert isinstance(online, dict) and isinstance(fixed, dict)

    online["policy"]["draft"]["enabled"] = False
    online["policy"]["draft"]["optimizer"] = None
    online["policy"]["draft"]["update_probe_enabled"] = False
    online["checkpointing"] = fixed["checkpointing"]
    online["logger"] = fixed["logger"]

    assert fixed == online


@pytest.mark.parametrize("config_path", [ONLINE_CONFIG, FIXED_CONFIG])
def test_both_arms_share_the_performance_oracle(config_path: Path) -> None:
    register_omegaconf_resolvers()
    config = load_config(config_path)

    assert config.grpo.seed == 42
    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 4
    assert config.policy.train_global_batch_size == 32
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.sequence_packing.enabled is False
    assert config.policy.generation.vllm_cfg.enforce_eager is False
    speculative = config.policy.generation.vllm_kwargs.speculative_config
    assert speculative.method == "dflash"
    assert speculative.num_speculative_tokens == 7
    assert speculative.model == "z-lab/Qwen3-8B-DFlash-b16"


def test_runner_is_fifty_step_non_profiled_wandb_measurement() -> None:
    script = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert f"readonly optimized_source_sha={OPTIMIZED_SOURCE_SHA}" in script
    assert "grpo.max_num_steps=50" in script
    assert "grpo.val_period=1000000" in script
    assert "grpo.val_at_start=false" in script
    assert "grpo.val_at_end=false" in script
    assert "checkpointing.enabled=false" in script
    assert "policy.draft.update_probe_enabled='${update_probe_enabled}'" in script
    assert "logger.wandb_enabled=true" in script
    assert "logger.tensorboard_enabled=false" in script
    assert "export WANDB__DISABLE_STATS=true" in script
    assert "export WANDB_DISABLE_STATS=true" not in script
    assert "unset NRL_NSYS_WORKER_PATTERNS" in script
    assert "unset NRL_NSYS_PROFILE_STEP_RANGE" in script
    assert "unset NRL_NSYS_EXTRA_OPTIONS" in script
    assert "Step 50" in script


def test_runner_preserves_mars_storage_layout() -> None:
    script = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert '[[ "${REMOTE_REPO}" == /home/* ]]' in script
    assert '[[ "${FINAL_DIR}" == /lustre/* ]]' in script
    assert 'readonly scratch_root="/raid/scratch/dflash-ab/${SLURM_JOB_ID}"' in script
    assert 'readonly ray_root="/raid/scratch/dflash-ab-ray/${SLURM_JOB_ID}"' in script
    assert script.index("trap archive EXIT") < script.index(
        'test "$(git -C "${REMOTE_REPO}" rev-parse HEAD)"'
    )


def test_submit_pair_uses_fresh_independent_runs(tmp_path: Path) -> None:
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo forecast >&2 ;; *) '
        'counter="${SBATCH_COUNTER}.n"; n=700; test -f "$counter" && n=$(cat "$counter"); '
        'n=$((n + 1)); printf "%s" "$n" > "$counter"; echo "$n" ;; esac\n'
    )
    sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(sbatch_log),
        "SBATCH_COUNTER": str(tmp_path / "counter"),
        "REMOTE_REPO": str(ROOT),
        "EXPECTED_HEAD": "b" * 40,
        "FINAL_ROOT": "/lustre/fake-dflash-ab",
        "CONTAINER": "/lustre/fake.sqsh",
        "TARGET_SNAPSHOT": "/lustre/target/b968",
        "DRAFTER_SNAPSHOT": "/lustre/draft/9b414",
        "SBATCH_ACCOUNT": "test-account",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
    }

    result = subprocess.run(
        ["bash", EXPERIMENT_DIR / "submit_pair.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    calls = sbatch_log.read_text().splitlines()
    assert len(calls) == 4
    forecasts = calls[:2]
    actual = calls[2:]
    assert all("--test-only" in call for call in forecasts)
    assert all("--test-only" not in call for call in actual)
    assert "ARM=fixed" in forecasts[0]
    assert "ARM=online" in forecasts[1]
    assert all("--dependency=" not in call for call in actual)
    assert "ARM=fixed" in actual[0]
    assert "ARM=online" in actual[1]
    run_ids = {
        field.removeprefix("WANDB_RUN_ID=")
        for call in calls
        for field in call.split(",")
        if field.startswith("WANDB_RUN_ID=")
    }
    assert len(run_ids) == 2
    assert "fixed_job=701" in result.stdout
    assert "online_job=702" in result.stdout
    assert (
        result.stdout.count("https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/")
        == 2
    )


def test_submit_pair_cancels_fixed_if_online_submission_fails(tmp_path: Path) -> None:
    sbatch_log = tmp_path / "sbatch.log"
    scancel_log = tmp_path / "scancel.log"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo forecast >&2 ;; '
        '*"ARM=fixed,"*) echo 801 ;; *"ARM=online,"*) exit 9 ;; esac\n'
    )
    sbatch.chmod(0o755)
    scancel = tmp_path / "scancel"
    scancel.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SCANCEL_CALL_LOG"\n')
    scancel.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(sbatch_log),
        "SCANCEL_CALL_LOG": str(scancel_log),
        "REMOTE_REPO": str(ROOT),
        "EXPECTED_HEAD": "b" * 40,
        "FINAL_ROOT": "/lustre/fake-dflash-ab-failure",
        "CONTAINER": "/lustre/fake.sqsh",
        "TARGET_SNAPSHOT": "/lustre/target/b968",
        "DRAFTER_SNAPSHOT": "/lustre/draft/9b414",
        "SBATCH_ACCOUNT": "test-account",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
    }

    result = subprocess.run(
        ["bash", EXPERIMENT_DIR / "submit_pair.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert scancel_log.read_text().strip() == "801"
    assert "online submission failed; cancelled fixed job 801" in result.stderr


def test_monitor_is_filtered_and_polls_for_five_minutes() -> None:
    script = (EXPERIMENT_DIR / "monitor_pair.sh").read_text()

    assert "for pass in 1 2 3 4 5" in script
    assert "sleep 60" in script
    assert 'squeue -j "${fixed_job},${online_job}"' in script
    assert "squeue --me" not in script
