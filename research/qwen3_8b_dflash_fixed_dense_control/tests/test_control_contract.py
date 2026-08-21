import importlib.util
import os
from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


ROOT = Path(__file__).parents[3]
CONTROL_DIR = ROOT / "research/qwen3_8b_dflash_fixed_dense_control"
ONLINE_CONFIG = ROOT / "research/qwen3_8b_dflash_online_cp1/config.yaml"
CONTROL_CONFIG = CONTROL_DIR / "config.yaml"
CAPTURE_SIZES = [
    1,
    2,
    4,
    6,
    8,
    10,
    12,
    16,
    18,
    20,
    24,
    28,
    30,
    32,
    36,
    40,
    42,
    48,
    50,
    56,
    60,
    64,
    70,
    80,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
]


def _module(name: str):
    path = CONTROL_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_control_has_only_declared_science_deltas() -> None:
    register_omegaconf_resolvers()
    online = OmegaConf.to_container(load_config(ONLINE_CONFIG), resolve=True)
    control = OmegaConf.to_container(load_config(CONTROL_CONFIG), resolve=True)
    assert isinstance(online, dict) and isinstance(control, dict)

    online["policy"]["draft"]["enabled"] = False
    online["policy"]["draft"]["optimizer"] = None
    online["policy"]["draft"]["update_probe_enabled"] = False
    online["checkpointing"]["checkpoint_dir"] = control["checkpointing"][
        "checkpoint_dir"
    ]
    online["logger"]["log_dir"] = control["logger"]["log_dir"]
    online["logger"]["wandb"] = control["logger"]["wandb"]

    assert control == online


def test_fixed_control_keeps_generation_and_training_oracle_contract() -> None:
    register_omegaconf_resolvers()
    config = load_config(CONTROL_CONFIG)

    assert config.grpo.max_num_steps == 1000
    assert config.grpo.seed == 42
    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 4
    assert config.grpo.val_at_start is True
    assert config.grpo.val_at_end is True
    assert config.grpo.max_val_samples == 4
    assert config.policy.max_total_sequence_length == 4096
    assert config.policy.train_global_batch_size == 32
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.pipeline_model_parallel_size == 1
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.draft.enabled is False
    assert config.policy.draft.optimizer is None
    assert config.policy.draft.update_probe_enabled is False
    assert config.policy.generation.vllm_cfg.tensor_parallel_size == 1
    assert config.policy.generation.vllm_cfg.enforce_eager is False
    spec = config.policy.generation.vllm_kwargs.speculative_config
    assert spec.method == "dflash"
    assert spec.model == "z-lab/Qwen3-8B-DFlash-b16"
    assert spec.num_speculative_tokens == 7
    compile_config = config.policy.generation.vllm_kwargs.compilation_config
    assert compile_config.cudagraph_mode == "PIECEWISE"
    assert compile_config.cudagraph_capture_sizes == CAPTURE_SIZES
    assert config.data.train.dataset_name == "DAPOMath17K"
    assert config.data.train.seed == 42
    assert config.logger.wandb.project == "sna-nemo-rl-online-drafter"
    assert config.logger.wandb.entity == "nvidia"
    assert config.logger.wandb.config.draft_training_enabled is False
    assert config.logger.wandb.config.draft_refit_enabled is False


def test_gate_requires_target_training_refit_generation_and_validation() -> None:
    validator = _module("validate_gate.py")
    metrics = {
        "train/loss": {"1": 0.5, "2": 0.4},
        "train/grad_norm": {"1": 0.2, "2": 0.3},
        "train/vllm/spec_acceptance_rate": {"1": 0.4, "2": 0.42},
    }
    log = 2 * (
        "Generating trajectories\n"
        "Performing policy generation refit\n"
        "Capturing CUDA graphs (PIECEWISE)\n"
        "Graph capturing finished\n"
    )

    validator.validate_gate(metrics, log)
    validator.validate_validation_history(
        [
            {"_step": 0, "validation/accuracy": 0.0, "validation/avg_length": 128},
            {"_step": 2, "validation/accuracy": 0.25, "validation/avg_length": 256},
        ]
    )

    with pytest.raises(RuntimeError, match="policy generation refit"):
        validator.validate_gate(
            metrics, log.replace("Performing policy generation refit\n", "")
        )


def test_manifest_binds_fixed_control_and_one_persistent_run(tmp_path: Path) -> None:
    contract = _module("resume_contract.py")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    manifest_path = tmp_path / "gate-manifest.json"
    identity = {
        "git_sha": "a" * 40,
        "checkpoint_root": checkpoint_root,
        "wandb_run_id": "fresh123",
        "target_revision": "b968",
        "drafter_revision": "9b414",
        "container_sha256": "6940",
    }

    contract.write_manifest(manifest_path, **identity)
    manifest = contract.validate_manifest(
        manifest_path,
        **{key: value for key, value in identity.items() if key != "wandb_run_id"},
    )
    assert manifest["wandb_run_id"] == "fresh123"
    assert manifest["draft_training_enabled"] is False
    assert manifest["draft_refit_enabled"] is False
    assert manifest["fixed_public_drafter"] is True


def test_submit_chain_uses_one_run_and_arm_local_dependencies(tmp_path: Path) -> None:
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo forecast >&2 ;; *) '
        'counter="${SBATCH_COUNTER}.n"; n=100; test -f "$counter" && n=$(cat "$counter"); '
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
        "FINAL_DIR": "/lustre/fake-fixed-control",
        "CONTAINER": "/lustre/fake.sqsh",
        "TARGET_SNAPSHOT": "/lustre/target/b968",
        "DRAFTER_SNAPSHOT": "/lustre/draft/9b414",
        "SBATCH_ACCOUNT": "test-account",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
    }

    result = subprocess.run(
        ["bash", CONTROL_DIR / "submit_chain.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    calls = sbatch_log.read_text().splitlines()
    assert len(calls) == 8
    actual = calls[1::2]
    run_ids = {
        field.removeprefix("WANDB_RUN_ID=")
        for call in calls
        for field in call.split(",")
        if field.startswith("WANDB_RUN_ID=")
    }
    assert len(run_ids) == 1
    assert "--dependency=" not in actual[0]
    assert "STAGE_MIN_STEP=2" in actual[0]
    assert "STAGE_DEADLINE=00:00:50:00" in actual[0]
    assert "afterok:101" in actual[1]
    assert "afterok:102" in actual[2]
    assert "afterok:103" in actual[3]
    assert "STAGE_MIN_STEP=350" in actual[1]
    assert "STAGE_MIN_STEP=700" in actual[2]
    assert "STAGE_MIN_STEP=1000" in actual[3]
    assert all(
        "WANDB_PROJECT=sna-nemo-rl-online-drafter"  # pragma: allowlist secret
        in call
        for call in calls
    )
    assert "wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/" in result.stdout
