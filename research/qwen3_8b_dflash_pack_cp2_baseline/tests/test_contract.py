import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[1]


def _module(name: str) -> ModuleType:
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm", "draft_enabled"), (("fixed", False), ("online", True))
)
def test_runtime_contract_is_packed_tp2_cp2_k5(
    arm: str, draft_enabled: bool
) -> None:
    contract = _module("runtime_contract.py")
    cell = contract.resolve_arm(arm)
    overrides = contract.runtime_overrides(
        cell,
        target_snapshot="/lustre/target/b968826d9c46dd6066d109eabc6255188de91218",
        drafter_snapshot="/lustre/draft/9b41424b7109f9c5413454f481b09a82b85333f4",
        scratch_root="/raid/scratch/123/fixed",
        wandb_run_id="run-id",
        expected_head="a" * 40,
    )

    assert "grpo.seed=42" in overrides
    assert "grpo.max_num_steps=30" in overrides
    assert "grpo.num_prompts_per_step=8" in overrides
    assert "grpo.num_generations_per_prompt=4" in overrides
    assert "policy.train_global_batch_size=32" in overrides
    assert "policy.train_micro_batch_size=1" in overrides
    assert "policy.sequence_packing.enabled=true" in overrides
    assert "policy.megatron_cfg.tensor_model_parallel_size=2" in overrides
    assert "policy.megatron_cfg.context_parallel_size=2" in overrides
    assert "policy.megatron_cfg.sequence_parallel=true" in overrides
    assert "policy.make_sequence_length_divisible_by=16" in overrides
    assert "policy.draft.gamma=5" in overrides
    assert "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5" in overrides
    assert f"policy.draft.enabled={str(draft_enabled).lower()}" in overrides


def test_unknown_arm_is_rejected() -> None:
    contract = _module("runtime_contract.py")
    with pytest.raises(ValueError, match="Unsupported baseline arm"):
        contract.resolve_arm("adaptive")


@pytest.mark.parametrize("arm", ("fixed", "online"))
def test_runtime_overrides_compose_against_derived_pr11_config(arm: str) -> None:
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    contract = _module("runtime_contract.py")
    overrides = contract.runtime_overrides(
        contract.resolve_arm(arm),
        target_snapshot="/lustre/target/b968826d9c46dd6066d109eabc6255188de91218",
        drafter_snapshot="/lustre/draft/9b41424b7109f9c5413454f481b09a82b85333f4",
        scratch_root="/raid/scratch/123/arm",
        wandb_run_id="run-id",
        expected_head="a" * 40,
    )

    config = parse_hydra_overrides(load_config(ROOT / "config.yaml"), list(overrides))

    assert config.logger.wandb.tags[-1] == arm
    assert config.policy.sequence_packing.enabled is True
    assert config.policy.megatron_cfg.context_parallel_size == 2
    assert config.data.validation is None
    assert config.grpo.val_period == 0
    assert config.grpo.val_at_start is False
    assert config.grpo.val_at_end is False


def test_runner_pins_all_immutable_inputs_and_packing_recipe() -> None:
    runner = (ROOT / "run_pair_oci_hsg.sbatch").read_text()

    for marker in (
        "443e7243ae2a235b6dcd8f4918fea86e693630a9",
        "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44",
        "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941",
        'readonly scratch_root="/raid/scratch/${SLURM_JOB_ID}"',
        "dataset-manifest.json",
        "verify-pair",
        "resolved-parity.json",
        "parity.py",
        "draft_refit_finalize=complete",
        "#SBATCH --time=04:00:00",
        "#SBATCH --partition=batch",
    ):
        assert marker in runner
    assert "#SBATCH --time=04:30:00" not in runner
    assert "#SBATCH --partition=batch_long" not in runner
    assert "adaptive" not in runner.lower()
    assert "fixed interval" not in runner.lower()


def test_parity_contract_allows_only_arm_owned_state() -> None:
    parity = _module("parity.py")
    assert parity.allowed_difference_paths() == {
        "logger.wandb.config.ab_arm",
        "logger.wandb.config.draft_refit_enabled",
        "logger.wandb.config.draft_training_enabled",
        "logger.wandb.name",
        "logger.wandb.tags",
        "policy.draft.enabled",
    }
