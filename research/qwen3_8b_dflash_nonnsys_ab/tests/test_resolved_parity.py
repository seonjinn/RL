import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_nonnsys_ab"
ONLINE_CONFIG = ROOT / "research/qwen3_8b_dflash_online_cp1/config.yaml"
FIXED_CONFIG = ROOT / "research/qwen3_8b_dflash_fixed_dense_control/config.yaml"


def _parity_module() -> ModuleType:
    path = EXPERIMENT_DIR / "resolved_parity.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(module: ModuleType, arm: str) -> object:
    return module.RuntimeInputs(
        arm=arm,
        target_snapshot="/lustre/target/b968",
        drafter_snapshot="/lustre/draft/9b414",
        scratch_root="/raid/scratch/dflash-ab/PARITY",
        wandb_run_id="PARITY",
        wandb_project="sna-nemo-rl-online-drafter",
        expected_head="0" * 40,
    )


def test_runtime_overrides_are_arm_invariant_except_declared_metadata() -> None:
    module = _parity_module()
    fixed = module.runtime_overrides(_inputs(module, "fixed"))
    online = module.runtime_overrides(_inputs(module, "online"))

    fixed_common = {
        item
        for item in fixed
        if not item.startswith("++logger.wandb.config.ab_arm=")
        and not item.startswith("logger.wandb.config.draft_training_enabled=")
    }
    online_common = {
        item
        for item in online
        if not item.startswith("++logger.wandb.config.ab_arm=")
        and not item.startswith("logger.wandb.config.draft_training_enabled=")
    }

    assert fixed_common == online_common
    assert "++logger.wandb.entity=nvidia" in fixed_common
    assert "++logger.wandb.project=sna-nemo-rl-online-drafter" in fixed_common
    assert "policy.draft.update_probe_enabled=false" in fixed_common
    assert "checkpointing.enabled=false" in fixed_common
    assert "logger.tensorboard_enabled=false" in fixed_common


def test_full_resolved_pair_has_only_declared_differences() -> None:
    module = _parity_module()

    report = module.resolve_pair(
        online_config=ONLINE_CONFIG,
        fixed_config=FIXED_CONFIG,
        online_inputs=_inputs(module, "online"),
        fixed_inputs=_inputs(module, "fixed"),
    )

    assert report.unexpected_differences == ()
    assert set(report.allowed_differences) == {
        "logger.wandb.config.ab_arm",
        "logger.wandb.config.draft_refit_enabled",
        "logger.wandb.config.draft_training_enabled",
        "logger.wandb.config.fixed_public_drafter",
        "policy.draft.enabled",
        "policy.draft.optimizer",
    }
    assert report.fixed_update_probe_enabled is False
    assert report.online_update_probe_enabled is False
    assert report.common_fingerprint
