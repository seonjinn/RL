import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_nonnsys_ab"
BASE_PRODUCT_SHA = "79e80af96a13522e6049658663a8c40ab21e8314"
OPTIMIZED_PRODUCT_SHA = "f909e3d124bb663db4099e88f6846e55b0500912"


def _module(name: str):
    path = EXPERIMENT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_overrides_bind_source_identity_without_changing_online_arm() -> None:
    parity = _module("resolved_parity.py")
    inputs = parity.RuntimeInputs(
        arm="online",
        target_snapshot="/lustre/target",
        drafter_snapshot="/lustre/drafter",
        scratch_root="/raid/scratch/run",
        wandb_run_id="run123",
        wandb_project="sna-nemo-rl-online-drafter",
        expected_head="a" * 40,
        product_source_sha=OPTIMIZED_PRODUCT_SHA,
        source_arm="optimized",
    )

    overrides = parity.runtime_overrides(inputs)

    assert "++logger.wandb.config.ab_arm=online" in overrides
    assert "++logger.wandb.config.source_arm=optimized" in overrides
    assert (
        f"++logger.wandb.config.product_source_sha={OPTIMIZED_PRODUCT_SHA}" in overrides
    )
    assert "policy.draft.update_probe_enabled=false" in overrides
    assert "grpo.max_num_steps=50" in overrides


def test_runner_binds_product_and_harness_heads_to_online_config() -> None:
    script = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert ': "${PRODUCT_HEAD:?Set exact product source SHA}"' in script
    assert ': "${SOURCE_ARM:?Set base or optimized}"' in script
    assert ': "${PARITY_PROOF:?Set source parity proof}"' in script
    assert "readonly arm=online" in script
    assert 'test "${ARM}" = "${arm}"' in script
    assert '"${product_head}" "${EXPECTED_HEAD}"' in script
    assert 'python3 "${source_parity}" validate-proof' in script
    assert "--product-source-sha '${product_head}'" in script
    assert "--source-arm '${SOURCE_ARM}'" in script


def test_source_pair_submitter_has_strict_two_arm_forecast_and_monitoring() -> None:
    script = (EXPERIMENT_DIR / "submit_source_pair.sh").read_text()

    assert f"readonly base_product_head={BASE_PRODUCT_SHA}" in script
    assert f"readonly optimized_product_head={OPTIMIZED_PRODUCT_SHA}" in script
    assert script.count('submit "base"') == 2
    assert script.count('submit "optimized"') == 2
    assert "ARM=online" in script
    assert '"${source_parity}" check' in script
    assert '"${monitor_script}" "${base_job}" "${optimized_job}"' in script


def test_source_parity_locks_exact_product_delta() -> None:
    parity = _module("source_parity.py")

    assert parity.BASE_PRODUCT_SHA == BASE_PRODUCT_SHA
    assert parity.OPTIMIZED_PRODUCT_SHA == OPTIMIZED_PRODUCT_SHA
    assert parity.EXPECTED_PRODUCT_DELTA == (
        "nemo_rl/algorithms/loss/wrapper.py",
        "nemo_rl/models/megatron/draft/step_state.py",
        "tests/unit/algorithms/test_draft_loss_wrapper.py",
        "tests/unit/models/megatron/test_draft_step_state.py",
    )
    assert "git ls-files --others --exclude-standard" in parity.SUBMODULE_CLEAN_COMMAND
