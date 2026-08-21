import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


ROOT = Path(__file__).parents[1]
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFTER_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
CONTAINER_SHA = "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"


def _contract():
    spec = importlib.util.spec_from_file_location(
        "matrix_contract", ROOT / "runtime_contract.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_config(
    *,
    arm: str,
    wandb_run_id: str,
    source_sha: str,
    runtime_sha: str | None = None,
) -> dict:
    contract = _contract().arm_contract(arm)
    speculative = None
    if contract.method is not None:
        speculative = {
            "method": contract.method,
            "num_speculative_tokens": contract.k,
        }
    provenance = {
        "matrix_arm": arm,
        "git_sha": source_sha,
        "target_revision": TARGET_REVISION,
        "drafter_revision": DRAFTER_REVISION,
        "training_horizon_steps": 1000,
        "draft_training_enabled": contract.draft_enabled,
        "draft_refit_enabled": contract.draft_enabled,
        "speculator_type": contract.method,
        "k": contract.k,
        "sequence_packing": False,
        "sequence_parallel": False,
    }
    if runtime_sha is not None:
        provenance.update(
            checkpoint_source_sha=source_sha,
            runtime_git_sha=runtime_sha,
        )
    return {
        "grpo": {
            "seed": 42,
            "max_num_steps": 1000,
            "num_prompts_per_step": 8,
            "num_generations_per_prompt": 4,
        },
        "policy": {
            "train_global_batch_size": 32,
            "sequence_packing": {"enabled": False},
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "sequence_parallel": False,
            },
            "draft": {
                "enabled": contract.draft_enabled,
                "gamma": contract.k or 7,
                "model_name": f"/models/{DRAFTER_REVISION}",
            },
            "generation": {"vllm_kwargs": {"speculative_config": speculative}},
        },
        "data": {
            "train": [{"dataset_name": "DAPOMath17K", "seed": 42}],
            "validation": [{"dataset_name": "DAPOMathAIME2024"}],
        },
        "logger": {
            "wandb": {
                "id": wandb_run_id,
                "project": "sna-nemo-rl-online-drafter",
                "config": provenance,
            }
        },
    }


def _write_checkpoint(
    checkpoint_root: Path,
    *,
    step: int,
    config: dict,
) -> None:
    step_dir = checkpoint_root / f"step_{step}"
    step_dir.mkdir(parents=True)
    (step_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (step_dir / "training_info.json").write_text(json.dumps({"current_step": step}))


@pytest.mark.parametrize(
    ("arm", "draft_enabled", "method", "k"),
    [
        ("baseline", False, None, None),
        ("dflash-fixed-k5", False, "dflash", 5),
        ("dflash-fixed-k7", False, "dflash", 7),
        ("dflash-k5", True, "dflash", 5),
        ("dflash-k7", True, "dflash", 7),
    ],
)
def test_arm_configs_are_matched(
    arm: str, draft_enabled: bool, method: str | None, k: int | None
) -> None:
    contract = _contract()
    config = yaml.safe_load((ROOT / f"{arm}.yaml").read_text())

    contract.validate_arm_config(
        arm,
        config,
        expected_draft_enabled=draft_enabled,
        expected_method=method,
        expected_k=k,
    )


def test_runner_and_submitter_are_fail_closed() -> None:
    runner = (ROOT / "run_oci_hsg.sbatch").read_text()
    submitter = (ROOT / "submit_chain.sh").read_text()

    for marker in (
        "sequence_packing.enabled=false",
        "megatron_cfg.sequence_parallel=false",
        "train_global_batch_size=32",
        "num_prompts_per_step=8",
        "num_generations_per_prompt=4",
        "cudagraph_mode: PIECEWISE",
    ):
        assert marker in runner
    assert "sbatch --test-only" in submitter
    assert "afterok:" in submitter
    assert "segment1 04:00:00 350" in submitter
    assert "segment2 04:00:00 700" in submitter
    assert "segment3 04:00:00 1000" in submitter
    assert "sna-nemo-rl-online-drafter" in submitter


def test_resume_progress_does_not_depend_on_transient_startup_markers() -> None:
    runner = (ROOT / "run_oci_hsg.sbatch").read_text()

    gated_markers = """if [[ "${IS_GATE}" == 1 ]]; then
  grep -Fq "cudagraph_mode: PIECEWISE" "${train_log}"
  grep -Fq "Graph capturing finished" "${train_log}"
fi"""
    assert gated_markers in runner
    assert runner.index(gated_markers) < runner.index("current_step=")


def test_existing_checkpoint_adoption_is_bound_to_source_and_wandb(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint_root = tmp_path / "checkpoints"
    source_sha = "6" * 40
    runtime_sha = "f" * 40
    wandb_run_id = "existing1"
    _write_checkpoint(
        checkpoint_root,
        step=87,
        config=_checkpoint_config(
            arm="baseline",
            wandb_run_id=wandb_run_id,
            source_sha=source_sha,
        ),
    )
    manifest = tmp_path / "resume-manifest.json"

    payload = contract.adopt_existing(
        arm="baseline",
        checkpoint_root=checkpoint_root,
        manifest=manifest,
        runtime_git_sha=runtime_sha,
        checkpoint_source_sha=source_sha,
        expected_wandb_run_id=wandb_run_id,
        target_revision=TARGET_REVISION,
        drafter_revision=DRAFTER_REVISION,
        container_sha256=CONTAINER_SHA,
    )

    assert payload["schema_version"] == 2
    assert payload["adopted_step"] == 87
    assert payload["checkpoint_source_sha"] == source_sha
    assert payload["runtime_git_sha"] == runtime_sha
    assert payload["wandb_run_id"] == wandb_run_id

    _write_checkpoint(
        checkpoint_root,
        step=100,
        config=_checkpoint_config(
            arm="baseline",
            wandb_run_id=wandb_run_id,
            source_sha=source_sha,
            runtime_sha=runtime_sha,
        ),
    )
    validated = contract.validate_resume_manifest(
        arm="baseline",
        checkpoint_root=checkpoint_root,
        manifest=manifest,
        runtime_git_sha=runtime_sha,
        checkpoint_source_sha=source_sha,
        expected_wandb_run_id=wandb_run_id,
        target_revision=TARGET_REVISION,
        drafter_revision=DRAFTER_REVISION,
        container_sha256=CONTAINER_SHA,
    )
    assert validated["adopted_step"] == 87


def test_existing_checkpoint_adoption_rejects_config_drift(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint_root = tmp_path / "checkpoints"
    config = _checkpoint_config(
        arm="dflash-k5",
        wandb_run_id="existing2",
        source_sha="6" * 40,
    )
    config["policy"]["sequence_packing"]["enabled"] = True
    _write_checkpoint(checkpoint_root, step=101, config=config)

    with pytest.raises(ValueError, match="sequence_packing"):
        contract.adopt_existing(
            arm="dflash-k5",
            checkpoint_root=checkpoint_root,
            manifest=tmp_path / "resume-manifest.json",
            runtime_git_sha="f" * 40,
            checkpoint_source_sha="6" * 40,
            expected_wandb_run_id="existing2",
            target_revision=TARGET_REVISION,
            drafter_revision=DRAFTER_REVISION,
            container_sha256=CONTAINER_SHA,
        )


def test_matrix_resume_submits_one_direct_to_1000_job_per_arm() -> None:
    submitter = (ROOT / "submit_resume_matrix.sh").read_text()
    runner = (ROOT / "run_oci_hsg.sbatch").read_text()

    assert "preflight_all" in submitter
    assert "submit_all" in submitter
    assert submitter.index("preflight_all") < submitter.index("submit_all")
    assert "baseline 6b041659" in submitter
    assert "dflash-k5 6b041659" in submitter
    assert "dflash-k7 6b041659" in submitter
    assert "dflash-fixed-k5 242ead65" in submitter
    assert "dflash-fixed-k7 242ead65" in submitter
    assert 'readonly milestones="1000"' in submitter
    assert "400 700 1000" not in submitter
    assert "--dependency=" not in submitter
    assert "WANDB_RESUME=must" in submitter
    assert "CHECKPOINT_RUNTIME_SHA=af5979b04ddd446a813980ae6cedd1871ebabaa0" in (
        submitter
    )
    assert '--runtime-git-sha "${CHECKPOINT_RUNTIME_SHA}"' in runner
    assert "+logger.wandb.config.runtime_git_sha='${CHECKPOINT_RUNTIME_SHA}'" in runner
    assert "+logger.wandb.config.runner_git_sha='${EXPECTED_HEAD}'" in runner


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported matrix arm"):
        _contract().arm_contract("unknown")


@pytest.mark.parametrize(
    ("arm", "draft_enabled", "method", "k"),
    [
        ("baseline", False, None, None),
        ("dflash-fixed-k5", False, "dflash", 5),
        ("dflash-fixed-k7", False, "dflash", 7),
        ("dflash-k5", True, "dflash", 5),
        ("dflash-k7", True, "dflash", 7),
    ],
)
def test_resolved_recipe_preserves_the_fair_matrix(
    arm: str, draft_enabled: bool, method: str | None, k: int | None
) -> None:
    register_omegaconf_resolvers()
    config = load_config(ROOT / f"{arm}.yaml")

    assert config.grpo.seed == 42
    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 4
    assert config.policy.train_global_batch_size == 32
    assert config.policy.sequence_packing.enabled is False
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.policy.megatron_cfg.sequence_parallel is False
    assert config.policy.draft.enabled is draft_enabled
    speculative = config.policy.generation.vllm_kwargs.speculative_config
    if method is None:
        assert speculative is None
    else:
        assert speculative.method == method
        assert speculative.num_speculative_tokens == k
        assert config.policy.draft.gamma == k
    assert config.policy.generation.vllm_kwargs.compilation_config.cudagraph_mode == (
        "PIECEWISE"
    )
    assert config.data.train.dataset_name == "DAPOMath17K"
    assert config.logger.wandb.project == "sna-nemo-rl-online-drafter"
    assert config.logger.wandb.config.draft_training_enabled is draft_enabled
    assert config.logger.wandb.config.draft_refit_enabled is draft_enabled
    assert config.logger.wandb.config.speculator_type == method
    assert config.logger.wandb.config.k == k
    assert config.logger.wandb.config.sequence_packing is False
    assert config.logger.wandb.config.sequence_parallel is False
