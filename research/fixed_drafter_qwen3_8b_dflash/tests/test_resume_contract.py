import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _load_resume_contract() -> ModuleType:
    module_path = EXPERIMENT_DIR / "resume_contract.py"
    spec = importlib.util.spec_from_file_location("dflash_resume_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("previous_step", "target_step"),
    [
        (1, 350),
        (100, 350),
        (300, 350),
        (350, 700),
        (600, 700),
        (700, 1000),
        (900, 1000),
    ],
)
def test_resume_endpoint_contract(previous_step: int, target_step: int) -> None:
    contract = _load_resume_contract()

    assert contract.validate_transition(previous_step, target_step) == (
        previous_step,
        target_step,
    )


@pytest.mark.parametrize(
    ("previous_step", "target_step"),
    [(0, 350), (99, 350), (1, 700), (350, 1000), (700, 900), (999, 1000)],
)
def test_non_chain_resume_transition_fails_loudly(
    previous_step: int, target_step: int
) -> None:
    contract = _load_resume_contract()

    with pytest.raises(ValueError, match="bounded recovery transition"):
        contract.validate_transition(previous_step, target_step)


def _write_complete_checkpoint(checkpoint_root: Path, step: int) -> Path:
    step_dir = checkpoint_root / f"step_{step}"
    weights_dir = step_dir / "policy" / "weights"
    iteration_dir = weights_dir / "iter_0000000"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text(
        json.dumps({"sharded_backend": "torch_dist", "sharded_backend_version": 1})
    )
    (iteration_dir / ".metadata").write_bytes(b"torch-dist-metadata")
    (iteration_dir / "__0_0.distcp").write_bytes(b"weights-and-optimizer")
    (weights_dir / "latest_checkpointed_iteration.txt").write_text("0")
    (weights_dir / "latest_train_state.pt").write_bytes(b"train-state")
    (step_dir / "train_dataloader.pt").write_bytes(b"dataloader")
    (step_dir / "training_info.json").write_text(
        json.dumps(
            {
                "current_step": step,
                "total_steps": step,
                "consumed_samples": 8 * step,
            }
        )
    )
    return step_dir


def test_checkpoint_preflight_requires_exact_complete_predecessor(
    tmp_path: Path,
) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "checkpoints"
    expected = _write_complete_checkpoint(checkpoint_root, 350)

    assert contract.validate_checkpoint(checkpoint_root, expected_step=350) == expected


@pytest.mark.parametrize(
    "mutation",
    ["later_step", "wrong_total", "wrong_consumed", "missing_optimizer", "tmp_step"],
)
def test_checkpoint_preflight_rejects_incomplete_or_ambiguous_state(
    tmp_path: Path, mutation: str
) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "checkpoints"
    step_dir = _write_complete_checkpoint(checkpoint_root, 350)

    if mutation == "later_step":
        _write_complete_checkpoint(checkpoint_root, 351)
    elif mutation == "wrong_total":
        info = json.loads((step_dir / "training_info.json").read_text())
        info["total_steps"] = 349
        (step_dir / "training_info.json").write_text(json.dumps(info))
    elif mutation == "wrong_consumed":
        info = json.loads((step_dir / "training_info.json").read_text())
        info["consumed_samples"] = 0
        (step_dir / "training_info.json").write_text(json.dumps(info))
    elif mutation == "missing_optimizer":
        (step_dir / "policy" / "weights" / "iter_0000000" / "metadata.json").unlink()
    else:
        (checkpoint_root / "tmp_step_351").mkdir()

    with pytest.raises(ValueError):
        contract.validate_checkpoint(checkpoint_root, expected_step=350)


def test_wandb_run_id_is_recovered_from_successful_gate_log(tmp_path: Path) -> None:
    contract = _load_resume_contract()
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "wandb: View run at https://wandb.ai/nvidia/project/runs/abc123xyz\n"
    )

    assert contract.extract_wandb_run_id(train_log) == "abc123xyz"


def test_gate_manifest_binds_checkpoint_wandb_and_k(tmp_path: Path) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "k003" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    manifest_path = tmp_path / "k003" / "gate-manifest.json"
    contract.write_gate_manifest(
        manifest_path,
        dflash_k=3,
        git_sha="abc123",
        checkpoint_root=checkpoint_root,
        wandb_run_id="wandb-k3",
        target_revision="target-rev",
        drafter_revision="draft-rev",
        container_sha256="container-sha",
    )

    manifest = contract.validate_gate_manifest(
        manifest_path,
        dflash_k=3,
        git_sha="abc123",
        checkpoint_root=checkpoint_root,
        target_revision="target-rev",
        drafter_revision="draft-rev",
        container_sha256="container-sha",
    )
    assert manifest["wandb_run_id"] == "wandb-k3"

    with pytest.raises(ValueError, match="dflash_k"):
        contract.validate_gate_manifest(
            manifest_path,
            dflash_k=5,
            git_sha="abc123",
            checkpoint_root=checkpoint_root,
            target_revision="target-rev",
            drafter_revision="draft-rev",
            container_sha256="container-sha",
        )


def test_resume_runner_reuses_checkpoint_and_wandb_identity() -> None:
    runner = (EXPERIMENT_DIR / "run_resume_oci_hsg.sbatch").read_text()

    assert "${CHECKPOINT_DIR}" in runner
    assert "grpo.max_num_steps='${TARGET_TOTAL_STEPS}'" in runner
    assert "checkpointing.save_period=100" in runner
    assert "+logger.wandb.id='${wandb_run_id}'" in runner
    assert "+logger.wandb.resume=must" in runner
    assert "resume_contract.py" in runner
    assert "GATE_MANIFEST" in runner
    assert "--verify-nemo-resume-paths" in runner
    assert '--expected-step "${EXPECTED_PREVIOUS_STEP}"' in runner
    assert '--expected-step "${TARGET_TOTAL_STEPS}"' in runner


@pytest.mark.parametrize(
    "runner_name", ["run_oci_hsg.sbatch", "run_resume_oci_hsg.sbatch"]
)
def test_every_segment_requires_positive_cuda_graph_evidence(
    runner_name: str,
) -> None:
    runner = (EXPERIMENT_DIR / runner_name).read_text()

    assert 'grep -Fq "Capturing CUDA graphs (PIECEWISE)"' in runner
    assert 'grep -Fq "Graph capturing finished"' in runner
    assert "resume_contract.py" in runner


def test_gate_runner_writes_arm_bound_manifest() -> None:
    runner = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert "--create-gate-manifest" in runner
    assert '"${RUN_DIR}/gate-manifest.json"' in runner


def test_submitter_builds_four_independent_serial_chains() -> None:
    submitter = (EXPERIMENT_DIR / "submit_resume_chains.sh").read_text()

    assert "for dflash_k in 3 5 7 9" in submitter
    assert 'transitions=("1:350" "350:700" "700:1000")' in submitter
    assert '--dependency="afterok:${previous_job_id}"' in submitter
    assert 'checkpoint_dir="${gate_run_dir}/checkpoints"' in submitter
    assert 'gate_manifest="${gate_run_dir}/gate-manifest.json"' in submitter
    assert "sbatch --test-only" in submitter
    assert "sbatch --parsable" in submitter
    assert "afterok:${GATE_JOB_K3}:${GATE_JOB_K5}" not in submitter
