import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
TRAINING_HORIZON_STEPS = 1000


def _load_resume_contract() -> ModuleType:
    module_path = EXPERIMENT_DIR / "resume_contract.py"
    spec = importlib.util.spec_from_file_location("dflash_resume_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_checkpoint(
    checkpoint_root: Path,
    step: int,
    *,
    horizon_steps: int = TRAINING_HORIZON_STEPS,
) -> Path:
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
    (step_dir / "config.yaml").write_text(f"grpo:\n  max_num_steps: {horizon_steps}\n")
    return step_dir


def test_checkpoint_and_progress_keep_one_1000_step_scheduler_horizon(
    tmp_path: Path,
) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "checkpoints"
    expected = _write_complete_checkpoint(checkpoint_root, 417)

    assert (
        contract.validate_checkpoint(
            checkpoint_root,
            expected_step=417,
            expected_horizon_steps=TRAINING_HORIZON_STEPS,
        )
        == expected
    )
    assert contract.validate_progress(417, 701) == (417, 701)
    assert contract.validate_progress(701, 1000) == (701, 1000)


def test_latest_step_cli_is_a_standalone_query(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _write_complete_checkpoint(checkpoint_root, 417)

    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "resume_contract.py"),
            "--checkpoint-dir",
            str(checkpoint_root),
            "--print-latest-step",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "417"


@pytest.mark.parametrize(
    "mutation",
    ["wrong_horizon", "wrong_total", "wrong_consumed", "missing_optimizer", "tmp"],
)
def test_checkpoint_rejects_incomplete_or_mixed_horizon_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "checkpoints"
    step_dir = _write_complete_checkpoint(checkpoint_root, 417)

    if mutation == "wrong_horizon":
        (step_dir / "config.yaml").write_text("grpo:\n  max_num_steps: 350\n")
    elif mutation == "wrong_total":
        info = json.loads((step_dir / "training_info.json").read_text())
        info["total_steps"] = 416
        (step_dir / "training_info.json").write_text(json.dumps(info))
    elif mutation == "wrong_consumed":
        info = json.loads((step_dir / "training_info.json").read_text())
        info["consumed_samples"] = 0
        (step_dir / "training_info.json").write_text(json.dumps(info))
    elif mutation == "missing_optimizer":
        (step_dir / "policy" / "weights" / "iter_0000000" / "metadata.json").unlink()
    else:
        (checkpoint_root / "tmp_step_418").mkdir()

    with pytest.raises(ValueError):
        contract.validate_checkpoint(
            checkpoint_root,
            expected_step=417,
            expected_horizon_steps=TRAINING_HORIZON_STEPS,
        )


@pytest.mark.parametrize(
    ("previous_step", "current_step"),
    [(0, 1), (1, 1), (701, 700), (999, 1001), (1000, 1000)],
)
def test_invalid_resume_progress_fails_loudly(
    previous_step: int,
    current_step: int,
) -> None:
    contract = _load_resume_contract()

    with pytest.raises(ValueError, match="progress"):
        contract.validate_progress(previous_step, current_step)


def test_gate_manifest_binds_checkpoint_wandb_k_and_horizon(tmp_path: Path) -> None:
    contract = _load_resume_contract()
    checkpoint_root = tmp_path / "k005" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    manifest_path = tmp_path / "k005" / "gate-manifest.json"
    kwargs = {
        "dflash_k": 5,
        "git_sha": "abc123",
        "checkpoint_root": checkpoint_root,
        "wandb_run_id": "wandb-k5",
        "target_revision": "target-rev",
        "drafter_revision": "draft-rev",
        "container_sha256": "container-sha",
        "training_horizon_steps": TRAINING_HORIZON_STEPS,
    }
    contract.write_gate_manifest(manifest_path, **kwargs)

    manifest = contract.validate_gate_manifest(
        manifest_path,
        **{key: value for key, value in kwargs.items() if key != "wandb_run_id"},
    )
    assert manifest["wandb_run_id"] == "wandb-k5"

    with pytest.raises(ValueError, match="training_horizon_steps"):
        contract.validate_gate_manifest(
            manifest_path,
            **{
                **{
                    key: value for key, value in kwargs.items() if key != "wandb_run_id"
                },
                "training_horizon_steps": 350,
            },
        )


def test_gate_and_resume_runners_keep_the_same_1000_step_horizon() -> None:
    gate = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()
    resume = (EXPERIMENT_DIR / "run_resume_oci_hsg.sbatch").read_text()

    for runner in (gate, resume):
        assert "grpo.max_num_steps='${TRAINING_HORIZON_STEPS}'" in runner
        assert "TRAINING_HORIZON_STEPS" in runner
        assert "checkpointing.checkpoint_must_save_by" in runner
        assert "step1000-seed42" in runner
        assert "training_horizon_steps='${TRAINING_HORIZON_STEPS}'" in runner
        assert 'grep -Fq "Capturing CUDA graphs (PIECEWISE)"' in runner
        assert 'grep -Fq "Graph capturing finished"' in runner
    assert "00:00:00:01" in gate
    assert "00:03:30:00" in resume
    assert "+logger.wandb.resume=must" in resume


def test_submitter_builds_independent_k5_and_k7_time_bounded_chains() -> None:
    submitter = (EXPERIMENT_DIR / "submit_resume_chains.sh").read_text()

    assert "for dflash_k in 5 7" in submitter
    assert "for chunk in 1 2 3 4" in submitter
    assert '--dependency="afterok:${previous_job_id}"' in submitter
    assert "sbatch --test-only" in submitter
    assert "sbatch --parsable" in submitter
    assert "afterok:${GATE_JOB_K5}:${GATE_JOB_K7}" not in submitter
    assert "TARGET_TOTAL_STEPS" not in submitter


def test_submitter_does_not_depend_on_a_purged_completed_gate(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "sbatch-calls.txt"
    counter = tmp_path / "counter.txt"
    counter.write_text("9000")
    (fake_bin / "sacct").write_text("#!/bin/bash\necho COMPLETED\n")
    (fake_bin / "sbatch").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'echo "$*" >> "${SBATCH_CALL_LOG}"\n'
        'if [[ " $* " == *" --test-only "* ]]; then\n'
        "  exit 0\n"
        "fi\n"
        'next=$(( $(cat "${SBATCH_COUNTER}") + 1 ))\n'
        'echo "${next}" > "${SBATCH_COUNTER}"\n'
        'echo "${next}"\n'
    )
    for executable in (fake_bin / "sacct", fake_bin / "sbatch"):
        executable.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(call_log),
        "SBATCH_COUNTER": str(counter),
        "REMOTE_REPO": str(tmp_path / "repo"),
        "EXPECTED_HEAD": "a" * 40,
        "RUN_ROOT": str(tmp_path / "runs"),
        "CONTAINER": str(tmp_path / "container.sqsh"),
        "TARGET_SNAPSHOT": str(tmp_path / "target"),
        "DRAFTER_SNAPSHOT": str(tmp_path / "drafter"),
        "WANDB_API_KEY": "test-only-placeholder",
        "GATE_JOB_K5": "1005",
        "GATE_RUN_DIR_K5": str(tmp_path / "gate-k5"),
        "GATE_JOB_K7": "1007",
        "GATE_RUN_DIR_K7": str(tmp_path / "gate-k7"),
    }
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit_resume_chains.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert len(calls) == 16
    assert "--dependency=afterok:1005" not in calls[0]
    assert "--dependency=afterok:1005" not in calls[1]
    assert "--dependency=afterok:9001" in calls[2]
    assert "--dependency=afterok:1007" not in calls[8]
    assert "--dependency=afterok:1007" not in calls[9]
    assert "--dependency=afterok:9005" in calls[10]
