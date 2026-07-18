import json
import subprocess
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.profile_dynamic_sd import (
    BATCH_SIZES,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DRAFTER_REPO_ID,
    DRAFTER_REVISION,
    DYNAMIC_PATCHER_RELATIVE_PATH,
    MANIFEST_NAME,
    TARGET_REPO_ID,
    TARGET_REVISION,
    build_jobs,
    build_runtime_command,
    build_sbatch_command,
    build_venv_setup_command,
    main,
    snapshot_path,
)


def test_build_jobs_covers_six_independent_fixed_k_runs(tmp_path: Path) -> None:
    jobs = build_jobs(tmp_path / "profile")

    assert [job.k for job in jobs] == list(range(6))
    assert [job.job_name for job in jobs] == [
        f"nemorl-qwen32-dynamicsd-k{k}" for k in range(6)
    ]
    assert [job.output_dir for job in jobs] == [
        tmp_path / "profile" / f"k-{k}" for k in range(6)
    ]
    assert len({job.job_name for job in jobs}) == 6


def test_runtime_command_uses_matched_snapshots_and_profile_worker(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    profile_root = tmp_path / "profile"
    hf_home = tmp_path / "hf_home"
    target = snapshot_path(hf_home, TARGET_REPO_ID, TARGET_REVISION)
    drafter = snapshot_path(hf_home, DRAFTER_REPO_ID, DRAFTER_REVISION)
    prompt_template = repo_dir / "examples" / "prompts" / "cot.txt"

    command = build_runtime_command(
        build_jobs(profile_root)[5],
        repo_dir=repo_dir,
        profile_root=profile_root,
        target_snapshot=target,
        drafter_snapshot=drafter,
        prompt_template=prompt_template,
    )

    assert command[:2] == ("env", "CUDA_VISIBLE_DEVICES=0,1")
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in command
    assert "NRL_FORCE_REBUILD_VENVS=true" not in command
    assert (
        "PYTHONPATH=/tmp/nemorl-v0251-qwen32-dynamicsd-k5/profile/"
        f"lib/python3.13/site-packages:{repo_dir}"
    ) in command
    assert "/tmp/nemorl-v0251-qwen32-dynamicsd-k5/profile/bin/python" in command
    assert (
        str(
            repo_dir
            / "experiments"
            / "vllm_0251_drafter_matrix"
            / "dynamic_profile_worker.py"
        )
        in command
    )
    assert "run-k" in command
    assert command[command.index("--k") + 1] == "5"
    assert command[command.index("--root") + 1] == str(profile_root)
    assert command[command.index("--target-snapshot") + 1] == str(target)
    assert command[command.index("--drafter-snapshot") + 1] == str(drafter)
    assert command[command.index("--prompt-template") + 1] == str(prompt_template)


@pytest.mark.parametrize(
    ("mode", "mode_flag"),
    (("test-only", "--test-only"), ("submit", "--parsable")),
)
def test_sbatch_command_clears_singleton_and_uses_bounded_ray_sub(
    tmp_path: Path,
    mode: str,
    mode_flag: str,
) -> None:
    repo_dir = tmp_path / "repo"
    job = build_jobs(tmp_path / "profile")[3]

    command = build_sbatch_command(job, repo_dir=repo_dir, mode=mode)

    assert mode_flag in command
    assert "--dependency=" in command
    assert "--account=coreai_dlalgo_llm" in command
    assert "--partition=gb200" in command
    assert "--nodes=1" in command
    assert "--ntasks-per-node=1" in command
    assert "--exclusive" in command
    assert "--time=05:00:00" in command
    assert "--segment=1" in command
    assert f"--job-name={job.job_name}" in command
    assert f"--output={job.output_dir / 'slurm-%j.out'}" in command
    assert command[-1] == str(repo_dir / "ray.sub")
    assert not any(part.startswith("--gres") for part in command)
    assert not any(
        part.startswith("--dependency=") and part != "--dependency=" for part in command
    )
    assert not any("singleton" in part for part in command)


def test_profile_venv_setup_applies_the_run_scoped_dynamic_patch(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    job = build_jobs(tmp_path / "profile")[0]

    command = build_venv_setup_command(job, repo_dir)
    joined = " ".join(command)

    assert "create_local_venv" in joined
    assert "Missing profile site-packages" in joined
    assert str(repo_dir / DYNAMIC_PATCHER_RELATIVE_PATH) in joined
    assert "NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY" in joined


def _write_runtime_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo_dir = tmp_path / "repo"
    worker = (
        repo_dir
        / "experiments"
        / "vllm_0251_drafter_matrix"
        / "dynamic_profile_worker.py"
    )
    worker.parent.mkdir(parents=True)
    worker.write_text("# worker\n", encoding="utf-8")
    patcher = repo_dir / DYNAMIC_PATCHER_RELATIVE_PATH
    patcher.parent.mkdir(parents=True, exist_ok=True)
    patcher.write_text("# patcher\n", encoding="utf-8")
    (repo_dir / "ray.sub").write_text("#!/bin/bash\n", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text(
        """
[project]
name = "unit"
version = "0.0.0"

[project.optional-dependencies]
vllm = ["vllm==0.25.1"]
""".lstrip(),
        encoding="utf-8",
    )
    prompt_template = repo_dir / "examples" / "prompts" / "cot.txt"
    prompt_template.parent.mkdir(parents=True)
    prompt_template.write_text("Solve: {}\n", encoding="utf-8")

    hf_home = tmp_path / "hf_home"
    for repo_id, revision in (
        (TARGET_REPO_ID, TARGET_REVISION),
        (DRAFTER_REPO_ID, DRAFTER_REVISION),
    ):
        snapshot = snapshot_path(hf_home, repo_id, revision)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    dataset_snapshot = (
        hf_home
        / "hub"
        / f"datasets--{DATASET_REPO_ID.replace('/', '--')}"
        / "snapshots"
        / DATASET_REVISION
    )
    dataset_snapshot.mkdir(parents=True)
    container = tmp_path / "container.sqsh"
    container.write_bytes(b"container")
    return repo_dir, hf_home, prompt_template, container


def test_submit_preflights_all_jobs_then_writes_secret_free_job_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir, hf_home, prompt_template, container = _write_runtime_inputs(tmp_path)
    profile_root = tmp_path / "profile"
    calls: list[tuple[str, ...]] = []
    submitted = iter(range(71000, 71006))

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert "NRL_FORCE_REBUILD_VENVS" not in environment
        assert environment["COMMAND"]
        assert "create_local_venv" in environment["SETUP_COMMAND"]
        assert "PY_EXECUTABLES.VLLM" in environment["SETUP_COMMAND"]
        assert "force_rebuild=True" not in environment["SETUP_COMMAND"]
        assert (
            "NEMO_RL_VENV_DIR=/tmp/nemorl-v0251-qwen32-dynamicsd-"
            in environment["SETUP_COMMAND"]
        )
        assert "'profile'" in environment["SETUP_COMMAND"]
        assert "WANDB_API_KEY" not in environment
        assert "HF_TOKEN" not in environment
        calls.append(command)
        stdout = "" if "--test-only" in command else f"{next(submitted)}\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("WANDB_API_KEY", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")

    exit_code = main(
        (
            "submit",
            "--repo-dir",
            str(repo_dir),
            "--output-dir",
            str(profile_root),
            "--hf-home",
            str(hf_home),
            "--container",
            str(container),
            "--mounts",
            f"{tmp_path}:{tmp_path}",
            "--prompt-template",
            str(prompt_template),
        )
    )

    assert exit_code == 0
    assert len(calls) == 12
    assert all("--test-only" in command for command in calls[:6])
    assert all("--parsable" in command for command in calls[6:])
    manifest = json.loads((profile_root / MANIFEST_NAME).read_text())
    assert manifest["status"] == "submitted"
    assert manifest["profile_contract"] == {
        "batch_sizes": list(BATCH_SIZES),
        "chunked_prefill": True,
        "cudagraph_capture_sizes": [],
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_revision": DATASET_REVISION,
        "draft_tensor_parallel_size": 1,
        "k_values": list(range(6)),
        "max_model_len": 4096,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 256,
        "profile_max_batch_size": 256,
        "num_prompts_per_batch_size": "batch_size * 20",
        "output_len": 256,
        "prefix_cache": False,
        "moe_backend": None,
        "runtime_vllm": "0.25.1",
        "target_tensor_parallel_size": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_runner": "MRv2",
    }
    assert [job["job_id"] for job in manifest["jobs"]] == [
        str(job_id) for job_id in range(71000, 71006)
    ]
    serialized = json.dumps(manifest)
    assert "must-not-leak" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert "HF_TOKEN" not in serialized
