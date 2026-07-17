import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _boundaries() -> ModuleType:
    return importlib.import_module(
        "experiments.vllm_0251_drafter_matrix.profile_dynamic_sd_boundaries"
    )


def test_direct_cli_entrypoint_resolves_repository_imports(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[2]
        / "experiments/vllm_0251_drafter_matrix/profile_dynamic_sd_boundaries.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_cells_matches_the_exact_boundary_matrix(tmp_path: Path) -> None:
    module = _boundaries()

    cells = module.build_cells(tmp_path / "profile")

    assert [(cell.batch_size, cell.k) for cell in cells] == [
        (34, 3),
        (34, 5),
        (35, 3),
        (35, 5),
        (75, 2),
        (75, 3),
        (76, 2),
        (76, 3),
        (85, 1),
        (85, 2),
        (86, 1),
        (86, 2),
    ]
    assert [cell.job_name for cell in cells] == [
        f"coreai_dlalgo_llm-dynamicsd.bs{batch_size}-k{k}"
        for batch_size, k in module.G_BOUNDARY_CELLS
    ]
    assert len({cell.job_name for cell in cells}) == 12
    assert len({cell.output_dir for cell in cells}) == 12


def test_runtime_command_runs_one_exact_cell_with_the_pinned_snapshots(
    tmp_path: Path,
) -> None:
    module = _boundaries()
    repo_dir = tmp_path / "repo"
    profile_root = tmp_path / "profile"
    hf_home = tmp_path / "hf-home"
    target = module.snapshot_path(
        hf_home, module.TARGET_REPO_ID, module.TARGET_REVISION
    )
    drafter = module.snapshot_path(
        hf_home, module.DRAFTER_REPO_ID, module.DRAFTER_REVISION
    )
    prompt_template = repo_dir / "examples" / "prompts" / "cot.txt"
    cell = module.build_cells(profile_root)[0]

    command = module.build_runtime_command(
        cell,
        repo_dir=repo_dir,
        profile_root=profile_root,
        target_snapshot=target,
        drafter_snapshot=drafter,
        prompt_template=prompt_template,
    )

    assert command[:2] == ("env", "CUDA_VISIBLE_DEVICES=0,1")
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert f"PYTHONPATH={repo_dir}" in command
    assert any("bs34-k3/profile/bin/python" in part for part in command)
    assert str(repo_dir / module.WORKER_RELATIVE_PATH) in command
    assert command[command.index("--root") + 1] == str(profile_root)
    assert command[command.index("--k") + 1] == "3"
    assert command[command.index("--batch-sizes") + 1 :] == ("34",)
    assert command[command.index("--target-snapshot") + 1] == str(target)
    assert command[command.index("--drafter-snapshot") + 1] == str(drafter)
    assert command[command.index("--prompt-template") + 1] == str(prompt_template)


@pytest.mark.parametrize(
    ("mode", "mode_flag"),
    (("test-only", "--test-only"), ("submit", "--parsable")),
)
def test_sbatch_command_is_the_exact_isolated_lyris_shape(
    tmp_path: Path,
    mode: str,
    mode_flag: str,
) -> None:
    module = _boundaries()
    repo_dir = tmp_path / "repo"
    cell = module.build_cells(tmp_path / "profile")[0]

    command = module.build_sbatch_command(cell, repo_dir=repo_dir, mode=mode)

    assert command[:2] == ("sbatch", mode_flag)
    assert "--dependency=" in command
    assert "--account=coreai_dlalgo_llm" in command
    assert "--partition=gb200" in command
    assert "--nodes=1" in command
    assert "--ntasks-per-node=1" in command
    assert "--exclusive" in command
    assert "--time=05:00:00" in command
    assert "--segment=1" in command
    assert f"--job-name={cell.job_name}" in command
    assert f"--output={cell.output_dir / 'slurm-%j.out'}" in command
    assert command[-1] == str(repo_dir / "ray.sub")
    assert not any(part.startswith("--gres") for part in command)
    assert not any(
        part.startswith("--dependency=") and part != "--dependency=" for part in command
    )


def _write_runtime_inputs(
    module: ModuleType, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    repo_dir = tmp_path / "repo"
    worker = repo_dir / module.WORKER_RELATIVE_PATH
    worker.parent.mkdir(parents=True)
    worker.write_text("# worker\n", encoding="utf-8")
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

    hf_home = tmp_path / "hf-home"
    for repo_id, revision in (
        (module.TARGET_REPO_ID, module.TARGET_REVISION),
        (module.DRAFTER_REPO_ID, module.DRAFTER_REVISION),
    ):
        snapshot = module.snapshot_path(hf_home, repo_id, revision)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    dataset_snapshot = (
        hf_home
        / "hub"
        / f"datasets--{module.DATASET_REPO_ID.replace('/', '--')}"
        / "snapshots"
        / module.DATASET_REVISION
    )
    dataset_snapshot.mkdir(parents=True)
    container = tmp_path / "container.sqsh"
    container.write_bytes(b"container")
    return repo_dir, hf_home, prompt_template, container


def test_submit_preflights_every_exact_cell_before_any_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _boundaries()
    repo_dir, hf_home, prompt_template, container = _write_runtime_inputs(
        module, tmp_path
    )
    profile_root = tmp_path / "profile"
    calls: list[tuple[str, ...]] = []
    submitted = iter(range(88000, 88012))

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["COMMAND"]
        assert "--batch-sizes" in environment["COMMAND"]
        assert "create_local_venv" in environment["SETUP_COMMAND"]
        assert "PY_EXECUTABLES.VLLM" in environment["SETUP_COMMAND"]
        assert environment["CONTAINER"] == str(container)
        assert environment["HF_HOME"] == str(hf_home)
        assert environment["MOUNTS"] == f"{tmp_path}:{tmp_path}"
        assert "WANDB_API_KEY" not in environment
        assert "HF_TOKEN" not in environment
        calls.append(command)
        stdout = "" if "--test-only" in command else f"{next(submitted)}\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("WANDB_API_KEY", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")

    exit_code = module.main(
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
    assert len(calls) == 24
    assert all("--test-only" in command for command in calls[:12])
    assert all("--parsable" in command for command in calls[12:])
    for preflight, submission in zip(calls[:12], calls[12:], strict=True):
        assert tuple(part for part in preflight if part != "--test-only") == tuple(
            part for part in submission if part != "--parsable"
        )

    manifest_path = profile_root / module.G_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "submitted"
    assert manifest["container"] == str(container)
    assert manifest["hf_home"] == str(hf_home)
    assert manifest["mounts"] == f"{tmp_path}:{tmp_path}"
    assert manifest["dataset"] == {
        "repo_id": module.DATASET_REPO_ID,
        "revision": module.DATASET_REVISION,
    }
    assert [(cell["batch_size"], cell["k"]) for cell in manifest["cells"]] == list(
        module.G_BOUNDARY_CELLS
    )
    assert [cell["job_id"] for cell in manifest["cells"]] == [
        str(job_id) for job_id in range(88000, 88012)
    ]
    for cell in manifest["cells"]:
        assert cell["result_path"] == str(
            profile_root / f"k-{cell['k']}" / f"bs-{cell['batch_size']}" / "result.json"
        )
        assert "--test-only" in cell["preflight_command"]
        assert "--parsable" in cell["submission_command"]
        assert "create_local_venv" in " ".join(cell["venv_setup_command"])
        assert cell["runtime_command"][-2:] == [
            "--batch-sizes",
            str(cell["batch_size"]),
        ]
    assert manifest["profile_contract"] == {
        "batch_sizes": [34, 35, 75, 76, 85, 86],
        "chunked_prefill": True,
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_revision": module.DATASET_REVISION,
        "draft_tensor_parallel_size": 1,
        "max_model_len": 4096,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 256,
        "num_prompts_per_batch_size": "batch_size * 20",
        "output_len": 256,
        "prefix_cache": False,
        "runtime_vllm": "0.25.1",
        "target_tensor_parallel_size": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_runner": "MRv2",
    }
    serialized = json.dumps(manifest)
    assert "must-not-leak" not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert "HF_TOKEN" not in serialized
    assert not list(profile_root.glob(f".{module.G_MANIFEST_NAME}.*.tmp"))
