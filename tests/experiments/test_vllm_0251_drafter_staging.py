import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.matrix import CheckpointSpec, G_CLUSTERS
from experiments.vllm_0251_drafter_matrix.stage_drafters import (
    DEFAULT_CONTAINER,
    DEFAULT_MOUNTS,
    MANIFEST_NAME,
    build_sbatch_command,
    collect_checkpoint_specs,
    prepare_worker_snapshot,
    run_stage,
    stage_targets,
    write_manifest,
)


def _write_complete_snapshot(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_collect_checkpoint_specs_uses_only_unique_matrix_checkpoint_identities() -> None:
    targets = collect_checkpoint_specs()

    assert [(target.repo_id, target.revision) for target in targets] == [
        (
            "AICP-Labs/qwen3-32b-dflash-en-zh",
            "68ccc7fd27b104271321b179a2959c759dce5eef",
        ),
        (
            "RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3",
            "3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87",
        ),
        (
            "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
            "edcff83783141eb9383e2bd6c33610d9a3104288",
        ),
        (
            "RedHatAI/Qwen3-30B-A3B-speculator.eagle3",
            "6afc5aa2477b923467fb9a8d906782b984a9a6ba",
        ),
        (
            "RedHatAI/Qwen3-32B-Thinking-speculator.eagle3",
            "a1403e07b73a66fc9ef561463631c31864616933",
        ),
        (
            "RedHatAI/Qwen3-32B-speculator.eagle3",
            "dc84fe7ff1db31efa824776f49c141fc8195eb47",
        ),
        (
            "amd/PARD-Qwen3-0.6B",
            "f9f650fbab180c26498817718f0db5cae8f25136",
        ),
        (
            "nvidia/Qwen3-235B-A22B-Eagle3",
            "33f3c01ce807376d1171301b9a148b1b28f239ba",
        ),
    ]


def test_stage_targets_downloads_to_the_exact_matrix_snapshot_path(
    tmp_path: Path,
) -> None:
    target = CheckpointSpec(
        model_key="unit",
        repo_id="org/model",
        revision="a" * 40,
    )
    calls: list[dict[str, object]] = []
    _write_complete_snapshot(target.snapshot_path(tmp_path))

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(target.snapshot_path(tmp_path))

    entries = stage_targets((target,), tmp_path, snapshot_download, "321")

    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "a" * 40,
            "cache_dir": tmp_path / "hub",
        }
    ]
    assert entries[0].status == "staged"
    assert entries[0].path == str(target.snapshot_path(tmp_path))
    assert entries[0].job_id == "321"


def test_stage_targets_fails_closed_when_download_returns_another_path(
    tmp_path: Path,
) -> None:
    target = CheckpointSpec(
        model_key="unit", repo_id="org/model", revision="b" * 40
    )

    with pytest.raises(RuntimeError, match="unexpected snapshot path"):
        stage_targets(
            (target,),
            tmp_path,
            lambda **_: str(tmp_path / "wrong"),
            None,
        )


def test_stage_targets_fails_closed_when_snapshot_has_no_weights(
    tmp_path: Path,
) -> None:
    target = CheckpointSpec(
        model_key="unit", repo_id="org/model", revision="e" * 40
    )
    snapshot = target.snapshot_path(tmp_path)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no model weights"):
        stage_targets((target,), tmp_path, lambda **_: str(snapshot), None)


def test_write_manifest_is_atomic_and_machine_readable(tmp_path: Path) -> None:
    target = CheckpointSpec(
        model_key="unit", repo_id="org/model", revision="c" * 40
    )
    _write_complete_snapshot(target.snapshot_path(tmp_path / "hf-home"))
    entries = stage_targets(
        (target,),
        tmp_path / "hf-home",
        lambda **_: str(target.snapshot_path(tmp_path / "hf-home")),
        "456",
    )

    manifest_path = write_manifest(tmp_path, entries)

    assert json.loads(manifest_path.read_text()) == {
        "checkpoints": [
            {
                "job_id": "456",
                "path": str(target.snapshot_path(tmp_path / "hf-home")),
                "repo_id": "org/model",
                "revision": "c" * 40,
                "status": "staged",
            }
        ]
    }
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("mode", "required_flag"),
    (("test-only", "--test-only"), ("submit", "--parsable")),
)
def test_sbatch_command_is_single_node_lyris_staging_without_gpu_request(
    tmp_path: Path, mode: str, required_flag: str
) -> None:
    command = build_sbatch_command(
        mode=mode,
        output_dir=tmp_path / "out",
        hf_home=G_CLUSTERS[0].hf_home,
        container=Path(DEFAULT_CONTAINER),
        mounts=DEFAULT_MOUNTS,
        wrapper_path=tmp_path / "submit_stage_drafters.sh",
        worker_path=tmp_path / "stage_drafters.py",
    )

    assert command[:2] == ("sbatch", required_flag)
    assert "--dependency=" in command
    assert "--account=coreai_dlalgo_llm" in command
    assert "--partition=gb200" in command
    assert "--nodes=1" in command
    assert "--segment=1" in command
    assert "--exclusive" not in command
    assert f"--export=HF_HOME={G_CLUSTERS[0].hf_home}" in command
    assert not any(part.startswith("--export=ALL") for part in command)
    assert ("--hold" in command) is (mode == "submit")
    assert f"--container-image={DEFAULT_CONTAINER}" in command
    assert f"--container-mounts={DEFAULT_MOUNTS}" in command
    assert not any(part.startswith("--gres") for part in command)
    assert not any("singleton" in part for part in command)
    wrapper_index = command.index(str(tmp_path / "submit_stage_drafters.sh"))
    assert command[wrapper_index + 1 : wrapper_index + 3] == (
        "--worker-script",
        str(tmp_path / "stage_drafters.py"),
    )
    assert command[wrapper_index + 3 :] == (
        "--output-dir",
        str(tmp_path / "out"),
        "--hf-home",
        str(G_CLUSTERS[0].hf_home),
    )


def test_submit_wrapper_separates_host_and_container_python() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "experiments/vllm_0251_drafter_matrix/submit_stage_drafters.sh"
    )

    text = wrapper.read_text()
    assert "Copyright (c) 2026, NVIDIA CORPORATION" in text
    assert "set -euo pipefail" in text
    assert 'if [[ "${1:-}" == "--worker-script" ]]' in text
    assert 'python_bin="/opt/nemo_rl_venv/bin/python"' in text
    assert 'python_bin="${PYTHON_BIN:-python3}"' in text
    assert '"${python_bin}" "${worker_script}" --worker "${worker_args[@]}"' in text
    assert "worker failed before terminal manifest" in text
    assert 'exec "${python_bin}"' in text


def test_prepare_worker_snapshot_is_content_addressed_and_complete(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkout/experiments/vllm_0251_drafter_matrix"
    source.mkdir(parents=True)
    worker = source / "stage_drafters.py"
    worker.write_text("worker\n", encoding="utf-8")
    (source / "matrix.py").write_text("matrix\n", encoding="utf-8")

    snapshot = prepare_worker_snapshot(tmp_path / "output", worker)

    assert snapshot.read_text(encoding="utf-8") == "worker\n"
    assert snapshot.with_name("matrix.py").read_text(encoding="utf-8") == "matrix\n"
    assert snapshot.is_relative_to(tmp_path / "output")
    assert oct(snapshot.stat().st_mode & 0o777) == "0o444"


def test_run_stage_records_terminal_failure_when_worker_initialization_fails(
    tmp_path: Path,
) -> None:
    target = CheckpointSpec(
        model_key="unit", repo_id="org/model", revision="d" * 40
    )

    with pytest.raises(RuntimeError, match="import failed"):
        run_stage(
            tmp_path,
            tmp_path / "hf-home",
            checkpoints=(target,),
            snapshot_download_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("import failed")
            ),
        )

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "import failed"
    assert manifest["checkpoints"][0]["status"] == "failed"


def test_show_cli_is_runnable_as_a_script(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[2]
        / "experiments/vllm_0251_drafter_matrix/stage_drafters.py"
    )

    result = subprocess.run(
        (sys.executable, str(script), "show", "--output-dir", str(tmp_path)),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["checkpoints"][0]["status"] == "planned"
