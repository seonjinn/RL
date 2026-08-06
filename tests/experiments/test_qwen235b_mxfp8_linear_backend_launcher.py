from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT / "experiments" / "qwen235b_mxfp8_linear_backends" / "submit_cluster.sh"
)
MATRIX_LAUNCHER = LAUNCHER.with_name("submit_matrix.sh")


def _dry_run(
    tmp_path: Path,
    backend: str,
    dependency_job_id: str = "",
    extra_env: dict[str, str] | None = None,
) -> str:
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    custom_vllm = tmp_path / "vllm"
    custom_vllm.mkdir(exist_ok=True)
    (custom_vllm / ".git").mkdir(exist_ok=True)

    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": backend,
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "EXPERIMENT_ROOT": str(tmp_path / backend),
        "WORK_ROOT": str(tmp_path),
        "DEPENDENCY_JOB_ID": dependency_job_id,
        "RUN_ID": "test-run",
    }
    if extra_env is not None:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _matrix_dry_run(tmp_path: Path, max_steps: str = "8") -> str:
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    custom_vllm = tmp_path / "vllm"
    custom_vllm.mkdir(exist_ok=True)
    (custom_vllm / ".git").mkdir(exist_ok=True)

    env = os.environ | {
        "ACTION": "dry-run",
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "DEPENDENCY_JOB_ID": "12345",
        "MAX_STEPS": max_steps,
        "RUN_ID": "test-run",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(MATRIX_LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_uses_ptyche_account_by_default(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "flashinfer_cutedsl")

    assert "--account=coreai_dlalgo_llm" in output


def test_matrix_submits_each_backend_without_afterok(tmp_path: Path) -> None:
    output = _matrix_dry_run(tmp_path)

    assert output.splitlines().count("backend=flashinfer_cutedsl") == 1
    assert output.splitlines().count("backend=flashinfer_cutlass") == 1
    assert "--dependency=afterok:" not in output


def test_matrix_submit_invokes_two_independent_sbatch_jobs(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%q \' "$@" >> "${SBATCH_LOG}"\n'
        "printf '\\n' >> \"${SBATCH_LOG}\"\n"
    )
    fake_sbatch.chmod(0o755)

    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    custom_vllm = tmp_path / "vllm"
    subprocess.run(["git", "init", "-q", str(custom_vllm)], check=True)
    (custom_vllm / "nemo-rl.env").write_text("# test environment\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(custom_vllm),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    expected_vllm_commit = subprocess.check_output(
        ["git", "-C", str(custom_vllm), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    source_root = tmp_path / "nemo-rl"
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    expected_nemo_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    env = os.environ | {
        "ACTION": "submit",
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "EXPECTED_NEMO_RL_BASE_COMMIT": expected_nemo_commit,
        "EXPECTED_VLLM_COMMIT": expected_vllm_commit,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REPO_DIR_OVERRIDE": str(source_root),
        "RUN_ID": "test-run",
        "SBATCH_LOG": str(sbatch_log),
        "WORK_ROOT": str(tmp_path),
    }
    env.pop("SLURM_ACCOUNT", None)
    result = subprocess.run(
        ["bash", str(MATRIX_LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    submissions = sbatch_log.read_text().splitlines()
    assert len(submissions) == 2
    assert any("--job-name=q235-mx-cutedsl-test-run" in line for line in submissions)
    assert any("--job-name=q235-mx-cutlass-test-run" in line for line in submissions)
    assert all("--dependency=" not in line for line in submissions)
    assert all("afterok" not in line for line in submissions)


def test_max_steps_changes_only_the_requested_run_length(tmp_path: Path) -> None:
    smoke = _dry_run(
        tmp_path,
        "flashinfer_cutedsl",
        extra_env={"MAX_STEPS": "2"},
    )
    measurement = _dry_run(
        tmp_path,
        "flashinfer_cutedsl",
        extra_env={"MAX_STEPS": "8"},
    )

    assert "grpo.max_num_steps=2" in smoke
    assert "grpo.max_num_steps=8" in measurement
    assert smoke.replace("grpo.max_num_steps=2", "grpo.max_num_steps=STEPS") == (
        measurement.replace("grpo.max_num_steps=8", "grpo.max_num_steps=STEPS")
    )


def test_qkvo_scope_changes_only_linear_backend(tmp_path: Path) -> None:
    outputs = {
        backend: _dry_run(tmp_path, backend)
        for backend in ("flashinfer_cutedsl", "flashinfer_cutlass")
    }

    for backend, output in outputs.items():
        assert "grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml" in output
        assert f"linear_backend={backend}" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "quantization_ignored_layer_kws=[lm_head,mlp.gate]" in output
        assert "moe_backend=flashinfer_trtllm" in output
        assert "cluster.num_nodes=16" in output
        assert "cluster.gpus_per_node=4" in output
        assert "cluster.segment_size=16" in output
        assert "grpo.max_num_steps=8" in output

    normalized = {
        backend: output.replace(backend, "LINEAR_BACKEND").replace(
            backend.removeprefix("flashinfer_"), "LINEAR_BACKEND"
        )
        for backend, output in outputs.items()
    }
    assert normalized["flashinfer_cutedsl"] == normalized["flashinfer_cutlass"]


def test_dry_run_captures_runtime_provenance_and_manifest(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "flashinfer_cutedsl")
    expected_nemo_rl_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    custom_vllm_root = tmp_path / "vllm"

    assert f"source {custom_vllm_root}/nemo-rl.env" in output
    assert "vllm_path = Path(vllm.__file__).resolve()" in output
    assert f'custom_vllm_root = Path("{custom_vllm_root}").resolve()' in output
    assert "vllm_path.is_relative_to(custom_vllm_root)" in output
    assert "runtime_nemo_rl_commit=$(git rev-parse HEAD)" in output
    assert expected_nemo_rl_commit in output
    assert "git status --porcelain --untracked-files=all" in output
    assert "runtime_vllm_commit=$(git -C" in output
    assert "run_manifest.json" in output
    assert '"model": "qwen3-235b"' in output
    assert '"linear_backend": "flashinfer_cutedsl"' in output


def test_matrix_isolates_explicit_output_root_by_backend(tmp_path: Path) -> None:
    explicit_root = tmp_path / "runs"
    env = os.environ | {
        "ACTION": "dry-run",
        "EXPERIMENT_ROOT": str(explicit_root),
        "RUN_ID": "test-run",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(MATRIX_LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert (
        result.stdout.splitlines().count(
            f"experiment_root={explicit_root / 'flashinfer_cutedsl'}"
        )
        == 1
    )
    assert (
        result.stdout.splitlines().count(
            f"experiment_root={explicit_root / 'flashinfer_cutlass'}"
        )
        == 1
    )


def test_rejects_non_baseline_backend(tmp_path: Path) -> None:
    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": "flashinfer_trtllm",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported BACKEND" in result.stderr


def test_adds_afterok_dependency_when_requested(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "flashinfer_cutedsl", dependency_job_id="12345")

    assert "--dependency=afterok:12345" in output
