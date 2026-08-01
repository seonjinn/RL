from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
SOURCE_DIGEST = "sha256:" + "a" * 64
SOURCE_COMMIT = "b" * 40
NEMORL_COMMIT = "b" * 40
BRIDGE_COMMIT = "c" * 40
MCORE_COMMIT = "d" * 40
TE_COMMIT = "e" * 40


def _load_runtime_probe() -> ModuleType:
    path = EXPERIMENT_DIR / "validate_container_runtime.py"
    spec = importlib.util.spec_from_file_location("container_runtime_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _runtime_modules(module: ModuleType, environment_root: Path) -> dict[str, object]:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"GPU-{index}"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            del index
            return 10, 0

    site_packages = environment_root / "lib" / "python3.13" / "site-packages"
    modules: dict[str, object] = {
        name: SimpleNamespace(__file__=str(site_packages / name / "__init__.py"))
        for name in module.REQUIRED_MODULE_DISTRIBUTIONS
    }
    modules["torch"] = SimpleNamespace(
        __file__=str(site_packages / "torch" / "__init__.py"),
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
    )
    return modules


def _run_script(
    relative_path: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    env.update(environment)
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / relative_path)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _stage_environment(container_dir: Path, **extra: str) -> dict[str, str]:
    environment = {
        "SOURCE_IMAGE": "nvcr.io/nvidian/nemo-rl@" + SOURCE_DIGEST,
        "SOURCE_DIGEST": SOURCE_DIGEST,
        "SOURCE_COMMIT": SOURCE_COMMIT,
        "OUTPUT_PREFIX": "nemo_rl_nightly_fixture",
        "CONTAINER_DIR": str(container_dir),
    }
    environment.update(extra)
    return environment


def test_runtime_probe_rejects_package_loaded_outside_uv_environment(
    tmp_path: Path,
) -> None:
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    modules = _runtime_modules(module, environment_root)
    modules["megatron.core"] = SimpleNamespace(
        __file__="/ambient/site-packages/megatron/core/__init__.py"
    )

    with pytest.raises(RuntimeError, match="megatron.core.*outside"):
        module.probe_runtime(
            expected_device_count=4,
            expected_environment_root=environment_root,
            expected_project_root=tmp_path / "project",
            importer=lambda name: modules[name],
            version_getter=lambda distribution: f"fixture-{distribution}",
            interpreter_path=environment_root / "bin" / "python",
            runtime_prefix=environment_root,
            environment={"UV_PROJECT_ENVIRONMENT": str(environment_root)},
        )


def test_runtime_probe_rejects_ambient_pythonpath(tmp_path: Path) -> None:
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    modules = _runtime_modules(module, environment_root)

    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        module.probe_runtime(
            expected_device_count=4,
            expected_environment_root=environment_root,
            expected_project_root=tmp_path / "project",
            importer=lambda name: modules[name],
            version_getter=lambda distribution: f"fixture-{distribution}",
            interpreter_path=environment_root / "bin" / "python",
            runtime_prefix=environment_root,
            environment={
                "PYTHONPATH": "/ambient/site-packages",
                "UV_PROJECT_ENVIRONMENT": str(environment_root),
            },
        )


def test_runtime_probe_allows_only_megatron_editables_from_project_root(
    tmp_path: Path,
) -> None:
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    project_root = tmp_path / "project"
    modules = _runtime_modules(module, environment_root)
    modules["megatron.core"] = SimpleNamespace(
        __file__=str(project_root / "megatron" / "core" / "__init__.py")
    )
    modules["megatron.bridge"] = SimpleNamespace(
        __file__=str(project_root / "megatron" / "bridge" / "__init__.py")
    )

    result = module.probe_runtime(
        expected_device_count=4,
        expected_environment_root=environment_root,
        expected_project_root=project_root,
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
        interpreter_path=environment_root / "bin" / "python",
        runtime_prefix=environment_root,
        environment={"UV_PROJECT_ENVIRONMENT": str(environment_root)},
    )

    assert result["expected_project_root"] == str(project_root)


def test_runtime_probe_reads_exact_transformer_engine_vcs_commit() -> None:
    module = _load_runtime_probe()
    commit = "a" * 40
    distribution = SimpleNamespace(
        read_text=lambda name: (
            json.dumps(
                {
                    "url": "https://github.com/NVIDIA/TransformerEngine.git",
                    "vcs_info": {"vcs": "git", "commit_id": commit},
                }
            )
            if name == "direct_url.json"
            else None
        )
    )

    assert (
        module._distribution_vcs_commit(
            "transformer-engine", distribution_getter=lambda name: distribution
        )
        == commit
    )


@pytest.mark.parametrize(
    ("relative_path", "environment"),
    (
        (
            "scripts/validate_oci_container_runtime.sub",
            {
                "CONTAINER": "/lustre/example/nightly.sqsh",
                "CONTAINER_SHA256": "c" * 64,
                "ARTIFACT_DIR": "/lustre/example/runtime-artifacts",
                "EXPECTED_NEMORL_SHA": NEMORL_COMMIT,
                "EXPECTED_BRIDGE_SHA": BRIDGE_COMMIT,
                "EXPECTED_MCORE_SHA": MCORE_COMMIT,
                "EXPECTED_TE_SHA": TE_COMMIT,
                "SOURCE_PROVENANCE_VERIFIER": str(
                    EXPERIMENT_DIR / "scripts" / "verify_source_provenance.sh"
                ),
            },
        ),
        (
            "scripts/stage_enroot_image.sbatch",
            _stage_environment(Path("/lustre/example/containers")),
        ),
    ),
)
def test_scheduler_preflight_invokes_real_sbatch_test_only(
    tmp_path: Path,
    relative_path: str,
    environment: dict[str, str],
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    _write_executable(
        fake_bin / "sbatch",
        '#!/bin/bash\nprintf \'%s\\n\' "$*" >"${SBATCH_LOG}"\n',
    )

    result = _run_script(
        relative_path,
        **environment,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SBATCH_LOG=str(sbatch_log),
        SBATCH_TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    submitted_arguments = sbatch_log.read_text()
    assert "--test-only" in submitted_arguments
    assert "--parsable" not in submitted_arguments
    assert "--export=ALL" not in submitted_arguments


def test_runtime_job_uses_worker_parity_uv_environment_and_exact_provenance(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    assert "#SBATCH --time=01:00:00" in source_wrapper.read_text()
    spool_dir = tmp_path / "slurm-spool" / "job733"
    spool_dir.mkdir(parents=True)
    spooled_wrapper = spool_dir / "slurm_script"
    spooled_wrapper.write_text(source_wrapper.read_text())
    spooled_wrapper.chmod(0o755)
    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"container")
    container_digest = hashlib.sha256(container.read_bytes()).hexdigest()
    artifact_dir = tmp_path / "artifacts"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.log"
    provenance_log = tmp_path / "source-provenance.log"
    provenance_verifier = tmp_path / "verify_source_provenance.sh"
    _write_executable(
        provenance_verifier,
        '#!/bin/bash\nset -euo pipefail\nprintf \'%s\\n\' "$@" >"${PROVENANCE_LOG}"\n',
    )
    _write_executable(
        fake_bin / "srun",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >"${SRUN_LOG}"
output=
while (($#)); do
  if [[ "$1" == "--output" ]]; then
    shift
    output=$1
  fi
  shift
done
printf '{"status":"passed"}\n' >"${output}"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHONPATH": "/ambient/site-packages",
            "SRUN_LOG": str(srun_log),
            "SLURM_JOB_ID": "733",
            "CONTAINER_RUNTIME_VALIDATOR": str(
                (EXPERIMENT_DIR / "validate_container_runtime.py").resolve()
            ),
            "PROJECT_ROOT": str(REPO_ROOT),
            "CONTAINER": str(container),
            "CONTAINER_SHA256": container_digest,
            "ARTIFACT_DIR": str(artifact_dir),
            "EXPECTED_NEMORL_SHA": NEMORL_COMMIT,
            "EXPECTED_BRIDGE_SHA": BRIDGE_COMMIT,
            "EXPECTED_MCORE_SHA": MCORE_COMMIT,
            "EXPECTED_TE_SHA": TE_COMMIT,
            "SOURCE_PROVENANCE_VERIFIER": str(provenance_verifier),
            "PROVENANCE_LOG": str(provenance_log),
        }
    )

    result = subprocess.run(
        ["bash", str(spooled_wrapper)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    command = srun_log.read_text()
    assert "--export=ALL" not in command
    assert "--export=NIL" in command
    assert "env -i" in command
    assert "HOME=/root" in command
    assert (
        "PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin:/opt/nemo_rl_venv/bin"
        in command
    )
    assert "UV_CACHE_DIR=/tmp" not in command
    assert "CUDA_HOME=/usr/local/cuda" in command
    assert "NRL_FORCE_REBUILD_VENVS=true" in command
    assert "UV_PROJECT_ENVIRONMENT=/tmp/nemo-rl-runtime-733" in command
    assert "uv run --locked --extra mcore" in command
    assert "--no-editable" not in command
    assert '--expected-environment-root "${environment_root}"' in command
    assert '--expected-project-root "${project_root}"' in command
    assert '--nemo-rl-commit "${nemo_rl_commit}"' in command
    assert '--bridge-commit "${bridge_commit}"' in command
    assert '--mcore-commit "${mcore_commit}"' in command
    assert '--uv-lock-sha256 "${uv_lock_sha256}"' in command
    assert '--expected-te-commit "${expected_te_commit}"' in command
    assert '--container-device "${container_device}"' in command
    assert '--container-inode "${container_inode}"' in command
    assert '--container-size "${container_size}"' in command
    assert '--container-mtime-seconds "${container_mtime_seconds}"' in command
    assert '--container-ctime-seconds "${container_ctime_seconds}"' in command
    assert NEMORL_COMMIT in command
    assert BRIDGE_COMMIT in command
    assert MCORE_COMMIT in command
    assert TE_COMMIT in command
    assert "/tmp/nemo-rl-runtime-733" in command
    assert "/ambient/site-packages" not in command
    assert f"{REPO_ROOT}:{REPO_ROOT}:ro" in command
    assert (artifact_dir / "oci-container-runtime-733.diagnostics.log").is_file()
    assert provenance_log.read_text().splitlines() == [
        str(REPO_ROOT),
        NEMORL_COMMIT,
        str(REPO_ROOT / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge"),
        BRIDGE_COMMIT,
        str(
            REPO_ROOT
            / "3rdparty"
            / "Megatron-Bridge-workspace"
            / "Megatron-Bridge"
            / "3rdparty"
            / "Megatron-LM"
        ),
        MCORE_COMMIT,
    ]


def test_runtime_job_rejects_mutable_container_symlink(tmp_path: Path) -> None:
    immutable_container = tmp_path / "nightly_immutable.sqsh"
    immutable_container.write_bytes(b"container")
    mutable_link = tmp_path / "nightly.sqsh"
    mutable_link.symlink_to(immutable_container)
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "734",
            "CONTAINER_RUNTIME_VALIDATOR": str(
                (EXPERIMENT_DIR / "validate_container_runtime.py").resolve()
            ),
            "PROJECT_ROOT": str(REPO_ROOT),
            "CONTAINER": str(mutable_link),
            "CONTAINER_SHA256": hashlib.sha256(
                immutable_container.read_bytes()
            ).hexdigest(),
            "ARTIFACT_DIR": str(tmp_path / "artifacts"),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "immutable container path must not be a symlink" in result.stderr.lower()
    assert not (tmp_path / "artifacts").exists()


def test_stage_retry_completes_metadata_for_already_published_image(
    tmp_path: Path,
) -> None:
    container_dir = tmp_path / "containers"
    container_dir.mkdir()
    date_stamp = subprocess.run(
        ["date", "+%Y%m%d"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output = container_dir / f"nemo_rl_nightly_fixture_{date_stamp}_91.sqsh"
    output.write_bytes(b"already imported image")
    Path(f"{output}.staging.txt").write_text(
        "\n".join(
            (
                f"source_image=nvcr.io/nvidian/nemo-rl@{SOURCE_DIGEST}",
                f"source_digest={SOURCE_DIGEST}",
                f"source_commit={SOURCE_COMMIT}",
                "slurm_job_id=91",
                "",
            )
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    enroot_log = tmp_path / "enroot.log"
    _write_executable(
        fake_bin / "enroot",
        "#!/bin/bash\nprintf 'unexpected import\\n' >\"${ENROOT_LOG}\"\nexit 99\n",
    )

    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        **_stage_environment(container_dir, SLURM_JOB_ID="91"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        ENROOT_LOG=str(enroot_log),
    )

    assert result.returncode == 0, result.stderr
    assert not enroot_log.exists()
    metadata = Path(f"{output}.metadata.txt")
    assert metadata.is_file()
    assert (
        f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}"
        in metadata.read_text()
    )
    assert (container_dir / "nemo_rl_nightly_fixture.sqsh").resolve() == output


def test_stage_retry_discards_matching_orphan_metadata_then_reimports(
    tmp_path: Path,
) -> None:
    container_dir = tmp_path / "containers"
    container_dir.mkdir()
    date_stamp = subprocess.run(
        ["date", "+%Y%m%d"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output = container_dir / f"nemo_rl_nightly_fixture_{date_stamp}_92.sqsh"
    metadata = Path(f"{output}.metadata.txt")
    metadata.write_text(
        "\n".join(
            (
                f"source_image=nvcr.io/nvidian/nemo-rl@{SOURCE_DIGEST}",
                f"source_digest={SOURCE_DIGEST}",
                f"source_commit={SOURCE_COMMIT}",
                "slurm_job_id=92",
                "sha256=" + "d" * 64,
                "",
            )
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    enroot_log = tmp_path / "enroot.log"
    _write_executable(
        fake_bin / "enroot",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >"${ENROOT_LOG}"
output=
while (($#)); do
  if [[ "$1" == "-o" ]]; then
    shift
    output=$1
  fi
  shift
done
printf 'new immutable image\n' >"${output}"
""",
    )

    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        **_stage_environment(container_dir, SLURM_JOB_ID="92"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        ENROOT_LOG=str(enroot_log),
    )

    assert result.returncode == 0, result.stderr
    assert "import -o" in enroot_log.read_text()
    assert output.read_text() == "new immutable image\n"
    assert (
        f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}"
        in metadata.read_text()
    )
    assert (container_dir / "nemo_rl_nightly_fixture.sqsh").resolve() == output


def test_stage_retry_refuses_conflicting_complete_provenance(tmp_path: Path) -> None:
    container_dir = tmp_path / "containers"
    container_dir.mkdir()
    date_stamp = subprocess.run(
        ["date", "+%Y%m%d"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output = container_dir / f"nemo_rl_nightly_fixture_{date_stamp}_93.sqsh"
    output.write_bytes(b"other image")
    metadata = Path(f"{output}.metadata.txt")
    metadata.write_text(
        "\n".join(
            (
                f"source_image=nvcr.io/nvidian/nemo-rl@{SOURCE_DIGEST}",
                f"source_digest={SOURCE_DIGEST}",
                f"source_commit={'e' * 40}",
                "slurm_job_id=93",
                f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}",
                "",
            )
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "enroot", "#!/bin/bash\nexit 99\n")

    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        **_stage_environment(container_dir, SLURM_JOB_ID="93"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert result.returncode != 0
    assert "immutable provenance mismatch" in result.stderr.lower()
    assert output.read_bytes() == b"other image"
    assert not (container_dir / "nemo_rl_nightly_fixture.sqsh").exists()


def test_stage_retry_never_adopts_symlink_as_immutable_image(tmp_path: Path) -> None:
    container_dir = tmp_path / "containers"
    container_dir.mkdir()
    date_stamp = subprocess.run(
        ["date", "+%Y%m%d"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output = container_dir / f"nemo_rl_nightly_fixture_{date_stamp}_94.sqsh"
    target = tmp_path / "unmanaged.sqsh"
    target.write_bytes(b"unmanaged image")
    output.symlink_to(target)
    Path(f"{output}.staging.txt").write_text(
        "\n".join(
            (
                f"source_image=nvcr.io/nvidian/nemo-rl@{SOURCE_DIGEST}",
                f"source_digest={SOURCE_DIGEST}",
                f"source_commit={SOURCE_COMMIT}",
                "slurm_job_id=94",
                "",
            )
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "enroot", "#!/bin/bash\nexit 99\n")

    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        **_stage_environment(container_dir, SLURM_JOB_ID="94"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert result.returncode != 0
    assert "immutable artifact must not be a symlink" in result.stderr.lower()
    assert target.read_bytes() == b"unmanaged image"
    assert not Path(f"{output}.metadata.txt").exists()
