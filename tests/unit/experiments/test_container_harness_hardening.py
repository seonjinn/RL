from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
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
PYTHON_VERSION = "3.13.13"
UV_VERSION = "0.11.18"


@dataclass(frozen=True)
class RuntimePayloadFixture:
    source_project_root: Path
    source_validator: Path
    source_lock: Path
    environment_root: Path
    copied_project_root: Path
    fake_bin: Path
    cuda_home: Path


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


def _runtime_payload() -> str:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    start = source.index("runtime_command='") + len("runtime_command='")
    return source[start : source.index("'\n\nset +e", start)]


def _stage_runtime_payload_fixture(
    tmp_path: Path,
    *,
    verifier_body: str,
    lock_text: str = "fixture lock\n",
) -> RuntimePayloadFixture:
    source_project_root = tmp_path / "source-repo"
    source_validator = (
        source_project_root
        / "experiments"
        / "cuda_graph"
        / "nemotron_thd_te_graph_20260731"
        / "validate_container_runtime.py"
    )
    source_verifier = (
        source_validator.parent / "scripts" / "verify_source_provenance.sh"
    )
    source_verifier.parent.mkdir(parents=True)
    source_validator.write_text("raise SystemExit('fixture validator must not run')\n")
    _write_executable(source_verifier, verifier_body)
    (source_project_root / "docker").mkdir()
    (source_project_root / "docker" / "Dockerfile").write_text(
        f"ARG UV_VERSION={UV_VERSION}\n"
    )
    (source_project_root / ".python-version").write_text(f"{PYTHON_VERSION}\n")
    (source_project_root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    source_lock = source_project_root / "uv.lock"
    source_lock.write_text(lock_text)
    bridge_root = (
        source_project_root
        / "3rdparty"
        / "Megatron-Bridge-workspace"
        / "Megatron-Bridge"
    )
    mcore_root = bridge_root / "3rdparty" / "Megatron-LM"
    mcore_root.mkdir(parents=True)
    for repository in (source_project_root, bridge_root, mcore_root):
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=repository,
            check=True,
        )
    (source_project_root / ".source-manifest.env").write_text("fixture_manifest=true\n")
    outer_exclude = source_project_root / ".git" / "info" / "exclude"
    outer_exclude.write_text(outer_exclude.read_text() + "\n.source-manifest.env\n")
    environment_root = tmp_path / "runtime-environment"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sha256sum",
        '#!/bin/sh\nexec /usr/bin/shasum -a 256 "$@"\n',
    )
    cuda_home = tmp_path / "cuda"
    cuda_home.joinpath("bin").mkdir(parents=True)
    _write_executable(
        cuda_home / "bin" / "nvcc",
        "#!/bin/sh\nprintf 'Cuda compilation tools, fixture release 13.2\\n'\n",
    )
    return RuntimePayloadFixture(
        source_project_root=source_project_root,
        source_validator=source_validator,
        source_lock=source_lock,
        environment_root=environment_root,
        copied_project_root=Path(f"{environment_root}-source"),
        fake_bin=fake_bin,
        cuda_home=cuda_home,
    )


def _run_runtime_payload(
    fixture: RuntimePayloadFixture,
    *,
    environment: dict[str, str],
    uv_lock_sha256: str | None = None,
) -> subprocess.CompletedProcess[str]:
    runtime_environment = environment.copy()
    runtime_environment.setdefault("CUDA_HOME", str(fixture.cuda_home))
    runtime_environment.setdefault("CUDACXX", str(fixture.cuda_home / "bin" / "nvcc"))
    runtime_environment["PATH"] = (
        f"{fixture.cuda_home / 'bin'}:{runtime_environment.get('PATH', '')}"
    )
    expected_uv_lock_sha256 = (
        uv_lock_sha256 or hashlib.sha256(fixture.source_lock.read_bytes()).hexdigest()
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            _runtime_payload(),
            "bash",
            str(fixture.source_project_root),
            str(fixture.source_validator),
            str(fixture.environment_root),
            str(fixture.source_project_root.parent / "container.sqsh"),
            "a" * 64,
            NEMORL_COMMIT,
            BRIDGE_COMMIT,
            MCORE_COMMIT,
            expected_uv_lock_sha256,
            TE_COMMIT,
            "1",
            "2",
            "3",
            "4",
            "5",
            "--output",
            str(fixture.source_project_root.parent / "runtime.json"),
        ],
        env=runtime_environment,
        check=False,
        capture_output=True,
        text=True,
    )


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


def test_runtime_probe_requires_exact_uv_managed_python(tmp_path: Path) -> None:
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    project_root = tmp_path / "project"
    python_install_dir = tmp_path / "uv-python-installations"
    base_python = (
        python_install_dir / "cpython-3.13.13-linux-aarch64-gnu" / "bin" / "python3.13"
    )
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"managed-python-fixture")
    uv_executable = tmp_path / f"uv-{UV_VERSION}-733" / "uv"
    uv_executable.parent.mkdir(parents=True)
    _write_executable(
        uv_executable,
        f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n",
    )
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
        expected_python_version=PYTHON_VERSION,
        expected_python_install_dir=python_install_dir,
        expected_uv_version=UV_VERSION,
        expected_uv_executable=uv_executable,
        expected_nvte_with_nccl_ep="0",
        optional_importer=lambda name: SimpleNamespace(__name__=name),
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
        interpreter_path=environment_root / "bin" / "python",
        base_interpreter_path=base_python,
        runtime_prefix=environment_root,
        python_version=PYTHON_VERSION,
        environment={
            "UV_PROJECT_ENVIRONMENT": str(environment_root),
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
            "UV_MANAGED_PYTHON": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_executable),
            "NVTE_WITH_NCCL_EP": "0",
        },
    )

    assert result["python_version"] == PYTHON_VERSION
    assert result["python_base_executable"] == str(base_python)
    assert result["uv_python_install_dir"] == str(python_install_dir)
    assert result["uv_version"] == UV_VERSION
    assert result["uv_executable"] == str(uv_executable)
    assert result["nvte_with_nccl_ep"] == "0"
    assert result["transformer_engine_nccl_ep_available"] is False
    assert result["transformer_engine_nccl_ep_symbols"] == []
    assert (
        result["uv_executable_sha256"]
        == hashlib.sha256(uv_executable.read_bytes()).hexdigest()
    )
    assert (
        result["python_base_executable_sha256"]
        == hashlib.sha256(base_python.read_bytes()).hexdigest()
    )

    with pytest.raises(RuntimeError, match="NVTE_WITH_NCCL_EP mismatch"):
        module.probe_runtime(
            expected_device_count=4,
            expected_nvte_with_nccl_ep="0",
            environment={"NVTE_WITH_NCCL_EP": "1"},
        )

    with pytest.raises(RuntimeError, match="NCCL-EP module is available"):
        module.probe_runtime(
            expected_device_count=4,
            expected_nvte_with_nccl_ep="0",
            optional_importer=lambda name: SimpleNamespace(
                **{symbol: object() for symbol in module.NCCL_EP_EXTENSION_SYMBOLS}
            ),
            environment={"NVTE_WITH_NCCL_EP": "0"},
        )

    with pytest.raises(RuntimeError, match="Python version mismatch"):
        module.probe_runtime(
            expected_device_count=4,
            expected_environment_root=environment_root,
            expected_project_root=project_root,
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
            importer=lambda name: modules[name],
            version_getter=lambda distribution: f"fixture-{distribution}",
            interpreter_path=environment_root / "bin" / "python",
            base_interpreter_path=base_python,
            runtime_prefix=environment_root,
            python_version="3.13.11",
            environment={
                "UV_PROJECT_ENVIRONMENT": str(environment_root),
                "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
                "UV_MANAGED_PYTHON": "1",
                "UV_PYTHON_DOWNLOADS": "never",
                "PINNED_UV_VERSION": UV_VERSION,
                "UV_EXECUTABLE": str(uv_executable),
            },
        )


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
        "PATH=/root/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:"
        "/opt/nemo_rl_venv/bin" in command
    )
    assert "UV_CACHE_DIR=/tmp" not in command
    assert "CUDA_HOME=/usr/local/cuda" in command
    assert "CUDACXX=/usr/local/cuda/bin/nvcc" in command
    assert "NRL_FORCE_REBUILD_VENVS=true" in command
    assert "NVTE_WITH_NCCL_EP=0" in command
    assert "UV_PROJECT_ENVIRONMENT=/tmp/nemo-rl-runtime-733" in command
    expected_uv_executable = artifact_dir / f"uv-{UV_VERSION}-733" / "uv"
    assert f"PINNED_UV_VERSION={UV_VERSION}" in command
    assert f"UV_EXECUTABLE={expected_uv_executable}" in command
    assert f"PATH={expected_uv_executable.parent}:" not in command
    assert "https://astral.sh/uv/${expected_uv_version}/install.sh" in command
    assert 'UV_UNMANAGED_INSTALL="${staging_uv_dir}"' in command
    assert "command -v sha256sum" in command
    assert 'if [[ -e "${uv_bin_dir}" || -L "${uv_bin_dir}" ]]' in command
    assert 'if [[ ! -x "${uv_executable}" ]]' not in command
    assert "mv --no-clobber --no-target-directory" in command
    assert '[[ ! -e "${staging_uv_dir}" && ! -L "${staging_uv_dir}" ]]' in command
    assert '"${uv_executable}" --version' in command
    assert f"UV_PYTHON={PYTHON_VERSION}" in command
    assert "UV_MANAGED_PYTHON=1" in command
    assert f"UV_PYTHON_INSTALL_DIR={artifact_dir}/uv-python-installations" in command
    assert (
        '"${uv_executable}" python install --managed-python --no-bin '
        '"${expected_python_version}"' in command
    )
    assert "UV_PYTHON_DOWNLOADS=never" in command
    assert command.index(
        '"${uv_executable}" python install --managed-python --no-bin '
        '"${expected_python_version}"'
    ) < command.index("UV_PYTHON_DOWNLOADS=never")
    assert (
        '"${uv_executable}" run --python "${expected_python_version}" --managed-python '
        "--locked --extra mcore --no-python-downloads" in command
    )
    assert "--no-editable" not in command
    assert '--expected-environment-root "${environment_root}"' in command
    assert '--expected-project-root "${project_root}"' in command
    assert '--nemo-rl-commit "${nemo_rl_commit}"' in command
    assert '--bridge-commit "${bridge_commit}"' in command
    assert '--mcore-commit "${mcore_commit}"' in command
    assert '--uv-lock-sha256 "${uv_lock_sha256}"' in command
    assert '--expected-te-commit "${expected_te_commit}"' in command
    assert '--expected-python-version "${expected_python_version}"' in command
    assert '--expected-python-install-dir "${python_install_dir}"' in command
    assert '--expected-uv-version "${expected_uv_version}"' in command
    assert '--expected-uv-executable "${uv_executable}"' in command
    assert '--expected-nvte-with-nccl-ep "0"' in command
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


def test_runtime_payload_rejects_missing_nvcc_before_staging_uv(
    tmp_path: Path,
) -> None:
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body="#!/bin/sh\nexit 0\n",
    )
    uv_stage_marker = tmp_path / "uv-stage-started"
    _write_executable(
        fixture.fake_bin / "curl",
        '#!/bin/sh\nprintf started >"${UV_STAGE_MARKER}"\nexit 97\n',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "CUDA_HOME": str(tmp_path / "missing-cuda"),
            "CUDACXX": "",
            "UV_STAGE_MARKER": str(uv_stage_marker),
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(tmp_path / f"uv-{UV_VERSION}-733" / "uv"),
        }
    )

    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 2
    assert "nvcc" in result.stderr.lower()
    assert not uv_stage_marker.exists()
    assert not fixture.copied_project_root.exists()


def test_runtime_payload_rejects_preseeded_uv_without_executing_it(
    tmp_path: Path,
) -> None:
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body="#!/bin/sh\nexit 0\n",
    )
    uv_bin_dir = tmp_path / f"uv-{UV_VERSION}-733"
    uv_bin_dir.mkdir()
    uv_marker = tmp_path / "preseeded-uv-executed"
    _write_executable(
        uv_bin_dir / "uv",
        "#!/bin/sh\n"
        'printf executed >"${UV_MARKER}"\n'
        f"printf 'uv {UV_VERSION} (preseeded fixture)\\n'\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_bin_dir / "uv"),
            "UV_MARKER": str(uv_marker),
        }
    )
    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 2
    assert "Pinned uv destination already exists" in result.stderr
    assert not uv_marker.exists()


def test_runtime_payload_builds_from_writable_verified_source_copy(
    tmp_path: Path,
) -> None:
    provenance_log = tmp_path / "copied-source-provenance.log"
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body=(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            'printf \'%s\\n\' "$@" >"${COPY_PROVENANCE_LOG}"\n'
            'mkdir "$1/nemo_rl.egg-info"\n'
            "exit 91\n"
        ),
    )
    uv_bin_dir = tmp_path / f"uv-{UV_VERSION}-733"
    uv_bin_dir.mkdir()
    _write_executable(
        uv_bin_dir / "uv",
        f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "COPY_PROVENANCE_LOG": str(provenance_log),
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_bin_dir / "uv"),
        }
    )
    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 91, result.stderr
    assert provenance_log.read_text().splitlines() == [
        str(fixture.copied_project_root),
        NEMORL_COMMIT,
        str(
            fixture.copied_project_root
            / "3rdparty"
            / "Megatron-Bridge-workspace"
            / "Megatron-Bridge"
        ),
        BRIDGE_COMMIT,
        str(
            fixture.copied_project_root
            / "3rdparty"
            / "Megatron-Bridge-workspace"
            / "Megatron-Bridge"
            / "3rdparty"
            / "Megatron-LM"
        ),
        MCORE_COMMIT,
    ]
    assert not fixture.copied_project_root.exists()
    assert not (fixture.source_project_root / "nemo_rl.egg-info").exists()


def test_runtime_payload_rejects_mutated_copied_uv_lock(tmp_path: Path) -> None:
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body=(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "printf 'mutated copied lock\\n' >\"$1/uv.lock\"\n"
        ),
        lock_text="immutable fixture lock\n",
    )
    uv_bin_dir = tmp_path / f"uv-{UV_VERSION}-733"
    uv_bin_dir.mkdir()
    _write_executable(
        uv_bin_dir / "uv",
        f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_bin_dir / "uv"),
        }
    )
    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 2
    assert "Runtime source copy uv.lock SHA256 mismatch" in result.stderr
    assert fixture.source_lock.read_text() == "immutable fixture lock\n"
    assert not fixture.copied_project_root.exists()


def test_runtime_payload_rejects_ignored_source_artifact_before_verifier(
    tmp_path: Path,
) -> None:
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body="#!/bin/sh\nexit 92\n",
    )
    (fixture.source_project_root / ".gitignore").write_text("nemo_rl.egg-info\n")
    escaped_target = tmp_path / "escaped-target"
    escaped_target.mkdir()
    ignored_artifact = fixture.source_project_root / "nemo_rl.egg-info"
    ignored_artifact.symlink_to(escaped_target, target_is_directory=True)
    uv_bin_dir = tmp_path / f"uv-{UV_VERSION}-733"
    uv_bin_dir.mkdir()
    _write_executable(
        uv_bin_dir / "uv",
        f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_bin_dir / "uv"),
        }
    )

    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 2
    assert "unexpected ignored path: NeMo-RL:nemo_rl.egg-info" in result.stderr
    assert ignored_artifact.is_symlink()
    assert list(escaped_target.iterdir()) == []
    assert not fixture.copied_project_root.exists()


def test_runtime_payload_cleans_workspace_after_late_uv_failure(tmp_path: Path) -> None:
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body="#!/bin/sh\nexit 0\n",
    )
    uv_executable = tmp_path / f"uv-{UV_VERSION}-733" / "uv"
    fake_uv_template = tmp_path / "fake-uv"
    python_install_dir = tmp_path / "uv-python-installations"
    base_python = (
        python_install_dir
        / f"cpython-{PYTHON_VERSION}-linux-aarch64-gnu"
        / "bin"
        / f"python{'.'.join(PYTHON_VERSION.split('.')[:2])}"
    )
    _write_executable(
        fake_uv_template,
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'case "${1:-}" in\n'
        "  --version)\n"
        f"    printf 'uv {UV_VERSION} (fixture)\\n'\n"
        "    ;;\n"
        "  python)\n"
        '    case "${2:-}" in\n'
        "      install)\n"
        f"        base_python={base_python}\n"
        '        mkdir -p "$(dirname "${base_python}")"\n'
        f"        printf '%s\\n' '#!/bin/sh' 'printf \"{PYTHON_VERSION}\\\\n\"' >\"${{base_python}}\"\n"
        '        chmod 755 "${base_python}"\n'
        "        ;;\n"
        "      find)\n"
        f"        printf '%s\\n' {base_python}\n"
        "        ;;\n"
        "      *) exit 94 ;;\n"
        "    esac\n"
        "    ;;\n"
        "  run)\n"
        '    mkdir -p "${UV_PROJECT_ENVIRONMENT}"\n'
        "    exit 93\n"
        "    ;;\n"
        "  *) exit 94 ;;\n"
        "esac\n",
    )
    _write_executable(
        fixture.fake_bin / "curl",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "output=\n"
        "while (($#)); do\n"
        "  if [[ \"$1\" == '-o' ]]; then\n"
        "    shift\n"
        "    output=$1\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "printf '%s\\n' '#!/bin/sh' 'set -eu' "
        "'mkdir -p \"$UV_UNMANAGED_INSTALL\"' "
        '\'cp "$FAKE_UV_TEMPLATE" "$UV_UNMANAGED_INSTALL/uv"\' '
        '\'chmod 755 "$UV_UNMANAGED_INSTALL/uv"\' >"${output}"\n'
        'chmod 755 "${output}"\n',
    )
    _write_executable(
        fixture.fake_bin / "mv",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "operands=()\n"
        'for argument in "$@"; do\n'
        '  case "${argument}" in\n'
        "    --no-clobber|--no-target-directory|--) ;;\n"
        '    *) operands+=("${argument}") ;;\n'
        "  esac\n"
        "done\n"
        "[[ ${#operands[@]} -eq 2 ]] || exit 95\n"
        '[[ ! -e "${operands[1]}" ]] || exit 96\n'
        'exec /bin/mv "${operands[0]}" "${operands[1]}"\n',
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "FAKE_UV_TEMPLATE": str(fake_uv_template),
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_executable),
            "UV_MANAGED_PYTHON": "1",
            "UV_PYTHON": PYTHON_VERSION,
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
            "UV_PROJECT_ENVIRONMENT": str(fixture.environment_root),
        }
    )
    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 93, result.stderr
    assert not fixture.environment_root.exists()
    assert not fixture.copied_project_root.exists()


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
