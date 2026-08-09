from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import re
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
PYTHON_VERSION = (REPO_ROOT / ".python-version").read_text().strip()
UV_VERSION_MATCH = re.search(
    r"^ARG UV_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$",
    (REPO_ROOT / "docker" / "Dockerfile").read_text(),
    re.MULTILINE,
)
assert UV_VERSION_MATCH is not None
UV_VERSION = UV_VERSION_MATCH.group(1)
RUNTIME_STAGE_CAPABILITY = "mcore-test-v1"
RUNTIME_TEST_REQUIREMENTS = (
    "pytest==9.1.1,iniconfig==2.3.0,packaging==26.2,pluggy==1.6.0,pygments==2.20.0"
)


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


def _load_runtime_stage_readonly_helper() -> ModuleType:
    path = EXPERIMENT_DIR / "make_runtime_stage_readonly.py"
    assert path.is_file(), "The runtime stage needs a retrying read-only helper"
    spec = importlib.util.spec_from_file_location("runtime_stage_readonly", path)
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
    modules["megatron.core.extensions.transformer_engine"] = SimpleNamespace(
        __file__=str(
            site_packages / "megatron" / "core" / "extensions" / "transformer_engine.py"
        ),
        TEColumnParallelGroupedLinear=object,
        TERowParallelGroupedLinear=object,
    )
    return modules


def test_runtime_probe_artifact_records_attestation_producer_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runtime_probe()
    output = tmp_path / "runtime.json"
    arguments = SimpleNamespace(
        runtime_attestation_job_id=734,
        container_image="/lustre/runtime.sqsh",
        container_sha256="a" * 64,
        nemo_rl_commit=NEMORL_COMMIT,
        bridge_commit=BRIDGE_COMMIT,
        mcore_commit=MCORE_COMMIT,
        uv_lock_sha256="f" * 64,
        expected_te_commit=TE_COMMIT,
        expected_te_version_base_commit=TE_COMMIT,
        expected_python_version=PYTHON_VERSION,
        expected_uv_version=UV_VERSION,
        expected_nvte_with_nccl_ep="0",
        runtime_feature_set="dropless_hybridep_nano16",
        excluded_packages="fast-hadamard-transform",
        torch_cuda_arch_list="10.0a",
        nvte_cuda_archs="100a",
        container_device=1,
        container_inode=2,
        container_size=3,
        container_mtime_seconds=4,
        container_ctime_seconds=5,
        expected_device_count=4,
        expected_environment_root=tmp_path / "environment",
        expected_project_root=tmp_path / "source",
        expected_python_install_dir=tmp_path / "python",
        expected_uv_executable=tmp_path / "uv",
        output=output,
    )
    monkeypatch.setattr(module, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        module,
        "probe_runtime",
        lambda **_: {
            "packages": {
                "transformer_engine.pytorch": {"version": "2.19.0.dev0+eeeeeeee"}
            }
        },
    )
    monkeypatch.setattr(module, "_distribution_vcs_commit", lambda _: TE_COMMIT)
    monkeypatch.setattr(
        module,
        "validate_transformer_engine_identities",
        lambda **_: {
            "transformer_engine_source_commit": TE_COMMIT,
            "transformer_engine_version_base_commit": TE_COMMIT,
        },
    )

    module.main()

    payload = json.loads(output.read_text())
    assert payload["status"] == "passed"
    assert payload["runtime_attestation_job_id"] == 734
    assert payload["deep_ep_vcs_commit"] == TE_COMMIT


def _run_script(
    relative_path: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    env.pop("RUNTIME_STAGE_CAPABILITY", None)
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
    start = source.index("stage_command='") + len("stage_command='")
    return source[start : source.index("'\n\nattestation_command=", start)]


def _attestation_payload() -> str:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    start = source.index("attestation_command='") + len("attestation_command='")
    return source[start : source.index("'\n\npayload=", start)]


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
    source_readonly_helper = source_validator.parent / "make_runtime_stage_readonly.py"
    source_verifier.parent.mkdir(parents=True)
    source_validator.write_text("raise SystemExit('fixture validator must not run')\n")
    source_readonly_helper.write_text(
        (EXPERIMENT_DIR / "make_runtime_stage_readonly.py").read_text()
    )
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
    runtime_stage_root = tmp_path / "runtime-stage"
    environment_root = runtime_stage_root / "environment"
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
        copied_project_root=runtime_stage_root / "source",
        fake_bin=fake_bin,
        cuda_home=cuda_home,
    )


def _run_runtime_payload(
    fixture: RuntimePayloadFixture,
    *,
    environment: dict[str, str],
    uv_lock_sha256: str | None = None,
    cuda_home: Path | None = None,
    cuda_compiler: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    runtime_environment = environment.copy()
    effective_cuda_home = cuda_home or fixture.cuda_home
    effective_cuda_compiler = (
        cuda_compiler
        if cuda_compiler is not None
        else effective_cuda_home / "bin" / "nvcc"
    )
    runtime_environment["CUDA_HOME"] = str(effective_cuda_home)
    runtime_environment["CUDACXX"] = str(effective_cuda_compiler)
    runtime_stage_root = fixture.environment_root.parent
    runtime_environment.update(
        {
            "UV_CACHE_DIR": str(runtime_stage_root / "build-cache"),
            "NVTE_CMAKE_BUILD_DIR": str(runtime_stage_root / "te-cmake"),
            "RUNTIME_STAGE_ROOT": str(runtime_stage_root),
            "ARTIFACT_DIR": str(runtime_stage_root.parent),
            "RUNTIME_STAGE_MARKER": str(
                runtime_stage_root.parent / "stage-markers" / "fixture.env"
            ),
            "RUNTIME_STAGE_MARKER_SHA256": "f" * 64,
            "SLURM_JOB_ID": "733",
            "RUNTIME_STAGE_JOB_ID": "733",
            "RUNTIME_STAGE_CPUS_PER_TASK": "32",
            "CMAKE_BUILD_PARALLEL_LEVEL": "32",
            "RUNTIME_BOOTSTRAP_PYTHON": sys.executable,
            "NVTE_CUDA_ARCHS": "100a",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
            "RUNTIME_TEST_REQUIREMENTS": RUNTIME_TEST_REQUIREMENTS,
        }
    )
    runtime_environment.setdefault("RUNTIME_FEATURE_SET", "te_eval_capability_8")
    runtime_environment.setdefault(
        "RUNTIME_EXCLUDED_PACKAGES",
        "causal-conv1d,deep-ep,fast-hadamard-transform,mamba-ssm",
    )
    runtime_environment["PATH"] = (
        f"{effective_cuda_home / 'bin'}:{runtime_environment.get('PATH', '')}"
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
            TE_COMMIT,
            "--output",
            str(fixture.source_project_root.parent / "runtime.json"),
            str(runtime_stage_root),
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
    modules["megatron.core.extensions.transformer_engine"] = SimpleNamespace(
        __file__=str(
            project_root / "megatron" / "core" / "extensions" / "transformer_engine.py"
        ),
        TEColumnParallelGroupedLinear=object,
        TERowParallelGroupedLinear=object,
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


@pytest.mark.parametrize(
    "feature_set", ("te_eval_capability_8", "bridge_forward_only_eval_8")
)
def test_runtime_probe_binds_narrow_te_eval_feature_set(
    tmp_path: Path, feature_set: str
) -> None:
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    project_root = tmp_path / "project"
    modules = _runtime_modules(module, environment_root)
    del modules["mamba_ssm"]
    del modules["causal_conv1d"]
    exclusions = (
        "causal-conv1d",
        "deep-ep",
        "fast-hadamard-transform",
        "mamba-ssm",
    )
    environment = {
        "UV_PROJECT_ENVIRONMENT": str(environment_root),
        "RUNTIME_FEATURE_SET": feature_set,
        "RUNTIME_EXCLUDED_PACKAGES": ",".join(exclusions),
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "NVTE_CUDA_ARCHS": "100a",
    }

    result = module.probe_runtime(
        expected_device_count=4,
        expected_environment_root=environment_root,
        expected_project_root=project_root,
        expected_runtime_feature_set=feature_set,
        expected_excluded_packages=exclusions,
        expected_torch_cuda_arch_list="10.0a",
        expected_nvte_cuda_archs="100a",
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
        interpreter_path=environment_root / "bin" / "python",
        runtime_prefix=environment_root,
        environment=environment,
    )

    assert result["runtime_feature_set"] == feature_set
    assert result["excluded_packages"] == list(exclusions)
    assert "mamba_ssm" not in result["packages"]
    assert "causal_conv1d" not in result["packages"]
    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST mismatch"):
        module.probe_runtime(
            expected_device_count=4,
            expected_runtime_feature_set=feature_set,
            expected_excluded_packages=exclusions,
            expected_torch_cuda_arch_list="10.0a",
            expected_nvte_cuda_archs="100a",
            environment={**environment, "TORCH_CUDA_ARCH_LIST": "10.0"},
        )


@pytest.mark.parametrize(
    "feature_set",
    ("dropless_hybridep_nano16", "dropless_hybridep_qwen235_64"),
)
def test_runtime_probe_binds_dropless_hybridep_feature_set(
    tmp_path: Path, feature_set: str
) -> None:
    """Every HybridEP gate must import DeepEP instead of trusting metadata."""
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    project_root = tmp_path / "project"
    modules = _runtime_modules(module, environment_root)
    modules["deep_ep"] = SimpleNamespace(
        __file__=str(
            environment_root
            / "lib"
            / "python3.13"
            / "site-packages"
            / "deep_ep"
            / "__init__.py"
        ),
        HybridEPBuffer=object,
    )
    exclusions = ("fast-hadamard-transform",)
    environment = {
        "UV_PROJECT_ENVIRONMENT": str(environment_root),
        "RUNTIME_FEATURE_SET": feature_set,
        "RUNTIME_EXCLUDED_PACKAGES": ",".join(exclusions),
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "NVTE_CUDA_ARCHS": "100a",
    }

    result = module.probe_runtime(
        expected_device_count=4,
        expected_environment_root=environment_root,
        expected_project_root=project_root,
        expected_runtime_feature_set=feature_set,
        expected_excluded_packages=exclusions,
        expected_torch_cuda_arch_list="10.0a",
        expected_nvte_cuda_archs="100a",
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
        interpreter_path=environment_root / "bin" / "python",
        runtime_prefix=environment_root,
        environment=environment,
    )

    assert result["runtime_feature_set"] == feature_set
    assert result["excluded_packages"] == list(exclusions)
    assert result["hybridep_buffer_available"] is True
    assert result["packages"]["deep_ep"]["distribution"] == "deep-ep"
    assert "mamba_ssm" in result["packages"]
    assert "causal_conv1d" in result["packages"]


@pytest.mark.parametrize(
    "feature_set",
    ("dropless_alltoall_qwen30_16", "dropless_alltoall_super32"),
)
def test_runtime_probe_binds_dropless_alltoall_feature_set(
    tmp_path: Path, feature_set: str
) -> None:
    """AlltoAll rows use the full locked MoE stack without requiring DeepEP."""
    module = _load_runtime_probe()
    environment_root = tmp_path / "runtime-venv"
    project_root = tmp_path / "project"
    modules = _runtime_modules(module, environment_root)
    exclusions = ("deep-ep", "fast-hadamard-transform")
    environment = {
        "UV_PROJECT_ENVIRONMENT": str(environment_root),
        "RUNTIME_FEATURE_SET": feature_set,
        "RUNTIME_EXCLUDED_PACKAGES": ",".join(exclusions),
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "NVTE_CUDA_ARCHS": "100a",
    }

    result = module.probe_runtime(
        expected_device_count=4,
        expected_environment_root=environment_root,
        expected_project_root=project_root,
        expected_runtime_feature_set=feature_set,
        expected_excluded_packages=exclusions,
        expected_torch_cuda_arch_list="10.0a",
        expected_nvte_cuda_archs="100a",
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
        interpreter_path=environment_root / "bin" / "python",
        runtime_prefix=environment_root,
        environment=environment,
    )

    assert result["runtime_feature_set"] == feature_set
    assert result["hybridep_buffer_available"] is None
    assert "deep_ep" not in result["packages"]


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


def test_runtime_probe_binds_te_package_version_to_source_commit() -> None:
    module = _load_runtime_probe()
    source_commit = "04a76c84423d9a4eb2f2010ef6692e347326cc00"

    assert module.validate_transformer_engine_identities(
        version="2.19.0.dev0+04a76c84",
        source_commit=source_commit,
        expected_source_commit=source_commit,
        expected_version_base_commit=source_commit,
    ) == {
        "transformer_engine_source_commit": source_commit,
        "transformer_engine_version_base_commit": source_commit,
    }
    with pytest.raises(RuntimeError, match="source commit mismatch"):
        module.validate_transformer_engine_identities(
            version="2.19.0.dev0+04a76c84",
            source_commit="bffde8f4a0a4eea9036dc753e28269247e5de69d",
            expected_source_commit=source_commit,
            expected_version_base_commit=source_commit,
        )
    with pytest.raises(RuntimeError, match="version-base commit mismatch"):
        module.validate_transformer_engine_identities(
            version="2.19.0.dev0+bffde8f4",
            source_commit=source_commit,
            expected_source_commit=source_commit,
            expected_version_base_commit=source_commit,
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
                "EXPECTED_TE_VERSION_BASE_SHA": TE_COMMIT,
                "RUNTIME_PHASE": "stage",
                "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
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
        '#!/bin/bash\nprintf \'%s\\n\' "$*" >"${FAKE_SBATCH_LOG}"\n',
    )

    result = _run_script(
        relative_path,
        **environment,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        FAKE_SBATCH_LOG=str(sbatch_log),
        SBATCH_TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    submitted_arguments = sbatch_log.read_text()
    assert "--test-only" in submitted_arguments
    assert "--parsable" not in submitted_arguments
    assert "--export=ALL" not in submitted_arguments
    if relative_path == "scripts/validate_oci_container_runtime.sub":
        assert "--time=00:45:00" in submitted_arguments
        assert (
            "--job-name=coreai_dlalgo_nemorl-cuda-graph.runtime-stage"
            in submitted_arguments
        )


def test_runtime_submitter_scrubs_reserved_sbatch_environment(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
printf 'ENV_SBATCH_GRES=%s\n' "${SBATCH_GRES-unset}" >"${FAKE_SBATCH_LOG}"
printf 'ENV_SBATCH_GPUS_PER_NODE=%s\n' "${SBATCH_GPUS_PER_NODE-unset}" >>"${FAKE_SBATCH_LOG}"
printf 'ENV_SBATCH_TEST_ONLY=%s\n' "${SBATCH_TEST_ONLY-unset}" >>"${FAKE_SBATCH_LOG}"
printf 'ENV_SBATCH_EXCLUSIVE=%s\n' "${SBATCH_EXCLUSIVE-unset}" >>"${FAKE_SBATCH_LOG}"
printf 'ENV_SBATCH_MEM=%s\n' "${SBATCH_MEM-unset}" >>"${FAKE_SBATCH_LOG}"
printf 'ARG=%s\n' "$@" >>"${FAKE_SBATCH_LOG}"
""",
    )

    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        CONTAINER="/lustre/example/nightly.sqsh",
        CONTAINER_SHA256="c" * 64,
        ARTIFACT_DIR="/lustre/example/runtime-artifacts",
        EXPECTED_NEMORL_SHA=NEMORL_COMMIT,
        EXPECTED_BRIDGE_SHA=BRIDGE_COMMIT,
        EXPECTED_MCORE_SHA=MCORE_COMMIT,
        EXPECTED_TE_SHA=TE_COMMIT,
        EXPECTED_TE_VERSION_BASE_SHA=TE_COMMIT,
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CAPABILITY=RUNTIME_STAGE_CAPABILITY,
        SOURCE_PROVENANCE_VERIFIER=str(
            EXPERIMENT_DIR / "scripts" / "verify_source_provenance.sh"
        ),
        STAGE_PARTITION="batch",
        SBATCH_GPUS_PER_NODE="4",
        SBATCH_GRES="none",
        SBATCH_TEST_ONLY="1",
        SBATCH_EXCLUSIVE="1",
        SBATCH_MEM="0",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        FAKE_SBATCH_LOG=str(sbatch_log),
    )

    assert result.returncode == 0, result.stderr
    submitted = sbatch_log.read_text()
    assert "ENV_SBATCH_GRES=unset" in submitted
    assert "ENV_SBATCH_GPUS_PER_NODE=unset" in submitted
    assert "ENV_SBATCH_TEST_ONLY=unset" in submitted
    assert "ENV_SBATCH_EXCLUSIVE=unset" in submitted
    assert "ENV_SBATCH_MEM=unset" in submitted
    assert "SBATCH_GPUS_PER_NODE=4" in submitted
    assert "SBATCH_GRES=none" in submitted
    assert "ARG=--gpus" not in submitted
    assert "ARG=--gres" not in submitted


def test_runtime_job_uses_worker_parity_uv_environment_and_exact_provenance(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    assert "#SBATCH --time=04:00:00" in source_wrapper.read_text()
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
    runtime_stage_root = artifact_dir / "staged-runtimes" / ("a" * 64)
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
            "EXPECTED_TE_VERSION_BASE_SHA": TE_COMMIT,
            "SOURCE_PROVENANCE_VERIFIER": str(provenance_verifier),
            "PROVENANCE_LOG": str(provenance_log),
            "RUNTIME_PHASE": "stage",
            "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
            "RUNTIME_TEST_REQUIREMENTS": RUNTIME_TEST_REQUIREMENTS,
            "RUNTIME_STAGE_CPUS_PER_TASK": "32",
            "SLURM_CPUS_PER_TASK": "32",
            "RUNTIME_STAGE_ROOT": str(runtime_stage_root),
            "RUNTIME_STAGE_MARKER": str(
                artifact_dir / "stage-markers" / f"{runtime_stage_root.name}.env"
            ),
            "RUNTIME_STAGE_MARKER_SHA256": "b" * 64,
            "RUNTIME_FEATURE_SET": "te_eval_capability_8",
            "RUNTIME_EXCLUDED_PACKAGES": (
                "causal-conv1d,deep-ep,fast-hadamard-transform,mamba-ssm"
            ),
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "NVTE_CUDA_ARCHS": "100a",
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
    assert "--no-container-mount-home" in command
    assert "env -i" in command
    assert "HOME=/root" in command
    assert (
        "PATH=/root/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:"
        "/opt/nemo_rl_venv/bin" in command
    )
    assert f"UV_CACHE_DIR={runtime_stage_root}/build-cache" in command
    assert f"NVTE_CMAKE_BUILD_DIR={runtime_stage_root}/te-cmake" in command
    assert "CMAKE_BUILD_PARALLEL_LEVEL=32" in command
    assert "--cpus-per-task=32" in command
    assert "/root/.cache/uv" not in command
    assert "CUDA_HOME=/usr/local/cuda" in command
    assert "CUDACXX=/usr/local/cuda/bin/nvcc" in command
    assert "NRL_FORCE_REBUILD_VENVS=true" in command
    assert "NVTE_WITH_NCCL_EP=0" in command
    assert f"UV_PROJECT_ENVIRONMENT={runtime_stage_root}/environment" in command
    expected_marker = artifact_dir / "stage-markers" / f"{runtime_stage_root.name}.env"
    assert command.count(f"RUNTIME_STAGE_MARKER={expected_marker}") == 1
    assert command.count("RUNTIME_STAGE_JOB_ID=733") == 1
    assert "RUNTIME_STAGE_MARKER= " not in command
    assert "RUNTIME_STAGE_JOB_ID= " not in command
    expected_uv_executable = runtime_stage_root / "uv" / "uv"
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
        '"${uv_executable}" sync --python "${expected_python_version}" --managed-python'
        in command
    )
    assert "--locked --extra mcore --group test --no-python-downloads" in command
    assert 'sync_command+=(--no-install-package "${excluded_package}")' in command
    assert (
        "RUNTIME_EXCLUDED_PACKAGES=causal-conv1d,deep-ep,"
        "fast-hadamard-transform,mamba-ssm" in command
    )
    assert "--no-editable" not in command
    assert '"schema=runtime-stage-v1"' in command
    assert (
        'mv --no-clobber --no-target-directory -- "${partial_marker}" "${marker}"'
        in command
    )
    assert '"${bootstrap_python}" "${source_readonly_helper}"' in command
    assert '--regular-file "${partial_marker}"' in command
    assert '"stage_cpus_per_task=${RUNTIME_STAGE_CPUS_PER_TASK}"' in command
    assert NEMORL_COMMIT in command
    assert BRIDGE_COMMIT in command
    assert MCORE_COMMIT in command
    assert TE_COMMIT in command
    assert str(runtime_stage_root) in command
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


def test_runtime_attestation_omits_gpu_tres_but_probes_four_devices(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    spool_dir = tmp_path / "slurm-spool" / "job734"
    spool_dir.mkdir(parents=True)
    spooled_wrapper = spool_dir / "slurm_script"
    spooled_wrapper.write_text(source_wrapper.read_text())
    spooled_wrapper.chmod(0o755)

    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"container")
    container_digest = hashlib.sha256(container.read_bytes()).hexdigest()
    artifact_dir = tmp_path / "artifacts"
    runtime_stage_root = artifact_dir / "staged-runtimes" / ("a" * 64)
    runtime_stage_marker = (
        artifact_dir / "stage-markers" / f"{runtime_stage_root.name}.env"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.log"
    provenance_verifier = tmp_path / "verify_source_provenance.sh"
    _write_executable(provenance_verifier, "#!/bin/sh\nexit 0\n")
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
: "${output:?}"
printf '{"status":"passed"}\n' >"${output}"
""",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SRUN_LOG": str(srun_log),
            "SLURM_JOB_ID": "734",
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
            "EXPECTED_TE_VERSION_BASE_SHA": TE_COMMIT,
            "SOURCE_PROVENANCE_VERIFIER": str(provenance_verifier),
            "RUNTIME_PHASE": "attest",
            "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
            "RUNTIME_TEST_REQUIREMENTS": RUNTIME_TEST_REQUIREMENTS,
            "RUNTIME_STAGE_ROOT": str(runtime_stage_root),
            "RUNTIME_STAGE_MARKER": str(runtime_stage_marker),
            "RUNTIME_STAGE_MARKER_SHA256": "b" * 64,
            "RUNTIME_STAGE_JOB_ID": "733",
            "RUNTIME_FEATURE_SET": "dropless_hybridep_nano16",
            "RUNTIME_EXCLUDED_PACKAGES": "fast-hadamard-transform",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "NVTE_CUDA_ARCHS": "100a",
            "SBATCH_GPUS_PER_NODE": "4",
            "SBATCH_GRES": "none",
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
    assert "--gpus-per-node" not in command
    assert "--gres=" not in command
    assert "--expected-device-count 4" in command
    assert "RUNTIME_ATTESTATION_JOB_ID=734" in command
    assert "UV_PYTHON_DOWNLOADS=never" in command
    assert '--runtime-attestation-job-id "${RUNTIME_ATTESTATION_JOB_ID}"' in command
    assert (artifact_dir / "oci-container-runtime-734.json").is_file()


def test_runtime_stage_readonly_retries_transient_lustre_eio(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    partial_marker = tmp_path / "partial-marker.env"
    partial_marker.write_text("marker\n")
    real_chmod = module.os.chmod
    chmod_attempts = 0

    def flaky_chmod(
        path: os.PathLike[str] | str,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal chmod_attempts
        if Path(path) == payload:
            chmod_attempts += 1
            if chmod_attempts == 1:
                raise OSError(errno.EIO, "transient Lustre metadata failure")
        real_chmod(path, mode, follow_symlinks=follow_symlinks)

    try:
        module.make_tree_readonly(
            stage_root,
            regular_files=(partial_marker,),
            chmod_fn=flaky_chmod,
            sleep_fn=lambda _: None,
        )

        assert chmod_attempts == 2
        assert payload.stat().st_mode & 0o222 == 0
        assert stage_root.stat().st_mode & 0o222 == 0
        assert partial_marker.stat().st_mode & 0o222 == 0
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)
        real_chmod(partial_marker, 0o600)


def test_runtime_stage_readonly_does_not_require_path_chmod_nofollow_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    real_chmod = module.os.chmod

    def unsupported_path_chmod(
        path: os.PathLike[str] | str,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        del path, mode, follow_symlinks
        raise NotImplementedError("path chmod no-follow is unavailable")

    monkeypatch.setattr(module.os, "chmod", unsupported_path_chmod)
    try:
        module.make_tree_readonly(stage_root)

        assert payload.stat().st_mode & 0o222 == 0
        assert stage_root.stat().st_mode & 0o222 == 0
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)


def test_runtime_stage_readonly_retries_default_fchmod_eio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    real_chmod = module.os.chmod
    real_fchmod = module.os.fchmod
    fchmod_calls = 0

    def flaky_fchmod(file_descriptor: int, mode: int) -> None:
        nonlocal fchmod_calls
        fchmod_calls += 1
        if fchmod_calls == 1:
            raise OSError(errno.EIO, "transient Lustre fchmod failure")
        real_fchmod(file_descriptor, mode)

    monkeypatch.setattr(module.os, "fchmod", flaky_fchmod)
    try:
        module.make_tree_readonly(stage_root, sleep_fn=lambda _: None)

        assert fchmod_calls >= 3
        assert payload.stat().st_mode & 0o222 == 0
        assert stage_root.stat().st_mode & 0o222 == 0
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)


def test_runtime_stage_readonly_retries_partial_verification_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    real_chmod = module.os.chmod
    real_walk = module.os.walk
    walk_calls = 0

    def flaky_walk(*args: object, **kwargs: object) -> object:
        nonlocal walk_calls
        walk_calls += 1
        if walk_calls != 2:
            return real_walk(*args, **kwargs)

        def partial_walk() -> object:
            yield str(stage_root), [], [payload.name]
            raise OSError(errno.EIO, "partial Lustre traversal", str(stage_root))

        return partial_walk()

    monkeypatch.setattr(module.os, "walk", flaky_walk)
    try:
        module.make_tree_readonly(stage_root, sleep_fn=lambda _: None)

        assert walk_calls == 4
        assert payload.stat().st_mode & 0o222 == 0
        assert stage_root.stat().st_mode & 0o222 == 0
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)


def test_runtime_stage_readonly_rejects_escaped_symlink_without_mutating_target(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    external_target = tmp_path / "external-target"
    external_target.write_text("external\n")
    stage_root.joinpath("escaped-link").symlink_to(external_target)
    original_mode = external_target.stat().st_mode

    with pytest.raises(ValueError, match="escapes trusted roots"):
        module.make_tree_readonly(stage_root)

    assert external_target.stat().st_mode == original_mode


def test_runtime_stage_readonly_allows_symlink_into_explicit_trusted_root(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    trusted_root = tmp_path / "python-installations"
    trusted_root.mkdir()
    trusted_target = trusted_root / "python"
    trusted_target.write_text("python\n")
    stage_root.joinpath("python-link").symlink_to(trusted_target)
    original_mode = trusted_target.stat().st_mode
    real_chmod = module.os.chmod
    try:
        module.make_tree_readonly(
            stage_root,
            trusted_symlink_roots=(trusted_root,),
        )

        assert trusted_target.stat().st_mode == original_mode
        assert stage_root.stat().st_mode & 0o222 == 0
    finally:
        real_chmod(stage_root, 0o700)


def test_runtime_stage_readonly_verify_only_rejects_writable_state(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    original_mode = payload.stat().st_mode

    with pytest.raises(RuntimeError, match="writable regular state"):
        module.verify_tree_readonly(stage_root)

    assert payload.stat().st_mode == original_mode


def test_runtime_stage_readonly_fails_after_bounded_persistent_eio(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    real_chmod = module.os.chmod
    chmod_attempts = 0

    def failing_chmod(
        path: os.PathLike[str] | str,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal chmod_attempts
        if Path(path) == payload:
            chmod_attempts += 1
            raise OSError(errno.EIO, "persistent Lustre metadata failure")
        real_chmod(path, mode, follow_symlinks=follow_symlinks)

    try:
        with pytest.raises(RuntimeError, match="exhausted 3 attempts"):
            module.make_tree_readonly(
                stage_root,
                max_attempts=3,
                chmod_fn=failing_chmod,
                sleep_fn=lambda _: None,
            )

        assert chmod_attempts == 3
        assert payload.stat().st_mode & 0o222 != 0
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)


def test_runtime_stage_readonly_retries_incomplete_lustre_scan(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    real_chmod = module.os.chmod
    real_lstat = module.os.lstat
    payload_scans = 0

    def flaky_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        nonlocal payload_scans
        if Path(path) == payload:
            payload_scans += 1
            if payload_scans == 1:
                raise OSError(errno.ESTALE, "transient Lustre scan failure")
        return real_lstat(path)

    try:
        module.make_tree_readonly(
            stage_root,
            lstat_fn=flaky_lstat,
            sleep_fn=lambda _: None,
        )

        assert payload_scans >= 2
        assert payload.stat().st_mode & 0o222 == 0
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)


def test_runtime_stage_readonly_does_not_retry_permission_error(
    tmp_path: Path,
) -> None:
    module = _load_runtime_stage_readonly_helper()
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    payload = stage_root / "payload.bin"
    payload.write_bytes(b"payload")
    real_chmod = module.os.chmod
    sleeps: list[float] = []

    def denied_chmod(
        path: os.PathLike[str] | str,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(path) == payload:
            raise PermissionError(errno.EPERM, "permission denied")
        real_chmod(path, mode, follow_symlinks=follow_symlinks)

    try:
        with pytest.raises(PermissionError, match="permission denied"):
            module.make_tree_readonly(
                stage_root,
                chmod_fn=denied_chmod,
                sleep_fn=sleeps.append,
            )

        assert sleeps == []
    finally:
        real_chmod(stage_root, 0o700)
        real_chmod(payload, 0o600)


def test_runtime_stage_readonly_cli_reports_operational_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_runtime_stage_readonly_helper()

    result = module.main(["--root", str(tmp_path / "missing-stage")])

    assert result == 2
    assert "Runtime stage read-only finalization failed" in capsys.readouterr().err


def test_runtime_stage_finalization_uses_retrying_readonly_helper() -> None:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    stage = source.split("stage_command='", 1)[1].split("'\n\nattestation_command=", 1)[
        0
    ]
    attestation = source.split("attestation_command='", 1)[1].split("'\n\npayload=", 1)[
        0
    ]

    assert "make_runtime_stage_readonly.py" in source
    assert '"${bootstrap_python}" "${source_readonly_helper}"' in source
    assert '--regular-file "${partial_marker}"' in source
    assert '--trusted-symlink-root "${runtime_stage_root}"' in stage
    assert '--trusted-symlink-root "${python_install_dir}"' in stage
    assert "--verify-only" in attestation
    assert '--regular-file "${marker}"' in attestation
    assert '--trusted-symlink-root "${runtime_stage_root}"' in attestation
    assert '--trusted-symlink-root "${python_install_dir}"' in attestation
    assert '"${runtime_python}" "${readonly_helper}"' not in attestation
    assert 'chmod a-w -- "${partial_marker}"' not in source
    assert 'chmod -R a-w -- "${runtime_stage_root}"' not in source
    assert 'find "${runtime_stage_root}"' not in stage
    assert 'find "${runtime_stage_root}"' not in attestation


def test_runtime_wrapper_separates_cpu_stage_from_gpu_attestation() -> None:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()

    assert "RUNTIME_PHASE=${RUNTIME_PHASE:-attest}" in source
    assert 'if [[ "${RUNTIME_PHASE}" == "stage" ]]' in source
    assert 'elif [[ "${SBATCH_GRES}" != "none" ]]' in source
    assert "srun_command=(srun --nodes=1 --ntasks=1)" in source
    assert 'srun_command+=("--gres=${SBATCH_GRES}")' in source
    assert '"--gpus-per-node=${SBATCH_GPUS_PER_NODE}"' not in source
    assert "NVTE_CUDA_ARCHS=100a" in source
    assert "RUNTIME_FEATURE_SET=${RUNTIME_FEATURE_SET:-te_eval_capability_8}" in source
    assert '"RUNTIME_FEATURE_SET=${RUNTIME_FEATURE_SET}"' in source
    assert "bridge_forward_only_eval_8" in source
    assert "dropless_hybridep_nano16" in source
    assert "dropless_alltoall_qwen30_16" in source
    assert "dropless_alltoall_super32" in source
    assert "dropless_hybridep_qwen235_64" in source
    assert "expected_runtime_exclusions=fast-hadamard-transform" in source
    assert "expected_runtime_exclusions=deep-ep,fast-hadamard-transform" in source
    assert '"RUNTIME_EXCLUDED_PACKAGES=${RUNTIME_EXCLUDED_PACKAGES}"' in source
    assert "RUNTIME_STAGE_CAPABILITY=${RUNTIME_STAGE_CAPABILITY:-}" in source
    assert "RUNTIME_BOOTSTRAP_PYTHON=/opt/nemo_rl_venv/bin/python" in source
    assert '"${RUNTIME_STAGE_CAPABILITY}" != "mcore-test-v1"' in source
    assert (
        "pytest==9.1.1,iniconfig==2.3.0,"
        "packaging==26.2,pluggy==1.6.0,pygments==2.20.0" in source
    )
    assert "TORCH_CUDA_ARCH_LIST=10.0a" in source
    assert (
        "RUNTIME_EXCLUDED_PACKAGES=${RUNTIME_EXCLUDED_PACKAGES:-causal-conv1d,"
        "deep-ep,fast-hadamard-transform,mamba-ssm}" in source
    )
    assert '"${uv_executable}" sync' in source
    assert "--group test" in source
    assert "--no-install-package" in source
    assert "runtime-stage-v1" in source
    assert "uv_lock_sha256" in source
    assert "RUNTIME_STAGE_MARKER_SHA256" in source
    assert '"stage_capability=${RUNTIME_STAGE_CAPABILITY}"' in source
    assert '"test_requirements=${RUNTIME_TEST_REQUIREMENTS}"' in source
    assert (
        '"${environment_root}/bin/python" - "${RUNTIME_TEST_REQUIREMENTS}" '
        '"${environment_root}" <<"PY"' in source
    )
    assert "importlib.metadata.version(distribution)" in source
    assert (
        'mv --no-clobber --no-target-directory -- "${partial_marker}" "${marker}"'
        in source
    )
    assert "--verify-only" in source
    assert '--trusted-symlink-root "${python_install_dir}"' in source
    assert "attestation_command='" in source
    attestation = source.split("attestation_command='", 1)[1].split("'\n\n", 1)[0]
    assert "uv run" not in attestation
    assert "python install" not in attestation
    assert "curl" not in attestation
    assert "cmake" not in attestation.lower()
    assert 'sha256sum "${marker}"' in attestation
    assert '"${runtime_python}" "${source_validator}"' in attestation


@pytest.mark.parametrize(
    ("feature_set", "excluded_packages"),
    (
        ("dropless_hybridep_nano16", "fast-hadamard-transform"),
        (
            "dropless_alltoall_qwen30_16",
            "deep-ep,fast-hadamard-transform",
        ),
        (
            "dropless_alltoall_super32",
            "deep-ep,fast-hadamard-transform",
        ),
        ("dropless_hybridep_qwen235_64", "fast-hadamard-transform"),
    ),
)
def test_runtime_wrapper_accepts_exact_dispatcher_exclusions(
    tmp_path: Path, feature_set: str, excluded_packages: str
) -> None:
    """The shell-side runtime table must distinguish AlltoAll from HybridEP."""
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        CONTAINER=str(tmp_path / "runtime.sqsh"),
        CONTAINER_SHA256="a" * 64,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        PROJECT_ROOT=str(REPO_ROOT),
        TEST_ONLY="1",
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CAPABILITY=RUNTIME_STAGE_CAPABILITY,
        RUNTIME_TEST_REQUIREMENTS=RUNTIME_TEST_REQUIREMENTS,
        RUNTIME_FEATURE_SET=feature_set,
        RUNTIME_EXCLUDED_PACKAGES=excluded_packages,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_stage_runs_exact_task2_root_suite_before_marker_publication(
    tmp_path: Path,
) -> None:
    runner = EXPERIMENT_DIR / "scripts" / "run_task2_root_tests.sh"
    assert runner.is_file()
    fake_python = tmp_path / "python"
    argument_log = tmp_path / "arguments.txt"
    fake_python.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" >"${TASK2_TEST_ARGUMENT_LOG:?}"\n'
    )
    fake_python.chmod(0o755)
    result_root = tmp_path / "results"

    result = subprocess.run(
        ["bash", str(runner), str(fake_python), str(result_root)],
        cwd=tmp_path,
        env={**os.environ, "TASK2_TEST_ARGUMENT_LOG": str(argument_log)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert argument_log.read_text().splitlines() == [
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={result_root}/tmp",
        f"--junitxml={result_root}/task-2-root.xml",
        "tests/unit/experiments/test_validate_te_runtime.py",
        "tests/unit/experiments/test_runtime_attestation.py",
        "tests/unit/experiments/test_container_harness_hardening.py",
        "tests/unit/experiments/test_mcore_standalone_driver.py",
        "tests/unit/experiments/test_matrix_submitters.py",
        "tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py",
    ]
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    stage = source.split("stage_command='", 1)[1].split("'\n\nattestation_command=", 1)[
        0
    ]
    test_index = stage.index('"${task2_root_test_runner}"')
    post_test_provenance_index = stage.index(
        '"${source_provenance_verifier}"', test_index
    )
    marker_index = stage.index(
        'mv --no-clobber --no-target-directory -- "${partial_marker}" "${marker}"'
    )
    assert test_index < post_test_provenance_index < marker_index


def test_task2_root_runner_removes_passing_pytest_basetemp(tmp_path: Path) -> None:
    runner = EXPERIMENT_DIR / "scripts" / "run_task2_root_tests.sh"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  case "${argument}" in\n'
        "    --basetemp=*) basetemp=${argument#*=} ;;\n"
        "  esac\n"
        "done\n"
        ': "${basetemp:?}"\n'
        'mkdir -p -- "${basetemp}"\n'
        'ln -s -- "${basetemp}/missing" "${basetemp}/broken-link"\n'
        "mkdir -p -- tests/unit/unit_results\n"
        "printf generated >tests/unit/unit_results.json\n"
        "printf generated >tests/unit/unit_results/result.json\n"
    )
    fake_python.chmod(0o755)
    result_root = tmp_path / "results"

    result = subprocess.run(
        ["bash", str(runner), str(fake_python), str(result_root)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result_root.is_dir()
    assert not (result_root / "tmp").exists()
    assert not (tmp_path / "tests/unit/unit_results.json").exists()
    assert not (tmp_path / "tests/unit/unit_results").exists()


def test_runtime_wrapper_requires_explicit_stage_capability(tmp_path: Path) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        CONTAINER=str(tmp_path / "runtime.sqsh"),
        CONTAINER_SHA256="a" * 64,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "must explicitly select mcore-test-v1" in result.stderr


def test_runtime_stage_publishes_marker_only_after_immutable_symlink_safe_audits() -> (
    None
):
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    runtime_environment = source.split("runtime_environment=(", 1)[1].split("\n)", 1)[0]
    stage = source.split("stage_command='", 1)[1].split("'\n\nattestation_command=", 1)[
        0
    ]
    attestation = source.split("attestation_command='", 1)[1].split("'\n\n", 1)[0]
    cleanup = stage.split("cleanup_runtime_workspace() {", 1)[1].split("\n}", 1)[0]

    assert cleanup.index('rm -f -- "${marker}"') < cleanup.index(
        'chmod -R u+w -- "${runtime_stage_root}"'
    )
    assert "2>/dev/null" not in cleanup
    assert "failed to restore runtime stage write permissions" in cleanup
    assert "failed to remove incomplete runtime stage" in cleanup
    assert (
        runtime_environment.count('"RUNTIME_STAGE_MARKER=${runtime_stage_marker}"') == 1
    )
    assert (
        runtime_environment.count('"RUNTIME_STAGE_JOB_ID=${runtime_stage_job_id}"') == 1
    )
    assert 'if [[ ! "${runtime_stage_job_id}" =~ ^[1-9][0-9]*$ ]]' in source
    assert "${ARTIFACT_DIR%/}/stage-markers/${runtime_stage_key}.env" in source
    assert 'find "${runtime_stage_root}"' not in stage
    assert 'find "${runtime_stage_root}"' not in attestation
    assert (
        ': "${RUNTIME_STAGE_MARKER:?Runtime stage payload requires RUNTIME_STAGE_MARKER}"'
        in stage
    )
    assert (
        ': "${RUNTIME_STAGE_JOB_ID:?Runtime stage payload requires RUNTIME_STAGE_JOB_ID}"'
        in stage
    )
    assert '"${RUNTIME_STAGE_JOB_ID}" >"${stage_job_record}"' in stage
    assert (
        ': "${RUNTIME_STAGE_MARKER:?Runtime attestation requires RUNTIME_STAGE_MARKER}"'
        in attestation
    )
    assert (
        ': "${RUNTIME_STAGE_JOB_ID:?Runtime attestation requires RUNTIME_STAGE_JOB_ID}"'
        in attestation
    )

    cleanup_index = stage.index('rm -rf -- "${uv_cache_dir}" "${te_cmake_dir}"')
    readonly_index = stage.index('"${bootstrap_python}" "${source_readonly_helper}"')
    marker_index = stage.index(
        'mv --no-clobber --no-target-directory -- "${partial_marker}" "${marker}"'
    )
    assert cleanup_index < readonly_index < marker_index
    attestation_helper_index = attestation.index("--verify-only")
    provenance_index = attestation.index('"${source_provenance_verifier}"')
    assert attestation_helper_index < provenance_index


def test_runtime_attestation_submitter_requires_completed_stage_job() -> None:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    submitter = source.split('if [[ -z "${SLURM_JOB_ID:-}" ]]', 1)[1].split(
        "unset PYTHONHOME", 1
    )[0]

    assert "RUNTIME_STAGE_JOB_ID" in submitter
    assert 'sacct -X -j "${RUNTIME_STAGE_JOB_ID}"' in submitter
    assert '"COMPLETED|0:0"' in submitter
    completed_gate = submitter.index('"COMPLETED|0:0"')
    marker_consumption = submitter.index('sha256sum "${RUNTIME_STAGE_MARKER}"')
    assert completed_gate < marker_consumption


def test_runtime_attestation_submitter_does_not_consume_running_stage_marker(
    tmp_path: Path,
) -> None:
    script = EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    artifact_dir = tmp_path / "artifacts"
    container = tmp_path / "runtime.sqsh"
    container.write_bytes(b"fixture container\n")
    base_environment = os.environ.copy()
    base_environment.update(
        {
            "CONTAINER": str(container),
            "CONTAINER_SHA256": "a" * 64,
            "ARTIFACT_DIR": str(artifact_dir),
            "PROJECT_ROOT": str(REPO_ROOT),
            "SOURCE_PROVENANCE_VERIFIER": str(
                EXPERIMENT_DIR / "scripts" / "verify_source_provenance.sh"
            ),
            "EXPECTED_NEMORL_SHA": NEMORL_COMMIT,
            "EXPECTED_BRIDGE_SHA": BRIDGE_COMMIT,
            "EXPECTED_MCORE_SHA": MCORE_COMMIT,
            "EXPECTED_TE_SHA": TE_COMMIT,
            "EXPECTED_TE_VERSION_BASE_SHA": TE_COMMIT,
            "RUNTIME_PHASE": "attest",
            "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
            "RUNTIME_TEST_REQUIREMENTS": RUNTIME_TEST_REQUIREMENTS,
            "TEST_ONLY": "1",
        }
    )
    render = subprocess.run(
        ["bash", str(script)],
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert render.returncode == 0, render.stderr
    rendered = dict(
        line.split("=", 1)
        for line in render.stdout.splitlines()
        if line.startswith("RUNTIME_STAGE_")
    )
    stage_root = Path(rendered["RUNTIME_STAGE_ROOT"])
    marker = Path(rendered["RUNTIME_STAGE_MARKER"])
    stage_root.mkdir(parents=True)
    (stage_root / "stage-job-id").write_text("733\n")
    marker.parent.mkdir(parents=True)
    marker.write_text("not consumed\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker_consumed = tmp_path / "marker-consumed"
    sbatch_called = tmp_path / "sbatch-called"
    _write_executable(
        fake_bin / "sacct",
        "#!/bin/sh\nprintf 'RUNNING|0:0\\n'\n",
    )
    _write_executable(
        fake_bin / "sha256sum",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  if [[ "${argument}" == "${RUNTIME_MARKER_TO_WATCH}" ]]; then\n'
        '    printf consumed >"${MARKER_CONSUMED}"\n'
        "  fi\n"
        "done\n"
        'exec /usr/bin/shasum -a 256 "$@"\n',
    )
    _write_executable(
        fake_bin / "sbatch",
        '#!/bin/sh\nprintf called >"${SBATCH_CALLED}"\n',
    )
    execution_environment = {
        **base_environment,
        "TEST_ONLY": "0",
        "RUNTIME_STAGE_JOB_ID": "733",
        "RUNTIME_MARKER_TO_WATCH": str(marker),
        "MARKER_CONSUMED": str(marker_consumed),
        "SBATCH_CALLED": str(sbatch_called),
        "PATH": f"{fake_bin}:{base_environment['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        env=execution_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "terminal COMPLETED" in result.stderr
    assert not marker_consumed.exists()
    assert not sbatch_called.exists()


@pytest.mark.parametrize("symlink_kind", ("broken", "escaped"))
def test_runtime_stage_audit_failure_never_publishes_marker(
    tmp_path: Path, symlink_kind: str
) -> None:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    stage = source.split("stage_command='", 1)[1].split("'\n\nattestation_command=", 1)[
        0
    ]
    tail = stage[stage.index('rm -rf -- "${uv_cache_dir}" "${te_cmake_dir}"') :]

    stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    stage_root.mkdir(parents=True)
    (stage_root / "payload").write_text("immutable\n")
    python_install_dir = tmp_path / "uv-python-installations"
    python_install_dir.mkdir()
    if symlink_kind == "broken":
        (stage_root / "unsafe-link").symlink_to(tmp_path / "missing")
        expected_error = "broken symlink"
    else:
        escaped_target = tmp_path / "escaped-target"
        escaped_target.write_text("outside\n")
        (stage_root / "unsafe-link").symlink_to(escaped_target)
        expected_error = "escapes trusted roots"

    marker_dir = tmp_path / "stage-markers"
    marker_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "realpath",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        '[[ "${1:-}" == -e ]] && shift\n'
        '[[ "${1:-}" == -- ]] && shift\n'
        f'exec "{sys.executable}" -c '
        "'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "
        '"$1"\n',
    )
    _write_executable(
        fake_bin / "chmod",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "arguments=()\n"
        'for argument in "$@"; do [[ "${argument}" == -- ]] || '
        'arguments+=("${argument}"); done\n'
        'exec /bin/chmod "${arguments[@]}"\n',
    )
    _write_executable(
        fake_bin / "python",
        f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
    )
    marker = marker_dir / f"{stage_root.name}.env"
    partial_marker = marker_dir / f".{stage_root.name}.733.partial"
    marker_lines = [
        "schema=runtime-stage-v1",
        f"stage_key={stage_root.name}",
        "container_sha256=" + "a" * 64,
        "uv_lock_sha256=" + "b" * 64,
        f"nemo_rl_commit={NEMORL_COMMIT}",
        f"bridge_commit={BRIDGE_COMMIT}",
        f"mcore_commit={MCORE_COMMIT}",
        f"te_source_commit={TE_COMMIT}",
        f"te_version_base_commit={TE_COMMIT}",
        f"python_version={PYTHON_VERSION}",
        f"uv_version={UV_VERSION}",
        "feature_set=te_eval_capability_8",
        "excluded_packages=causal-conv1d,deep-ep,fast-hadamard-transform,mamba-ssm",
        f"stage_capability={RUNTIME_STAGE_CAPABILITY}",
        f"test_requirements={RUNTIME_TEST_REQUIREMENTS}",
        "torch_cuda_arch_list=10.0a",
        "cuda_archs=100a",
        "stage_cpus_per_task=32",
    ]
    marker_sha256 = hashlib.sha256(
        ("\n".join(marker_lines) + "\n").encode()
    ).hexdigest()
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "uv_cache_dir": str(stage_root / "build-cache"),
            "te_cmake_dir": str(stage_root / "te-cmake"),
            "RUNTIME_STAGE_JOB_ID": "733",
            "stage_job_record": str(stage_root / "stage-job-id"),
            "container_sha256": "a" * 64,
            "uv_lock_sha256": "b" * 64,
            "nemo_rl_commit": NEMORL_COMMIT,
            "bridge_commit": BRIDGE_COMMIT,
            "mcore_commit": MCORE_COMMIT,
            "expected_te_commit": TE_COMMIT,
            "expected_te_version_base_commit": TE_COMMIT,
            "expected_python_version": PYTHON_VERSION,
            "expected_uv_version": UV_VERSION,
            "RUNTIME_FEATURE_SET": "te_eval_capability_8",
            "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
            "RUNTIME_TEST_REQUIREMENTS": RUNTIME_TEST_REQUIREMENTS,
            "RUNTIME_EXCLUDED_PACKAGES": (
                "causal-conv1d,deep-ep,fast-hadamard-transform,mamba-ssm"
            ),
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "NVTE_CUDA_ARCHS": "100a",
            "RUNTIME_STAGE_CPUS_PER_TASK": "32",
            "partial_marker": str(partial_marker),
            "RUNTIME_STAGE_MARKER_SHA256": marker_sha256,
            "runtime_stage_root": str(stage_root),
            "environment_root": str(tmp_path),
            "bootstrap_python": str(fake_bin / "python"),
            "source_readonly_helper": str(
                EXPERIMENT_DIR / "make_runtime_stage_readonly.py"
            ),
            "python_install_dir": str(python_install_dir),
            "marker": str(marker),
        }
    )

    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{tail}"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert not marker.exists()
    subprocess.run(["/bin/chmod", "-R", "u+w", stage_root], check=True)


def test_runtime_stage_readonly_helper_failure_never_publishes_marker(
    tmp_path: Path,
) -> None:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()
    stage = source.split("stage_command='", 1)[1].split("'\n\nattestation_command=", 1)[
        0
    ]
    tail = stage[stage.index("PYTHONDONTWRITEBYTECODE=1") :]

    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    environment_root = tmp_path / "environment"
    environment_root.joinpath("bin").mkdir(parents=True)
    _write_executable(
        environment_root / "bin" / "python",
        "#!/bin/sh\nexit 97\n",
    )
    partial_marker = tmp_path / ".partial.env"
    partial_marker.write_text("partial\n")
    marker = tmp_path / "published.env"
    environment = os.environ.copy()
    environment.update(
        {
            "environment_root": str(environment_root),
            "bootstrap_python": str(environment_root / "bin" / "python"),
            "source_readonly_helper": str(tmp_path / "readonly-helper.py"),
            "runtime_stage_root": str(stage_root),
            "partial_marker": str(partial_marker),
            "marker": str(marker),
            "python_install_dir": str(tmp_path / "python-installations"),
        }
    )

    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{tail}"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 97
    assert not marker.exists()


@pytest.mark.parametrize("marker_kind", ("missing", "symlink", "mismatch", "writable"))
def test_gpu_attestation_rejects_untrusted_runtime_stage(
    tmp_path: Path, marker_kind: str
) -> None:
    stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    stage_root.mkdir(parents=True)
    stage_job_record = stage_root / "stage-job-id"
    stage_job_record.write_text("733\n")
    readonly_helper = (
        stage_root
        / "source"
        / "experiments"
        / "cuda_graph"
        / "nemotron_thd_te_graph_20260731"
        / "make_runtime_stage_readonly.py"
    )
    readonly_helper.parent.mkdir(parents=True)
    readonly_helper.write_text(
        (EXPERIMENT_DIR / "make_runtime_stage_readonly.py").read_text()
    )
    source_readonly_helper = (
        tmp_path
        / "source"
        / "experiments"
        / "cuda_graph"
        / "nemotron_thd_te_graph_20260731"
        / "make_runtime_stage_readonly.py"
    )
    source_readonly_helper.parent.mkdir(parents=True)
    source_readonly_helper.write_text(
        (EXPERIMENT_DIR / "make_runtime_stage_readonly.py").read_text()
    )
    runtime_python = stage_root / "environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    _write_executable(
        runtime_python,
        f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
    )
    python_install_dir = tmp_path / "uv-python-installations"
    python_install_dir.mkdir()
    marker = tmp_path / "stage-markers" / f"{stage_root.name}.env"
    marker.parent.mkdir()
    expected_content = b"schema=runtime-stage-v1\n"
    expected_sha256 = hashlib.sha256(expected_content).hexdigest()
    if marker_kind == "symlink":
        target = tmp_path / "marker-target"
        target.write_bytes(expected_content)
        marker.symlink_to(target)
    elif marker_kind != "missing":
        marker.write_bytes(
            b"tampered\n" if marker_kind == "mismatch" else expected_content
        )
    if marker_kind != "writable":
        stage_root.chmod(0o555)
        stage_job_record.chmod(0o444)
        if marker.exists() and not marker.is_symlink():
            marker.chmod(0o444)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sha256sum",
        '#!/bin/sh\nexec /usr/bin/shasum -a 256 "$@"\n',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "RUNTIME_STAGE_MARKER": str(marker),
            "RUNTIME_STAGE_MARKER_SHA256": expected_sha256,
            "RUNTIME_STAGE_JOB_ID": "733",
            "RUNTIME_ATTESTATION_JOB_ID": "734",
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
            "RUNTIME_BOOTSTRAP_PYTHON": sys.executable,
        }
    )
    arguments = [
        "bash",
        "-c",
        _attestation_payload(),
        "bash",
        str(tmp_path / "source"),
        str(tmp_path / "source" / "validator.py"),
        str(stage_root / "environment"),
        str(tmp_path / "container.sqsh"),
        "a" * 64,
        NEMORL_COMMIT,
        BRIDGE_COMMIT,
        MCORE_COMMIT,
        "b" * 64,
        TE_COMMIT,
        "1",
        "2",
        "3",
        "4",
        "5",
        TE_COMMIT,
        "--output",
        str(tmp_path / "runtime.json"),
        str(stage_root),
    ]

    result = subprocess.run(
        arguments,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    stage_root.chmod(0o755)
    stage_job_record.chmod(0o644)
    if marker.exists() and not marker.is_symlink():
        marker.chmod(0o644)

    assert result.returncode == 2
    expected_error = {
        "missing": "completion marker is missing or unsafe",
        "symlink": "completion marker is missing or unsafe",
        "mismatch": "marker SHA256 mismatch",
        "writable": "contains writable regular state",
    }[marker_kind]
    assert expected_error in result.stderr


def test_gpu_attestation_rejects_escaped_runtime_python_before_execution(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    stage_root.mkdir(parents=True)
    stage_job_record = stage_root / "stage-job-id"
    stage_job_record.write_text("733\n")
    source_readonly_helper = (
        tmp_path
        / "source"
        / "experiments"
        / "cuda_graph"
        / "nemotron_thd_te_graph_20260731"
        / "make_runtime_stage_readonly.py"
    )
    source_readonly_helper.parent.mkdir(parents=True)
    source_readonly_helper.write_text(
        (EXPERIMENT_DIR / "make_runtime_stage_readonly.py").read_text()
    )
    escaped_python = tmp_path / "escaped-python"
    escaped_python_marker = tmp_path / "escaped-python-executed"
    _write_executable(
        escaped_python,
        '#!/bin/sh\nprintf executed >"${ESCAPED_PYTHON_MARKER}"\nexit 98\n',
    )
    runtime_python = stage_root / "environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(escaped_python)
    marker = tmp_path / "stage-markers" / f"{stage_root.name}.env"
    marker.parent.mkdir()
    marker.write_text("schema=runtime-stage-v1\n")
    marker_sha256 = hashlib.sha256(marker.read_bytes()).hexdigest()
    for path in sorted(stage_root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    stage_root.chmod(stage_root.stat().st_mode & ~0o222)
    marker.chmod(marker.stat().st_mode & ~0o222)
    python_install_dir = tmp_path / "uv-python-installations"
    python_install_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sha256sum",
        '#!/bin/sh\nexec /usr/bin/shasum -a 256 "$@"\n',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "RUNTIME_STAGE_MARKER": str(marker),
            "RUNTIME_STAGE_MARKER_SHA256": marker_sha256,
            "RUNTIME_STAGE_JOB_ID": "733",
            "RUNTIME_ATTESTATION_JOB_ID": "734",
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
            "RUNTIME_BOOTSTRAP_PYTHON": sys.executable,
            "ESCAPED_PYTHON_MARKER": str(escaped_python_marker),
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            _attestation_payload(),
            "bash",
            str(tmp_path / "source"),
            str(tmp_path / "source" / "validator.py"),
            str(stage_root / "environment"),
            str(tmp_path / "container.sqsh"),
            "a" * 64,
            NEMORL_COMMIT,
            BRIDGE_COMMIT,
            MCORE_COMMIT,
            "b" * 64,
            TE_COMMIT,
            "1",
            "2",
            "3",
            "4",
            "5",
            TE_COMMIT,
            "--output",
            str(tmp_path / "runtime.json"),
            str(stage_root),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    for path in sorted(stage_root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    stage_root.chmod(stage_root.stat().st_mode | 0o200)
    marker.chmod(marker.stat().st_mode | 0o200)
    assert result.returncode == 2
    assert "escapes trusted roots" in result.stderr
    assert not escaped_python_marker.exists()


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
            "UV_STAGE_MARKER": str(uv_stage_marker),
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(tmp_path / f"uv-{UV_VERSION}-733" / "uv"),
        }
    )

    result = _run_runtime_payload(
        fixture,
        environment=environment,
        cuda_home=tmp_path / "missing-cuda",
        cuda_compiler="",
    )

    assert result.returncode == 2
    assert "nvcc" in result.stderr.lower()
    assert not uv_stage_marker.exists()
    assert not fixture.copied_project_root.exists()


@pytest.mark.parametrize("path_kind", ("directory", "symlink"))
def test_runtime_payload_rejects_preexisting_keyed_stage_root(
    tmp_path: Path,
    path_kind: str,
) -> None:
    fixture = _stage_runtime_payload_fixture(
        tmp_path,
        verifier_body="#!/bin/sh\nexit 0\n",
    )
    state_path = fixture.environment_root.parent
    if path_kind == "directory":
        state_path.mkdir()
    else:
        symlink_target = tmp_path / "stage-target"
        symlink_target.mkdir()
        state_path.symlink_to(symlink_target, target_is_directory=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fixture.fake_bin}:/usr/bin:/bin",
            "CUDA_HOME": "/ambient/cuda",
            "CUDACXX": "/ambient/cuda/bin/nvcc",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(tmp_path / f"uv-{UV_VERSION}-733" / "uv"),
            "UV_CACHE_DIR": str(state_path / "build-cache"),
            "NVTE_CMAKE_BUILD_DIR": str(state_path / "te-cmake"),
        }
    )

    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 2
    assert "Runtime stage destination already exists" in result.stderr
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
            "UV_CACHE_DIR": "/ambient/runtime/build-cache",
            "NVTE_CMAKE_BUILD_DIR": "/ambient/runtime/te-cmake",
        }
    )
    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 2
    assert "Pinned uv destination already exists" in result.stderr
    assert not uv_marker.exists()


@pytest.mark.parametrize("outer_gitfile", (False, True))
def test_runtime_payload_builds_from_writable_verified_source_copy(
    tmp_path: Path,
    outer_gitfile: bool,
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
    if outer_gitfile:
        external_git_dir = tmp_path / "outer-worktree-gitdir"
        fixture.source_project_root.joinpath(".git").rename(external_git_dir)
        fixture.source_project_root.joinpath(".git").write_text(
            f"gitdir: {external_git_dir}\n"
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
        "  sync)\n"
        '    printf \'%s\\n\' "$@" >"${UV_SYNC_LOG}"\n'
        '    mkdir -p "${UV_CACHE_DIR}"\n'
        '    touch "${UV_CACHE_DIR}/build-marker"\n'
        '    mkdir -p "${NVTE_CMAKE_BUILD_DIR}"\n'
        '    touch "${NVTE_CMAKE_BUILD_DIR}/CMakeCache.txt"\n'
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
    uv_sync_log = tmp_path / "uv-sync.log"
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
            "UV_SYNC_LOG": str(uv_sync_log),
            "RUNTIME_FEATURE_SET": "dropless_alltoall_qwen30_16",
            "RUNTIME_EXCLUDED_PACKAGES": "deep-ep,fast-hadamard-transform",
        }
    )
    result = _run_runtime_payload(fixture, environment=environment)

    assert result.returncode == 93, result.stderr
    sync_arguments = uv_sync_log.read_text().splitlines()
    assert [
        sync_arguments[index + 1]
        for index, argument in enumerate(sync_arguments[:-1])
        if argument == "--no-install-package"
    ] == ["deep-ep", "fast-hadamard-transform"]
    assert not fixture.environment_root.exists()
    assert not fixture.copied_project_root.exists()
    assert not Path(f"{fixture.environment_root}-uv-cache").exists()
    assert not Path(f"{fixture.environment_root}-te-cmake").exists()


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
            "RUNTIME_STAGE_CAPABILITY": RUNTIME_STAGE_CAPABILITY,
            "RUNTIME_TEST_REQUIREMENTS": RUNTIME_TEST_REQUIREMENTS,
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
