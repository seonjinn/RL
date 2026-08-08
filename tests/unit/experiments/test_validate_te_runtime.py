from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TE_COMMIT = "bffde8f4a0a4eea9036dc753e28269247e5de69d"
EXPECTED_TE_SOURCE_COMMIT = "04a76c84423d9a4eb2f2010ef6692e347326cc00"
EXPECTED_TE_SOURCE_VERSION = "2.19.0.dev0+04a76c84"
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "nemotron_thd_te_graph_20260731"
    / "validate_te_runtime.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_te_runtime", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "transformer-engine"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "revision.txt").write_text("minimum\n")
    _git(repository, "add", "revision.txt")
    _git(repository, "commit", "-qm", "minimum runtime")
    minimum_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "revision.txt").write_text("newer\n")
    _git(repository, "commit", "-qam", "newer runtime")
    native_commit = _git(repository, "rev-parse", "HEAD")
    return repository, minimum_commit, native_commit


def _runtime_fixture(
    tmp_path: Path,
    *,
    version: str = "2.16.0.dev0",
    native_commit: str,
    source_repository: Path,
) -> tuple[Path, Path, Path, Path, str]:
    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"immutable-nightly-container")
    container_sha256 = hashlib.sha256(container.read_bytes()).hexdigest()
    site_packages = tmp_path / "runtime" / "site-packages"
    package = site_packages / "transformer_engine"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (package / "libtransformer_engine.so").write_bytes(b"native-core")
    provenance = tmp_path / "te-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "container": str(container.resolve()),
                "container_sha256": container_sha256,
                "install_prefix": str(site_packages.parent.resolve()),
                "site_packages": str(site_packages.resolve()),
                "source_repository": str(source_repository.resolve()),
                "transformer_engine_commit": native_commit,
                "transformer_engine_version": version,
            }
        )
    )
    return (
        provenance,
        site_packages,
        container,
        tmp_path / "validated.json",
        container_sha256,
    )


def test_validator_accepts_te_216_descendant_and_writes_machine_readable_output(
    tmp_path: Path,
) -> None:
    module = _load_module()
    repository, minimum_commit, native_commit = _source_repository(tmp_path)
    provenance, site_packages, container, output, digest = _runtime_fixture(
        tmp_path,
        native_commit=native_commit,
        source_repository=repository,
    )

    result = module.validate_runtime(
        provenance_path=provenance,
        site_packages=site_packages,
        container=container,
        expected_container_sha256=digest,
        source_repository=repository,
        minimum_commit=minimum_commit,
        output=output,
        validate_imports=False,
    )

    assert result["status"] == "passed"
    assert result["transformer_engine_version"] == "2.16.0.dev0"
    assert result["transformer_engine_commit"] == native_commit
    assert result["transformer_engine_source_commit"] == native_commit
    assert result["transformer_engine_version_base_commit"] == native_commit
    assert result["minimum_commit"] == minimum_commit
    assert result["ancestry_verified"] is True
    assert result["all_eval_callables_supported"] == "not_tested"
    assert result["mcore_eval_reuse_graph_io"] == "not_implemented"
    assert result["raw_te_eval_reuse_graph_io"] == "not_tested"
    assert result["test_row_id"] == "runtime_preflight"
    assert json.loads(output.read_text()) == result


def test_validator_rejects_native_revision_older_than_minimum(
    tmp_path: Path,
) -> None:
    module = _load_module()
    repository, older_commit, minimum_commit = _source_repository(tmp_path)
    provenance, site_packages, container, output, digest = _runtime_fixture(
        tmp_path,
        native_commit=older_commit,
        source_repository=repository,
    )

    with pytest.raises(ValueError, match="not at or after minimum commit"):
        module.validate_runtime(
            provenance_path=provenance,
            site_packages=site_packages,
            container=container,
            expected_container_sha256=digest,
            source_repository=repository,
            minimum_commit=minimum_commit,
            output=output,
            validate_imports=False,
        )

    assert not output.exists()


def test_validator_rejects_te_215_even_with_valid_commit(tmp_path: Path) -> None:
    module = _load_module()
    repository, minimum_commit, native_commit = _source_repository(tmp_path)
    provenance, site_packages, container, output, digest = _runtime_fixture(
        tmp_path,
        version="2.15.9",
        native_commit=native_commit,
        source_repository=repository,
    )

    with pytest.raises(ValueError, match="requires Transformer Engine >= 2.16"):
        module.validate_runtime(
            provenance_path=provenance,
            site_packages=site_packages,
            container=container,
            expected_container_sha256=digest,
            source_repository=repository,
            minimum_commit=minimum_commit,
            output=output,
            validate_imports=False,
        )


def test_validator_rejects_container_or_provenance_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    repository, minimum_commit, native_commit = _source_repository(tmp_path)
    provenance, site_packages, container, output, digest = _runtime_fixture(
        tmp_path,
        native_commit=native_commit,
        source_repository=repository,
    )

    with pytest.raises(ValueError, match="container SHA256 mismatch"):
        module.validate_runtime(
            provenance_path=provenance,
            site_packages=site_packages,
            container=container,
            expected_container_sha256="0" * 64,
            source_repository=repository,
            minimum_commit=minimum_commit,
            output=output,
            validate_imports=False,
        )
    assert digest != "0" * 64


def test_validator_requires_every_import_to_resolve_inside_runtime_prefix(
    tmp_path: Path,
) -> None:
    module = _load_module()
    site_packages = tmp_path / "runtime" / "site-packages"
    inside = site_packages / "transformer_engine" / "__init__.py"
    outside = tmp_path / "system" / "transformer_engine_torch.so"

    with pytest.raises(ValueError, match="resolved outside the native runtime"):
        module.validate_resolved_paths(
            {
                "transformer_engine": inside,
                "transformer_engine_torch": outside,
            },
            site_packages=site_packages,
        )


def test_validator_source_contains_no_clone_or_native_build_path() -> None:
    source = MODULE_PATH.read_text()

    assert "git clone" not in source
    assert "pip install" not in source
    assert "uv sync" not in source
    assert "cmake" not in source.lower()


def test_outer_project_pins_the_validated_te_runtime_by_full_commit() -> None:
    project_text = (REPO_ROOT / "pyproject.toml").read_text()
    project = tomllib.loads(project_text)

    assert (
        "transformer-engine[pytorch,core_cu13] @ "
        "git+https://github.com/seonjinn/TransformerEngine.git@"
        f"{EXPECTED_TE_SOURCE_COMMIT}" in project_text
    )
    assert "TransformerEngine.git@release_v2.15" not in project_text
    assert '"nvidia-cudnn-frontend==1.26.0"' in project_text
    assert '"nvidia-nccl-cu13==2.30.7"' in project_text
    dependency_metadata = {
        entry["name"]: entry for entry in project["tool"]["uv"]["dependency-metadata"]
    }
    assert dependency_metadata["transformer-engine"]["version"] == (
        EXPECTED_TE_SOURCE_VERSION
    )
    assert dependency_metadata["transformer-engine-torch"]["version"] == (
        EXPECTED_TE_SOURCE_VERSION
    )

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked_packages = {entry["name"]: entry for entry in lock["package"]}
    assert locked_packages["transformer-engine"]["version"] == (
        EXPECTED_TE_SOURCE_VERSION
    )
    locked_metadata = {
        entry["name"]: entry for entry in lock["manifest"]["dependency-metadata"]
    }
    assert locked_metadata["transformer-engine"]["version"] == (
        EXPECTED_TE_SOURCE_VERSION
    )
    assert locked_metadata["transformer-engine-torch"]["version"] == (
        EXPECTED_TE_SOURCE_VERSION
    )
