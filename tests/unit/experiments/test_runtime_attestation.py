from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "nemotron_thd_te_graph_20260731"
    / "verify_runtime_attestation.py"
)
NEMORL_COMMIT = "a" * 40
BRIDGE_COMMIT = "b" * 40
MCORE_COMMIT = "c" * 40
TE_COMMIT = "d" * 40
CONTAINER_SHA256 = "e" * 64


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_runtime_attestation", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    container = tmp_path / "nemo_rl_immutable.sqsh"
    container.write_bytes(b"fixture-container")
    lock = tmp_path / "uv.lock"
    lock.write_text("fixture-lock\n")
    lock_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest()
    container_stat = container.stat()
    attestation = tmp_path / "runtime-733.json"
    attestation.write_text(
        json.dumps(
            {
                "status": "passed",
                "container_image": str(container),
                "container_sha256": CONTAINER_SHA256,
                "container_device": container_stat.st_dev,
                "container_inode": container_stat.st_ino,
                "container_size": container_stat.st_size,
                "container_mtime_seconds": int(container_stat.st_mtime),
                "container_ctime_seconds": int(container_stat.st_ctime),
                "nemo_rl_commit": NEMORL_COMMIT,
                "bridge_commit": BRIDGE_COMMIT,
                "mcore_commit": MCORE_COMMIT,
                "uv_lock_sha256": lock_sha256,
                "expected_te_commit": TE_COMMIT,
                "transformer_engine_vcs_commit": TE_COMMIT,
                "device_count": 4,
                "expected_device_count": 4,
                "packages": {
                    "torch": {"version": "2.11.0"},
                    "transformer_engine.pytorch": {"version": "2.19.0.dev0+bffde8f4"},
                    "megatron.core": {"version": "0.16.0rc0"},
                    "megatron.bridge": {"version": "0.2.0"},
                    "mamba_ssm": {"version": "2.2.6.post3"},
                    "causal_conv1d": {"version": "1.5.3.post1"},
                    "cupy": {"version": "14.0.1"},
                    "grouped_gemm": {"version": "1.1.4"},
                },
            }
        )
    )
    return attestation, container, lock


def test_validator_accepts_exact_preflight_artifact_without_rehashing_container(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock = _fixture(tmp_path)
    container.chmod(0)
    try:
        result = module.validate_attestation(
            attestation=attestation,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit=NEMORL_COMMIT,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )
    finally:
        container.chmod(0o644)

    assert result["status"] == "passed"
    assert result["transformer_engine_vcs_commit"] == TE_COMMIT


def test_validator_rejects_mutated_container_identity_or_uv_lock(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock = _fixture(tmp_path)
    container.write_bytes(b"different-size-container")

    with pytest.raises(ValueError, match="container identity mismatch"):
        module.validate_attestation(
            attestation=attestation,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit=NEMORL_COMMIT,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )

    attestation, container, lock = _fixture(tmp_path)
    lock.write_text("mutated-lock\n")
    with pytest.raises(ValueError, match="uv.lock SHA256 mismatch"):
        module.validate_attestation(
            attestation=attestation,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit=NEMORL_COMMIT,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )


def test_validator_rejects_symlink_or_wrong_source_and_te_provenance(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock = _fixture(tmp_path)
    symlink = tmp_path / "runtime-latest.json"
    symlink.symlink_to(attestation)

    with pytest.raises(ValueError, match="attestation must not be a symlink"):
        module.validate_attestation(
            attestation=symlink,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit=NEMORL_COMMIT,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )

    payload = json.loads(attestation.read_text())
    payload["transformer_engine_vcs_commit"] = "f" * 40
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="attestation provenance mismatch"):
        module.validate_attestation(
            attestation=attestation,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit="0" * 40,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )


def test_validator_requires_complete_worker_stack_and_te_216_or_newer(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    del payload["packages"]["mamba_ssm"]
    payload["packages"]["transformer_engine.pytorch"]["version"] = "2.15.0"
    attestation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="missing required packages"):
        module.validate_attestation(
            attestation=attestation,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit=NEMORL_COMMIT,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )

    payload["packages"]["mamba_ssm"] = {"version": "2.2.6.post3"}
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Transformer Engine >= 2.16"):
        module.validate_attestation(
            attestation=attestation,
            container=container,
            expected_container_sha256=CONTAINER_SHA256,
            nemo_rl_commit=NEMORL_COMMIT,
            bridge_commit=BRIDGE_COMMIT,
            mcore_commit=MCORE_COMMIT,
            uv_lock=lock,
            expected_te_commit=TE_COMMIT,
            expected_device_count=4,
        )
