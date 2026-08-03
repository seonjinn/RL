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
PYTHON_VERSION = "3.13.13"
UV_VERSION = "0.11.18"


def _device_bindings(
    *, num_nodes: int = 2, gpus_per_node: int = 4
) -> list[dict[str, int]]:
    return [
        {
            "global_rank": node_rank * gpus_per_node + local_rank,
            "node_rank": node_rank,
            "local_rank": local_rank,
            "cuda_device_index": local_rank,
        }
        for node_rank in range(num_nodes)
        for local_rank in range(gpus_per_node)
    ]


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


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    container = tmp_path / "nemo_rl_immutable.sqsh"
    container.write_bytes(b"fixture-container")
    lock = tmp_path / "uv.lock"
    lock.write_text("fixture-lock\n")
    lock_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest()
    container_stat = container.stat()
    python_install_dir = tmp_path / "uv-python-installations"
    python_base_executable = (
        python_install_dir / "cpython-3.13.13-linux-aarch64-gnu" / "bin" / "python3.13"
    )
    python_base_executable.parent.mkdir(parents=True, exist_ok=True)
    python_base_executable.write_bytes(b"managed-python-fixture")
    uv_executable = tmp_path / f"uv-{UV_VERSION}-733" / "uv"
    uv_executable.parent.mkdir(parents=True, exist_ok=True)
    uv_executable.write_text(f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n")
    uv_executable.chmod(0o755)
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
                "expected_python_version": PYTHON_VERSION,
                "python_version": PYTHON_VERSION,
                "uv_python_install_dir": str(python_install_dir),
                "python_base_executable": str(python_base_executable),
                "python_base_executable_sha256": hashlib.sha256(
                    python_base_executable.read_bytes()
                ).hexdigest(),
                "expected_uv_version": UV_VERSION,
                "expected_nvte_with_nccl_ep": "0",
                "uv_version": UV_VERSION,
                "uv_executable": str(uv_executable),
                "nvte_with_nccl_ep": "0",
                "transformer_engine_nccl_ep_available": False,
                "transformer_engine_nccl_ep_symbols": [],
                "transformer_engine_grouped_linear_symbols": [
                    "TEColumnParallelGroupedLinear",
                    "TERowParallelGroupedLinear",
                ],
                "uv_executable_sha256": hashlib.sha256(
                    uv_executable.read_bytes()
                ).hexdigest(),
                "packages": {
                    "torch": {"version": "2.11.0"},
                    "transformer_engine.pytorch": {"version": "2.19.0.dev0+bffde8f4"},
                    "megatron.core": {"version": "0.16.0rc0"},
                    "megatron.core.extensions.transformer_engine": {
                        "version": "0.16.0rc0"
                    },
                    "megatron.bridge": {"version": "0.2.0"},
                    "mamba_ssm": {"version": "2.2.6.post3"},
                    "causal_conv1d": {"version": "1.5.3.post1"},
                    "cupy": {"version": "14.0.1"},
                },
            }
        )
    )
    return attestation, container, lock, python_install_dir, uv_executable


def test_validator_accepts_exact_preflight_artifact_without_rehashing_container(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )
    finally:
        container.chmod(0o644)

    assert result["status"] == "passed"
    assert result["transformer_engine_vcs_commit"] == TE_COMMIT
    assert result["nvte_with_nccl_ep"] == "0"
    assert result["transformer_engine_nccl_ep_available"] is False


def test_validator_rejects_wrong_nvte_nccl_ep_policy(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    payload["nvte_with_nccl_ep"] = "1"
    attestation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attestation provenance mismatch"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
            expected_nvte_with_nccl_ep="0",
        )

    module._require_nvte_environment(
        expected_nvte_with_nccl_ep="0",
        environment={"NVTE_WITH_NCCL_EP": "0"},
    )
    with pytest.raises(ValueError, match="process environment mismatch"):
        module._require_nvte_environment(
            expected_nvte_with_nccl_ep="0",
            environment={},
        )


def test_validator_rejects_mutated_container_identity_or_uv_lock(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )

    attestation, container, lock, python_install_dir, uv_executable = _fixture(
        tmp_path / "hash"
    )
    uv_executable.write_text("mutated uv\n")
    with pytest.raises(ValueError, match="uv executable SHA256 mismatch"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )

    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )


def test_validator_rejects_symlink_or_wrong_source_and_te_provenance(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )


def test_validator_requires_complete_worker_stack_and_te_216_or_newer(
    tmp_path: Path,
) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )


def test_validator_binds_typed_te_eval_runtime_contract(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    payload.update(
        {
            "runtime_feature_set": "te_eval_capability_8",
            "excluded_packages": [
                "causal-conv1d",
                "deep-ep",
                "fast-hadamard-transform",
                "mamba-ssm",
            ],
            "torch_cuda_arch_list": "10.0a",
            "nvte_cuda_archs": "100a",
        }
    )
    del payload["packages"]["mamba_ssm"]
    del payload["packages"]["causal_conv1d"]
    attestation.write_text(json.dumps(payload))
    contract = {
        "expected_runtime_feature_set": "te_eval_capability_8",
        "expected_excluded_packages": (
            "causal-conv1d",
            "deep-ep",
            "fast-hadamard-transform",
            "mamba-ssm",
        ),
        "expected_torch_cuda_arch_list": "10.0a",
        "expected_nvte_cuda_archs": "100a",
    }

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
        expected_python_version=PYTHON_VERSION,
        expected_python_install_dir=python_install_dir,
        expected_uv_version=UV_VERSION,
        expected_uv_executable=uv_executable,
        **contract,
    )

    assert result["runtime_feature_set"] == "te_eval_capability_8"
    payload["torch_cuda_arch_list"] = "10.0"
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature contract mismatch"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
            **contract,
        )


def test_validator_rejects_wrong_or_mutated_managed_python(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    payload["python_version"] = "3.13.11"
    attestation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attestation provenance mismatch"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )

    attestation, container, lock, python_install_dir, uv_executable = _fixture(
        tmp_path / "hash"
    )
    payload = json.loads(attestation.read_text())
    Path(payload["python_base_executable"]).write_bytes(b"mutated-python")

    with pytest.raises(ValueError, match="managed Python executable SHA256 mismatch"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )


def test_validator_rejects_symlinked_managed_python_paths(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(
        tmp_path / "install"
    )
    linked_install_dir = tmp_path / "install" / "uv-python-installations-link"
    linked_install_dir.symlink_to(python_install_dir, target_is_directory=True)
    payload = json.loads(attestation.read_text())
    payload["uv_python_install_dir"] = str(linked_install_dir)
    attestation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="install directory must not be a symlink"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=linked_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )

    attestation, container, lock, python_install_dir, uv_executable = _fixture(
        tmp_path / "base"
    )
    payload = json.loads(attestation.read_text())
    python_base_executable = Path(payload["python_base_executable"])
    python_target = python_base_executable.with_name("python3.13.real")
    python_base_executable.replace(python_target)
    python_base_executable.symlink_to(python_target.name)

    with pytest.raises(ValueError, match="executable must not be a symlink"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )


def test_validator_rejects_wrong_or_mutated_uv(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(
        tmp_path / "version"
    )
    payload = json.loads(attestation.read_text())
    payload["uv_version"] = "0.11.1"
    attestation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attestation provenance mismatch"):
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
            expected_python_version=PYTHON_VERSION,
            expected_python_install_dir=python_install_dir,
            expected_uv_version=UV_VERSION,
            expected_uv_executable=uv_executable,
        )


def test_matrix_validator_requires_exact_content_bound_rows(tmp_path: Path) -> None:
    module = _load_module()
    candidate_sha = "f" * 40
    candidate_dir = tmp_path / "mcore" / candidate_sha
    candidate_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "candidate_kind": "mcore",
        "candidate_sha": candidate_sha,
        "integration_sha": MCORE_COMMIT,
        "container_sha256": CONTAINER_SHA256,
        "transformer_engine_version": "2.19.0.dev0",
        "transformer_engine_source_commit": TE_COMMIT,
        "transformer_engine_version_base_commit": "e" * 40,
        "all_eval_callables_supported": True,
        "mcore_eval_reuse_graph_io": "not_implemented",
        "raw_te_eval_reuse_graph_io": True,
        "topology": {
            "world_size": 8,
            "num_nodes": 2,
            "gpus_per_node": 4,
            "joined_ranks": list(range(8)),
            "device_bindings": _device_bindings(),
        },
        "test_row_id": "te_eval_capability_8",
        "node_results": [],
    }
    (candidate_dir / "te_eval_capability_8.json").write_text(json.dumps(payload))

    results = module.validate_matrix_results(
        candidate_kind="mcore",
        candidate_sha=candidate_sha,
        integration_sha=MCORE_COMMIT,
        expected_container_sha256=CONTAINER_SHA256,
        expected_te_commit=TE_COMMIT,
        expected_te_version_base_commit="e" * 40,
        test_result_dir=tmp_path,
        required_rows=("te_eval_capability_8",),
    )

    assert results["te_eval_capability_8"] == payload
    (candidate_dir / "extra.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extra matrix result"):
        module.validate_matrix_results(
            candidate_kind="mcore",
            candidate_sha=candidate_sha,
            integration_sha=MCORE_COMMIT,
            expected_container_sha256=CONTAINER_SHA256,
            expected_te_commit=TE_COMMIT,
            expected_te_version_base_commit="e" * 40,
            test_result_dir=tmp_path,
            required_rows=("te_eval_capability_8",),
        )


def test_matrix_validator_rejects_duplicate_or_missing_device_slots(
    tmp_path: Path,
) -> None:
    module = _load_module()
    candidate_sha = "f" * 40
    candidate_dir = tmp_path / "mcore" / candidate_sha
    candidate_dir.mkdir(parents=True)
    bindings = _device_bindings()
    bindings[1] = {
        **bindings[1],
        "local_rank": 0,
        "cuda_device_index": 0,
    }
    payload = {
        "schema_version": 1,
        "status": "passed",
        "candidate_kind": "mcore",
        "candidate_sha": candidate_sha,
        "integration_sha": MCORE_COMMIT,
        "container_sha256": CONTAINER_SHA256,
        "transformer_engine_version": "2.19.0.dev0",
        "transformer_engine_source_commit": TE_COMMIT,
        "transformer_engine_version_base_commit": "e" * 40,
        "all_eval_callables_supported": True,
        "mcore_eval_reuse_graph_io": "not_implemented",
        "raw_te_eval_reuse_graph_io": True,
        "topology": {
            "world_size": 8,
            "num_nodes": 2,
            "gpus_per_node": 4,
            "joined_ranks": list(range(8)),
            "device_bindings": bindings,
        },
        "test_row_id": "te_eval_capability_8",
        "node_results": [],
    }
    (candidate_dir / "te_eval_capability_8.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="duplicate or missing"):
        module.validate_matrix_results(
            candidate_kind="mcore",
            candidate_sha=candidate_sha,
            integration_sha=MCORE_COMMIT,
            expected_container_sha256=CONTAINER_SHA256,
            expected_te_commit=TE_COMMIT,
            expected_te_version_base_commit="e" * 40,
            test_result_dir=tmp_path,
            required_rows=("te_eval_capability_8",),
        )


def test_matrix_validator_rejects_global_rank_bound_to_wrong_device_slot() -> None:
    module = _load_module()
    bindings = _device_bindings()
    bindings[0] = {**bindings[0], "global_rank": 4}
    bindings[4] = {**bindings[4], "global_rank": 0}

    with pytest.raises(ValueError, match="global rank.*device slot"):
        module._validate_device_bindings(
            bindings,
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
        )
