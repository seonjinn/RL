from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
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
DEEP_EP_COMMIT = "f" * 40
CONTAINER_SHA256 = "e" * 64
PYTHON_VERSION = "3.13.13"
UV_VERSION = "0.11.18"
TE_EVAL_NODES = [
    {
        "node": (
            "tests/unit_tests/transformer/test_cuda_graphs.py::"
            "test_te_make_graphed_callables_supports_eval_no_grad"
        ),
        "status": "passed",
        "exit_code": 0,
    },
    {
        "node": (
            "tests/unit_tests/transformer/test_cuda_graphs.py::"
            "test_te_eval_graph_input_output_buffer_reuse_capability"
        ),
        "status": "passed",
        "exit_code": 0,
    },
]


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


def _te_capability_evidence() -> dict[str, object]:
    return {
        "all_eval_callables_supported": True,
        "backward_executed": False,
        "fallback_forward_counter_increment": 1,
        "forward_invocations_after_capture": 3,
        "no_parameter_grads": True,
        "outputs_changed": True,
        "replay_forward_counter_increment": 0,
        "mcore_eval_reuse_graph_io": "not_implemented",
        "raw_te_eval_reuse_graph_io": True,
        "raw_te_eval_reuse_rejection": None,
        "raw_te_eval_reuse_eager_parity": True,
        "raw_te_eval_reuse_fallback_forward_counter_increment": 1,
        "raw_te_eval_reuse_no_parameter_grads": True,
        "raw_te_eval_reuse_outputs_changed": True,
        "raw_te_eval_reuse_replay_forward_counter_increment": 0,
    }


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
                "runtime_attestation_job_id": 733,
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


def _add_vllm_actor_runtime(payload: dict, uv_executable: Path) -> None:
    environment_root = uv_executable.parent.parent / "vllm-environment"
    python_executable = environment_root / "bin" / "python"
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n")
    python_executable.chmod(0o755)
    package = {
        "distribution": "vllm",
        "version": "0.25.1",
        "path": str(environment_root / "lib/python3.13/site-packages/vllm/__init__.py"),
    }
    payload["packages"]["vllm"] = package
    payload["actor_runtimes"] = {
        "vllm": {
            "python_executable": str(python_executable),
            "runtime_prefix": str(environment_root),
            "cuda_available": True,
            "device_count": 4,
            "excluded_packages": payload["excluded_packages"],
            "packages": {
                "torch": {"distribution": "torch", "version": "2.11.0"},
                "vllm": package,
            },
        }
    }


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


def test_validator_binds_attestation_to_producer_job(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)

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
        expected_runtime_attestation_job_id=733,
    )
    assert result["runtime_attestation_job_id"] == 733

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
            expected_runtime_attestation_job_id=734,
        )


def test_validator_rejects_partial_runtime_feature_contract(tmp_path: Path) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)

    with pytest.raises(ValueError, match="must be provided together"):
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
            expected_runtime_feature_set="dropless_hybridep_nano16",
        )


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


@pytest.mark.parametrize(
    "feature_set", ("te_eval_capability_8", "bridge_forward_only_eval_8")
)
def test_validator_binds_typed_te_eval_runtime_contract(
    tmp_path: Path, feature_set: str
) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    payload.update(
        {
            "runtime_feature_set": feature_set,
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
        "expected_runtime_feature_set": feature_set,
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

    assert result["runtime_feature_set"] == feature_set
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


@pytest.mark.parametrize(
    "feature_set",
    ("dropless_hybridep_nano16", "dropless_hybridep_qwen235_64"),
)
def test_validator_binds_dropless_hybridep_runtime_contract(
    tmp_path: Path, feature_set: str
) -> None:
    """HybridEP evidence must prove DeepEP is installed in the immutable runtime."""
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    payload.update(
        {
            "runtime_feature_set": feature_set,
            "excluded_packages": ["fast-hadamard-transform"],
            "torch_cuda_arch_list": "10.0a",
            "nvte_cuda_archs": "100a",
            "hybridep_buffer_available": True,
            "deep_ep_vcs_commit": DEEP_EP_COMMIT,
        }
    )
    payload["packages"]["deep_ep"] = {"version": "1.2.1"}
    _add_vllm_actor_runtime(payload, uv_executable)
    attestation.write_text(json.dumps(payload))
    contract = {
        "expected_runtime_feature_set": feature_set,
        "expected_excluded_packages": ("fast-hadamard-transform",),
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

    assert result["runtime_feature_set"] == feature_set
    assert result["deep_ep_vcs_commit"] == DEEP_EP_COMMIT
    payload["deep_ep_vcs_commit"] = "not-a-full-commit"
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="DeepEP VCS commit"):
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
    payload["deep_ep_vcs_commit"] = DEEP_EP_COMMIT
    payload["hybridep_buffer_available"] = False
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="HybridEPBuffer"):
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
    payload["hybridep_buffer_available"] = True
    del payload["packages"]["deep_ep"]
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="deep_ep"):
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
    payload["packages"]["deep_ep"] = {"version": "1.2.1"}
    del payload["packages"]["vllm"]
    attestation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="vllm"):
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


@pytest.mark.parametrize(
    "feature_set",
    ("dropless_alltoall_qwen30_16", "dropless_alltoall_super32"),
)
def test_validator_binds_dropless_alltoall_runtime_contract(
    tmp_path: Path, feature_set: str
) -> None:
    module = _load_module()
    attestation, container, lock, python_install_dir, uv_executable = _fixture(tmp_path)
    payload = json.loads(attestation.read_text())
    payload.update(
        {
            "runtime_feature_set": feature_set,
            "excluded_packages": ["deep-ep", "fast-hadamard-transform"],
            "torch_cuda_arch_list": "10.0a",
            "nvte_cuda_archs": "100a",
        }
    )
    _add_vllm_actor_runtime(payload, uv_executable)
    attestation.write_text(json.dumps(payload))

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
        expected_runtime_feature_set=feature_set,
        expected_excluded_packages=("deep-ep", "fast-hadamard-transform"),
        expected_torch_cuda_arch_list="10.0a",
        expected_nvte_cuda_archs="100a",
    )

    assert result["runtime_feature_set"] == feature_set


@pytest.mark.parametrize(
    ("feature_set", "excluded_packages"),
    (
        ("dropless_hybridep_nano16", ("fast-hadamard-transform",)),
        (
            "dropless_alltoall_qwen30_16",
            ("deep-ep", "fast-hadamard-transform"),
        ),
        (
            "dropless_alltoall_super32",
            ("deep-ep", "fast-hadamard-transform"),
        ),
        ("dropless_hybridep_qwen235_64", ("fast-hadamard-transform",)),
    ),
)
def test_mcore_submitter_resolves_exact_row_exclusions(
    tmp_path: Path,
    feature_set: str,
    excluded_packages: tuple[str, ...],
) -> None:
    """The submitter must export the same typed exclusions verified by leaf jobs."""
    submitter = (MODULE_PATH.parent / "submit_mcore_matrix.sh").read_text()
    start_marker = 'runtime_contract=$(SELECTION="${selection}" python3 - "${RUNTIME_ATTESTATION}" <<\'PY\'\n'
    start = submitter.index(start_marker) + len(start_marker)
    contract_program = submitter[start : submitter.index("\nPY\n)", start)]
    attestation = tmp_path / "runtime.json"
    attestation.write_text(
        json.dumps(
            {
                "runtime_feature_set": feature_set,
                "excluded_packages": list(excluded_packages),
                "torch_cuda_arch_list": "10.0a",
                "nvte_cuda_archs": "100a",
            }
        )
    )

    result = subprocess.run(
        [sys.executable, "-", str(attestation)],
        input=contract_program,
        env={**os.environ, "SELECTION": f"{feature_set}\t16\t4\t4"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split("\t") == [
        feature_set,
        ",".join(excluded_packages),
        "10.0a",
        "100a",
    ]


@pytest.mark.parametrize(
    ("row_id", "excluded_packages"),
    (
        ("dropless_hybridep_nano16", ("fast-hadamard-transform",)),
        (
            "dropless_alltoall_qwen30_16",
            ("deep-ep", "fast-hadamard-transform"),
        ),
        (
            "dropless_alltoall_super32",
            ("deep-ep", "fast-hadamard-transform"),
        ),
        ("dropless_hybridep_qwen235_64", ("fast-hadamard-transform",)),
    ),
)
def test_profile_runtime_contract_selects_exact_moe_row(
    row_id: str, excluded_packages: tuple[str, ...]
) -> None:
    module = _load_module()

    assert module._runtime_contract_for_rows(
        candidate_kind="mcore",
        required_rows=(row_id,),
    ) == (
        row_id,
        excluded_packages,
    )


def test_matrix_mode_uses_profile_uv_as_independent_expected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    profile_uv = tmp_path / "profile-runtime" / "uv"
    attested_uv = tmp_path / "different-attested-runtime" / "uv"
    attestation = tmp_path / "runtime.json"
    attestation.write_text("{}")
    profile = tmp_path / "profile.env"
    profile.write_text(
        f"RUNTIME_ATTESTATION={attestation}\n"
        f"CONTAINER={tmp_path / 'container.sqsh'}\n"
        f"CONTAINER_SHA256={CONTAINER_SHA256}\n"
        f"EXPECTED_NEMORL_SHA={NEMORL_COMMIT}\n"
        f"EXPECTED_BRIDGE_SHA={BRIDGE_COMMIT}\n"
        f"EXPECTED_MCORE_SHA={MCORE_COMMIT}\n"
        f"EXPECTED_TE_SHA={TE_COMMIT}\n"
        f"EXPECTED_TE_VERSION_BASE_SHA={TE_COMMIT}\n"
        "SBATCH_GPUS_PER_NODE=4\n"
        "RUNTIME_PREFLIGHT_JOB_ID=733\n"
        f"UV_EXECUTABLE={profile_uv}\n"
    )
    runtime_payload = {
        "expected_nvte_with_nccl_ep": "0",
        "expected_python_version": PYTHON_VERSION,
        "uv_python_install_dir": str(tmp_path / "uv-python-installations"),
        "expected_uv_version": UV_VERSION,
        "uv_executable": str(attested_uv),
    }
    captured: dict[str, object] = {}

    def capture_attestation(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return runtime_payload

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            profile_file=profile,
            candidate_kind="mcore",
            candidate_sha="f" * 40,
            test_result_dir=tmp_path,
            required_rows="dropless_hybridep_nano16",
        ),
    )
    monkeypatch.setattr(module, "_read_attestation", lambda _: runtime_payload)
    monkeypatch.setattr(module, "_require_nvte_environment", lambda **_: None)
    monkeypatch.setattr(module, "validate_attestation", capture_attestation)
    monkeypatch.setattr(module, "validate_matrix_results", lambda **_: {})

    module.main()

    assert captured["expected_uv_executable"] == profile_uv
    assert captured["expected_uv_executable"] != Path(runtime_payload["uv_executable"])


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
        "capability_evidence": _te_capability_evidence(),
        "topology": {
            "world_size": 8,
            "num_nodes": 2,
            "gpus_per_node": 4,
            "joined_ranks": list(range(8)),
            "device_bindings": _device_bindings(),
        },
        "test_row_id": "te_eval_capability_8",
        "node_results": TE_EVAL_NODES,
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
    payload["node_results"] = []
    (candidate_dir / "te_eval_capability_8.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pytest node results"):
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
    payload["node_results"] = TE_EVAL_NODES
    (candidate_dir / "te_eval_capability_8.json").write_text(json.dumps(payload))
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("all_eval_callables_supported", False),
        ("outputs_changed", False),
        ("no_parameter_grads", False),
        ("replay_forward_counter_increment", 1),
        ("raw_te_eval_reuse_eager_parity", False),
        ("raw_te_eval_reuse_replay_forward_counter_increment", 1),
    ),
)
def test_matrix_validator_rejects_false_te_capability_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _load_module()
    candidate_sha = "f" * 40
    candidate_dir = tmp_path / "mcore" / candidate_sha
    candidate_dir.mkdir(parents=True)
    evidence = {**_te_capability_evidence(), field: value}
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
        "capability_evidence": evidence,
        "topology": {
            "world_size": 8,
            "num_nodes": 2,
            "gpus_per_node": 4,
            "joined_ranks": list(range(8)),
            "device_bindings": _device_bindings(),
        },
        "test_row_id": "te_eval_capability_8",
        "node_results": TE_EVAL_NODES,
    }
    (candidate_dir / "te_eval_capability_8.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="capability evidence"):
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


def test_matrix_validator_rejects_missing_te_capability_evidence(
    tmp_path: Path,
) -> None:
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
        "node_results": TE_EVAL_NODES,
    }
    (candidate_dir / "te_eval_capability_8.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="capability evidence"):
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
        "capability_evidence": _te_capability_evidence(),
        "topology": {
            "world_size": 8,
            "num_nodes": 2,
            "gpus_per_node": 4,
            "joined_ranks": list(range(8)),
            "device_bindings": bindings,
        },
        "test_row_id": "te_eval_capability_8",
        "node_results": TE_EVAL_NODES,
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
