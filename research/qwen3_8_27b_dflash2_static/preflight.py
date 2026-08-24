#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Mapping
from typing import TypedDict


_DFLASH2_MODULES = (
    "vllm.model_executor.models.qwen3_dflash2",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator",
)
_V2_TRUE_VALUES = frozenset({"1", "true", "yes"})
_EXPECTED_IMAGE_ENVIRONMENT = {
    "NRL_VLLM_SOURCE_COMMIT": "f94666b60d4c58ec0807d22c837cfae322a1dde9",
    "NRL_VLLM_SOURCE_INDEX_DIGEST": (
        "sha256:f50b406f696712019a673e317a0db6e029c430cf81ec7bdea2ebd7111e55aef7"
    ),
    "NRL_VLLM_SOURCE_ARM64_DIGEST": (
        "sha256:4db6d42b66ad393faa3da7341db580f443b7aeb9a7de5597cd11b724eabff6f6"
    ),
    "NRL_DFLASH2_MERGE_ANCESTOR": "b389ac29465b33f9e9c534df221ea3c129e9793f",
}


class RuntimeFingerprint(TypedDict):
    vllm_version: str
    dflash2_modules: dict[str, bool]
    v2_model_runner: bool
    python_version: str
    image_contract: dict[str, str]


class NemoActorFingerprint(TypedDict):
    runtime: RuntimeFingerprint
    actor_executable: str
    system_executable: str
    actor_module_path: str
    vllm_module_path: str


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def validate_runtime(
    *,
    vllm_version: str,
    has_dflash2_capability: bool,
    uses_v2_runner: bool,
) -> None:
    if not has_dflash2_capability:
        raise RuntimeError(f"vLLM {vllm_version} is not a DFlash2-capable vLLM runtime")
    if not uses_v2_runner:
        raise RuntimeError(
            "DFlash2 requires VLLM_USE_V2_MODEL_RUNNER=1 and the vLLM V2 model runner"
        )


def validate_image_contract_environment(
    environ: Mapping[str, str],
) -> dict[str, str]:
    values = {name: environ.get(name, "") for name in _EXPECTED_IMAGE_ENVIRONMENT}
    mismatches = {
        name: value
        for name, value in values.items()
        if value != _EXPECTED_IMAGE_ENVIRONMENT[name]
    }
    if mismatches:
        raise RuntimeError(
            "runtime image contract is missing or mismatched: "
            + ", ".join(sorted(mismatches))
        )
    return values


def runtime_fingerprint() -> RuntimeFingerprint:
    """Inspect the installed vLLM without importing its CUDA runtime."""
    version = importlib.metadata.version("vllm")
    modules = {name: _module_exists(name) for name in _DFLASH2_MODULES}
    uses_v2_runner = (
        os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").lower() in _V2_TRUE_VALUES
    )
    validate_runtime(
        vllm_version=version,
        has_dflash2_capability=all(modules.values()),
        uses_v2_runner=uses_v2_runner,
    )
    image_contract = validate_image_contract_environment(os.environ)
    return RuntimeFingerprint(
        vllm_version=version,
        dflash2_modules=modules,
        v2_model_runner=uses_v2_runner,
        python_version=platform.python_version(),
        image_contract=image_contract,
    )


def validate_nemo_actor_runtime(
    *,
    actor_executable: str,
    system_executable: str,
    actor_module_path: str,
    vllm_module_path: str,
) -> None:
    if actor_executable != system_executable:
        raise RuntimeError(
            "NeMo-RL VllmGenerationWorker must use the current system interpreter"
        )
    if not actor_module_path or "nemo_rl/models/generation/vllm/vllm_worker.py" not in (
        actor_module_path
    ):
        raise RuntimeError("actual NeMo-RL VllmGenerationWorker was not imported")
    if not vllm_module_path or "/vllm/__init__.py" not in vllm_module_path:
        raise RuntimeError("actual vLLM package was not imported")


def nemo_actor_fingerprint() -> NemoActorFingerprint:
    """Import the exact NeMo-RL actor and prove which vLLM interpreter it uses."""
    import vllm

    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.models.generation.vllm import vllm_worker

    actor_fqn = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    actor_executable = ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]
    actor_module_path = str(vllm_worker.__file__ or "")
    vllm_module_path = str(vllm.__file__ or "")
    validate_nemo_actor_runtime(
        actor_executable=actor_executable,
        system_executable=sys.executable,
        actor_module_path=actor_module_path,
        vllm_module_path=vllm_module_path,
    )
    return NemoActorFingerprint(
        runtime=runtime_fingerprint(),
        actor_executable=actor_executable,
        system_executable=sys.executable,
        actor_module_path=actor_module_path,
        vllm_module_path=vllm_module_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-current-pin-to-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--nemo-actor-json", action="store_true")
    args = parser.parse_args()

    version = importlib.metadata.version("vllm")
    try:
        fingerprint = (
            nemo_actor_fingerprint() if args.nemo_actor_json else runtime_fingerprint()
        )
    except RuntimeError:
        if args.expect_current_pin_to_fail and version == "0.25.1":
            print("DFLASH2_RUNTIME_PREFLIGHT=EXPECTED_PIN_REJECTION")
            return
        raise
    if args.expect_current_pin_to_fail:
        raise RuntimeError(
            "expected the NeMo-RL vLLM 0.25.1 pin, but found a capable runtime"
        )
    if args.json or args.nemo_actor_json:
        print(json.dumps(fingerprint, sort_keys=True))
    else:
        print(f"DFLASH2_RUNTIME_PREFLIGHT=PASS,vllm={version},runner=v2")


if __name__ == "__main__":
    main()
