#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
from typing import TypedDict


_DFLASH2_MODULES = (
    "vllm.model_executor.models.qwen3_dflash2",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator",
)
_V2_TRUE_VALUES = frozenset({"1", "true", "yes"})


class RuntimeFingerprint(TypedDict):
    vllm_version: str
    dflash2_modules: dict[str, bool]
    v2_model_runner: bool
    python_version: str


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
    return RuntimeFingerprint(
        vllm_version=version,
        dflash2_modules=modules,
        v2_model_runner=uses_v2_runner,
        python_version=platform.python_version(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-current-pin-to-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    version = importlib.metadata.version("vllm")
    try:
        fingerprint = runtime_fingerprint()
    except RuntimeError:
        if args.expect_current_pin_to_fail and version == "0.25.1":
            print("DFLASH2_RUNTIME_PREFLIGHT=EXPECTED_PIN_REJECTION")
            return
        raise
    if args.expect_current_pin_to_fail:
        raise RuntimeError(
            "expected the NeMo-RL vLLM 0.25.1 pin, but found a capable runtime"
        )
    if args.json:
        print(json.dumps(fingerprint, sort_keys=True))
    else:
        print(f"DFLASH2_RUNTIME_PREFLIGHT=PASS,vllm={version},runner=v2")


if __name__ == "__main__":
    main()
