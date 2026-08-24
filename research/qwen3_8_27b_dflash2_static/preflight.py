from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os


_DFLASH2_MODULES = (
    "vllm.model_executor.models.qwen3_dflash2",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator",
)


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
        raise RuntimeError("DFlash2 requires the vLLM V2 model runner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-current-pin-to-fail", action="store_true")
    args = parser.parse_args()

    version = importlib.metadata.version("vllm")
    has_dflash2_capability = all(_module_exists(name) for name in _DFLASH2_MODULES)
    uses_v2_runner = os.environ.get("VLLM_USE_V1", "0").lower() not in {
        "1",
        "true",
        "yes",
    }
    try:
        validate_runtime(
            vllm_version=version,
            has_dflash2_capability=has_dflash2_capability,
            uses_v2_runner=uses_v2_runner,
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
    print(f"DFLASH2_RUNTIME_PREFLIGHT=PASS,vllm={version},runner=v2")


if __name__ == "__main__":
    main()
