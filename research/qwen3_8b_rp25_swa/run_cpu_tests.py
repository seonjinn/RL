# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks; bypass GPU package initializers, not tested implementations.

Run with uv --no-project and torch, numpy, pytest, pydantic, omegaconf,
megatron-core installed. This is not the pinned-container integration gate.
"""

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _namespace(name: str, path: Path) -> None:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)


def _install_cpu_namespaces() -> None:
    megatron = importlib.import_module("megatron")
    megatron_root = Path(next(iter(megatron.__path__)))
    for suffix in ("core", "core.transformer", "core.tensor_parallel"):
        _namespace(f"megatron.{suffix}", megatron_root / suffix.replace(".", "/"))
    for name in (
        "nemo_rl",
        "nemo_rl.models",
        "nemo_rl.models.megatron",
        "nemo_rl.models.megatron.draft",
        "nemo_rl.models.policy",
    ):
        _namespace(name, REPO_ROOT / name.replace(".", "/"))


# Spawned CPU distributed-test workers need the same dependency boundary.
_install_cpu_namespaces()


if __name__ == "__main__":
    print(
        "CPU-only checks: package initialization bypassed; GPU integration not covered"
    )
    tests = [
        "tests/unit/models/policy/test_draft_attention_config.py",
        "tests/unit/models/megatron/test_dflash_model.py",
        "tests/unit/models/megatron/test_dflash_block_attention.py",
        "tests/unit/models/megatron/test_dflash_cp_attention.py",
        "tests/unit/models/megatron/test_dflash_block_plan.py",
        "tests/unit/models/megatron/test_dspark_cp_plan.py",
        "tests/unit/models/megatron/test_dspark_provider.py",
    ]
    raise SystemExit(
        pytest.main(
            [
                "--noconftest",
                "-q",
                "--tb=short",
                "--maxfail=0",
                "-k",
                "not cuda and not benchmark and not default_",
                *[str(REPO_ROOT / test) for test in tests],
            ]
        )
    )
