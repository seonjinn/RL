"""Integration tests for the custom vLLM Ray environment contract."""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


MXFP8_CONFIG_ENV_VAR = "VLLM_MXFP8_DENSE_CONFIG_FILE"
MXFP8_CONFIG_NAME = "qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json"


class _EnvModule(ModuleType):
    VLLM_CONFIG_ROOT = str(Path(__file__).with_name(".missing-vllm-config"))
    environment_variables: dict[str, object] = {}

    def __getattr__(self, name: str) -> str:
        if name in {
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY",
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
        }:
            return os.getenv(name, "")
        raise AttributeError(name)


def _custom_vllm_root() -> Path:
    repository_root = Path(__file__).resolve().parents[4]
    configured_root = os.environ.get("NEMO_RL_CUSTOM_VLLM_SOURCE")
    if configured_root is not None:
        return Path(configured_root)
    return repository_root.parent / "vllm-v0202-mxfp8-adaptive-nemorl"


def _load_custom_vllm_ray_env() -> ModuleType:
    module_path = _custom_vllm_root() / "vllm/ray/ray_env.py"
    if not module_path.is_file():
        pytest.skip(
            "custom vLLM 0.20.2 checkout not found; set NEMO_RL_CUSTOM_VLLM_SOURCE"
        )

    spec = importlib.util.spec_from_file_location(
        "_standalone_custom_vllm_ray_env", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load custom vLLM Ray environment module: {module_path}"
        )

    fake_vllm = ModuleType("vllm")
    fake_envs = _EnvModule("vllm.envs")
    fake_logger = ModuleType("vllm.logger")
    fake_logger.init_logger = logging.getLogger  # type: ignore[attr-defined]
    original_modules = {
        name: sys.modules.get(name) for name in ("vllm", "vllm.envs", "vllm.logger")
    }
    try:
        sys.modules["vllm"] = fake_vllm
        sys.modules["vllm.envs"] = fake_envs
        sys.modules["vllm.logger"] = fake_logger
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_custom_vllm_forwards_mxfp8_config_via_native_vllm_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv(MXFP8_CONFIG_ENV_VAR, MXFP8_CONFIG_NAME)
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", raising=False)
    ray_env = _load_custom_vllm_ray_env()

    assert "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY" not in os.environ
    env_vars_to_copy = ray_env.get_env_vars_to_copy()
    copied_env = {
        name: os.environ[name] for name in env_vars_to_copy if name in os.environ
    }
    assert copied_env[MXFP8_CONFIG_ENV_VAR] == MXFP8_CONFIG_NAME
