import importlib.util
import sys
import types
from pathlib import Path

import pytest


REGISTRY_PATH = (
    Path(__file__).parents[3]
    / "nemo_rl"
    / "distributed"
    / "ray_actor_environment_registry.py"
)


class _PythonExecutables:
    SYSTEM = "/system/bin/python"
    VLLM = "uv run --extra vllm"
    SGLANG = "uv run --extra sglang"
    MCORE = "uv run --extra mcore"
    TRTLLM = "uv run --extra trtllm"
    FSDP = "uv run --extra fsdp"
    AUTOMODEL = "uv run --extra automodel"
    NEMO_GYM = "uv run --extra nemo_gym"


def _load_registry(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    virtual_cluster = types.ModuleType("nemo_rl.distributed.virtual_cluster")
    virtual_cluster.PY_EXECUTABLES = _PythonExecutables
    modelopt_registry = types.ModuleType("nemo_rl.modelopt.registry")
    modelopt_registry.MODELOPT_ACTOR_REGISTRY = {}
    monkeypatch.setitem(
        sys.modules, "nemo_rl.distributed.virtual_cluster", virtual_cluster
    )
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.registry", modelopt_registry)
    spec = importlib.util.spec_from_file_location(
        "_test_ray_actor_environment_registry", REGISTRY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tier_specific_python_executables_override_uv_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", raising=False)
    monkeypatch.setenv(
        "NEMO_RL_MCORE_PY_EXECUTABLE", "/runtime/mcore-environment/bin/python"
    )
    monkeypatch.setenv(
        "NEMO_RL_VLLM_PY_EXECUTABLE", "/runtime/vllm-environment/bin/python"
    )

    registry = _load_registry(monkeypatch)

    assert registry.MCORE_EXECUTABLE == "/runtime/mcore-environment/bin/python"
    assert registry.VLLM_EXECUTABLE == "/runtime/vllm-environment/bin/python"
    assert registry.SGLANG_EXECUTABLE == _PythonExecutables.SGLANG
    for actor in (
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
        "nemo_rl.algorithms.async_utils.ReplayBuffer",
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
    ):
        assert (
            registry.ACTOR_ENVIRONMENT_REGISTRY[actor]
            == "/runtime/vllm-environment/bin/python"
        )


@pytest.mark.parametrize(
    "variable",
    ("NEMO_RL_MCORE_PY_EXECUTABLE", "NEMO_RL_VLLM_PY_EXECUTABLE"),
)
def test_tier_specific_python_executable_must_be_absolute(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.delenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", raising=False)
    monkeypatch.setenv(variable, "relative/bin/python")

    with pytest.raises(ValueError, match=f"{variable} must be an absolute path"):
        _load_registry(monkeypatch)
