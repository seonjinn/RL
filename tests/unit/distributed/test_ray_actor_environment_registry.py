import importlib

import pytest

import nemo_rl.distributed.ray_actor_environment_registry as registry
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


_ENV_KEYS = (
    "NEMO_RL_VLLM_PY_EXECUTABLE",
    "NEMO_RL_SGLANG_PY_EXECUTABLE",
    "NEMO_RL_MCORE_PY_EXECUTABLE",
    "NEMO_RL_ASYNC_UTILS_PY_EXECUTABLE",
)


@pytest.fixture(autouse=True)
def restore_registry(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(registry)
    yield
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(registry)


def test_actor_registry_uses_default_executables_without_env_overrides():
    reloaded = importlib.reload(registry)

    assert (
        reloaded.get_actor_python_env(
            "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
        )
        == PY_EXECUTABLES.VLLM
    )
    assert (
        reloaded.get_actor_python_env(
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        )
        == PY_EXECUTABLES.MCORE
    )
    assert (
        reloaded.get_actor_python_env(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
        )
        == PY_EXECUTABLES.VLLM
    )


def test_actor_registry_uses_env_overridden_executables(monkeypatch):
    monkeypatch.setenv("NEMO_RL_VLLM_PY_EXECUTABLE", "/opt/ray_venvs/vllm/bin/python")
    monkeypatch.setenv("NEMO_RL_MCORE_PY_EXECUTABLE", "/opt/ray_venvs/mcore/bin/python")
    monkeypatch.setenv(
        "NEMO_RL_ASYNC_UTILS_PY_EXECUTABLE", "/opt/ray_venvs/async/bin/python"
    )

    reloaded = importlib.reload(registry)

    assert (
        reloaded.get_actor_python_env(
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
        )
        == "/opt/ray_venvs/vllm/bin/python"
    )
    assert (
        reloaded.get_actor_python_env(
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        )
        == "/opt/ray_venvs/mcore/bin/python"
    )
    assert (
        reloaded.get_actor_python_env(
            "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
        )
        == "/opt/ray_venvs/async/bin/python"
    )
