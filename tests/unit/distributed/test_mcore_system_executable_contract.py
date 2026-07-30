import importlib
import sys
from pathlib import Path

import pytest

import nemo_rl.distributed.ray_actor_environment_registry as registry


@pytest.fixture(autouse=True)
def reset_mcore_system_contract(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEMO_RL_REQUIRE_SYSTEM_MCORE", raising=False)
    monkeypatch.delenv("NEMO_RL_MCORE_SYSTEM_PYTHON", raising=False)
    importlib.reload(registry)
    yield
    monkeypatch.delenv("NEMO_RL_REQUIRE_SYSTEM_MCORE", raising=False)
    monkeypatch.delenv("NEMO_RL_MCORE_SYSTEM_PYTHON", raising=False)
    importlib.reload(registry)


def test_mcore_uses_its_normal_uv_environment_without_the_experiment_contract() -> None:
    reloaded = importlib.reload(registry)

    assert (
        reloaded.ACTOR_ENVIRONMENT_REGISTRY[
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        ]
        == reloaded.PY_EXECUTABLES.MCORE
    )


def test_mcore_experiment_contract_requires_an_expected_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_RL_REQUIRE_SYSTEM_MCORE", "1")

    with pytest.raises(RuntimeError, match="NEMO_RL_MCORE_SYSTEM_PYTHON"):
        importlib.reload(registry)


def test_mcore_experiment_contract_rejects_a_mismatched_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_python = tmp_path / "expected" / "bin" / "python"
    expected_python.parent.mkdir(parents=True)
    expected_python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setenv("NEMO_RL_REQUIRE_SYSTEM_MCORE", "1")
    monkeypatch.setenv("NEMO_RL_MCORE_SYSTEM_PYTHON", str(expected_python))

    with pytest.raises(RuntimeError, match="must match sys.executable"):
        importlib.reload(registry)


def test_mcore_experiment_contract_uses_the_exact_system_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_python = tmp_path / "locked-mcore" / "bin" / "python"
    expected_python.parent.mkdir(parents=True)
    expected_python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr(sys, "executable", str(expected_python))
    monkeypatch.setattr(sys, "prefix", str(expected_python.parent.parent))
    monkeypatch.setenv("NEMO_RL_REQUIRE_SYSTEM_MCORE", "1")
    monkeypatch.setenv("NEMO_RL_MCORE_SYSTEM_PYTHON", str(expected_python))

    reloaded = importlib.reload(registry)

    assert (
        reloaded.ACTOR_ENVIRONMENT_REGISTRY[
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        ]
        == reloaded.PY_EXECUTABLES.SYSTEM
    )
    assert (
        reloaded.ACTOR_ENVIRONMENT_REGISTRY[
            "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
        ]
        == reloaded.PY_EXECUTABLES.VLLM
    )
