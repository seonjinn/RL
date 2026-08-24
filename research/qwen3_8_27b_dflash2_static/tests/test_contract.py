from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(name: str) -> dict[str, object]:
    with (EXPERIMENT_ROOT / name).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def _load_preflight():
    path = EXPERIMENT_ROOT / "preflight.py"
    spec = importlib.util.spec_from_file_location("dflash2_static_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_and_dflash2_hold_target_and_workload_constant() -> None:
    baseline = _load_yaml("baseline.yaml")
    dflash2 = _load_yaml("dflash2.yaml")

    assert baseline["model"] == dflash2["model"] == "Qwen/Qwen3.8-27B"
    assert baseline["workload"] == dflash2["workload"]
    assert baseline["engine"] == dflash2["engine"]
    assert "speculative_config" not in baseline


def test_dflash2_arm_uses_the_published_static_vllm_contract() -> None:
    config = _load_yaml("dflash2.yaml")

    assert config["speculative_config"] == {
        "method": "dflash",
        "model": "incoai/Qwen3.8-27B-DFlash2",
        "num_speculative_tokens": 7,
    }
    assert config["mode"] == "static_rollout"
    assert config["draft_refit"] is False
    assert config["online_draft_training"] is False


def test_preflight_rejects_the_current_nemo_rl_vllm_pin() -> None:
    preflight = _load_preflight()

    with pytest.raises(RuntimeError, match="DFlash2-capable vLLM"):
        preflight.validate_runtime(
            vllm_version="0.25.1",
            has_dflash2_capability=False,
            uses_v2_runner=False,
        )


def test_preflight_accepts_a_capable_v2_runtime() -> None:
    preflight = _load_preflight()

    preflight.validate_runtime(
        vllm_version="0.26.0.dev0",
        has_dflash2_capability=True,
        uses_v2_runner=True,
    )
