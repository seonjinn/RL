import json
import math
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.models.generation import validate_vllm_speculative_config
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.sd_toggle.config import (
    CalibrationConfig,
    HardwareConfig,
    ModelConfig,
    PerGammaCalibration,
    SDToggleConfig,
)
from nemo_rl.models.generation.vllm import tail_gate as tail_gate_module
from nemo_rl.models.generation.vllm.tail_gate import (
    TailGateConfig,
    TailGateController,
    TailGateObservation,
)


def make_roofline_gate(expected_accept_length: float = 3.0) -> TailGateConfig:
    roofline_config = SDToggleConfig(
        hardware=HardwareConfig(gpu="test", tp=1, BW_eff=1.0e12),
        model=ModelConfig(
            name="test-model",
            W_t=3.0e10,
            W_d=1.0e9,
            C_dense=1.0e11,
            C_attn=1.0e7,
            kappa_theoretical=1,
        ),
        calibration=CalibrationConfig(
            eta_d=1.0,
            kappa_eff=1.0,
            F_eff=1.0e15,
            per_gamma={
                5: PerGammaCalibration(c_T=1.0, c_D=1.0, c_V=1.0),
            },
        ),
    )
    return TailGateConfig(
        mode="roofline",
        threshold=32,
        consecutive_checks=3,
        gamma=5,
        roofline_config=roofline_config,
        expected_accept_length=expected_accept_length,
    )


def test_threshold_gate_requires_ramp_and_consecutive_checks():
    gate = TailGateController(
        TailGateConfig(mode="threshold", threshold=32, consecutive_checks=3, gamma=5)
    )
    assert not gate.observe(TailGateObservation(8, 2048, True)).enabled
    assert not gate.observe(TailGateObservation(64, 2048, True)).enabled
    assert not gate.observe(TailGateObservation(32, 4096, True)).enabled
    assert not gate.observe(TailGateObservation(31, 4097, True)).enabled
    decision = gate.observe(TailGateObservation(30, 4098, True))
    assert decision.enabled
    assert decision.just_activated
    assert gate.observe(TailGateObservation(64, 4099, True)).enabled


def test_gate_reset_keeps_previous_rollout_acceptance():
    gate = TailGateController(make_roofline_gate())
    gate.finish_rollout(accepted_tokens=60, num_drafts=20, validation=False)
    assert gate.expected_accept_length == 4.0
    assert not gate.enabled


def test_zero_cycle_does_not_replace_acceptance():
    gate = TailGateController(make_roofline_gate(expected_accept_length=2.5))
    gate.finish_rollout(accepted_tokens=0, num_drafts=0, validation=False)
    assert gate.expected_accept_length == 2.5


def test_validation_rollout_does_not_replace_acceptance():
    gate = TailGateController(make_roofline_gate(expected_accept_length=2.5))
    gate.finish_rollout(accepted_tokens=100, num_drafts=20, validation=True)
    assert gate.expected_accept_length == 2.5


def test_roofline_prediction_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = TailGateController(make_roofline_gate())
    gate.observe(TailGateObservation(64, 2048, True))

    def fail_prediction(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid roofline prediction")

    monkeypatch.setattr(tail_gate_module, "predict_decision", fail_prediction)

    with pytest.raises(ValueError, match="invalid roofline prediction"):
        gate.observe(TailGateObservation(32, 4096, True))


def _write_roofline_config(
    path: Path,
    *,
    gamma: int = 5,
    target_tp: int = 1,
    model_updates: dict[str, object] | None = None,
) -> None:
    model: dict[str, object] = {
        "name": "test-model",
        "W_t": 3.0e10,
        "W_d": 1.0e9,
        "C_dense": 1.0e11,
        "C_attn": 1.0e7,
        "kappa_theoretical": 1,
    }
    model.update(model_updates or {})
    payload = {
        "hardware": {"gpu": "test", "tp": target_tp, "BW_eff": 1.0e12},
        "model": model,
        "calibration": {
            "eta_d": 1.0,
            "kappa_eff": 1.0,
            "F_eff": 1.0e15,
            "per_gamma": {
                str(gamma): {"c_T": 1.0, "c_D": 1.0, "c_V": 1.0},
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _vllm_tail_gate_config(**updates: object) -> VllmConfig:
    speculative_config: dict[str, Any] = {
        "method": "eagle3",
        "model": "/tmp/eagle3-model",
        "num_speculative_tokens": 5,
        "sd_tail_gate_mode": "roofline",
        "sd_tail_gate_threshold": 32,
        "sd_tail_gate_consecutive_checks": 10,
        "sd_tail_gate_margin": 0.05,
        "sd_tail_gate_off_mode": "advance_only",
    }
    speculative_config.update(updates)
    return {
        "backend": "vllm",
        "model_name": "test-model",
        "max_new_tokens": 32,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "skip_tokenizer_init": True,
            "async_engine": False,
            "kv_cache_dtype": "auto",
        },
        "vllm_kwargs": {"speculative_config": speculative_config},
    }


def _validate_tail_gate_config(config: VllmConfig) -> None:
    validate_vllm_speculative_config(
        config,
        has_refit_draft_weights=False,
        is_eval=False,
    )


def test_roofline_validation_loads_calibration_before_engine_creation(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(missing_path))

    with pytest.raises(OSError):
        _validate_tail_gate_config(config)

    assert "scheduler_cls" not in config["vllm_kwargs"]


def test_roofline_validation_rejects_malformed_calibration(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "malformed.json"
    calibration_path.write_text("{", encoding="utf-8")
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(calibration_path))

    with pytest.raises(json.JSONDecodeError):
        _validate_tail_gate_config(config)


def test_roofline_validation_requires_exact_k_fit(tmp_path: Path) -> None:
    calibration_path = tmp_path / "k3.json"
    _write_roofline_config(calibration_path, gamma=3)
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(calibration_path))

    with pytest.raises(ValueError, match="exact K=5 fit"):
        _validate_tail_gate_config(config)


@pytest.mark.parametrize("margin", [math.nan, math.inf, -0.01, "invalid"])
def test_roofline_validation_rejects_invalid_margin(
    tmp_path: Path,
    margin: object,
) -> None:
    calibration_path = tmp_path / "roofline.json"
    _write_roofline_config(calibration_path)
    config = _vllm_tail_gate_config(
        sd_tail_gate_config_path=str(calibration_path),
        sd_tail_gate_margin=margin,
    )

    with pytest.raises(ValueError, match="margin"):
        _validate_tail_gate_config(config)


def test_tail_gate_validation_rejects_unsupported_off_mode(tmp_path: Path) -> None:
    calibration_path = tmp_path / "roofline.json"
    _write_roofline_config(calibration_path)
    config = _vllm_tail_gate_config(
        sd_tail_gate_config_path=str(calibration_path),
        sd_tail_gate_off_mode="skip",
    )

    with pytest.raises(ValueError, match="sd_tail_gate_off_mode=advance_only"):
        _validate_tail_gate_config(config)


def test_roofline_validation_rejects_invalid_model_data(tmp_path: Path) -> None:
    calibration_path = tmp_path / "invalid-model.json"
    _write_roofline_config(calibration_path, model_updates={"W_t": 0.0})
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(calibration_path))

    with pytest.raises(ValueError, match="model.W_t"):
        _validate_tail_gate_config(config)


def test_roofline_validation_rejects_target_model_mismatch(tmp_path: Path) -> None:
    calibration_path = tmp_path / "wrong-model.json"
    _write_roofline_config(
        calibration_path,
        model_updates={"name": "different-model"},
    )
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(calibration_path))

    with pytest.raises(ValueError, match="target model"):
        _validate_tail_gate_config(config)


def test_roofline_validation_rejects_target_tp_mismatch(tmp_path: Path) -> None:
    calibration_path = tmp_path / "wrong-tp.json"
    _write_roofline_config(calibration_path, target_tp=2)
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(calibration_path))

    with pytest.raises(ValueError, match="tensor parallel size"):
        _validate_tail_gate_config(config)


def test_roofline_validation_accepts_exact_finite_calibration(tmp_path: Path) -> None:
    calibration_path = tmp_path / "roofline.json"
    _write_roofline_config(calibration_path)
    config = _vllm_tail_gate_config(sd_tail_gate_config_path=str(calibration_path))

    _validate_tail_gate_config(config)

    assert config["vllm_kwargs"]["scheduler_cls"].endswith("TailGatedScheduler")


@pytest.mark.parametrize("mode", ["off", "threshold"])
def test_non_roofline_modes_do_not_load_calibration(mode: str) -> None:
    config = _vllm_tail_gate_config(
        sd_tail_gate_mode=mode,
        sd_tail_gate_margin=math.nan,
        sd_tail_gate_config_path="/missing/calibration.json",
    )

    _validate_tail_gate_config(config)

    if mode == "off":
        assert "scheduler_cls" not in config["vllm_kwargs"]
    else:
        assert config["vllm_kwargs"]["scheduler_cls"].endswith("TailGatedScheduler")
