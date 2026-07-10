import json
from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm.sd_toggle import load_config
from nemo_rl.models.generation.vllm.sd_toggle.predict import should_enable_sd
from nemo_rl.models.generation.vllm.sd_toggle.roofline import predict_speedup


def make_calibration(tmp_path: Path, *, BW_eff: float = 1.0e12):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "hardware": {"gpu": "test", "tp": 1, "BW_eff": BW_eff},
                "model": {
                    "name": "test-model",
                    "W_t": 3.0e10,
                    "W_d": 1.0e9,
                    "C_dense": 1.0e11,
                    "C_attn": 1.0e7,
                    "kappa_theoretical": 1,
                },
                "calibration": {
                    "eta_d": 1.0,
                    "kappa_eff": 1.0,
                    "F_eff": 1.0e15,
                    "per_gamma": {"5": {"c_D": 0.0, "c_V": 0.0}},
                },
            }
        )
    )
    return load_config(calibration_path)


def test_roofline_enables_only_above_margin(tmp_path: Path):
    config = make_calibration(tmp_path)
    speedup = predict_speedup(B=4, S=4096, gamma=5, L_accept=3.0, config=config)
    assert should_enable_sd(config, 4, 4096, 5, 3.0, margin=speedup - 1.0)
    assert not should_enable_sd(config, 4, 4096, 5, 3.0, margin=speedup - 1.0 + 1e-6)


def test_roofline_rejects_non_finite_prediction(tmp_path: Path):
    config = make_calibration(tmp_path, BW_eff=0.0)
    with pytest.raises(ValueError, match="finite"):
        should_enable_sd(config, 4, 4096, 5, 3.0, margin=0.05)
