from nemo_rl.models.generation.vllm.sd_toggle.config import (
    CalibrationConfig,
    HardwareConfig,
    ModelConfig,
    SDToggleConfig,
)
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
            c_D=1.0,
            c_V=1.0,
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
    gate.finish_rollout(accepted_tokens=40, num_drafts=20, validation=False)
    assert gate.expected_accept_length == 3.0
    assert not gate.enabled


def test_validation_rollout_does_not_replace_acceptance():
    gate = TailGateController(make_roofline_gate(expected_accept_length=2.5))
    gate.finish_rollout(accepted_tokens=100, num_drafts=20, validation=True)
    assert gate.expected_accept_length == 2.5
