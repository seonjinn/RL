import ast
import importlib.metadata
from types import SimpleNamespace

import pytest
import torch

from experiments.mxfp8_moe_tactic_audit.flashinfer_adapter import (
    MoeKernelCase,
    assert_supported_flashinfer,
    cache_key_for_case,
    enumerate_valid_tactics,
    force_stock_tactic,
    force_tactic,
    normalize_tactic_pair,
)
from experiments.mxfp8_moe_tactic_audit.schema import (
    ReplayProfile,
    RoutingSignature,
    TacticPair,
)


def _profile() -> ReplayProfile:
    signature = RoutingSignature(
        schema_version=1,
        model_revision="qwen3-30ba3b-test",
        layer_family="routed_experts",
        num_tokens=23,
        global_num_experts=4,
        local_num_experts=4,
        top_k=2,
        hidden_size=64,
        intermediate_size=32,
        expert_counts=(12, 12, 11, 11),
        sampled_gpu_time_us=17.5,
        tp_size=1,
        ep_size=1,
        dp_size=16,
        cuda_graph_state="trace-eager",
        weight_layout="MajorK",
        quantization="MXFP8",
        runtime_fingerprint="runtime-sha256",
    )
    return ReplayProfile.from_signature(signature, weight=1.0)


def _case() -> MoeKernelCase:
    profile = _profile()
    return MoeKernelCase(
        profile=profile,
        hidden_states=torch.empty((23, 64)),
        hidden_states_scale=torch.empty((23, 2)),
        gemm1_weights=torch.empty((4, 64, 64)),
        gemm1_weights_scale=torch.empty((4, 2, 2)),
        gemm2_weights=torch.empty((4, 64, 32)),
        gemm2_weights_scale=torch.empty((4, 2, 1)),
        output=torch.empty((23, 64), dtype=torch.bfloat16),
        activation_type=3,
        routing_method_type=4,
        local_expert_offset=0,
    )


def test_adapter_rejects_unpinned_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.6.14")

    with pytest.raises(RuntimeError, match="requires FlashInfer 0.6.13"):
        assert_supported_flashinfer()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [([17, 23], TacticPair(17, 23)), ((17, 23), TacticPair(17, 23))],
)
def test_normalize_tactic_pair(raw: object, expected: TacticPair) -> None:
    assert normalize_tactic_pair(raw) == expected


@pytest.mark.parametrize("raw", [[17], [17, 23, 42], [-1, 23], [True, 23], "17,23"])
def test_normalize_tactic_pair_rejects_invalid_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_tactic_pair(raw)


def test_force_tactic_restores_only_audit_cache_state_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = SimpleNamespace(
        _file_configs={"existing": ("OtherRunner", 7)},
        profiling_cache={("existing",): (0, 7, None)},
        untouched={"sentinel": object()},
    )
    original_untouched = tuner.untouched
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )
    cache_key = str(
        (
            "flashinfer::trtllm_fp8_block_scale_moe",
            "MoERunner",
            ((16, 2048),),
            (),
        )
    )

    with pytest.raises(RuntimeError, match="tactic crashed"):
        with force_tactic(cache_key, TacticPair(17, 23)):
            assert tuner._file_configs == {cache_key: ("MoERunner", [17, 23])}
            assert tuner.profiling_cache == {}
            raise RuntimeError("tactic crashed")

    assert tuner._file_configs == {"existing": ("OtherRunner", 7)}
    assert tuner.profiling_cache == {("existing",): (0, 7, None)}
    assert tuner.untouched is original_untouched


def test_force_tactic_rejects_a_key_for_an_unrelated_operation() -> None:
    cache_key = str(("flashinfer::other", "MoERunner", ((16, 2048),), ()))

    with pytest.raises(ValueError, match="exact FlashInfer MoE file key"):
        with force_tactic(cache_key, TacticPair(17, 23)):
            pass


def test_force_stock_tactic_inserts_literal_fallback_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = SimpleNamespace(
        _file_configs={"existing": ("OtherRunner", 7)},
        profiling_cache={("existing",): (0, 7, None)},
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )
    cache_key = str(
        (
            "flashinfer::trtllm_fp8_block_scale_moe",
            "MoERunner",
            ((16, 2048),),
            (),
        )
    )

    with force_stock_tactic(cache_key):
        assert tuner._file_configs == {cache_key: ("MoERunner", [-1, -1])}
        assert tuner.profiling_cache == {}

    assert tuner._file_configs == {"existing": ("OtherRunner", 7)}
    assert tuner.profiling_cache == {("existing",): (0, 7, None)}


def test_enumerate_valid_tactics_uses_mxfp8_trtllm_gen_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    moe_op = SimpleNamespace(
        trtllm_get_valid_moe_configs=lambda *args: (
            calls.append(args) or [[17, 23], (5, 11), [17, 23]]
        )
    )
    module_factory = lambda: SimpleNamespace(build_and_load=lambda: moe_op)
    dtype_enum = SimpleNamespace(MxE4m3="mxe4m3")
    fp8_enum = SimpleNamespace(MxFp8="mxfp8")
    weight_layout = SimpleNamespace(MajorK=SimpleNamespace(value=0))
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._load_flashinfer_bindings",
        lambda: (module_factory, dtype_enum, fp8_enum, weight_layout),
    )

    tactics = enumerate_valid_tactics(_case())

    assert tactics == (TacticPair(5, 11), TacticPair(17, 23))
    assert calls == [
        (
            "mxe4m3",
            "mxe4m3",
            "mxfp8",
            2,
            64,
            32,
            4,
            3,
            True,
            0,
            False,
            23,
            False,
        )
    ]


def test_cache_key_for_case_matches_moe_input_profile_layout() -> None:
    case = _case()

    final_key = cache_key_for_case(case, has_gemm1_lora_delta=False)
    intermediate_key = cache_key_for_case(case, has_gemm1_lora_delta=True)

    final = ast.literal_eval(final_key)
    intermediate = ast.literal_eval(intermediate_key)
    assert final[:2] == (
        "flashinfer::trtllm_fp8_block_scale_moe",
        "MoERunner",
    )
    assert final[2] == (
        (16, 64),
        (0,),
        (16, 2),
        (0,),
        (16, 64),
        (16, 2),
        (0,),
        (0,),
    )
    assert intermediate[2][6] == (16, 2, 64)
    assert final[3] == intermediate[3] == ()
