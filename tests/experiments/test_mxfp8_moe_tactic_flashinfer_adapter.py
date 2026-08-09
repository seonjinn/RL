import ast
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mxfp8_moe_tactic_audit import flashinfer_adapter
from experiments.mxfp8_moe_tactic_audit.flashinfer_adapter import (
    PREPACKED_ARTIFACT_FORMAT,
    IntermediateApiUnavailable,
    MoeKernelCase,
    MoePairResult,
    TacticDispatchError,
    _prepare_synthetic_weights,
    assert_supported_flashinfer,
    cache_key_for_case,
    enumerate_valid_tactics,
    force_stock_tactic,
    force_tactic,
    load_prepacked_weights,
    normalize_tactic_pair,
    observed_forced_cache_event,
    run_moe_pair,
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
        weight_layout="MajorK",
        use_shuffled_weight=True,
        prepacked_weight_format=PREPACKED_ARTIFACT_FORMAT,
    )


def _cache_key() -> str:
    return str(
        (
            "flashinfer::trtllm_fp8_block_scale_moe",
            "MoERunner",
            ((16, 2048),),
            (),
        )
    )


def _artifact_payload() -> dict[str, object]:
    return {
        "metadata": {
            "format": PREPACKED_ARTIFACT_FORMAT,
            "flashinfer_version": "0.6.13",
            "model_revision": "qwen3-30ba3b-test",
            "quantization": "MXFP8",
            "weight_layout": "MajorK",
            "use_shuffled_weight": True,
            "activation": "SwiGLU",
            "gated_rows_reordered": True,
            "matrix_a_shuffled": True,
            "matrix_sf_a_shuffled": True,
            "global_num_experts": 4,
            "local_num_experts": 4,
            "hidden_size": 64,
            "intermediate_size": 32,
            "local_expert_offset": 0,
        },
        "gemm1_weights": torch.zeros((4, 64, 64), dtype=torch.float8_e4m3fn),
        "gemm1_weights_scale": torch.zeros((4, 64, 2), dtype=torch.uint8),
        "gemm2_weights": torch.zeros((4, 64, 32), dtype=torch.float8_e4m3fn),
        "gemm2_weights_scale": torch.zeros((4, 64, 1), dtype=torch.uint8),
    }


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
        _logged_file_hits={("other", "OtherRunner")},
        untouched={"sentinel": object()},
    )
    original_untouched = tuner.untouched
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )
    cache_key = _cache_key()

    with pytest.raises(RuntimeError, match="tactic crashed"):
        with force_tactic(cache_key, TacticPair(17, 23)):
            assert tuner._file_configs == {cache_key: ("MoERunner", [17, 23])}
            assert tuner.profiling_cache == {}
            raise RuntimeError("tactic crashed")

    assert tuner._file_configs == {"existing": ("OtherRunner", 7)}
    assert tuner.profiling_cache == {("existing",): (0, 7, None)}
    assert tuner._logged_file_hits == {("other", "OtherRunner")}
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
        _logged_file_hits=set(),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )
    cache_key = _cache_key()

    with force_stock_tactic(cache_key):
        assert tuner._file_configs == {cache_key: ("MoERunner", [-1, -1])}
        assert tuner.profiling_cache == {}
        tuner._logged_file_hits.add(
            ("flashinfer::trtllm_fp8_block_scale_moe", "MoERunner")
        )

    assert tuner._file_configs == {"existing": ("OtherRunner", 7)}
    assert tuner.profiling_cache == {("existing",): (0, 7, None)}
    assert tuner._logged_file_hits == set()


@pytest.mark.parametrize(
    ("stock_fallback", "expected"),
    [(False, "cache hit"), (True, "fallback")],
)
def test_cache_event_is_observed_from_active_runtime_dispatch(
    stock_fallback: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuner = SimpleNamespace(
        _file_configs={}, profiling_cache={}, _logged_file_hits=set()
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )
    cache_key = _cache_key()
    context = (
        force_stock_tactic(cache_key)
        if stock_fallback
        else force_tactic(cache_key, TacticPair(17, 23))
    )

    with context:
        tuner._logged_file_hits.add(
            ("flashinfer::trtllm_fp8_block_scale_moe", "MoERunner")
        )
        assert observed_forced_cache_event(cache_key) == expected


def test_force_tactic_rejects_missing_exact_file_hit_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = SimpleNamespace(
        _file_configs={"existing": ("OtherRunner", 7)},
        profiling_cache={("existing",): (0, 7, None)},
        _logged_file_hits={("other", "OtherRunner")},
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )

    with pytest.raises(TacticDispatchError, match="did not log an exact file hit"):
        with force_tactic(_cache_key(), TacticPair(17, 23)):
            pass

    assert tuner._file_configs == {"existing": ("OtherRunner", 7)}
    assert tuner.profiling_cache == {("existing",): (0, 7, None)}
    assert tuner._logged_file_hits == {("other", "OtherRunner")}


def test_force_tactic_rejects_wrong_tactic_hit_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = SimpleNamespace(
        _file_configs={"existing": ("OtherRunner", 7)},
        profiling_cache={("existing",): (0, 7, None)},
        _logged_file_hits=set(),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._get_autotuner",
        lambda: tuner,
    )
    cache_key = _cache_key()

    with pytest.raises(TacticDispatchError, match="wrong tactic"):
        with force_tactic(cache_key, TacticPair(17, 23)):
            tuner._file_configs[cache_key] = ("MoERunner", [5, 11])
            tuner._logged_file_hits.add(
                ("flashinfer::trtllm_fp8_block_scale_moe", "MoERunner")
            )

    assert tuner._file_configs == {"existing": ("OtherRunner", 7)}
    assert tuner.profiling_cache == {("existing",): (0, 7, None)}
    assert tuner._logged_file_hits == set()


def test_load_prepacked_weights_rejects_missing_preparation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact_payload()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    del metadata["matrix_sf_a_shuffled"]
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)

    with pytest.raises(ValueError, match="matrix_sf_a_shuffled"):
        load_prepacked_weights(Path("weights.pt"), _profile(), torch.device("cpu"))


def test_load_prepacked_weights_accepts_exact_kernel_ready_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact_payload()
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)

    prepared = load_prepacked_weights(
        Path("weights.pt"), _profile(), torch.device("cpu")
    )

    assert prepared.gemm1_weights is payload["gemm1_weights"]
    assert prepared.gemm1_weights_scale is payload["gemm1_weights_scale"]
    assert prepared.local_expert_offset == 0


def test_load_prepacked_weights_preserves_runtime_blocked_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact_payload()
    payload["gemm1_weights"] = torch.zeros((4, 2, 32, 64), dtype=torch.float8_e4m3fn)
    payload["gemm1_weights_scale"] = torch.zeros((4, 2, 2), dtype=torch.float8_e8m0fnu)
    payload["gemm2_weights"] = torch.zeros((4, 64, 1, 32), dtype=torch.float8_e4m3fn)
    payload["gemm2_weights_scale"] = torch.zeros((4, 2, 1), dtype=torch.float8_e8m0fnu)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)

    prepared = load_prepacked_weights(
        Path("weights.pt"), _profile(), torch.device("cpu")
    )

    assert prepared.gemm1_weights is payload["gemm1_weights"]
    assert prepared.gemm1_weights_scale is payload["gemm1_weights_scale"]
    assert prepared.gemm2_weights is payload["gemm2_weights"]
    assert prepared.gemm2_weights_scale is payload["gemm2_weights_scale"]


def test_load_prepacked_weights_rejects_mismatched_layout_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact_payload()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["weight_layout"] = "BlockMajorK"
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)

    with pytest.raises(ValueError, match="weight_layout"):
        load_prepacked_weights(Path("weights.pt"), _profile(), torch.device("cpu"))


def test_synthetic_weights_run_upstream_gated_reorder_and_shuffle_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], int | None]] = []

    def quantize(
        tensor: torch.Tensor, swizzled: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not swizzled
        return (
            tensor.to(torch.float8_e4m3fn),
            torch.zeros(
                (*tensor.shape[:-1], tensor.shape[-1] // 32), dtype=torch.uint8
            ),
        )

    def reorder(tensor: torch.Tensor) -> torch.Tensor:
        calls.append(("reorder", tuple(tensor.shape), None))
        return tensor

    def shuffle_a(tensor: torch.Tensor, tile: int) -> torch.Tensor:
        calls.append(("shuffle_a", tuple(tensor.shape), tile))
        return tensor

    def shuffle_sf(tensor: torch.Tensor, tile: int) -> torch.Tensor:
        calls.append(("shuffle_sf", tuple(tensor.shape), tile))
        return tensor

    flashinfer = SimpleNamespace(
        mxfp8_quantize=quantize,
        reorder_rows_for_gated_act_gemm=reorder,
        shuffle_matrix_a=shuffle_a,
        shuffle_matrix_sf_a=shuffle_sf,
    )

    prepared = _prepare_synthetic_weights(_profile(), torch.device("cpu"), flashinfer)

    assert [name for name, _, _ in calls].count("reorder") == 8
    assert [name for name, _, _ in calls].count("shuffle_a") == 8
    assert [name for name, _, _ in calls].count("shuffle_sf") == 8
    assert all(tile == 128 for name, _, tile in calls if name.startswith("shuffle"))
    assert prepared.gemm1_weights.dtype == torch.float8_e4m3fn
    assert prepared.gemm1_weights_scale.dtype == torch.uint8


def test_run_moe_pair_returns_typed_activated_intermediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate = torch.ones((23, 2, 32), dtype=torch.bfloat16)
    runtime = SimpleNamespace(
        ActivationType=SimpleNamespace(Swiglu=SimpleNamespace(value=3)),
        RoutingMethodType=SimpleNamespace(RenormalizeNaive=SimpleNamespace(value=4)),
    )
    routed_moe = lambda **_kwargs: (
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        intermediate,
    )
    fp8_enum = SimpleNamespace(MxFp8="mxfp8")
    weight_layout = SimpleNamespace(MajorK=SimpleNamespace(value=0))
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._load_moe_runtime",
        lambda: (runtime, routed_moe, fp8_enum, weight_layout),
    )

    result = run_moe_pair(
        _case(),
        torch.zeros((23, 2), dtype=torch.int32),
        do_finalize=False,
        gemm1_lora_delta=torch.zeros((23, 2, 64), dtype=torch.bfloat16),
    )

    assert isinstance(result, MoePairResult)
    assert result.final_output is None
    assert result.activated_intermediate is not None
    assert torch.equal(result.activated_intermediate, intermediate)


@pytest.mark.parametrize("error", [TypeError("bad signature"), NotImplementedError()])
def test_run_moe_pair_normalizes_intermediate_invocation_failures(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def routed_moe(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._load_moe_runtime",
        lambda: (
            object(),
            routed_moe,
            SimpleNamespace(MxFp8="mxfp8"),
            SimpleNamespace(MajorK=SimpleNamespace(value=0)),
        ),
    )

    with pytest.raises(IntermediateApiUnavailable):
        run_moe_pair(
            _case(),
            torch.zeros((23, 2), dtype=torch.int32),
            do_finalize=False,
            gemm1_lora_delta=torch.zeros((23, 2, 64), dtype=torch.bfloat16),
        )


def test_run_monolithic_moe_pair_matches_vllm_call_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    blocked_gemm1 = torch.empty((4, 2, 32, 64), dtype=torch.float8_e4m3fn)
    blocked_gemm2 = torch.empty((4, 64, 1, 32), dtype=torch.float8_e4m3fn)
    object.__setattr__(case, "gemm1_weights", blocked_gemm1)
    object.__setattr__(case, "gemm2_weights", blocked_gemm2)
    router_logits = torch.zeros((23, 4), dtype=torch.bfloat16)
    final_output = torch.ones((23, 64), dtype=torch.bfloat16)
    calls: list[dict[str, object]] = []
    major_k = object()

    def monolithic_moe(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return final_output

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._load_monolithic_moe_runtime",
        lambda: (
            monolithic_moe,
            SimpleNamespace(MxFp8="mxfp8"),
            SimpleNamespace(MajorK=major_k),
        ),
    )

    result = flashinfer_adapter.run_monolithic_moe_pair(case, router_logits)

    assert result.final_output is final_output
    assert result.activated_intermediate is None
    assert calls == [
        {
            "routing_logits": router_logits,
            "routing_bias": None,
            "hidden_states": case.hidden_states,
            "hidden_states_scale": case.hidden_states_scale,
            "gemm1_weights": blocked_gemm1,
            "gemm1_weights_scale": case.gemm1_weights_scale,
            "gemm1_alpha": None,
            "gemm1_beta": None,
            "gemm1_clamp_limit": None,
            "gemm2_weights": blocked_gemm2,
            "gemm2_weights_scale": case.gemm2_weights_scale,
            "num_experts": 4,
            "top_k": 2,
            "n_group": None,
            "topk_group": None,
            "intermediate_size": 32,
            "local_expert_offset": 0,
            "local_num_experts": 4,
            "routed_scaling_factor": None,
            "routing_method_type": 4,
            "use_shuffled_weight": True,
            "weight_layout": major_k,
            "fp8_quantization_type": "mxfp8",
            "activation_type": 3,
        }
    ]


def test_run_monolithic_moe_pair_propagates_flashinfer_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected_moe(**_kwargs: object) -> None:
        raise TypeError("runtime prepacked shape rejected")

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.flashinfer_adapter._load_monolithic_moe_runtime",
        lambda: (
            rejected_moe,
            SimpleNamespace(MxFp8="mxfp8"),
            SimpleNamespace(MajorK=object()),
        ),
    )

    with pytest.raises(TypeError, match="runtime prepacked shape rejected"):
        flashinfer_adapter.run_monolithic_moe_pair(
            _case(), torch.zeros((23, 4), dtype=torch.bfloat16)
        )


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
        (32, 64),
        (0,),
        (32, 2),
        (32,),
        (32, 64),
        (32, 2),
        (0,),
        (0,),
    )
    assert intermediate[2][6] == (32, 2, 64)
    assert final[3] == intermediate[3] == ()


def test_cache_key_for_monolithic_case_includes_router_logits_shape() -> None:
    case = _case()
    router_logits = torch.empty((23, 4), dtype=torch.bfloat16)

    key = ast.literal_eval(
        cache_key_for_case(
            case,
            has_gemm1_lora_delta=False,
            router_logits=router_logits,
        )
    )

    assert key[2] == (
        (32, 64),
        (32, 4),
        (32,),
        (32,),
        (32, 64),
        (32, 2),
        (0,),
        (0,),
    )
