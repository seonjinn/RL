from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from typing import cast

import pytest
import torch

from experiments.mxfp8_moe_tactic_audit.flashinfer_adapter import MoeKernelCase
from experiments.mxfp8_moe_tactic_audit.schema import (
    ReplayProfile,
    RoutingSignature,
    TacticMeasurement,
    TacticPair,
)
from experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics import (
    IntermediateApiUnavailable,
    _profile_tactic_cuda,
    main,
    profile_tactic,
    reconstruct_topk,
)


def _profile(
    *,
    expert_counts: tuple[int, ...] = (2, 1, 1, 0),
    num_tokens: int = 2,
    top_k: int = 2,
) -> ReplayProfile:
    signature = RoutingSignature(
        schema_version=1,
        model_revision="qwen3-30ba3b-test",
        layer_family="routed_experts",
        num_tokens=num_tokens,
        global_num_experts=len(expert_counts),
        local_num_experts=len(expert_counts),
        top_k=top_k,
        hidden_size=2048,
        intermediate_size=768,
        expert_counts=expert_counts,
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


def test_reconstruct_topk_reproduces_histogram_without_duplicate_experts() -> None:
    profile = _profile()

    packed_topk, topk_weights = reconstruct_topk(profile, torch.device("cpu"))

    topk_ids = torch.bitwise_right_shift(packed_topk, 16)
    histogram = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=4)
    assert tuple(histogram.tolist()) == profile.signature.expert_counts
    assert all(len(set(row)) == profile.signature.top_k for row in topk_ids.tolist())
    assert topk_weights.dtype == torch.bfloat16
    torch.testing.assert_close(
        topk_weights.sum(dim=1),
        torch.ones(profile.signature.num_tokens, dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_reconstruct_topk_is_deterministic_for_signature_key() -> None:
    profile = _profile()

    first = reconstruct_topk(profile, torch.device("cpu"))
    second = reconstruct_topk(profile, torch.device("cpu"))

    assert all(
        torch.equal(left, right) for left, right in zip(first, second, strict=True)
    )


def test_reconstruct_topk_rejects_histogram_requiring_duplicate_expert() -> None:
    profile = _profile(expert_counts=cast(tuple[int, ...], (3, 1)), num_tokens=2)

    with pytest.raises(ValueError, match="same expert twice"):
        reconstruct_topk(profile, torch.device("cpu"))


def _case(profile: ReplayProfile | None = None) -> MoeKernelCase:
    replay_profile = profile or _profile()
    signature = replay_profile.signature
    return MoeKernelCase(
        profile=replay_profile,
        hidden_states=torch.empty((signature.num_tokens, signature.hidden_size)),
        hidden_states_scale=torch.empty(
            (signature.num_tokens, signature.hidden_size // 32)
        ),
        gemm1_weights=torch.empty(0),
        gemm1_weights_scale=torch.empty(0),
        gemm2_weights=torch.empty(0),
        gemm2_weights_scale=torch.empty(0),
        output=torch.zeros(
            (signature.num_tokens, signature.hidden_size), dtype=torch.bfloat16
        ),
        activation_type=3,
        routing_method_type=4,
        local_expert_offset=0,
    )


def test_profile_tactic_requires_three_warmups_and_ten_repetitions() -> None:
    with pytest.raises(ValueError, match="warmups must be at least 3"):
        profile_tactic(_case(), TacticPair(1, 2), warmups=2, repetitions=10)
    with pytest.raises(ValueError, match="repetitions must be at least 10"):
        profile_tactic(_case(), TacticPair(1, 2), warmups=3, repetitions=9)


def test_profile_tactic_serializes_a_row_for_tactic_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._profile_tactic_cuda",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("kernel crash")),
    )

    measurement = profile_tactic(_case(), TacticPair(1, 2))

    assert measurement.failure == "RuntimeError: kernel crash"
    assert not measurement.finite
    assert TacticMeasurement.from_json(measurement.to_json()) == measurement


def test_profile_tactic_fails_closed_when_intermediate_api_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._profile_tactic_cuda",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            IntermediateApiUnavailable("unexpected return contract")
        ),
    )

    measurement = profile_tactic(_case(), TacticPair(1, 2))

    assert measurement.failure == "flashinfer_intermediate_api_unavailable"
    assert not measurement.finite


def test_profile_tactic_uses_paired_graph_replay_and_cold_l2_each_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    tactic = TacticPair(1, 2)
    run_modes: list[bool] = []
    forced: list[tuple[str, TacticPair | None]] = []
    graph_replays = 0
    cold_touches = 0
    original_zeros = torch.zeros
    original_empty = torch.empty

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.reconstruct_topk",
        lambda _profile, _device: (
            torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
            torch.full((2, 2), 0.5, dtype=torch.bfloat16),
        ),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.cache_key_for_case",
        lambda _case, *, has_gemm1_lora_delta: (
            "intermediate-key" if has_gemm1_lora_delta else "final-key"
        ),
    )

    @contextmanager
    def fake_force(key: str, selected: TacticPair | None = None):
        forced.append((key, selected))
        yield

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.force_stock_tactic",
        lambda key: fake_force(key),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.force_tactic",
        lambda key, selected: fake_force(key, selected),
    )

    intermediate = torch.ones((2, 2, 768), dtype=torch.bfloat16)

    def fake_run_moe_pair(
        _case: MoeKernelCase,
        _packed_topk: torch.Tensor,
        *,
        do_finalize: bool,
        gemm1_lora_delta: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        run_modes.append(do_finalize)
        if do_finalize:
            assert gemm1_lora_delta is None
            return (_case.output,)
        assert gemm1_lora_delta is not None
        return (torch.empty(0), torch.empty(0), torch.empty(0), intermediate)

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.run_moe_pair",
        fake_run_moe_pair,
    )

    def cpu_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    class FakeColdL2:
        def add_(self, _value: int) -> None:
            nonlocal cold_touches
            cold_touches += 1

    def cpu_empty(*args: object, **kwargs: object) -> torch.Tensor | FakeColdL2:
        if kwargs.get("dtype") == torch.uint8 and len(args) == 1:
            return FakeColdL2()
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    class FakeGraph:
        def replay(self) -> None:
            nonlocal graph_replays
            graph_replays += 1

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 0.004

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda _graph: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Properties", (), {"L2_cache_size": 1024})(),
    )
    object.__setattr__(
        case,
        "hidden_states",
        type("CudaTensor", (), {"device": torch.device("cuda"), "shape": (2, 2048)})(),
    )

    result = _profile_tactic_cuda(case, tactic, warmups=3, repetitions=10)

    assert run_modes.count(True) == 6
    assert run_modes.count(False) == 3
    assert forced == [
        ("final-key", None),
        ("intermediate-key", None),
        ("intermediate-key", tactic),
        ("final-key", tactic),
    ]
    assert graph_replays == 10
    assert cold_touches == 10
    assert result.median_us == result.p95_us == 4.0
    assert result.finite and result.deterministic


def test_cli_writes_one_serializable_row_per_tactic_and_continues_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    profiles_path = tmp_path / "selected_profiles.json"
    profiles_path.write_text(
        json.dumps({"selected_profiles": [profile.to_json()]}), encoding="ascii"
    )
    output_path = tmp_path / "measurements.jsonl"
    case = _case(profile)
    tactics = (TacticPair(1, 2), TacticPair(3, 4))
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.build_kernel_case",
        lambda _profile, _device: case,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.enumerate_valid_tactics",
        lambda _case: tactics,
    )

    def fake_profile_tactic(
        _case: MoeKernelCase,
        tactic: TacticPair,
        warmups: int = 3,
        repetitions: int = 10,
    ) -> TacticMeasurement:
        return TacticMeasurement(
            signature_key=profile.signature_key,
            tactic=tactic,
            median_us=4.0 if tactic == tactics[0] else 0.0,
            p95_us=4.5 if tactic == tactics[0] else 0.0,
            cv=0.02 if tactic == tactics[0] else 0.0,
            warmups=warmups,
            repetitions=repetitions,
            finite=tactic == tactics[0],
            deterministic=tactic == tactics[0],
            max_abs_error=0.0,
            cosine_similarity=1.0 if tactic == tactics[0] else 0.0,
            failure=None if tactic == tactics[0] else "RuntimeError: kernel crash",
        )

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.profile_tactic",
        fake_profile_tactic,
    )

    assert (
        main(
            [
                "--profiles",
                str(profiles_path),
                "--profile-limit",
                "1",
                "--tactic-limit",
                "2",
                "--warmups",
                "3",
                "--repetitions",
                "10",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="ascii").splitlines()
    ]
    assert len(rows) == 2
    measurements = [TacticMeasurement.from_json(row) for row in rows]
    assert [measurement.tactic for measurement in measurements] == list(tactics)
    assert measurements[1].failure == "RuntimeError: kernel crash"
