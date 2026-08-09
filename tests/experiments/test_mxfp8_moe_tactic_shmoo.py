from contextlib import contextmanager, nullcontext
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch  # pyright: ignore[reportMissingImports]

from experiments.mxfp8_moe_tactic_audit.flashinfer_adapter import (
    PREPACKED_ARTIFACT_FORMAT,
    IntermediateApiUnavailable,
    MoeKernelCase,
    MoePairResult,
)
from experiments.mxfp8_moe_tactic_audit.schema import (
    ReplayProfile,
    RoutingSignature,
    TacticMeasurement,
    TacticPair,
)
from experiments.mxfp8_moe_tactic_audit.nsys_to_component_csv import convert
from experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics import (
    FC1_CUMULATIVE,
    PAIR_CUMULATIVE,
    _nsys_component_range,
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


def test_actual_shmoo_range_converts_to_mean_component_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the producer label rather than constructing an idealized tag."""
    labels: list[str] = []
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_push", lambda label: labels.append(label)
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    for component in (FC1_CUMULATIVE, PAIR_CUMULATIVE):
        with _nsys_component_range(
            _case(),
            TacticPair(1, 2),
            "stock",
            component,
            comparison_tactic=TacticPair(3, 4),
            cache_event="fallback",
        ):
            pass

    raw = tmp_path / "nvtx.csv"
    with raw.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Range", "Instances", "Total Time (ns)"]
        )
        writer.writeheader()
        writer.writerow(
            {"Range": labels[0], "Instances": 2, "Total Time (ns)": 120_000}
        )
        writer.writerow(
            {"Range": labels[1], "Instances": 2, "Total Time (ns)": 200_000}
        )
    output = tmp_path / "components.csv"
    convert(raw, output)

    rows = list(csv.DictReader(output.open(encoding="ascii")))
    assert [row["component"] for row in rows] == ["FC1/GEMM1", "FC2/GEMM2"]
    assert all(row["cache_event"] == "fallback" for row in rows)
    assert all(row["call_count"] == "2" for row in rows)
    assert [row["mean_us"] for row in rows] == ["60", "40"]
    assert all("median_us" not in row for row in rows)


def test_pair_only_range_is_preserved_without_false_fc1_fc2_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_push", lambda label: labels.append(label)
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    with _nsys_component_range(
        _case(),
        TacticPair(1, 2),
        "candidate",
        PAIR_CUMULATIVE,
        comparison_tactic=TacticPair(1, 2),
        cache_event="cache hit",
    ):
        pass

    raw = tmp_path / "nvtx.csv"
    with raw.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Range", "Instances", "Total Time (ns)"]
        )
        writer.writeheader()
        writer.writerow(
            {"Range": labels[0], "Instances": 2, "Total Time (ns)": 200_000}
        )
    output = tmp_path / "components.csv"
    convert(raw, output)

    rows = list(csv.DictReader(output.open(encoding="ascii")))
    assert len(rows) == 1
    assert rows[0]["component"] == "FC1+FC2/GEMM1+GEMM2"
    assert rows[0]["mean_us"] == "100"


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
        weight_layout="MajorK",
        use_shuffled_weight=True,
        prepacked_weight_format=PREPACKED_ARTIFACT_FORMAT,
    )


def test_profile_tactic_requires_exactly_three_warmups_and_ten_repetitions() -> None:
    with pytest.raises(ValueError, match="warmups must equal 3"):
        profile_tactic(_case(), TacticPair(1, 2), warmups=2, repetitions=10)
    with pytest.raises(ValueError, match="warmups must equal 3"):
        profile_tactic(_case(), TacticPair(1, 2), warmups=4, repetitions=10)
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


def test_profile_tactic_pair_only_does_not_require_intermediate_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        median_us=7.0,
        p95_us=8.0,
        cv=0.01,
        finite=True,
        deterministic=True,
        max_abs_error=0.0,
        cosine_similarity=1.0,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._profile_tactic_pair_cuda",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._profile_tactic_cuda",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            IntermediateApiUnavailable("unexpected return contract")
        ),
    )

    measurement = profile_tactic(_case(), TacticPair(1, 2), pair_only=True)

    assert measurement.failure is None
    assert measurement.median_us == 7.0
    assert measurement.finite


@pytest.mark.parametrize("repeated_intermediate_nan", [False, True])
def test_profile_tactic_uses_paired_graph_replay_and_cold_l2_each_time(
    repeated_intermediate_nan: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    tactic = TacticPair(1, 2)

    class FakeGraph:
        def __init__(self) -> None:
            self.do_finalize: bool | None = None

        def replay(self) -> None:
            nonlocal graph_replays
            graph_replays += 1
            replay_ranges.append((active_range, self.do_finalize))

    run_modes: list[bool] = []
    forced: list[tuple[str, TacticPair | None]] = []
    nsys_ranges: list[tuple[str, str, TacticPair, str]] = []
    graph_replays = 0
    cold_touches = 0
    active_range: str | None = None
    synchronized_ranges: list[str | None] = []
    replay_ranges: list[tuple[str | None, bool | None]] = []
    capturing_graph: FakeGraph | None = None
    original_zeros = torch.zeros
    original_empty = torch.empty
    intermediate_calls = 0

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
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.observed_forced_cache_event",
        lambda _key: "cache hit",
    )

    @contextmanager
    def fake_nsys_range(
        _case: MoeKernelCase,
        selected: TacticPair,
        arm: str,
        component: str,
        *,
        comparison_tactic: TacticPair,
        cache_event: str,
    ):
        nonlocal active_range
        assert comparison_tactic == tactic
        assert active_range is None
        active_range = component
        nsys_ranges.append((arm, component, selected, cache_event))
        try:
            yield
        finally:
            active_range = None

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._nsys_component_range",
        fake_nsys_range,
    )

    intermediate = torch.ones((2, 2, 768), dtype=torch.bfloat16)
    nonfinite_intermediate = intermediate.clone()
    nonfinite_intermediate[0, 0, 0] = float("nan")

    def fake_run_moe_pair(
        _case: MoeKernelCase,
        _packed_topk: torch.Tensor,
        *,
        do_finalize: bool,
        gemm1_lora_delta: torch.Tensor | None,
    ) -> MoePairResult:
        nonlocal intermediate_calls
        assert active_range is None
        if capturing_graph is not None:
            capturing_graph.do_finalize = do_finalize
        run_modes.append(do_finalize)
        if do_finalize:
            assert gemm1_lora_delta is None
            return MoePairResult(final_output=_case.output, activated_intermediate=None)
        assert gemm1_lora_delta is not None
        intermediate_calls += 1
        selected_intermediate = (
            nonfinite_intermediate
            if repeated_intermediate_nan and intermediate_calls == 8
            else intermediate
        )
        return MoePairResult(
            final_output=None, activated_intermediate=selected_intermediate
        )

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

    @contextmanager
    def fake_graph_context(graph: FakeGraph):
        nonlocal capturing_graph
        assert capturing_graph is None
        capturing_graph = graph
        try:
            yield
        finally:
            capturing_graph = None

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            synchronized_ranges.append(active_range)

        def elapsed_time(self, _other: object) -> float:
            return 0.004

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", fake_graph_context)
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

    result = _profile_tactic_cuda(
        case,
        tactic,
        warmups=3,
        repetitions=10,
        stock_tactics={
            "final-key": TacticPair(5, 6),
            "intermediate-key": TacticPair(7, 8),
        },
    )

    assert run_modes.count(True) == 8
    assert run_modes.count(False) == 8
    assert forced == [
        ("final-key", TacticPair(5, 6)),
        ("intermediate-key", TacticPair(7, 8)),
        ("intermediate-key", tactic),
        ("final-key", tactic),
    ]
    assert nsys_ranges == [
        *(("stock", PAIR_CUMULATIVE, TacticPair(5, 6), "cache hit"),) * 10,
        *(("stock", FC1_CUMULATIVE, TacticPair(7, 8), "cache hit"),) * 10,
        *(("candidate", FC1_CUMULATIVE, tactic, "cache hit"),) * 10,
        *(("candidate", PAIR_CUMULATIVE, tactic, "cache hit"),) * 10,
    ]
    assert graph_replays == 40
    assert cold_touches == 40
    assert all(component is not None for component, _mode in replay_ranges)
    assert all(
        (component == PAIR_CUMULATIVE) is do_finalize
        for component, do_finalize in replay_ranges
    )
    assert synchronized_ranges == [None] * 40
    assert result.median_us == result.p95_us == 4.0
    assert result.finite is not repeated_intermediate_nan
    assert result.deterministic is not repeated_intermediate_nan


def test_stock_intermediate_invocation_failure_is_normalized_before_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    object.__setattr__(
        case,
        "hidden_states",
        type("CudaTensor", (), {"device": torch.device("cuda"), "shape": (2, 2048)})(),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.reconstruct_topk",
        lambda _profile, _device: (
            torch.zeros((2, 2), dtype=torch.int32),
            torch.full((2, 2), 0.5, dtype=torch.bfloat16),
        ),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.cache_key_for_case",
        lambda _case, *, has_gemm1_lora_delta: (
            "intermediate" if has_gemm1_lora_delta else "final"
        ),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.force_stock_tactic",
        lambda _key: nullcontext(),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.force_tactic",
        lambda *_args: (_ for _ in ()).throw(AssertionError("candidate forced")),
    )
    original_zeros = torch.zeros

    def cpu_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)

    def fake_run(
        _case: MoeKernelCase,
        _packed: torch.Tensor,
        *,
        do_finalize: bool,
        gemm1_lora_delta: torch.Tensor | None,
    ) -> MoePairResult:
        if do_finalize:
            return MoePairResult(final_output=_case.output, activated_intermediate=None)
        raise TypeError("zero-LoRA contract unavailable")

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.run_moe_pair",
        fake_run,
    )

    def fake_profile_component(*_args: object, **kwargs: object) -> object:
        if kwargs["component"] == FC1_CUMULATIVE:
            raise TypeError("zero-LoRA contract unavailable")
        return SimpleNamespace(outputs=(case.output,), timings_us=(4.0,) * 10)

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._profile_component_replays",
        fake_profile_component,
    )

    with pytest.raises(IntermediateApiUnavailable):
        _profile_tactic_cuda(case, tactic=TacticPair(1, 2), warmups=3, repetitions=10)


@pytest.mark.parametrize("bad_reference", ["final", "intermediate"])
def test_nonfinite_stock_reference_fails_before_candidate_profiling(
    bad_reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case()
    object.__setattr__(
        case,
        "hidden_states",
        type("CudaTensor", (), {"device": torch.device("cuda"), "shape": (2, 2048)})(),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.reconstruct_topk",
        lambda _profile, _device: (
            torch.zeros((2, 2), dtype=torch.int32),
            torch.full((2, 2), 0.5, dtype=torch.bfloat16),
        ),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.cache_key_for_case",
        lambda _case, *, has_gemm1_lora_delta: (
            "intermediate" if has_gemm1_lora_delta else "final"
        ),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.force_stock_tactic",
        lambda _key: nullcontext(),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.force_tactic",
        lambda *_args: (_ for _ in ()).throw(AssertionError("candidate forced")),
    )
    original_zeros = torch.zeros

    def cpu_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    final = torch.zeros((2, 2048), dtype=torch.bfloat16)
    intermediate = torch.zeros((2, 2, 768), dtype=torch.bfloat16)
    if bad_reference == "final":
        final[0, 0] = float("nan")
    else:
        intermediate[0, 0, 0] = float("inf")

    def fake_run(
        _case: MoeKernelCase,
        _packed: torch.Tensor,
        *,
        do_finalize: bool,
        gemm1_lora_delta: torch.Tensor | None,
    ) -> MoePairResult:
        return (
            MoePairResult(final_output=final, activated_intermediate=None)
            if do_finalize
            else MoePairResult(final_output=None, activated_intermediate=intermediate)
        )

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.run_moe_pair",
        fake_run,
    )

    def fake_profile_component(*_args: object, **kwargs: object) -> object:
        if kwargs["arm"] != "stock":
            raise AssertionError("candidate profiled")
        output = final if kwargs["component"] == PAIR_CUMULATIVE else intermediate
        return SimpleNamespace(outputs=(output,), timings_us=(4.0,) * 10)

    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics._profile_component_replays",
        fake_profile_component,
    )

    with pytest.raises(RuntimeError, match="stock .* reference is not finite"):
        _profile_tactic_cuda(case, TacticPair(1, 2), warmups=3, repetitions=10)


def test_cli_rejects_broader_source_less_invocation(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--profiles",
                str(tmp_path / "profiles.json"),
                "--output",
                str(tmp_path / "measurements.jsonl"),
            ]
        )


def test_exact_brief_smoke_args_use_marked_bounded_synthetic_source(
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
    build_calls: list[tuple[Path | None, bool]] = []
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.build_kernel_case",
        lambda _profile, _device, *, weights_path, synthetic_smoke: (
            build_calls.append((weights_path, synthetic_smoke)) or case
        ),
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.enumerate_valid_tactics",
        lambda _case: tactics,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.profile_tactic",
        lambda _case, tactic, warmups=3, repetitions=10, pair_only=False: TacticMeasurement(
            signature_key=profile.signature_key,
            tactic=tactic,
            median_us=4.0,
            p95_us=4.5,
            cv=0.02,
            warmups=warmups,
            repetitions=repetitions,
            finite=True,
            deterministic=True,
            max_abs_error=0.0,
            cosine_similarity=1.0,
            failure=None,
        ),
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
    assert build_calls == [(None, True)]
    assert len(rows) == 2
    assert all(row["synthetic"] is True for row in rows)


def test_cli_writes_one_serializable_row_per_tactic_and_continues_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    profiles_path = tmp_path / "selected_profiles.json"
    profiles_path.write_text(
        json.dumps({"selected_profiles": [profile.to_json()]}), encoding="ascii"
    )
    output_path = tmp_path / "measurements.jsonl"
    weights_path = tmp_path / "prepacked.pt"
    stock_cache_path = tmp_path / "stock-cache.json"
    stock_cache_path.write_text(
        json.dumps({"cache-key": ["MoERunner", [1, 2]]}), encoding="ascii"
    )
    case = _case(profile)
    tactics = (TacticPair(1, 2), TacticPair(3, 4))
    build_calls: list[tuple[Path | None, bool]] = []
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.assert_supported_flashinfer",
        lambda: None,
    )
    monkeypatch.setattr(
        "experiments.mxfp8_moe_tactic_audit.shmoo_moe_tactics.build_kernel_case",
        lambda _profile, _device, *, weights_path, synthetic_smoke: (
            build_calls.append((weights_path, synthetic_smoke)) or case
        ),
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
        stock_tactics: object = None,
        pair_only: bool = False,
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
                "--weights",
                str(weights_path),
                "--stock-cache",
                str(stock_cache_path),
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
    assert build_calls == [(weights_path, False)]
    measurements = [TacticMeasurement.from_json(row) for row in rows]
    assert [measurement.tactic for measurement in measurements] == list(tactics)
    assert measurements[1].failure == "RuntimeError: kernel crash"
