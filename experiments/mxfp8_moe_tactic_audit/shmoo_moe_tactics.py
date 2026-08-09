"""Workload-replayed FlashInfer MXFP8 MoE tactic shmoo."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import cast

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as functional  # pyright: ignore[reportMissingImports]

try:
    from .flashinfer_adapter import (
        IntermediateApiUnavailable,
        MoeKernelCase,
        MoePairResult,
        TacticDispatchError,
        assert_monolithic_replay_supported,
        assert_supported_flashinfer,
        build_kernel_case,
        cache_key_for_case,
        enumerate_valid_tactics,
        force_stock_tactic,
        force_tactic,
        observed_forced_cache_event,
        run_monolithic_moe_pair,
        run_moe_pair,
    )
    from .schema import ReplayProfile, TacticMeasurement, TacticPair
except ImportError:  # pragma: no cover - direct script execution
    from flashinfer_adapter import (
        IntermediateApiUnavailable,
        MoeKernelCase,
        MoePairResult,
        TacticDispatchError,
        assert_monolithic_replay_supported,
        assert_supported_flashinfer,
        build_kernel_case,
        cache_key_for_case,
        enumerate_valid_tactics,
        force_stock_tactic,
        force_tactic,
        observed_forced_cache_event,
        run_monolithic_moe_pair,
        run_moe_pair,
    )
    from schema import ReplayProfile, TacticMeasurement, TacticPair


FC1_CUMULATIVE = "FC1/GEMM1 cumulative"
PAIR_CUMULATIVE = "FC1+FC2/GEMM1+GEMM2 cumulative"


@dataclass(frozen=True)
class _ProfileResult:
    median_us: float
    p95_us: float
    cv: float
    finite: bool
    deterministic: bool
    max_abs_error: float
    cosine_similarity: float


@dataclass(frozen=True)
class _ReplayResult:
    outputs: tuple[torch.Tensor, ...]
    timings_us: tuple[float, ...]


def _seeded_order(signature_key: str, index: int) -> bytes:
    return sha256(f"{signature_key}:{index}".encode("ascii")).digest()


def _reconstruct_expert_rows(profile: ReplayProfile) -> tuple[tuple[int, ...], ...]:
    signature = profile.signature
    if signature.top_k > signature.global_num_experts:
        raise ValueError("routing would require the same expert twice for a token")
    if max(signature.expert_counts) > signature.num_tokens:
        raise ValueError("routing would require the same expert twice for a token")

    token_experts: list[list[int]] = [[] for _ in range(signature.num_tokens)]
    token_loads = [0] * signature.num_tokens
    cursor = int(profile.signature_key[:16], 16) % signature.num_tokens
    experts = sorted(
        range(signature.global_num_experts),
        key=lambda expert: (
            -signature.expert_counts[expert],
            _seeded_order(profile.signature_key, expert),
        ),
    )
    for expert in experts:
        count = signature.expert_counts[expert]
        available = [
            token
            for token in range(signature.num_tokens)
            if token_loads[token] < signature.top_k
        ]
        if count > len(available):
            raise ValueError("routing would require the same expert twice for a token")
        selected = sorted(
            available,
            key=lambda token: (
                token_loads[token],
                (token - cursor) % signature.num_tokens,
            ),
        )[:count]
        for token in selected:
            token_experts[token].append(expert)
            token_loads[token] += 1
        if selected:
            cursor = (selected[-1] + 1) % signature.num_tokens

    if any(load != signature.top_k for load in token_loads):
        raise ValueError(
            "expert histogram cannot be assigned to fixed-width top-k routing"
        )
    for token, row in enumerate(token_experts):
        row.sort(
            key=lambda expert: _seeded_order(
                profile.signature_key, token * signature.global_num_experts + expert
            )
        )
    return tuple(tuple(row) for row in token_experts)


def reconstruct_topk(
    profile: ReplayProfile, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct deterministic packed routing with the exact expert histogram."""
    signature = profile.signature
    token_experts = _reconstruct_expert_rows(profile)

    topk_ids = torch.tensor(token_experts, dtype=torch.int32, device=device)
    topk_weights = torch.full(
        (signature.num_tokens, signature.top_k),
        1.0 / signature.top_k,
        dtype=torch.bfloat16,
        device=device,
    )
    if signature.top_k > 1:
        topk_weights[:, -1] = 1 - topk_weights[:, :-1].sum(dim=1)
    packed_topk = (topk_ids << 16) | topk_weights.view(torch.int16).to(torch.int32)
    return packed_topk, topk_weights


def reconstruct_router_logits(
    profile: ReplayProfile, device: torch.device
) -> torch.Tensor:
    """Reconstruct dense BF16 logits whose top-k has the exact histogram."""
    signature = profile.signature
    topk_ids = torch.tensor(
        _reconstruct_expert_rows(profile), dtype=torch.int64, device=device
    )
    router_logits = torch.full(
        (signature.num_tokens, signature.global_num_experts),
        -1.0,
        dtype=torch.bfloat16,
        device=device,
    )
    router_logits.scatter_(1, topk_ids, 1.0)
    return router_logits


def _final_output(result: MoePairResult) -> torch.Tensor:
    if result.final_output is None:
        raise RuntimeError("adapter returned no final FC2 output")
    return result.final_output


def _intermediate_output(result: MoePairResult) -> torch.Tensor:
    if result.activated_intermediate is None:
        raise IntermediateApiUnavailable(
            "adapter returned no activated FC1 intermediate"
        )
    return result.activated_intermediate


def _tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_float = left.float().flatten()
    right_float = right.float().flatten()
    if not torch.isfinite(left_float).all() or not torch.isfinite(right_float).all():
        return 0.0
    if (
        torch.count_nonzero(left_float).item() == 0
        and torch.count_nonzero(right_float).item() == 0
    ):
        return 1.0
    return float(functional.cosine_similarity(left_float, right_float, dim=0).item())


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.float() - right.float()).abs()
    finite_max = torch.finfo(torch.float32).max
    return float(
        torch.nan_to_num(
            difference, nan=finite_max, posinf=finite_max, neginf=finite_max
        )
        .max()
        .item()
    )


def _l2_size_bytes(device: torch.device) -> int:
    properties = torch.cuda.get_device_properties(device)
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(properties, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 256 * 1024 * 1024


@contextmanager
def _nsys_component_range(
    case: MoeKernelCase,
    tactic: TacticPair,
    arm: str,
    component: str,
    *,
    comparison_tactic: TacticPair,
    cache_event: str,
    router_logits: torch.Tensor | None = None,
) -> Iterator[None]:
    """Emit the metadata NSys needs to produce one non-fabricated component row."""
    cache_key = cache_key_for_case(
        case,
        has_gemm1_lora_delta=component == FC1_CUMULATIVE,
        router_logits=router_logits,
    )
    label = "|".join(
        (
            "MXFP8_MOE_AUDIT",
            f"signature_key={case.profile.signature_key}",
            f"cache_key={cache_key}",
            f"arm={arm}",
            f"component={component}",
            f"tactic={tactic.gemm1},{tactic.gemm2}",
            f"comparison_tactic={comparison_tactic.gemm1},{comparison_tactic.gemm2}",
            f"cache_event={cache_event}",
            f"call_weight={case.profile.call_count}",
        )
    )
    pushed = False
    try:
        torch.cuda.nvtx.range_push(label)
        pushed = True
    except RuntimeError:
        # CPU-only unit tests have no NVTX library; an actual NSys run must emit ranges.
        pass
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def _profile_component_replays(
    case: MoeKernelCase,
    routing_input: torch.Tensor,
    tactic: TacticPair,
    *,
    cache_key: str,
    arm: str,
    component: str,
    comparison_tactic: TacticPair,
    warmups: int,
    repetitions: int,
    use_stock_fallback: bool,
    zero_delta: torch.Tensor | None,
    monolithic_replay: bool = False,
) -> _ReplayResult:
    """Profile only graph replays after setup and observe the active cache event."""
    if component not in {FC1_CUMULATIVE, PAIR_CUMULATIVE}:
        raise ValueError(f"unsupported cumulative component: {component}")
    do_finalize = component == PAIR_CUMULATIVE
    if monolithic_replay and not do_finalize:
        raise ValueError("monolithic replay supports only the full FC1+FC2 pair")
    delta = None if do_finalize else zero_delta

    def run_pair() -> MoePairResult:
        if monolithic_replay:
            return run_monolithic_moe_pair(case, routing_input)
        return run_moe_pair(
            case,
            routing_input,
            do_finalize=do_finalize,
            gemm1_lora_delta=delta,
        )

    force_context = (
        force_stock_tactic(cache_key)
        if use_stock_fallback
        else force_tactic(cache_key, tactic)
    )
    with force_context:
        for _ in range(warmups):
            run_pair()
        torch.cuda.synchronize(case.hidden_states.device)
        cache_event = observed_forced_cache_event(cache_key)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_result = run_pair()
        graph_output = (
            _final_output(captured_result)
            if do_finalize
            else _intermediate_output(captured_result)
        )
        cold_l2 = torch.empty(
            _l2_size_bytes(case.hidden_states.device) + 4 * 1024 * 1024,
            dtype=torch.uint8,
            device=case.hidden_states.device,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timings_us: list[float] = []
        outputs: list[torch.Tensor] = []
        for _ in range(repetitions):
            cold_l2.add_(1)
            start.record()
            range_context = (
                _nsys_component_range(
                    case,
                    tactic,
                    arm,
                    component,
                    comparison_tactic=comparison_tactic,
                    cache_event=cache_event,
                    router_logits=routing_input,
                )
                if monolithic_replay
                else _nsys_component_range(
                    case,
                    tactic,
                    arm,
                    component,
                    comparison_tactic=comparison_tactic,
                    cache_event=cache_event,
                )
            )
            with range_context:
                graph.replay()
                end.record()
            end.synchronize()
            timings_us.append(float(start.elapsed_time(end) * 1000.0))
            outputs.append(graph_output.clone())
    return _ReplayResult(tuple(outputs), tuple(timings_us))


def _load_stock_tactics(path: Path) -> dict[str, TacticPair]:
    """Load the exact stock-cache tactics used to tag NSys ranges."""
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read stock cache {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("stock cache must be a JSON object")
    tactics: dict[str, TacticPair] = {}
    for cache_key, value in payload.items():
        if cache_key == "_metadata":
            continue
        if not isinstance(value, list) or not value or value[0] != "MoERunner":
            continue
        if (
            not isinstance(cache_key, str)
            or len(value) != 2
            or not isinstance(value[1], list)
            or len(value[1]) != 2
        ):
            raise ValueError("stock cache contains an invalid MoERunner tactic")
        tactics[cache_key] = TacticPair(value[1][0], value[1][1])
    if not tactics:
        raise ValueError("stock cache has no MoERunner tactics")
    return tactics


def _profile_tactic_cuda(
    case: MoeKernelCase,
    tactic: TacticPair,
    *,
    warmups: int,
    repetitions: int,
    stock_tactics: Mapping[str, TacticPair] | None = None,
) -> _ProfileResult:
    if case.hidden_states.device.type != "cuda":
        raise ValueError("MXFP8 MoE tactic profiling requires CUDA")
    packed_topk, _ = reconstruct_topk(case.profile, case.hidden_states.device)
    original_routing = packed_topk.clone()
    signature = case.profile.signature
    final_key = cache_key_for_case(case, has_gemm1_lora_delta=False)
    intermediate_key = cache_key_for_case(case, has_gemm1_lora_delta=True)
    stock_final_tactic = None if stock_tactics is None else stock_tactics.get(final_key)
    stock_intermediate_tactic = (
        None if stock_tactics is None else stock_tactics.get(intermediate_key)
    )
    if stock_tactics is not None and (
        stock_final_tactic is None or stock_intermediate_tactic is None
    ):
        raise ValueError("stock cache has no tactic for a replayed component key")
    zero_delta = torch.zeros(
        signature.num_tokens,
        signature.top_k,
        2 * signature.intermediate_size,
        dtype=torch.bfloat16,
        device=case.hidden_states.device,
    )

    stock_final_replays = _profile_component_replays(
        case,
        packed_topk,
        tactic if stock_final_tactic is None else stock_final_tactic,
        cache_key=final_key,
        arm="stock",
        component=PAIR_CUMULATIVE,
        comparison_tactic=tactic,
        warmups=warmups,
        repetitions=repetitions,
        use_stock_fallback=stock_final_tactic is None,
        zero_delta=zero_delta,
    )
    try:
        stock_intermediate_replays = _profile_component_replays(
            case,
            packed_topk,
            tactic if stock_intermediate_tactic is None else stock_intermediate_tactic,
            cache_key=intermediate_key,
            arm="stock",
            component=FC1_CUMULATIVE,
            comparison_tactic=tactic,
            warmups=warmups,
            repetitions=repetitions,
            use_stock_fallback=stock_intermediate_tactic is None,
            zero_delta=zero_delta,
        )
    except TacticDispatchError:
        raise
    except Exception as error:
        raise IntermediateApiUnavailable(
            "FlashInfer stock intermediate preflight is unavailable"
        ) from error

    stock_final = stock_final_replays.outputs[0]
    stock_intermediate = stock_intermediate_replays.outputs[0]
    if not bool(torch.isfinite(stock_final).all().item()):
        raise RuntimeError("stock FC2 output reference is not finite")
    if not bool(torch.isfinite(stock_intermediate).all().item()):
        raise RuntimeError("stock FC1 intermediate reference is not finite")

    candidate_intermediate_replays = _profile_component_replays(
        case,
        packed_topk,
        tactic,
        cache_key=intermediate_key,
        arm="candidate",
        component=FC1_CUMULATIVE,
        comparison_tactic=tactic,
        warmups=warmups,
        repetitions=repetitions,
        use_stock_fallback=False,
        zero_delta=zero_delta,
    )
    candidate_final_replays = _profile_component_replays(
        case,
        packed_topk,
        tactic,
        cache_key=final_key,
        arm="candidate",
        component=PAIR_CUMULATIVE,
        comparison_tactic=tactic,
        warmups=warmups,
        repetitions=repetitions,
        use_stock_fallback=False,
        zero_delta=zero_delta,
    )
    candidate_intermediate = candidate_intermediate_replays.outputs[0]
    candidate_final = candidate_final_replays.outputs[0]
    timings_us = list(candidate_final_replays.timings_us)
    replay_outputs = list(candidate_final_replays.outputs)

    if not torch.equal(packed_topk, original_routing):
        raise RuntimeError("FlashInfer modified packed top-k routing inputs")
    all_outputs = [
        candidate_final,
        candidate_intermediate,
        *candidate_intermediate_replays.outputs,
        *replay_outputs,
    ]
    finite = all(bool(torch.isfinite(output).all().item()) for output in all_outputs)
    deterministic = all(
        torch.equal(candidate_intermediate, output)
        for output in candidate_intermediate_replays.outputs
    ) and all(torch.equal(candidate_final, output) for output in replay_outputs)
    max_abs_error = max(
        _max_abs_error(candidate_final, stock_final),
        _max_abs_error(candidate_intermediate, stock_intermediate),
        *(_max_abs_error(output, stock_final) for output in replay_outputs),
    )
    cosine_similarity = min(
        _tensor_cosine(candidate_final, stock_final),
        _tensor_cosine(candidate_intermediate, stock_intermediate),
        *(_tensor_cosine(output, stock_final) for output in replay_outputs),
    )
    median_us = float(statistics.median(timings_us))
    sorted_timings = sorted(timings_us)
    p95_us = float(sorted_timings[math.ceil(0.95 * len(sorted_timings)) - 1])
    mean_us = statistics.fmean(timings_us)
    cv = float(statistics.pstdev(timings_us) / mean_us) if mean_us > 0 else 0.0
    return _ProfileResult(
        median_us=median_us,
        p95_us=p95_us,
        cv=cv,
        finite=finite,
        deterministic=deterministic,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
    )


def _profile_tactic_pair_cuda(
    case: MoeKernelCase,
    tactic: TacticPair,
    *,
    warmups: int,
    repetitions: int,
    stock_tactics: Mapping[str, TacticPair] | None = None,
    monolithic_replay: bool = False,
) -> _ProfileResult:
    """Profile the complete FC1+FC2 tactic pair when FC1 output is private."""
    if case.hidden_states.device.type != "cuda":
        raise ValueError("MXFP8 MoE tactic profiling requires CUDA")
    if monolithic_replay:
        routing_input = reconstruct_router_logits(
            case.profile, case.hidden_states.device
        )
    else:
        routing_input, _ = reconstruct_topk(case.profile, case.hidden_states.device)
    original_routing = routing_input.clone()
    final_key = cache_key_for_case(
        case,
        has_gemm1_lora_delta=False,
        router_logits=routing_input if monolithic_replay else None,
    )
    stock_tactic = None if stock_tactics is None else stock_tactics.get(final_key)
    if stock_tactics is not None and stock_tactic is None:
        raise ValueError("stock cache has no tactic for the replayed pair key")

    stock_replays = _profile_component_replays(
        case,
        routing_input,
        tactic if stock_tactic is None else stock_tactic,
        cache_key=final_key,
        arm="stock",
        component=PAIR_CUMULATIVE,
        comparison_tactic=tactic,
        warmups=warmups,
        repetitions=repetitions,
        use_stock_fallback=stock_tactic is None,
        zero_delta=None,
        monolithic_replay=monolithic_replay,
    )
    candidate_replays = _profile_component_replays(
        case,
        routing_input,
        tactic,
        cache_key=final_key,
        arm="candidate",
        component=PAIR_CUMULATIVE,
        comparison_tactic=tactic,
        warmups=warmups,
        repetitions=repetitions,
        use_stock_fallback=False,
        zero_delta=None,
        monolithic_replay=monolithic_replay,
    )
    stock_output = stock_replays.outputs[0]
    candidate_output = candidate_replays.outputs[0]
    if not torch.equal(routing_input, original_routing):
        raise RuntimeError("FlashInfer modified routing inputs")
    all_outputs = [stock_output, *candidate_replays.outputs]
    finite = all(bool(torch.isfinite(output).all().item()) for output in all_outputs)
    deterministic = all(
        torch.equal(candidate_output, output) for output in candidate_replays.outputs
    )
    max_abs_error = max(
        _max_abs_error(output, stock_output) for output in candidate_replays.outputs
    )
    cosine_similarity = min(
        _tensor_cosine(output, stock_output) for output in candidate_replays.outputs
    )
    timings_us = list(candidate_replays.timings_us)
    median_us = float(statistics.median(timings_us))
    sorted_timings = sorted(timings_us)
    p95_us = float(sorted_timings[math.ceil(0.95 * len(sorted_timings)) - 1])
    mean_us = statistics.fmean(timings_us)
    cv = float(statistics.pstdev(timings_us) / mean_us) if mean_us > 0 else 0.0
    return _ProfileResult(
        median_us=median_us,
        p95_us=p95_us,
        cv=cv,
        finite=finite,
        deterministic=deterministic,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
    )


def _failure_measurement(
    case: MoeKernelCase,
    tactic: TacticPair,
    *,
    warmups: int,
    repetitions: int,
    failure: str,
) -> TacticMeasurement:
    return TacticMeasurement(
        signature_key=case.profile.signature_key,
        tactic=tactic,
        median_us=0.0,
        p95_us=0.0,
        cv=0.0,
        warmups=warmups,
        repetitions=repetitions,
        finite=False,
        deterministic=False,
        max_abs_error=0.0,
        cosine_similarity=0.0,
        failure=failure,
    )


def profile_tactic(
    case: MoeKernelCase,
    tactic: TacticPair,
    warmups: int = 3,
    repetitions: int = 10,
    stock_tactics: Mapping[str, TacticPair] | None = None,
    pair_only: bool = False,
    monolithic_replay: bool = False,
) -> TacticMeasurement:
    """Profile and qualify one paired FC1/FC2 tactic without leaking failures."""
    if warmups != 3:
        raise ValueError("warmups must equal 3")
    if repetitions < 10:
        raise ValueError("repetitions must be at least 10")
    if monolithic_replay and not pair_only:
        raise ValueError("monolithic replay requires pair_only")
    try:
        profiler = _profile_tactic_pair_cuda if pair_only else _profile_tactic_cuda
        if monolithic_replay:
            result = _profile_tactic_pair_cuda(
                case,
                tactic,
                warmups=warmups,
                repetitions=repetitions,
                stock_tactics=stock_tactics,
                monolithic_replay=True,
            )
        else:
            result = profiler(
                case,
                tactic,
                warmups=warmups,
                repetitions=repetitions,
                stock_tactics=stock_tactics,
            )
    except IntermediateApiUnavailable:
        return _failure_measurement(
            case,
            tactic,
            warmups=warmups,
            repetitions=repetitions,
            failure="flashinfer_intermediate_api_unavailable",
        )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}".replace("\n", " ")
        return _failure_measurement(
            case,
            tactic,
            warmups=warmups,
            repetitions=repetitions,
            failure=failure,
        )
    return TacticMeasurement(
        signature_key=case.profile.signature_key,
        tactic=tactic,
        median_us=result.median_us,
        p95_us=result.p95_us,
        cv=result.cv,
        warmups=warmups,
        repetitions=repetitions,
        finite=result.finite,
        deterministic=result.deterministic,
        max_abs_error=result.max_abs_error,
        cosine_similarity=result.cosine_similarity,
        failure=None,
    )


def _load_profiles(path: Path) -> tuple[ReplayProfile, ...]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, Mapping):
        raise ValueError("profiles artifact must be a JSON object")
    raw_profiles = payload.get("selected_profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("profiles artifact must contain selected_profiles")
    return tuple(
        ReplayProfile.from_json(cast(Mapping[str, object], raw_profile))
        for raw_profile in raw_profiles
    )


def _parse_tactic_pair(value: str) -> TacticPair:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("tactic pair must be GEMM1,GEMM2")
    try:
        return TacticPair(*(int(part) for part in parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tactic pair must contain integer IDs"
        ) from error


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    weight_source = parser.add_mutually_exclusive_group()
    weight_source.add_argument("--weights", type=Path)
    weight_source.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--profile-limit", type=int)
    parser.add_argument("--tactic-limit", type=int)
    parser.add_argument("--tactic-pair", type=_parse_tactic_pair, action="append")
    parser.add_argument("--pair-only", action="store_true")
    parser.add_argument("--monolithic-replay", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stock-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shmoo and append one JSONL row for every paired tactic."""
    args = _parse_args(argv)
    if args.warmups != 3:
        raise SystemExit("warmups must equal 3")
    if args.repetitions < 10:
        raise SystemExit("repetitions must be at least 10")
    if args.profile_limit is not None and args.profile_limit <= 0:
        raise SystemExit("profile-limit must be positive")
    if args.tactic_limit is not None and args.tactic_limit <= 0:
        raise SystemExit("tactic-limit must be positive")
    if args.monolithic_replay and not args.pair_only:
        raise SystemExit("--monolithic-replay requires --pair-only")
    if args.monolithic_replay and (
        args.profile_limit is None or args.tactic_limit is None
    ):
        raise SystemExit(
            "--monolithic-replay requires explicit --profile-limit and --tactic-limit"
        )
    source_less = args.weights is None and not args.synthetic_smoke
    bounded_implicit_smoke = (
        source_less and args.profile_limit == 1 and args.tactic_limit == 2
    )
    if source_less and not bounded_implicit_smoke:
        raise SystemExit(
            "provide --weights or --synthetic-smoke unless using the bounded "
            "--profile-limit=1 --tactic-limit=2 smoke"
        )
    synthetic_source = args.synthetic_smoke or bounded_implicit_smoke
    if not synthetic_source and args.stock_cache is None:
        raise SystemExit("real shmoo runs require --stock-cache for NSys stock tactics")
    stock_tactics = (
        None if args.stock_cache is None else _load_stock_tactics(args.stock_cache)
    )
    if args.monolithic_replay:
        assert_monolithic_replay_supported()
    else:
        assert_supported_flashinfer()
    profiles = _load_profiles(args.profiles)
    if args.profile_limit is not None:
        profiles = profiles[: args.profile_limit]
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as output_file:
        for replay_profile in profiles:
            case = build_kernel_case(
                replay_profile,
                device,
                weights_path=args.weights,
                synthetic_smoke=synthetic_source,
            )
            tactics = enumerate_valid_tactics(case)
            if args.tactic_pair:
                requested = tuple(args.tactic_pair)
                legal_tactics = set(tactics)
                missing = [tactic for tactic in requested if tactic not in legal_tactics]
                if missing:
                    raise ValueError(f"requested tactic is not legal: {missing[0]}")
                tactics = requested
            if args.tactic_limit is not None:
                tactics = tactics[: args.tactic_limit]
            for tactic in tactics:
                if stock_tactics is None:
                    if args.monolithic_replay:
                        measurement = profile_tactic(
                            case,
                            tactic,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            pair_only=True,
                            monolithic_replay=True,
                        )
                    else:
                        measurement = profile_tactic(
                            case,
                            tactic,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            pair_only=args.pair_only,
                        )
                else:
                    if args.monolithic_replay:
                        measurement = profile_tactic(
                            case,
                            tactic,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            stock_tactics=stock_tactics,
                            pair_only=True,
                            monolithic_replay=True,
                        )
                    else:
                        measurement = profile_tactic(
                            case,
                            tactic,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            stock_tactics=stock_tactics,
                            pair_only=args.pair_only,
                        )
                row = measurement.to_json()
                if synthetic_source:
                    row["synthetic"] = True
                output_file.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                output_file.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
