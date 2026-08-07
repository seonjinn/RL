"""Workload-replayed FlashInfer MXFP8 MoE tactic shmoo."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
        assert_supported_flashinfer,
        build_kernel_case,
        cache_key_for_case,
        enumerate_valid_tactics,
        force_stock_tactic,
        force_tactic,
        run_moe_pair,
    )
    from .schema import ReplayProfile, TacticMeasurement, TacticPair
except ImportError:  # pragma: no cover - direct script execution
    from flashinfer_adapter import (
        IntermediateApiUnavailable,
        MoeKernelCase,
        MoePairResult,
        TacticDispatchError,
        assert_supported_flashinfer,
        build_kernel_case,
        cache_key_for_case,
        enumerate_valid_tactics,
        force_stock_tactic,
        force_tactic,
        run_moe_pair,
    )
    from schema import ReplayProfile, TacticMeasurement, TacticPair


@dataclass(frozen=True)
class _ProfileResult:
    median_us: float
    p95_us: float
    cv: float
    finite: bool
    deterministic: bool
    max_abs_error: float
    cosine_similarity: float


def _seeded_order(signature_key: str, index: int) -> bytes:
    return sha256(f"{signature_key}:{index}".encode("ascii")).digest()


def reconstruct_topk(
    profile: ReplayProfile, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct deterministic packed routing with the exact expert histogram."""
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


def _profile_tactic_cuda(
    case: MoeKernelCase,
    tactic: TacticPair,
    *,
    warmups: int,
    repetitions: int,
) -> _ProfileResult:
    if case.hidden_states.device.type != "cuda":
        raise ValueError("MXFP8 MoE tactic profiling requires CUDA")
    packed_topk, _ = reconstruct_topk(case.profile, case.hidden_states.device)
    original_routing = packed_topk.clone()
    signature = case.profile.signature
    final_key = cache_key_for_case(case, has_gemm1_lora_delta=False)
    intermediate_key = cache_key_for_case(case, has_gemm1_lora_delta=True)
    zero_delta = torch.zeros(
        signature.num_tokens,
        signature.top_k,
        2 * signature.intermediate_size,
        dtype=torch.bfloat16,
        device=case.hidden_states.device,
    )

    with force_stock_tactic(final_key):
        stock_final = _final_output(
            run_moe_pair(case, packed_topk, do_finalize=True, gemm1_lora_delta=None)
        ).clone()
    try:
        with force_stock_tactic(intermediate_key):
            stock_intermediate = _intermediate_output(
                run_moe_pair(
                    case,
                    packed_topk,
                    do_finalize=False,
                    gemm1_lora_delta=zero_delta,
                )
            ).clone()
    except TacticDispatchError:
        raise
    except Exception as error:
        raise IntermediateApiUnavailable(
            "FlashInfer stock intermediate preflight is unavailable"
        ) from error

    if not bool(torch.isfinite(stock_final).all().item()):
        raise RuntimeError("stock FC2 output reference is not finite")
    if not bool(torch.isfinite(stock_intermediate).all().item()):
        raise RuntimeError("stock FC1 intermediate reference is not finite")

    with force_tactic(intermediate_key, tactic):
        candidate_intermediate = _intermediate_output(
            run_moe_pair(
                case,
                packed_topk,
                do_finalize=False,
                gemm1_lora_delta=zero_delta,
            )
        ).clone()
        repeated_intermediate = _intermediate_output(
            run_moe_pair(
                case,
                packed_topk,
                do_finalize=False,
                gemm1_lora_delta=zero_delta,
            )
        ).clone()

    timings_us: list[float] = []
    replay_outputs: list[torch.Tensor] = []
    with force_tactic(final_key, tactic):
        candidate_final = _final_output(
            run_moe_pair(case, packed_topk, do_finalize=True, gemm1_lora_delta=None)
        ).clone()
        for _ in range(warmups):
            run_moe_pair(case, packed_topk, do_finalize=True, gemm1_lora_delta=None)
        torch.cuda.synchronize(case.hidden_states.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_result = run_moe_pair(
                case, packed_topk, do_finalize=True, gemm1_lora_delta=None
            )
        graph_output = _final_output(captured_result)
        cold_l2 = torch.empty(
            _l2_size_bytes(case.hidden_states.device) + 4 * 1024 * 1024,
            dtype=torch.uint8,
            device=case.hidden_states.device,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repetitions):
            cold_l2.add_(1)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings_us.append(float(start.elapsed_time(end) * 1000.0))
            replay_outputs.append(graph_output.clone())

    if not torch.equal(packed_topk, original_routing):
        raise RuntimeError("FlashInfer modified packed top-k routing inputs")
    all_outputs = [candidate_final, candidate_intermediate, *replay_outputs]
    finite = all(bool(torch.isfinite(output).all().item()) for output in all_outputs)
    deterministic = torch.equal(candidate_intermediate, repeated_intermediate) and all(
        torch.equal(candidate_final, output) for output in replay_outputs
    )
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
) -> TacticMeasurement:
    """Profile and qualify one paired FC1/FC2 tactic without leaking failures."""
    if warmups != 3:
        raise ValueError("warmups must equal 3")
    if repetitions < 10:
        raise ValueError("repetitions must be at least 10")
    try:
        result = _profile_tactic_cuda(
            case, tactic, warmups=warmups, repetitions=repetitions
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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    weight_source = parser.add_mutually_exclusive_group(required=True)
    weight_source.add_argument("--weights", type=Path)
    weight_source.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--profile-limit", type=int)
    parser.add_argument("--tactic-limit", type=int)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
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
                synthetic_smoke=args.synthetic_smoke,
            )
            tactics = enumerate_valid_tactics(case)
            if args.tactic_limit is not None:
                tactics = tactics[: args.tactic_limit]
            for tactic in tactics:
                measurement = profile_tactic(
                    case,
                    tactic,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                )
                output_file.write(
                    json.dumps(
                        measurement.to_json(),
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
