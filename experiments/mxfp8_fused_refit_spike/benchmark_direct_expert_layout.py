#!/usr/bin/env python3

import argparse
import json
from collections.abc import Callable

import torch


def _measure_ms(fn: Callable[[], None], warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def _row_permutations(
    w13: torch.Tensor, w2: torch.Tensor, intermediate_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    epilogue_tile_m = 128
    w13_perm = get_shuffle_matrix_a_row_indices(w13[0], epilogue_tile_m)
    reorder = get_reorder_rows_for_gated_act_gemm_row_indices(w13[0])
    w13_perm = reorder[w13_perm].to(w13.device)
    w2_perm = get_shuffle_matrix_a_row_indices(w2[0], epilogue_tile_m).to(w2.device)

    swap = torch.cat(
        (
            torch.arange(intermediate_size, 2 * intermediate_size, device=w13.device),
            torch.arange(0, intermediate_size, device=w13.device),
        )
    )
    return w13_perm, swap[w13_perm], w2_perm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--quantize-warmup", type=int, default=3)
    parser.add_argument("--quantize-repetitions", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float8_e4m3fn
    w13_source = torch.empty(
        (args.experts, 2 * args.intermediate_size, args.hidden_size),
        dtype=dtype,
        device=device,
    )
    w2_source = torch.empty(
        (args.experts, args.hidden_size, args.intermediate_size),
        dtype=dtype,
        device=device,
    )
    w13_source.copy_(
        (torch.arange(w13_source.shape[1], device=device) % 16)
        .to(dtype)
        .view(1, -1, 1)
    )
    w2_source.copy_(
        (torch.arange(w2_source.shape[1], device=device) % 16)
        .to(dtype)
        .view(1, -1, 1)
    )
    w13_live = torch.empty_like(w13_source)
    w2_live = torch.empty_like(w2_source)
    w13_scratch = torch.empty_like(w13_source, dtype=torch.uint8)
    w2_scratch = torch.empty_like(w2_source, dtype=torch.uint8)

    w13_perm, w13_source_rows, w2_source_rows = _row_permutations(
        w13_source, w2_source, args.intermediate_size
    )

    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )

    def current_path() -> None:
        w13_live.copy_(w13_source)
        w2_live.copy_(w2_source)
        w13_swapped = swap_w13_to_w31(w13_live)
        torch.index_select(
            w13_swapped.view(torch.uint8),
            1,
            w13_perm,
            out=w13_scratch,
        )
        torch.index_select(
            w2_live.view(torch.uint8), 1, w2_source_rows, out=w2_scratch
        )
        w13_live.copy_(w13_scratch.view(dtype))
        w2_live.copy_(w2_scratch.view(dtype))

    def direct_path() -> None:
        torch.index_select(
            w13_source.view(torch.uint8), 1, w13_source_rows, out=w13_live.view(torch.uint8)
        )
        torch.index_select(
            w2_source.view(torch.uint8), 1, w2_source_rows, out=w2_live.view(torch.uint8)
        )

    def composed_path() -> None:
        w13_live.copy_(w13_source)
        w2_live.copy_(w2_source)
        torch.index_select(
            w13_live.view(torch.uint8), 1, w13_source_rows, out=w13_scratch
        )
        torch.index_select(
            w2_live.view(torch.uint8), 1, w2_source_rows, out=w2_scratch
        )
        w13_live.copy_(w13_scratch.view(dtype))
        w2_live.copy_(w2_scratch.view(dtype))

    current_path()
    current_w13 = w13_live.clone()
    current_w2 = w2_live.clone()
    composed_path()
    torch.cuda.synchronize()
    if not torch.equal(current_w13, w13_live) or not torch.equal(current_w2, w2_live):
        raise AssertionError("composed layout output does not match the current path")
    direct_path()
    torch.cuda.synchronize()
    if not torch.equal(current_w13, w13_live) or not torch.equal(current_w2, w2_live):
        raise AssertionError("direct layout output does not match the current path")

    current_ms = _measure_ms(current_path, args.warmup, args.repetitions)
    composed_ms = _measure_ms(composed_path, args.warmup, args.repetitions)
    direct_ms = _measure_ms(direct_path, args.warmup, args.repetitions)

    from flashinfer import mxfp8_quantize

    w13_bf16 = torch.empty_like(w13_source, dtype=torch.bfloat16)
    w2_bf16 = torch.empty_like(w2_source, dtype=torch.bfloat16)
    w13_bf16.copy_(w13_source)
    w2_bf16.copy_(w2_source)

    def current_quantize_path() -> None:
        w13_quantized, _ = mxfp8_quantize(
            w13_bf16, is_sf_swizzled_layout=False, alignment=32
        )
        w2_quantized, _ = mxfp8_quantize(
            w2_bf16, is_sf_swizzled_layout=False, alignment=32
        )
        w13_live.copy_(w13_quantized)
        w2_live.copy_(w2_quantized)
        w13_swapped = swap_w13_to_w31(w13_live)
        torch.index_select(
            w13_swapped.view(torch.uint8), 1, w13_perm, out=w13_scratch
        )
        torch.index_select(
            w2_live.view(torch.uint8), 1, w2_source_rows, out=w2_scratch
        )
        w13_live.copy_(w13_scratch.view(dtype))
        w2_live.copy_(w2_scratch.view(dtype))

    def direct_quantize_path() -> None:
        w13_quantized, _ = mxfp8_quantize(
            w13_bf16, is_sf_swizzled_layout=False, alignment=32
        )
        w2_quantized, _ = mxfp8_quantize(
            w2_bf16, is_sf_swizzled_layout=False, alignment=32
        )
        torch.index_select(
            w13_quantized.view(torch.uint8),
            1,
            w13_source_rows,
            out=w13_live.view(torch.uint8),
        )
        torch.index_select(
            w2_quantized.view(torch.uint8),
            1,
            w2_source_rows,
            out=w2_live.view(torch.uint8),
        )

    def composed_quantize_path() -> None:
        w13_quantized, _ = mxfp8_quantize(
            w13_bf16, is_sf_swizzled_layout=False, alignment=32
        )
        w2_quantized, _ = mxfp8_quantize(
            w2_bf16, is_sf_swizzled_layout=False, alignment=32
        )
        w13_live.copy_(w13_quantized)
        w2_live.copy_(w2_quantized)
        torch.index_select(
            w13_live.view(torch.uint8), 1, w13_source_rows, out=w13_scratch
        )
        torch.index_select(
            w2_live.view(torch.uint8), 1, w2_source_rows, out=w2_scratch
        )
        w13_live.copy_(w13_scratch.view(dtype))
        w2_live.copy_(w2_scratch.view(dtype))

    current_quantize_ms = _measure_ms(
        current_quantize_path, args.quantize_warmup, args.quantize_repetitions
    )
    composed_quantize_ms = _measure_ms(
        composed_quantize_path, args.quantize_warmup, args.quantize_repetitions
    )
    direct_quantize_ms = _measure_ms(
        direct_quantize_path, args.quantize_warmup, args.quantize_repetitions
    )
    tensor_bytes = w13_source.numel() + w2_source.numel()
    result = {
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "experts": args.experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
        },
        "weight_bytes": tensor_bytes,
        "current_ms": current_ms,
        "composed_ms": composed_ms,
        "direct_ms": direct_ms,
        "composed_speedup": current_ms / composed_ms,
        "speedup": current_ms / direct_ms,
        "composed_latency_reduction_pct": 100.0
        * (current_ms - composed_ms)
        / current_ms,
        "latency_reduction_pct": 100.0 * (current_ms - direct_ms) / current_ms,
        "quantize_and_layout": {
            "current_ms": current_quantize_ms,
            "composed_ms": composed_quantize_ms,
            "direct_ms": direct_quantize_ms,
            "composed_speedup": current_quantize_ms / composed_quantize_ms,
            "speedup": current_quantize_ms / direct_quantize_ms,
            "composed_latency_reduction_pct": 100.0
            * (current_quantize_ms - composed_quantize_ms)
            / current_quantize_ms,
            "latency_reduction_pct": 100.0
            * (current_quantize_ms - direct_quantize_ms)
            / current_quantize_ms,
        },
        "current_full_tensor_scratch_bytes": tensor_bytes,
        "direct_full_tensor_scratch_bytes": 0,
        "value_parity": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
