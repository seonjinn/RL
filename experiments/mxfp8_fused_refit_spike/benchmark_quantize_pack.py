#!/usr/bin/env python3
"""Measure producer-side MXFP8 quantization and IPC packing upper bounds."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch


def _measure_wall_ms(fn: Callable[[], None], warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000 / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    from flashinfer import mxfp8_quantize

    device = torch.device("cuda")
    bf16 = torch.bfloat16
    value_dtype = torch.float8_e4m3fn
    scale_dtype = torch.uint8
    shape = (args.experts, args.intermediate_size, args.hidden_size)
    w1 = torch.full(shape, 0.5, dtype=bf16, device=device)
    w3 = torch.full(shape, -0.25, dtype=bf16, device=device)
    w2 = torch.full(
        (args.experts, args.hidden_size, args.intermediate_size),
        0.75,
        dtype=bf16,
        device=device,
    )
    sources = (w1, w3, w2)

    current_values = [torch.empty_like(source, dtype=value_dtype) for source in sources]
    current_scales = [
        torch.empty(
            (*source.shape[:-1], source.shape[-1] // 32),
            dtype=scale_dtype,
            device=device,
        )
        for source in sources
    ]
    batched_values = [torch.empty_like(destination) for destination in current_values]
    batched_scales = [torch.empty_like(destination) for destination in current_scales]

    def current_quantize_only() -> None:
        for source in sources:
            for expert_id in range(args.experts):
                mxfp8_quantize(
                    source[expert_id],
                    is_sf_swizzled_layout=False,
                    alignment=32,
                )

    def current_quantize_pack() -> None:
        for source, value_destination, scale_destination in zip(
            sources, current_values, current_scales, strict=True
        ):
            for expert_id in range(args.experts):
                value, scale = mxfp8_quantize(
                    source[expert_id],
                    is_sf_swizzled_layout=False,
                    alignment=32,
                )
                value_destination[expert_id].copy_(value)
                scale_destination[expert_id].copy_(scale)

    def batched_quantize_only() -> None:
        for source in sources:
            mxfp8_quantize(
                source,
                is_sf_swizzled_layout=False,
                alignment=32,
            )

    def batched_quantize_pack() -> None:
        for source, value_destination, scale_destination in zip(
            sources, batched_values, batched_scales, strict=True
        ):
            value, scale = mxfp8_quantize(
                source,
                is_sf_swizzled_layout=False,
                alignment=32,
            )
            value_destination.copy_(value)
            scale_destination.copy_(scale)

    current_quantize_pack()
    batched_quantize_pack()
    torch.cuda.synchronize()
    for current, batched in zip(current_values, batched_values, strict=True):
        if not torch.equal(current, batched):
            raise AssertionError("batched MXFP8 values differ from per-expert values")
    for current, batched in zip(current_scales, batched_scales, strict=True):
        if not torch.equal(current, batched):
            raise AssertionError("batched MXFP8 scales differ from per-expert scales")

    current_quantize_only_ms = _measure_wall_ms(
        current_quantize_only, args.warmup, args.repetitions
    )
    current_quantize_pack_ms = _measure_wall_ms(
        current_quantize_pack, args.warmup, args.repetitions
    )
    batched_quantize_only_ms = _measure_wall_ms(
        batched_quantize_only, args.warmup, args.repetitions
    )
    batched_quantize_pack_ms = _measure_wall_ms(
        batched_quantize_pack, args.warmup, args.repetitions
    )

    bf16_bytes = sum(source.nbytes for source in sources)
    packed_bytes = sum(tensor.nbytes for tensor in current_values + current_scales)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "experts": args.experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
        },
        "bf16_source_bytes": bf16_bytes,
        "mxfp8_packed_bytes": packed_bytes,
        "per_expert_calls": 3 * args.experts,
        "batched_calls": 3,
        "current_quantize_only_ms": current_quantize_only_ms,
        "current_quantize_pack_ms": current_quantize_pack_ms,
        "batched_quantize_only_ms": batched_quantize_only_ms,
        "batched_quantize_pack_ms": batched_quantize_pack_ms,
        "batched_pack_speedup": current_quantize_pack_ms / batched_quantize_pack_ms,
        "batched_pack_latency_reduction_pct": 100
        * (current_quantize_pack_ms - batched_quantize_pack_ms)
        / current_quantize_pack_ms,
        "direct_output_upper_bound_speedup": current_quantize_pack_ms
        / batched_quantize_only_ms,
        "direct_output_upper_bound_latency_reduction_pct": 100
        * (current_quantize_pack_ms - batched_quantize_only_ms)
        / current_quantize_pack_ms,
        "value_and_scale_parity": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
