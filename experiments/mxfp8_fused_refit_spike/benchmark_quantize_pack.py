#!/usr/bin/env python3
"""Measure producer-side MXFP8 quantization and IPC packing upper bounds."""

from __future__ import annotations

import argparse
import importlib
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


def _find_internal_quant_module():
    from flashinfer.quantization.fp8_quantization import (
        get_mxfp8_quantization_sm100_module,
    )

    public_fn = get_mxfp8_quantization_sm100_module().mxfp8_quantize_sm100
    for cell in public_fn.__closure__ or ():
        module = cell.cell_contents
        if hasattr(module, "mxfp8_quantize"):
            return module
    raise RuntimeError("FlashInfer internal MXFP8 quantization module is unavailable")


def _linear_sf_layout():
    for module_name in (
        "flashinfer.tllm_enums",
        "flashinfer.fp4_quantization",
        "flashinfer",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        enum = getattr(module, "SfLayout", None)
        if enum is not None and hasattr(enum, "layout_linear"):
            return enum.layout_linear
    raise RuntimeError("FlashInfer SfLayout.layout_linear is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--chunk-sizes", default="4,8,16,32,64,128")
    args = parser.parse_args()
    chunk_sizes = [
        size
        for value in args.chunk_sizes.split(",")
        if 0 < (size := int(value)) <= args.experts
    ]
    if not chunk_sizes:
        raise ValueError("chunk_sizes must contain a value in [1, experts]")

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
    separate_sources = tuple(
        tuple(source[expert_id].clone() for expert_id in range(args.experts))
        for source in sources
    )
    stack_buffers = [torch.empty_like(source) for source in sources]

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
    direct_values = [torch.empty_like(destination) for destination in current_values]
    direct_scales = [torch.empty_like(destination) for destination in current_scales]

    def current_quantize_only() -> None:
        for source in sources:
            for expert_id in range(args.experts):
                _value, scale = mxfp8_quantize(
                    source[expert_id],
                    is_sf_swizzled_layout=False,
                    alignment=32,
                )
                torch.where(scale == 0, torch.ones_like(scale), scale)

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
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                value_destination[expert_id].copy_(value)
                scale_destination[expert_id].copy_(
                    scale.view_as(scale_destination[expert_id])
                )

    def batched_quantize_only() -> None:
        for source in sources:
            _value, scale = mxfp8_quantize(
                source.reshape(-1, source.shape[-1]),
                is_sf_swizzled_layout=False,
                alignment=32,
            )
            torch.where(scale == 0, torch.ones_like(scale), scale)

    def batched_quantize_pack() -> None:
        for source, value_destination, scale_destination in zip(
            sources, batched_values, batched_scales, strict=True
        ):
            value, scale = mxfp8_quantize(
                source.reshape(-1, source.shape[-1]),
                is_sf_swizzled_layout=False,
                alignment=32,
            )
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            value_destination.copy_(value.view_as(value_destination))
            scale_destination.copy_(scale.view_as(scale_destination))

    def stack_only() -> None:
        for expert_sources, stack_destination in zip(
            separate_sources, stack_buffers, strict=True
        ):
            torch.stack(expert_sources, dim=0, out=stack_destination)

    def stacked_batched_quantize_pack() -> None:
        for (
            expert_sources,
            stack_destination,
            value_destination,
            scale_destination,
        ) in zip(
            separate_sources,
            stack_buffers,
            batched_values,
            batched_scales,
            strict=True,
        ):
            torch.stack(expert_sources, dim=0, out=stack_destination)
            value, scale = mxfp8_quantize(
                stack_destination.reshape(-1, stack_destination.shape[-1]),
                is_sf_swizzled_layout=False,
                alignment=32,
            )
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            value_destination.copy_(value.view_as(value_destination))
            scale_destination.copy_(scale.view_as(scale_destination))

    def chunked_quantize_pack(chunk_size: int) -> None:
        for (
            expert_sources,
            stack_destination,
            value_destination,
            scale_destination,
        ) in zip(
            separate_sources,
            stack_buffers,
            batched_values,
            batched_scales,
            strict=True,
        ):
            for start in range(0, args.experts, chunk_size):
                end = min(start + chunk_size, args.experts)
                count = end - start
                stack_chunk = stack_destination[:count]
                torch.stack(expert_sources[start:end], dim=0, out=stack_chunk)
                value, scale = mxfp8_quantize(
                    stack_chunk.reshape(-1, stack_chunk.shape[-1]),
                    is_sf_swizzled_layout=False,
                    alignment=32,
                )
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                value_destination[start:end].copy_(
                    value.view_as(value_destination[start:end])
                )
                scale_destination[start:end].copy_(
                    scale.view_as(scale_destination[start:end])
                )

    internal_module = None
    internal_error = None
    try:
        internal_module = _find_internal_quant_module()
        linear_layout = _linear_sf_layout()
    except Exception as error:  # noqa: BLE001
        internal_error = str(error)

    def direct_per_expert_quantize_pack() -> None:
        if internal_module is None:
            raise RuntimeError(internal_error)
        for source, value_destination, scale_destination in zip(
            sources, direct_values, direct_scales, strict=True
        ):
            for expert_id in range(args.experts):
                internal_module.mxfp8_quantize(
                    source[expert_id],
                    value_destination[expert_id],
                    scale_destination[expert_id].view(-1),
                    linear_layout.value,
                    32,
                    True,
                )
                scale_destination[expert_id].clamp_min_(1)

    def direct_batched_quantize_pack() -> None:
        if internal_module is None:
            raise RuntimeError(internal_error)
        for source, value_destination, scale_destination in zip(
            sources, direct_values, direct_scales, strict=True
        ):
            internal_module.mxfp8_quantize(
                source.reshape(-1, source.shape[-1]),
                value_destination.view(-1, value_destination.shape[-1]),
                scale_destination.view(-1),
                linear_layout.value,
                32,
                True,
            )
            scale_destination.clamp_min_(1)

    def stacked_direct_batched_quantize_pack() -> None:
        if internal_module is None:
            raise RuntimeError(internal_error)
        for (
            expert_sources,
            stack_destination,
            value_destination,
            scale_destination,
        ) in zip(
            separate_sources,
            stack_buffers,
            direct_values,
            direct_scales,
            strict=True,
        ):
            torch.stack(expert_sources, dim=0, out=stack_destination)
            internal_module.mxfp8_quantize(
                stack_destination.reshape(-1, stack_destination.shape[-1]),
                value_destination.view(-1, value_destination.shape[-1]),
                scale_destination.view(-1),
                linear_layout.value,
                32,
                True,
            )
            scale_destination.clamp_min_(1)

    current_quantize_pack()
    batched_quantize_pack()
    torch.cuda.synchronize()
    for current, batched in zip(current_values, batched_values, strict=True):
        if not torch.equal(current, batched):
            raise AssertionError("batched MXFP8 values differ from per-expert values")
    for current, batched in zip(current_scales, batched_scales, strict=True):
        if not torch.equal(current, batched):
            raise AssertionError("batched MXFP8 scales differ from per-expert scales")
    stacked_batched_quantize_pack()
    torch.cuda.synchronize()
    for current, batched in zip(current_values, batched_values, strict=True):
        if not torch.equal(current, batched):
            raise AssertionError("stacked MXFP8 values differ from per-expert values")
    for current, batched in zip(current_scales, batched_scales, strict=True):
        if not torch.equal(current, batched):
            raise AssertionError("stacked MXFP8 scales differ from per-expert scales")
    if internal_module is not None:
        direct_per_expert_quantize_pack()
        torch.cuda.synchronize()
        for current, direct in zip(current_values, direct_values, strict=True):
            if not torch.equal(current, direct):
                raise AssertionError("direct MXFP8 values differ from public values")
        for current, direct in zip(current_scales, direct_scales, strict=True):
            if not torch.equal(current, direct):
                raise AssertionError("direct MXFP8 scales differ from public scales")
        stacked_direct_batched_quantize_pack()
        torch.cuda.synchronize()
        for current, direct in zip(current_values, direct_values, strict=True):
            if not torch.equal(current, direct):
                raise AssertionError(
                    "stacked direct MXFP8 values differ from public values"
                )
        for current, direct in zip(current_scales, direct_scales, strict=True):
            if not torch.equal(current, direct):
                raise AssertionError(
                    "stacked direct MXFP8 scales differ from public scales"
                )

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
    stack_only_ms = _measure_wall_ms(stack_only, args.warmup, args.repetitions)
    stacked_batched_quantize_pack_ms = _measure_wall_ms(
        stacked_batched_quantize_pack, args.warmup, args.repetitions
    )
    chunked_results = {}
    max_expert_numel = max(source[0].numel() for source in sources)
    max_scale_numel = max(source[0].numel() // 32 for source in sources)
    for chunk_size in chunk_sizes:
        latency_ms = _measure_wall_ms(
            lambda chunk_size=chunk_size: chunked_quantize_pack(chunk_size),
            args.warmup,
            args.repetitions,
        )
        scratch_bytes = (
            chunk_size * max_expert_numel * torch.empty((), dtype=bf16).element_size()
        )
        pending_source_bytes = 2 * scratch_bytes
        quantized_output_bytes = chunk_size * (
            max_expert_numel * torch.empty((), dtype=value_dtype).element_size()
            + max_scale_numel * torch.empty((), dtype=scale_dtype).element_size()
        )
        chunked_results[str(chunk_size)] = {
            "calls": 3 * ((args.experts + chunk_size - 1) // chunk_size),
            "latency_ms": latency_ms,
            "speedup": current_quantize_pack_ms / latency_ms,
            "scratch_bytes": scratch_bytes,
            "additional_live_bytes_upper_bound": (
                pending_source_bytes + scratch_bytes + quantized_output_bytes
            ),
        }
    direct_per_expert_ms = None
    direct_batched_ms = None
    stacked_direct_batched_ms = None
    if internal_module is not None:
        direct_per_expert_ms = _measure_wall_ms(
            direct_per_expert_quantize_pack, args.warmup, args.repetitions
        )
        direct_batched_ms = _measure_wall_ms(
            direct_batched_quantize_pack, args.warmup, args.repetitions
        )
        stacked_direct_batched_ms = _measure_wall_ms(
            stacked_direct_batched_quantize_pack,
            args.warmup,
            args.repetitions,
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
        "stack_only_ms": stack_only_ms,
        "stacked_batched_quantize_pack_ms": stacked_batched_quantize_pack_ms,
        "chunked_quantize_pack": chunked_results,
        "direct_per_expert_quantize_pack_ms": direct_per_expert_ms,
        "direct_batched_quantize_pack_ms": direct_batched_ms,
        "stacked_direct_batched_quantize_pack_ms": stacked_direct_batched_ms,
        "internal_output_api_error": internal_error,
        "batched_pack_speedup": current_quantize_pack_ms / batched_quantize_pack_ms,
        "batched_pack_latency_reduction_pct": 100
        * (current_quantize_pack_ms - batched_quantize_pack_ms)
        / current_quantize_pack_ms,
        "direct_output_upper_bound_speedup": current_quantize_pack_ms
        / batched_quantize_only_ms,
        "direct_output_upper_bound_latency_reduction_pct": 100
        * (current_quantize_pack_ms - batched_quantize_only_ms)
        / current_quantize_pack_ms,
        "direct_batched_speedup": (
            current_quantize_pack_ms / direct_batched_ms
            if direct_batched_ms is not None
            else None
        ),
        "direct_batched_latency_reduction_pct": (
            100
            * (current_quantize_pack_ms - direct_batched_ms)
            / current_quantize_pack_ms
            if direct_batched_ms is not None
            else None
        ),
        "stacked_batched_speedup": current_quantize_pack_ms
        / stacked_batched_quantize_pack_ms,
        "stacked_batched_latency_reduction_pct": 100
        * (current_quantize_pack_ms - stacked_batched_quantize_pack_ms)
        / current_quantize_pack_ms,
        "stacked_direct_batched_speedup": (
            current_quantize_pack_ms / stacked_direct_batched_ms
            if stacked_direct_batched_ms is not None
            else None
        ),
        "stacked_direct_batched_latency_reduction_pct": (
            100
            * (current_quantize_pack_ms - stacked_direct_batched_ms)
            / current_quantize_pack_ms
            if stacked_direct_batched_ms is not None
            else None
        ),
        "value_and_scale_parity": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
