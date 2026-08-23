#!/usr/bin/env python3

import argparse
import json
from collections.abc import Callable

import torch


def _measure_ms(
    fn: Callable[[], None], *, device: torch.device, warmup: int, repetitions: int
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    for _ in range(repetitions):
        fn()
    end.record(torch.cuda.current_stream(device))
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--destination-device", type=int, default=1)
    args = parser.parse_args()

    source_device = torch.device("cuda", args.source_device)
    destination_device = torch.device("cuda", args.destination_device)
    if not torch.cuda.can_device_access_peer(
        args.source_device, args.destination_device
    ):
        raise RuntimeError(
            f"GPU {args.source_device} cannot access GPU {args.destination_device}"
        )

    value_dtype = torch.float8_e4m3fn
    scale_dtype = torch.uint8
    e = args.experts
    i = args.intermediate_size
    k = args.hidden_size

    def source(shape: tuple[int, ...], dtype: torch.dtype, offset: int) -> torch.Tensor:
        with torch.cuda.device(source_device):
            values = torch.arange(shape[-2], device=source_device, dtype=torch.int32)
            values = ((values + offset) % 16).to(dtype).view(1, -1, 1)
            return values.expand(shape).contiguous()

    w1 = source((e, i, k), value_dtype, 1)
    w3 = source((e, i, k), value_dtype, 3)
    w2 = source((e, k, i), value_dtype, 5)
    s1 = source((e, i, k // 32), scale_dtype, 7)
    s3 = source((e, i, k // 32), scale_dtype, 9)
    s2 = source((e, k, i // 32), scale_dtype, 11)

    with torch.cuda.device(source_device):
        prepared_w13 = torch.cat((w1, w3), dim=1)
        prepared_s13 = torch.cat((s1, s3), dim=1)

    with torch.cuda.device(destination_device):
        dst_w13 = torch.empty(
            (e, 2 * i, k), dtype=value_dtype, device=destination_device
        )
        dst_w2 = torch.empty((e, k, i), dtype=value_dtype, device=destination_device)
        dst_s13 = torch.empty(
            (e, 2 * i, k // 32), dtype=scale_dtype, device=destination_device
        )
        dst_s2 = torch.empty(
            (e, k, i // 32), dtype=scale_dtype, device=destination_device
        )

    def per_expert_copy() -> None:
        with torch.cuda.device(destination_device):
            for expert in range(e):
                dst_w13[expert, :i].copy_(w1[expert], non_blocking=True)
                dst_w13[expert, i:].copy_(w3[expert], non_blocking=True)
                dst_w2[expert].copy_(w2[expert], non_blocking=True)
                dst_s13[expert, :i].copy_(s1[expert], non_blocking=True)
                dst_s13[expert, i:].copy_(s3[expert], non_blocking=True)
                dst_s2[expert].copy_(s2[expert], non_blocking=True)

    def prepared_batched_copy() -> None:
        with torch.cuda.device(destination_device):
            dst_w13.copy_(prepared_w13, non_blocking=True)
            dst_w2.copy_(w2, non_blocking=True)
            dst_s13.copy_(prepared_s13, non_blocking=True)
            dst_s2.copy_(s2, non_blocking=True)

    per_expert_copy()
    torch.cuda.synchronize(destination_device)
    reference = tuple(tensor.clone() for tensor in (dst_w13, dst_w2, dst_s13, dst_s2))
    prepared_batched_copy()
    torch.cuda.synchronize(destination_device)
    for actual, expected in zip((dst_w13, dst_w2, dst_s13, dst_s2), reference):
        if not torch.equal(actual, expected):
            raise AssertionError("batched expert copy does not match per-expert copy")

    per_expert_ms = _measure_ms(
        per_expert_copy,
        device=destination_device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    prepared_batched_ms = _measure_ms(
        prepared_batched_copy,
        device=destination_device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    payload_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (prepared_w13, w2, prepared_s13, s2)
    )
    print(
        json.dumps(
            {
                "source_gpu": torch.cuda.get_device_name(source_device),
                "destination_gpu": torch.cuda.get_device_name(destination_device),
                "shape": {
                    "experts": e,
                    "hidden_size": k,
                    "intermediate_size": i,
                },
                "payload_bytes": payload_bytes,
                "per_expert_copy_count": 6 * e,
                "prepared_batched_copy_count": 4,
                "per_expert_ms": per_expert_ms,
                "prepared_batched_ms": prepared_batched_ms,
                "speedup": per_expert_ms / prepared_batched_ms,
                "latency_reduction_pct": 100.0
                * (per_expert_ms - prepared_batched_ms)
                / per_expert_ms,
                "value_parity": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
