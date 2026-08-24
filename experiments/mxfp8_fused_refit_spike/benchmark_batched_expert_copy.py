#!/usr/bin/env python3

import argparse
import json
import time
from collections.abc import Callable

import torch


def _measure(
    fn: Callable[[], None], *, device: torch.device, warmup: int, repetitions: int
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    wall_start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    end.record(torch.cuda.current_stream(device))
    end.synchronize()
    wall_ms = 1000.0 * (time.perf_counter() - wall_start) / repetitions
    return start.elapsed_time(end) / repetitions, wall_ms


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
        remote_w1_views = list(w1.unbind())
        remote_w3_views = list(w3.unbind())
        remote_w2_views = list(w2.unbind())
        remote_s1_views = list(s1.unbind())
        remote_s3_views = list(s3.unbind())
        remote_s2_views = list(s2.unbind())

    with torch.cuda.device(destination_device):
        local_w1 = w1.to(destination_device)
        local_w3 = w3.to(destination_device)
        local_w2 = w2.to(destination_device)
        local_s1 = s1.to(destination_device)
        local_s3 = s3.to(destination_device)
        local_s2 = s2.to(destination_device)
        w1_views = list(local_w1.unbind())
        w3_views = list(local_w3.unbind())
        w2_views = list(local_w2.unbind())
        s1_views = list(local_s1.unbind())
        s3_views = list(local_s3.unbind())
        s2_views = list(local_s2.unbind())
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
        stack_w13_half = torch.empty_like(local_w1)
        stack_w2 = torch.empty_like(local_w2)
        stack_s13_half = torch.empty_like(local_s1)
        stack_s2 = torch.empty_like(local_s2)
        dst_w1_views = list(dst_w13[:, :i].unbind())
        dst_w3_views = list(dst_w13[:, i:].unbind())
        dst_w2_views = list(dst_w2.unbind())
        dst_s1_views = list(dst_s13[:, :i].unbind())
        dst_s3_views = list(dst_s13[:, i:].unbind())
        dst_s2_views = list(dst_s2.unbind())

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

    def per_expert_local_copy() -> None:
        with torch.cuda.device(destination_device):
            for expert in range(e):
                dst_w13[expert, :i].copy_(local_w1[expert], non_blocking=True)
                dst_w13[expert, i:].copy_(local_w3[expert], non_blocking=True)
                dst_w2[expert].copy_(local_w2[expert], non_blocking=True)
                dst_s13[expert, :i].copy_(local_s1[expert], non_blocking=True)
                dst_s13[expert, i:].copy_(local_s3[expert], non_blocking=True)
                dst_s2[expert].copy_(local_s2[expert], non_blocking=True)

    def receiver_stack_copy() -> None:
        with torch.cuda.device(destination_device):
            torch.stack(w1_views, out=stack_w13_half)
            dst_w13[:, :i].copy_(stack_w13_half)
            torch.stack(w3_views, out=stack_w13_half)
            dst_w13[:, i:].copy_(stack_w13_half)
            torch.stack(w2_views, out=stack_w2)
            dst_w2.copy_(stack_w2)
            torch.stack(s1_views, out=stack_s13_half)
            dst_s13[:, :i].copy_(stack_s13_half)
            torch.stack(s3_views, out=stack_s13_half)
            dst_s13[:, i:].copy_(stack_s13_half)
            torch.stack(s2_views, out=stack_s2)
            dst_s2.copy_(stack_s2)

    def foreach_peer_copy() -> None:
        with torch.cuda.device(destination_device):
            torch._foreach_copy_(dst_w1_views, remote_w1_views, non_blocking=True)
            torch._foreach_copy_(dst_w3_views, remote_w3_views, non_blocking=True)
            torch._foreach_copy_(dst_w2_views, remote_w2_views, non_blocking=True)
            torch._foreach_copy_(dst_s1_views, remote_s1_views, non_blocking=True)
            torch._foreach_copy_(dst_s3_views, remote_s3_views, non_blocking=True)
            torch._foreach_copy_(dst_s2_views, remote_s2_views, non_blocking=True)

    per_expert_copy()
    torch.cuda.synchronize(destination_device)
    reference = tuple(tensor.clone() for tensor in (dst_w13, dst_w2, dst_s13, dst_s2))
    prepared_batched_copy()
    torch.cuda.synchronize(destination_device)
    for actual, expected in zip((dst_w13, dst_w2, dst_s13, dst_s2), reference):
        if not torch.equal(actual, expected):
            raise AssertionError("batched expert copy does not match per-expert copy")
    receiver_stack_copy()
    torch.cuda.synchronize(destination_device)
    for actual, expected in zip((dst_w13, dst_w2, dst_s13, dst_s2), reference):
        if not torch.equal(actual, expected):
            raise AssertionError("receiver stack copy does not match per-expert copy")
    foreach_peer_copy()
    torch.cuda.synchronize(destination_device)
    for actual, expected in zip((dst_w13, dst_w2, dst_s13, dst_s2), reference):
        if not torch.equal(actual, expected):
            raise AssertionError("foreach peer copy does not match per-expert copy")

    per_expert_gpu_ms, per_expert_wall_ms = _measure(
        per_expert_copy,
        device=destination_device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    per_expert_local_gpu_ms, per_expert_local_wall_ms = _measure(
        per_expert_local_copy,
        device=destination_device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    receiver_stack_gpu_ms, receiver_stack_wall_ms = _measure(
        receiver_stack_copy,
        device=destination_device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    foreach_peer_gpu_ms, foreach_peer_wall_ms = _measure(
        foreach_peer_copy,
        device=destination_device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    prepared_batched_gpu_ms, prepared_batched_wall_ms = _measure(
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
                "receiver_stack_op_count": 6,
                "foreach_peer_op_count": 6,
                "prepared_batched_copy_count": 4,
                "per_expert_gpu_ms": per_expert_gpu_ms,
                "per_expert_wall_ms": per_expert_wall_ms,
                "per_expert_local_gpu_ms": per_expert_local_gpu_ms,
                "per_expert_local_wall_ms": per_expert_local_wall_ms,
                "receiver_stack_gpu_ms": receiver_stack_gpu_ms,
                "receiver_stack_wall_ms": receiver_stack_wall_ms,
                "foreach_peer_gpu_ms": foreach_peer_gpu_ms,
                "foreach_peer_wall_ms": foreach_peer_wall_ms,
                "prepared_batched_gpu_ms": prepared_batched_gpu_ms,
                "prepared_batched_wall_ms": prepared_batched_wall_ms,
                "receiver_stack_wall_speedup": per_expert_local_wall_ms
                / receiver_stack_wall_ms,
                "foreach_peer_wall_speedup": per_expert_wall_ms
                / foreach_peer_wall_ms,
                "prepared_batched_wall_speedup": per_expert_wall_ms
                / prepared_batched_wall_ms,
                "receiver_stack_wall_reduction_pct": 100.0
                * (per_expert_local_wall_ms - receiver_stack_wall_ms)
                / per_expert_local_wall_ms,
                "foreach_peer_wall_reduction_pct": 100.0
                * (per_expert_wall_ms - foreach_peer_wall_ms)
                / per_expert_wall_ms,
                "prepared_batched_wall_reduction_pct": 100.0
                * (per_expert_wall_ms - prepared_batched_wall_ms)
                / per_expert_wall_ms,
                "value_parity": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
