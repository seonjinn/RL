#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def random_bytes(*shape: int) -> torch.Tensor:
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")


def measure(
    fn: Callable[[], tuple[torch.Tensor, ...]], repeats: int
) -> tuple[float, float, float]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        output = fn()
    end.record()
    end.synchronize()
    assert output
    wall_ms = (time.perf_counter() - wall_start) * 1_000 / repeats
    cuda_ms = start.elapsed_time(end) / repeats
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return wall_ms, cuda_ms, peak_gib


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    num_layers = 48
    num_experts = 128
    hidden_size = 2048
    intermediate_size = 768
    w13_rows = 2 * intermediate_size
    epilogue_tile_m = 128

    from nemo_rl.models.generation.vllm.quantization import fp8

    w13_weight = random_bytes(num_experts, w13_rows, hidden_size).view(
        torch.float8_e4m3fn
    )
    w2_weight = random_bytes(num_experts, hidden_size, intermediate_size).view(
        torch.float8_e4m3fn
    )
    w13_scale = random_bytes(num_experts, w13_rows, hidden_size // 32)
    w2_scale = random_bytes(num_experts, hidden_size, intermediate_size // 32)
    layer = SimpleNamespace()

    def baseline() -> tuple[torch.Tensor, ...]:
        return fp8._shuffle_mxfp8_moe_per_expert(
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
            True,
            epilogue_tile_m,
        )

    def optimized() -> tuple[torch.Tensor, ...]:
        return fp8._shuffle_mxfp8_moe_batched(
            layer,
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
            True,
            epilogue_tile_m,
        )

    reference = baseline()
    candidate = optimized()
    for actual, expected in zip(candidate, reference, strict=True):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    del reference, candidate
    torch.cuda.empty_cache()

    for _ in range(args.warmup):
        baseline()
        optimized()
    torch.cuda.synchronize()

    samples: dict[str, list[dict[str, float]]] = {
        "baseline": [],
        "optimized": [],
    }
    functions = {"baseline": baseline, "optimized": optimized}
    for trial in range(args.trials):
        order = (
            ("baseline", "optimized") if trial % 2 == 0 else ("optimized", "baseline")
        )
        for arm in order:
            wall_ms, cuda_ms, peak_gib = measure(functions[arm], args.repeats)
            samples[arm].append(
                {
                    "wall_ms_per_layer": wall_ms,
                    "cuda_ms_per_layer": cuda_ms,
                    "peak_allocated_gib": peak_gib,
                }
            )

    summary = {}
    for arm, arm_samples in samples.items():
        wall_ms = statistics.median(s["wall_ms_per_layer"] for s in arm_samples)
        cuda_ms = statistics.median(s["cuda_ms_per_layer"] for s in arm_samples)
        peak_gib = max(s["peak_allocated_gib"] for s in arm_samples)
        summary[arm] = {
            "wall_ms_per_layer": wall_ms,
            "cuda_ms_per_layer": cuda_ms,
            "estimated_48_layer_wall_ms": wall_ms * num_layers,
            "peak_allocated_gib": peak_gib,
        }

    baseline_ms = summary["baseline"]["wall_ms_per_layer"]
    optimized_ms = summary["optimized"]["wall_ms_per_layer"]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "shapes": {
            "w13_weight": list(w13_weight.shape),
            "w2_weight": list(w2_weight.shape),
            "w13_scale": list(w13_scale.shape),
            "w2_scale": list(w2_scale.shape),
        },
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "trials": args.trials,
        },
        "summary": summary,
        "speedup": baseline_ms / optimized_ms,
        "estimated_48_layer_savings_ms": (baseline_ms - optimized_ms) * num_layers,
        "bit_exact": True,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
