"""Byte-exact MXFP8 transport comparison through NeMo-RL's actual dispatcher."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard

from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
from nemo_rl.weight_sync import xferdtensor as xfer
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo
from nemo_rl.weight_sync.xferdtensor_python import clear_xferdtensor_python_caches


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    src_dim: int
    dst_dim: int | None


def local_slice(
    shape: tuple[int, ...], dim: int | None, coordinate: int, count: int
) -> tuple[slice, ...]:
    region = [slice(0, size) for size in shape]
    if dim is not None:
        if shape[dim] % count:
            raise ValueError(
                f"{shape} cannot be evenly split on axis {dim} over {count}"
            )
        width = shape[dim] // count
        region[dim] = slice(coordinate * width, (coordinate + 1) * width)
    return tuple(region)


def expected_bytes(
    region: tuple[slice, ...], role: str, version: int, device: torch.device
) -> torch.Tensor:
    code = torch.tensor(version * 17, dtype=torch.int32, device=device)
    for axis, section in enumerate(region):
        shape = [1] * len(region)
        shape[axis] = section.stop - section.start
        coordinates = torch.arange(
            section.start, section.stop, dtype=torch.int32, device=device
        ).reshape(shape)
        code = code + coordinates * (11, 37, 53)[axis]
    if role == "weight":
        return code.remainder(120).to(torch.uint8).contiguous()
    return (code.remainder(200) + 20).to(torch.uint8).contiguous()


def provenance() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "nccl4py", "nccl-extensions", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "source_sha": os.environ.get("SOURCE_SHA"),
        "container": os.environ.get("CONTAINER"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "machine": platform.machine(),
        "packages": packages,
        "torch_nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(),
        "real_op_available": xfer._reshard is not None,
        "real_op_module": getattr(xfer._reshard, "__module__", None),
    }


def run_case(
    case: Case,
    backend: str,
    group: StatelessProcessGroup,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    rank, world = dist.get_rank(), dist.get_world_size()
    count = world // 2
    source = rank < count
    coordinate = rank if source else rank - count
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    stream = torch.cuda.current_stream()
    src_mesh = MeshInfo(torch.arange(count))
    dst_mesh = MeshInfo(torch.arange(count, world))
    src_placements = (Shard(case.src_dim),)
    dst_placements = (Replicate() if case.dst_dim is None else Shard(case.dst_dim),)
    scale_shape = (*case.shape[:-1], case.shape[-1] // 32)
    components: list[tuple[str, torch.Tensor, tuple[int, ...], tuple[slice, ...]]] = []
    for role, shape in (("weight", case.shape), ("weight_scale", scale_shape)):
        region = local_slice(
            shape, case.src_dim if source else case.dst_dim, coordinate, count
        )
        local_shape = tuple(section.stop - section.start for section in region)
        dtype = torch.float8_e4m3fn if role == "weight" else torch.uint8
        components.append(
            (role, torch.empty(local_shape, device=device, dtype=dtype), shape, region)
        )

    def transfer() -> None:
        context = nullcontext()
        if backend == "native-grouped":
            from nccl import m2n

            context = m2n.group()
        with context:
            for _role, tensor, shape, _region in components:
                ref = xfer.DTensorRef(tensor, shape)
                xfer.xferdtensor(
                    ref if source else None,
                    src_mesh,
                    src_placements,
                    None if source else ref,
                    dst_mesh,
                    dst_placements,
                    group,
                    stream,
                )

    parity: list[dict[str, Any]] = []
    cold_ms: list[float] = []
    for version in (1, 2, 3):
        for role, tensor, _shape, region in components:
            if source:
                tensor.view(torch.uint8).copy_(
                    expected_bytes(region, role, version, device)
                )
            else:
                tensor.view(torch.uint8).fill_(255)
        stream.synchronize()
        dist.barrier()
        start = time.perf_counter()
        transfer()
        stream.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        mismatches = 0
        if not source:
            for role, tensor, _shape, region in components:
                expected = expected_bytes(region, role, version, device)
                mismatches += int(
                    torch.count_nonzero(tensor.view(torch.uint8) != expected).item()
                )
        counters = torch.tensor([mismatches], dtype=torch.int64)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        if counters.item():
            raise AssertionError(
                f"{case.name} version {version}: {counters.item()} wrong bytes"
            )
        parity.append({"version": version, "mismatched_bytes": 0})
        times: list[float | None] = [None] * world
        dist.all_gather_object(times, elapsed)
        cold_ms.append(max(float(value) for value in times if value is not None))

    for _ in range(warmup):
        transfer()
        stream.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        dist.barrier()
        started = time.perf_counter()
        transfer()
        stream.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    gathered: list[list[float] | None] = [None] * world
    dist.all_gather_object(gathered, samples)
    maxima = [
        max(values[index] for values in gathered if values is not None)
        for index in range(iterations)
    ]
    destination_bytes = (
        sum(tensor.numel() * tensor.element_size() for _, tensor, _, _ in components)
        if not source
        else 0
    )
    total_bytes = torch.tensor([destination_bytes], dtype=torch.int64)
    dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    return {
        "case": case.name,
        "backend": backend,
        "shape": case.shape,
        "src_dim": case.src_dim,
        "dst_dim": case.dst_dim,
        "parity": parity,
        "cold_update_ms": cold_ms,
        "steady_max_rank_ms": maxima,
        "steady_mean_ms": statistics.mean(maxima),
        "steady_median_ms": statistics.median(maxima),
        "aggregate_destination_bytes": int(total_bytes.item()),
        "aggregate_destination_gib_per_s": total_bytes.item()
        / (1024**3)
        / (statistics.mean(maxima) / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("python", "native", "native-grouped"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world < 4 or world % 2:
        raise ValueError("Use an even number of GPUs, at least four")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    info = provenance()
    print(json.dumps({"rank": rank, "provenance": info}), flush=True)
    if args.backend != "python" and xfer._reshard is None:
        raise RuntimeError(
            "Native M2N is required but unavailable; refusing Python fallback"
        )
    os.environ.pop("NRL_XFERDTENSOR_GOLDEN", None)
    os.environ["NRL_XFERDTENSOR_PYTHON"] = "1" if args.backend == "python" else "0"
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    group = StatelessProcessGroup(
        os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]) + 1, rank, world
    )
    group.init_nccl_communicator(int(os.environ["LOCAL_RANK"]))
    cases = (
        Case("experts-to-replicated", (16, 128, 256), 0, None),
        Case("experts-to-k", (16, 128, 256), 0, 2),
        Case("k-to-experts", (16, 128, 256), 2, 0),
        Case("dense-k-to-row", (256, 256), 1, 0),
        Case("qwen30-expert-projection", (128, 768, 2048), 0, None),
    )
    results = []
    try:
        for case in cases:
            result = run_case(case, args.backend, group, args.warmup, args.iterations)
            results.append(result)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {"provenance": info, "world_size": world, "results": results},
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(result), flush=True)
        torch.cuda.synchronize()
        dist.barrier()
        clear_xferdtensor_python_caches(group)
        if args.backend != "python":
            from nccl import m2n

            m2n.finalize()
        if group.nccl_communicator is not None:
            group.nccl_communicator.destroy()
        dist.destroy_process_group()
    except BaseException:
        group.abort()
        raise


if __name__ == "__main__":
    main()
