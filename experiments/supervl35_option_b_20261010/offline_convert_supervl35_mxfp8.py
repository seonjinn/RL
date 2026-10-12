from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from offline_mxfp8_scope import build_scope


BLOCK_SIZE = 32
FP8_DTYPE = torch.float8_e4m3fn
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_shards(input_dir: Path) -> tuple[dict[str, str], dict[str, TensorSpec]]:
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map: {index_path}")
    elif (input_dir / "model.safetensors").is_file():
        with safe_open(
            input_dir / "model.safetensors", framework="pt", device="cpu"
        ) as reader:
            weight_map = {name: "model.safetensors" for name in reader.keys()}
    else:
        raise FileNotFoundError(f"No safetensors checkpoint in {input_dir}")

    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        if (
            not isinstance(name, str)
            or not isinstance(shard, str)
            or Path(shard).name != shard
        ):
            raise ValueError(f"Invalid index entry: {name!r} -> {shard!r}")
        by_shard[shard].add(name)

    specs: dict[str, TensorSpec] = {}
    for shard, indexed_names in sorted(by_shard.items()):
        source = input_dir / shard
        if not source.is_file():
            raise FileNotFoundError(source)
        with safe_open(source, framework="pt", device="cpu") as reader:
            actual_names = set(reader.keys())
            if actual_names != indexed_names:
                raise ValueError(f"Shard/index key mismatch in {source}")
            for name in actual_names:
                slice_info = reader.get_slice(name)
                spec = TensorSpec(tuple(slice_info.get_shape()), slice_info.get_dtype())
                if spec.dtype not in DTYPE_BYTES:
                    raise ValueError(f"Unsupported dtype {spec.dtype} for {name}")
                specs[name] = spec
    return weight_map, specs


def expected_output_specs(
    specs: dict[str, TensorSpec], targets: set[str]
) -> dict[str, TensorSpec]:
    result = dict(specs)
    for name in targets:
        spec = specs[name]
        if spec.dtype != "BF16" or len(spec.shape) != 2 or spec.shape[-1] % BLOCK_SIZE:
            raise ValueError(f"MXFP8 requires BF16 rank-2 K%32=0: {name} {spec}")
        scale = f"{name}_scale"
        if scale in specs:
            raise ValueError(f"Generated scale name already exists: {scale}")
        result[name] = TensorSpec(spec.shape, "F8_E4M3")
        result[scale] = TensorSpec((spec.shape[0], spec.shape[1] // BLOCK_SIZE), "U8")
    return result


def quantize_weight(
    weight: torch.Tensor, device: torch.device, chunk_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(weight.shape)
    rows, columns = shape
    values = torch.empty(shape, dtype=FP8_DTYPE, device="cpu")
    scales = torch.empty((rows, columns // BLOCK_SIZE), dtype=torch.uint8, device="cpu")
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    tiny = torch.finfo(torch.float32).tiny

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        chunk = weight[start:stop].to(device=device, dtype=torch.float32)
        if not torch.isfinite(chunk).all():
            raise ValueError("Source weight contains a non-finite value")
        blocked = chunk.reshape(stop - start, columns // BLOCK_SIZE, BLOCK_SIZE)
        amax = blocked.abs().amax(dim=-1)
        exponent = (torch.ceil(torch.log2(amax.clamp(min=tiny) / fp8_max)) + 127).clamp(
            0, 254
        )
        if ((exponent == 0) & (amax != 0)).any():
            raise ValueError("Nonzero MXFP8 block requires reserved E8M0 scale byte 0")
        scaled = blocked * torch.exp2(127.0 - exponent).unsqueeze(-1)
        values[start:stop].copy_(
            scaled.reshape(stop - start, columns).to(FP8_DTYPE).cpu()
        )
        scales[start:stop].copy_(
            torch.where(exponent == 0, 1, exponent).to(torch.uint8).cpu()
        )
    return values, scales


def validate_shard(path: Path, expected: dict[str, TensorSpec]) -> None:
    with safe_open(path, framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(expected):
            raise ValueError(f"Output shard has wrong tensor names: {path}")
        for name, spec in expected.items():
            slice_info = reader.get_slice(name)
            if (
                tuple(slice_info.get_shape()) != spec.shape
                or slice_info.get_dtype() != spec.dtype
            ):
                raise ValueError(f"Output shard has wrong shape/dtype: {path} {name}")


def list_auxiliary_files(input_dir: Path, shards: set[str]) -> list[Path]:
    excluded = shards | {"model.safetensors.index.json", "config.json"}
    auxiliary: list[Path] = []
    total_bytes = 0
    pending = [input_dir]
    while pending:
        parent = pending.pop()
        for source in parent.iterdir():
            relative = source.relative_to(input_dir)
            if parent == input_dir and (
                source.name in excluded or source.name.startswith(".")
            ):
                continue
            if source.is_symlink():
                raise ValueError(
                    f"Auxiliary symlink requires explicit handling: {source}"
                )
            if source.is_dir():
                if len(relative.parts) > 4:
                    raise ValueError(f"Auxiliary directory is too deep: {source}")
                pending.append(source)
            elif source.is_file():
                auxiliary.append(source)
                total_bytes += source.stat().st_size
                if len(auxiliary) > 2000 or total_bytes > 1024**3:
                    raise ValueError("Auxiliary files exceed 2000 files or 1 GiB")
    return auxiliary


def copy_auxiliary_files(
    input_dir: Path, output_dir: Path, auxiliary: list[Path]
) -> None:
    for source in auxiliary:
        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def convert(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (
        not input_dir.is_dir()
        or input_dir == output_dir
        or output_dir.is_relative_to(input_dir)
    ):
        raise ValueError(
            "Input must exist and output must be outside the input directory"
        )
    config_path = input_dir / "config.json"
    config = read_json(config_path)
    if config.get("quantization_config"):
        raise ValueError("Source already declares quantization_config; expected BF16")
    layer_types = config.get("llm_config", {}).get("layers_block_type")
    if not isinstance(layer_types, list) or not all(
        isinstance(x, str) for x in layer_types
    ):
        raise ValueError("Expected llm_config.layers_block_type in SuperVL3.5 config")

    weight_map, specs = inspect_shards(input_dir)
    targets, ignored = build_scope(
        weight_map,
        layer_types,
        first_bf16=args.first_bf16,
        last_bf16=args.last_bf16,
    )
    output_specs = expected_output_specs(specs, targets)
    input_bytes = sum(spec.nbytes for spec in specs.values())
    output_bytes = sum(spec.nbytes for spec in output_specs.values())
    shards = sorted(set(weight_map.values()))
    auxiliary = list_auxiliary_files(input_dir, set(shards))
    print(
        f"input={input_dir} output={output_dir} layers={len(layer_types)} "
        f"quantized_weights={len(targets)} ignored_modules={len(ignored)} "
        f"shards={len(shards)} bytes={input_bytes}->{output_bytes} "
        f"auxiliary_files={len(auxiliary)}",
        flush=True,
    )
    if args.dry_run:
        return

    if args.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Conversion requires a CUDA GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".convert.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {
            "format_version": 1,
            "input_dir": str(input_dir),
            "source_config_sha256": sha256(config_path),
            "source_index_sha256": (
                sha256(input_dir / "model.safetensors.index.json")
                if (input_dir / "model.safetensors.index.json").is_file()
                else None
            ),
            "first_bf16": args.first_bf16,
            "last_bf16": args.last_bf16,
            "scope": "middle_routed_experts_only",
        }
        plan_path = output_dir / "conversion-plan.json"
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise ValueError(
                    "Existing output was created with a different conversion plan"
                )
        elif set(output_dir.iterdir()) != {output_dir / ".convert.lock"}:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        else:
            write_json_atomic(plan_path, plan)

        output_map = dict(weight_map)
        for name in targets:
            output_map[f"{name}_scale"] = weight_map[name]
        names_by_shard: dict[str, list[str]] = defaultdict(list)
        for name, shard in weight_map.items():
            names_by_shard[shard].append(name)

        for position, shard in enumerate(shards, start=1):
            expected = {
                name: output_specs[name]
                for name, mapped in output_map.items()
                if mapped == shard
            }
            destination = output_dir / shard
            if destination.is_file():
                validate_shard(destination, expected)
                print(f"[{position}/{len(shards)}] reuse {shard}", flush=True)
                continue
            print(f"[{position}/{len(shards)}] convert {shard}", flush=True)
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(input_dir / shard, framework="pt", device="cpu") as reader:
                for name in sorted(names_by_shard[shard]):
                    source = reader.get_tensor(name)
                    if name in targets:
                        value, scale = quantize_weight(
                            source, args.device, args.chunk_rows
                        )
                        tensors[name] = value
                        tensors[f"{name}_scale"] = scale
                    else:
                        tensors[name] = source
                temporary = output_dir / f".{shard}.incomplete"
                save_file(tensors, temporary, metadata={"format": "pt"})
            validate_shard(temporary, expected)
            os.replace(temporary, destination)
            del tensors
            if args.max_shards and position >= args.max_shards:
                print(
                    "Stopped after requested shards; checkpoint is not finalized",
                    flush=True,
                )
                return

        copy_auxiliary_files(input_dir, output_dir, auxiliary)
        config["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignore": ignored,
            "ignored_layers": ignored,
        }
        write_json_atomic(output_dir / "config.json", config)
        write_json_atomic(
            output_dir / "model.safetensors.index.json",
            {"metadata": {"total_size": output_bytes}, "weight_map": output_map},
        )
        print(f"Conversion complete: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SuperVL3.5 middle routed experts to MXFP8"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--first-bf16", type=int, default=2)
    parser.add_argument("--last-bf16", type=int, default=6)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--chunk-rows", type=int, default=16384)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows <= 0 or args.max_shards < 0:
        parser.error("--chunk-rows must be positive and --max-shards nonnegative")
    return args


if __name__ == "__main__":
    convert(parse_args())
