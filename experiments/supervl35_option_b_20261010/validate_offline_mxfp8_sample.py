from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def validate(input_dir: Path, output_dir: Path, shards: list[str]) -> None:
    with (input_dir / "model.safetensors.index.json").open(encoding="utf-8") as stream:
        source_map = json.load(stream)["weight_map"]
    results: list[dict[str, object]] = []
    checked_projections: set[str] = set()
    checked_bf16 = False

    for shard in shards:
        source_names = sorted(
            name for name, mapped in source_map.items() if mapped == shard
        )
        with (
            safe_open(input_dir / shard, framework="pt", device="cpu") as source,
            safe_open(output_dir / shard, framework="pt", device="cpu") as output,
        ):
            for name in source_names:
                scale_name = f"{name}_scale"
                if scale_name in output.keys() and name.endswith(".weight"):
                    projection = name.rsplit(".", 2)[-2]
                    if projection in checked_projections:
                        continue
                    original = source.get_tensor(name).float()
                    values = output.get_tensor(name)
                    scales = output.get_tensor(scale_name)
                    if (
                        values.dtype != torch.float8_e4m3fn
                        or scales.dtype != torch.uint8
                    ):
                        raise ValueError(f"Unexpected MXFP8 dtype for {name}")
                    if (scales == 0).any():
                        raise ValueError(f"Reserved E8M0 scale byte 0 in {name}")
                    dequantized = values.float().reshape(
                        *scales.shape, 32
                    ) * torch.exp2(scales.float() - 127).unsqueeze(-1)
                    relative_rmse = (
                        torch.linalg.vector_norm(
                            (dequantized.reshape_as(original) - original).flatten()
                        )
                        / torch.linalg.vector_norm(original.flatten())
                    ).item()
                    if not relative_rmse < 0.05:
                        raise ValueError(
                            f"MXFP8 relative RMSE too high for {name}: {relative_rmse}"
                        )
                    results.append({"name": name, "relative_rmse": relative_rmse})
                    checked_projections.add(projection)
                elif not checked_bf16 and name.endswith(".weight"):
                    if not torch.equal(
                        source.get_tensor(name), output.get_tensor(name)
                    ):
                        raise ValueError(f"Unquantized weight changed: {name}")
                    results.append({"name": name, "bf16_bit_identical": True})
                    checked_bf16 = True

    if checked_projections != {"up_proj", "down_proj"} or not checked_bf16:
        raise ValueError(
            f"Missing sample weights: projections={checked_projections}, bf16={checked_bf16}"
        )
    print(json.dumps({"shards": shards, "samples": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("shards", nargs="+")
    parsed = parser.parse_args()
    validate(parsed.input_dir, parsed.output_dir, parsed.shards)
