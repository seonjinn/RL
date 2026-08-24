# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

DFlash2Architecture: TypeAlias = Literal[
    "DFlash2DraftModel",
    "Qwen3DFlash2DraftModel",
]

_DFLASH2_ARCHITECTURES = frozenset({"DFlash2DraftModel", "Qwen3DFlash2DraftModel"})
_PLAIN_DFLASH_ARCHITECTURES = frozenset({"DFlashDraftModel", "Qwen3DFlashDraftModel"})
_CONFIG_DTYPE_TO_SAFETENSORS = {
    "bfloat16": "BF16",
    "float16": "F16",
    "float32": "F32",
}
_SAFETENSORS_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


class DFlash2CheckpointContractError(ValueError):
    """A checkpoint cannot be safely recognized as DFlash2."""


@dataclass(frozen=True, slots=True)
class DFlash2TensorMetadata:
    """Shape and safetensors dtype for one checkpoint tensor."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class DFlash2CheckpointContract:
    """Validated, immutable identity and geometry of a DFlash2 checkpoint."""

    architecture: DFlash2Architecture
    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: tuple[int, ...]
    num_hidden_layers: int
    tensor_count: int
    dtype: str


@dataclass(frozen=True, slots=True)
class _DFlash2Geometry:
    architecture: DFlash2Architecture
    dtype: str
    hidden_size: int
    intermediate_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    num_hidden_layers: int
    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: tuple[int, ...]


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DFlash2CheckpointContractError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise DFlash2CheckpointContractError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _require_int(
    values: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DFlash2CheckpointContractError(f"{context}.{name} must be an integer")
    return value


def _require_positive_int(
    values: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> int:
    value = _require_int(values, name, context=context)
    if value <= 0:
        raise DFlash2CheckpointContractError(f"{context}.{name} must be positive")
    return value


def _architecture_name(config: Mapping[str, object]) -> str:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, Sequence)
        or isinstance(architectures, (str, bytes))
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
    ):
        raise DFlash2CheckpointContractError(
            "architectures must contain exactly one DFlash2 architecture"
        )
    return architectures[0]


def _dflash2_architecture_if_present(
    config: Mapping[str, object],
) -> DFlash2Architecture | None:
    architectures = config.get("architectures")
    if not isinstance(architectures, Sequence) or isinstance(
        architectures, (str, bytes)
    ):
        return None
    matches = [
        architecture
        for architecture in architectures
        if isinstance(architecture, str) and architecture in _DFLASH2_ARCHITECTURES
    ]
    if not matches:
        return None
    if len(architectures) != 1 or len(matches) != 1:
        raise DFlash2CheckpointContractError(
            "architectures must contain exactly one DFlash2 architecture"
        )
    return cast(DFlash2Architecture, matches[0])


def _require_architecture(config: Mapping[str, object]) -> DFlash2Architecture:
    architecture = _architecture_name(config)
    if architecture in _PLAIN_DFLASH_ARCHITECTURES:
        raise DFlash2CheckpointContractError(
            f"plain DFlash architecture {architecture!r} cannot be loaded as DFlash2"
        )
    if architecture not in _DFLASH2_ARCHITECTURES:
        raise DFlash2CheckpointContractError(
            f"unsupported DFlash2 architecture {architecture!r}"
        )
    return cast(DFlash2Architecture, architecture)


def _require_target_layer_ids(
    dflash_config: Mapping[str, object],
    *,
    num_hidden_layers: int,
    num_target_layers: int,
) -> tuple[int, ...]:
    raw_ids = dflash_config.get("target_layer_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise DFlash2CheckpointContractError(
            "dflash_config.target_layer_ids must be a non-empty list"
        )
    if any(
        isinstance(layer_id, bool) or not isinstance(layer_id, int)
        for layer_id in raw_ids
    ):
        raise DFlash2CheckpointContractError(
            "dflash_config.target_layer_ids must contain integers"
        )
    layer_ids = cast(list[int], raw_ids)
    if len(layer_ids) != num_hidden_layers:
        raise DFlash2CheckpointContractError(
            "DFlash2 requires one target tap per draft layer"
        )
    if len(set(layer_ids)) != len(layer_ids) or any(
        layer_id < 0 or layer_id >= num_target_layers for layer_id in layer_ids
    ):
        raise DFlash2CheckpointContractError(
            "dflash_config.target_layer_ids must be unique and within the target"
        )
    return tuple(layer_ids)


def _parse_geometry(config: Mapping[str, object]) -> _DFlash2Geometry:
    architecture = _require_architecture(config)
    dflash_config = _require_mapping(config.get("dflash_config"), name="dflash_config")
    hidden_size = _require_positive_int(config, "hidden_size", context="config")
    intermediate_size = _require_positive_int(
        config, "intermediate_size", context="config"
    )
    head_dim = _require_positive_int(config, "head_dim", context="config")
    num_attention_heads = _require_positive_int(
        config, "num_attention_heads", context="config"
    )
    num_key_value_heads = _require_positive_int(
        config, "num_key_value_heads", context="config"
    )
    num_hidden_layers = _require_positive_int(
        config, "num_hidden_layers", context="config"
    )
    num_target_layers = _require_positive_int(
        config, "num_target_layers", context="config"
    )
    vocab_size = _require_positive_int(config, "vocab_size", context="config")

    raw_dtype = config.get("dtype")
    if not isinstance(raw_dtype, str) or raw_dtype not in _CONFIG_DTYPE_TO_SAFETENSORS:
        raise DFlash2CheckpointContractError(
            "config.dtype must be one of bfloat16, float16, or float32"
        )
    dtype = _CONFIG_DTYPE_TO_SAFETENSORS[raw_dtype]

    block_size = _require_positive_int(
        dflash_config, "block_size", context="dflash_config"
    )
    if block_size != 8:
        raise DFlash2CheckpointContractError("DFlash2 requires block_size=8")
    conv_kernel_size = _require_positive_int(
        dflash_config, "conv_kernel_size", context="dflash_config"
    )
    if conv_kernel_size > block_size:
        raise DFlash2CheckpointContractError(
            "dflash_config.conv_kernel_size must not exceed block_size"
        )
    conv_group_size = _require_positive_int(
        dflash_config, "conv_group_size", context="dflash_config"
    )
    if hidden_size % conv_group_size != 0:
        raise DFlash2CheckpointContractError(
            "dflash_config.conv_group_size must divide hidden_size"
        )
    selector_rank = _require_positive_int(
        dflash_config, "selector_rank", context="dflash_config"
    )
    selector_top_k = _require_positive_int(
        dflash_config, "selector_top_k", context="dflash_config"
    )
    if selector_top_k > vocab_size:
        raise DFlash2CheckpointContractError(
            "dflash_config.selector_top_k must not exceed vocab_size"
        )
    mask_token_id = _require_int(
        dflash_config, "mask_token_id", context="dflash_config"
    )
    if not 0 <= mask_token_id < vocab_size:
        raise DFlash2CheckpointContractError(
            "dflash_config.mask_token_id must be within the vocabulary"
        )
    target_layer_ids = _require_target_layer_ids(
        dflash_config,
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
    )
    return _DFlash2Geometry(
        architecture=architecture,
        dtype=dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        block_size=block_size,
        conv_kernel_size=conv_kernel_size,
        conv_group_size=conv_group_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        target_layer_ids=target_layer_ids,
    )


def _tensor(shape: tuple[int, ...], dtype: str) -> DFlash2TensorMetadata:
    return DFlash2TensorMetadata(shape=shape, dtype=dtype)


def _expected_tensor_metadata(
    geometry: _DFlash2Geometry,
) -> dict[str, DFlash2TensorMetadata]:
    hidden_size = geometry.hidden_size
    dtype = geometry.dtype
    metadata = {
        "candidate_selector.hidden_projection.weight": _tensor(
            (geometry.selector_rank, hidden_size), dtype
        ),
        "candidate_selector.predecessor_codebook": _tensor(
            (geometry.vocab_size, geometry.selector_rank), dtype
        ),
        "candidate_selector.successor_codebook": _tensor(
            (geometry.vocab_size, geometry.selector_rank), dtype
        ),
        "fc.weight": _tensor(
            (hidden_size, hidden_size * geometry.num_hidden_layers), dtype
        ),
        "hidden_norm.weight": _tensor((hidden_size,), dtype),
        "norm.weight": _tensor((hidden_size,), dtype),
    }
    query_width = geometry.num_attention_heads * geometry.head_dim
    kv_width = geometry.num_key_value_heads * geometry.head_dim
    conv_projection_width = (
        2 * geometry.conv_kernel_size * (hidden_size // geometry.conv_group_size)
    )
    for layer_id in range(geometry.num_hidden_layers):
        prefix = f"layers.{layer_id}"
        metadata.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor((hidden_size,), dtype),
                f"{prefix}.mlp.down_proj.weight": _tensor(
                    (hidden_size, geometry.intermediate_size), dtype
                ),
                f"{prefix}.mlp.gate_proj.weight": _tensor(
                    (geometry.intermediate_size, hidden_size), dtype
                ),
                f"{prefix}.mlp.up_proj.weight": _tensor(
                    (geometry.intermediate_size, hidden_size), dtype
                ),
                f"{prefix}.post_attention_layernorm.weight": _tensor(
                    (hidden_size,), dtype
                ),
                f"{prefix}.self_attn.k_norm.weight": _tensor(
                    (geometry.head_dim,), dtype
                ),
                f"{prefix}.self_attn.k_proj.weight": _tensor(
                    (kv_width, hidden_size), dtype
                ),
                f"{prefix}.self_attn.o_proj.weight": _tensor(
                    (hidden_size, query_width), dtype
                ),
                f"{prefix}.self_attn.q_norm.weight": _tensor(
                    (geometry.head_dim,), dtype
                ),
                f"{prefix}.self_attn.q_proj.weight": _tensor(
                    (query_width, hidden_size), dtype
                ),
                f"{prefix}.self_attn.v_proj.weight": _tensor(
                    (kv_width, hidden_size), dtype
                ),
            }
        )
        for component in ("attention_conv", "mlp_conv"):
            metadata[f"{prefix}.{component}.base_kernel"] = _tensor(
                (2, geometry.conv_kernel_size, hidden_size), dtype
            )
            metadata[f"{prefix}.{component}.kernel_projection.weight"] = _tensor(
                (conv_projection_width, hidden_size), dtype
            )
    return metadata


def expected_dflash2_tensor_metadata(
    config: Mapping[str, object],
) -> dict[str, DFlash2TensorMetadata]:
    """Compose the exact published DFlash2 tensor schema from checkpoint config."""
    return _expected_tensor_metadata(_parse_geometry(config))


def _coerce_tensor_metadata(
    name: str,
    value: object,
) -> DFlash2TensorMetadata:
    if isinstance(value, DFlash2TensorMetadata):
        return value
    values = _require_mapping(value, name=f"tensor metadata for {name}")
    raw_shape = values.get("shape")
    if (
        not isinstance(raw_shape, Sequence)
        or isinstance(raw_shape, (str, bytes))
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in raw_shape
        )
    ):
        raise DFlash2CheckpointContractError(
            f"tensor {name!r} shape must contain non-negative integers"
        )
    raw_dtype = values.get("dtype")
    if not isinstance(raw_dtype, str):
        raise DFlash2CheckpointContractError(f"tensor {name!r} dtype must be a string")
    return DFlash2TensorMetadata(
        shape=tuple(cast(Sequence[int], raw_shape)), dtype=raw_dtype
    )


def validate_dflash2_checkpoint_contract(
    config: Mapping[str, object],
    tensor_metadata: Mapping[str, object],
) -> DFlash2CheckpointContract:
    """Validate identity and the exact shape/dtype of every published tensor."""
    geometry = _parse_geometry(config)
    if not all(isinstance(key, str) for key in tensor_metadata):
        raise DFlash2CheckpointContractError("state-dict keys must be strings")
    actual = {
        name: _coerce_tensor_metadata(name, value)
        for name, value in tensor_metadata.items()
    }
    expected = _expected_tensor_metadata(geometry)
    missing = sorted(expected.keys() - actual.keys())
    if missing:
        raise DFlash2CheckpointContractError(
            f"missing DFlash2 tensors ({len(missing)}): {missing[:8]}"
        )
    unexpected = sorted(actual.keys() - expected.keys())
    if unexpected:
        raise DFlash2CheckpointContractError(
            f"unexpected DFlash2 tensors ({len(unexpected)}): {unexpected[:8]}"
        )
    for name, expected_metadata in expected.items():
        actual_metadata = actual[name]
        if actual_metadata.shape != expected_metadata.shape:
            raise DFlash2CheckpointContractError(
                f"DFlash2 tensor {name!r} shape {actual_metadata.shape} does not "
                f"match expected shape {expected_metadata.shape}"
            )
        if actual_metadata.dtype != expected_metadata.dtype:
            raise DFlash2CheckpointContractError(
                f"DFlash2 tensor {name!r} dtype {actual_metadata.dtype!r} does "
                f"not match expected dtype {expected_metadata.dtype!r}"
            )
    return DFlash2CheckpointContract(
        architecture=geometry.architecture,
        block_size=geometry.block_size,
        conv_kernel_size=geometry.conv_kernel_size,
        conv_group_size=geometry.conv_group_size,
        selector_rank=geometry.selector_rank,
        selector_top_k=geometry.selector_top_k,
        target_layer_ids=geometry.target_layer_ids,
        num_hidden_layers=geometry.num_hidden_layers,
        tensor_count=len(expected),
        dtype=geometry.dtype,
    )


def _read_json_mapping(path: Path, *, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DFlash2CheckpointContractError(
            f"could not read {description} at {path}"
        ) from exc
    return _require_mapping(value, name=description)


def _read_safetensors_header(path: Path) -> dict[str, DFlash2TensorMetadata]:
    try:
        with path.open("rb") as checkpoint_file:
            raw_header_length = checkpoint_file.read(8)
            if len(raw_header_length) != 8:
                raise DFlash2CheckpointContractError(
                    f"safetensors file {path} has a truncated header length"
                )
            header_length = struct.unpack("<Q", raw_header_length)[0]
            if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise DFlash2CheckpointContractError(
                    f"safetensors file {path} has invalid header length {header_length}"
                )
            raw_header = checkpoint_file.read(header_length)
    except OSError as exc:
        raise DFlash2CheckpointContractError(
            f"could not read safetensors checkpoint {path}"
        ) from exc
    if len(raw_header) != header_length:
        raise DFlash2CheckpointContractError(
            f"safetensors file {path} has a truncated JSON header"
        )
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DFlash2CheckpointContractError(
            f"safetensors file {path} has an invalid JSON header"
        ) from exc
    header_mapping = _require_mapping(header, name=f"safetensors header {path}")
    metadata: dict[str, DFlash2TensorMetadata] = {}
    payload_bytes = 0
    for name, value in header_mapping.items():
        if name == "__metadata__":
            continue
        metadata[name] = _coerce_tensor_metadata(name, value)
        values = _require_mapping(value, name=f"tensor metadata for {name}")
        offsets = values.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in offsets
            )
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise DFlash2CheckpointContractError(
                f"tensor {name!r} has invalid safetensors data_offsets"
            )
        tensor_metadata = metadata[name]
        dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(tensor_metadata.dtype)
        if dtype_bytes is None:
            raise DFlash2CheckpointContractError(
                f"tensor {name!r} uses unsupported dtype {tensor_metadata.dtype!r}"
            )
        expected_bytes = dtype_bytes
        for dimension in tensor_metadata.shape:
            expected_bytes *= dimension
        if offsets[1] - offsets[0] != expected_bytes:
            raise DFlash2CheckpointContractError(
                f"tensor {name!r} data_offsets do not match its shape and dtype"
            )
        payload_bytes = max(payload_bytes, offsets[1])
    try:
        file_bytes = path.stat().st_size
    except OSError as exc:
        raise DFlash2CheckpointContractError(
            f"could not stat safetensors checkpoint {path}"
        ) from exc
    expected_file_bytes = 8 + header_length + payload_bytes
    if file_bytes != expected_file_bytes:
        raise DFlash2CheckpointContractError(
            f"safetensors file {path} payload size does not match tensor offsets: "
            f"file has {file_bytes} bytes, expected {expected_file_bytes}"
        )
    return metadata


def _read_checkpoint_tensor_metadata(
    checkpoint_root: Path,
) -> dict[str, DFlash2TensorMetadata]:
    if checkpoint_root.is_file():
        if checkpoint_root.suffix != ".safetensors":
            raise DFlash2CheckpointContractError(
                "DFlash2 checkpoint inspection requires safetensors"
            )
        return _read_safetensors_header(checkpoint_root)

    index_path = checkpoint_root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_mapping(index_path, description="safetensors index")
        weight_map = _require_mapping(index.get("weight_map"), name="weight_map")
        if not all(isinstance(name, str) for name in weight_map.values()):
            raise DFlash2CheckpointContractError(
                "safetensors index weight_map values must be strings"
            )
        shard_names = sorted({cast(str, name) for name in weight_map.values()})
        merged: dict[str, DFlash2TensorMetadata] = {}
        for shard_name in shard_names:
            shard_metadata = _read_safetensors_header(checkpoint_root / shard_name)
            indexed_names = {
                name for name, value in weight_map.items() if value == shard_name
            }
            if indexed_names != set(shard_metadata):
                raise DFlash2CheckpointContractError(
                    f"safetensors index entries for {shard_name!r} do not exactly "
                    "match its tensor header"
                )
            duplicates = sorted(merged.keys() & shard_metadata.keys())
            if duplicates:
                raise DFlash2CheckpointContractError(
                    f"duplicate DFlash2 tensors across shards: {duplicates[:8]}"
                )
            merged.update(shard_metadata)
        if set(weight_map) != set(merged):
            raise DFlash2CheckpointContractError(
                "safetensors index does not exactly match shard tensor headers"
            )
        return merged

    single_file = checkpoint_root / "model.safetensors"
    if single_file.is_file():
        return _read_safetensors_header(single_file)
    shards = sorted(checkpoint_root.glob("model-*.safetensors"))
    if not shards:
        raise DFlash2CheckpointContractError(
            "DFlash2 checkpoint inspection requires model safetensors files"
        )
    merged = {}
    for shard_path in shards:
        shard_metadata = _read_safetensors_header(shard_path)
        duplicates = sorted(merged.keys() & shard_metadata.keys())
        if duplicates:
            raise DFlash2CheckpointContractError(
                f"duplicate DFlash2 tensors across shards: {duplicates[:8]}"
            )
        merged.update(shard_metadata)
    return merged


def _load_checkpoint_config(
    checkpoint_source: str | Path,
) -> tuple[Mapping[str, object], Path]:
    source_path = Path(checkpoint_source)
    if source_path.is_file():
        config_path = source_path.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return _read_json_mapping(
            config_path, description="DFlash2 config"
        ), source_path
    if source_path.is_dir():
        config_path = source_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return _read_json_mapping(
            config_path, description="DFlash2 config"
        ), source_path
    raise FileNotFoundError(str(checkpoint_source))


def inspect_dflash2_checkpoint_if_present(
    checkpoint_source: str | Path,
    *,
    revision: str | None = None,
) -> DFlash2CheckpointContract | None:
    """Validate a DFlash2 checkpoint, or return ``None`` for a non-DFlash2 one."""
    source_path = Path(checkpoint_source)
    try:
        config, checkpoint_root = _load_checkpoint_config(checkpoint_source)
    except FileNotFoundError:
        if source_path.exists():
            return None
        try:
            from huggingface_hub import hf_hub_download, snapshot_download

            config_path = Path(
                hf_hub_download(
                    repo_id=str(checkpoint_source),
                    filename="config.json",
                    revision=revision,
                )
            )
            config = _read_json_mapping(config_path, description="DFlash2 config")
            architecture = _dflash2_architecture_if_present(config)
            if architecture is None:
                return None
            checkpoint_root = Path(
                snapshot_download(
                    repo_id=str(checkpoint_source),
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "model.safetensors",
                        "model-*.safetensors",
                        "model.safetensors.index.json",
                    ],
                )
            )
        except DFlash2CheckpointContractError:
            raise
        except Exception as exc:
            raise DFlash2CheckpointContractError(
                f"could not resolve DFlash2 checkpoint {checkpoint_source!s}"
            ) from exc

    architecture = _dflash2_architecture_if_present(config)
    if architecture is None:
        return None
    tensor_metadata = _read_checkpoint_tensor_metadata(checkpoint_root)
    return validate_dflash2_checkpoint_contract(config, tensor_metadata)
