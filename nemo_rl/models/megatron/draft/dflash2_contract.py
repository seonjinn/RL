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

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

DFlash2Architecture: TypeAlias = Literal[
    "DFlash2DraftModel",
    "Qwen3DFlash2DraftModel",
]

_DFLASH2_ARCHITECTURES = frozenset({"DFlash2DraftModel", "Qwen3DFlash2DraftModel"})
_PLAIN_DFLASH_ARCHITECTURES = frozenset({"DFlashDraftModel", "Qwen3DFlashDraftModel"})
_CONV_FEATURE_PATTERN = re.compile(r"^layers\.\d+\.(?:attention_conv|mlp_conv)\.")


class DFlash2CheckpointContractError(ValueError):
    """A checkpoint cannot be safely recognized as DFlash2."""


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


def _require_architecture(config: Mapping[str, object]) -> DFlash2Architecture:
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
    architecture = architectures[0]
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


def _required_feature_keys(num_hidden_layers: int) -> frozenset[str]:
    keys = {
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
    }
    for layer_id in range(num_hidden_layers):
        for component in ("attention_conv", "mlp_conv"):
            keys.add(f"layers.{layer_id}.{component}.base_kernel")
            keys.add(f"layers.{layer_id}.{component}.kernel_projection.weight")
    return frozenset(keys)


def _validate_feature_state_dict(
    state_dict_keys: Iterable[str],
    *,
    num_hidden_layers: int,
) -> None:
    keys = set(state_dict_keys)
    if not all(isinstance(key, str) for key in keys):
        raise DFlash2CheckpointContractError("state-dict keys must be strings")
    required = _required_feature_keys(num_hidden_layers)
    missing = sorted(required.difference(keys))
    if missing:
        raise DFlash2CheckpointContractError(
            f"missing DFlash2 tensors ({len(missing)}): {missing[:8]}"
        )
    feature_keys = {
        key
        for key in keys
        if key.startswith("candidate_selector.")
        or _CONV_FEATURE_PATTERN.match(key) is not None
    }
    unexpected = sorted(feature_keys.difference(required))
    if unexpected:
        raise DFlash2CheckpointContractError(
            f"unexpected DFlash2 tensors ({len(unexpected)}): {unexpected[:8]}"
        )


def validate_dflash2_checkpoint_contract(
    config: Mapping[str, object],
    state_dict_keys: Iterable[str],
) -> DFlash2CheckpointContract:
    """Validate DFlash2 checkpoint identity, geometry, and feature tensors."""
    architecture = _require_architecture(config)
    dflash_config = _require_mapping(
        config.get("dflash_config"),
        name="dflash_config",
    )
    hidden_size = _require_positive_int(config, "hidden_size", context="config")
    num_hidden_layers = _require_positive_int(
        config,
        "num_hidden_layers",
        context="config",
    )
    num_target_layers = _require_positive_int(
        config,
        "num_target_layers",
        context="config",
    )
    vocab_size = _require_positive_int(config, "vocab_size", context="config")

    block_size = _require_positive_int(
        dflash_config,
        "block_size",
        context="dflash_config",
    )
    if block_size != 8:
        raise DFlash2CheckpointContractError("DFlash2 requires block_size=8")
    conv_kernel_size = _require_positive_int(
        dflash_config,
        "conv_kernel_size",
        context="dflash_config",
    )
    conv_group_size = _require_positive_int(
        dflash_config,
        "conv_group_size",
        context="dflash_config",
    )
    if hidden_size % conv_group_size != 0:
        raise DFlash2CheckpointContractError(
            "dflash_config.conv_group_size must divide hidden_size"
        )
    selector_rank = _require_positive_int(
        dflash_config,
        "selector_rank",
        context="dflash_config",
    )
    selector_top_k = _require_positive_int(
        dflash_config,
        "selector_top_k",
        context="dflash_config",
    )
    if selector_top_k > vocab_size:
        raise DFlash2CheckpointContractError(
            "dflash_config.selector_top_k must not exceed vocab_size"
        )
    mask_token_id = _require_int(
        dflash_config,
        "mask_token_id",
        context="dflash_config",
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
    _validate_feature_state_dict(
        state_dict_keys,
        num_hidden_layers=num_hidden_layers,
    )
    return DFlash2CheckpointContract(
        architecture=architecture,
        block_size=block_size,
        conv_kernel_size=conv_kernel_size,
        conv_group_size=conv_group_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        target_layer_ids=target_layer_ids,
        num_hidden_layers=num_hidden_layers,
    )
