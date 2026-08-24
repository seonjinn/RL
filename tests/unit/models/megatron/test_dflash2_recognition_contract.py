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

import importlib.util
import sys
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module(name: str, relative_path: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"could not load {path}", pytrace=False)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


draft_config = _load_module(
    "dflash2_test_draft_config",
    "nemo_rl/models/policy/draft_config.py",
)
speculator_runtime = _load_module(
    "dflash2_test_speculator_runtime",
    "nemo_rl/models/generation/vllm/speculator_runtime.py",
)


@cache
def _dflash2_contract() -> ModuleType:
    try:
        return _load_module(
            "dflash2_test_checkpoint_contract",
            "nemo_rl/models/megatron/draft/dflash2_contract.py",
        )
    except FileNotFoundError:
        pytest.fail("DFlash2 checkpoint contract is not implemented", pytrace=False)


def _draft_values() -> dict[str, object]:
    return {
        "enabled": True,
        "model_name": "incoai/Qwen3.8-27B-DFlash2",
        "block_size": 8,
        "num_speculative_tokens": 7,
        "mask_token_id": 248070,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_hidden_state_layer_ids": [5, 19, 33, 47, 61],
    }


def _checkpoint_config(
    architecture: str = "DFlash2DraftModel",
) -> dict[str, object]:
    return {
        "architectures": [architecture],
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_target_layers": 64,
        "vocab_size": 248320,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
    }


def _required_dflash2_keys(num_layers: int = 5) -> set[str]:
    keys = {
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
    }
    for layer_id in range(num_layers):
        for component in ("attention_conv", "mlp_conv"):
            keys.add(f"layers.{layer_id}.{component}.base_kernel")
            keys.add(f"layers.{layer_id}.{component}.kernel_projection.weight")
    return keys


def test_dflash2_config_is_distinct_and_preserves_published_geometry() -> None:
    config = draft_config.DFlash2DraftConfig.model_validate(_draft_values())

    assert config.speculator_type == "dflash2"
    assert config.block_size == 8
    assert config.num_speculative_tokens == 7
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.target_hidden_state_layer_ids == [5, 19, 33, 47, 61]
    assert not isinstance(config, draft_config.DFlashDraftConfig)


def test_draft_union_dispatches_dflash2_without_dflash_downgrade() -> None:
    config = TypeAdapter(draft_config.DraftConfig).validate_python(_draft_values())

    assert isinstance(config, draft_config.DFlash2DraftConfig)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"block_size": 7}, "8"),
        ({"num_speculative_tokens": 8}, "7"),
        ({"conv_kernel_size": 0}, "greater than 0"),
        ({"conv_group_size": 0}, "greater than 0"),
        ({"selector_rank": 0}, "greater than 0"),
        ({"selector_top_k": 0}, "greater than 0"),
        ({"target_hidden_state_layer_ids": []}, "at least 1"),
        ({"target_hidden_state_layer_ids": [5, 5]}, "unique"),
        ({"target_hidden_state_layer_ids": [-1, 5]}, "non-negative"),
    ],
)
def test_dflash2_config_rejects_invalid_contract(
    override: Mapping[str, object], message: str
) -> None:
    values = _draft_values()
    values.update(override)

    with pytest.raises(ValidationError, match=message):
        draft_config.DFlash2DraftConfig.model_validate(values)


@pytest.mark.parametrize(
    "architecture",
    ["DFlash2DraftModel", "Qwen3DFlash2DraftModel"],
)
def test_checkpoint_contract_recognizes_both_published_architecture_names(
    architecture: str,
) -> None:
    dflash2_contract = _dflash2_contract()
    contract = dflash2_contract.validate_dflash2_checkpoint_contract(
        _checkpoint_config(architecture),
        _required_dflash2_keys(),
    )

    assert contract.architecture == architecture
    assert contract.block_size == 8
    assert contract.target_layer_ids == (5, 19, 33, 47, 61)


@pytest.mark.parametrize(
    "architecture",
    ["DFlashDraftModel", "Qwen3DFlashDraftModel"],
)
def test_checkpoint_contract_rejects_architecture_downgrade(
    architecture: str,
) -> None:
    dflash2_contract = _dflash2_contract()
    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match="plain DFlash architecture",
    ):
        dflash2_contract.validate_dflash2_checkpoint_contract(
            _checkpoint_config(architecture),
            _required_dflash2_keys(),
        )


@pytest.mark.parametrize("missing_key", sorted(_required_dflash2_keys()))
def test_checkpoint_contract_requires_every_dflash2_feature_tensor(
    missing_key: str,
) -> None:
    dflash2_contract = _dflash2_contract()
    keys = _required_dflash2_keys()
    keys.remove(missing_key)

    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match="missing DFlash2 tensors",
    ):
        dflash2_contract.validate_dflash2_checkpoint_contract(
            _checkpoint_config(),
            keys,
        )


@pytest.mark.parametrize(
    "unexpected_key",
    [
        "layers.5.attention_conv.base_kernel",
        "layers.0.mlp_conv.bias",
        "candidate_selector.hidden_projection.bias",
    ],
)
def test_checkpoint_contract_rejects_unknown_dflash2_feature_tensor(
    unexpected_key: str,
) -> None:
    dflash2_contract = _dflash2_contract()
    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match="unexpected DFlash2 tensors",
    ):
        dflash2_contract.validate_dflash2_checkpoint_contract(
            _checkpoint_config(),
            _required_dflash2_keys() | {unexpected_key},
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dflash_config", "block_size"), 16, "block_size=8"),
        (("dflash_config", "conv_kernel_size"), 0, "conv_kernel_size"),
        (("dflash_config", "conv_group_size"), 3, "must divide hidden_size"),
        (("dflash_config", "selector_rank"), 0, "selector_rank"),
        (("dflash_config", "selector_top_k"), 248321, "selector_top_k"),
        (
            ("dflash_config", "target_layer_ids"),
            [5, 19, 33, 47, 64],
            "target_layer_ids",
        ),
        (("num_hidden_layers",), 4, "one target tap per draft layer"),
    ],
)
def test_checkpoint_config_rejects_incompatible_geometry(
    path: tuple[str, ...], value: object, message: str
) -> None:
    dflash2_contract = _dflash2_contract()
    config = _checkpoint_config()
    if len(path) == 1:
        config[path[0]] = value
    else:
        nested = config[path[0]]
        assert isinstance(nested, dict)
        nested[path[1]] = value

    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match=message,
    ):
        dflash2_contract.validate_dflash2_checkpoint_contract(
            config,
            _required_dflash2_keys(),
        )


def test_runtime_contract_requires_seven_speculative_tokens() -> None:
    speculator_runtime.validate_speculator_runtime_contract(
        speculator_type="dflash2",
        num_speculative_tokens=7,
    )

    with pytest.raises(
        speculator_runtime.SpeculatorRuntimeError,
        match="num_speculative_tokens=7",
    ):
        speculator_runtime.validate_speculator_runtime_contract(
            speculator_type="dflash2",
            num_speculative_tokens=8,
        )


def test_live_runtime_refit_rejects_recognized_dflash2() -> None:
    with pytest.raises(
        speculator_runtime.SpeculatorRuntimeError,
        match="live refit is not implemented",
    ):
        speculator_runtime.DraftRuntimeAdapter.resolve(
            SimpleNamespace(get_draft_model=lambda: object()),
            speculator_type="dflash2",
            num_speculative_tokens=7,
            vllm_version="0.27.1",
            pp_rank=0,
            pp_size=1,
        )


@pytest.mark.parametrize("speculator_type", ["dflash", "dspark"])
def test_existing_runtime_variants_do_not_require_dflash2_geometry(
    speculator_type: str,
) -> None:
    draft_model = object()

    adapter = speculator_runtime.DraftRuntimeAdapter.resolve(
        SimpleNamespace(get_draft_model=lambda: draft_model),
        speculator_type=speculator_type,
        vllm_version="0.27.1",
        pp_rank=0,
        pp_size=1,
    )

    assert adapter.model is draft_model


@pytest.mark.parametrize("capability", ["training", "refit"])
def test_online_dflash2_capabilities_fail_with_explicit_error(
    capability: str,
) -> None:
    config = draft_config.DFlash2DraftConfig.model_validate(_draft_values())

    with pytest.raises(
        draft_config.DraftCapabilityError,
        match=f"DFlash2 {capability}",
    ):
        draft_config.require_draft_capability(config, capability=capability)


def test_enabled_dflash2_does_not_silently_request_plain_refit() -> None:
    config = draft_config.DFlash2DraftConfig.model_validate(_draft_values())

    with pytest.raises(
        draft_config.DraftCapabilityError,
        match="DFlash2 refit",
    ):
        draft_config.draft_refit_enabled(config)
