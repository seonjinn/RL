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

import ast
import importlib.util
import json
import struct
import sys
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[4]

_PUBLIC_DFLASH2_REPO = "incoai/Qwen3.8-27B-DFlash2"
_PUBLIC_DFLASH2_REVISION = (
    "dedf8df68adfb1afeaf7b7480c0a0243108177b4"  # pragma: allowlist secret
)
_PUBLIC_DFLASH2_CONFIG_SHA256 = (
    "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
)  # pragma: allowlist secret
_PUBLIC_DFLASH2_HEADER_BYTES = 8_928
_PUBLIC_DFLASH2_HEADER_SHA256 = (
    "0c2c70601b30f8d1ca7d5794b817779ba2dcf1956cfc7d4f83e87091e1ab7c8c"
)  # pragma: allowlist secret


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
            "nemo_rl/models/dflash2_contract.py",
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
        "dtype": "bfloat16",
        "head_dim": 128,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
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


def _required_dflash2_metadata(
    *,
    hidden_size: int = 5120,
    intermediate_size: int = 17408,
    head_dim: int = 128,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    vocab_size: int = 248320,
    num_layers: int = 5,
    conv_kernel_size: int = 2,
    conv_group_size: int = 16,
    selector_rank: int = 256,
) -> dict[str, dict[str, object]]:
    metadata = {
        "candidate_selector.hidden_projection.weight": {
            "shape": [selector_rank, hidden_size],
            "dtype": "BF16",
        },
        "candidate_selector.predecessor_codebook": {
            "shape": [vocab_size, selector_rank],
            "dtype": "BF16",
        },
        "candidate_selector.successor_codebook": {
            "shape": [vocab_size, selector_rank],
            "dtype": "BF16",
        },
        "fc.weight": {
            "shape": [hidden_size, hidden_size * num_layers],
            "dtype": "BF16",
        },
        "hidden_norm.weight": {"shape": [hidden_size], "dtype": "BF16"},
        "norm.weight": {"shape": [hidden_size], "dtype": "BF16"},
    }
    for layer_id in range(num_layers):
        prefix = f"layers.{layer_id}"
        metadata.update(
            {
                f"{prefix}.input_layernorm.weight": {
                    "shape": [hidden_size],
                    "dtype": "BF16",
                },
                f"{prefix}.mlp.down_proj.weight": {
                    "shape": [hidden_size, intermediate_size],
                    "dtype": "BF16",
                },
                f"{prefix}.mlp.gate_proj.weight": {
                    "shape": [intermediate_size, hidden_size],
                    "dtype": "BF16",
                },
                f"{prefix}.mlp.up_proj.weight": {
                    "shape": [intermediate_size, hidden_size],
                    "dtype": "BF16",
                },
                f"{prefix}.post_attention_layernorm.weight": {
                    "shape": [hidden_size],
                    "dtype": "BF16",
                },
                f"{prefix}.self_attn.k_norm.weight": {
                    "shape": [head_dim],
                    "dtype": "BF16",
                },
                f"{prefix}.self_attn.k_proj.weight": {
                    "shape": [num_key_value_heads * head_dim, hidden_size],
                    "dtype": "BF16",
                },
                f"{prefix}.self_attn.o_proj.weight": {
                    "shape": [hidden_size, num_attention_heads * head_dim],
                    "dtype": "BF16",
                },
                f"{prefix}.self_attn.q_norm.weight": {
                    "shape": [head_dim],
                    "dtype": "BF16",
                },
                f"{prefix}.self_attn.q_proj.weight": {
                    "shape": [num_attention_heads * head_dim, hidden_size],
                    "dtype": "BF16",
                },
                f"{prefix}.self_attn.v_proj.weight": {
                    "shape": [num_key_value_heads * head_dim, hidden_size],
                    "dtype": "BF16",
                },
            }
        )
        for component in ("attention_conv", "mlp_conv"):
            metadata[f"{prefix}.{component}.base_kernel"] = {
                "shape": [2, conv_kernel_size, hidden_size],
                "dtype": "BF16",
            }
            metadata[f"{prefix}.{component}.kernel_projection.weight"] = {
                "shape": [
                    2 * conv_kernel_size * (hidden_size // conv_group_size),
                    hidden_size,
                ],
                "dtype": "BF16",
            }
    return metadata


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
        ({"conv_kernel_size": 9}, "must not exceed block_size"),
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
        _required_dflash2_metadata(),
    )

    assert contract.architecture == architecture
    assert contract.block_size == 8
    assert contract.target_layer_ids == (5, 19, 33, 47, 61)


def test_published_checkpoint_provenance_and_exact_81_tensor_schema() -> None:
    dflash2_contract = _dflash2_contract()

    metadata = dflash2_contract.expected_dflash2_tensor_metadata(
        _checkpoint_config()
    )

    assert _PUBLIC_DFLASH2_REPO == "incoai/Qwen3.8-27B-DFlash2"
    assert len(_PUBLIC_DFLASH2_REVISION) == 40
    assert len(_PUBLIC_DFLASH2_CONFIG_SHA256) == 64
    assert _PUBLIC_DFLASH2_HEADER_BYTES == 8_928
    assert len(_PUBLIC_DFLASH2_HEADER_SHA256) == 64
    assert metadata == {
        name: dflash2_contract.DFlash2TensorMetadata(
            shape=tuple(value["shape"]),
            dtype=value["dtype"],
        )
        for name, value in _required_dflash2_metadata().items()
    }
    assert len(metadata) == 81


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
            _required_dflash2_metadata(),
        )


@pytest.mark.parametrize("missing_key", sorted(_required_dflash2_metadata()))
def test_checkpoint_contract_requires_every_published_tensor(
    missing_key: str,
) -> None:
    dflash2_contract = _dflash2_contract()
    metadata = _required_dflash2_metadata()
    metadata.pop(missing_key)

    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match="missing DFlash2 tensors",
    ):
        dflash2_contract.validate_dflash2_checkpoint_contract(
            _checkpoint_config(),
            metadata,
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
            {
                **_required_dflash2_metadata(),
                unexpected_key: {"shape": [1], "dtype": "BF16"},
            },
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dflash_config", "block_size"), 16, "block_size=8"),
        (("dflash_config", "conv_kernel_size"), 0, "conv_kernel_size"),
        (
            ("dflash_config", "conv_kernel_size"),
            9,
            "must not exceed block_size",
        ),
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
            _required_dflash2_metadata(),
        )


@pytest.mark.parametrize(
    ("tensor_name", "field", "value", "message"),
    [
        ("fc.weight", "shape", [1, 1], "fc.weight.*shape"),
        (
            "layers.0.attention_conv.base_kernel",
            "shape",
            [1, 2, 5120],
            "attention_conv.base_kernel.*shape",
        ),
        (
            "candidate_selector.hidden_projection.weight",
            "dtype",
            "F16",
            "hidden_projection.weight.*dtype",
        ),
    ],
)
def test_checkpoint_contract_rejects_wrong_tensor_shape_or_dtype(
    tensor_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    dflash2_contract = _dflash2_contract()
    metadata = _required_dflash2_metadata()
    metadata[tensor_name][field] = value

    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match=message,
    ):
        dflash2_contract.validate_dflash2_checkpoint_contract(
            _checkpoint_config(),
            metadata,
        )


def _write_safetensors_metadata(
    path: Path,
    metadata: Mapping[str, Mapping[str, object]],
) -> None:
    offset = 0
    header: dict[str, object] = {}
    dtype_size = {"BF16": 2}
    for name, tensor in metadata.items():
        shape = tensor["shape"]
        dtype = tensor["dtype"]
        assert isinstance(shape, list)
        assert isinstance(dtype, str)
        size = dtype_size[dtype]
        for dimension in shape:
            assert isinstance(dimension, int)
            size *= dimension
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)
    with path.open("r+b") as checkpoint_file:
        checkpoint_file.truncate(8 + len(header_bytes) + offset)


def test_local_checkpoint_inspection_validates_config_and_full_header(
    tmp_path: Path,
) -> None:
    dflash2_contract = _dflash2_contract()
    config = _checkpoint_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_safetensors_metadata(
        tmp_path / "model.safetensors",
        _required_dflash2_metadata(),
    )

    contract = dflash2_contract.inspect_dflash2_checkpoint_if_present(tmp_path)

    assert contract is not None
    assert contract.tensor_count == 81
    assert contract.dtype == "BF16"


def test_local_checkpoint_inspection_rejects_truncated_tensor_payload(
    tmp_path: Path,
) -> None:
    dflash2_contract = _dflash2_contract()
    (tmp_path / "config.json").write_text(json.dumps(_checkpoint_config()))
    checkpoint_path = tmp_path / "model.safetensors"
    _write_safetensors_metadata(checkpoint_path, _required_dflash2_metadata())
    with checkpoint_path.open("r+b") as checkpoint_file:
        checkpoint_file.truncate(_PUBLIC_DFLASH2_HEADER_BYTES + 8)

    with pytest.raises(
        dflash2_contract.DFlash2CheckpointContractError,
        match="payload size",
    ):
        dflash2_contract.inspect_dflash2_checkpoint_if_present(tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3"},
        {"architectures": ["Qwen3ForCausalLM", "Qwen3ForSequenceClassification"]},
    ],
)
def test_checkpoint_inspection_ignores_non_dflash2_configs(
    tmp_path: Path,
    config: Mapping[str, object],
) -> None:
    dflash2_contract = _dflash2_contract()
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert dflash2_contract.inspect_dflash2_checkpoint_if_present(tmp_path) is None


def test_static_vllm_startup_detects_dflash2_under_dflash_method(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_safetensors_metadata(
        tmp_path / "model.safetensors",
        _required_dflash2_metadata(),
    )

    contract = speculator_runtime.validate_vllm_speculative_startup(
        {
            "method": "dflash",
            "model": str(tmp_path),
            "num_speculative_tokens": 7,
        }
    )

    assert contract is not None
    assert contract.architecture == "DFlash2DraftModel"


def test_static_vllm_startup_rejects_dflash2_wrong_k(tmp_path: Path) -> None:
    config = _checkpoint_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_safetensors_metadata(
        tmp_path / "model.safetensors",
        _required_dflash2_metadata(),
    )

    with pytest.raises(
        speculator_runtime.SpeculatorRuntimeError,
        match="num_speculative_tokens=7",
    ):
        speculator_runtime.validate_vllm_speculative_startup(
            {
                "method": "dflash",
                "model": str(tmp_path),
                "num_speculative_tokens": 6,
            }
        )


def test_vllm_worker_wires_static_checkpoint_validation_before_startup() -> None:
    source = (REPO_ROOT / "nemo_rl/models/generation/vllm/vllm_worker.py").read_text()
    module = ast.parse(source)
    load_model = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_model"
    )
    calls = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
        ): node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }

    assert "validate_vllm_speculative_startup" in calls
    assert calls["validate_vllm_speculative_startup"] < calls["_create_engine"]


def test_plain_dflash_loader_uses_guarded_common_checkpoint_route() -> None:
    source = (REPO_ROOT / "nemo_rl/models/megatron/draft/utils.py").read_text()
    module = ast.parse(source)
    loader = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_hf_weights_to_dflash"
    )
    calls = {
        node.func.id
        for node in ast.walk(loader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_load_checkpoint_state" in calls


def test_common_megatron_checkpoint_route_guards_dflash2_before_loading() -> None:
    source = (REPO_ROOT / "nemo_rl/models/megatron/draft/utils.py").read_text()
    module = ast.parse(source)
    loader = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint_state"
    )
    calls = [
        node.func.id
        for node in ast.walk(loader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert "inspect_dflash2_checkpoint_if_present" in calls


def test_vllm_refit_setup_uses_dflash2_refit_boundary() -> None:
    source = (REPO_ROOT / "nemo_rl/models/generation/vllm/vllm_backend.py").read_text()
    module = ast.parse(source)
    prepare_refit = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_refit_info"
    )
    calls = {
        node.func.id
        for node in ast.walk(prepare_refit)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "validate_vllm_refit_boundary" in calls


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


def test_dflash2_refit_boundary_allows_target_only_updates() -> None:
    speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DFlash2DraftModel"])
        ),
    )

    assert (
        speculator_runtime.validate_vllm_refit_boundary(
            speculative_config,
            state_dict_names=("model.layers.0.weight",),
        )
        == "dflash2"
    )


def test_dflash2_refit_boundary_rejects_draft_weight_updates() -> None:
    speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DFlash2DraftModel"])
        ),
    )

    with pytest.raises(
        speculator_runtime.SpeculatorRuntimeError,
        match="live refit is not implemented",
    ):
        speculator_runtime.validate_vllm_refit_boundary(
            speculative_config,
            state_dict_names=("draft.layers.0.weight",),
        )


def test_vllm_loaded_dflash2_architecture_is_not_treated_as_plain_dflash() -> None:
    speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DFlash2DraftModel"])
        ),
    )

    assert (
        speculator_runtime.resolve_vllm_speculator_type(speculative_config) == "dflash2"
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
