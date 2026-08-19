from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig


_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _tiny_config() -> DFlashBodyConfig:
    return DFlashBodyConfig(
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=2,
        num_target_taps=2,
        rope_theta=10_000.0,
    )


def test_qwen3_8b_defaults_are_pinned_and_frozen() -> None:
    config = DFlashBodyConfig()

    assert config.hidden_size == 4096
    assert config.intermediate_size == 12288
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.num_hidden_layers == 5
    assert config.num_target_taps == 5
    assert config.rope_theta == 1_000_000.0
    assert config.rms_norm_eps == 1e-6
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 8  # type: ignore[misc]


def test_exact_public_body_state_dict_schema_and_shapes() -> None:
    with torch.device("meta"):
        body = DFlashBody(DFlashBodyConfig())
    state = body.state_dict()
    expected_keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    expected_keys.update(
        f"layers.{layer_index}.{suffix}"
        for layer_index in range(5)
        for suffix in _LAYER_SUFFIXES
    )

    assert set(state) == expected_keys
    assert tuple(state["fc.weight"].shape) == (4096, 20480)
    assert tuple(state["hidden_norm.weight"].shape) == (4096,)
    assert tuple(state["norm.weight"].shape) == (4096,)
    for layer_index in range(5):
        prefix = f"layers.{layer_index}."
        assert tuple(state[prefix + "input_layernorm.weight"].shape) == (4096,)
        assert tuple(state[prefix + "self_attn.q_proj.weight"].shape) == (
            4096,
            4096,
        )
        assert tuple(state[prefix + "self_attn.k_proj.weight"].shape) == (
            1024,
            4096,
        )
        assert tuple(state[prefix + "self_attn.v_proj.weight"].shape) == (
            1024,
            4096,
        )
        assert tuple(state[prefix + "self_attn.o_proj.weight"].shape) == (
            4096,
            4096,
        )
        assert tuple(state[prefix + "self_attn.q_norm.weight"].shape) == (128,)
        assert tuple(state[prefix + "self_attn.k_norm.weight"].shape) == (128,)
        assert tuple(state[prefix + "post_attention_layernorm.weight"].shape) == (4096,)
        assert tuple(state[prefix + "mlp.gate_proj.weight"].shape) == (
            12288,
            4096,
        )
        assert tuple(state[prefix + "mlp.up_proj.weight"].shape) == (
            12288,
            4096,
        )
        assert tuple(state[prefix + "mlp.down_proj.weight"].shape) == (
            4096,
            12288,
        )


def test_body_owns_no_target_embedding_head_or_mask_tensor() -> None:
    body = DFlashBody(_tiny_config())
    forbidden_fragments = ("embed_tokens", "lm_head", "mask_embedding", "mask_token")

    assert all(
        fragment not in name
        for name in body.state_dict()
        for fragment in forbidden_fragments
    )
    assert all(
        fragment not in name
        for name, _ in body.named_modules()
        for fragment in forbidden_fragments
    )


def test_strict_state_dict_load_accepts_exact_schema_and_rejects_drift() -> None:
    torch.manual_seed(41)
    source = DFlashBody(_tiny_config())
    exact_state = source.state_dict()
    restored = DFlashBody(_tiny_config())

    incompatible = restored.load_state_dict(exact_state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    for name, parameter in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter)

    missing_state = dict(exact_state)
    del missing_state["norm.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing_state, strict=True)
    unexpected_state = dict(exact_state)
    unexpected_state["mask_embedding.weight"] = torch.empty(1, 8)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(unexpected_state, strict=True)


def test_sharded_state_dict_preserves_public_names_and_prefix() -> None:
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    body = DFlashBody(_tiny_config())
    sharded = body.sharded_state_dict(prefix="draft.")

    assert set(sharded) == {f"draft.{name}" for name in body.state_dict()}
    for name, sharded_tensor in sharded.items():
        assert isinstance(sharded_tensor, mapping.ShardedTensor)
        assert sharded_tensor.key == name
        assert tuple(sharded_tensor.global_shape) == tuple(sharded_tensor.data.shape)


def test_megatron_sharded_checkpoint_round_trip(tmp_path: Path) -> None:
    dist_checkpointing = pytest.importorskip("megatron.core.dist_checkpointing")
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    torch.manual_seed(43)
    source = DFlashBody(_tiny_config())
    sharded = source.sharded_state_dict(prefix="draft.")
    assert all(isinstance(value, mapping.ShardedTensor) for value in sharded.values())

    checkpoint_dir = tmp_path / "dflash_dcp"
    dist_checkpointing.save({"model": sharded}, str(checkpoint_dir))
    restored = DFlashBody(_tiny_config())
    template = restored.sharded_state_dict(prefix="draft.")
    loaded: dict[str, Any] = dist_checkpointing.load(
        {"model": template},
        str(checkpoint_dir),
    )
    unprefixed = {
        name.removeprefix("draft."): tensor for name, tensor in loaded["model"].items()
    }
    incompatible = restored.load_state_dict(unprefixed, strict=True)

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    for name, parameter in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter)
