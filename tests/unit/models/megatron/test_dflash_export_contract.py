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

import pytest
import torch

import nemo_rl.models.megatron.draft as draft_api
import nemo_rl.models.megatron.draft.utils as draft_utils
from nemo_rl.models.megatron.draft.utils import (
    validate_dflash_export_state_dict,
)


pytestmark = pytest.mark.mcore


def test_raw_dflash_export_is_not_a_public_api() -> None:
    """Raw TP-local state must not masquerade as an interoperable export."""
    assert not hasattr(draft_api, "export_dflash_weights")
    assert not hasattr(draft_utils, "export_dflash_weights")


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "lm_head.weight",
        "module.draft_model.output_layer.weight",
        "draft.mask_embedding.weight",
        "module.mask_token",
    ],
)
def test_dflash_export_rejects_target_owned_components(forbidden_key: str) -> None:
    """Target-head and mask-token ownership violations fail before export."""
    with pytest.raises(ValueError, match=forbidden_key):
        validate_dflash_export_state_dict({forbidden_key: torch.ones(1)})


def test_dflash_export_checks_components_instead_of_substrings() -> None:
    """Related body parameter names are not rejected by substring matching."""
    allowed = {
        "head_projection.weight": torch.ones(1),
        "output_layernorm.weight": torch.ones(1),
        "mask_tokenizer_projection.weight": torch.ones(1),
    }

    validate_dflash_export_state_dict(allowed)


def test_dflash_body_export_is_logical_and_excludes_target_owned_weights() -> None:
    from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
    from nemo_rl.models.megatron.draft.utils import export_dflash_weights_to_hf

    body = DFlashBody(
        DFlashBodyConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_target_taps=2,
        )
    )

    exported = dict(export_dflash_weights_to_hf(body))

    assert set(exported) == set(body.state_dict())
    assert not any("lm_head" in name for name in exported)
    assert not any("embedding" in name for name in exported)
    assert not any("mask_token" in name for name in exported)
    for name, tensor in body.state_dict().items():
        torch.testing.assert_close(exported[name], tensor)


def test_dflash_tp_export_gathers_one_flat_dtype_device_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP reconstruction preserves order and values with one bucket gather."""
    config = type(
        "Config",
        (),
        {
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "num_target_taps": 2,
        },
    )()
    rank_shards = [
        (
            "fc.weight",
            torch.arange(16, dtype=torch.bfloat16).view(2, 8),
            torch.arange(16, 32, dtype=torch.bfloat16).view(2, 8),
            0,
        ),
        (
            "layers.0.self_attn.q_proj.weight",
            torch.arange(32, 40, dtype=torch.bfloat16).view(2, 4),
            torch.arange(40, 48, dtype=torch.bfloat16).view(2, 4),
            0,
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            torch.arange(48, 56, dtype=torch.bfloat16).view(4, 2),
            torch.arange(56, 64, dtype=torch.bfloat16).view(4, 2),
            1,
        ),
        (
            "layers.0.mlp.down_proj.weight",
            torch.arange(64, 76, dtype=torch.bfloat16).view(4, 3),
            torch.arange(76, 88, dtype=torch.bfloat16).view(4, 3),
            1,
        ),
    ]
    hidden_norm = torch.arange(4, dtype=torch.bfloat16)
    final_norm = torch.arange(4, 8, dtype=torch.bfloat16)
    local_state = {
        "fc.weight": rank_shards[0][1],
        "hidden_norm.weight": hidden_norm,
        "layers.0.self_attn.q_proj.weight": rank_shards[1][1],
        "layers.0.self_attn.o_proj.weight": rank_shards[2][1],
        "layers.0.mlp.down_proj.weight": rank_shards[3][1],
        "norm.weight": final_norm,
    }
    model = type(
        "Model",
        (),
        {"config": config, "state_dict": lambda self: local_state},
    )()
    reference = {
        name: torch.cat((rank_zero, rank_one), dim=split_axis).contiguous()
        for name, rank_zero, rank_one, split_axis in rank_shards
    }
    reference["hidden_norm.weight"] = hidden_norm
    reference["norm.weight"] = final_norm
    tp_group = object()
    gather_calls = 0
    flat_rank_zero = torch.cat(
        [rank_zero.contiguous().view(-1) for _, rank_zero, _, _ in rank_shards]
    )
    flat_rank_one = torch.cat(
        [rank_one.contiguous().view(-1) for _, _, rank_one, _ in rank_shards]
    )

    def fake_all_gather(
        gathered: list[torch.Tensor],
        local_bucket: torch.Tensor,
        *,
        group: object,
    ) -> None:
        nonlocal gather_calls
        gather_calls += 1
        assert group is tp_group
        gathered[0].copy_(local_bucket)
        if local_bucket.numel() == flat_rank_zero.numel():
            torch.testing.assert_close(local_bucket, flat_rank_zero)
            gathered[1].copy_(flat_rank_one)
            return
        for _, rank_zero, rank_one, _ in rank_shards:
            if torch.equal(local_bucket.flatten(), rank_zero.flatten()):
                gathered[1].copy_(rank_one)
                return
        raise AssertionError(
            f"unexpected local bucket shape {tuple(local_bucket.shape)}"
        )

    monkeypatch.setattr(draft_utils, "unwrap_model", lambda wrapped: wrapped)
    monkeypatch.setattr(
        draft_utils.parallel_state, "model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        draft_utils.parallel_state,
        "get_tensor_model_parallel_group",
        lambda: tp_group,
    )
    monkeypatch.setattr(
        draft_utils.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(draft_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(draft_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(draft_utils.dist, "all_gather", fake_all_gather)

    exported = draft_utils.export_dflash_weights_to_hf(model)
    exported_names = [name for name, _ in exported]
    exported_tensors = [tensor for _, tensor in exported]
    reference_names = list(local_state)
    reference_tensors = [reference[name] for name in reference_names]

    assert exported_names == reference_names
    for actual, expected in zip(exported_tensors, reference_tensors, strict=True):
        torch.testing.assert_close(actual, expected)
    assert dict(exported)["hidden_norm.weight"] is hidden_norm
    assert dict(exported)["norm.weight"] is final_norm
    assert gather_calls == 1
