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

"""Dense-oracle tests for DFlash structured block attention."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor


pytestmark = pytest.mark.mcore

_PLAN_MODULE = "nemo_rl.models.megatron.draft.block_plan"
_ATTENTION_MODULE = "nemo_rl.models.megatron.draft.block_attention"


def _load_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(
            f"DFlash production contract is missing: {error}",
            pytrace=False,
        )


def _load_attention_contract() -> tuple[type[Any], Any]:
    plan_module = _load_module(_PLAN_MODULE)
    attention_module = _load_module(_ATTENTION_MODULE)
    return plan_module.DFlashBatchPlan, attention_module.dflash_block_attention


def _make_plan(
    plan_type: type[Any],
    *,
    token_valid_mask: Tensor,
    sample_rows: list[int],
    anchor_positions: list[int],
    slot_valid: Tensor,
) -> Any:
    batch_size, sequence_length = token_valid_mask.shape
    num_blocks, block_size = slot_valid.shape
    assert num_blocks == len(sample_rows) == len(anchor_positions)
    assert num_blocks % batch_size == 0
    sample_rows_tensor = torch.tensor(sample_rows, dtype=torch.int64)
    anchors_tensor = torch.tensor(anchor_positions, dtype=torch.int64)
    sequence_positions = torch.arange(sequence_length)
    trunk_lengths = (
        token_valid_mask[sample_rows_tensor]
        & (sequence_positions.unsqueeze(0) < anchors_tensor.unsqueeze(1))
    ).sum(dim=1)
    safe_positions = torch.clamp(
        anchors_tensor.unsqueeze(1) + torch.arange(block_size),
        min=0,
        max=sequence_length - 1,
    )
    loss_mask = slot_valid.clone()
    loss_mask[:, 0] = False
    return plan_type(
        token_valid_mask=token_valid_mask,
        sample_rows=sample_rows_tensor,
        anchor_ids=torch.arange(num_blocks, dtype=torch.int64),
        anchor_positions=anchors_tensor,
        trunk_lengths=trunk_lengths,
        query_positions=safe_positions,
        label_positions=safe_positions.clone(),
        block_valid=slot_valid.any(dim=1),
        slot_valid=slot_valid,
        loss_mask=loss_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        anchors_per_sample=num_blocks // batch_size,
        gamma=block_size - 1,
        block_size=block_size,
    )


def _dense_attention_oracle(
    *,
    plan: Any,
    trunk_q: Tensor,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Independent scalar-loop implementation of the written visibility rules."""
    batch_size, sequence_length, num_query_heads, head_dim = trunk_q.shape
    num_kv_heads = trunk_k.shape[2]
    heads_per_group = num_query_heads // num_kv_heads
    effective_scale = head_dim**-0.5 if scale is None else scale
    trunk_output = torch.zeros_like(trunk_q)
    block_output = torch.zeros_like(block_q)

    for batch_index in range(batch_size):
        for query_position in range(sequence_length):
            if not bool(plan.token_valid_mask[batch_index, query_position]):
                continue
            visible_positions = [
                key_position
                for key_position in range(query_position + 1)
                if bool(plan.token_valid_mask[batch_index, key_position])
            ]
            for query_head in range(num_query_heads):
                kv_head = query_head // heads_per_group
                query = trunk_q[batch_index, query_position, query_head]
                keys = torch.stack(
                    [
                        trunk_k[batch_index, key_position, kv_head]
                        for key_position in visible_positions
                    ]
                )
                values = torch.stack(
                    [
                        trunk_v[batch_index, key_position, kv_head]
                        for key_position in visible_positions
                    ]
                )
                probabilities = torch.softmax(
                    torch.mv(keys, query) * effective_scale,
                    dim=0,
                )
                trunk_output[batch_index, query_position, query_head] = (
                    probabilities.unsqueeze(0) @ values
                ).squeeze(0)

    num_blocks, block_size = plan.slot_valid.shape
    for block_index in range(num_blocks):
        sample_row = int(plan.sample_rows[block_index])
        anchor_position = int(plan.anchor_positions[block_index])
        visible_trunk_positions = [
            position
            for position in range(sequence_length)
            if position < anchor_position
            and bool(plan.token_valid_mask[sample_row, position])
        ]
        visible_block_positions = [
            position
            for position in range(block_size)
            if bool(plan.slot_valid[block_index, position])
        ]
        for query_position in range(block_size):
            if not bool(plan.slot_valid[block_index, query_position]):
                continue
            for query_head in range(num_query_heads):
                kv_head = query_head // heads_per_group
                query = block_q[block_index, query_position, query_head]
                keys = [
                    trunk_k[sample_row, position, kv_head]
                    for position in visible_trunk_positions
                ] + [
                    block_k[block_index, position, kv_head]
                    for position in visible_block_positions
                ]
                values = [
                    trunk_v[sample_row, position, kv_head]
                    for position in visible_trunk_positions
                ] + [
                    block_v[block_index, position, kv_head]
                    for position in visible_block_positions
                ]
                stacked_keys = torch.stack(keys)
                stacked_values = torch.stack(values)
                probabilities = torch.softmax(
                    torch.mv(stacked_keys, query) * effective_scale,
                    dim=0,
                )
                block_output[block_index, query_position, query_head] = (
                    probabilities.unsqueeze(0) @ stacked_values
                ).squeeze(0)

    return trunk_output, block_output


def _clone_with_grad(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads",
    [
        pytest.param(2, 2, id="mha"),
        pytest.param(4, 2, id="gqa"),
    ],
)
def test_dense_fp32_forward_and_qkv_gradient_parity(
    num_query_heads: int,
    num_kv_heads: int,
) -> None:
    """Catches wrong visibility, GQA head mapping, scaling, or backward math."""
    plan_type, attention = _load_attention_contract()
    token_valid_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    plan = _make_plan(
        plan_type,
        token_valid_mask=token_valid_mask,
        sample_rows=[0, 0, 0, 1],
        anchor_positions=[0, 3, 4, 2],
        slot_valid=torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ]
        ),
    )
    generator = torch.Generator().manual_seed(1234)
    head_dim = 3
    tensors = (
        torch.randn((2, 4, num_query_heads, head_dim), generator=generator),
        torch.randn((2, 4, num_kv_heads, head_dim), generator=generator),
        torch.randn((2, 4, num_kv_heads, head_dim), generator=generator),
        torch.randn((4, 3, num_query_heads, head_dim), generator=generator),
        torch.randn((4, 3, num_kv_heads, head_dim), generator=generator),
        torch.randn((4, 3, num_kv_heads, head_dim), generator=generator),
    )
    production_inputs = _clone_with_grad(tensors)
    oracle_inputs = _clone_with_grad(tensors)

    production_outputs = attention(
        plan=plan,
        trunk_q=production_inputs[0],
        trunk_k=production_inputs[1],
        trunk_v=production_inputs[2],
        block_q=production_inputs[3],
        block_k=production_inputs[4],
        block_v=production_inputs[5],
    )
    oracle_outputs = _dense_attention_oracle(
        plan=plan,
        trunk_q=oracle_inputs[0],
        trunk_k=oracle_inputs[1],
        trunk_v=oracle_inputs[2],
        block_q=oracle_inputs[3],
        block_k=oracle_inputs[4],
        block_v=oracle_inputs[5],
    )

    assert len(production_outputs) == 2
    torch.testing.assert_close(production_outputs[0], oracle_outputs[0])
    torch.testing.assert_close(production_outputs[1], oracle_outputs[1])

    trunk_weight = torch.randn(production_outputs[0].shape, generator=generator)
    block_weight = torch.randn(production_outputs[1].shape, generator=generator)
    production_loss = (production_outputs[0] * trunk_weight).sum() + (
        production_outputs[1] * block_weight
    ).sum()
    oracle_loss = (oracle_outputs[0] * trunk_weight).sum() + (
        oracle_outputs[1] * block_weight
    ).sum()
    production_gradients = torch.autograd.grad(production_loss, production_inputs)
    oracle_gradients = torch.autograd.grad(oracle_loss, oracle_inputs)

    for production_gradient, oracle_gradient in zip(
        production_gradients,
        oracle_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            production_gradient,
            oracle_gradient,
            atol=2e-5,
            rtol=2e-5,
        )


def test_block_queries_cover_empty_remainder_and_full_trunk_boundaries() -> None:
    """Catches inclusion of the anchor or exclusion of valid prefix boundaries."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 4), dtype=torch.bool),
        sample_rows=[0, 0, 0],
        anchor_positions=[0, 2, 4],
        slot_valid=torch.ones((3, 2), dtype=torch.bool),
    )
    trunk_q = torch.zeros((1, 4, 1, 1))
    trunk_k = torch.zeros((1, 4, 1, 1))
    trunk_v = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    block_q = torch.zeros((3, 2, 1, 1))
    block_k = torch.zeros((3, 2, 1, 1))
    block_v = torch.tensor(
        [
            [[[10.0]], [[20.0]]],
            [[[10.0]], [[20.0]]],
            [[[10.0]], [[20.0]]],
        ]
    )

    _, block_output = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )

    expected = torch.tensor([15.0, 8.25, 40.0 / 6.0]).reshape(3, 1, 1, 1)
    torch.testing.assert_close(block_output, expected.expand(-1, 2, -1, -1))


def test_duplicate_anchors_and_multiple_rows_remain_block_local() -> None:
    """Catches cross-block and cross-sample K/V leakage."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((2, 2), dtype=torch.bool),
        sample_rows=[0, 0, 1, 1],
        anchor_positions=[1, 1, 2, 2],
        slot_valid=torch.ones((4, 2), dtype=torch.bool),
    )
    trunk_q = torch.zeros((2, 2, 1, 1))
    trunk_k = torch.zeros((2, 2, 1, 1))
    trunk_v = torch.tensor([1.0, 3.0, 100.0, 300.0]).reshape(2, 2, 1, 1)
    block_q = torch.zeros((4, 2, 1, 1))
    block_k = torch.zeros((4, 2, 1, 1))
    block_v = torch.tensor(
        [
            [[[5.0]], [[7.0]]],
            [[[50.0]], [[70.0]]],
            [[[500.0]], [[700.0]]],
            [[[5000.0]], [[7000.0]]],
        ]
    )

    _, baseline = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )
    changed_block_v = block_v.clone()
    changed_block_v[1:] = 1_000_000.0
    _, changed = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=changed_block_v,
    )

    torch.testing.assert_close(baseline[0], torch.full_like(baseline[0], 13.0 / 3.0))
    torch.testing.assert_close(changed[0], baseline[0])
    assert not torch.equal(baseline[1], baseline[0])
    assert not torch.equal(baseline[2], baseline[0])


def test_invalid_block_queries_are_zero_with_finite_isolated_gradients() -> None:
    """Catches NaNs or gradient flow through masked block queries and K/V slots."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        sample_rows=[0],
        anchor_positions=[1],
        slot_valid=torch.tensor([[True, False, True]]),
    )
    generator = torch.Generator().manual_seed(77)
    inputs = _clone_with_grad(
        (
            torch.randn((1, 2, 2, 3), generator=generator),
            torch.randn((1, 2, 1, 3), generator=generator),
            torch.randn((1, 2, 1, 3), generator=generator),
            torch.randn((1, 3, 2, 3), generator=generator),
            torch.randn((1, 3, 1, 3), generator=generator),
            torch.randn((1, 3, 1, 3), generator=generator),
        )
    )

    trunk_output, block_output = attention(
        plan=plan,
        trunk_q=inputs[0],
        trunk_k=inputs[1],
        trunk_v=inputs[2],
        block_q=inputs[3],
        block_k=inputs[4],
        block_v=inputs[5],
    )
    loss = trunk_output.square().sum() + block_output.square().sum()
    gradients = torch.autograd.grad(loss, inputs)

    assert torch.equal(block_output[:, 1], torch.zeros_like(block_output[:, 1]))
    assert torch.isfinite(trunk_output).all()
    assert torch.isfinite(block_output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.equal(gradients[3][:, 1], torch.zeros_like(gradients[3][:, 1]))
    assert torch.equal(gradients[4][:, 1], torch.zeros_like(gradients[4][:, 1]))
    assert torch.equal(gradients[5][:, 1], torch.zeros_like(gradients[5][:, 1]))
