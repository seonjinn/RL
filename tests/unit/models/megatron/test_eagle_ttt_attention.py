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

import pytest
import torch

from nemo_rl.models.megatron.draft.eagle_ttt import (
    EagleTTTAttentionPlan,
    EagleTTTKVCache,
    eagle_ttt_attention,
)


def _dense_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    compute_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    scores = (
        torch.einsum(
            "bhqd,bhkd->bhqk",
            query.to(compute_dtype),
            key.to(compute_dtype),
        )
        * scale
    )
    scores = scores.masked_fill(~visible[None, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, value.to(compute_dtype)).to(
        query.dtype
    )


def _expand_gqa(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    repeats = query_heads // tensor.shape[1]
    return tensor.repeat_interleave(repeats, dim=1)


@pytest.mark.parametrize("pass_index", [1, 3])
@pytest.mark.parametrize("query_heads,kv_heads", [(4, 4), (4, 2)])
def test_ttt_attention_matches_dense_output_and_all_gradients(
    pass_index: int, query_heads: int, kv_heads: int
) -> None:
    torch.manual_seed(17 + pass_index)
    batch, sequence, head_dim = 2, 5, 8
    plan = EagleTTTAttentionPlan(
        pass_index=pass_index, sequence_length=sequence, max_steps=4
    )

    query = torch.randn(
        batch, query_heads, sequence, head_dim, dtype=torch.float64, requires_grad=True
    )
    trunk_key = torch.randn(
        batch, kv_heads, sequence, head_dim, dtype=torch.float64, requires_grad=True
    )
    trunk_value = torch.randn_like(trunk_key, requires_grad=True)
    branch_keys = tuple(
        torch.randn_like(trunk_key, requires_grad=True) for _ in range(pass_index)
    )
    branch_values = tuple(
        torch.randn_like(trunk_value, requires_grad=True) for _ in range(pass_index)
    )
    cache = EagleTTTKVCache.empty(max_steps=4).with_trunk(trunk_key, trunk_value)
    for key, value in zip(branch_keys, branch_values):
        cache = cache.append_branch(key, value)

    actual = eagle_ttt_attention(query=query, cache=cache, plan=plan)
    key = torch.cat((trunk_key, *branch_keys), dim=2)
    value = torch.cat((trunk_value, *branch_values), dim=2)
    visible = plan.visibility_mask(device=query.device)
    expected = _dense_reference(
        query,
        _expand_gqa(key, query_heads),
        _expand_gqa(value, query_heads),
        visible,
        head_dim**-0.5,
    )
    torch.testing.assert_close(actual, expected)

    upstream = torch.randn_like(actual)
    actual_inputs = (query, trunk_key, trunk_value, *branch_keys, *branch_values)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, actual_inputs, upstream)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_plan_exposes_inference_aligned_offsets_and_visibility() -> None:
    plan = EagleTTTAttentionPlan(pass_index=2, sequence_length=4, max_steps=4)

    torch.testing.assert_close(plan.rope_positions(), torch.tensor([2, 3, 4, 5]))
    assert plan.teacher_offset == 3
    expected = torch.tensor(
        [
            [
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
            ],
        ]
    )
    torch.testing.assert_close(plan.visibility_mask(), expected)


def test_cache_is_immutable_and_preserves_tensor_storage() -> None:
    trunk_key = torch.randn(1, 2, 4, 8)
    trunk_value = torch.randn_like(trunk_key)
    branch_key = torch.randn_like(trunk_key)
    branch_value = torch.randn_like(trunk_value)

    empty = EagleTTTKVCache.empty(max_steps=4)
    trunk = empty.with_trunk(trunk_key, trunk_value)
    branch = trunk.append_branch(branch_key, branch_value)

    assert empty.trunk_key is None
    assert trunk.branches_key == ()
    assert branch.trunk_key is trunk_key
    assert branch.trunk_value is trunk_value
    assert branch.branches_key[0] is branch_key
    assert branch.branches_value[0] is branch_value


@pytest.mark.parametrize(
    "pass_index,max_steps",
    [(-1, 4), (4, 4), (1, 1), (0, 5)],
)
def test_plan_rejects_unbounded_or_invalid_passes(
    pass_index: int, max_steps: int
) -> None:
    with pytest.raises(ValueError):
        EagleTTTAttentionPlan(
            pass_index=pass_index, sequence_length=4, max_steps=max_steps
        )


def test_later_pass_requires_one_branch_per_completed_ttt_step() -> None:
    plan = EagleTTTAttentionPlan(pass_index=2, sequence_length=4, max_steps=4)
    key = torch.randn(1, 2, 4, 8)
    cache = EagleTTTKVCache.empty(max_steps=4).with_trunk(key, key)
    cache = cache.append_branch(key, key)

    with pytest.raises(ValueError, match="two branch"):
        eagle_ttt_attention(query=key, cache=cache, plan=plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_heads,kv_heads", [(4, 4), (4, 2)])
@pytest.mark.parametrize("pass_index", [1, 3])
def test_cuda_flex_trunk_matches_dense_output_and_all_gradients(
    dtype: torch.dtype, query_heads: int, kv_heads: int, pass_index: int
) -> None:
    torch.manual_seed(91 + pass_index)
    batch, sequence, head_dim = 2, 9, 16
    device = torch.device("cuda")
    plan = EagleTTTAttentionPlan(
        pass_index=pass_index, sequence_length=sequence, max_steps=4
    )
    query = torch.randn(
        batch,
        query_heads,
        sequence,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    trunk_key = torch.randn(
        batch,
        kv_heads,
        sequence,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    trunk_value = torch.randn_like(trunk_key, requires_grad=True)
    branch_keys = tuple(
        torch.randn_like(trunk_key, requires_grad=True) for _ in range(pass_index)
    )
    branch_values = tuple(
        torch.randn_like(trunk_value, requires_grad=True) for _ in range(pass_index)
    )
    cache = EagleTTTKVCache.empty(max_steps=4).with_trunk(trunk_key, trunk_value)
    for key, value in zip(branch_keys, branch_values, strict=True):
        cache = cache.append_branch(key, value)

    actual = eagle_ttt_attention(query=query, cache=cache, plan=plan)
    key = torch.cat((trunk_key, *branch_keys), dim=2)
    value = torch.cat((trunk_value, *branch_values), dim=2)
    expected = _dense_reference(
        query,
        _expand_gqa(key, query_heads),
        _expand_gqa(value, query_heads),
        plan.visibility_mask(device=device),
        head_dim**-0.5,
    )
    tolerance = 5e-2 if dtype is torch.bfloat16 else 4e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    upstream = torch.randn_like(actual)
    inputs = (query, trunk_key, trunk_value, *branch_keys, *branch_values)
    actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, inputs, upstream)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_backward_state_does_not_concatenate_branch_kv() -> None:
    batch, heads, sequence, head_dim = 1, 2, 64, 16
    plan = EagleTTTAttentionPlan(pass_index=3, sequence_length=sequence, max_steps=4)
    query = torch.randn(
        batch,
        heads,
        sequence,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    cache = EagleTTTKVCache.empty(max_steps=4).with_trunk(key, value)
    for _ in range(plan.pass_index):
        cache = cache.append_branch(
            torch.randn_like(key, requires_grad=True),
            torch.randn_like(value, requires_grad=True),
        )

    saved_shapes: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = eagle_ttt_attention(query=query, cache=cache, plan=plan)
        output.square().mean().backward()

    concatenated_length = sequence * (plan.pass_index + 1)
    assert not any(
        len(shape) == 4 and shape[-2] == concatenated_length and shape[-1] == head_dim
        for shape in saved_shapes
    )
