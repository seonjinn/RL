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

import importlib
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


def _load_symbols():
    """Load the pure TTT module without importing optional Megatron dependencies."""
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.eagle_ttt")
    return (
        module.EagleTTTAttentionPlan,
        module.EagleTTTState,
        module.EagleTTTStoragePlan,
        module.eagle_ttt_attention,
    )


def _dense_reference(
    *,
    query: torch.Tensor,
    trunk_key: torch.Tensor,
    trunk_value: torch.Tensor,
    branch_keys: tuple[torch.Tensor, ...],
    branch_values: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    query_heads = query.shape[1]
    repeats = query_heads // trunk_key.shape[1]
    key = torch.cat((trunk_key, *branch_keys), dim=2).repeat_interleave(repeats, dim=1)
    value = torch.cat((trunk_value, *branch_values), dim=2).repeat_interleave(
        repeats, dim=1
    )
    sequence = query.shape[2]
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * query.shape[-1] ** -0.5
    trunk_visible = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    branch_visible = torch.eye(sequence, dtype=torch.bool).repeat(1, len(branch_keys))
    visible = torch.cat((trunk_visible, branch_visible), dim=1)
    probabilities = scores.masked_fill(~visible[None, None], float("-inf")).softmax(
        dim=-1
    )
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, value)


@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
@pytest.mark.parametrize("query_heads,kv_heads", [(4, 4), (4, 2)])
def test_attention_matches_dense_output_and_every_input_gradient(
    pass_count: int,
    query_heads: int,
    kv_heads: int,
) -> None:
    (
        EagleTTTAttentionPlan,
        EagleTTTState,
        _,
        eagle_ttt_attention,
    ) = _load_symbols()
    torch.manual_seed(31 + pass_count)
    batch, sequence, head_dim = 1, 5, 8
    query = torch.randn(
        batch,
        query_heads,
        sequence,
        head_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    trunk_key = torch.randn(
        batch,
        kv_heads,
        sequence,
        head_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    trunk_value = torch.randn_like(trunk_key, requires_grad=True)
    branch_keys = tuple(
        torch.randn_like(trunk_key, requires_grad=True) for _ in range(pass_count - 1)
    )
    branch_values = tuple(
        torch.randn_like(trunk_value, requires_grad=True) for _ in range(pass_count - 1)
    )
    state = EagleTTTState.from_trunk(
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for branch_key, branch_value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(branch_key=branch_key, branch_value=branch_value)
    plan = EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )

    actual = eagle_ttt_attention(query=query, state=state, plan=plan)
    expected = _dense_reference(
        query=query,
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        branch_keys=branch_keys,
        branch_values=branch_values,
    )
    torch.testing.assert_close(actual, expected)

    upstream = torch.randn_like(actual)
    differentiable_inputs = (
        query,
        trunk_key,
        trunk_value,
        *branch_keys,
        *branch_values,
    )
    actual_gradients = torch.autograd.grad(
        actual, differentiable_inputs, upstream, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, differentiable_inputs, upstream)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("sequence_length", [8_192, 32_768, 262_144])
@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_rope_positions_and_retained_storage_scale_linearly(
    sequence_length: int,
    pass_count: int,
) -> None:
    EagleTTTAttentionPlan, _, EagleTTTStoragePlan, _ = _load_symbols()
    plan = EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence_length,
    )
    positions = plan.rope_positions()
    assert positions.dtype == torch.int64
    assert positions.shape == (sequence_length,)
    assert positions[0].item() == 0
    assert positions[-1].item() == sequence_length - 1

    storage = EagleTTTStoragePlan(
        batch_size=2,
        kv_heads=4,
        sequence_length=sequence_length,
        head_dim=16,
        dtype=torch.bfloat16,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 40,
    )
    bytes_per_pass = 2 * 2 * 4 * sequence_length * 16 * 2
    assert storage.retained_bytes == pass_count * bytes_per_pass
    assert all(
        not isinstance(getattr(plan, field.name), torch.Tensor)
        for field in fields(plan)
    )


def test_flex_attention_is_compiled_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.eagle_ttt")
    module._compiled_flex_attention.cache_clear()
    compile_calls: list[tuple[Any, bool]] = []

    def compile_stub(function: Any, *, dynamic: bool) -> Any:
        compile_calls.append((function, dynamic))
        return function

    monkeypatch.setattr(torch, "compile", compile_stub)
    compiled = module._compiled_flex_attention()

    assert callable(compiled)
    assert len(compile_calls) == 1
    assert compile_calls[0][1] is True
    module._compiled_flex_attention.cache_clear()


def test_budget_and_maximum_are_rejected_before_state_construction() -> None:
    _, EagleTTTState, EagleTTTStoragePlan, _ = _load_symbols()
    with pytest.raises(ValueError, match="configured maximum"):
        EagleTTTStoragePlan(
            batch_size=1,
            kv_heads=1,
            sequence_length=32_768,
            head_dim=8,
            dtype=torch.bfloat16,
            pass_count=9,
            max_passes=8,
            activation_budget_bytes=1 << 30,
        )
    with pytest.raises(ValueError, match="activation budget"):
        EagleTTTStoragePlan(
            batch_size=1,
            kv_heads=1,
            sequence_length=262_144,
            head_dim=128,
            dtype=torch.bfloat16,
            pass_count=8,
            max_passes=8,
            activation_budget_bytes=1,
        )

    key = torch.randn(1, 1, 2, 4)
    with pytest.raises(ValueError, match="configured maximum"):
        EagleTTTState.from_trunk(
            trunk_key=key,
            trunk_value=key,
            pass_count=9,
            max_passes=8,
            activation_budget_bytes=1 << 20,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_backward_does_not_save_sequence_square_mask_or_packed_pass_kv() -> None:
    (
        EagleTTTAttentionPlan,
        EagleTTTState,
        _,
        eagle_ttt_attention,
    ) = _load_symbols()
    sequence, head_dim, pass_count = 128, 16, 4
    query = torch.randn(
        1,
        2,
        sequence,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    state = EagleTTTState.from_trunk(
        trunk_key=key,
        trunk_value=value,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for _ in range(pass_count - 1):
        state = state.append_branch(
            branch_key=torch.randn_like(key, requires_grad=True),
            branch_value=torch.randn_like(value, requires_grad=True),
        )
    plan = EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )
    saved_shapes: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        eagle_ttt_attention(query=query, state=state, plan=plan).sum().backward()

    assert torch.Size((sequence, sequence)) not in saved_shapes
    assert not any(
        len(shape) == 4 and shape[-2] == sequence * pass_count for shape in saved_shapes
    )
