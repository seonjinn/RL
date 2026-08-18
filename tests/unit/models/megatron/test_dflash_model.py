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

"""Internal DFlash model behavior and checkpoint contract tests."""

from __future__ import annotations

import importlib
import io
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor, nn


pytestmark = pytest.mark.mcore

_PLAN_MODULE = "nemo_rl.models.megatron.draft.block_plan"
_MODEL_MODULE = "nemo_rl.models.megatron.draft.dflash"


def _load_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(
            f"DFlash production contract is missing: {error}",
            pytrace=False,
        )


def _load_model_contract() -> tuple[ModuleType, type[nn.Module]]:
    plan_module = _load_module(_PLAN_MODULE)
    model_module = _load_module(_MODEL_MODULE)
    return plan_module, model_module.DFlashModel


def _build_plan(
    plan_module: ModuleType,
    token_valid_mask: Tensor,
    sample_ids: Tensor,
) -> Any:
    return plan_module.build_dflash_batch_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=2,
        gamma=2,
        optimizer_step=7,
        seed=11,
    )


def _new_model(model_type: type[nn.Module]) -> nn.Module:
    return model_type(
        target_hidden_size=6,
        draft_hidden_size=8,
        num_target_hidden_taps=2,
        num_layers=2,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=16,
    )


def test_model_forward_backward_uses_caller_tensors_and_draft_parameters() -> None:
    """Catches detached target features, unused draft weights, and wrong output shape."""
    plan_module, model_type = _load_model_contract()
    plan = _build_plan(
        plan_module,
        torch.ones((2, 6), dtype=torch.bool),
        torch.tensor([101, 202], dtype=torch.int64),
    )
    torch.manual_seed(123)
    model = _new_model(model_type)
    target_hidden_taps = torch.randn((2, 2, 6, 6), requires_grad=True)
    input_embeddings = torch.randn((4, 3, 6), requires_grad=True)

    output = model(
        plan=plan,
        target_hidden_taps=target_hidden_taps,
        input_embeddings=input_embeddings,
    )
    loss = output.square().mean() + output.sum() * 0.01
    loss.backward()

    assert output.shape == (4, 3, 6)
    assert torch.isfinite(output).all()
    assert target_hidden_taps.grad is not None
    assert torch.isfinite(target_hidden_taps.grad).all()
    assert target_hidden_taps.grad.abs().sum() > 0
    assert input_embeddings.grad is not None
    assert torch.isfinite(input_embeddings.grad).all()
    assert input_embeddings.grad.abs().sum() > 0

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_parameters
    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, f"unused trainable draft parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_model_owns_neither_target_head_nor_mask_embedding() -> None:
    """Catches accidental duplication of parameters that remain target-owned."""
    _, model_type = _load_model_contract()
    model = _new_model(model_type)
    parameter_names = {name for name, _ in model.named_parameters()}

    assert not any("lm_head" in name for name in parameter_names)
    assert not any("mask_embedding" in name for name in parameter_names)
    assert not any("mask_token" in name for name in parameter_names)


def test_model_zeros_slots_for_rows_without_a_full_emitted_block() -> None:
    """Catches invalid-row output leakage into the later live target head."""
    plan_module, model_type = _load_model_contract()
    plan = _build_plan(
        plan_module,
        torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, False, False, False, False],
            ]
        ),
        torch.tensor([101, 202], dtype=torch.int64),
    )
    torch.manual_seed(456)
    model = _new_model(model_type)
    target_hidden_taps = torch.randn((2, 2, 6, 6))
    input_embeddings = torch.randn((4, 3, 6))

    output = model(
        plan=plan,
        target_hidden_taps=target_hidden_taps,
        input_embeddings=input_embeddings,
    )

    assert plan.block_valid[:2].all()
    assert not plan.block_valid[2:].any()
    assert torch.equal(output[2:], torch.zeros_like(output[2:]))
    assert torch.isfinite(output).all()


def test_state_dict_save_load_round_trip_is_exact() -> None:
    """Catches missing, renamed, or non-restored internal DFlash state."""
    plan_module, model_type = _load_model_contract()
    plan = _build_plan(
        plan_module,
        torch.ones((2, 6), dtype=torch.bool),
        torch.tensor([101, 202], dtype=torch.int64),
    )
    torch.manual_seed(789)
    source_model = _new_model(model_type).eval()
    target_hidden_taps = torch.randn((2, 2, 6, 6))
    input_embeddings = torch.randn((4, 3, 6))
    expected_output = source_model(
        plan=plan,
        target_hidden_taps=target_hidden_taps,
        input_embeddings=input_embeddings,
    )
    checkpoint = io.BytesIO()
    torch.save(source_model.state_dict(), checkpoint)

    torch.manual_seed(999)
    restored_model = _new_model(model_type).eval()
    checkpoint.seek(0)
    restored_model.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored_output = restored_model(
        plan=plan,
        target_hidden_taps=target_hidden_taps,
        input_embeddings=input_embeddings,
    )

    source_state = source_model.state_dict()
    restored_state = restored_model.state_dict()
    assert source_state.keys() == restored_state.keys()
    for name in source_state:
        assert torch.equal(source_state[name], restored_state[name]), name
    assert torch.equal(expected_output, restored_output)
