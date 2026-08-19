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
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_module():
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.eagle_ttt")


def _storage(module, *, pass_count: int = 2):
    return module.EagleTTTStoragePlan(
        batch_size=1,
        kv_heads=2,
        sequence_length=6,
        head_dim=4,
        dtype=torch.float32,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 20,
        layer_count=1,
        hidden_size=8,
        rope_dim=4,
    )


def _layout(module):
    return module.EagleTTTSequenceLayout.from_cu_seqlens(
        cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32),
        sequence_length=6,
    )


class _FakeEagleModule(torch.nn.Module):
    def __init__(self, core_attention: torch.nn.Module) -> None:
        super().__init__()
        self.core_attention = core_attention
        self.weight = torch.nn.Parameter(torch.randn(8, 8))

    def forward(
        self,
        *,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rotary_pos_emb: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del embeddings, rotary_pos_emb
        sequence, batch, hidden = hidden_states.shape
        query = (hidden_states * 1).reshape(sequence, batch, 2, hidden // 2)
        output = self.core_attention(
            query,
            query,
            query,
            attention_mask,
            attn_mask_type=None,
            attention_bias=None,
            packed_seq_params=packed_seq_params,
        )
        return output, output


def _core(module, *, context_parallel_size: int = 1):
    config = SimpleNamespace(context_parallel_size=context_parallel_size)
    return module.EagleTTTCoreAttention(
        config=config,
        layer_number=1,
        attn_mask_type=None,
        attention_type="self",
        cp_comm_type=None,
        softmax_scale=None,
        pg_collection=None,
    )


def test_layer_spec_adapter_is_construction_time_and_does_not_mutate_default() -> None:
    module = _load_module()
    original_core = object()
    original = SimpleNamespace(
        submodules=SimpleNamespace(
            self_attention=SimpleNamespace(
                submodules=SimpleNamespace(core_attention=original_core)
            )
        )
    )

    adapted = module.with_eagle_ttt_core_attention(original)

    assert adapted is not original
    assert (
        adapted.submodules.self_attention.submodules.core_attention
        is module.EagleTTTCoreAttention
    )
    assert original.submodules.self_attention.submodules.core_attention is original_core


def test_concrete_session_captures_post_rope_trunk_and_branch_in_pass_order() -> None:
    module = _load_module()
    core = _core(module)
    model = _FakeEagleModule(core)
    session = module.MCoreEagleTTTSession(model)
    layout = _layout(module)
    storage = _storage(module)
    ledger = module.EagleTTTResourceLedger(limit_bytes=1 << 20)
    hidden = torch.randn(6, 1, 8, requires_grad=True)
    embeddings = torch.randn_like(hidden)
    session.begin(
        layout=layout,
        storage_plan=storage,
        excluded_tensors=(hidden, embeddings),
        resource_ledger=ledger,
    )

    first_plan = module.EagleTTTAttentionPlan(
        pass_index=0,
        pass_count=2,
        max_passes=8,
        sequence_length=6,
    )
    first, _ = session(
        embeddings=embeddings,
        hidden_states=hidden,
        plan=first_plan,
        rope_positions=torch.arange(6),
    )
    assert first.shape == hidden.shape
    assert core.state is not None
    assert len(core.state.branch_keys) == 0
    torch.testing.assert_close(
        core.state.trunk_key,
        hidden.reshape(6, 1, 2, 4).permute(1, 2, 0, 3),
    )

    second_plan = module.EagleTTTAttentionPlan(
        pass_index=1,
        pass_count=2,
        max_passes=8,
        sequence_length=6,
    )
    second, _ = session(
        embeddings=embeddings,
        hidden_states=first.detach(),
        plan=second_plan,
        rope_positions=torch.arange(6),
    )
    assert second.shape == hidden.shape
    assert core.state is not None
    assert len(core.state.branch_keys) == 1
    assert ledger.owned_bytes > 0

    session.reset()
    assert core.state is None
    assert core.plan is None
    session.reset()


def test_session_fails_closed_before_mutation_for_cp_and_pass_order() -> None:
    module = _load_module()
    layout = _layout(module)
    storage = _storage(module)
    ledger = module.EagleTTTResourceLedger(limit_bytes=1 << 20)
    cp_core = _core(module, context_parallel_size=2)
    cp_session = module.MCoreEagleTTTSession(_FakeEagleModule(cp_core))

    with pytest.raises(ValueError, match="context parallel"):
        cp_session.begin(
            layout=layout,
            storage_plan=storage,
            excluded_tensors=(),
            resource_ledger=ledger,
        )
    assert cp_core.state is None
    assert cp_core.plan is None

    core = _core(module)
    session = module.MCoreEagleTTTSession(_FakeEagleModule(core))
    session.begin(
        layout=layout,
        storage_plan=storage,
        excluded_tensors=(),
        resource_ledger=ledger,
    )
    skipped_plan = module.EagleTTTAttentionPlan(
        pass_index=1,
        pass_count=2,
        max_passes=8,
        sequence_length=6,
    )
    with pytest.raises(ValueError, match="expected pass 0"):
        session(
            embeddings=torch.zeros(6, 1, 8),
            hidden_states=torch.zeros(6, 1, 8),
            plan=skipped_plan,
            rope_positions=torch.arange(6),
        )
    assert core.state is None
    session.reset()


def test_core_attention_rejects_unrepresented_dense_mask_and_clears_state() -> None:
    module = _load_module()
    core = _core(module)
    session = module.MCoreEagleTTTSession(_FakeEagleModule(core))
    session.begin(
        layout=_layout(module),
        storage_plan=_storage(module),
        excluded_tensors=(),
        resource_ledger=module.EagleTTTResourceLedger(limit_bytes=1 << 20),
    )
    core.begin_pass(
        module.EagleTTTAttentionPlan(
            pass_index=0,
            pass_count=2,
            max_passes=8,
            sequence_length=6,
        )
    )
    query = torch.randn(6, 1, 2, 4)

    with pytest.raises(ValueError, match="dense attention_mask"):
        core(
            query,
            query,
            query,
            torch.zeros(1, 1, 6, 6, dtype=torch.bool),
            attn_mask_type=None,
            attention_bias=None,
            packed_seq_params=None,
        )

    session.reset()
    assert core.state is None
    assert core.plan is None


def test_session_excludes_preexisting_model_parameters_from_resource_budget() -> None:
    module = _load_module()

    class RecordingLedger(module.EagleTTTResourceLedger):
        def __init__(self) -> None:
            super().__init__(limit_bytes=1 << 20)
            self.excluded: list[torch.Tensor] = []

        def exclude(self, tensors: tuple[torch.Tensor, ...]) -> None:
            self.excluded.extend(tensors)
            super().exclude(tensors)

    model = _FakeEagleModule(_core(module))
    ledger = RecordingLedger()
    session = module.MCoreEagleTTTSession(model)
    session.begin(
        layout=_layout(module),
        storage_plan=_storage(module),
        excluded_tensors=(),
        resource_ledger=ledger,
    )

    assert any(tensor is model.weight for tensor in ledger.excluded)
    session.reset()
