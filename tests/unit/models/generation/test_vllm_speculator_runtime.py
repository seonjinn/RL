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
from torch import nn

from nemo_rl.models.generation.vllm.speculator_runtime import (
    MarkovRuntimeLayout,
    bind_live_target_io,
    prepare_markov_loader_weight,
)


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)


class _LanguageModel(nn.Module):
    def __init__(self, *, owns_head: bool) -> None:
        super().__init__()
        self.model = _InnerModel()
        if owns_head:
            self.lm_head = nn.Linear(4, 8, bias=False)


class _TargetModel(nn.Module):
    def __init__(self, *, conditional: bool) -> None:
        super().__init__()
        self.language_model = _LanguageModel(owns_head=conditional)
        if not conditional:
            self.model = self.language_model.model
            self.lm_head = nn.Linear(4, 8, bias=False)

    def get_language_model(self) -> nn.Module:
        return self.language_model


class _DraftModel(nn.Module):
    def __init__(self, *, owns_io: bool = False) -> None:
        super().__init__()
        self.model = _InnerModel()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.has_own_embed_tokens = owns_io
        self.has_own_lm_head = owns_io


@pytest.mark.parametrize("conditional", [False, True])
def test_bind_live_target_io_supports_vllm_025_and_027_target_layouts(
    conditional: bool,
) -> None:
    target = _TargetModel(conditional=conditional)
    draft = _DraftModel()
    target_language_model = target.get_language_model()
    expected_head = target_language_model.lm_head if conditional else target.lm_head

    bind_live_target_io(draft_model=draft, target_model=target, share_embedding=True)

    assert draft.model.embed_tokens is target_language_model.model.embed_tokens
    assert draft.lm_head is expected_head


def test_bind_live_target_io_can_skip_embedding_for_pipeline_parallelism() -> None:
    target = _TargetModel(conditional=False)
    draft = _DraftModel()
    original_embedding = draft.model.embed_tokens

    bind_live_target_io(draft_model=draft, target_model=target, share_embedding=False)

    assert draft.model.embed_tokens is original_embedding
    assert draft.lm_head is target.lm_head


@pytest.mark.parametrize("flag", ["has_own_embed_tokens", "has_own_lm_head"])
def test_bind_live_target_io_rejects_stale_draft_owned_io(flag: str) -> None:
    target = _TargetModel(conditional=False)
    draft = _DraftModel()
    setattr(draft, flag, True)

    with pytest.raises(ValueError, match="body-only"):
        bind_live_target_io(
            draft_model=draft, target_model=target, share_embedding=True
        )


def test_prepare_markov_weight_keeps_full_tensor_for_replicated_runtime() -> None:
    weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    prepared = prepare_markov_loader_weight(
        name="model.markov_head.markov_w2.weight",
        weight=weight,
        target_shape=(8, 2),
        global_vocab_size=8,
        tp_size=2,
    )

    assert prepared.tensor is weight
    assert prepared.runtime_layout is MarkovRuntimeLayout.REPLICATED


def test_prepare_markov_weight_keeps_full_tensor_for_sharded_runtime_loader() -> None:
    weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    prepared = prepare_markov_loader_weight(
        name="model.markov_head.markov_w2.weight",
        weight=weight,
        target_shape=(4, 2),
        global_vocab_size=8,
        tp_size=2,
    )

    assert prepared.tensor is weight
    assert prepared.runtime_layout is MarkovRuntimeLayout.VOCAB_SHARDED


@pytest.mark.parametrize("target_shape", [(4, 2), (8, 2)])
def test_prepare_markov_weight_rejects_local_transport_without_collective(
    target_shape: tuple[int, int],
) -> None:
    local_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    with pytest.raises(ValueError, match="component-aware gather"):
        prepare_markov_loader_weight(
            name="model.markov_head.markov_w2.weight",
            weight=local_weight,
            target_shape=target_shape,
            global_vocab_size=8,
            tp_size=2,
        )


@pytest.mark.parametrize(
    "target_shape,global_vocab_size,tp_size",
    [((5, 2), 8, 2), ((4, 3), 8, 2), ((4, 2), 7, 2), ((4, 2), 8, 0)],
)
def test_prepare_markov_weight_rejects_ambiguous_layouts(
    target_shape: tuple[int, int], global_vocab_size: int, tp_size: int
) -> None:
    weight = torch.zeros(8, 2)

    with pytest.raises(ValueError):
        prepare_markov_loader_weight(
            name="model.markov_head.markov_w1.weight",
            weight=weight,
            target_shape=target_shape,
            global_vocab_size=global_vocab_size,
            tp_size=tp_size,
        )
