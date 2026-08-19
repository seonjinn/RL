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
    adapt_markov_weight_to_runtime,
    bind_live_target_io,
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


def test_adapt_markov_weight_keeps_replicated_runtime_tensor() -> None:
    weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    adapted = adapt_markov_weight_to_runtime(
        name="model.markov_head.markov_w2.weight",
        weight=weight,
        target_shape=(8, 2),
        tp_rank=1,
        tp_size=2,
    )

    assert adapted is weight


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_adapt_markov_weight_selects_vllm_025_vocab_shard(tp_rank: int) -> None:
    weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    adapted = adapt_markov_weight_to_runtime(
        name="model.markov_head.markov_w2.weight",
        weight=weight,
        target_shape=(4, 2),
        tp_rank=tp_rank,
        tp_size=2,
    )

    torch.testing.assert_close(adapted, weight[tp_rank * 4 : (tp_rank + 1) * 4])
    assert adapted.is_contiguous()


def test_adapt_markov_weight_rejects_local_to_replicated_without_collective() -> None:
    local_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    with pytest.raises(ValueError, match="component-aware gather"):
        adapt_markov_weight_to_runtime(
            name="model.markov_head.markov_w2.weight",
            weight=local_weight,
            target_shape=(8, 2),
            tp_rank=0,
            tp_size=2,
        )


@pytest.mark.parametrize(
    "target_shape,tp_rank,tp_size",
    [((5, 2), 0, 2), ((4, 3), 0, 2), ((4, 2), 2, 2), ((4, 2), 0, 0)],
)
def test_adapt_markov_weight_rejects_ambiguous_layouts(
    target_shape: tuple[int, int], tp_rank: int, tp_size: int
) -> None:
    weight = torch.zeros(8, 2)

    with pytest.raises(ValueError):
        adapt_markov_weight_to_runtime(
            name="model.markov_head.markov_w1.weight",
            weight=weight,
            target_shape=target_shape,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
