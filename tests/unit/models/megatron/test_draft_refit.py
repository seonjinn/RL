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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_rl.models.megatron.draft import utils as draft_utils
from nemo_rl.models.megatron.draft.utils import (
    prepare_draft_weight_for_refit,
    prepare_draft_weights_for_refit,
)


class _DraftModel(nn.Module):
    def __init__(
        self,
        *,
        target_vocab_size: int = 12,
        draft_vocab_size: int = 8,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )


@pytest.mark.mcore
def test_markov_w2_reconstructs_tp_shards_exactly_once(monkeypatch) -> None:
    draft_model = _DraftModel()
    rank0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    rank1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    calls = []

    def gather(local_weight):
        calls.append(local_weight)
        return [rank0, rank1]

    monkeypatch.setattr(draft_utils, "_all_gather_tp_shards", gather)

    result = prepare_draft_weight_for_refit(
        draft_model=draft_model,
        name="model.markov_head.markov_w2.weight",
        weight=rank0,
    )

    assert calls == [rank0]
    torch.testing.assert_close(result, torch.cat([rank0, rank1], dim=0))


@pytest.mark.mcore
@pytest.mark.parametrize(
    "name",
    [
        "model.markov_head.markov_w1.weight",
        "model.markov_head.markov_w2.weight",
        "model.layers.0.mlp.down_proj.weight",
    ],
)
def test_global_or_non_markov_draft_weight_skips_collective(monkeypatch, name) -> None:
    draft_model = _DraftModel()
    global_vocab_size = (
        draft_model.config.vocab_size
        if name.endswith("markov_w1.weight")
        else draft_model.config.draft_vocab_size
    )
    weight = torch.arange(
        global_vocab_size * 2,
        dtype=torch.float32,
    ).reshape(global_vocab_size, 2)

    def unexpected_gather(_local_weight):
        raise AssertionError("full-global and non-Markov weights must not gather")

    monkeypatch.setattr(draft_utils, "_all_gather_tp_shards", unexpected_gather)

    result = prepare_draft_weight_for_refit(
        draft_model=draft_model,
        name=name,
        weight=weight,
    )

    assert result is weight


@pytest.mark.mcore
def test_markov_w1_rejects_tp_local_shape_without_collective(monkeypatch) -> None:
    draft_model = _DraftModel()
    local_weight = torch.zeros(draft_model.config.vocab_size // 2, 2)

    def unexpected_gather(_local_weight):
        raise AssertionError("replicated markov_w1 must not gather")

    monkeypatch.setattr(draft_utils, "_all_gather_tp_shards", unexpected_gather)

    with pytest.raises(ValueError, match="markov_w1.*global vocab dimension"):
        prepare_draft_weight_for_refit(
            draft_model=draft_model,
            name="model.markov_head.markov_w1.weight",
            weight=local_weight,
        )


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("weight", "error"),
    [
        (torch.zeros(8), "rank-2"),
        (torch.zeros(3, 2), "cannot form global vocab dimension"),
    ],
)
def test_markov_refit_rejects_invalid_local_shape(weight, error) -> None:
    with pytest.raises(ValueError, match=error):
        prepare_draft_weight_for_refit(
            draft_model=_DraftModel(),
            name="model.markov_head.markov_w2.weight",
            weight=weight,
        )


@pytest.mark.mcore
def test_markov_components_use_distinct_target_and_draft_vocabularies(
    monkeypatch,
) -> None:
    draft_model = _DraftModel(target_vocab_size=12, draft_vocab_size=8)
    full_w1 = torch.zeros(12, 2)
    full_w2 = torch.zeros(8, 2)

    def unexpected_gather(_local_weight):
        raise AssertionError("global Markov weights must not gather")

    monkeypatch.setattr(draft_utils, "_all_gather_tp_shards", unexpected_gather)

    assert (
        prepare_draft_weight_for_refit(
            draft_model=draft_model,
            name="model.markov_head.markov_w1.weight",
            weight=full_w1,
        )
        is full_w1
    )
    assert (
        prepare_draft_weight_for_refit(
            draft_model=draft_model,
            name="model.markov_head.markov_w2.weight",
            weight=full_w2,
        )
        is full_w2
    )


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("name", "weight"),
    [
        ("model.confidence_head.proj.weight", torch.zeros(1, 10)),
        ("model.confidence_head.proj.bias", torch.zeros(1)),
    ],
)
def test_confidence_components_are_explicitly_replicated(
    monkeypatch,
    name,
    weight,
) -> None:
    def unexpected_gather(_local_weight):
        raise AssertionError("confidence weights must not gather")

    monkeypatch.setattr(draft_utils, "_all_gather_tp_shards", unexpected_gather)

    result = prepare_draft_weight_for_refit(
        draft_model=_DraftModel(),
        name=name,
        weight=weight,
    )

    assert result is weight


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("name", "weight", "error"),
    [
        ("model.confidence_head.proj.weight", torch.zeros(2, 10), "shape"),
        ("model.confidence_head.proj.bias", torch.zeros(2), "shape"),
    ],
)
def test_confidence_components_reject_non_replicated_shapes(
    name, weight, error
) -> None:
    with pytest.raises(ValueError, match=error):
        prepare_draft_weight_for_refit(
            draft_model=_DraftModel(),
            name=name,
            weight=weight,
        )


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("weights", "error"),
    [
        (
            [
                ("model.markov_head.markov_w1.weight", torch.zeros(12, 2)),
            ],
            "missing.*markov_w2",
        ),
        (
            [
                ("model.confidence_head.proj.weight", torch.zeros(1, 10)),
            ],
            "missing.*confidence.*bias",
        ),
        (
            [
                ("model.layers.0.norm.weight", torch.zeros(10)),
                ("model.layers.0.norm.weight", torch.zeros(10)),
            ],
            "duplicate draft refit weight",
        ),
    ],
)
def test_draft_refit_collection_rejects_missing_or_duplicate_components(
    weights,
    error,
) -> None:
    with pytest.raises(ValueError, match=error):
        prepare_draft_weights_for_refit(
            draft_model=_DraftModel(),
            weights=weights,
        )


@pytest.mark.mcore
def test_markov_w2_rejects_incomplete_tp_reconstruction(monkeypatch) -> None:
    local_weight = torch.zeros(4, 2)
    monkeypatch.setattr(
        draft_utils,
        "_all_gather_tp_shards",
        lambda _local_weight: [local_weight],
    )

    with pytest.raises(ValueError, match="reconstructed shape"):
        prepare_draft_weight_for_refit(
            draft_model=_DraftModel(),
            name="model.markov_head.markov_w2.weight",
            weight=local_weight,
        )


@pytest.mark.mcore
def test_policy_export_prepares_markov_weight_before_refit_manifest(
    monkeypatch,
) -> None:
    from nemo_rl.models.megatron import draft as draft_module
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    class _Bridge:
        def export_hf_weights(self, *_args, **_kwargs):
            return iter(())

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.cfg = {}
    worker.model = object()
    worker.megatron_bridge = _Bridge()
    worker.refit_conversion_tasks = []
    worker.draft_model = _DraftModel()
    rank0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    rank1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    calls = []

    monkeypatch.setattr(
        draft_module,
        "export_eagle_weights_to_hf",
        lambda _model: [
            ("model.markov_head.markov_w1.weight", torch.zeros(12, 2)),
            ("model.markov_head.markov_w2.weight", rank0),
        ],
    )

    def gather(local_weight):
        calls.append(local_weight)
        return [rank0, rank1]

    monkeypatch.setattr(draft_utils, "_all_gather_tp_shards", gather)

    output = list(worker._iter_params_with_optional_kv_scales())

    assert calls == [rank0]
    assert output[1][0] == "draft.model.markov_head.markov_w2.weight"
    torch.testing.assert_close(output[1][1], torch.cat([rank0, rank1], dim=0))
