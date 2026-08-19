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
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_provider():
    loss_package_name = "nemo_rl.algorithms.loss"
    loss_package = ModuleType(loss_package_name)
    loss_package.__path__ = [str(Path(__file__).parents[4] / "nemo_rl/algorithms/loss")]
    sys.modules[loss_package_name] = loss_package
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.eagle_provider")


class _FakeStructuredPassRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden_states = kwargs["hidden_states"]
        embeddings = kwargs["embeddings"]
        output = hidden_states + embeddings
        return output, output * 0.5


def _project(hidden_states: torch.Tensor) -> torch.Tensor:
    weight = torch.arange(
        hidden_states.shape[-1] * 7,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).reshape(hidden_states.shape[-1], 7)
    return hidden_states @ weight


def _soft_ce_pass_loss(
    module,
    *,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    token_chunk_size: int,
):
    def pass_loss(*, hidden_states: torch.Tensor, plan):
        offset = plan.teacher_offset
        logits = _project(hidden_states).transpose(0, 1)
        return module.streaming_vocab_parallel_soft_ce(
            student_logits=logits[:, :-offset],
            teacher_logits=teacher[:, offset:],
            mask=valid[:, offset:],
            token_chunk_size=token_chunk_size,
            tp_group=None,
        )

    return pass_loss


@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_provider_runs_structured_passes_without_dense_masks(pass_count: int) -> None:
    module = _load_provider()
    runner = _FakeStructuredPassRunner()
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    sequence, batch, hidden = 12, 2, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    output = provider.forward_loss_stats(
        pass_runner=runner,
        pass_loss=_soft_ce_pass_loss(
            module,
            teacher=teacher,
            valid=valid,
            token_chunk_size=4,
        ),
        target_trunk_states=target,
        input_embeds=embeds,
        pass_count=pass_count,
        pass_weights=torch.ones(pass_count),
    )

    assert output.stats.numerators.shape == (pass_count,)
    assert output.stats.counts.shape == (pass_count,)
    assert output.stats.weights.shape == (pass_count,)
    assert [plan.pass_index for plan in output.plans] == list(range(pass_count))
    assert len(runner.calls) == pass_count
    assert not hasattr(output, "pass_logits")
    for index, call in enumerate(runner.calls):
        plan = call["plan"]
        assert plan.pass_index == index
        rope_positions = call["rope_positions"]
        torch.testing.assert_close(rope_positions, torch.arange(sequence))
        assert "attention_mask" not in call


@pytest.mark.parametrize("pass_count", [2, 4])
def test_modelopt_static_rope_table_covers_every_mcore_cache_slice(
    pass_count: int,
) -> None:
    module = _load_provider()
    sequence = 11
    base_rotary = torch.arange(sequence * 3).reshape(sequence, 1, 1, 3)
    for pass_index in range(pass_count):
        plan = module.EagleTTTAttentionPlan(
            pass_index=pass_index,
            pass_count=pass_count,
            max_passes=8,
            sequence_length=sequence,
        )
        rotary = module.modelopt_static_rotary_table(
            base_rotary_pos_emb=base_rotary,
            plan=plan,
        )
        sequence_start = pass_index * sequence
        sequence_end = sequence_start + sequence
        torch.testing.assert_close(rotary[sequence_start:sequence_end], base_rotary)
        assert rotary.shape[0] == sequence_end


def test_one_pass_matches_direct_loss_and_gradient() -> None:
    module = _load_provider()
    sequence, batch, hidden = 6, 2, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    output = provider.forward_loss_stats(
        pass_runner=_FakeStructuredPassRunner(),
        pass_loss=_soft_ce_pass_loss(
            module,
            teacher=teacher,
            valid=valid,
            token_chunk_size=4,
        ),
        target_trunk_states=target,
        input_embeds=embeds,
        pass_count=1,
        pass_weights=torch.tensor([1.0]),
    )

    direct_logits = _project(target + embeds).transpose(0, 1)
    direct_stats = module.streaming_vocab_parallel_soft_ce(
        student_logits=direct_logits[:, :-1],
        teacher_logits=teacher[:, 1:],
        mask=valid[:, 1:],
        token_chunk_size=4,
        tp_group=None,
    )
    torch.testing.assert_close(output.stats.numerators, direct_stats.numerators)
    torch.testing.assert_close(output.stats.counts, direct_stats.counts)
    actual_gradients = torch.autograd.grad(output.stats.numerators.sum(), (target, embeds))
    expected_gradients = torch.autograd.grad(
        direct_stats.numerators.sum(), (target, embeds)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_cross_pass_state_is_explicitly_stop_gradient() -> None:
    module = _load_provider()
    sequence, batch, hidden, pass_count = 10, 1, 4, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    runner = _FakeStructuredPassRunner()
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    output = provider.forward_loss_stats(
        pass_runner=runner,
        pass_loss=_soft_ce_pass_loss(
            module,
            teacher=teacher,
            valid=valid,
            token_chunk_size=8,
        ),
        target_trunk_states=target,
        input_embeds=embeds,
        pass_count=pass_count,
        pass_weights=torch.ones(pass_count),
    )

    assert runner.calls[0]["hidden_states"] is target
    assert all(
        not call["hidden_states"].requires_grad for call in runner.calls[1:]
    )
    assert not output.final_branch_state.requires_grad
    target_gradient = torch.autograd.grad(
        output.stats.numerators.sum(), target, retain_graph=True
    )[0]
    first_pass = _soft_ce_pass_loss(
        module,
        teacher=teacher,
        valid=valid,
        token_chunk_size=8,
    )(
        hidden_states=target + embeds,
        plan=output.plans[0],
    )
    expected_gradient = torch.autograd.grad(first_pass.numerators.sum(), target)[0]
    torch.testing.assert_close(target_gradient, expected_gradient)


def test_full_and_split_additive_stats_match_with_unequal_weights() -> None:
    module = _load_provider()
    sequence, batch, hidden, pass_count = 9, 4, 4, 4
    target = torch.randn(sequence, batch, hidden)
    embeds = torch.randn_like(target)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    weights = torch.tensor([1.0, 0.6, 0.3, 0.1])
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )

    def run(start: int, end: int):
        return provider.forward_loss_stats(
            pass_runner=_FakeStructuredPassRunner(),
            pass_loss=_soft_ce_pass_loss(
                module,
                teacher=teacher[start:end],
                valid=valid[start:end],
                token_chunk_size=5,
            ),
            target_trunk_states=target[:, start:end],
            input_embeds=embeds[:, start:end],
            pass_count=pass_count,
            pass_weights=weights,
        ).stats

    full = run(0, batch)
    left = run(0, 1)
    right = run(1, batch)
    torch.testing.assert_close(full.numerators, left.numerators + right.numerators)
    torch.testing.assert_close(full.counts, left.counts + right.counts)
    torch.testing.assert_close(full.weights, weights)


@pytest.mark.parametrize("sequence", [32_768, 262_144])
def test_long_context_provider_retains_no_square_state_or_vocab_logits(
    sequence: int,
) -> None:
    module = _load_provider()
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    target = torch.zeros(sequence, 1, 1)
    seen_shapes: list[torch.Size] = []

    def pass_loss(*, hidden_states: torch.Tensor, plan):
        seen_shapes.append(hidden_states.shape)
        return module.DraftLossStats(
            numerators=hidden_states[:1].reshape(1).float() + plan.pass_index,
            counts=torch.ones(1),
            weights=torch.ones(1),
        )

    output = provider.forward_loss_stats(
        pass_runner=_FakeStructuredPassRunner(),
        pass_loss=pass_loss,
        target_trunk_states=target,
        input_embeds=torch.ones_like(target),
        pass_count=4,
        pass_weights=torch.ones(4),
    )
    assert output.stats.numerators.shape == (4,)
    assert output.final_branch_state.numel() == sequence
    assert seen_shapes == [torch.Size((sequence, 1, 1))] * 4
    assert not any(len(shape) >= 2 and shape[-2:] == (sequence, sequence) for shape in seen_shapes)


@pytest.mark.mcore
def test_pinned_modelopt_and_megatron_public_contracts() -> None:
    megatron_eagle = pytest.importorskip(
        "modelopt.torch.speculative.plugins.megatron_eagle"
    )
    contexts = pytest.importorskip("megatron.core.inference.contexts")
    assert list(inspect.signature(megatron_eagle.EagleModule.forward).parameters) == [
        "self",
        "embeddings",
        "hidden_states",
        "attention_mask",
        "rotary_pos_emb",
        "inference_params",
        "packed_seq_params",
        "inference_context",
        "extra_block_kwargs",
    ]
    assert list(
        inspect.signature(megatron_eagle.set_multi_step_attention_mask).parameters
    ) == ["attn_mask", "step"]
    assert "detach()" in inspect.getsource(
        megatron_eagle.EagleModule._eagle3_layer_forward_hook
    )
    assert list(inspect.signature(contexts.StaticInferenceContext).parameters) == [
        "max_batch_size",
        "max_sequence_length",
        "use_flashinfer_fused_rope",
    ]
