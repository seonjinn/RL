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


class _FakeInferenceContext:
    def __init__(self, max_batch_size: int, max_sequence_length: int) -> None:
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.sequence_len_offset = 0
        self.key_value_memory_dict: dict[object, object] = {}


class _FakeEagleModule:
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


@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_provider_runs_bounded_passes_with_public_call_shape(pass_count: int) -> None:
    module = _load_provider()
    mask_steps: list[int] = []
    rope_offsets: list[int] = []

    def mask_helper(mask: torch.Tensor, step: int) -> torch.Tensor:
        mask_steps.append(step)
        return mask

    def rotary_provider(sequence_length: int, offset: int = 0) -> torch.Tensor:
        rope_offsets.append(offset)
        return torch.arange(offset, offset + sequence_length)

    eagle_module = _FakeEagleModule()
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
        token_chunk_size=4,
        inference_context_factory=_FakeInferenceContext,
        multi_step_mask_helper=mask_helper,
    )
    sequence, batch, hidden = 12, 2, 4
    target_trunk = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeddings = torch.randn_like(target_trunk, requires_grad=True)
    output = provider.forward(
        eagle_module=eagle_module,
        project_logits=_project,
        target_trunk_states=target_trunk,
        input_embeds=embeddings,
        attention_mask=torch.zeros(batch, 1, sequence, sequence, dtype=torch.bool),
        pass_count=pass_count,
        rotary_provider=rotary_provider,
    )

    assert len(output.pass_logits) == pass_count
    assert len(output.branch_states) == pass_count
    assert [plan.pass_index for plan in output.plans] == list(range(pass_count))
    assert mask_steps == list(range(pass_count))
    assert rope_offsets == list(range(pass_count))
    assert len(eagle_module.calls) == pass_count
    context = eagle_module.calls[0]["inference_context"]
    assert isinstance(context, _FakeInferenceContext)
    assert context.max_batch_size == batch
    assert context.max_sequence_length == sequence * pass_count
    assert context.sequence_len_offset == sequence * pass_count


def test_one_pass_matches_direct_eagle_output_loss_and_gradient() -> None:
    module = _load_provider()
    sequence, batch, hidden = 6, 2, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
        token_chunk_size=4,
        inference_context_factory=_FakeInferenceContext,
        multi_step_mask_helper=lambda mask, _step: mask,
    )
    output = provider.forward(
        eagle_module=_FakeEagleModule(),
        project_logits=_project,
        target_trunk_states=target,
        input_embeds=embeds,
        attention_mask=torch.zeros(batch, 1, sequence, sequence, dtype=torch.bool),
        pass_count=1,
        rotary_provider=lambda length, offset=0: torch.arange(offset, offset + length),
    )
    stats = provider.loss_stats(
        output=output,
        teacher_logits=teacher,
        valid_mask=valid,
        pass_weights=torch.tensor([1.0]),
        tp_group=None,
    )

    direct_logits = _project(target + embeds).transpose(0, 1)
    direct_stats = module.streaming_vocab_parallel_soft_ce(
        student_logits=direct_logits[:, :-1],
        teacher_logits=teacher[:, 1:],
        mask=valid[:, 1:],
        token_chunk_size=4,
        tp_group=None,
    )
    torch.testing.assert_close(output.pass_logits[0], direct_logits)
    torch.testing.assert_close(stats.numerators, direct_stats.numerators)
    torch.testing.assert_close(stats.counts, direct_stats.counts)
    actual_gradients = torch.autograd.grad(stats.numerators.sum(), (target, embeds))
    expected_gradients = torch.autograd.grad(
        direct_stats.numerators.sum(), (target, embeds)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_loss_reaches_target_trunk_and_every_prior_pass() -> None:
    module = _load_provider()
    sequence, batch, hidden, pass_count = 10, 1, 4, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
        token_chunk_size=8,
        inference_context_factory=_FakeInferenceContext,
        multi_step_mask_helper=lambda mask, _step: mask,
    )
    output = provider.forward(
        eagle_module=_FakeEagleModule(),
        project_logits=_project,
        target_trunk_states=target,
        input_embeds=embeds,
        attention_mask=torch.zeros(batch, 1, sequence, sequence, dtype=torch.bool),
        pass_count=pass_count,
        rotary_provider=lambda length, offset=0: torch.arange(offset, offset + length),
    )
    teacher = torch.randn(batch, sequence, 7)
    stats = provider.loss_stats(
        output=output,
        teacher_logits=teacher,
        valid_mask=torch.ones(batch, sequence, dtype=torch.bool),
        pass_weights=torch.tensor([1.0, 0.75, 0.5, 0.25]),
        tp_group=None,
    )
    gradients = torch.autograd.grad(
        stats.normalized(normalization_counts=stats.counts),
        (target, *output.branch_states[:-1]),
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.count_nonzero(gradient) > 0
        for gradient in gradients
        if gradient is not None
    )


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
        token_chunk_size=5,
        inference_context_factory=_FakeInferenceContext,
        multi_step_mask_helper=lambda mask, _step: mask,
    )

    def run(start: int, end: int):
        output = provider.forward(
            eagle_module=_FakeEagleModule(),
            project_logits=_project,
            target_trunk_states=target[:, start:end],
            input_embeds=embeds[:, start:end],
            attention_mask=torch.zeros(
                end - start, 1, sequence, sequence, dtype=torch.bool
            ),
            pass_count=pass_count,
            rotary_provider=lambda length, offset=0: torch.arange(
                offset, offset + length
            ),
        )
        return provider.loss_stats(
            output=output,
            teacher_logits=teacher[start:end],
            valid_mask=valid[start:end],
            pass_weights=weights,
            tp_group=None,
        )

    full = run(0, batch)
    left = run(0, 1)
    right = run(1, batch)
    torch.testing.assert_close(full.numerators, left.numerators + right.numerators)
    torch.testing.assert_close(full.counts, left.counts + right.counts)
    torch.testing.assert_close(full.weights, weights)


def test_32k_provider_smoke_is_executable_without_retained_square_state() -> None:
    module = _load_provider()
    sequence = 32_768
    provider = module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=1 << 30,
        token_chunk_size=128,
        inference_context_factory=_FakeInferenceContext,
        multi_step_mask_helper=lambda mask, _step: mask,
    )
    target = torch.zeros(sequence, 1, 1)
    output = provider.forward(
        eagle_module=_FakeEagleModule(),
        project_logits=lambda hidden: hidden,
        target_trunk_states=target,
        input_embeds=torch.ones_like(target),
        attention_mask=None,
        pass_count=4,
        rotary_provider=lambda length, offset=0: torch.arange(offset, offset + length),
    )
    assert [logits.shape for logits in output.pass_logits] == [
        torch.Size((1, sequence, 1))
    ] * 4
    assert all(state.numel() == sequence for state in output.branch_states)


@pytest.mark.mcore
def test_pinned_modelopt_and_megatron_public_signatures() -> None:
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
    assert list(inspect.signature(contexts.StaticInferenceContext).parameters) == [
        "max_batch_size",
        "max_sequence_length",
        "use_flashinfer_fused_rope",
    ]
