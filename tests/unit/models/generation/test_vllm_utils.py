# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.utils import (
    R3_MISSING_ROUTE_SENTINEL,
    aggregate_spec_decode_counters,
    attach_routed_experts_to_chat_response_choices,
    compute_spec_decode_metrics,
    extract_generated_token_logprobs,
    format_prompt_for_vllm_generation,
    model_dump_chat_response_with_routed_experts,
    pad_and_align_routed_expert_indices,
)


def _logprob(value: float) -> SimpleNamespace:
    return SimpleNamespace(logprob=value)


def test_extract_generated_token_logprobs_returns_chosen_token_values():
    values = extract_generated_token_logprobs(
        [11, 12],
        [{11: _logprob(-0.25)}, {12: _logprob(-0.5)}],
    )

    assert values == [-0.25, -0.5]


def test_extract_generated_token_logprobs_allows_empty_generation():
    assert extract_generated_token_logprobs([], None) == []


@pytest.mark.parametrize("token_logprobs", [None, []])
def test_extract_generated_token_logprobs_rejects_missing_values(token_logprobs):
    with pytest.raises(RuntimeError, match="did not return processed logprobs"):
        extract_generated_token_logprobs([11], token_logprobs)


def test_extract_generated_token_logprobs_rejects_length_mismatch():
    with pytest.raises(RuntimeError, match="count does not match"):
        extract_generated_token_logprobs(
            [11, 12],
            [{11: _logprob(-0.25)}],
        )


def test_extract_generated_token_logprobs_rejects_missing_chosen_token():
    with pytest.raises(RuntimeError, match="chosen token 11"):
        extract_generated_token_logprobs(
            [11],
            [{99: _logprob(-0.25)}],
        )


def test_extract_generated_token_logprobs_rejects_nonfinite_value():
    with pytest.raises(RuntimeError, match="non-finite"):
        extract_generated_token_logprobs(
            [11],
            [{11: _logprob(float("nan"))}],
        )


def _mk_inputs(batch_size: int = 2, seq_len: int = 5):
    input_ids = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
    # make second example shorter
    input_lengths = torch.tensor([seq_len, seq_len - 2])
    return input_ids, input_lengths


def test_vllm_utils_regular_llm_path():
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        }
    )
    prompts = format_prompt_for_vllm_generation(data)
    assert isinstance(prompts, list) and len(prompts) == 2
    # first has full length
    assert prompts[0]["prompt_token_ids"] == input_ids[0].tolist()
    # second trimmed by input_lengths
    assert prompts[1]["prompt_token_ids"] == input_ids[1, : input_lengths[1]].tolist()


def test_vllm_utils_vlm_with_images_and_text():
    # Batch with two samples
    # both have content; first has one image, second has two images
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["<s>user: hi</s>", "<s>user: hello</s>"],
            "vllm_images": [["img1"], ["img2a", "img2b"]],
        }
    )

    prompts = format_prompt_for_vllm_generation(data)
    assert len(prompts) == 2
    assert prompts[0]["prompt"] == "<s>user: hi</s>"
    assert prompts[0]["multi_modal_data"]["image"] == "img1"
    assert prompts[1]["prompt"] == "<s>user: hello</s>"
    assert prompts[1]["multi_modal_data"]["image"] == ["img2a", "img2b"]


def test_vllm_utils_vlm_with_audio_and_video_intent_path():
    """IntentTrain/IntentBench rollouts must surface both modalities to vLLM.

    Asserts ``multi_modal_data`` contains a ``video`` key built from
    ``vllm_videos`` AND an ``audio`` key built from ``vllm_audios`` for the
    same prompt. This is the regression bar for AC-3 of the audio+video
    intent recipe; if either key is dropped at this site, vLLM rolls out a
    text-only / single-modality prompt and the smoke run silently degrades.
    """
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["<s>user: q1</s>", "<s>user: q2</s>"],
            "vllm_videos": [["frames-1"], ["frames-2"]],
            "vllm_audios": [[("audio-1", 16000)], [("audio-2", 16000)]],
            "task_name": ["intent-train", "intent-bench"],
        }
    )

    prompts = format_prompt_for_vllm_generation(data)
    assert len(prompts) == 2
    for i, prompt in enumerate(prompts):
        assert "multi_modal_data" in prompt, (
            f"prompt {i} missing multi_modal_data: keys={list(prompt)}"
        )
        mm = prompt["multi_modal_data"]
        assert "video" in mm, (
            f"prompt {i} dropped vllm_videos -> multi_modal_data['video']: "
            f"keys={list(mm)}"
        )
        assert "audio" in mm, (
            f"prompt {i} dropped vllm_audios -> multi_modal_data['audio']: "
            f"keys={list(mm)}"
        )
    # The independent-streams path explicitly does not set
    # mm_processor_kwargs={"use_audio_in_video": True} (Round 1 BitLesson
    # BL-20260428-omni-use-audio-in-video). If a future change re-introduces
    # that flag this assertion will need to be updated together with vLLM
    # acceptance evidence.
    for prompt in prompts:
        assert "mm_processor_kwargs" not in prompt


def test_vllm_utils_vlm_with_video_only():
    """Video-only path (no audio, no images) produces multi_modal_data with video key only."""
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["<s>user: q1</s>", "<s>user: q2</s>"],
            "vllm_videos": [["frames-1"], ["frames-2"]],
        }
    )

    prompts = format_prompt_for_vllm_generation(data)
    assert len(prompts) == 2
    for i, prompt in enumerate(prompts):
        assert "multi_modal_data" in prompt, f"prompt {i} missing multi_modal_data"
        mm = prompt["multi_modal_data"]
        assert "video" in mm, f"prompt {i} missing video key"
        assert "audio" not in mm, f"prompt {i} should not have audio key"
        assert "image" not in mm, f"prompt {i} should not have image key"


def test_vllm_utils_vlm_with_empty_videos_fallback_to_tokens():
    """Empty vllm_videos (per-sample) should fall back to prompt_token_ids."""
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["a", "b"],
            "vllm_videos": [[], []],
        }
    )
    prompts = format_prompt_for_vllm_generation(data)
    assert all("prompt_token_ids" in p for p in prompts)


def test_vllm_utils_vlm_with_missing_images_fallback_to_tokens():
    input_ids, input_lengths = _mk_inputs()
    # images None triggers fallback
    data_none = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["a", "b"],
            "vllm_images": None,
        }
    )
    prompts = format_prompt_for_vllm_generation(data_none)
    assert all("prompt_token_ids" in p for p in prompts)

    # images empty per sample also triggers fallback
    data_empty = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["a", "b"],
            "vllm_images": [[], []],
        }
    )
    prompts = format_prompt_for_vllm_generation(data_empty)
    assert all("prompt_token_ids" in p for p in prompts)


def test_vllm_utils_vlm_with_none_content_fallback_to_tokens_and_sample_idx():
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": [None, None],
            "vllm_images": [["img"], ["img"]],
        }
    )
    # even though images provided, None content should fallback to tokens
    prompts_all = format_prompt_for_vllm_generation(data)
    assert len(prompts_all) == 2
    assert all("prompt_token_ids" in p for p in prompts_all)

    # single-sample API
    p0 = format_prompt_for_vllm_generation(data, sample_idx=0)
    p1 = format_prompt_for_vllm_generation(data, sample_idx=1)
    assert isinstance(p0, dict) and isinstance(p1, dict)
    assert "prompt_token_ids" in p0 and "prompt_token_ids" in p1


def test_normalize_routed_experts_full_sequence_alignment():
    class Output:
        pass

    request_output = Output()
    completion_output = Output()
    completion_output.routed_experts = torch.arange(5 * 3 * 2).reshape(5, 3, 2)

    routed_experts = pad_and_align_routed_expert_indices(
        request_output,
        completion_output,
        valid_length=6,
        padded_length=8,
        device=torch.device("cpu"),
    )

    assert routed_experts.shape == (8, 3, 2)
    assert routed_experts.dtype == torch.int32
    assert torch.equal(
        routed_experts[:5], completion_output.routed_experts.to(torch.int32)
    )
    expected_default_route = torch.tensor([0, 1], dtype=torch.int32).view(1, 1, 2)
    assert torch.equal(routed_experts[5:], expected_default_route.expand(3, 3, 2))


def test_normalize_routed_experts_concatenates_prompt_and_decode():
    class Output:
        pass

    request_output = Output()
    completion_output = Output()
    request_output.prompt_routed_experts = torch.ones(2, 1, 2, dtype=torch.int32)
    completion_output.routed_experts = 2 * torch.ones(3, 1, 2, dtype=torch.int32)

    routed_experts = pad_and_align_routed_expert_indices(
        request_output,
        completion_output,
        valid_length=5,
        padded_length=5,
        device=torch.device("cpu"),
    )

    expected_default_route = torch.tensor([0, 1], dtype=torch.int32).view(1, 1, 2)
    assert torch.equal(routed_experts[:2], request_output.prompt_routed_experts)
    assert torch.equal(routed_experts[2:4], completion_output.routed_experts[:2])
    assert torch.equal(routed_experts[4:], expected_default_route.expand(1, 1, 2))


def test_normalize_routed_experts_uses_valid_dummy_route_for_missing_last_token():
    class Output:
        pass

    request_output = Output()
    completion_output = Output()
    completion_output.routed_experts = torch.tensor(
        [
            [[4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [10, 11, 12]],
        ],
        dtype=torch.int32,
    )

    routed_experts = pad_and_align_routed_expert_indices(
        request_output,
        completion_output,
        valid_length=3,
        padded_length=5,
        device=torch.device("cpu"),
    )

    expected_default_route = torch.tensor([0, 1, 2], dtype=torch.int32).view(1, 1, 3)
    assert torch.equal(routed_experts[:2], completion_output.routed_experts)
    assert torch.equal(routed_experts[2:], expected_default_route.expand(3, 2, 3))


def test_normalize_routed_experts_keeps_final_token_dummy_even_if_vllm_returns_route():
    class Output:
        pass

    request_output = Output()
    completion_output = Output()
    completion_output.routed_experts = torch.tensor(
        [
            [[4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [10, 11, 12]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.int32,
    )

    routed_experts = pad_and_align_routed_expert_indices(
        request_output,
        completion_output,
        valid_length=3,
        padded_length=3,
        device=torch.device("cpu"),
    )

    expected_default_route = torch.tensor([0, 1, 2], dtype=torch.int32).view(1, 1, 3)
    assert torch.equal(routed_experts[:2], completion_output.routed_experts[:2])
    assert torch.equal(routed_experts[2:], expected_default_route.expand(1, 2, 3))


def test_normalize_routed_experts_strict_mode_marks_missing_routes_for_fallback():
    class Output:
        pass

    request_output = Output()
    request_output.num_cached_tokens = 4
    completion_output = Output()
    completion_output.routed_experts = torch.ones(2, 1, 2, dtype=torch.int32)

    routed_experts, stats = pad_and_align_routed_expert_indices(
        request_output,
        completion_output,
        valid_length=6,
        padded_length=6,
        device=torch.device("cpu"),
        require_complete_routed_experts=True,
        return_stats=True,
    )

    assert stats == {
        "actual_routes": 2,
        "expected_routes": 5,
        "missing_routes": 3,
        "surplus_routes": 0,
    }
    assert torch.equal(routed_experts[:2], completion_output.routed_experts)
    assert torch.equal(
        routed_experts[2:5],
        torch.full((3, 1, 2), R3_MISSING_ROUTE_SENTINEL, dtype=torch.int32),
    )
    expected_default_route = torch.tensor([0, 1], dtype=torch.int32).view(1, 1, 2)
    assert torch.equal(routed_experts[5:], expected_default_route)


def test_normalize_routed_experts_can_reject_missing_routes_when_fallback_disabled():
    class Output:
        pass

    request_output = Output()
    request_output.num_cached_tokens = 4
    completion_output = Output()
    completion_output.routed_experts = torch.ones(2, 1, 2, dtype=torch.int32)

    with pytest.raises(ValueError, match="incomplete routed_experts"):
        pad_and_align_routed_expert_indices(
            request_output,
            completion_output,
            valid_length=6,
            padded_length=6,
            device=torch.device("cpu"),
            require_complete_routed_experts=True,
            allow_missing_routed_experts_fallback=False,
        )


def test_normalize_routed_experts_strict_mode_rejects_surplus_routes():
    class Output:
        pass

    request_output = Output()
    request_output.num_cached_tokens = 0
    completion_output = Output()
    completion_output.routed_experts = torch.ones(4, 1, 2, dtype=torch.int32)

    with pytest.raises(ValueError, match="too many routed_experts routes"):
        pad_and_align_routed_expert_indices(
            request_output,
            completion_output,
            valid_length=3,
            padded_length=3,
            device=torch.device("cpu"),
            require_complete_routed_experts=True,
        )


def test_attach_routed_experts_to_chat_response_choices_reassociates_by_choice_index():
    final_res = SimpleNamespace(
        prompt_token_ids=[101, 102, 103],
        prompt_routed_experts=torch.tensor([[[10]], [[11]]], dtype=torch.int32),
        outputs=[
            SimpleNamespace(
                index=1,
                token_ids=[201, 202],
                routed_experts=torch.tensor([[[31]], [[32]]], dtype=torch.int32),
            ),
            SimpleNamespace(
                index=0,
                token_ids=[200],
                routed_experts=torch.tensor([[[30]]], dtype=torch.int32),
            ),
        ],
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(index=0, message=SimpleNamespace()),
            SimpleNamespace(index=1, message=SimpleNamespace()),
        ]
    )

    attach_routed_experts_to_chat_response_choices(
        response,
        final_res,
        device=torch.device("cpu"),
    )

    assert response.choices[0].message.routed_experts == [
        [[10]],
        [[11]],
        [[30]],
        [[0]],
    ]
    assert response.choices[1].message.routed_experts == [
        [[10]],
        [[11]],
        [[31]],
        [[32]],
        [[0]],
    ]


def test_attach_routed_experts_to_chat_response_choices_requires_routed_experts():
    final_res = SimpleNamespace(
        prompt_token_ids=[101, 102],
        outputs=[SimpleNamespace(index=0, token_ids=[200])],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace())]
    )

    with pytest.raises(RuntimeError, match="did not include routed_experts"):
        attach_routed_experts_to_chat_response_choices(
            response,
            final_res,
            device=torch.device("cpu"),
        )


def test_attach_routed_experts_to_chat_response_choices_warns_on_missing_routes():
    final_res = SimpleNamespace(
        prompt_token_ids=[101, 102, 103],
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=[200, 201],
                routed_experts=torch.tensor([[[10]], [[11]]], dtype=torch.int32),
            )
        ],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace())]
    )
    logger = MagicMock()

    attach_routed_experts_to_chat_response_choices(
        response,
        final_res,
        device=torch.device("cpu"),
        logger=logger,
    )

    logger.warning.assert_called_once_with(
        "R3 router replay fallback: vLLM returned incomplete "
        "routed_experts for chat choice_idx=%d, "
        "missing_token_routes=%d, actual_routes=%d, "
        "expected_routes=%d. Megatron will use its own router "
        "for those missing token routes.",
        0,
        2,
        2,
        4,
    )
    assert response.choices[0].message.routed_experts == [
        [[10]],
        [[11]],
        [[R3_MISSING_ROUTE_SENTINEL]],
        [[R3_MISSING_ROUTE_SENTINEL]],
        [[0]],
    ]


def test_attach_routed_experts_to_chat_response_choices_raises_for_unmatched_choice():
    final_res = SimpleNamespace(
        prompt_token_ids=[101, 102],
        outputs=[
            SimpleNamespace(
                index=1,
                token_ids=[200],
                routed_experts=torch.tensor([[[10]], [[11]]], dtype=torch.int32),
            )
        ],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace())]
    )

    with pytest.raises(RuntimeError, match=r"missing_choice_indices=\[0\]"):
        attach_routed_experts_to_chat_response_choices(
            response,
            final_res,
            device=torch.device("cpu"),
        )


def test_model_dump_chat_response_with_routed_experts_preserves_dynamic_field():
    routed_experts = [[[1]], [[2]]]

    class Response:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(routed_experts=routed_experts),
            )
        ]

        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                        }
                    }
                ]
            }

    response_dict = model_dump_chat_response_with_routed_experts(Response())

    assert response_dict["choices"][0]["message"]["routed_experts"] == routed_experts


@pytest.mark.vllm
def test_vllm_speculative_decoding_patch_removed():
    # The speculative decoding patch was fixed upstream in vLLM >= 0.14.0:
    # https://github.com/vllm-project/vllm/pull/30319
    # Verify the patch function has been removed from the codebase.
    import importlib

    vllm_worker = importlib.import_module("nemo_rl.models.generation.vllm.vllm_worker")
    assert not hasattr(vllm_worker, "_patch_vllm_speculative_decoding_post_step"), (
        "_patch_vllm_speculative_decoding_post_step still exists in vllm_worker.py "
        "but vLLM >= 0.14.0 includes the upstream fix. Please remove it."
    )


def test_aggregate_spec_decode_counters():
    """Test aggregation of speculative decoding counters from multiple workers."""
    worker_metrics = [
        {
            "vllm:spec_decode_num_drafts": 100.0,
            "vllm:spec_decode_num_draft_tokens": 300.0,
            "vllm:spec_decode_num_accepted_tokens": 240.0,
            "other_metric": 999.0,  # Should be ignored
        },
        {
            "vllm:spec_decode_num_drafts": 150.0,
            "vllm:spec_decode_num_draft_tokens": 450.0,
            "vllm:spec_decode_num_accepted_tokens": 360.0,
        },
    ]

    counters = aggregate_spec_decode_counters(worker_metrics)

    assert counters["vllm:spec_decode_num_drafts"] == 250.0
    assert counters["vllm:spec_decode_num_draft_tokens"] == 750.0
    assert counters["vllm:spec_decode_num_accepted_tokens"] == 600.0
    assert "other_metric" not in counters


def test_compute_spec_decode_metrics():
    """Test computation of speculative decoding metrics from counter snapshots."""
    start_counters = {
        "vllm:spec_decode_num_drafts": 100.0,
        "vllm:spec_decode_num_draft_tokens": 300.0,
        "vllm:spec_decode_num_accepted_tokens": 200.0,
    }
    end_counters = {
        "vllm:spec_decode_num_drafts": 200.0,
        "vllm:spec_decode_num_draft_tokens": 600.0,
        "vllm:spec_decode_num_accepted_tokens": 440.0,
    }

    metrics = compute_spec_decode_metrics(start_counters, end_counters)

    # Delta values
    assert metrics["vllm/spec_num_drafts"] == 100.0
    assert metrics["vllm/spec_num_draft_tokens"] == 300.0
    assert metrics["vllm/spec_num_accepted_tokens"] == 240.0

    # Derived metrics
    # acceptance_length = 1 + (accepted / drafts) = 1 + (240 / 100) = 3.4
    assert math.isclose(metrics["vllm/spec_acceptance_length"], 3.4, rel_tol=1e-6)
    # acceptance_rate = accepted / draft_tokens = 240 / 300 = 0.8
    assert math.isclose(metrics["vllm/spec_acceptance_rate"], 0.8, rel_tol=1e-6)


def test_compute_spec_decode_metrics_derives_tail_gate_metrics() -> None:
    start_counters: dict[str | tuple[str, int], float] = {
        "vllm:spec_decode_tail_gate_decisions": 10.0,
        "vllm:spec_decode_tail_gate_enabled_steps": 2.0,
        "vllm:spec_decode_tail_gate_disabled_steps": 8.0,
        "vllm:spec_decode_tail_gate_activations": 3.0,
        "vllm:spec_decode_tail_gate_active_requests_sum": 100.0,
        "vllm:spec_decode_tail_gate_mean_sequence_length_sum": 32000.0,
        "vllm:spec_decode_tail_gate_activation_batch_sum": 48.0,
        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": 24576.0,
        "vllm:spec_decode_tail_gate_predicted_speedup_sum": 3.0,
        "vllm:spec_decode_tail_gate_predicted_speedup_count": 3.0,
        "vllm:spec_decode_tail_gate_k_0_steps": 8.0,
        "vllm:spec_decode_tail_gate_k_5_steps": 2.0,
    }
    end_counters: dict[str | tuple[str, int], float] = {
        "vllm:spec_decode_tail_gate_decisions": 14.0,
        "vllm:spec_decode_tail_gate_enabled_steps": 3.0,
        "vllm:spec_decode_tail_gate_disabled_steps": 11.0,
        "vllm:spec_decode_tail_gate_activations": 4.0,
        "vllm:spec_decode_tail_gate_active_requests_sum": 180.0,
        "vllm:spec_decode_tail_gate_mean_sequence_length_sum": 72000.0,
        "vllm:spec_decode_tail_gate_activation_batch_sum": 64.0,
        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": 32768.0,
        "vllm:spec_decode_tail_gate_predicted_speedup_sum": 5.24,
        "vllm:spec_decode_tail_gate_predicted_speedup_count": 5.0,
        "vllm:spec_decode_tail_gate_k_0_steps": 11.0,
        "vllm:spec_decode_tail_gate_k_5_steps": 3.0,
    }

    metrics = compute_spec_decode_metrics(start_counters, end_counters)

    assert metrics["vllm/tail_gate_enabled_step_ratio"] == pytest.approx(0.25)
    assert metrics["vllm/tail_gate_activation_batch"] == pytest.approx(16.0)
    assert metrics["vllm/tail_gate_activation_seq_len"] == pytest.approx(8192.0)
    assert metrics["vllm/tail_gate_predicted_speedup"] == pytest.approx(1.12)
    assert metrics["vllm/tail_gate_advance_only_step_ratio"] == pytest.approx(0.75)
    assert metrics["vllm/tail_gate_k_0_steps"] == pytest.approx(3.0)
    assert metrics["vllm/tail_gate_k_5_steps"] == pytest.approx(1.0)
    assert metrics["vllm/tail_gate_k_0_step_ratio"] == pytest.approx(0.75)
    assert metrics["vllm/tail_gate_k_5_step_ratio"] == pytest.approx(0.25)


def test_compute_spec_decode_metrics_tail_gate_zero_denominators_are_zero() -> None:
    counters: dict[str | tuple[str, int], float] = {
        "vllm:spec_decode_tail_gate_decisions": 0.0,
        "vllm:spec_decode_tail_gate_enabled_steps": 0.0,
        "vllm:spec_decode_tail_gate_disabled_steps": 0.0,
        "vllm:spec_decode_tail_gate_activations": 0.0,
        "vllm:spec_decode_tail_gate_active_requests_sum": 0.0,
        "vllm:spec_decode_tail_gate_mean_sequence_length_sum": 0.0,
        "vllm:spec_decode_tail_gate_activation_batch_sum": 0.0,
        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": 0.0,
        "vllm:spec_decode_tail_gate_predicted_speedup_sum": 0.0,
        "vllm:spec_decode_tail_gate_predicted_speedup_count": 0.0,
        "vllm:spec_decode_tail_gate_k_0_steps": 0.0,
    }

    metrics = compute_spec_decode_metrics(counters, counters)

    for metric_name in (
        "vllm/tail_gate_enabled_step_ratio",
        "vllm/tail_gate_activation_batch",
        "vllm/tail_gate_activation_seq_len",
        "vllm/tail_gate_predicted_speedup",
        "vllm/tail_gate_advance_only_step_ratio",
        "vllm/tail_gate_k_0_step_ratio",
    ):
        assert metrics[metric_name] == 0.0
        assert not math.isnan(metrics[metric_name])


def test_compute_spec_decode_metrics_includes_cudagraph_coverage() -> None:
    start_counters = {
        "vllm:spec_decode_cudagraph_target_calls_none": 10.0,
        "vllm:spec_decode_cudagraph_target_calls_piecewise": 20.0,
        "vllm:spec_decode_cudagraph_target_unpadded_tokens_none": 100.0,
        "vllm:spec_decode_cudagraph_target_unpadded_tokens_piecewise": 200.0,
        "vllm:spec_decode_cudagraph_target_padded_tokens_piecewise": 240.0,
        "vllm:spec_decode_cudagraph_target_fallback_missing_key": 10.0,
        "vllm:spec_decode_cudagraph_draft_calls_full": 5.0,
        "vllm:spec_decode_cudagraph_draft_unpadded_tokens_full": 50.0,
        "vllm:spec_decode_cudagraph_draft_padded_tokens_full": 50.0,
        "vllm:spec_decode_cudagraph_draft_decode_calls_none": 20.0,
        "vllm:spec_decode_cudagraph_draft_decode_unpadded_tokens_none": 20.0,
    }
    end_counters = {
        "vllm:spec_decode_cudagraph_target_calls_none": 12.0,
        "vllm:spec_decode_cudagraph_target_calls_piecewise": 28.0,
        "vllm:spec_decode_cudagraph_target_unpadded_tokens_none": 120.0,
        "vllm:spec_decode_cudagraph_target_unpadded_tokens_piecewise": 280.0,
        "vllm:spec_decode_cudagraph_target_padded_tokens_piecewise": 336.0,
        "vllm:spec_decode_cudagraph_target_fallback_missing_key": 12.0,
        "vllm:spec_decode_cudagraph_draft_calls_full": 15.0,
        "vllm:spec_decode_cudagraph_draft_unpadded_tokens_full": 150.0,
        "vllm:spec_decode_cudagraph_draft_padded_tokens_full": 150.0,
        "vllm:spec_decode_cudagraph_draft_decode_calls_none": 100.0,
        "vllm:spec_decode_cudagraph_draft_decode_unpadded_tokens_none": 100.0,
    }

    metrics = compute_spec_decode_metrics(start_counters, end_counters)

    assert metrics["vllm/cudagraph_target_total_calls"] == 10.0
    assert metrics["vllm/cudagraph_target_graph_calls"] == 8.0
    assert metrics["vllm/cudagraph_target_eager_calls"] == 2.0
    assert math.isclose(metrics["vllm/cudagraph_target_graph_call_ratio"], 0.8)
    assert math.isclose(metrics["vllm/cudagraph_target_eager_call_ratio"], 0.2)
    assert math.isclose(metrics["vllm/cudagraph_target_graph_token_ratio"], 0.8)
    assert math.isclose(metrics["vllm/cudagraph_target_eager_token_ratio"], 0.2)
    assert math.isclose(metrics["vllm/cudagraph_target_padding_overhead_ratio"], 0.2)
    assert metrics["vllm/cudagraph_target_fallback_missing_key"] == 2.0
    assert metrics["vllm/cudagraph_draft_graph_call_ratio"] == 1.0
    assert metrics["vllm/cudagraph_draft_decode_eager_call_ratio"] == 1.0
    assert metrics["vllm/cudagraph_all_total_calls"] == 100.0
    assert math.isclose(metrics["vllm/cudagraph_all_eager_call_ratio"], 0.82)
    assert math.isclose(metrics["vllm/cudagraph_all_graph_call_ratio"], 0.18)
