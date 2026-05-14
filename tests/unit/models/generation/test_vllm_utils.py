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

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.models.generation.vllm.utils import (
    aggregate_spec_decode_counters,
    compute_spec_decode_metrics,
    extract_sampled_token_logprob,
    format_prompt_for_vllm_generation,
    _compute_video_timestamps,
)
from nemo_rl.models.generation.vllm.vllm_generation import _build_compact_mm_payload


def _mk_inputs(batch_size: int = 2, seq_len: int = 5):
    input_ids = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
    # make second example shorter
    input_lengths = torch.tensor([seq_len, seq_len - 2])
    return input_ids, input_lengths


def _summarize_image_payload(image_payload):
    if isinstance(image_payload, list):
        return [_summarize_image_payload(image) for image in image_payload]
    if isinstance(image_payload, Image.Image):
        return {
            "kind": "pil",
            "mode": image_payload.mode,
            "size": list(image_payload.size),
        }
    return image_payload


def _summarize_prompt(prompt: dict) -> dict:
    summary = dict(prompt)
    multi_modal_data = summary.get("multi_modal_data")
    if isinstance(multi_modal_data, dict) and "image" in multi_modal_data:
        summary["multi_modal_data"] = {
            **multi_modal_data,
            "image": _summarize_image_payload(multi_modal_data["image"]),
        }
    return summary


class _FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob


def test_extract_sampled_token_logprob_uses_sampled_token_not_first_entry():
    logprob_dict = {
        11: _FakeLogprob(-0.1),
        42: _FakeLogprob(-7.5),
    }

    assert extract_sampled_token_logprob(logprob_dict, 42) == pytest.approx(-7.5)


def test_extract_sampled_token_logprob_accepts_int_equivalent_keys():
    logprob_dict = {
        "42": _FakeLogprob(-3.25),
    }

    assert extract_sampled_token_logprob(logprob_dict, 42) == pytest.approx(-3.25)


def test_extract_sampled_token_logprob_returns_none_for_missing_token():
    logprob_dict = {
        11: _FakeLogprob(-0.1),
    }

    assert extract_sampled_token_logprob(logprob_dict, 42) is None


def test_compute_video_timestamps_defaults_to_sft_v2_duration(monkeypatch):
    monkeypatch.delenv("NRL_VIDEO_SAMPLING_STYLE", raising=False)

    frame_count, timestamps = _compute_video_timestamps(
        total_duration=120.0,
        num_frames=256,
        total_frames_in_file=3000,
        original_num_frames=256,
        temporal_patch_size=2,
    )

    assert frame_count == 240
    assert timestamps[:3] == pytest.approx([0.24791666, 0.74375, 1.23958333])


def test_compute_video_timestamps_current_fixed_uses_requested_frame_count(monkeypatch):
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "current_fixed")

    frame_count, timestamps = _compute_video_timestamps(
        total_duration=120.0,
        num_frames=32,
        total_frames_in_file=3000,
        original_num_frames=32,
        temporal_patch_size=2,
    )

    assert frame_count == 32
    assert timestamps[:3] == pytest.approx([1.859375, 5.578125, 9.296875])


def test_compute_video_timestamps_sft_v2_duration_uses_num_frames_as_cap(
    monkeypatch,
):
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "sft_v2_duration")

    short_count, short_timestamps = _compute_video_timestamps(
        total_duration=5.12,
        num_frames=32,
        total_frames_in_file=128,
        original_num_frames=32,
        temporal_patch_size=2,
    )
    capped_long_count, capped_long_timestamps = _compute_video_timestamps(
        total_duration=120.0,
        num_frames=128,
        total_frames_in_file=3000,
        original_num_frames=128,
        temporal_patch_size=2,
    )
    full_sft_long_count, _ = _compute_video_timestamps(
        total_duration=120.0,
        num_frames=256,
        total_frames_in_file=3000,
        original_num_frames=256,
        temporal_patch_size=2,
    )

    assert short_count == 10
    assert short_timestamps[:3] == pytest.approx([0.206, 0.618, 1.03])
    assert capped_long_count == 128
    assert capped_long_timestamps[:3] == pytest.approx(
        [0.46484375, 1.39453125, 2.32421875]
    )
    assert full_sft_long_count == 240


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


def test_vllm_utils_vlm_loads_local_image_paths(tmp_path):
    input_ids, input_lengths = _mk_inputs(batch_size=1)
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (10, 12), color=(1, 2, 3)).save(image_path)

    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths[:1],
            "vllm_content": ["<s>user: hi</s>"],
            "vllm_images": [[str(image_path)]],
        }
    )

    prompt = format_prompt_for_vllm_generation(data, sample_idx=0)

    image_payload = prompt["multi_modal_data"]["image"]
    assert isinstance(image_payload, Image.Image)
    assert image_payload.size == (10, 12)


def test_vllm_utils_vlm_forwards_parity_mm_processor_kwargs_from_packed_imgs_sizes():
    input_ids, input_lengths = _mk_inputs()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["prompt-a", "prompt-b"],
            "vllm_images": [["img1"], ["img2a", "img2b"]],
            "imgs_sizes": PackedTensor(
                [
                    torch.tensor([[4, 5]], dtype=torch.int32),
                    torch.tensor([[6, 4], [8, 8]], dtype=torch.int32),
                ],
                dim_to_pack=0,
            ),
            "vllm_max_num_tiles": [12, None],
            "vllm_max_num_patches": [None, 256],
        }
    )

    prompts = format_prompt_for_vllm_generation(data)

    assert prompts[0]["mm_processor_kwargs"] == {
        "max_num_tiles": 12,
        "precomputed_imgs_sizes": [[4, 5]],
    }
    assert prompts[1]["mm_processor_kwargs"] == {
        "max_num_patches": 256,
        "precomputed_imgs_sizes": [[6, 4], [8, 8]],
    }


def test_vllm_utils_vlm_forwards_imgs_sizes_from_list_form():
    input_ids, input_lengths = _mk_inputs(batch_size=1)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths[:1],
            "vllm_content": ["prompt-a"],
            "vllm_images": [["img1"]],
            "imgs_sizes": [[[10, 12]]],
        }
    )

    prompt = format_prompt_for_vllm_generation(data, sample_idx=0)

    assert prompt["mm_processor_kwargs"] == {"precomputed_imgs_sizes": [[10, 12]]}


def test_vllm_utils_omni_video_as_images_rewrites_prompt_and_forwards_sizes(
    monkeypatch,
):
    input_ids, input_lengths = _mk_inputs(batch_size=1)

    def fake_load_video_frames(video_path, num_frames=8, temporal_patch_size=1):
        assert video_path == "video.mp4"
        assert num_frames == 2
        assert temporal_patch_size == 1
        return np.zeros((2, 12, 10, 3), dtype=np.uint8)

    monkeypatch.setenv("NRL_VIDEO_PROMPT_STYLE", "current_plain_native")
    monkeypatch.setenv("NRL_VLLM_VIDEO_AS_IMAGES", "1")
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.utils.load_video_frames",
        fake_load_video_frames,
    )

    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths[:1],
            "vllm_content": ["prefix <video> suffix"],
            "vllm_images": [[]],
            "vllm_videos": [["video.mp4"]],
            "vllm_num_frames": [2],
            "vllm_temporal_patch_size": [1],
            "imgs_sizes": [[[12, 10], [12, 10]]],
        }
    )

    prompt = format_prompt_for_vllm_generation(data, sample_idx=0)

    assert prompt["prompt"] == "prefix <image><image> suffix"
    image_payload = prompt["multi_modal_data"]["image"]
    assert isinstance(image_payload, list)
    assert len(image_payload) == 2
    assert all(isinstance(image, Image.Image) for image in image_payload)
    assert "video" not in prompt["multi_modal_data"]
    assert prompt["mm_processor_kwargs"] == {
        "max_num_tiles": 1,
        "video_as_images": True,
        "precomputed_imgs_sizes": [[12, 10], [12, 10]],
    }


def test_vllm_utils_omni_native_video_keeps_video_payload(monkeypatch):
    input_ids, input_lengths = _mk_inputs(batch_size=1)

    def fake_load_video_frames_with_metadata(
        video_path, num_frames=8, temporal_patch_size=1
    ):
        assert video_path == "video.mp4"
        assert num_frames == 2
        assert temporal_patch_size == 2
        return np.zeros((2, 12, 10, 3), dtype=np.uint8), {
            "fps": 2.0,
            "frames_indices": [0, 1],
            "do_sample_frames": False,
        }

    monkeypatch.delenv("NRL_VLLM_VIDEO_AS_IMAGES", raising=False)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.utils.load_video_frames_with_metadata",
        fake_load_video_frames_with_metadata,
    )

    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths[:1],
            "vllm_content": ["prefix <video> suffix"],
            "vllm_videos": [["video.mp4"]],
            "vllm_num_frames": [2],
            "vllm_temporal_patch_size": [2],
            "vllm_max_num_patches": [1008],
        }
    )

    prompt = format_prompt_for_vllm_generation(data, sample_idx=0)

    assert prompt["prompt"] == "prefix <video> suffix"
    assert "image" not in prompt["multi_modal_data"]
    video_payload = prompt["multi_modal_data"]["video"]
    frames, metadata = video_payload
    assert frames.shape == (2, 12, 10, 3)
    assert metadata["frames_indices"] == [0, 1]
    assert prompt["mm_processor_kwargs"] == {"max_num_patches": 1008}


def test_vllm_utils_sft_v2_grouped_video_uses_native_video_metadata_sidechannel(
    monkeypatch,
):
    input_ids, input_lengths = _mk_inputs(batch_size=1)

    def fake_load_video_frames_with_metadata(
        video_path, num_frames=8, temporal_patch_size=1
    ):
        assert video_path == "video.mp4"
        assert num_frames == 4
        assert temporal_patch_size == 2
        return np.zeros((4, 12, 10, 3), dtype=np.uint8), {
            "fps": 1.0,
            "frames_indices": [0, 1, 2, 3],
            "do_sample_frames": False,
        }

    monkeypatch.setenv("NRL_VIDEO_PROMPT_STYLE", "sft_v2_grouped")
    monkeypatch.setenv("NRL_VLLM_VIDEO_AS_IMAGES", "1")
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.utils.load_video_frames_with_metadata",
        fake_load_video_frames_with_metadata,
    )

    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths[:1],
            "vllm_content": ["This is a video:\n<video>\nQuestion?"],
            "vllm_videos": [["video.mp4"]],
            "vllm_num_frames": [4],
            "vllm_temporal_patch_size": [2],
            "vllm_video_prompt_style": ["sft_v2_grouped"],
            "vllm_video_frame_indices": [[[10, 20, 30, 40]]],
            "vllm_video_fps": [[5.0]],
            "vllm_max_num_patches": [1008],
        }
    )

    prompt = format_prompt_for_vllm_generation(data, sample_idx=0)

    assert prompt["prompt"] == "This is a video:\n<video>\nQuestion?"
    assert "image" not in prompt["multi_modal_data"]
    frames, metadata = prompt["multi_modal_data"]["video"]
    assert frames.shape == (4, 12, 10, 3)
    assert metadata["frames_indices"] == [10, 20, 30, 40]
    assert metadata["fps"] == 5.0
    assert prompt["mm_processor_kwargs"] == {"max_num_patches": 1008}


def test_vllm_utils_vlm_compact_payload_matches_raw_prompt_format_for_local_paths(
    tmp_path,
):
    input_ids = torch.arange(15).view(3, 5)
    input_lengths = torch.tensor([5, 4, 3])
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (10, 12), color=(1, 2, 3)).save(image_path)
    raw_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "vllm_content": ["prompt-a", "prompt-a", None],
            "vllm_images": [[str(image_path)], [str(image_path)], []],
            "imgs_sizes": PackedTensor(
                [
                    torch.tensor([[4, 5]], dtype=torch.int32),
                    torch.tensor([[4, 5]], dtype=torch.int32),
                    torch.tensor([[9, 9]], dtype=torch.int32),
                ],
                dim_to_pack=0,
            ),
            "vllm_max_num_tiles": [12, 12, None],
        }
    )

    raw_prompts = format_prompt_for_vllm_generation(raw_data)

    compact_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "imgs_sizes": raw_data["imgs_sizes"],
            "vllm_mm_compact_payload": _build_compact_mm_payload(
                SlicedDataDict(dict(raw_data))
            ),
        }
    )

    compact_prompts = format_prompt_for_vllm_generation(compact_data)

    assert [_summarize_prompt(prompt) for prompt in compact_prompts] == [
        _summarize_prompt(prompt) for prompt in raw_prompts
    ]


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


@pytest.mark.vllm
def test_vllm_speculative_decoding_patch_still_needed():
    # This test reminds to remove the vLLM patch when no longer needed.
    # The patch was fixed upstream: https://github.com/vllm-project/vllm/pull/30319
    # When this test fails, remove _patch_vllm_speculative_decoding_post_step()
    # from nemo_rl/models/generation/vllm/vllm_worker.py
    from importlib.metadata import version

    from packaging.version import Version

    assert Version(version("vllm")) < Version("0.14.0"), (
        "vLLM >= 0.14.0 includes the speculative decoding fix from "
        "https://github.com/vllm-project/vllm/pull/30319. "
        "Please remove the _patch_vllm_speculative_decoding_post_step() function "
        "from nemo_rl/models/generation/vllm/vllm_worker.py"
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
