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

import torch
from PIL import Image

from nemo_rl.data.processors import (
    _append_sft_v2_grouped_video_content,
    _collapse_video_frame_token_wrappers,
    _get_video_prompt_style,
    _timestamps_from_video_metadata,
)


def test_video_prompt_style_defaults_to_sft_v2_grouped(monkeypatch):
    monkeypatch.delenv("NRL_VIDEO_PROMPT_STYLE", raising=False)

    assert _get_video_prompt_style() == "sft_v2_grouped"


def test_sft_v2_grouped_video_content_uses_one_vllm_video_marker_and_all_frames():
    frames = [Image.new("RGB", (2, 2), color=(i, i, i)) for i in range(4)]
    user_content = []
    vllm_content = []

    _append_sft_v2_grouped_video_content(
        user_content=user_content,
        vllm_content=vllm_content,
        video_path="video.mp4",
        frames=frames,
        timestamps=[0.0, 0.5, 1.0, 1.5],
        temporal_patch_size=2,
    )

    user_text = "".join(
        item.get("text", "") for item in user_content if item["type"] == "text"
    )
    assert user_text == (
        "This is a video:\n"
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds: \n"
        "Frame 3 sampled at 1.00 seconds and frame 4 sampled at 1.50 seconds: \n"
    )
    assert sum(item["type"] == "image" for item in user_content) == 4
    assert vllm_content == [
        {"type": "text", "text": "This is a video:"},
        {"type": "video", "video": "video.mp4"},
    ]


def test_timestamps_from_video_metadata_match_vllm_integer_frame_duration():
    timestamps = _timestamps_from_video_metadata(
        {"fps": 2.5, "frames_indices": [1, 3, 5, 7]},
        num_frames=4,
    )

    assert timestamps == [0.4, 1.2, 2.0, 2.8]


def test_video_frame_collapse_preserves_non_video_image_wrappers():
    img_start_id = 101
    img_end_id = 102
    user_ids = torch.tensor(
        [
            1,
            img_start_id,
            11,
            img_end_id,
            2,
            img_start_id,
            21,
            img_end_id,
            3,
            img_start_id,
            22,
            img_end_id,
            4,
            img_start_id,
            31,
            img_end_id,
            5,
        ],
        dtype=torch.int64,
    )

    collapsed = _collapse_video_frame_token_wrappers(
        user_ids,
        video_flags=[False, True, True, False],
        temporal_patch_size=2,
        img_start_id=img_start_id,
        img_end_id=img_end_id,
    )

    assert collapsed.tolist() == [
        1,
        img_start_id,
        11,
        img_end_id,
        2,
        img_start_id,
        21,
        img_end_id,
        3,
        4,
        img_start_id,
        31,
        img_end_id,
        5,
    ]
