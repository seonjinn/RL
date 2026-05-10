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

"""Video dataset for VLM RL training.

Expected JSONL format:
    {
        "videos": ["path/to/video.mp4"],
        "images": ["path/to/image.jpg"],   // optional, for mixed image+video data
        "question": "What happens in this video?",
        "answer": "A cat jumps on the table",
        "dataset": "video_qa",
        "verifier": "string-match"
    }

Videos are loaded and uniformly sampled into N frames, which are then treated
as images by the VLM pipeline (each frame becomes an <image> token in the prompt).
The number of frames is controlled by data.num_frames in the training config.
"""

import json
import os
import random
import re
from typing import Any, Optional

_DATASET_DEBUG = os.environ.get("NRL_DATASET_DEBUG", "0") == "1"
_FORMAT_DEBUG = os.environ.get("NRL_DATASET_FORMAT_DEBUG", "0") == "1"

from datasets import Dataset, Features, Sequence, Value

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets.blend_v1 import get_verifier, unify_answer_format
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.environments.mmpr_filtered_reward import question_asks_for_reasoning


def _parse_bool(value) -> bool:
    """Safely convert a value that may be a bool or a string like ``"false"``."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


class VideoDataset(RawDataset):
    """Dataset class for loading video QA data from JSONL files."""

    def __init__(
        self,
        train_data_path: Optional[str] = None,
        data_path: Optional[str] = None,
        prompt_file: Optional[str] = None,
        val_size: int = 0,
        split_validation_size: float = 0,
        seed: int = 42,
        task_name: str = "video_dataset",
        **kwargs,
    ):
        self.task_name = task_name
        path = train_data_path or data_path
        if not path:
            raise ValueError("VideoDataset requires a JSONL path")
        full_dataset = self._load_jsonl(path)
        if val_size > 0 and len(full_dataset) > val_size:
            cutoff = len(full_dataset) - val_size
            val_dataset = full_dataset.select(range(cutoff, len(full_dataset)))
            train_dataset = full_dataset.select(range(cutoff))
        else:
            train_dataset = full_dataset
            val_dataset = None
        self.dataset = train_dataset
        self.val_dataset = val_dataset
        self.split_train_validation(split_validation_size, seed)
        self.formatted_ds = {"train": self.dataset, "validation": self.val_dataset}
        self.task_spec = TaskDataSpec(task_name=self.task_name, prompt_file=prompt_file)

    def set_task_spec(self, data_config: dict):
        super().set_task_spec(data_config)
        self.task_spec.num_frames = data_config.get("num_frames", 8)
        self.task_spec.max_num_tiles = data_config.get("max_num_tiles", None)
        self.task_spec.max_num_patches = data_config.get("max_num_patches", None)
        self.task_spec.use_audio = data_config.get("use_audio", False)
        self.task_spec.max_audio_duration = data_config.get("max_audio_duration", None)
        if _DATASET_DEBUG:
            print(
                f"[VideoDataset] task={self.task_name} num_frames={self.task_spec.num_frames} "
                f"max_num_tiles={self.task_spec.max_num_tiles} "
                f"max_num_patches={self.task_spec.max_num_patches} "
                f"use_audio={self.task_spec.use_audio} "
                f"max_audio_duration={self.task_spec.max_audio_duration}"
            )

    def _load_jsonl(self, path: str) -> Dataset:
        """Load a JSONL with video/image paths and conversations into a Dataset."""
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                videos = row.get("videos") or ([row["video"]] if "video" in row else [])
                images = row.get("images") or ([row["image"]] if "image" in row else [])

                if not videos and not images:
                    continue

                question = row["question"].strip()
                # Clean up any existing placeholder tokens
                for token in ("<image>", "<video>"):
                    question = question.replace(token, "").strip()

                question = unify_answer_format(question)

                answer = row.get("answer", "")
                # Image-only rows: use blend_v1-style hardcoded verifier mapping
                if images and not videos and "dataset" in row:
                    verifier = get_verifier(row)
                else:
                    verifier = row.get("verifier", "string-match")

                rows.append(
                    {
                        "videos": videos,
                        "images": images,
                        "question": question,
                        "answer": answer,
                        "verifier": verifier,
                        "task_name": self.task_name,
                        "load_audio_flag": _parse_bool(row.get("load_audio_flag", False)),
                    }
                )
        if not rows:
            raise ValueError(
                f"No valid rows loaded from {path}. "
                "Every row was skipped (each row must contain a non-empty 'videos' or 'images' field)."
            )
        features = Features(
            {
                "videos": Sequence(Value("string")),
                "images": Sequence(Value("string")),
                "question": Value("string"),
                "answer": Value("string"),
                "verifier": Value("string"),
                "task_name": Value("string"),
                "load_audio_flag": Value("bool"),
            }
        )
        return Dataset.from_list(rows, features=features)


def format_video_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format VideoDataset into an OpenAI-API-like message log.

    Videos are represented as {"type": "video", "video": path} in the user
    content. The data processor will load the video, extract frames, and
    convert them to images for the model. The number of frames is controlled
    by TaskDataSpec.num_frames (set from data.num_frames in the training config).

    Images are included as {"type": "image", "image": path} as usual.
    """
    user_content = []

    # Add video entries first
    for video_path in example.get("videos", []):
        user_content.append({
            "type": "video",
            "video": video_path,
        })

    # Add image entries
    for image_path in example.get("images", []):
        user_content.append({"type": "image", "image": image_path})

    # Add text question
    question = example["question"]
    for token in ("<image>", "<video>"):
        question = question.replace(token, "").strip()
    # Ensure the question instructs the model to use \boxed{} format,
    # matching the image dataset's unify_answer_format() behavior.
    # Without this, the mmpr_filtered reward function cannot extract answers.
    if "\\boxed{" not in question:
        question = question + "\nPlease put the final answer within \\boxed{...}."
    user_content.append({
        "type": "text",
        "text": question,
    })

    think_flag = "think" if question_asks_for_reasoning(question) else "nothink"
    assistant_content = f"{think_flag}:{example['verifier']}:{example['answer']}"

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": example.get("task_name", "video_dataset"),
        "load_audio_flag": example.get("load_audio_flag", False),
    }

    if _FORMAT_DEBUG:
        _dbg_vids = example.get("videos", [])
        _dbg_imgs = example.get("images", [])
        print(f"[FMT_VIDEO_DEBUG] videos={_dbg_vids}")
        print(f"[FMT_VIDEO_DEBUG] images={_dbg_imgs}")
        print(f"[FMT_VIDEO_DEBUG] question (original first 300)={example['question'][:300]!r}")
        print(f"[FMT_VIDEO_DEBUG] question (cleaned first 300)={question[:300]!r}")
        print(f"[FMT_VIDEO_DEBUG] user_content has {len(user_content)} entries:")
        for _dci, _dc in enumerate(user_content):
            _dctype = _dc.get("type", "?")
            _dcrest = {k: v for k, v in _dc.items() if k != "type"}
            print(f"[FMT_VIDEO_DEBUG]   [{_dci}] type={_dctype} rest={str(_dcrest)[:200]}")

    return ret


# ---------------------------------------------------------------------------
#  Regex for extracting multiple-choice options from question text.
#  Matches patterns like:
#    A. The answer is ...
#    B. Another option ...
#  Options are separated by newlines (or end-of-string).
# ---------------------------------------------------------------------------
_MC_OPTION_RE = re.compile(
    r"(?:^|\n)\s*([A-D])\.\s+(.*?)(?=\n\s*[A-D]\.\s|\Z)",
    re.DOTALL,
)


def format_video_mpo_dataset(
    example: dict[str, Any],
    include_audio: bool = False,
) -> dict[str, Any]:
    """Convert a video multiple-choice QA sample into MPO preference-pair format.

    The function parses answer options A-D embedded in the question text, then
    uses the correct answer as the **chosen** completion (rank 0) and a randomly
    selected incorrect answer as the **rejected** completion (rank 1).

    Args:
        example: Dataset row with videos, question, answer, verifier fields.
        include_audio: If True, also add an audio entry for each video so the
            omni model processes the audio track alongside video frames.

    Returns the canonical MPO dict::

        {
            "context": [{"role": "user", "content": [<video>, <audio>, <text>]}],
            "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": "..."}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": "..."}]},
            ],
        }
    """
    user_content: list[dict[str, Any]] = []

    # Add video entries (and optionally audio extracted from the same video).
    # Video is placed BEFORE audio to match Megatron SFT ordering:
    # SFT cooker uses tags=["vis_video", "vis_sound"] → <image>... then <so_embedding>...
    load_audio_flag = _parse_bool(example.get("load_audio_flag", False))
    for video_path in example.get("videos", []):
        user_content.append({"type": "video", "video": video_path})
        if include_audio and load_audio_flag:
            user_content.append({"type": "audio", "audio": video_path})

    # Add image entries (for mixed image+video data)
    for image_path in example.get("images", []):
        user_content.append({"type": "image", "image": image_path})

    # Clean up question text
    question = example["question"]
    for token in ("<image>", "<video>"):
        question = question.replace(token, "").strip()

    user_content.append({"type": "text", "text": question})

    # --- Build chosen / rejected completions from MCQ options ---------------
    correct_letter = example["answer"].strip().rstrip(".").upper()

    # Try to parse individual options from the question text
    options = {m.group(1): m.group(2).strip() for m in _MC_OPTION_RE.finditer(question)}

    if options and correct_letter in options:
        # Build full-text chosen answer  (e.g. "A. The shift from ...")
        chosen_text = f"{correct_letter}. {options[correct_letter]}"
        # Pick a random wrong option for the rejected answer
        wrong_letters = [k for k in options if k != correct_letter]
        rejected_letter = random.choice(wrong_letters)
        rejected_text = f"{rejected_letter}. {options[rejected_letter]}"
    else:
        # Fallback: cannot parse options – use the raw answer for chosen and
        # a generic wrong marker for rejected.
        chosen_text = example["answer"].strip()
        rejected_text = "I'm not sure."

    return {
        "context": [{"role": "user", "content": user_content}],
        "completions": [
            {
                "rank": 0,
                "completion": [{"role": "assistant", "content": chosen_text}],
            },
            {
                "rank": 1,
                "completion": [{"role": "assistant", "content": rejected_text}],
            },
        ],
    }


class OmniVideoDataset(VideoDataset):
    """Video dataset alias whose rows keep task_name='omni_video_dataset'."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("task_name", "omni_video_dataset")
        super().__init__(*args, **kwargs)
