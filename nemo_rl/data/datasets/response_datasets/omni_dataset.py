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

"""Omni dataset for image/video/audio QA RL training.

Expected JSONL format:
    {
        "videos": ["path/to/video.mp4"],
        "images": ["path/to/image.jpg"],
        "audios": ["path/to/audio.wav"],
        "question": "What is happening in this clip?",
        "answer": "A person is speaking",
        "dataset": "audio_qa",
        "verifier": "string-match",
        "load_audio_flag": true,
        "enable_thinking": true
    }

Rows may contain any combination of images, videos, and audios. Standalone
audio QA is also supported via either ``"audio"`` or ``"audios"``.

For MPO training, rows may optionally provide direct preference completions via
``chosen_response`` / ``rejected_response`` (or ``chosen`` / ``rejected``), or
the canonical ``context`` / ``completions`` preference schema. When neither is
present, MPO falls back to deriving chosen/rejected completions from the MCQ
options in ``question`` and the correct ``answer``.

For video rows, ``load_audio_flag`` preserves the existing behavior of letting
the training pipeline extract the audio track from the video path when desired.
Explicit ``audios`` entries are supported in addition to that mechanism.
"""

import json
import os
import random
import re
from typing import Any, Optional

from datasets import Dataset, Features, Sequence, Value

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets.blend_v1 import (
    get_verifier,
    unify_answer_format,
)
from nemo_rl.data.datasets.response_datasets.video_dataset import VideoDataset
from nemo_rl.data.interfaces import TaskDataSpec

_DATASET_DEBUG = os.environ.get("NRL_DATASET_DEBUG", "0") == "1"
_FORMAT_DEBUG = os.environ.get("NRL_DATASET_FORMAT_DEBUG", "0") == "1"

_MEDIA_PLACEHOLDER_TOKENS = (
    "<image>",
    "<video>",
    "<audio>",
    "<so_embedding>",
    "<so_start>",
    "<so_end>",
)

_MC_OPTION_RE = re.compile(
    r"(?:^|\n)\s*([A-D])\.\s+(.*?)(?=\n\s*[A-D]\.\s|\Z)",
    re.DOTALL,
)

OMNI_FEATURES = Features(
    {
        "videos": Sequence(Value("string")),
        "images": Sequence(Value("string")),
        "audios": Sequence(Value("string")),
        "question": Value("string"),
        "answer": Value("string"),
        "verifier": Value("string"),
        "task_name": Value("string"),
        "load_audio_flag": Value("bool"),
        "enable_thinking": Value("bool"),
        "chosen_response": Value("string"),
        "rejected_response": Value("string"),
        "system": Value("string"),
        "context_json": Value("string"),
        "completions_json": Value("string"),
    }
)


def _parse_bool(value: Any) -> bool:
    """Safely convert a value that may be a bool or a string like ``"false"``."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _normalize_path_list(
    row: dict[str, Any],
    plural_key: str,
    singular_key: str,
) -> list[str]:
    """Normalize either ``key`` or ``keys`` into a list of strings."""
    value = row.get(plural_key)
    if value is None:
        value = row.get(singular_key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _clean_question(question: str, verifier: str) -> str:
    """Strip multimodal placeholder tokens and normalize answer formatting."""
    cleaned = question.strip()
    for token in _MEDIA_PLACEHOLDER_TOKENS:
        cleaned = cleaned.replace(token, "").strip()
    if verifier == "asr":
        # ASR answers are graded on the raw transcript, so no boxed instruction.
        return cleaned
    return unify_answer_format(cleaned)


def _build_mpo_completions(example: dict[str, Any]) -> tuple[str, str]:
    """Build chosen/rejected completions for MPO from a multiple-choice prompt."""
    question = example["question"]
    correct_letter = example["answer"].strip().rstrip(".").upper()
    options = {m.group(1): m.group(2).strip() for m in _MC_OPTION_RE.finditer(question)}

    if options and correct_letter in options:
        chosen_text = f"{correct_letter}. {options[correct_letter]}"
        wrong_letters = [letter for letter in options if letter != correct_letter]
        rejected_letter = random.choice(wrong_letters)
        rejected_text = f"{rejected_letter}. {options[rejected_letter]}"
    else:
        chosen_text = example["answer"].strip()
        rejected_text = "I'm not sure."

    return chosen_text, rejected_text


def _normalize_context_messages(context: Any) -> list[dict[str, Any]]:
    """Normalize raw preference context into message-list form."""
    if isinstance(context, str):
        return [{"role": "user", "content": context}]
    return list(context)


def _normalize_completion_messages(completions: Any) -> list[dict[str, Any]]:
    """Normalize raw preference completions into the canonical MPO structure."""
    normalized: list[dict[str, Any]] = []
    for completion in completions:
        completion_dict = dict(completion)
        completion_messages = completion_dict.get("completion", "")
        if isinstance(completion_messages, str):
            completion_messages = [
                {"role": "assistant", "content": completion_messages}
            ]
        else:
            completion_messages = list(completion_messages)
        completion_dict["completion"] = completion_messages
        normalized.append(completion_dict)
    return normalized


def _build_omni_row(row: dict[str, Any], task_name: str) -> Optional[dict[str, Any]]:
    """Convert a JSONL row into the unified omni schema."""
    videos = _normalize_path_list(row, "videos", "video")
    images = _normalize_path_list(row, "images", "image")
    audios = _normalize_path_list(row, "audios", "audio")
    enable_thinking = _parse_bool(row["enable_thinking"]) if row.get("enable_thinking") is not None else True
    has_preference_pair = row.get("context") is not None and row.get("completions") is not None
    has_text_prompt = row.get("question") is not None

    if not videos and not images and not audios and not has_preference_pair and not has_text_prompt:
        return None

    question = ""
    if row.get("question") is not None:
        question = _clean_question(row["question"], row.get("verifier", "string-match"))

    if images and not videos and "dataset" in row and not has_preference_pair:
        verifier = get_verifier(row)
    else:
        verifier = row.get("verifier", "string-match")

    return {
        "videos": videos,
        "images": images,
        "audios": audios,
        "question": question,
        "answer": row.get("answer", ""),
        "verifier": verifier,
        "task_name": task_name,
        "load_audio_flag": _parse_bool(row.get("load_audio_flag", False)),
        "enable_thinking": enable_thinking,
        "chosen_response": str(
            row.get("chosen_response", row.get("chosen", "")) or ""
        ),
        "rejected_response": str(
            row.get("rejected_response", row.get("rejected", "")) or ""
        ),
        "system": str(row.get("system", "") or ""),
        "context_json": (
            json.dumps(row["context"], ensure_ascii=False)
            if has_preference_pair
            else ""
        ),
        "completions_json": (
            json.dumps(row["completions"], ensure_ascii=False)
            if has_preference_pair
            else ""
        ),
    }


class OmniDataset(VideoDataset):
    """Dataset class for loading image/video/audio QA data from JSONL files."""

    def __init__(
        self,
        train_data_path: Optional[str] = None,
        data_path: Optional[str] = None,
        prompt_file: Optional[str] = None,
        val_size: int = 0,
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.task_name = "omni_dataset"
        path = train_data_path or data_path
        if not path:
            raise ValueError("OmniDataset requires a JSONL path")

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
        RawDataset.set_task_spec(self, data_config)
        self.task_spec.num_frames = data_config.get("num_frames", 8)
        self.task_spec.max_num_tiles = data_config.get("max_num_tiles", None)
        self.task_spec.max_num_patches = data_config.get("max_num_patches", None)
        self.task_spec.use_audio = data_config.get("use_audio", True)
        self.task_spec.max_audio_duration = data_config.get("max_audio_duration", None)
        if _DATASET_DEBUG:
            print(
                f"[OmniDataset] task={self.task_name} num_frames={self.task_spec.num_frames} "
                f"max_num_tiles={self.task_spec.max_num_tiles} "
                f"max_num_patches={self.task_spec.max_num_patches} "
                f"use_audio={self.task_spec.use_audio} "
                f"max_audio_duration={self.task_spec.max_audio_duration}"
            )

    def _load_jsonl(self, path: str) -> Dataset:
        """Load a JSONL with image/video/audio paths and QA into a Dataset."""
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                parsed_row = _build_omni_row(row, task_name=self.task_name)
                if parsed_row is not None:
                    rows.append(parsed_row)

        if not rows:
            raise ValueError(
                f"No valid rows loaded from {path}. "
                "Each row must contain either multimodal media fields "
                "('videos', 'images', or 'audios') or a canonical "
                "'context'/'completions' preference pair."
            )

        return Dataset.from_list(rows, features=OMNI_FEATURES)


def format_omni_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format OmniDataset into an OpenAI-API-like message log."""
    user_content = []
    verifier = example["verifier"]

    for video_path in example.get("videos", []):
        user_content.append({"type": "video", "video": video_path})

    for image_path in example.get("images", []):
        user_content.append({"type": "image", "image": image_path})

    for audio_path in example.get("audios", []):
        user_content.append({"type": "audio", "audio": audio_path})

    question = example["question"]
    for token in _MEDIA_PLACEHOLDER_TOKENS:
        question = question.replace(token, "").strip()
    if verifier != "asr" and "\\boxed{" not in question:
        question = question + "\nPlease put the final answer within \\boxed{...}."

    user_content.append({"type": "text", "text": question})

    enable_thinking = example.get("enable_thinking", True)
    think_flag = "think" if enable_thinking else "nothink"
    assistant_content = f"{think_flag}:{verifier}:{example['answer']}"

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": example.get("task_name", "omni_dataset"),
        "load_audio_flag": example.get("load_audio_flag", False),
        "enable_thinking": enable_thinking,
    }

    if _FORMAT_DEBUG:
        print(f"[FMT_OMNI_DEBUG] videos={example.get('videos', [])}")
        print(f"[FMT_OMNI_DEBUG] images={example.get('images', [])}")
        print(f"[FMT_OMNI_DEBUG] audios={example.get('audios', [])}")
        print(
            f"[FMT_OMNI_DEBUG] question (original first 300)={example['question'][:300]!r}"
        )
        print(f"[FMT_OMNI_DEBUG] question (cleaned first 300)={question[:300]!r}")
        print(f"[FMT_OMNI_DEBUG] user_content has {len(user_content)} entries:")
        for content_idx, content_item in enumerate(user_content):
            content_type = content_item.get("type", "?")
            content_rest = {
                key: value for key, value in content_item.items() if key != "type"
            }
            print(
                f"[FMT_OMNI_DEBUG]   [{content_idx}] type={content_type} "
                f"rest={str(content_rest)[:200]}"
            )
        print(
            f"[FMT_OMNI_DEBUG] enable_thinking={enable_thinking!r}"
        )

    return ret


def format_omni_mpo_dataset(
    example: dict[str, Any],
    include_audio: bool = False,
) -> dict[str, Any]:
    """Convert an omni sample into MPO preference-pair format.

    Preference pairs are resolved in this order:
    1. Use canonical ``context`` / ``completions`` if present.
    2. Use direct ``chosen_response`` / ``rejected_response`` text if present.
    3. Otherwise derive chosen/rejected completions from MCQ options.
    """
    if example.get("context_json") and example.get("completions_json"):
        return {
            "context": _normalize_context_messages(
                json.loads(example["context_json"])
            ),
            "completions": _normalize_completion_messages(
                json.loads(example["completions_json"])
            ),
        }

    user_content: list[dict[str, Any]] = []

    for video_path in example.get("videos", []):
        user_content.append({"type": "video", "video": video_path})
        if include_audio and example.get("load_audio_flag", False):
            user_content.append({"type": "audio", "audio": video_path})

    for image_path in example.get("images", []):
        user_content.append({"type": "image", "image": image_path})

    for audio_path in example.get("audios", []):
        user_content.append({"type": "audio", "audio": audio_path})

    question = example["question"]
    for token in _MEDIA_PLACEHOLDER_TOKENS:
        question = question.replace(token, "").strip()
    user_content.append({"type": "text", "text": question})

    context_messages: list[dict[str, Any]] = []
    if example.get("system"):
        context_messages.append({"role": "system", "content": example["system"]})
    context_messages.append({"role": "user", "content": user_content})

    chosen_text = example.get("chosen_response", "").strip()
    rejected_text = example.get("rejected_response", "").strip()
    if chosen_text or rejected_text:
        if not chosen_text or not rejected_text:
            raise ValueError(
                "Omni MPO samples with direct completions must provide both "
                "chosen_response and rejected_response."
            )
    else:
        chosen_text, rejected_text = _build_mpo_completions(example)

    return {
        "context": context_messages,
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
