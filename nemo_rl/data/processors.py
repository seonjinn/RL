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

"""Contains data processors for evaluation."""

import json
import logging
import math
import os
from typing import Any, Dict, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    PreferenceDatumSpec,
    TaskDataProcessFnCallable,
    TaskDataSpec,
    VLMMessageLogType,
)
from nemo_rl.data.llm_message_utils import get_formatted_message_log

TokenizerType = PreTrainedTokenizerBase
logger = logging.getLogger(__name__)
_DERIVE_VLLM_MAX_NUM_PATCHES_ENV = "NEMO_RL_VLLM_DERIVE_MAX_NUM_PATCHES"


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def helpsteer3_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a HelpSteer3 preference datum into a DatumSpec for GRPO training.

    This function converts HelpSteer3 preference data to work with GRPO by:
    1. Using the context as the prompt
    2. Using the preferred completion as the target response
    3. Creating a reward signal based on preference scores
    """
    # Extract context and completions from HelpSteer3 format
    context = datum_dict["context"]
    preferred_completion = datum_dict["response"]

    # Build the conversation from context
    message_log: LLMMessageLogType = []

    # Add context messages
    if isinstance(context, list):
        for msg in context:
            message_log.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )
    else:
        # If context is a string, treat it as a user message
        message_log.append(
            {
                "role": "user",
                "content": context,
            }
        )

    # Add the preferred completion as the target
    for completion_msg in preferred_completion:
        message_log.append(
            {
                "role": completion_msg["role"],
                "content": completion_msg["content"],
            }
        )

    # Apply chat template and tokenize
    formatted_conversation = tokenizer.apply_chat_template(
        message_log,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=True,
    )

    # Tokenize the entire conversation
    full_tokens = tokenizer(
        formatted_conversation,
        return_tensors="pt",
        add_special_tokens=False,  # Already added by chat template
    )["input_ids"][0]

    # For simplicity, assign all tokens to the first message
    # In a more sophisticated implementation, you might want to split tokens properly
    message_log[0]["token_ids"] = full_tokens
    message_log[0]["content"] = formatted_conversation

    # Clear token_ids for other messages to avoid double counting
    for i in range(1, len(message_log)):
        message_log[i]["token_ids"] = tokenizer("", return_tensors="pt")["input_ids"][
            0
        ]  # Empty tensor

    length = sum(len(m["token_ids"]) for m in message_log)

    # Create ground truth from the preferred completion for environment evaluation
    ground_truth = " ".join([msg["content"] for msg in preferred_completion])
    extra_env_info = {"ground_truth": ground_truth}

    loss_multiplier = 1.0
    if length > max_seq_length:
        # Truncate if too long
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(
                    max_seq_length // len(message_log), len(chat_message["token_ids"])
                )
            ]
        loss_multiplier = 0.0  # Reduce loss for truncated sequences

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def sft_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""
    # optional preprocessor
    if datum_dict["task_name"] == "clevr-cogent":
        from nemo_rl.data.datasets.response_datasets.clevr import (
            format_clevr_cogent_dataset,
        )

        datum_dict = format_clevr_cogent_dataset(datum_dict)

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
        tools=datum_dict.get("tools", None),  # Pass tools from data if present
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def preference_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> PreferenceDatumSpec:
    """Process a datum dictionary for RM/DPO training.

    Examples:
        ```{doctest}
        >>> from transformers import AutoTokenizer
        >>> from nemo_rl.data.interfaces import TaskDataSpec
        >>> from nemo_rl.data.processors import preference_preprocessor
        >>>
        >>> # Initialize tokenizer and task spec
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        >>> ## set a passthrough chat template for simplicity
        >>> tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        >>> task_spec = TaskDataSpec(task_name="test_preference")
        >>>
        >>> datum = {
        ...     "context": [{"role": "user", "content": "What is 2+2?"}],
        ...     "completions": [
        ...         {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
        ...         {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
        ...     ]
        ... }
        >>>
        >>> processed = preference_preprocessor(datum, task_spec, tokenizer, max_seq_length=128, idx=0)  # doctest: +ELLIPSIS
        <BLANKLINE>
        ...
        >>> len(processed["message_log_chosen"])
        2
        >>> processed["message_log_chosen"][0]["content"]
        '<|begin_of_text|>What is 2+2?'
        >>> processed["message_log_chosen"][-1]["content"]
        '4<|eot_id|>'
        >>> processed["message_log_rejected"][-1]["content"]
        '5<|eot_id|>'
        >>>
        >>> # context can also contain multiple turns
        >>> datum = {
        ...     "context": [{"role": "user", "content": "I have a question."}, {"role": "assistant", "content": "Sure!"}, {"role": "user", "content": "What is 2+2?"}],
        ...     "completions": [
        ...         {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
        ...         {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
        ...     ]
        ... }
        >>> processed = preference_preprocessor(datum, task_spec, tokenizer, max_seq_length=128, idx=0)
        >>> len(processed["message_log_chosen"])
        4
        >>> processed["message_log_chosen"][1]["content"]
        'Sure!'
        >>> processed["message_log_chosen"][-1]["content"]
        '4<|eot_id|>'
        >>> processed["message_log_rejected"][-1]["content"]
        '5<|eot_id|>'

        ```
    """
    assert len(datum_dict["completions"]) == 2, (
        "RM/DPO training supports only two completions"
    )
    # Lower rank is preferred
    if datum_dict["completions"][0]["rank"] < datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][0]
        rejected_completion = datum_dict["completions"][1]
    elif datum_dict["completions"][0]["rank"] > datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][1]
        rejected_completion = datum_dict["completions"][0]
    else:
        raise NotImplementedError(
            "Ties are not supported yet. You can use the following command to filter out ties: `cat <PathToPreferenceDataset> | jq 'select(.completions[0].rank != .completions[1].rank)'`."
        )

    messages_chosen = datum_dict["context"] + chosen_completion["completion"]
    messages_rejected = datum_dict["context"] + rejected_completion["completion"]

    message_log_chosen = get_formatted_message_log(
        messages_chosen, tokenizer, task_data_spec
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, tokenizer, task_data_spec
    )

    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        logging.warning(
            f"Sequence length {max(length_chosen, length_rejected)} exceeds max_seq_length {max_seq_length}. Ignoring example."
        )

        # make smaller and mask out
        for message in message_log_chosen:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
        for message in message_log_rejected:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
        loss_multiplier = 0.0

        length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
        length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

        # safeguard against edge case where there are too many turns to fit within the max length
        assert max(length_chosen, length_rejected) <= max_seq_length

    output: PreferenceDatumSpec = {
        "message_log_chosen": message_log_chosen,
        "message_log_rejected": message_log_rejected,
        "length_chosen": length_chosen,
        "length_rejected": length_rejected,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


# Example of a generic math data processor
def math_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["expected_answer"])
    extra_env_info = {"ground_truth": solution}

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        problem = task_data_spec.prompt.format(problem)
    user_message = {"role": "user", "content": problem}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for indiv_message in message_log:
            indiv_message["token_ids"] = indiv_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def math_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Reward Model Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []
    formatted_content = (
        task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    )
    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


_MEDIA_PLACEHOLDER_TOKENS = (
    "<image>",
    "<video>",
    "<audio>",
    "<so_embedding>",
    "<so_start>",
    "<so_end>",
)

_VIDEO_PROMPT_STYLE_ENV = "NRL_VIDEO_PROMPT_STYLE"
_VIDEO_PROMPT_STYLE_SFT_V2_GROUPED = "sft_v2_grouped"
_VIDEO_PROMPT_STYLE_DEFAULT = _VIDEO_PROMPT_STYLE_SFT_V2_GROUPED
_SUPPORTED_VIDEO_PROMPT_STYLES = {
    _VIDEO_PROMPT_STYLE_SFT_V2_GROUPED,
}


def _get_video_prompt_style() -> str:
    style = os.environ.get(_VIDEO_PROMPT_STYLE_ENV, _VIDEO_PROMPT_STYLE_DEFAULT)
    style = style.strip().lower()
    if style not in _SUPPORTED_VIDEO_PROMPT_STYLES:
        supported = ", ".join(sorted(_SUPPORTED_VIDEO_PROMPT_STYLES))
        raise ValueError(
            f"Unsupported {_VIDEO_PROMPT_STYLE_ENV}={style!r}; supported: {supported}"
        )
    return style


def _timestamps_from_video_metadata(
    metadata: dict[str, Any] | None,
    num_frames: int,
) -> list[float]:
    if num_frames <= 0:
        return []
    if metadata is None:
        return [float(i) for i in range(num_frames)]

    frames_indices = metadata.get("frames_indices")
    if torch.is_tensor(frames_indices):
        frames_indices = frames_indices.tolist()
    elif hasattr(frames_indices, "tolist") and not isinstance(
        frames_indices, (list, tuple)
    ):
        frames_indices = frames_indices.tolist()

    fps = float(metadata.get("fps") or 0.0)
    if isinstance(frames_indices, (list, tuple)) and fps > 0:
        frame_duration_ms = int(1000 / fps)
        return [
            int(frames_indices[i]) * frame_duration_ms / 1000.0
            for i in range(min(num_frames, len(frames_indices)))
        ]

    duration = float(metadata.get("duration") or 0.0)
    if duration > 0:
        if num_frames == 1:
            return [duration / 2.0]
        effective_span = max(duration - 1.0, 0.0)
        segment_size = effective_span / num_frames
        return [segment_size * (i + 0.5) for i in range(num_frames)]

    return [float(i) for i in range(num_frames)]


def _format_sft_v2_grouped_video_line(
    frame_start_index: int,
    timestamps: list[float],
) -> str:
    group_frames = []
    for offset, timestamp in enumerate(timestamps):
        frame_number = frame_start_index + offset + 1
        frame_label = "Frame" if offset == 0 else "frame"
        group_frames.append(
            f"{frame_label} {frame_number} sampled at {timestamp:.2f} seconds"
        )
    return " and ".join(group_frames) + ": "


def _append_sft_v2_grouped_video_content(
    *,
    user_content: list[dict[str, Any]],
    vllm_content: list[dict[str, Any]],
    video_path: str,
    frames: list[Image.Image],
    timestamps: list[float],
    temporal_patch_size: int,
) -> None:
    """Append SFT-v2 grouped video prompt text for policy and native-vLLM paths."""
    temporal_patch_size = max(1, int(temporal_patch_size))
    user_content.append({"type": "text", "text": "This is a video:\n"})
    vllm_content.append({"type": "text", "text": "This is a video:"})
    vllm_content.append({"type": "video", "video": video_path})

    for frame_start in range(0, len(frames), temporal_patch_size):
        group_frames = frames[frame_start : frame_start + temporal_patch_size]
        group_timestamps = timestamps[frame_start : frame_start + len(group_frames)]
        if len(group_timestamps) < len(group_frames):
            group_timestamps.extend(
                float(frame_start + i)
                for i in range(len(group_timestamps), len(group_frames))
            )
        user_content.append(
            {
                "type": "text",
                "text": _format_sft_v2_grouped_video_line(
                    frame_start, group_timestamps
                ),
            }
        )
        for frame in group_frames:
            user_content.append(
                {"type": "image", "image": frame, "_is_video_frame": True}
            )
        user_content.append({"type": "text", "text": "\n"})


def _video_metadata_frame_indices(metadata: dict[str, Any] | None) -> list[int]:
    if metadata is None:
        return []
    frames_indices = metadata.get("frames_indices")
    if torch.is_tensor(frames_indices):
        frames_indices = frames_indices.tolist()
    elif hasattr(frames_indices, "tolist") and not isinstance(
        frames_indices, (list, tuple)
    ):
        frames_indices = frames_indices.tolist()
    if not isinstance(frames_indices, (list, tuple)):
        return []
    return [int(index) for index in frames_indices]


def _collapse_video_frame_token_wrappers(
    user_ids: torch.Tensor,
    *,
    video_flags: list[bool],
    temporal_patch_size: int,
    img_start_id: int | None,
    img_end_id: int | None,
) -> torch.Tensor:
    if temporal_patch_size <= 1 or img_start_id is None or img_end_id is None:
        return user_ids

    keep = torch.ones(len(user_ids), dtype=torch.bool, device=user_ids.device)
    image_idx = 0
    video_frame_idx = 0
    for start in (user_ids == img_start_id).nonzero(as_tuple=True)[0]:
        end_matches = (user_ids[start:] == img_end_id).nonzero(as_tuple=True)[0]
        if len(end_matches) == 0:
            break
        end = start + end_matches[0].item()
        is_video_frame = image_idx < len(video_flags) and video_flags[image_idx]
        if is_video_frame:
            if video_frame_idx % temporal_patch_size != 0:
                keep[start : end + 1] = False
            video_frame_idx += 1
        else:
            video_frame_idx = 0
        image_idx += 1
    return user_ids[keep]


def _strip_media_tokens_from_text(text: str) -> str:
    for token in _MEDIA_PLACEHOLDER_TOKENS:
        text = text.replace(token, "")
    return text


def _flatten_multimodal_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    if isinstance(content, str):
        return message

    parts: list[str] = []
    for item in content:
        item_type = item.get("type")
        if item_type == "text":
            parts.append(item.get("text", ""))
        elif item_type == "image":
            parts.append("<image>")
        elif item_type == "video":
            parts.append("<video>")
        elif item_type == "audio":
            parts.append("<so_embedding>")
    return {**message, "content": "\n".join(part for part in parts if part)}


def _message_for_chat_template(
    processor: AutoProcessor,
    message: dict[str, Any],
    *,
    force_flatten: bool = False,
) -> dict[str, Any]:
    if not force_flatten and hasattr(processor, "conversation_preprocessor"):
        return processor.conversation_preprocessor(message)
    return _flatten_multimodal_message(message)


def _apply_chat_template_compat(processor: AutoProcessor, messages: list[dict[str, Any]], **kwargs):
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        kwargs = dict(kwargs)
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def _get_audio_duration_seconds(audio_path: str) -> float:
    ext = os.path.splitext(audio_path)[1].lower()
    video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".ts"}

    if ext not in video_extensions:
        try:
            import soundfile as sf

            info = sf.info(audio_path)
            if info.duration > 0:
                return float(info.duration)
        except Exception:
            pass

    try:
        from decord import VideoReader
        from decord import cpu as decord_cpu

        vr = VideoReader(audio_path, ctx=decord_cpu(), num_threads=1)
        fps = vr.get_avg_fps()
        if fps > 0 and len(vr) > 0:
            return float(len(vr) / fps)
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(audio_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and n_frames > 0:
                return float(n_frames / fps)
        cap.release()
    except Exception:
        pass

    return 30.0


def _derive_vllm_max_num_patches(
    message: dict[str, Any],
    processor: AutoProcessor,
    task_data_spec: TaskDataSpec,
    mm_kwargs: dict[str, Any],
) -> int | None:
    explicit_max_num_patches = task_data_spec.max_num_patches or mm_kwargs.get(
        "max_num_patches"
    )
    if explicit_max_num_patches is not None:
        return int(explicit_max_num_patches)

    if _env_enabled(_DERIVE_VLLM_MAX_NUM_PATCHES_ENV, "0") and "imgs_sizes" in message:
        patch_size = int(getattr(processor, "patch_size", 16))
        sizes = (
            message["imgs_sizes"].tolist()
            if hasattr(message["imgs_sizes"], "tolist")
            else list(message["imgs_sizes"])
        )
        if sizes:
            return max((int(h) // patch_size) * (int(w) // patch_size) for h, w in sizes)
    return None


def _compute_vision_expansion_from_processor(
    message: dict[str, Any],
    processor: AutoProcessor,
    media_num_frames: list[int],
    num_post_collapse_placeholders: int,
    temporal_patch_size: int,
) -> int:
    if "imgs_sizes" not in message or num_post_collapse_placeholders == 0:
        return 0

    imgs_sizes = message["imgs_sizes"]
    sizes = imgs_sizes.tolist() if hasattr(imgs_sizes, "tolist") else list(imgs_sizes)
    if not sizes:
        return 0

    if hasattr(processor, "compute_num_embeddings"):
        per_frame_embeds = [
            int(processor.compute_num_embeddings(int(h), int(w))) for h, w in sizes
        ]
    else:
        patch_size = int(getattr(processor, "patch_size", 16))
        downsample_ratio = float(getattr(processor, "downsample_ratio", 0.5))
        reduction_factor = max(1, int(round(1 / downsample_ratio)))
        per_frame_embeds = [
            (int(h) // patch_size) * (int(w) // patch_size) // (reduction_factor**2)
            for h, w in sizes
        ]

    if temporal_patch_size > 1 and media_num_frames:
        total_embeds = 0
        offset = 0
        for num_frames in media_num_frames:
            frame_embeds = per_frame_embeds[offset : offset + num_frames]
            offset += num_frames
            if not frame_embeds:
                continue
            if num_frames > 1:
                per_tubelet = sum(frame_embeds) // max(num_frames, 1)
                total_embeds += per_tubelet * math.ceil(num_frames / temporal_patch_size)
            else:
                total_embeds += sum(frame_embeds)
        if offset < len(per_frame_embeds):
            total_embeds += sum(per_frame_embeds[offset:])
    else:
        total_embeds = sum(per_frame_embeds)

    return max(0, int(total_embeds) - int(num_post_collapse_placeholders))


def _extract_text_from_content_items(content_items: list[dict[str, Any]]) -> str:
    text_parts = [
        str(content.get("text", ""))
        for content in content_items
        if content.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part).strip()


def _build_masked_vlm_datum(
    *,
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int | None,
    idx: int,
    task_name: str,
    extra_env_info: dict[str, Any],
    content_items: list[dict[str, Any]],
    reason: str,
    exc: Exception,
) -> DatumSpec:
    """Return a tiny masked DatumSpec for bad media rows."""
    fallback_text = _extract_text_from_content_items(content_items)
    if task_data_spec.prompt and fallback_text:
        fallback_text = task_data_spec.prompt.format(fallback_text)
    if not fallback_text:
        fallback_text = "Skipped invalid multimodal sample."

    chat_template_kwargs: dict[str, Any] = {}
    if "enable_thinking" in datum_dict:
        chat_template_kwargs["enable_thinking"] = bool(datum_dict["enable_thinking"])

    system_message: dict[str, Any] = {
        "role": "system",
        "content": task_data_spec.system_prompt or "",
    }
    user_message: dict[str, Any] = {"role": "user", "content": fallback_text}
    system_for_chat = _message_for_chat_template(processor, system_message, force_flatten=True)
    user_for_chat = _message_for_chat_template(processor, user_message, force_flatten=True)

    system_only: dict[str, Any] = _apply_chat_template_compat(
        processor,
        [system_for_chat],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        **chat_template_kwargs,
    )
    message_both: dict[str, Any] = _apply_chat_template_compat(
        processor,
        [system_for_chat, user_for_chat],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **chat_template_kwargs,
    )
    sys_len = system_only["input_ids"].shape[1]
    system_token_ids = message_both["input_ids"][0][:sys_len]
    user_token_ids = message_both["input_ids"][0][sys_len:]

    if max_seq_length is not None:
        token_limit = max(1, min(16, max_seq_length // 2))
        system_token_ids = system_token_ids[:token_limit]
        user_token_ids = user_token_ids[:token_limit]

    if user_token_ids.numel() == 0:
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        token_id = eos_token_id if eos_token_id is not None else pad_token_id
        if token_id is None:
            token_id = 0
        user_token_ids = torch.tensor([int(token_id)], dtype=torch.int64)

    message_log: VLMMessageLogType = [
        {**system_message, "token_ids": system_token_ids},
        {**user_message, "token_ids": user_token_ids},
    ]
    length = sum(len(message["token_ids"]) for message in message_log)
    logger.warning(
        "Masking VLM sample idx=%s task=%s after media load failure (%s): %s",
        idx,
        task_name,
        reason,
        exc,
    )

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 0.0,
        "idx": idx,
        "task_name": task_name,
        "vision_expansion": 0,
        "collapse_savings": 0,
        "vllm_content": None,
        "vllm_images": [],
        "vllm_videos": [],
        "vllm_num_frames": task_data_spec.num_frames,
        "vllm_temporal_patch_size": task_data_spec.video_temporal_patch_size,
        "vllm_video_prompt_style": _get_video_prompt_style(),
        "vllm_video_frame_indices": [],
        "vllm_video_fps": [],
        "vllm_audio_paths": [],
        "vllm_audio_waveforms": [],
        "vllm_max_audio_duration": task_data_spec.max_audio_duration,
        "vllm_max_num_tiles": task_data_spec.max_num_tiles
        if task_data_spec.use_tiling
        else None,
        "vllm_max_num_patches": task_data_spec.max_num_patches,
    }


def vlm_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process image/video/audio/text VLM rows into a GRPO DatumSpec."""
    from nemo_rl.data.datasets.response_datasets.blend import format_blend_dataset
    from nemo_rl.data.datasets.response_datasets.blend_v1 import (
        format_blend_v1_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.clevr import (
        format_clevr_cogent_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.geometry3k import (
        format_geometry3k_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.mmpr_tiny import (
        format_mmpr_tiny_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.omni_dataset import (
        format_omni_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.refcoco import format_refcoco_dataset
    from nemo_rl.data.datasets.response_datasets.video_dataset import (
        format_video_dataset,
    )
    from nemo_rl.data.multimodal_utils import (
        PackedTensor,
        get_dim_to_pack_along,
        get_multimodal_keys_from_processor,
        resolve_to_image,
    )
    from nemo_rl.models.generation.vllm.utils import (
        load_audio_waveform,
        load_video_frames_with_metadata,
    )

    task_name = task_data_spec.task_name or datum_dict.get("task_name")
    if task_name == "clevr-cogent":
        datum_dict = format_clevr_cogent_dataset(datum_dict)
    elif task_name == "refcoco":
        datum_dict = format_refcoco_dataset(datum_dict)
    elif task_name == "geometry3k":
        datum_dict = format_geometry3k_dataset(datum_dict)
    elif task_name == "mmpr_tiny":
        datum_dict = format_mmpr_tiny_dataset(datum_dict)
    elif task_name == "blend":
        datum_dict = format_blend_dataset(datum_dict)
    elif task_name == "blend_v1":
        datum_dict = format_blend_v1_dataset(datum_dict)
    elif task_name in ("video_dataset", "omni_video_dataset"):
        datum_dict = format_video_dataset(datum_dict)
    elif task_name == "omni_dataset":
        datum_dict = format_omni_dataset(datum_dict)
    else:
        raise ValueError(f"No data processor for task {task_name}")

    raw_messages = datum_dict["messages"]
    problem = raw_messages[0]["content"]
    extra_env_info = {"ground_truth": raw_messages[1]["content"]}
    sample_load_audio = bool(datum_dict.get("load_audio_flag", False))
    video_prompt_style = _get_video_prompt_style()

    chat_template_kwargs: dict[str, Any] = {}
    if "enable_thinking" in datum_dict:
        chat_template_kwargs["enable_thinking"] = bool(datum_dict["enable_thinking"])

    system_message: dict[str, Any] = {
        "role": "system",
        "content": task_data_spec.system_prompt or "",
    }
    user_message: dict[str, Any] = {"role": "user", "content": []}
    vllm_user_message: dict[str, Any] = {"role": "user", "content": []}

    image_paths: list[Any] = []
    video_paths: list[str] = []
    video_frame_indices: list[list[int]] = []
    video_fps: list[float] = []
    media_num_frames: list[int] = []

    content_items = problem if isinstance(problem, list) else [{"type": "text", "text": str(problem)}]
    for content in content_items:
        content_type = content.get("type")
        if content_type == "image":
            image_ref = content["image"]
            image_paths.append(image_ref)
            media_num_frames.append(1)
            user_message["content"].append({"type": "image", "image": image_ref})
            vllm_user_message["content"].append({"type": "image", "image": image_ref})
        elif content_type == "video":
            video_path = content["video"]
            video_paths.append(video_path)
            want_audio = task_data_spec.use_audio and sample_load_audio
            cached_waveform = None
            cached_duration = 0.0
            video_metadata = None
            if want_audio:
                try:
                    frames_np, video_metadata = load_video_frames_with_metadata(
                        video_path,
                        num_frames=task_data_spec.num_frames,
                        temporal_patch_size=task_data_spec.video_temporal_patch_size,
                    )
                    cached_waveform = load_audio_waveform(
                        video_path,
                        target_sr=16000,
                        max_duration=task_data_spec.max_audio_duration,
                        force_pyav=True,
                    )
                    cached_duration = (
                        0.0
                        if cached_waveform is None
                        else len(cached_waveform) / 16000.0
                    )
                except Exception as exc:
                    return _build_masked_vlm_datum(
                        datum_dict=datum_dict,
                        task_data_spec=task_data_spec,
                        processor=processor,
                        max_seq_length=max_seq_length,
                        idx=idx,
                        task_name=task_name,
                        extra_env_info=extra_env_info,
                        content_items=content_items,
                        reason=f"video+audio:{video_path}",
                        exc=exc,
                    )
            else:
                try:
                    frames_np, video_metadata = load_video_frames_with_metadata(
                        video_path,
                        num_frames=task_data_spec.num_frames,
                        temporal_patch_size=task_data_spec.video_temporal_patch_size,
                    )
                except Exception as exc:
                    return _build_masked_vlm_datum(
                        datum_dict=datum_dict,
                        task_data_spec=task_data_spec,
                        processor=processor,
                        max_seq_length=max_seq_length,
                        idx=idx,
                        task_name=task_name,
                        extra_env_info=extra_env_info,
                        content_items=content_items,
                        reason=f"video:{video_path}",
                        exc=exc,
                    )
            frames = [Image.fromarray(frame) for frame in frames_np]
            media_num_frames.append(len(frames))
            video_frame_indices.append(_video_metadata_frame_indices(video_metadata))
            video_fps.append(float((video_metadata or {}).get("fps") or 0.0))
            _append_sft_v2_grouped_video_content(
                user_content=user_message["content"],
                vllm_content=vllm_user_message["content"],
                video_path=video_path,
                frames=frames,
                timestamps=_timestamps_from_video_metadata(
                    video_metadata, len(frames)
                ),
                temporal_patch_size=task_data_spec.video_temporal_patch_size,
            )
            if want_audio:
                user_message["content"].append(
                    {
                        "type": "audio",
                        "audio": video_path,
                        "_cached_waveform": cached_waveform,
                        "_cached_audio_duration": cached_duration,
                    }
                )
                vllm_user_message["content"].append(
                    {"type": "audio", "audio": video_path}
                )
        elif content_type == "audio":
            audio_path = content["audio"]
            try:
                cached_waveform = load_audio_waveform(
                    audio_path,
                    target_sr=16000,
                    max_duration=task_data_spec.max_audio_duration,
                    raise_on_failure=True,
                )
            except Exception as exc:
                return _build_masked_vlm_datum(
                    datum_dict=datum_dict,
                    task_data_spec=task_data_spec,
                    processor=processor,
                    max_seq_length=max_seq_length,
                    idx=idx,
                    task_name=task_name,
                    extra_env_info=extra_env_info,
                    content_items=content_items,
                    reason=f"audio:{audio_path}",
                    exc=exc,
                )
            cached_duration = (
                float(cached_waveform.shape[-1] / 16000) if cached_waveform is not None else 0.0
            )
            user_message["content"].append(
                {
                    "type": "audio",
                    "audio": audio_path,
                    "_cached_waveform": cached_waveform,
                    "_cached_audio_duration": cached_duration,
                }
            )
            vllm_user_message["content"].append({"type": "audio", "audio": audio_path})
        elif content_type == "text":
            text = _strip_media_tokens_from_text(str(content.get("text", "")))
            text_content = {
                "type": "text",
                "text": task_data_spec.prompt.format(text)
                if task_data_spec.prompt
                else text,
            }
            user_message["content"].append(text_content)
            vllm_user_message["content"].append(text_content)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    native_video_prompt = bool(video_paths)
    system_message_for_chat = _message_for_chat_template(
        processor, system_message, force_flatten=native_video_prompt
    )
    user_message_for_chat = _message_for_chat_template(
        processor, vllm_user_message, force_flatten=native_video_prompt
    )

    string_formatted_dialog = _apply_chat_template_compat(
        processor,
        [system_message_for_chat, user_message_for_chat],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )

    audio_paths: list[str] = []
    cached_audio_waveforms: list[Any] = []
    cached_audio_durations: list[float] = []
    content_without_audio: list[dict[str, Any]] = []
    for content in user_message["content"]:
        if content["type"] == "audio":
            audio_paths.append(content["audio"])
            cached_audio_waveforms.append(content.get("_cached_waveform"))
            cached_audio_durations.append(float(content.get("_cached_audio_duration", 0.0) or 0.0))
            content_without_audio.append({"type": "text", "text": "<so_embedding>"})
        else:
            content_without_audio.append(content)

    resolved_processor_content: list[dict[str, Any]] = []
    for content in content_without_audio:
        if content["type"] != "image":
            resolved_processor_content.append(content)
            continue
        try:
            resolved_image = resolve_to_image(content["image"])
        except Exception as exc:
            return _build_masked_vlm_datum(
                datum_dict=datum_dict,
                task_data_spec=task_data_spec,
                processor=processor,
                max_seq_length=max_seq_length,
                idx=idx,
                task_name=task_name,
                extra_env_info=extra_env_info,
                content_items=content_items,
                reason=f"image:{content['image']}",
                exc=exc,
            )
        resolved_processor_content.append({**content, "image": resolved_image})

    user_message_for_processor = {
        "role": "user",
        "content": resolved_processor_content,
    }
    system_message_for_processor = _flatten_multimodal_message(system_message)

    mm_kwargs: dict[str, Any] = {}
    if task_data_spec.use_tiling and task_data_spec.max_num_tiles is not None:
        mm_kwargs["max_num_tiles"] = task_data_spec.max_num_tiles
    if task_data_spec.max_num_patches is not None:
        mm_kwargs["max_num_patches"] = task_data_spec.max_num_patches

    video_flags = [
        bool(content.get("_is_video_frame", False))
        for content in user_message_for_processor["content"]
        if content.get("type") == "image"
    ]
    if any(video_flags):
        mm_kwargs["video_flags"] = video_flags
        mm_kwargs["video_temporal_patch_size"] = task_data_spec.video_temporal_patch_size
        if task_data_spec.video_target_num_patches is not None:
            mm_kwargs["video_target_num_patches"] = task_data_spec.video_target_num_patches
        mm_kwargs["video_maintain_aspect_ratio"] = task_data_spec.video_maintain_aspect_ratio

    exact_audio_tokens = 0
    if audio_paths and hasattr(processor, "estimate_audio_tokens"):
        for audio_index, audio_path in enumerate(audio_paths):
            cached_duration = cached_audio_durations[audio_index] if audio_index < len(cached_audio_durations) else 0.0
            audio_duration = cached_duration if cached_duration > 0 else _get_audio_duration_seconds(audio_path)
            exact_audio_tokens += int(
                processor.estimate_audio_tokens(
                    audio_duration,
                    max_duration=task_data_spec.max_audio_duration,
                )
            ) + 2

    if "max_num_patches" not in mm_kwargs:
        text_estimate = 200
        mm_kwargs["num_tokens_available"] = max(
            1024,
            max_seq_length
            - exact_audio_tokens
            - text_estimate
            - task_data_spec.min_generation_tokens,
        )

    orig_max_num_tiles = None
    if (
        mm_kwargs.get("max_num_tiles") is not None
        and hasattr(processor, "image_processor")
        and hasattr(processor.image_processor, "max_num_tiles")
    ):
        orig_max_num_tiles = processor.image_processor.max_num_tiles
        processor.image_processor.max_num_tiles = task_data_spec.max_num_tiles

    system_only: dict[str, Any] = _apply_chat_template_compat(
        processor,
        [system_message_for_processor],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        **chat_template_kwargs,
    )
    message_both: dict[str, Any] = _apply_chat_template_compat(
        processor,
        [system_message_for_processor, user_message_for_processor],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **chat_template_kwargs,
        **mm_kwargs,
    )
    if orig_max_num_tiles is not None:
        processor.image_processor.max_num_tiles = orig_max_num_tiles

    if audio_paths:
        if not hasattr(processor, "expand_audio_tokens"):
            raise ValueError(
                f"Audio inputs are present for task {task_name}, but processor "
                f"{type(processor).__name__} does not implement expand_audio_tokens()."
            )
        waveforms_to_pass = (
            cached_audio_waveforms
            if any(waveform is not None for waveform in cached_audio_waveforms)
            else None
        )
        try:
            message_both = processor.expand_audio_tokens(
                message_both,
                audio_paths,
                audio_waveforms=waveforms_to_pass,
                max_audio_duration=task_data_spec.max_audio_duration,
                sound_clip_duration=task_data_spec.sound_clip_duration,
                sound_clip_min_duration=task_data_spec.sound_clip_min_duration,
            )
        except Exception as exc:
            return _build_masked_vlm_datum(
                datum_dict=datum_dict,
                task_data_spec=task_data_spec,
                processor=processor,
                max_seq_length=max_seq_length,
                idx=idx,
                task_name=task_name,
                extra_env_info=extra_env_info,
                content_items=content_items,
                reason=f"audio_expand:{','.join(audio_paths)}",
                exc=exc,
            )

    sys_len = system_only["input_ids"].shape[1]
    system_message["token_ids"] = message_both["input_ids"][0][:sys_len]
    user_message["token_ids"] = message_both["input_ids"][0][sys_len:]

    if task_data_spec.video_temporal_patch_size > 1 and video_paths:
        tokenizer = processor.tokenizer
        img_start_id = tokenizer.convert_tokens_to_ids("<img>")
        img_end_id = tokenizer.convert_tokens_to_ids("</img>")
        user_message["token_ids"] = _collapse_video_frame_token_wrappers(
            user_message["token_ids"],
            video_flags=video_flags,
            temporal_patch_size=task_data_spec.video_temporal_patch_size,
            img_start_id=img_start_id,
            img_end_id=img_end_id,
        )

    multimodal_keys = get_multimodal_keys_from_processor(processor)
    extra_multimodal_keys = {
        "pixel_values_flat",
        "image_num_patches",
        "sound_clips",
    }
    all_multimodal_keys = list(
        dict.fromkeys(
            list(multimodal_keys)
            + [key for key in extra_multimodal_keys if key in message_both]
        )
    )
    for key in all_multimodal_keys:
        if key not in message_both:
            continue
        if key == "sound_clips":
            clips = message_both["sound_clips"]
            clip_tensors = [
                torch.from_numpy(clip).float()
                if isinstance(clip, np.ndarray)
                else torch.as_tensor(clip).float()
                for clip in clips
            ]
            lengths = torch.tensor(
                [clip.shape[0] for clip in clip_tensors], dtype=torch.int32
            )
            user_message["sound_clips"] = PackedTensor(clip_tensors, dim_to_pack=0)
            user_message["sound_length"] = PackedTensor(lengths, dim_to_pack=0)
            user_message["sound_clip_duration"] = task_data_spec.sound_clip_duration
            user_message["sound_clip_min_duration"] = task_data_spec.sound_clip_min_duration
        else:
            user_message[key] = PackedTensor(
                message_both[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    if "imgs_sizes" in message_both:
        user_message["imgs_sizes"] = PackedTensor(message_both["imgs_sizes"], dim_to_pack=0)
        if media_num_frames:
            user_message["num_frames"] = PackedTensor(
                torch.tensor(media_num_frames, dtype=torch.int32), dim_to_pack=0
            )

    if "token_type_ids" in message_both:
        system_message["token_type_ids"] = message_both["token_type_ids"][0][:sys_len]
        user_message["token_type_ids"] = message_both["token_type_ids"][0][sys_len:]

    for entry in user_message.get("content", []):
        if entry.get("type") == "image" and isinstance(entry.get("image"), Image.Image):
            entry["image"] = "__video_frame__"
        entry.pop("_cached_waveform", None)
        entry.pop("_cached_audio_duration", None)

    message_log: VLMMessageLogType = [system_message, user_message]
    length = sum(len(m["token_ids"]) for m in message_log)

    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    img_start_id = tokenizer.convert_tokens_to_ids("<img>")
    num_vision_tokens = (
        sum(int((m["token_ids"] == image_token_id).sum()) for m in message_log)
        if image_token_id is not None
        else 0
    )
    num_image_groups = (
        sum(int((m["token_ids"] == img_start_id).sum()) for m in message_log)
        if img_start_id is not None
        else 0
    )
    post_collapse_placeholders = num_image_groups if num_image_groups > 0 else num_vision_tokens
    collapse_savings = max(0, num_vision_tokens - post_collapse_placeholders)
    vision_expansion = _compute_vision_expansion_from_processor(
        message_both,
        processor,
        media_num_frames,
        post_collapse_placeholders,
        task_data_spec.video_temporal_patch_size,
    )
    expanded_length = length - collapse_savings + vision_expansion

    vllm_max_num_tiles = task_data_spec.max_num_tiles if task_data_spec.use_tiling else None
    vllm_max_num_patches = _derive_vllm_max_num_patches(
        message_both, processor, task_data_spec, mm_kwargs
    )

    loss_multiplier = 1.0
    if max_seq_length is not None and expanded_length >= max_seq_length:
        vllm_kwargs = {
            "vllm_content": None,
            "vllm_images": [],
            "vllm_videos": [],
            "vllm_num_frames": task_data_spec.num_frames,
            "vllm_temporal_patch_size": task_data_spec.video_temporal_patch_size,
            "vllm_video_prompt_style": video_prompt_style,
            "vllm_video_frame_indices": [],
            "vllm_video_fps": [],
            "vllm_audio_paths": [],
            "vllm_audio_waveforms": [],
            "vllm_max_audio_duration": task_data_spec.max_audio_duration,
            "vllm_max_num_tiles": vllm_max_num_tiles,
            "vllm_max_num_patches": vllm_max_num_patches,
        }
        image_token_ids = [
            tokenizer.convert_tokens_to_ids(token)
            for token in ("<img>", "<image>", "</img>")
        ]
        image_token_ids = [token_id for token_id in image_token_ids if token_id is not None]
        for chat_message in message_log:
            token_ids = chat_message["token_ids"][: min(4, max_seq_length // len(message_log))]
            for image_token in image_token_ids:
                token_ids = token_ids[token_ids != image_token]
            chat_message["token_ids"] = token_ids
            for key, value in list(chat_message.items()):
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)
        length = sum(len(m["token_ids"]) for m in message_log)
        loss_multiplier = 0.0
    else:
        cached_waveforms = (
            cached_audio_waveforms
            if any(waveform is not None for waveform in cached_audio_waveforms)
            else []
        )
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": image_paths,
            "vllm_videos": video_paths,
            "vllm_num_frames": task_data_spec.num_frames,
            "vllm_temporal_patch_size": task_data_spec.video_temporal_patch_size,
            "vllm_video_prompt_style": video_prompt_style,
            "vllm_video_frame_indices": video_frame_indices,
            "vllm_video_fps": video_fps,
            "vllm_audio_paths": audio_paths,
            "vllm_audio_waveforms": cached_waveforms,
            "vllm_max_audio_duration": task_data_spec.max_audio_duration,
            "vllm_max_num_tiles": vllm_max_num_tiles,
            "vllm_max_num_patches": vllm_max_num_patches,
        }

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_name,
        "vision_expansion": vision_expansion,
        "collapse_savings": collapse_savings,
        **vllm_kwargs,
    }
    return output


def _construct_multichoice_prompt(
    prompt: str, question: str, options: dict[str, str]
) -> str:
    """Construct prompt from question and options."""
    output = prompt
    output += f"\n\nQuestion: {question}\nOptions:\n"
    output += "\n".join(
        [
            f"{letter}) {option}"
            for letter, option in options.items()
            if option is not None
        ]
    )
    return output


def multichoice_qa_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for multiple-choice problems."""
    question = datum_dict["question"]
    answer = str(datum_dict["answer"])
    options = datum_dict["options"]
    extra_env_info = {"ground_truth": answer}
    if "subject" in datum_dict:
        extra_env_info.update({"subject": datum_dict["subject"]})

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        question = _construct_multichoice_prompt(
            task_data_spec.prompt, question, options
        )
    user_message = {"role": "user", "content": question}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def nemo_gym_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for Nemo Gym.

    In text mode (3rd arg is a tokenizer) the message_log is a placeholder because
    NeMo-Gym builds the real prompt server-side from `responses_create_params`.
    In VLM mode (3rd arg is an AutoProcessor) we delegate to
    `nemo_gym_example_to_nemo_rl_datum_spec` so the first-turn user message carries
    the HF-processor token layout (<img>/<image>×N/</img>) and multimodal data
    (pixel_values, imgs_sizes, ...) that Megatron needs to identify image regions.
    """
    extra_env_info = json.loads(datum_dict["extra_env_info"])

    # VLM mode is signalled by the caller passing an AutoProcessor or a local
    # processor wrapper. Some Omni wrappers intentionally expose the tokenizer
    # and chat-template API without an `image_processor` attribute.
    is_multimodal_processor = hasattr(tokenizer, "image_processor") or (
        hasattr(tokenizer, "tokenizer") and hasattr(tokenizer, "apply_chat_template")
    )
    if is_multimodal_processor:
        from nemo_rl.environments.nemo_gym import nemo_gym_example_to_nemo_rl_datum_spec

        datum = nemo_gym_example_to_nemo_rl_datum_spec(
            extra_env_info, idx, processor=tokenizer, max_seq_length=max_seq_length,
        )
        # Honor the dataset's task_name rather than the hardcoded "nemo_gym" default.
        datum["task_name"] = datum_dict["task_name"]
        return datum

    output: DatumSpec = {
        # load to dict format here since `Dataset` cannot handle nested structure well in `NemoGymDataset`
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": datum_dict["task_name"],
        # fake keys for compatibility with the current GRPO implementation
        "message_log": [{"role": "user", "content": "", "token_ids": torch.tensor([])}],
        "length": 0,
    }
    return output


# Processor registry. Key is the processor name, value is the processor function.
# Note: We cast the literal dict to Dict[str, TaskDataProcessFnCallable] because
# type checkers see each concrete function's signature as a distinct callable type.
# Without the cast, the registry's inferred type becomes a union of those specific
# callables, which is not assignable to the uniform TaskDataProcessFnCallable.
# The cast asserts our intent that all entries conform to the common callable protocol.
PROCESSOR_REGISTRY: Dict[str, TaskDataProcessFnCallable] = cast(
    Dict[str, TaskDataProcessFnCallable],
    {
        "default": math_hf_data_processor,
        "helpsteer3_data_processor": helpsteer3_data_processor,
        "math_data_processor": math_data_processor,
        "math_hf_data_processor": math_hf_data_processor,
        "multichoice_qa_processor": multichoice_qa_processor,
        "sft_processor": sft_processor,
        "vlm_hf_data_processor": vlm_hf_data_processor,
        "nemo_gym_data_processor": nemo_gym_data_processor,
    },
)


def register_processor(
    processor_name: str, processor_function: TaskDataProcessFnCallable
) -> None:
    if processor_name in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor name {processor_name} already registered")
    PROCESSOR_REGISTRY[processor_name] = processor_function

    print(f"[INFO] Dataset processor {processor_name} registered")
