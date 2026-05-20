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
import os
from dataclasses import dataclass
from typing import Any, NotRequired, Optional, Protocol, TypedDict, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.data.multimodal_utils import PackedTensor

# OpenAI-API-like message log, but every messsage may contain associated tensors (i.e. tokenized strings and logprobs) in addition to the original "content" string
LLMMessageLogType = list[dict[str, Union[str, torch.Tensor]]]
VLMMessageLogType = list[dict[str, Union[str, torch.Tensor, PackedTensor]]]

# Flattened message log where all tensors and data are concatenated together for a conversation
# Converts a conversation from list-of-turns format to key-value format with concatenated tensors
FlatMessagesType = dict[str, Union[list[str], torch.Tensor]]

PathLike = Union[str, "os.PathLike[Any]"]
TokenizerType = PreTrainedTokenizerBase


class DatumSpec(TypedDict):
    message_log: LLMMessageLogType | VLMMessageLogType
    length: int  # total (concatenated) length of the message tensors
    extra_env_info: Optional[dict[str, Any]]
    loss_multiplier: float  # multiplier for the loss for this datum. 0 to mask out (say the sample is invalid)
    idx: int
    task_name: NotRequired[str]
    stop_strings: NotRequired[list[str]]  # Optional stop strings for generation
    __extra__: NotRequired[Any]  # This allows additional fields of any type


class PreferenceDatumSpec(TypedDict):
    message_log_chosen: LLMMessageLogType
    message_log_rejected: LLMMessageLogType
    length_chosen: int
    length_rejected: int
    loss_multiplier: float
    idx: int


@dataclass
class TaskDataSpec:
    task_name: Optional[str] = None
    # prompt
    prompt_file: Optional[PathLike] = None

    system_prompt_file: Optional[PathLike] = None

    # Image/video processing -- optional overrides of model vision_config defaults.
    # When None, the processor falls back to the model's vision_config values.
    num_frames: int = 8
    use_tiling: bool = False  # False = simple resize (1 tile); True = InternVL multi-tile splitting
    use_dynamic_resolution: Optional[bool] = None  # None = auto-detect from model; True/False = explicit override
    max_num_tiles: Optional[int] = None  # only used when use_tiling=True; controls tile count
    max_num_patches: Optional[int] = None  # None = vision_config default; set = override per-image patch cap
    video_target_num_patches: Optional[int] = None  # fixed per-frame patch budget for video (mirrors Megatron's --video-target-num-patches)

    # conv3d temporal compression -- when > 1, every T consecutive video
    # frames are grouped into one tubelet by the RADIO vision encoder.
    # Placeholder tokens are only emitted for primary frames (every T-th).
    video_temporal_patch_size: int = 1

    # When True the video target resolution preserves the source aspect
    # ratio (matching Megatron SFT's --video-maintain-aspect-ratio).
    video_maintain_aspect_ratio: bool = True

    # audio processing -- when True the data processor extracts audio from
    # videos and includes it as audio content alongside vision frames.
    use_audio: bool = False
    max_audio_duration: Optional[float] = None  # seconds; clip audio longer than this
    sound_clip_duration: float = 30.0  # seconds; max clip length for audio splitting (matches SFT --sound-clip-duration)
    sound_clip_min_duration: float = 0.1  # seconds; minimum tail clip length after splitting

    # Budget-aware vision resolution: minimum tokens reserved for generation
    # when auto-computing max_num_patches from max_seq_length.
    min_generation_tokens: int = 2000

    def __post_init__(self) -> None:
        def load_prompt_file(
            prompt_file: Optional[PathLike],
        ) -> Optional[str]:
            """Load prompt from file if it exists, otherwise return as is."""
            if prompt_file is None:
                return None
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Prompt file {prompt_file} not found")

        # Load prompts from files if they exist
        self.system_prompt = load_prompt_file(self.system_prompt_file)
        self.prompt = load_prompt_file(self.prompt_file)

    def copy_defaults(self, from_spec: "TaskDataSpec") -> None:
        """Apply default values from another Task instance for any None attributes."""
        default_attrs = {
            "system_prompt": from_spec.system_prompt,
            "prompt": from_spec.prompt,
        }

        for attr_name, default_value in default_attrs.items():
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, default_value)


class TaskDataProcessFnCallable(Protocol):
    """A callable that processes a loaded datum dictionary into a DatumSpec."""

    def __call__(
        self,
        datum_dict: dict[str, Any],
        task_data_spec: TaskDataSpec,
        tokenizer: TokenizerType,
        max_seq_length: int | None,
        idx: int,
    ) -> DatumSpec:
        raise NotImplementedError("Task data process not implemented")


class TaskDataPreProcessFnCallable(Protocol):
    """A callable that preprocesses a raw datum dict before the main processing step.

    Used by datasets that need to transform raw entries (e.g. resolve file paths,
    format conversations) before tokenization.
    """

    def __call__(self, datum_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Task data preprocess not implemented")
