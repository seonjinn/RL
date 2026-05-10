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
from dataclasses import dataclass, field
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


_UNSET = object()


@dataclass(init=False)
class TaskDataSpec:
    task_name: Optional[str] = None
    # prompt
    prompt_file: Optional[PathLike] = None

    system_prompt_file: Optional[PathLike] = None

    # Image/video processing overrides. None means the processor/model default
    # remains authoritative.
    num_frames: int = 8
    use_tiling: bool = False
    use_dynamic_resolution: Optional[bool] = None
    max_num_tiles: Optional[int] = None
    max_num_patches: Optional[int] = None
    video_target_num_patches: Optional[int] = None

    # Conv3D video tubelet grouping.
    video_temporal_patch_size: int = 1
    video_maintain_aspect_ratio: bool = True

    # Audio processing controls.
    use_audio: bool = False
    max_audio_duration: Optional[float] = None
    sound_clip_duration: float = 30.0
    sound_clip_min_duration: float = 0.1

    # Minimum prompt-side budget reserved for generation when dynamic
    # resolution computes image/video patch budgets from max_seq_length.
    min_generation_tokens: int = 2000

    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    _explicit_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    _DEFAULTS = {
        "num_frames": 8,
        "use_tiling": False,
        "use_dynamic_resolution": None,
        "max_num_tiles": None,
        "max_num_patches": None,
        "video_target_num_patches": None,
        "video_temporal_patch_size": 1,
        "video_maintain_aspect_ratio": True,
        "use_audio": False,
        "max_audio_duration": None,
        "sound_clip_duration": 30.0,
        "sound_clip_min_duration": 0.1,
        "min_generation_tokens": 2000,
    }

    def __init__(
        self,
        task_name: Optional[str] = None,
        prompt_file: Optional[PathLike] = None,
        system_prompt_file: Optional[PathLike] = None,
        *,
        num_frames: Any = _UNSET,
        use_tiling: Any = _UNSET,
        use_dynamic_resolution: Any = _UNSET,
        max_num_tiles: Any = _UNSET,
        max_num_patches: Any = _UNSET,
        video_target_num_patches: Any = _UNSET,
        video_temporal_patch_size: Any = _UNSET,
        video_maintain_aspect_ratio: Any = _UNSET,
        use_audio: Any = _UNSET,
        max_audio_duration: Any = _UNSET,
        sound_clip_duration: Any = _UNSET,
        sound_clip_min_duration: Any = _UNSET,
        min_generation_tokens: Any = _UNSET,
    ) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        self.system_prompt_file = system_prompt_file
        self._explicit_fields = set()
        if prompt_file is not None:
            self._explicit_fields.add("prompt")
        if system_prompt_file is not None:
            self._explicit_fields.add("system_prompt")

        field_values = {
            "num_frames": num_frames,
            "use_tiling": use_tiling,
            "use_dynamic_resolution": use_dynamic_resolution,
            "max_num_tiles": max_num_tiles,
            "max_num_patches": max_num_patches,
            "video_target_num_patches": video_target_num_patches,
            "video_temporal_patch_size": video_temporal_patch_size,
            "video_maintain_aspect_ratio": video_maintain_aspect_ratio,
            "use_audio": use_audio,
            "max_audio_duration": max_audio_duration,
            "sound_clip_duration": sound_clip_duration,
            "sound_clip_min_duration": sound_clip_min_duration,
            "min_generation_tokens": min_generation_tokens,
        }
        for field_name, value in field_values.items():
            if value is _UNSET:
                setattr(self, field_name, self._DEFAULTS[field_name])
            else:
                setattr(self, field_name, value)
                self._explicit_fields.add(field_name)
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
        """Apply defaults from another task spec when fields are unset.

        Concrete Omni fields can legitimately be set to their class defaults
        (for example ``num_frames=8``).  Track constructor-provided fields so
        default-copy inheritance does not overwrite those explicit values.
        """
        default_attrs = {
            "system_prompt": from_spec.system_prompt,
            "prompt": from_spec.prompt,
            "num_frames": from_spec.num_frames,
            "use_tiling": from_spec.use_tiling,
            "use_dynamic_resolution": from_spec.use_dynamic_resolution,
            "max_num_tiles": from_spec.max_num_tiles,
            "max_num_patches": from_spec.max_num_patches,
            "video_target_num_patches": from_spec.video_target_num_patches,
            "video_temporal_patch_size": from_spec.video_temporal_patch_size,
            "video_maintain_aspect_ratio": from_spec.video_maintain_aspect_ratio,
            "use_audio": from_spec.use_audio,
            "max_audio_duration": from_spec.max_audio_duration,
            "sound_clip_duration": from_spec.sound_clip_duration,
            "sound_clip_min_duration": from_spec.sound_clip_min_duration,
            "min_generation_tokens": from_spec.min_generation_tokens,
        }

        for attr_name, default_value in default_attrs.items():
            if attr_name in self._explicit_fields:
                continue
            if attr_name in self._DEFAULTS or getattr(self, attr_name) is None:
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
