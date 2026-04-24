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

"""Omni (vision+audio) dynamic resolution processor.

Extends DynamicResolutionProcessor with audio support: computes the correct
number of <so_embedding> placeholder tokens per audio clip and expands them
in the tokenized sequence.
"""

import math
import os
from typing import Optional, Union

_DEBUG = os.environ.get("NRL_DEBUG", "0") == "1"

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature, PretrainedConfig

from nemo_rl.models.nano_v3_vl.dynamic_resolution_processor import (
    DynamicResolutionProcessor,
    _flatten_images,
)

AUDIO_INPUT_TAG = "<so_embedding>"
AUDIO_CONTEXT = "<so_embedding>"


class OmniDynamicResolutionProcessor(DynamicResolutionProcessor):
    """Dynamic resolution processor with audio (sound) modality support.

    Inherits all vision processing from DynamicResolutionProcessor and adds:
    - Audio placeholder expansion: each <so_embedding> tag is expanded into
      the correct number of repeated tokens based on waveform length.
    - sound_clips output: raw waveform arrays passed through for model consumption.
    """

    model_input_names = ["pixel_values", "imgs_sizes", "sound_clips"]

    def __init__(
        self,
        tokenizer,
        config: PretrainedConfig,
        *,
        chat_template: Optional[str] = None,
    ):
        super().__init__(tokenizer, config, chat_template=chat_template)

        sound_config = getattr(config, "sound_config", None)
        self.audio_sampling_rate = (
            getattr(sound_config, "sampling_rate", 16000) if sound_config else 16000
        )
        self.audio_subsampling_factor = (
            getattr(sound_config, "subsampling_factor", 8) if sound_config else 8
        )
        self.audio_hop_length = 160  # 10ms at 16kHz
        self.audio_n_fft = 512
        self.audio_subsampling_conv_kernel_size = (
            getattr(sound_config, "subsampling_conv_kernel_size", 3)
            if sound_config
            else 3
        )
        self.audio_subsampling_conv_stride = (
            getattr(sound_config, "subsampling_conv_stride", 2) if sound_config else 2
        )
        self.audio_clip_duration_s = (
            getattr(sound_config, "clip_duration_s", 30) if sound_config else 30
        )
        self.audio_clip_min_duration_s = (
            getattr(sound_config, "clip_min_duration_s", 0.1) if sound_config else 0.1
        )

    def _normalize_audio_length(
        self,
        audio_len: int,
        clip_duration_s: Optional[float] = None,
        clip_min_duration_s: Optional[float] = None,
    ) -> int:
        """Match vllm's ParakeetExtractor._normalize_audio_length."""
        _cd = (
            clip_duration_s
            if clip_duration_s is not None
            else self.audio_clip_duration_s
        )
        _cmd = (
            clip_min_duration_s
            if clip_min_duration_s is not None
            else self.audio_clip_min_duration_s
        )
        clip_target = int(round(_cd * self.audio_sampling_rate))
        tail_min = int(round(_cmd * self.audio_sampling_rate))
        audio_len = max(audio_len, tail_min)
        tail_rem = audio_len % clip_target
        if 0 < tail_rem < tail_min:
            audio_len += tail_min - tail_rem
        return audio_len

    def _conv_subsampling_output_length(self, num_frames: int) -> int:
        """Replicate FastConformer subsampling output length.

        Each subsampling conv layer uses padding = kernel_size // 2:
          out = floor((in + 2*padding - kernel_size) / stride) + 1
        Number of layers = log(subsampling_factor) / log(stride).
        """
        k = self.audio_subsampling_conv_kernel_size
        s = self.audio_subsampling_conv_stride
        p = k // 2
        num_layers = round(math.log(self.audio_subsampling_factor) / math.log(s))
        length = num_frames
        for _ in range(num_layers):
            length = (length + 2 * p - k) // s + 1
        return max(1, length)

    def _split_clip_sample_counts(
        self,
        audio_len: int,
        clip_duration_s: Optional[float] = None,
        clip_min_duration_s: Optional[float] = None,
    ) -> list[int]:
        """Per-clip sample counts matching SFT AudioTransformParakeetStrategy.compute_params.

        For audio <= clip_duration_s (30s): single clip.
        For audio > clip_duration_s: full clips + possibly shorter last clip
        with last clip >= min_duration (0.1s = 1600 samples @ 16kHz).
        """
        _cd = (
            clip_duration_s
            if clip_duration_s is not None
            else self.audio_clip_duration_s
        )
        _cmd = (
            clip_min_duration_s
            if clip_min_duration_s is not None
            else self.audio_clip_min_duration_s
        )
        clip_target = int(round(_cd * self.audio_sampling_rate))
        tail_min = int(round(_cmd * self.audio_sampling_rate))
        norm_len = self._normalize_audio_length(audio_len, _cd, _cmd)
        if norm_len <= clip_target:
            return [norm_len]
        num_clips = math.ceil(norm_len / clip_target)
        remainder = norm_len % clip_target
        last = clip_target if remainder == 0 else max(remainder, tail_min)
        return [clip_target] * (num_clips - 1) + [last]

    def _compute_audio_num_tokens(
        self,
        waveform: np.ndarray,
        clip_duration_s: Optional[float] = None,
        clip_min_duration_s: Optional[float] = None,
    ) -> int:
        """Number of audio placeholder tokens for a waveform.

        Uses sum-of-per-clip tokens matching SFT compute_params line 185:
            num_embeddings = sum(estimate_audio_num_embeddings(n) for n in clip_samples)
        """
        clip_samples = self._split_clip_sample_counts(
            len(waveform), clip_duration_s, clip_min_duration_s
        )
        total = sum(
            self._conv_subsampling_output_length(cs // self.audio_hop_length)
            for cs in clip_samples
        )
        if _DEBUG:
            _old = self._conv_subsampling_output_length(
                self._normalize_audio_length(
                    len(waveform), clip_duration_s, clip_min_duration_s
                )
                // self.audio_hop_length
            )
            print(
                f"[HF_CLIP_SPLIT] _compute_audio_num_tokens: "
                f"waveform_len={len(waveform)} "
                f"num_clips={len(clip_samples)} clip_samples={clip_samples} "
                f"sum_of_per_clip_tokens={total} "
                f"old_full_sequence_tokens={_old} "
                f"diff={total - _old}",
                flush=True,
            )
        return total

    def estimate_audio_tokens(
        self,
        duration_seconds: float,
        max_duration: Optional[float] = None,
    ) -> int:
        """Compute exact audio token count from duration (no waveform needed).

        Uses sum-of-per-clip tokens matching SFT and updated vLLM.

        Args:
            duration_seconds: Duration of the audio clip in seconds.
            max_duration: If set, clip duration to this maximum first.
        """
        if max_duration is not None:
            duration_seconds = min(duration_seconds, max_duration)
        audio_len = int(round(duration_seconds * self.audio_sampling_rate))
        clip_samples = self._split_clip_sample_counts(audio_len)
        tokens = sum(
            self._conv_subsampling_output_length(cs // self.audio_hop_length)
            for cs in clip_samples
        )
        if _DEBUG:
            print(
                f"[AUDIO_ESTIMATE] duration={duration_seconds:.2f}s "
                f"max_duration={max_duration} "
                f"samples={audio_len} num_clips={len(clip_samples)} "
                f"clip_samples={clip_samples} tokens={tokens}",
                flush=True,
            )
        return tokens

    def _add_audio_placeholders(
        self,
        text: list[str],
        audio_num_tokens: list[int],
    ) -> list[str]:
        """Expand each <so_embedding> tag into the correct number of tokens."""
        if not audio_num_tokens:
            return text
        results_lst = []
        idx = 0
        for t in text:
            while AUDIO_INPUT_TAG in t:
                num_tokens = audio_num_tokens[idx] if idx < len(audio_num_tokens) else 1
                t = t.replace(
                    AUDIO_INPUT_TAG, "<|audio_placeholder|>" * num_tokens, 1
                )
                idx += 1
            t = t.replace("<|audio_placeholder|>", AUDIO_CONTEXT)
            results_lst.append(t)
        return results_lst

    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
        audio: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        if not isinstance(text, list):
            text = [text]

        max_num_tiles: Optional[int] = kwargs.pop("max_num_tiles", None)
        max_num_patches: Optional[int] = kwargs.pop("max_num_patches", None)
        num_tokens_available: Optional[int] = kwargs.pop("num_tokens_available", None)
        video_flags: Optional[list[bool]] = kwargs.pop("video_flags", None)
        video_temporal_patch_size: int = kwargs.pop("video_temporal_patch_size", 1)
        video_target_num_patches: Optional[int] = kwargs.pop(
            "video_target_num_patches", None
        )
        video_maintain_aspect_ratio: bool = kwargs.pop(
            "video_maintain_aspect_ratio", True
        )

        if _DEBUG:
            print("[OMNI_PROC_DEBUG] __call__ entry:")
            print(
                f"[OMNI_PROC_DEBUG]   text type={type(text).__name__}, len={len(text)}"
            )
            print(
                f"[OMNI_PROC_DEBUG]   images type={type(images).__name__ if images is not None else 'None'}"
            )
            print(
                f"[OMNI_PROC_DEBUG]   audio type={type(audio).__name__ if audio is not None else 'None'}"
            )
            print(
                f"[OMNI_PROC_DEBUG]   max_num_tiles={max_num_tiles} max_num_patches={max_num_patches}"
            )
            print(f"[OMNI_PROC_DEBUG]   num_tokens_available={num_tokens_available}")
            print(
                f"[OMNI_PROC_DEBUG]   video_flags={video_flags[:5] if video_flags else None}{'...' if video_flags and len(video_flags) > 5 else ''}"
            )
            print(
                f"[OMNI_PROC_DEBUG]   kwargs keys={list(kwargs.keys())}"
            )

        flat_images = _flatten_images(images) if images is not None else []
        use_static = max_num_tiles is not None and len(flat_images) > 0

        # Vision
        if use_static:
            all_tiles: list[torch.Tensor] = []
            num_tiles_per_image: list[int] = []
            for image in flat_images:
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected PIL Image, got {type(image)}")
                tiles, n_tiles = self.preprocess_image_static(image, max_num_tiles)
                all_tiles.append(tiles)
                num_tiles_per_image.append(n_tiles)
            processed_text = self._add_image_placeholders_static(
                text, num_tiles_per_image
            )
        else:
            pixel_values_list, imgs_sizes_list = self._resolve_dynamic_images(
                flat_images,
                max_num_patches,
                num_tokens_available=num_tokens_available,
                video_flags=video_flags,
                video_temporal_patch_size=video_temporal_patch_size,
                video_target_num_patches=video_target_num_patches,
                video_maintain_aspect_ratio=video_maintain_aspect_ratio,
            )
            processed_text = self._add_image_placeholders_dynamic(
                text, imgs_sizes_list
            )

        # Audio
        audio_clips: list[np.ndarray] = []
        audio_num_tokens: list[int] = []
        if audio is not None:
            if isinstance(audio, np.ndarray):
                audio = [audio]
            for clip in audio:
                if isinstance(clip, np.ndarray):
                    waveform = clip.squeeze() if clip.ndim > 1 else clip
                elif isinstance(clip, torch.Tensor):
                    waveform = clip.numpy().squeeze()
                else:
                    raise ValueError(f"Unsupported audio type: {type(clip)}")
                audio_clips.append(waveform)
                audio_num_tokens.append(self._compute_audio_num_tokens(waveform))
            processed_text = self._add_audio_placeholders(
                processed_text, audio_num_tokens
            )

        # Tokenize
        text_inputs = self.tokenizer(
            processed_text,
            return_tensors=kwargs.get("return_tensors"),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )

        result = BatchFeature(data=dict(text_inputs))

        if audio_clips:
            result["sound_clips"] = audio_clips

        if use_static and all_tiles:
            result["pixel_values_flat"] = torch.cat(all_tiles, dim=0)
            result["image_num_patches"] = torch.tensor(
                num_tiles_per_image, dtype=torch.int32
            )
        elif not use_static and pixel_values_list:
            import torch.nn.functional as F

            max_h = max(s[0] for s in imgs_sizes_list)
            max_w = max(s[1] for s in imgs_sizes_list)
            padded_pvs = []
            for pv, (h, w) in zip(pixel_values_list, imgs_sizes_list):
                pad_h = max_h - h
                pad_w = max_w - w
                if pad_h > 0 or pad_w > 0:
                    pv = F.pad(pv, (0, pad_w, 0, pad_h), value=0)
                padded_pvs.append(pv)

            result["pixel_values"] = torch.stack(padded_pvs)
            result["imgs_sizes"] = torch.tensor(imgs_sizes_list, dtype=torch.int32)

        return result

    @staticmethod
    def _load_audio_from_path(
        audio_path: str,
        target_sr: int = 16000,
        max_duration: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Load audio from a file path.

        Delegates to the shared ``load_audio_waveform`` utility in
        ``nemo_rl.models.generation.vllm.utils`` to avoid code duplication.
        Both paths use soundfile for pure audio files and PyAV for video
        containers, resampling to *target_sr* via librosa.

        Returns None if the file has no audio stream or audio is empty.
        """
        from nemo_rl.models.generation.vllm.utils import load_audio_waveform

        return load_audio_waveform(
            audio_path, target_sr=target_sr, max_duration=max_duration
        )

    def expand_audio_tokens(
        self,
        tokenized: dict,
        audio_paths: list[str],
        *,
        audio_waveforms: Optional[list[Optional[np.ndarray]]] = None,
        max_audio_duration: Optional[float] = None,
        sound_clip_duration: Optional[float] = None,
        sound_clip_min_duration: Optional[float] = None,
    ) -> dict:
        """Load audio from paths, expand <so_embedding> placeholders in tokenized output.

        Called after apply_chat_template which renders each audio entry as a
        single <so_embedding> token.  This method:
        1. Loads each audio file (supports video containers via PyAV)
        2. Computes the correct number of audio tokens per clip
        3. Replaces each single <so_embedding> with N repeated copies
        4. Stores the loaded waveforms in ``sound_clips`` for the model

        Args:
            tokenized: Tokenized output from apply_chat_template.
            audio_paths: List of audio/video file paths.
            audio_waveforms: Optional pre-decoded waveforms (from combined
                video+audio decode). When ``waveforms[i]`` is not ``None``,
                ``_load_audio_from_path`` is skipped for that index.
            max_audio_duration: If set, clip each audio to at most this many seconds.
            sound_clip_duration: Override for clip splitting boundary (seconds).
                When ``None``, uses the value from the model's sound_config
                (default 30).  Must match the value used by the Megatron
                training path (``_prepare_sound_data``) to avoid token count
                mismatches.
            sound_clip_min_duration: Override for minimum tail clip length
                (seconds).  When ``None``, uses sound_config default (0.1).
        """
        _effective_cd = (
            sound_clip_duration
            if sound_clip_duration is not None
            else self.audio_clip_duration_s
        )
        _effective_cmd = (
            sound_clip_min_duration
            if sound_clip_min_duration is not None
            else self.audio_clip_min_duration_s
        )
        if _DEBUG:
            print(
                f"[AUDIO_CFG] expand_audio_tokens: effective_clip_duration={_effective_cd} "
                f"effective_clip_min_duration={_effective_cmd} "
                f"(explicit={sound_clip_duration is not None})",
                flush=True,
            )

        audio_token_id = self.tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT)
        audio_start_id = self.tokenizer.convert_tokens_to_ids("<so_start>")
        audio_end_id = self.tokenizer.convert_tokens_to_ids("<so_end>")
        input_ids = tokenized["input_ids"][0]

        audio_clips = []
        new_ids = []
        audio_idx = 0
        for tid in input_ids.tolist():
            if tid == audio_token_id and audio_idx < len(audio_paths):
                cached = (
                    audio_waveforms[audio_idx]
                    if audio_waveforms is not None and audio_idx < len(audio_waveforms)
                    else None
                )
                if cached is not None:
                    waveform = cached
                    if _DEBUG:
                        print(
                            f"[HF_AUDIO_EXPAND_DEBUG] clip={audio_idx} "
                            f"using cached waveform (len={len(waveform)}, "
                            f"skipped file I/O for {audio_paths[audio_idx].split('/')[-1]})",
                            flush=True,
                        )
                else:
                    waveform = self._load_audio_from_path(
                        audio_paths[audio_idx],
                        target_sr=self.audio_sampling_rate,
                        max_duration=max_audio_duration,
                    )
                if waveform is None:
                    from nemo_rl.models.generation.vllm.utils import AudioLoadError

                    raise AudioLoadError(
                        audio_paths[audio_idx],
                        reason=f"audio clip {audio_idx} returned None waveform; "
                        f"cannot expand <so_embedding> placeholder without valid audio",
                    )
                audio_clips.append(waveform)
                num_tokens = self._compute_audio_num_tokens(
                    waveform,
                    sound_clip_duration,
                    sound_clip_min_duration,
                )
                if _DEBUG:
                    _raw_len = len(waveform)
                    _norm_len = self._normalize_audio_length(
                        _raw_len,
                        sound_clip_duration,
                        sound_clip_min_duration,
                    )
                    _n_frames = _norm_len // self.audio_hop_length
                    print(
                        f"[HF_AUDIO_EXPAND_DEBUG] clip={audio_idx} "
                        f"waveform_len={_raw_len} "
                        f"normalized_len={_norm_len} "
                        f"num_frames={_n_frames} "
                        f"hop={self.audio_hop_length} "
                        f"subsamp_factor={self.audio_subsampling_factor} "
                        f"kernel={self.audio_subsampling_conv_kernel_size} "
                        f"stride={self.audio_subsampling_conv_stride} "
                        f"num_tokens={num_tokens}",
                        flush=True,
                    )
                new_ids.append(audio_start_id)
                new_ids.extend([audio_token_id] * num_tokens)
                new_ids.append(audio_end_id)
                audio_idx += 1
            else:
                new_ids.append(tid)

        tokenized["input_ids"] = torch.tensor([new_ids], dtype=input_ids.dtype)
        if "attention_mask" in tokenized:
            tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
        if audio_clips:
            tokenized["sound_clips"] = audio_clips
        return tokenized


def is_omni_model(config: PretrainedConfig) -> bool:
    """Check if model is an omni model (dynamic resolution + sound_config)."""
    from nemo_rl.models.nano_v3_vl import is_dynamic_resolution_model

    if not is_dynamic_resolution_model(config):
        return False
    return getattr(config, "sound_config", None) is not None
