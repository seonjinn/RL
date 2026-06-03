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

import math
import os
from typing import Any, Optional, Union

import numpy as np

_DEBUG = os.environ.get("NRL_DEBUG", "0") == "1"

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BatchFeature, PretrainedConfig
from transformers.processing_utils import ProcessorMixin

from nemo_rl.utils.vlm_debug import (
    debug_enabled,
    stable_hash,
    tensor_summary,
    write_stage,
)

def _wrap_enable_thinking(kwargs: dict) -> dict:
    """Mirror ``enable_thinking`` into both template styles.

    Some Nemotron-Omni checkpoints read ``enable_thinking`` as a top-level
    Jinja variable, while older templates read it from the nested
    ``chat_template_kwargs`` dict. Keep the top-level kwarg intact and also
    mirror it into ``chat_template_kwargs`` so both template variants behave
    consistently.
    """
    if "enable_thinking" in kwargs:
        val = kwargs["enable_thinking"]
        ct_kw = dict(kwargs.get("chat_template_kwargs", {}) or {})
        ct_kw["enable_thinking"] = val
        kwargs["chat_template_kwargs"] = ct_kw
        if _DEBUG:
            print(
                f"[THINK_FIX] _wrap_enable_thinking: "
                f"enable_thinking={val} top_level_preserved=True "
                f"chat_template_kwargs={ct_kw}"
            )
    return kwargs


from nemo_rl.models.nemotron_h_nano_vl.image_processing import (
    dynamic_preprocess as _internvl_dynamic_preprocess,
)

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None

DEFAULT_NUM_TILES = 12

# Incoming prompt tags
IMG_INPUT_TAG = "<image>"
# Preprocessed prompt placeholders
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"


def _flatten_images(images):
    """Recursively flatten nested lists of images into a flat list."""
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, list):
        result = []
        for item in images:
            result.extend(_flatten_images(item))
        return result
    return [images]


def _text_boundary(
    text: str | None,
    token: str,
    context: int = 80,
) -> dict[str, Any] | None:
    if text is None:
        return None
    index = text.find(token)
    if index < 0:
        return {"token": token, "index": -1, "snippet": None}
    start = max(0, index - context)
    end = min(len(text), index + len(token) + context)
    return {
        "token": token,
        "index": index,
        "snippet": text[start:end],
    }


def _first_tensor_row(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    return value[0]


def _tensor_shape(value: Any) -> list[int] | None:
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def _json_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return None
    return value


def _token_count(sequence: torch.Tensor | None, token_id: int | None) -> int | None:
    if sequence is None or token_id is None:
        return None
    return int((sequence == token_id).sum().item())


def _write_processor_output_debug(
    processor: "DynamicResolutionProcessor",
    rendered_text: list[str],
    processed_text: list[str],
    batch: BatchFeature,
    *,
    path_type: str,
    max_num_tiles: Optional[int] = None,
    max_num_patches: Optional[int] = None,
    num_tokens_available: Optional[int] = None,
    video_flags: Optional[list[bool]] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    if not debug_enabled():
        return

    input_ids = _first_tensor_row(batch.get("input_ids"))
    attention_mask = _first_tensor_row(batch.get("attention_mask"))
    pixel_values = batch.get("pixel_values")
    pixel_values_name = "pixel_values"
    if pixel_values is None:
        pixel_values = batch.get("pixel_values_flat")
        pixel_values_name = "pixel_values_flat"
    pixel_values_summary = (
        tensor_summary(pixel_values_name, pixel_values)
        if torch.is_tensor(pixel_values)
        else None
    )

    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
    image_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END)

    payload: dict[str, Any] = {
        "checkpoint": getattr(processor, "name_or_path", None)
        or getattr(tokenizer, "name_or_path", None),
        "processor_class": type(processor).__name__,
        "processor_model_input_names": list(
            getattr(processor, "model_input_names", []) or []
        ),
        "path_type": path_type,
        "rendered_text_hash": stable_hash(rendered_text),
        "rendered_text_boundary": _text_boundary(
            rendered_text[0] if rendered_text else None, IMG_INPUT_TAG
        ),
        "processed_text_hash": stable_hash(processed_text),
        "processed_text_boundary": _text_boundary(
            processed_text[0] if processed_text else None, IMG_START
        ),
        "input_ids_shape": _tensor_shape(batch.get("input_ids")),
        "input_ids_len": int(input_ids.shape[0]) if input_ids is not None else None,
        "input_ids_hash": stable_hash(input_ids.tolist()) if input_ids is not None else None,
        "attention_mask_shape": _tensor_shape(batch.get("attention_mask")),
        "attention_mask_len": (
            int(attention_mask.shape[0]) if attention_mask is not None else None
        ),
        "attention_mask_hash": (
            stable_hash(attention_mask.tolist()) if attention_mask is not None else None
        ),
        "image_token_count": _token_count(input_ids, image_token_id),
        "image_start_count": _token_count(input_ids, image_start_token_id),
        "image_end_count": _token_count(input_ids, image_end_token_id),
        "pixel_values_shape": _tensor_shape(pixel_values),
        "pixel_values_summary": pixel_values_summary,
        "imgs_sizes": _json_list(batch.get("imgs_sizes")),
        "image_num_patches": _json_list(batch.get("image_num_patches")),
        "max_num_tiles": (
            max_num_tiles
            if max_num_tiles is not None
            else getattr(processor, "max_num_tiles", None)
        ),
        "max_num_patches": (
            max_num_patches
            if max_num_patches is not None
            else getattr(processor, "max_num_patches", None)
        ),
        "num_tokens_available": num_tokens_available,
        "video_flag_count": sum(1 for flag in video_flags if flag) if video_flags else 0,
    }
    if extra_payload:
        payload.update(extra_payload)
    write_stage("processor_output", payload)


class DynamicResolutionProcessor(ProcessorMixin):
    """Dual-mode image processor for VLMs (Nano v3 VL / Omni).

    Mirrors the 3rdparty vLLM ``nano_nemotron_vl.py`` behavior:

    * **Dynamic resolution** (default, when ``max_num_tiles`` is *not* set):
      resizes images to variable dimensions constrained by ``max_num_patches``
      from the model's ``vision_config``.  Returns ``pixel_values`` +
      ``imgs_sizes`` for the RADIO vision encoder.

    * **Static InternVL tiling** (when ``max_num_tiles`` *is* set):
      splits images into fixed ``image_size x image_size`` tiles, matching
      vLLM's ``image_to_pixel_values`` path.  Returns ``pixel_values_flat`` +
      ``image_num_patches``.

    Both ``max_num_tiles`` and ``max_num_patches`` can be overridden per-call
    via ``**kwargs`` (plumbed from the data config).  When absent, the
    model's ``vision_config`` defaults are used.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"
    model_input_names = ["pixel_values", "imgs_sizes"]
    image_token = "<image>"

    def __init__(
        self,
        tokenizer,
        config: PretrainedConfig,
        *,
        chat_template: Optional[str] = None,
    ):
        super().__init__(tokenizer, chat_template=chat_template)
        self.config = config

        vision_args = getattr(config.vision_config, "args", {}) or {}
        self.patch_size = getattr(config.vision_config, "patch_size", 16)
        self.min_num_patches = vision_args.get("min_num_patches", 1024)
        self.max_num_patches = vision_args.get("max_num_patches", 13312)
        self.downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.pixel_shuffle = getattr(config, "pixel_shuffle", True)

        self.image_size = getattr(config, "force_image_size", 512)
        self.use_thumbnail = getattr(config, "use_thumbnail", True)
        self.num_image_token = int(
            (self.image_size // self.patch_size) ** 2
            * (self.downsample_ratio**2)
        )

        norm_mean = vision_args.get("norm_mean", [0.48145466, 0.4578275, 0.40821073])
        norm_std = vision_args.get("norm_std", [0.26862954, 0.26130258, 0.27577711])
        self.norm_mean = torch.tensor(norm_mean)
        self.norm_std = torch.tensor(norm_std)

        print(
            f"[{type(self).__name__}] initialized: "
            f"patch_size={self.patch_size} image_size={self.image_size} "
            f"max_num_patches={self.max_num_patches} min_num_patches={self.min_num_patches} "
            f"downsample_ratio={self.downsample_ratio} num_image_token={self.num_image_token} "
            f"use_thumbnail={self.use_thumbnail}"
        )

    @staticmethod
    def conversation_preprocessor(message: dict) -> dict:
        """Flatten multimodal content lists into plain strings for chat templates.

        Chat templates (Jinja2) that do ``message.content | string`` will
        produce Python repr for list content.  This method converts the list
        into the same string that the Nemotron multimodal template branch would
        build before its common sanitization/trim step.
        """
        content = message.get("content")
        if not isinstance(content, list):
            return message

        text = ""
        num_images = 0
        num_videos = 0
        num_audios = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            ctype = item.get("type", "")
            if ctype in ("image", "image_url"):
                num_images += 1
            elif ctype in ("video", "video_url"):
                num_videos += 1
            elif ctype in ("audio", "audio_url"):
                num_audios += 1
            elif ctype == "text":
                text += item.get("text", "")

        if IMG_INPUT_TAG in text:
            num_images = 0
        if "<video>" in text:
            num_videos = 0
        if "<so_embedding>" in text:
            num_audios = 0

        mm_content = ""
        if num_images > 1:
            image_tags = [
                f"<image {image_idx + 1}>{IMG_INPUT_TAG}"
                for image_idx in range(num_images)
            ]
            mm_content = " ".join(image_tags) + "\n"
        elif num_images == 1:
            mm_content = f"{IMG_INPUT_TAG}\n"

        mm_content += "<video>\n" * num_videos
        mm_content += "<so_embedding>\n" * num_audios
        return {**message, "content": mm_content + text.lstrip("\n")}

    def compute_num_embeddings(self, height: int, width: int) -> int:
        """Compute number of image embeddings for given dimensions.

        This must match vLLM's DynamicResolutionImageTiler._get_num_embeddings().
        Formula: (height // patch_size) * (width // patch_size) // downsample_ratio²
        """
        reduction_factor = int(1 / self.downsample_ratio)
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        return num_patches // (reduction_factor**2)

    @staticmethod
    def video_target_resolution(
        orig_w: int,
        orig_h: int,
        video_target_num_patches: int,
        patch_size: int,
        pixel_shuffle: bool,
        maintain_aspect_ratio: bool = True,
    ) -> tuple[int, int]:
        """Compute target resolution for a video frame (SFT-compatible).

        Ported from Megatron SFT ``DynamicResolutionImageTilingStrategy.process_media``
        (``is_video=True`` branch).  Also matches vLLM
        ``_compute_aspect_preserving_size``.

        Returns:
            ``(target_width, target_height)`` in pixels.
        """
        if maintain_aspect_ratio:
            aspect = orig_w / max(orig_h, 1)
            ph = round(math.sqrt(video_target_num_patches / aspect))
            pw = round(math.sqrt(video_target_num_patches * aspect))
        else:
            side = int(math.sqrt(video_target_num_patches))
            ph = pw = side

        ph, pw = max(1, ph), max(1, pw)

        required_divisor = 2 if pixel_shuffle else 1
        if required_divisor > 1:
            rem_h = ph % required_divisor
            rem_w = pw % required_divisor
            ph_up = ph + (required_divisor - rem_h if rem_h else 0)
            ph_down = ph - rem_h
            pw_up = pw + (required_divisor - rem_w if rem_w else 0)
            pw_down = pw - rem_w
            if ph_up * pw_up <= video_target_num_patches:
                ph, pw = ph_up, pw_up
            else:
                ph = max(required_divisor, ph_down)
                pw = max(required_divisor, pw_down)

        if _DEBUG:
            print(
                f"[VIDEO_TARGET_RES] ({orig_w}x{orig_h}) -> "
                f"({pw * patch_size}x{ph * patch_size}) "
                f"patches=({pw}x{ph}={pw * ph}) "
                f"target_num_patches={video_target_num_patches} "
                f"maintain_aspect={maintain_aspect_ratio}",
                flush=True,
            )

        return pw * patch_size, ph * patch_size

    def compute_params(
        self,
        images: list[Image.Image],
        num_tokens_available: int,
        video_flags: Optional[list[bool]] = None,
        video_temporal_patch_size: int = 1,
        video_target_num_patches: Optional[int] = None,
        video_maintain_aspect_ratio: bool = True,
    ) -> list[int]:
        """SFT-style iterative token budget allocation across images.

        Ported from Megatron SFT ``DynamicResolutionImageTilingStrategy.compute_params``
        (image_processing.py).  Distributes a shared token budget across all
        images/video-frames so the total vision tokens stay within the sequence
        length limit.

        Args:
            images: Flat list of PIL images (regular images + video frames).
            num_tokens_available: Post-reduction vision token budget (LLM token
                space).  The method internally expands by pixel_shuffle / conv
                merging factors.
            video_flags: Per-image boolean; ``True`` for video frames, ``False``
                for regular images.  ``None`` means all images.
            video_temporal_patch_size: Conv3D temporal patch size (T).
            video_target_num_patches: Fixed patch target for video frames.
            video_maintain_aspect_ratio: Aspect-ratio preservation for video.

        Returns:
            List of per-image ``max_num_patches`` values that
            ``compute_target_resolution`` / ``preprocess_image`` should use.
        """
        if not images:
            return []

        if video_flags is None:
            video_flags = [False] * len(images)

        num_images = sum(1 for f in video_flags if not f)
        num_video_frames = sum(1 for f in video_flags if f)

        budget = num_tokens_available * (4 if self.pixel_shuffle else 1)

        if video_temporal_patch_size > 1 and num_video_frames > 0:
            if num_images == 0:
                budget *= video_temporal_patch_size
            else:
                imgs_frac = num_images / len(images)
                vid_frac = num_video_frames / len(images)
                budget = int(
                    budget * (imgs_frac + vid_frac * video_temporal_patch_size)
                )

        budget = max(budget, self.min_num_patches * len(images))

        per_image_budgets: list[int] = []
        for is_video in video_flags:
            if is_video and video_target_num_patches is not None:
                per_image_budgets.append(video_target_num_patches)
            else:
                per_image_budgets.append(
                    max(min(budget, self.max_num_patches), self.min_num_patches)
                )

        if _DEBUG:
            print(
                f"[COMPUTE_PARAMS] entry: n_images={num_images} "
                f"n_video_frames={num_video_frames} "
                f"num_tokens_available={num_tokens_available} "
                f"expanded_budget={budget} "
                f"video_temporal_patch_size={video_temporal_patch_size} "
                f"video_target_num_patches={video_target_num_patches}",
                flush=True,
            )

        for iteration in range(10):
            token_counts: list[int] = []
            for img, img_budget, is_video in zip(images, per_image_budgets, video_flags):
                if is_video and video_target_num_patches is not None:
                    tw, th = self.video_target_resolution(
                        img.width,
                        img.height,
                        video_target_num_patches,
                        self.patch_size,
                        self.pixel_shuffle,
                        maintain_aspect_ratio=video_maintain_aspect_ratio,
                    )
                    count = (th // self.patch_size) * (tw // self.patch_size)
                else:
                    th, tw = self.compute_target_resolution(img, img_budget)
                    count = (th // self.patch_size) * (tw // self.patch_size)
                token_counts.append(count)

            total = sum(token_counts)

            if _DEBUG and iteration == 0:
                print(
                    f"[COMPUTE_PARAMS] iter={iteration} "
                    f"total_patches={total} budget={budget} "
                    f"per_image_counts={token_counts[:5]}{'...' if len(token_counts) > 5 else ''}",
                    flush=True,
                )

            if total <= budget:
                if _DEBUG:
                    print(
                        f"[COMPUTE_PARAMS] converged iter={iteration} "
                        f"total={total} <= budget={budget}",
                        flush=True,
                    )
                return per_image_budgets

            scaling_factor = budget / total
            new_budgets = [
                max(self.min_num_patches, int(tc * scaling_factor))
                for tc in token_counts
            ]
            scaled_down = any(
                new_budgets[i] < per_image_budgets[i]
                for i in range(len(per_image_budgets))
                if not (video_flags[i] and video_target_num_patches is not None)
            )
            if not scaled_down:
                for i in range(len(per_image_budgets)):
                    if not (video_flags[i] and video_target_num_patches is not None):
                        per_image_budgets[i] = self.min_num_patches
            else:
                for i in range(len(per_image_budgets)):
                    if not (video_flags[i] and video_target_num_patches is not None):
                        per_image_budgets[i] = new_budgets[i]

            if _DEBUG:
                print(
                    f"[COMPUTE_PARAMS] iter={iteration} over budget: "
                    f"total={total} > budget={budget} "
                    f"scaling_factor={scaling_factor:.4f} "
                    f"scaled_down={scaled_down}",
                    flush=True,
                )

        if _DEBUG:
            _final_total = sum(token_counts)
            print(
                f"[COMPUTE_PARAMS] WARNING: did not converge after 10 iters, "
                f"final_total={_final_total} budget={budget}",
                flush=True,
            )
        return per_image_budgets

    def compute_target_resolution(
        self,
        image: Image.Image,
        max_num_patches: Optional[int] = None,
    ) -> tuple[int, int]:
        """Compute dynamic target resolution for an image.

        Ported from vLLM's DynamicResolutionImageTiler.process_media().

        Args:
            image: PIL image to compute resolution for.
            max_num_patches: Per-image patch cap override. When ``None``,
                falls back to ``self.max_num_patches`` from vision_config.
        """
        effective_max = (
            max_num_patches if max_num_patches is not None else self.max_num_patches
        )

        orig_width, orig_height = image.size
        closest_patch_height = round(orig_height / self.patch_size + 0.5)
        closest_patch_width = round(orig_width / self.patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        factor = min(math.sqrt(effective_max / patches), 1.0)
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        target_patches = target_patch_height * target_patch_width
        if (
            effective_max > self.min_num_patches
            and target_patches < self.min_num_patches
        ):
            up_factor = math.sqrt(self.min_num_patches / target_patches)
            up_h = math.ceil(up_factor * target_patch_height)
            up_w = math.ceil(up_factor * target_patch_width)
            if effective_max >= up_h * up_w:
                if _DEBUG:
                    print(
                        f"[MIN_PATCHES_UPSCALE] applied: "
                        f"({target_patch_height}x{target_patch_width}={target_patches}) -> "
                        f"({up_h}x{up_w}={up_h * up_w}) "
                        f"min_num_patches={self.min_num_patches} "
                        f"effective_max={effective_max} "
                        f"image=({orig_width}x{orig_height})",
                        flush=True,
                    )
                target_patch_height, target_patch_width = up_h, up_w
            elif _DEBUG:
                print(
                    f"[MIN_PATCHES_SKIP] upscale would exceed budget: "
                    f"({up_h}x{up_w}={up_h * up_w}) > effective_max={effective_max} "
                    f"keeping ({target_patch_height}x{target_patch_width}={target_patches}) "
                    f"image=({orig_width}x{orig_height})",
                    flush=True,
                )
        elif _DEBUG and target_patches < self.min_num_patches:
            print(
                f"[MIN_PATCHES_SKIP] budget too small: "
                f"effective_max={effective_max} <= min_num_patches={self.min_num_patches} "
                f"target=({target_patch_height}x{target_patch_width}={target_patches}) "
                f"image=({orig_width}x{orig_height})",
                flush=True,
            )

        if self.pixel_shuffle:
            required_divisor = 2
            rem_h = target_patch_height % required_divisor
            if rem_h != 0:
                inc_h = required_divisor - rem_h
                if (target_patch_height + inc_h) * target_patch_width <= effective_max:
                    target_patch_height += inc_h
                else:
                    target_patch_height = max(
                        required_divisor, target_patch_height - rem_h
                    )

            rem_w = target_patch_width % required_divisor
            if rem_w != 0:
                inc_w = required_divisor - rem_w
                if target_patch_height * (target_patch_width + inc_w) <= effective_max:
                    target_patch_width += inc_w
                else:
                    target_patch_width = max(
                        required_divisor, target_patch_width - rem_w
                    )

        target_height = target_patch_height * self.patch_size
        target_width = target_patch_width * self.patch_size
        return target_height, target_width

    # Dynamic resolution preprocessing
    def preprocess_image(
        self,
        image: Image.Image,
        max_num_patches: Optional[int] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess a single image using dynamic resolution."""
        target_h, target_w = self.compute_target_resolution(image, max_num_patches)
        resized = image.resize((target_w, target_h), Image.BICUBIC)
        # Match vLLM's DynamicResolutionImageTiler ordering: resize first,
        # then convert non-RGB images just before tensorization.
        if resized.mode != "RGB":
            resized = resized.convert("RGB")

        tensor = transforms.ToTensor()(resized)
        tensor = (tensor - self.norm_mean.view(3, 1, 1)) / self.norm_std.view(
            3, 1, 1
        )

        return tensor, (target_h, target_w)

    # Static InternVL tiling preprocessing
    def preprocess_image_static(
        self,
        image: Image.Image,
        max_num_tiles: int,
    ) -> tuple[torch.Tensor, int]:
        """Preprocess a single image using InternVL-style static tiling.

        Returns:
            (stacked_tiles, num_tiles): tiles tensor of shape
            ``[num_tiles, 3, image_size, image_size]`` and the tile count.
        """
        tile_images = _internvl_dynamic_preprocess(
            image,
            image_size=self.image_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=self.use_thumbnail,
        )
        stacked = torch.stack(tile_images)
        stacked = (
            stacked - self.norm_mean.view(1, 3, 1, 1)
        ) / self.norm_std.view(1, 3, 1, 1)
        return stacked, stacked.shape[0]

    # Placeholder expansion
    def _add_image_placeholders_dynamic(
        self,
        text: list[str],
        imgs_sizes_list: list[list[int]],
    ) -> list[str]:
        """Expand ``<image>`` tags using dynamic-resolution embedding counts."""
        if len(imgs_sizes_list) == 0:
            return text

        if _DEBUG:
            print(
                f"[ADD_IMG_PLACEHOLDER_DEBUG] dynamic mode: text count={len(text)}, imgs_sizes count={len(imgs_sizes_list)}"
            )

        results_lst = []
        for t in text:
            parts = t.split(IMG_INPUT_TAG)
            assert len(parts) - 1 == len(imgs_sizes_list), (
                f"Number of {IMG_INPUT_TAG} tokens ({len(parts) - 1}) "
                f"doesn't match number of images ({len(imgs_sizes_list)})"
            )
            result = parts[0]
            for (h, w), part in zip(imgs_sizes_list, parts[1:]):
                num_embeddings = self.compute_num_embeddings(h, w)
                image_placeholder = IMG_START + IMG_CONTEXT * num_embeddings + IMG_END
                result += image_placeholder + part
            results_lst.append(result)
        return results_lst

    def _add_image_placeholders_static(
        self,
        text: list[str],
        num_tiles_per_image: list[int],
    ) -> list[str]:
        """Expand ``<image>`` tags using static-tiling token counts."""
        if len(num_tiles_per_image) == 0:
            return text

        if _DEBUG:
            print(
                f"[ADD_IMG_PLACEHOLDER_DEBUG] static mode: text count={len(text)}, tiles_per_image={num_tiles_per_image}"
            )

        results_lst = []
        for t in text:
            parts = t.split(IMG_INPUT_TAG)
            assert len(parts) - 1 == len(num_tiles_per_image), (
                f"Number of {IMG_INPUT_TAG} tokens ({len(parts) - 1}) "
                f"doesn't match number of images ({len(num_tiles_per_image)})"
            )
            result = parts[0]
            for num_tiles, part in zip(num_tiles_per_image, parts[1:]):
                num_embeddings = num_tiles * self.num_image_token
                image_placeholder = IMG_START + IMG_CONTEXT * num_embeddings + IMG_END
                result += image_placeholder + part
            results_lst.append(result)
        return results_lst

    # keep old name as alias for backwards compatibility
    def _add_image_placeholders(self, text, imgs_sizes_list):
        return self._add_image_placeholders_dynamic(text, imgs_sizes_list)

    # Shared dynamic-resolution helper
    def _resolve_dynamic_images(
        self,
        flat_images: list[Image.Image],
        max_num_patches: Optional[int],
        num_tokens_available: Optional[int] = None,
        video_flags: Optional[list[bool]] = None,
        video_temporal_patch_size: int = 1,
        video_target_num_patches: Optional[int] = None,
        video_maintain_aspect_ratio: bool = True,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """Resolve per-image resolutions and preprocess.

        When *num_tokens_available* is provided, uses SFT-style shared
        budgeting via :meth:`compute_params`.  Otherwise falls back to the
        flat per-image *max_num_patches* (backward compatible).

        Returns:
            ``(pixel_values_list, imgs_sizes_list)``
        """
        if num_tokens_available is not None and flat_images:
            per_image_budgets = self.compute_params(
                flat_images,
                num_tokens_available,
                video_flags=video_flags,
                video_temporal_patch_size=video_temporal_patch_size,
                video_target_num_patches=video_target_num_patches,
                video_maintain_aspect_ratio=video_maintain_aspect_ratio,
            )
            if _DEBUG:
                print(
                    f"[RESOLVE_DYN] shared_budget path: "
                    f"num_tokens_available={num_tokens_available} "
                    f"n_images={len(flat_images)} "
                    f"budgets={per_image_budgets[:5]}{'...' if len(per_image_budgets) > 5 else ''}",
                    flush=True,
                )
        else:
            per_image_budgets = [max_num_patches] * len(flat_images)
            if _DEBUG and flat_images:
                print(
                    f"[RESOLVE_DYN] per_image_flat path: "
                    f"max_num_patches={max_num_patches} n_images={len(flat_images)}",
                    flush=True,
                )

        if video_flags is None:
            _vflags = [False] * len(flat_images)
        else:
            _vflags = video_flags

        pixel_values_list: list[torch.Tensor] = []
        imgs_sizes_list: list[list[int]] = []

        for image, budget, is_vid in zip(flat_images, per_image_budgets, _vflags):
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")
            if is_vid and video_target_num_patches is not None:
                target_w, target_h = self.video_target_resolution(
                    image.width,
                    image.height,
                    video_target_num_patches,
                    self.patch_size,
                    self.pixel_shuffle,
                    maintain_aspect_ratio=video_maintain_aspect_ratio,
                )
                if image.mode != "RGB":
                    image = image.convert("RGB")
                arr = np.asarray(image, dtype=np.uint8)
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                if t.shape[2] != target_h or t.shape[3] != target_w:
                    t = F.interpolate(
                        t,
                        size=(target_h, target_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )
                tensor = t.squeeze(0) / 255.0
                tensor = (tensor - self.norm_mean.view(3, 1, 1)) / self.norm_std.view(
                    3, 1, 1
                )
                pixel_values_list.append(tensor)
                imgs_sizes_list.append([target_h, target_w])
            else:
                pv, (h, w) = self.preprocess_image(image, budget)
                pixel_values_list.append(pv)
                imgs_sizes_list.append([h, w])

        return pixel_values_list, imgs_sizes_list

    # Main entry point
    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
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

        flat_images = _flatten_images(images) if images is not None else []

        if max_num_tiles is not None and flat_images:
            return self._call_static(text, flat_images, max_num_tiles, **kwargs)
        else:
            return self._call_dynamic(
                text,
                flat_images,
                max_num_patches,
                num_tokens_available=num_tokens_available,
                video_flags=video_flags,
                video_temporal_patch_size=video_temporal_patch_size,
                video_target_num_patches=video_target_num_patches,
                video_maintain_aspect_ratio=video_maintain_aspect_ratio,
                **kwargs,
            )

    # Dynamic resolution path
    def _call_dynamic(
        self,
        text: list[str],
        flat_images: list[Image.Image],
        max_num_patches: Optional[int],
        *,
        num_tokens_available: Optional[int] = None,
        video_flags: Optional[list[bool]] = None,
        video_temporal_patch_size: int = 1,
        video_target_num_patches: Optional[int] = None,
        video_maintain_aspect_ratio: bool = True,
        **kwargs,
    ) -> BatchFeature:
        pixel_values_list, imgs_sizes_list = self._resolve_dynamic_images(
            flat_images,
            max_num_patches,
            num_tokens_available=num_tokens_available,
            video_flags=video_flags,
            video_temporal_patch_size=video_temporal_patch_size,
            video_target_num_patches=video_target_num_patches,
            video_maintain_aspect_ratio=video_maintain_aspect_ratio,
        )

        processed_text = self._add_image_placeholders_dynamic(text, imgs_sizes_list)

        text_inputs = self.tokenizer(
            processed_text,
            return_tensors=kwargs.get("return_tensors"),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )

        result = BatchFeature(data=dict(text_inputs))

        if pixel_values_list:
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

        _write_processor_output_debug(
            self,
            text,
            processed_text,
            result,
            path_type="dynamic",
            max_num_patches=max_num_patches,
            num_tokens_available=num_tokens_available,
            video_flags=video_flags,
        )
        return result

    # Static InternVL tiling path
    def _call_static(
        self,
        text: list[str],
        flat_images: list[Image.Image],
        max_num_tiles: int,
        **kwargs,
    ) -> BatchFeature:
        all_tiles: list[torch.Tensor] = []
        num_tiles_per_image: list[int] = []

        for image in flat_images:
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")
            tiles, n_tiles = self.preprocess_image_static(image, max_num_tiles)
            all_tiles.append(tiles)
            num_tiles_per_image.append(n_tiles)

        processed_text = self._add_image_placeholders_static(text, num_tiles_per_image)

        text_inputs = self.tokenizer(
            processed_text,
            return_tensors=kwargs.get("return_tensors"),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )

        result = BatchFeature(data=dict(text_inputs))

        if all_tiles:
            result["pixel_values_flat"] = torch.cat(all_tiles, dim=0)
            result["image_num_patches"] = torch.tensor(
                num_tiles_per_image, dtype=torch.int32
            )

        _write_processor_output_debug(
            self,
            text,
            processed_text,
            result,
            path_type="static",
            max_num_tiles=max_num_tiles,
        )
        return result

    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        """Override to handle multimodal content lists in messages.

        The base Jinja2 chat template does ``message.content | string`` which
        produces the Python repr of list content instead of inserting
        ``<image>`` tokens.  This override:
        1. Extracts PIL images from content lists.
        2. Flattens content lists to strings with ``<image>`` placeholders via
           ``conversation_preprocessor``.
        3. Renders the template with the flattened messages.
        4. For tokenize=True, calls ``self.__call__`` with extracted images.
        """
        images = []
        preprocessed = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)
                preprocessed.append(self.conversation_preprocessor(msg))
            else:
                preprocessed.append(msg)

        if not tokenize:
            kwargs = _wrap_enable_thinking(kwargs)
            result = super().apply_chat_template(
                preprocessed, tokenize=False, **kwargs
            )
            if _DEBUG and isinstance(result, str):
                _tail = result[-80:].replace("\n", "\\n")
                _has_open_think = result.endswith("<think>\n")
                _has_closed_think = "<think></think>" in result[-40:]
                print(
                    f"[THINK_FIX] apply_chat_template(tokenize=False): "
                    f"open_think={_has_open_think} closed_think={_has_closed_think} "
                    f"tail={_tail!r}"
                )
            return result

        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        enable_thinking = kwargs.pop("enable_thinking", None)
        render_kwargs = {"add_generation_prompt": add_generation_prompt}
        if enable_thinking is not None:
            render_kwargs["enable_thinking"] = enable_thinking
            render_kwargs["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }
        if _DEBUG:
            print(
                f"[THINK_FIX] apply_chat_template(tokenize=True): "
                f"enable_thinking={enable_thinking} render_kwargs={render_kwargs}"
            )
        rendered_text = super().apply_chat_template(
            preprocessed,
            tokenize=False,
            **render_kwargs,
        )
        if _DEBUG:
            _tail = rendered_text[-80:].replace("\n", "\\n")
            print(f"[THINK_FIX] rendered_text tail={_tail!r}")
        return self(
            text=rendered_text,
            images=images or None,
            **kwargs,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def is_dynamic_resolution_model(config: PretrainedConfig) -> bool:
    """Check if model uses dynamic resolution (not static InternVL tiling)."""
    if not hasattr(config, "vision_config"):
        return False
    vision_args = getattr(config.vision_config, "args", None)
    if vision_args is None:
        return False
    return "min_num_patches" in vision_args and "max_num_patches" in vision_args
