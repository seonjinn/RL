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
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BatchFeature, PretrainedConfig
from transformers.processing_utils import ProcessorMixin

from nemo_rl.models.nemotron_h_nano_vl.image_processing import (
    dynamic_preprocess as _internvl_dynamic_preprocess,
)

_DEBUG = os.environ.get("NRL_DEBUG", "0") == "1"

# Configure PIL to handle large images without warnings.
Image.MAX_IMAGE_PIXELS = None

IMG_INPUT_TAG = "<image>"
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"


def _wrap_enable_thinking(kwargs: dict) -> dict:
    """Mirror ``enable_thinking`` into both template styles."""

    if "enable_thinking" in kwargs:
        val = kwargs["enable_thinking"]
        ct_kw = dict(kwargs.get("chat_template_kwargs", {}) or {})
        ct_kw["enable_thinking"] = val
        kwargs["chat_template_kwargs"] = ct_kw
    return kwargs


def _flatten_images(images):
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, list):
        flattened = []
        for item in images:
            flattened.extend(_flatten_images(item))
        return flattened
    return [images]


class DynamicResolutionProcessor(ProcessorMixin):
    """Nemotron image processor that preserves dynamic-resolution sizing."""

    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"
    model_input_names = ["pixel_values", "imgs_sizes"]
    image_token = IMG_CONTEXT

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
            (self.image_size // self.patch_size) ** 2 * (self.downsample_ratio**2)
        )

        norm_mean = vision_args.get("norm_mean", [0.48145466, 0.4578275, 0.40821073])
        norm_std = vision_args.get("norm_std", [0.26862954, 0.26130258, 0.27577711])
        self.norm_mean = torch.tensor(norm_mean)
        self.norm_std = torch.tensor(norm_std)

        if _DEBUG:
            print(
                f"[{type(self).__name__}] patch_size={self.patch_size} "
                f"image_size={self.image_size} max_num_patches={self.max_num_patches} "
                f"min_num_patches={self.min_num_patches} "
                f"downsample_ratio={self.downsample_ratio} "
                f"pixel_shuffle={self.pixel_shuffle}",
                flush=True,
            )

    @staticmethod
    def conversation_preprocessor(message: dict) -> dict:
        """Flatten multimodal content lists into plain strings for chat templates."""

        content = message.get("content")
        if not isinstance(content, list):
            return message

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type", "")
            if item_type == "image":
                parts.append(IMG_INPUT_TAG)
            elif item_type == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return {**message, "content": "".join(parts)}

    def compute_num_embeddings(self, height: int, width: int) -> int:
        reduction_factor = int(1 / self.downsample_ratio)
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        return num_patches // (reduction_factor**2)

    def compute_target_resolution(
        self,
        image: Image.Image,
        max_num_patches: Optional[int] = None,
    ) -> tuple[int, int]:
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
        if effective_max > self.min_num_patches and target_patches < self.min_num_patches:
            up_factor = math.sqrt(self.min_num_patches / target_patches)
            up_h = math.ceil(up_factor * target_patch_height)
            up_w = math.ceil(up_factor * target_patch_width)
            if effective_max >= up_h * up_w:
                target_patch_height, target_patch_width = up_h, up_w

        if self.pixel_shuffle:
            required_divisor = 2
            rem_h = target_patch_height % required_divisor
            if rem_h != 0:
                inc_h = required_divisor - rem_h
                if (target_patch_height + inc_h) * target_patch_width <= effective_max:
                    target_patch_height += inc_h
                else:
                    target_patch_height = max(required_divisor, target_patch_height - rem_h)

            rem_w = target_patch_width % required_divisor
            if rem_w != 0:
                inc_w = required_divisor - rem_w
                if target_patch_height * (target_patch_width + inc_w) <= effective_max:
                    target_patch_width += inc_w
                else:
                    target_patch_width = max(required_divisor, target_patch_width - rem_w)

        target_height = target_patch_height * self.patch_size
        target_width = target_patch_width * self.patch_size
        return target_height, target_width

    def preprocess_image(
        self,
        image: Image.Image,
        max_num_patches: Optional[int] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_h, target_w = self.compute_target_resolution(image, max_num_patches)
        resized = image.resize((target_w, target_h), Image.BICUBIC)

        tensor = transforms.ToTensor()(resized)
        tensor = (tensor - self.norm_mean.view(3, 1, 1)) / self.norm_std.view(3, 1, 1)
        return tensor, (target_h, target_w)

    def preprocess_image_static(
        self,
        image: Image.Image,
        max_num_tiles: int,
    ) -> tuple[torch.Tensor, int]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tile_images = _internvl_dynamic_preprocess(
            image,
            image_size=self.image_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=self.use_thumbnail,
        )
        stacked = torch.stack(tile_images)
        stacked = (stacked - self.norm_mean.view(1, 3, 1, 1)) / self.norm_std.view(
            1, 3, 1, 1
        )
        return stacked, stacked.shape[0]

    def _add_image_placeholders_dynamic(
        self,
        text: list[str],
        imgs_sizes_list: list[list[int]],
    ) -> list[str]:
        if len(imgs_sizes_list) == 0:
            return text

        results = []
        for item in text:
            parts = item.split(IMG_INPUT_TAG)
            assert len(parts) - 1 == len(imgs_sizes_list), (
                f"Number of {IMG_INPUT_TAG} tokens ({len(parts) - 1}) "
                f"doesn't match number of images ({len(imgs_sizes_list)})"
            )
            result = parts[0]
            for (height, width), suffix in zip(imgs_sizes_list, parts[1:]):
                num_embeddings = self.compute_num_embeddings(height, width)
                placeholder = IMG_START + IMG_CONTEXT * num_embeddings + IMG_END
                result += placeholder + suffix
            results.append(result)
        return results

    def _add_image_placeholders_static(
        self,
        text: list[str],
        num_tiles_per_image: list[int],
    ) -> list[str]:
        if len(num_tiles_per_image) == 0:
            return text

        results = []
        for item in text:
            parts = item.split(IMG_INPUT_TAG)
            assert len(parts) - 1 == len(num_tiles_per_image), (
                f"Number of {IMG_INPUT_TAG} tokens ({len(parts) - 1}) "
                f"doesn't match number of images ({len(num_tiles_per_image)})"
            )
            result = parts[0]
            for num_tiles, suffix in zip(num_tiles_per_image, parts[1:]):
                num_embeddings = num_tiles * self.num_image_token
                placeholder = IMG_START + IMG_CONTEXT * num_embeddings + IMG_END
                result += placeholder + suffix
            results.append(result)
        return results

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
        kwargs.pop("num_tokens_available", None)
        kwargs.pop("video_flags", None)
        kwargs.pop("video_temporal_patch_size", None)
        kwargs.pop("video_target_num_patches", None)
        kwargs.pop("video_maintain_aspect_ratio", None)

        flat_images = _flatten_images(images) if images is not None else []
        if max_num_tiles is not None and flat_images:
            return self._call_static(text, flat_images, max_num_tiles, **kwargs)
        return self._call_dynamic(text, flat_images, max_num_patches, **kwargs)

    def _call_dynamic(
        self,
        text: list[str],
        flat_images: list[Image.Image],
        max_num_patches: Optional[int],
        **kwargs,
    ) -> BatchFeature:
        pixel_values_list: list[torch.Tensor] = []
        imgs_sizes_list: list[list[int]] = []
        for image in flat_images:
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(image)}")
            pixel_values, (height, width) = self.preprocess_image(image, max_num_patches)
            pixel_values_list.append(pixel_values)
            imgs_sizes_list.append([height, width])

        processed_text = self._add_image_placeholders_dynamic(text, imgs_sizes_list)
        text_inputs = self.tokenizer(
            processed_text,
            return_tensors=kwargs.get("return_tensors"),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )

        result = BatchFeature(data=dict(text_inputs))
        if pixel_values_list:
            max_h = max(size[0] for size in imgs_sizes_list)
            max_w = max(size[1] for size in imgs_sizes_list)
            padded_pvs = []
            for pixel_values, (height, width) in zip(pixel_values_list, imgs_sizes_list):
                pad_h = max_h - height
                pad_w = max_w - width
                if pad_h > 0 or pad_w > 0:
                    pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), value=0)
                padded_pvs.append(pixel_values)

            result["pixel_values"] = torch.stack(padded_pvs)
            result["imgs_sizes"] = torch.tensor(imgs_sizes_list, dtype=torch.int32)

        return result

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
            tiles, num_tiles = self.preprocess_image_static(image, max_num_tiles)
            all_tiles.append(tiles)
            num_tiles_per_image.append(num_tiles)

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
        return result

    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        """Handle multimodal content lists before rendering the chat template."""

        images = []
        preprocessed = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image = item.get("image")
                        if isinstance(image, Image.Image):
                            images.append(image)
                preprocessed.append(self.conversation_preprocessor(message))
            else:
                preprocessed.append(message)

        if not tokenize:
            kwargs = _wrap_enable_thinking(kwargs)
            return super().apply_chat_template(preprocessed, tokenize=False, **kwargs)

        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        enable_thinking = kwargs.pop("enable_thinking", None)
        render_kwargs = {"add_generation_prompt": add_generation_prompt}
        if enable_thinking is not None:
            render_kwargs["enable_thinking"] = enable_thinking
            render_kwargs["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }

        rendered_text = super().apply_chat_template(
            preprocessed,
            tokenize=False,
            **render_kwargs,
        )
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
    """Check if the model uses dynamic-resolution image sizing."""

    if not hasattr(config, "vision_config"):
        return False
    vision_args = getattr(config.vision_config, "args", None)
    if vision_args is None:
        return False
    return "min_num_patches" in vision_args and "max_num_patches" in vision_args
