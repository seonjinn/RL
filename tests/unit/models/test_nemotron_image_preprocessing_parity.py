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

import copy
import os
from functools import lru_cache

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from nemo_rl.models.nano_v3_vl.dynamic_resolution_processor import (
    DynamicResolutionProcessor,
)

DEFAULT_MODEL = (
    "/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/"
    "mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-truncated"
)
DEFAULT_FIXTURE_PROMPT = (
    "While hanging Christmas lights for neighbors, Bella counted the number "
    "of broken lights on each string. How many strings had exactly 16 broken "
    "lights?\nPlease answer the question and put the final answer within \\boxed{}."
)
IMAGE_CASES = [
    pytest.param((1, 1), id="square_very_small_1x1"),
    pytest.param((65, 65), id="square_small_65x65"),
    pytest.param((513, 513), id="square_medium_513x513"),
    pytest.param((2048, 2048), id="square_large_2048x2048"),
    pytest.param((4097, 4097), id="square_very_large_4097x4097"),
    pytest.param((13, 5), id="wide_very_small_13x5"),
    pytest.param((97, 49), id="wide_small_97x49"),
    pytest.param((1025, 513), id="wide_medium_1025x513"),
    pytest.param((2048, 512), id="wide_large_2048x512"),
    pytest.param((4097, 1025), id="wide_very_large_4097x1025"),
    pytest.param((5, 13), id="tall_very_small_5x13"),
    pytest.param((49, 97), id="tall_small_49x97"),
    pytest.param((513, 1025), id="tall_medium_513x1025"),
    pytest.param((512, 2048), id="tall_large_512x2048"),
    pytest.param((1025, 4097), id="tall_very_large_1025x4097"),
]


def _build_messages(question: str) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "/no_think"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": question},
            ],
        },
    ]


@lru_cache(maxsize=1)
def _load_model_assets(model_name: str):
    if not os.path.exists(model_name):
        pytest.skip(f"Model checkpoint not found: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Keep the test focused on the real image path. The audio extractor is
    # unrelated to this parity check and currently depends on container packaging.
    if getattr(config, "sound_config", None) is not None:
        config = copy.deepcopy(config)
        config.sound_config = None
    return tokenizer, config


def _make_test_image(mode: str, image_dims: tuple[int, int]) -> Image.Image:
    width, height = image_dims
    seed = width * 1_000_003 + height * 10_007 + (0 if mode == "RGB" else 1)
    rng = np.random.default_rng(seed)
    channels = 4 if mode == "RGBA" else 3
    array = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    return Image.fromarray(array, mode=mode)


def _get_prompt_text(tokenizer) -> str:
    return tokenizer.apply_chat_template(
        _build_messages(DEFAULT_FIXTURE_PROMPT),
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize_vllm_processed_text(tokenizer, prompt_text: str, replacement: str) -> torch.Tensor:
    processed_text = prompt_text.replace("<image>", replacement, 1)
    return tokenizer(
        processed_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]


@pytest.mark.vllm
@pytest.mark.parametrize("image_dims", IMAGE_CASES)
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_dynamic_preprocess_matches_vllm_017(
    mode: str, image_dims: tuple[int, int]
):
    vllm_nano = pytest.importorskip("vllm.model_executor.models.nano_nemotron_vl")
    model_name = os.environ.get("NRL_NEMOTRON_VL_MODEL", DEFAULT_MODEL)
    tokenizer, config = _load_model_assets(model_name)

    image = _make_test_image(mode, image_dims)
    prompt_text = _get_prompt_text(tokenizer)
    super_processor = DynamicResolutionProcessor(tokenizer, copy.deepcopy(config))
    vllm_processor = vllm_nano.NanoNemotronVLProcessor(
        config=copy.deepcopy(config),
        tokenizer=tokenizer,
        max_model_len=16384,
    )

    sans_images = prompt_text.replace("<image>", "")
    text_prompt_length = len(
        tokenizer(sans_images, add_special_tokens=False).input_ids
    )
    assert vllm_processor.dynamic_tiler is not None
    num_tokens_available = vllm_processor.dynamic_tiler.max_num_tokens_available(
        text_prompt_length
    )

    super_batch = super_processor(
        text=prompt_text,
        images=image,
        num_tokens_available=num_tokens_available,
        return_tensors="pt",
    )
    vllm_batch = vllm_processor(text=prompt_text, images=image, return_tensors="pt")

    super_sizes = super_batch["imgs_sizes"].tolist()
    assert super_sizes == [list(size) for size in vllm_batch["imgs_sizes"]]

    target_h, target_w = super_sizes[0]
    super_tensor = super_batch["pixel_values"][0, :, :target_h, :target_w]
    vllm_tensor = vllm_batch["pixel_values_flat"][0]
    torch.testing.assert_close(super_tensor, vllm_tensor, rtol=0, atol=0)

    vllm_input_ids = _tokenize_vllm_processed_text(
        tokenizer,
        prompt_text,
        vllm_processor.get_image_repl(
            vllm_batch["num_tokens_per_image"][0],
            None,
        ).full,
    )
    torch.testing.assert_close(
        super_batch["input_ids"],
        vllm_input_ids,
    )
    assert vllm_batch["num_tokens_per_image"] == [
        super_processor.compute_num_embeddings(target_h, target_w)
    ]


@pytest.mark.vllm
@pytest.mark.parametrize("image_dims", IMAGE_CASES)
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_vllm_nemotron_processor_stays_dynamic_with_max_num_tiles(
    mode: str, image_dims: tuple[int, int]
):
    vllm_nano = pytest.importorskip("vllm.model_executor.models.nano_nemotron_vl")
    model_name = os.environ.get("NRL_NEMOTRON_VL_MODEL", DEFAULT_MODEL)
    tokenizer, config = _load_model_assets(model_name)

    image = _make_test_image(mode, image_dims)
    prompt_text = _get_prompt_text(tokenizer)
    max_num_tiles = 4
    super_processor = DynamicResolutionProcessor(tokenizer, copy.deepcopy(config))
    vllm_processor = vllm_nano.NanoNemotronVLProcessor(
        config=copy.deepcopy(config),
        tokenizer=tokenizer,
        max_model_len=16384,
    )

    super_batch = super_processor(
        text=prompt_text,
        images=image,
        max_num_tiles=max_num_tiles,
        return_tensors="pt",
    )
    direct_vllm_batch = vllm_processor(
        text=prompt_text,
        images=image,
        max_num_tiles=max_num_tiles,
        return_tensors="pt",
    )

    assert isinstance(direct_vllm_batch["pixel_values_flat"], list)
    assert "imgs_sizes" in direct_vllm_batch
    assert "num_tokens_per_image" in direct_vllm_batch
    assert "image_num_patches" not in direct_vllm_batch
    # Dynamic mode can legitimately land on 512x512 for small square images
    # because the processor enforces a minimum patch budget. The reliable
    # signal is the output schema: dynamic mode returns imgs_sizes +
    # num_tokens_per_image, whereas static mode returns image_num_patches.

    static_num_patches = int(torch.as_tensor(super_batch["image_num_patches"])[0].item())
    static_vllm_input_ids = _tokenize_vllm_processed_text(
        tokenizer,
        prompt_text,
        vllm_processor.get_image_repl(
            static_num_patches * vllm_processor.num_image_token,
            static_num_patches,
        ).full,
    )
    torch.testing.assert_close(
        super_batch["input_ids"],
        static_vllm_input_ids,
    )


@pytest.mark.vllm
@pytest.mark.parametrize("image_dims", IMAGE_CASES)
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_static_helper_preprocess_matches_vllm_017(
    mode: str, image_dims: tuple[int, int]
):
    vllm_nano = pytest.importorskip("vllm.model_executor.models.nano_nemotron_vl")
    model_name = os.environ.get("NRL_NEMOTRON_VL_MODEL", DEFAULT_MODEL)
    tokenizer, config = _load_model_assets(model_name)

    image = _make_test_image(mode, image_dims)
    prompt_text = _get_prompt_text(tokenizer)
    max_num_tiles = 4
    super_processor = DynamicResolutionProcessor(tokenizer, copy.deepcopy(config))
    vllm_processor = vllm_nano.NanoNemotronVLProcessor(
        config=copy.deepcopy(config),
        tokenizer=tokenizer,
        max_model_len=16384,
    )

    super_batch = super_processor(
        text=prompt_text,
        images=image,
        max_num_tiles=max_num_tiles,
        return_tensors="pt",
    )
    vllm_pixel_values_flat = vllm_nano.image_to_pixel_values(
        image,
        input_size=vllm_processor.image_size,
        max_num=max_num_tiles,
        use_thumbnail=vllm_processor.use_thumbnail,
        idx=0,
    )
    vllm_pixel_values_flat = vllm_nano.input_conditioner(
        vllm_pixel_values_flat,
        vllm_processor.norm_mean,
        vllm_processor.norm_std,
    )

    torch.testing.assert_close(
        super_batch["pixel_values_flat"],
        vllm_pixel_values_flat,
        rtol=0,
        atol=0,
    )
    static_num_patches = int(torch.as_tensor(super_batch["image_num_patches"])[0].item())
    static_vllm_input_ids = _tokenize_vllm_processed_text(
        tokenizer,
        prompt_text,
        vllm_processor.get_image_repl(
            static_num_patches * vllm_processor.num_image_token,
            static_num_patches,
        ).full,
    )
    torch.testing.assert_close(super_batch["input_ids"], static_vllm_input_ids)
