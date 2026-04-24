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

from __future__ import annotations

import json
import os
from typing import Any, cast

import torch
import vllm
from PIL import Image
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation
from nemo_rl.models.generation.vllm.vllm_generation import _build_compact_mm_payload

DEFAULT_MODEL = (
    "/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/"
    "mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-truncated"
)
DEFAULT_FIXTURE_IMAGE = (
    "/lustre/fs1/portfolios/coreai/users/aroshanghias/data/mmpr_miniscule/processed/"
    "MMPR-Tiny/images/10189_0.png"
)
DEFAULT_FIXTURE_PROMPT = (
    "While hanging Christmas lights for neighbors, Bella counted the number "
    "of broken lights on each string. How many strings had exactly 16 broken "
    "lights?\nPlease answer the question and put the final answer within \\boxed{}."
)


def _emit(stage: str, **payload: object) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


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


def main() -> int:
    model_name = os.environ.get("NRL_NEMOTRON_VL_MODEL", DEFAULT_MODEL)
    fixture_image = os.environ.get("NRL_NEMOTRON_VL_FIXTURE_IMAGE", DEFAULT_FIXTURE_IMAGE)
    fixture_prompt = os.environ.get(
        "NRL_NEMOTRON_VL_FIXTURE_PROMPT", DEFAULT_FIXTURE_PROMPT
    )
    max_new_tokens = int(os.environ.get("NRL_GATE6_MAX_NEW_TOKENS", "16"))

    # TODO: Once the container ships a matching torchvision/vLLM stack, restore
    # the full Nemotron processor/registration path for this smoke instead of the
    # tokenizer-only workaround below.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt_text = tokenizer.apply_chat_template(
        _build_messages(fixture_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer([prompt_text], return_tensors="pt")

    with Image.open(fixture_image) as image:
        width, height = image.size

    raw_data = BatchedDataDict(
        {
            "input_ids": tokenized.input_ids,
            "input_lengths": torch.tensor([tokenized.input_ids.shape[1]]),
            "vllm_content": [prompt_text],
            "vllm_images": [[fixture_image]],
            "imgs_sizes": [[[height, width]]],
        }
    )
    raw_prompt = format_prompt_for_vllm_generation(raw_data, sample_idx=0)
    assert isinstance(raw_prompt, dict)

    compact_data = BatchedDataDict(
        {
            "input_ids": tokenized.input_ids,
            "input_lengths": torch.tensor([tokenized.input_ids.shape[1]]),
            "imgs_sizes": raw_data["imgs_sizes"],
            "vllm_mm_compact_payload": _build_compact_mm_payload(
                SlicedDataDict(dict(raw_data))
            ),
        }
    )
    compact_prompt = format_prompt_for_vllm_generation(compact_data, sample_idx=0)
    assert isinstance(compact_prompt, dict)
    assert compact_prompt == raw_prompt, "compact prompt must match raw prompt"
    mm_data = cast(dict[str, Any], compact_prompt.get("multi_modal_data", {}))
    image_payload = mm_data.get("image")
    image_count = len(image_payload) if isinstance(image_payload, list) else 1

    _emit(
        "vllm_runtime_init",
        model_name=model_name,
        fixture_image=fixture_image,
        vllm_version=getattr(vllm, "__version__", "unknown"),
        prompt_type="multimodal" if "multi_modal_data" in compact_prompt else "token_ids",
        image_count=image_count,
        mm_processor_kwargs=compact_prompt.get("mm_processor_kwargs"),
    )

    llm = vllm.LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_num_seqs=1,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate([compact_prompt], sampling_params)
    generated = outputs[0].outputs[0]

    _emit(
        "vllm_generate_result",
        output_token_count=len(generated.token_ids),
        text_snippet=generated.text[:200],
    )
    print("SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
