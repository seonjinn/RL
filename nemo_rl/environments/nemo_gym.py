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
import base64
import copy
import io
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import ray
import requests
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
)
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_PORT_RANGE_HIGH,
    DEFAULT_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


def get_nemo_gym_uv_cache_dir() -> Optional[str]:
    """Return the uv cache directory inside a container, or None outside one.

    Inside a container (NRL_CONTAINER=1), returns the uv cache location so Gym
    stores its caches in the expected shared path. Returns None outside a
    container, meaning the caller should omit this arg and let Gym create the
    cache locally (the default when you may not be able to write to /opt).
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def get_nemo_gym_venv_dir() -> Optional[str]:
    """Return the NeMo Gym venv directory from NEMO_GYM_VENV_DIR, or None.

    Returns the value of NEMO_GYM_VENV_DIR if set, otherwise None. When None
    the caller should omit this arg and let Gym create venvs locally (the
    default when a container is not used since you may not be able to write
    to /opt).
    """
    return os.environ.get("NEMO_GYM_VENV_DIR")


def resolve_to_image(image_path_or_image: "str | Image.Image") -> Image.Image:
    """Resolve a local path / URL / data-URL to a PIL.Image."""
    if isinstance(image_path_or_image, Image.Image):
        return image_path_or_image

    if image_path_or_image.startswith(("http://", "https://")):
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    elif image_path_or_image.startswith("data:"):
        _, encoded = image_path_or_image.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    else:
        return Image.open(image_path_or_image).convert("RGB")


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def encode_images_in_examples(nemo_gym_examples: list[dict]) -> list[dict]:
    """Walk examples and replace local image paths with base64 data URLs.

    Operates in-place on each example's responses_create_params.input[].content[]
    items of type 'input_image'.
    """
    for example in nemo_gym_examples:
        input_items = example.get("responses_create_params", {}).get("input", [])
        for item in input_items:
            for part in item.get("content", []):
                if not isinstance(part, dict) or part.get("type") != "input_image":
                    continue
                url = part.get("image_url", "")
                if url.startswith(("http://", "https://", "data:", "file://")):
                    continue
                # Local filesystem path — encode as data URL
                part["image_url"] = image_to_data_url(resolve_to_image(url))
    return nemo_gym_examples


def _normalize_image_placeholders(text: str, replacement: str) -> str:
    text = text.replace("<img>", "")
    text = text.replace("</img>", "")
    text = text.replace("<image>", replacement)
    return text


def _strip_image_placeholders(text: str) -> str:
    return _normalize_image_placeholders(text, "")


def _count_image_payloads(nemo_gym_example: dict) -> int:
    count = 0
    input_items = nemo_gym_example.get("responses_create_params", {}).get("input", [])
    for item in input_items:
        content = item.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("input_image", "image", "image_url"):
                count += 1
                continue
            if part.get("image_url") is not None or part.get("image") is not None:
                count += 1
    return count


def _example_has_image_payload(nemo_gym_example: dict) -> bool:
    return _count_image_payloads(nemo_gym_example) > 0


def _strip_text_image_placeholders_in_example(nemo_gym_example: dict) -> dict:
    sanitized = copy.deepcopy(nemo_gym_example)
    input_items = sanitized.get("responses_create_params", {}).get("input", [])
    for item in input_items:
        content = item.get("content", "")
        if isinstance(content, str):
            item["content"] = _strip_image_placeholders(content)
            continue
        if not isinstance(content, list):
            continue
        for idx, part in enumerate(content):
            if isinstance(part, str):
                content[idx] = _strip_image_placeholders(part)
                continue
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and part.get("type") in (
                "input_text",
                "text",
            ):
                part["text"] = _strip_image_placeholders(text_value)
    return sanitized


def sanitize_nemo_gym_example_image_placeholders(nemo_gym_example: dict) -> dict:
    """Normalize literal image placeholders in text fields.

    The OpenAI-style content list already carries images as structured
    ``input_image`` items. The HF processor turns those items into image
    blocks, so literal ``<image>`` strings in adjacent text would double-count
    images and can trip the dynamic-resolution placeholder assertion. For
    multimodal rows, strip the literal anchors and let the structured image
    items drive placement. For text-only rows, replace ``<image>`` with the word
    ``image`` so special image tokens do not enter a prompt without payloads.
    """
    image_payload_count = _count_image_payloads(nemo_gym_example)
    sanitized = copy.deepcopy(nemo_gym_example)

    replacement = "" if image_payload_count > 0 else "image"
    input_items = sanitized.get("responses_create_params", {}).get("input", [])
    for item in input_items:
        content = item.get("content", "")
        if isinstance(content, str):
            item["content"] = _normalize_image_placeholders(content, replacement)
            continue
        if not isinstance(content, list):
            continue
        for idx, part in enumerate(content):
            if isinstance(part, str):
                content[idx] = _normalize_image_placeholders(part, replacement)
                continue
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str):
                part["text"] = _normalize_image_placeholders(text_value, replacement)
    return sanitized


def _make_overlength_filtered_nemo_gym_example(nemo_gym_example: dict) -> dict:
    """Replace the rollout request with a tiny text-only fallback.

    NeMo-Gym uses ``extra_env_info`` as the source-of-truth request payload during
    rollout, so setting ``loss_multiplier=0`` alone is not enough to prevent the
    original oversized prompt from reaching vLLM. For filtered samples we preserve
    the example metadata, but swap the model input with a minimal user message.
    """
    filtered = copy.deepcopy(nemo_gym_example)
    filtered.setdefault("responses_create_params", {})
    filtered["responses_create_params"]["input"] = [
        {
            "role": "user",
            "type": "message",
            "content": [
                {
                    "type": "input_text",
                    "text": "Discarded overlength sample.",
                }
            ],
        }
    ]
    return filtered


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    ray_num_gpus_per_node: Optional[int]
    ray_namespace: Optional[str]
    initial_global_config_dict: Dict[str, Any]
    invalid_tool_call_patterns: Optional[List[str]]  # Substrings in assistant text content that indicate an invalid tool call (default: ["<tool_call>", "</tool_call>", "<function_call>", "</function_call>"])
    thinking_tags: Optional[List[str]]  # Thinking tags to check for malformed usage (default: ["<think>", "</think>"])


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg

    def _spinup(self) -> None:
        self.node_ip = _get_node_ip_local()
        port_range_low = self.cfg.get("port_range_low", DEFAULT_PORT_RANGE_LOW)
        port_range_high = self.cfg.get("port_range_high", DEFAULT_PORT_RANGE_HIGH)
        self.head_server_port = _get_free_port_local(port_range_low, port_range_high)

        from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
        from nemo_gym.rollout_collection import RolloutCollectionHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        from omegaconf import DictConfig

        RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
        assert __file__.endswith(RELATIVE_PATH)

        initial_global_config_dict = (
            self.cfg.get("initial_global_config_dict") or dict()
        )
        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = (
            "dummy_key"  # No key necessary for training.
        )
        initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]

        # Gym servers default to 5000-5999, below the OS ephemeral floor (9000
        # on OCI-HSG).  See ray.sub port layout comment for the full map.
        _gym_port_low = self.cfg.get("port_range_low", 5000)
        _gym_port_high = self.cfg.get("port_range_high", 5999)
        if _gym_port_low < 5000 or _gym_port_high > 5999:
            print(
                f"WARNING: Gym port range [{_gym_port_low}, {_gym_port_high}) is outside "
                f"the expected 5000-5999 band. Check ray.sub port layout for conflicts."
            )
        initial_global_config_dict["port_range_low"] = _gym_port_low
        initial_global_config_dict["port_range_high"] = _gym_port_high

        initial_global_config_dict.setdefault(
            "global_aiohttp_connector_limit_per_host", 16_384
        )
        initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)
        print(
            f"""Set global_aiohttp_connector_limit_per_host={initial_global_config_dict["global_aiohttp_connector_limit_per_host"]} and global_aiohttp_connector_limit={initial_global_config_dict["global_aiohttp_connector_limit"]}.
Depending on your data shape, you may want to change these values."""
        )

        # Get Ray head node address if Ray is initialized
        assert ray.is_initialized(), (
            "Ray must be initialized before using NeMo-Gym environment"
        )
        ray_context = ray.get_runtime_context()
        assert ray_context.gcs_address, "Ray must have a GCS address"

        initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address
        print(f"Ray head node address: {ray_context.gcs_address}")

        ray_namespace = self.cfg.get("ray_namespace", None)
        if ray_namespace is not None:
            initial_global_config_dict["ray_namespace"] = ray_namespace
            print(f"Ray namespace: {ray_namespace}")

        initial_global_config_dict["ray_num_gpus_per_node"] = self.cfg[
            "ray_num_gpus_per_node"
        ]
        print(
            f"Ray num GPUs per node: {initial_global_config_dict['ray_num_gpus_per_node']}"
        )

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rollout_max_attempts_to_avoid_lp_nan = initial_global_config_dict.pop(
            "rollout_max_attempts_to_avoid_lp_nan", 1
        )

        assert self.rollout_max_attempts_to_avoid_lp_nan >= 1, (
            "`rollout_max_attempts_to_avoid_lp_nan` must be at least 1"
        )

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute()
                / "nemo_gym_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            ),
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
        original_message_logs: Optional[list[list[dict]]] = None,
        max_total_tokens_per_sample: Optional[int] = None,
    ) -> list[dict]:
        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens()

        timer = Timer(context={"worker": "nemo_gym"})

        encode_images_in_examples(nemo_gym_examples)

        timer.start("_run_rollouts_total")
        max_attempts, trial = self.rollout_max_attempts_to_avoid_lp_nan, 0
        while trial < max_attempts:
            nemo_gym_num_rows = len(nemo_gym_examples)
            nemo_gym_result_iterator = self.rch.run_examples(
                examples=nemo_gym_examples,
                head_server_config=self.head_server_config,
            )

            nemo_rl_rowidxs = []
            nemo_rl_results = []
            for task in nemo_gym_result_iterator:
                with timer.time(label=f"{timer_prefix}/await_results"):
                    nemo_gym_row, nemo_gym_result = await task

                with timer.time(label=f"{timer_prefix}/postprocess_results"):
                    rowidx = nemo_gym_row["_rowidx"]
                    original_message_log = (
                        original_message_logs[rowidx]
                        if original_message_logs is not None
                        else None
                    )
                    nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                        nemo_gym_result,
                        tokenizer,
                        original_message_log=original_message_log,
                        max_total_tokens_per_sample=max_total_tokens_per_sample,
                        rowidx=rowidx,
                    )

                nemo_rl_rowidxs.append(rowidx)
                nemo_rl_results.append(nemo_rl_result)

            # determine if generation_logprobs contain NaN; if not, break;
            logprob_contains_nan = False
            for nemo_rl_result in nemo_rl_results:
                for message in nemo_rl_result["message_log"]:
                    if (
                        "generation_logprobs" in message
                        and message["generation_logprobs"] is not None
                    ):
                        if torch.isnan(message["generation_logprobs"]).any():
                            logprob_contains_nan = True
                            break
            if logprob_contains_nan:
                trial += 1
                print(
                    f"Generation logprobs contain NaN; retrying... (trial {trial}/{max_attempts})"
                )
                continue
            else:
                break

        nemo_rl_sort_results = [None] * nemo_gym_num_rows
        for rowidx, result in zip(nemo_rl_rowidxs, nemo_rl_results):
            nemo_rl_sort_results[rowidx] = result
        nemo_rl_results = nemo_rl_sort_results

        timer.stop("_run_rollouts_total")
        timing_metrics = timer.get_timing_metrics("sum")
        total_time = timing_metrics.pop("_run_rollouts_total")
        timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
            100 * timing_metrics[f"{timer_prefix}/postprocess_results"] / total_time
        )

        return nemo_rl_results, timing_metrics

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self,
        nemo_gym_result: dict,
        tokenizer: PreTrainedTokenizerBase,
        original_message_log: Optional[list[dict]] = None,
        max_total_tokens_per_sample: Optional[int] = None,
        rowidx: Optional[int] = None,
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        nemo_rl_message_log = []
        seen_token_ids = torch.tensor([], dtype=torch.int64)
        # Track cumulative token count across turns to enforce the policy token
        # budget (max_total_tokens_per_sample). If a turn would push us past the
        # budget, we truncate the assistant tokens and flag the sample.
        current_policy_token_count = 0
        truncated_for_policy_max_length = False

        # Extract multimodal data (pixel_values, imgs_sizes, etc.) from the original
        # message_log. The original message_log was created by the HF processor and
        # contains pixel_values that are not available in the vLLM nemo_gym response.
        multimodal_data: Dict[str, Any] = {}
        if original_message_log:
            for msg in original_message_log:
                if msg.get("role") == "user":
                    for key in list(msg.keys()):
                        if key not in ("role", "content", "token_ids"):
                            multimodal_data[key] = msg[key]

        batch_decode_items = []  # Collect (output_item_dict, prompt_token_ids, generation_token_ids) for batch decode
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            prompt_token_ids_tensor = torch.tensor(
                output_item_dict["prompt_token_ids"], dtype=torch.int64
            )
            n_seen = len(seen_token_ids)
            if n_seen > 0:
                assert torch.equal(
                    seen_token_ids, prompt_token_ids_tensor[:n_seen]
                ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids.tolist()}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
"""

            n_seen = len(seen_token_ids)

            # Create tensors for new tokens
            new_prompt_token_ids = torch.tensor(
                output_item_dict["prompt_token_ids"][n_seen:], dtype=torch.int64
            )
            generation_token_ids = torch.tensor(
                output_item_dict["generation_token_ids"], dtype=torch.int64
            )
            generation_logprobs = torch.tensor(
                output_item_dict["generation_log_probs"], dtype=torch.float32
            )

            # On the first multimodal turn, override the user's token_ids with the HF
            # processor's version (including <img>/<image>×N/</img> wrappers) so that
            # Megatron's collapse_multimodal_tokens() can identify image regions and
            # honor pixel_values. seen_token_ids continues to track the original vLLM
            # ids (without wrappers) so the prefix assertion above stays valid across
            # multi-turn rollouts.
            if multimodal_data and not nemo_rl_message_log and original_message_log:
                hf_token_ids = torch.cat(
                    [msg["token_ids"] for msg in original_message_log], dim=0
                )
                # Opt-in debug: when HF-rendered prompt tokens diverge from vLLM's,
                # print the first mismatch window so we can spot tokenizer/template
                # drift early. Disabled by default; set NRL_DEBUG_PROMPT_TOKENS=1.
                if os.environ.get("NRL_DEBUG_PROMPT_TOKENS", "0") == "1":
                    hf_prompt_token_ids = hf_token_ids.tolist()
                    vllm_prompt_token_ids = new_prompt_token_ids.tolist()
                    min_len = min(len(hf_prompt_token_ids), len(vllm_prompt_token_ids))
                    first_diff_idx = next(
                        (
                            idx
                            for idx in range(min_len)
                            if hf_prompt_token_ids[idx] != vllm_prompt_token_ids[idx]
                        ),
                        None,
                    )
                    has_mismatch = (
                        first_diff_idx is not None
                        or len(hf_prompt_token_ids) != len(vllm_prompt_token_ids)
                    )
                    if has_mismatch:
                        window_start = max(0, (first_diff_idx or min_len) - 5)
                        window_end = min(
                            max(len(hf_prompt_token_ids), len(vllm_prompt_token_ids)),
                            (first_diff_idx or min_len) + 6,
                        )
                        print(
                            "[NEMO_GYM_PROMPT_TOKEN_MISMATCH] "
                            f"rowidx={rowidx} "
                            f"len_hf={len(hf_prompt_token_ids)} "
                            f"len_vllm={len(vllm_prompt_token_ids)} "
                            f"first_diff={first_diff_idx} "
                            f"hf_window={hf_prompt_token_ids[window_start:window_end]} "
                            f"vllm_window={vllm_prompt_token_ids[window_start:window_end]}",
                            flush=True,
                        )
                user_token_ids = hf_token_ids
            else:
                user_token_ids = new_prompt_token_ids

            user_message: Dict[str, Any] = {
                "role": "user",
                "content": "",
                "token_ids": user_token_ids,
            }
            # Attach multimodal data only on the first turn (it represents the initial
            # image input from the original DatumSpec).
            if multimodal_data and len(nemo_rl_message_log) == 0:
                user_message.update(multimodal_data)

            # Policy token-budget enforcement: if appending this turn's assistant
            # tokens would push us past `max_total_tokens_per_sample`, truncate them
            # and flag the sample so downstream metrics can distinguish vLLM
            # max-tokens truncation from policy-budget truncation.
            if max_total_tokens_per_sample is not None:
                remaining_assistant_tokens = max(
                    0,
                    max_total_tokens_per_sample
                    - current_policy_token_count
                    - len(user_token_ids),
                )
                if len(generation_token_ids) > remaining_assistant_tokens:
                    generation_token_ids = generation_token_ids[
                        :remaining_assistant_tokens
                    ]
                    generation_logprobs = generation_logprobs[
                        :remaining_assistant_tokens
                    ]
                    truncated_for_policy_max_length = True

            nemo_rl_message_log.append(user_message)

            # Valid tool calls go through the structured API (tool_calls field) and get
            # executed by NeMo-Gym. If tool call patterns appear in the text content instead,
            # the call was invalid and never executed — flag it so training can penalize it.
            invalid_tool_call_patterns = self.cfg.get("invalid_tool_call_patterns") or ["<tool_call>", "</tool_call>", "<function_call>", "</function_call>"]
            is_invalid_tool_call = False

            # NeMo-Gym only attaches generation_token_ids to the last output item of a
            # model call (see vllm_model/app.py postprocess_chat_response). So this item
            # is guaranteed to be the final thing the model produced for this turn.
            # If it's a reasoning item, the model output only reasoning (no content/tool calls).
            is_output_message = "content" in output_item_dict and len(output_item_dict["content"]) > 0 and "text" in output_item_dict["content"][0]
            is_reasoning_message = output_item_dict.get("type") == "reasoning" and len(output_item_dict["summary"]) > 0 and "text" in output_item_dict["summary"][0]

            if is_output_message:
                assistant_message_content = output_item_dict["content"][0]["text"]
                if any(pattern in assistant_message_content for pattern in invalid_tool_call_patterns):
                    is_invalid_tool_call = True
            elif is_reasoning_message:
                assistant_message_content = output_item_dict["summary"][0]["text"]
                if any(pattern in assistant_message_content for pattern in invalid_tool_call_patterns):
                    is_invalid_tool_call = True

            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": generation_token_ids,
                    "generation_logprobs": generation_logprobs,
                    "is_invalid_tool_call": is_invalid_tool_call,
                }
            )
            current_policy_token_count += len(user_token_ids) + len(
                generation_token_ids
            )

            seen_token_ids = torch.cat(
                [seen_token_ids, new_prompt_token_ids, generation_token_ids]
            )

            # We pop to remove larger tensors from logging.
            prompt_token_ids_for_decode = output_item_dict.pop("prompt_token_ids")
            generation_token_ids_for_decode = output_item_dict.pop(
                "generation_token_ids"
            )

            output_item_dict.pop("generation_log_probs")

            batch_decode_items.append(
                (
                    output_item_dict,
                    prompt_token_ids_for_decode,
                    generation_token_ids_for_decode,
                )
            )

            # Stop processing further turns once the policy budget has been hit —
            # any remaining output_items would only contribute discarded tokens.
            if truncated_for_policy_max_length:
                break

        if batch_decode_items:
            prompt_token_ids_batch = [item[1] for item in batch_decode_items]
            generation_token_ids_batch = [item[2] for item in batch_decode_items]

            prompt_strs = tokenizer.batch_decode(prompt_token_ids_batch)
            generation_strs = tokenizer.batch_decode(generation_token_ids_batch)

            for (output_item_dict, _, _), prompt_str, generation_str in zip(
                batch_decode_items, prompt_strs, generation_strs
            ):
                output_item_dict["prompt_str"] = prompt_str
                output_item_dict["generation_str"] = generation_str

        if not nemo_rl_message_log:
            input_messages = nemo_gym_result["responses_create_params"]["input"]
            prompt_token_ids = tokenizer.apply_chat_template(
                input_messages, tokenize=True
            )
            raise ValueError(
                f"NeMo Gym returned a result with no generation data. "
                f"This typically means the prompt for the first turn already exceeds the vLLM max_model_len, "
                f"so vLLM rejected the request before any tokens could be generated.\n"
                f"  Prompt length: {len(prompt_token_ids)} tokens.\n"
                f"  → Fix: increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len` "
                f"to a value larger than {len(prompt_token_ids)}."
            )

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
            "truncated_for_policy_max_length": truncated_for_policy_max_length,
        }

    def shutdown(self) -> None:
        self.rh.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo-Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


########################################
# Global config utils
########################################


def setup_nemo_gym_config(config, tokenizer) -> None:
    generation_config = config["policy"]["generation"]

    # Enable the http server. Requires both async engine and the expose_http_server flag
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True

    # Stop strings or token ids are not supported
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None


########################################
# Data utils
########################################


# We do some light preprocessing here to make our data format compatible with nemo rl format
def nemo_gym_example_to_nemo_rl_datum_spec(
    nemo_gym_example: dict,
    idx: int,
    processor: Optional[Any] = None,
    max_seq_length: Optional[int] = None,
) -> DatumSpec:
    nemo_gym_example = sanitize_nemo_gym_example_image_placeholders(nemo_gym_example)

    if processor is None:
        return DatumSpec(
            message_log=[
                {"role": "user", "content": "", "token_ids": torch.tensor([])}
            ],  # Fake message
            length=0,
            extra_env_info=nemo_gym_example,
            loss_multiplier=1.0,
            idx=idx,
            task_name="nemo_gym",
            stop_strings=None,
            token_ids=[],
        )

    # Extract messages from nemo_gym format
    input_messages = nemo_gym_example.get("responses_create_params", {}).get("input", [])

    # Build user message only (no system message — NeMo-Gym sends empty system to vLLM)
    user_message = {"role": "user", "content": []}

    for msg in input_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            if isinstance(content, str):
                user_message["content"].append({"type": "text", "text": content})
            elif isinstance(content, list):
                user_message["content"] = content

    # Build user_message with PIL.Image objects for the HF processor
    user_message_with_images = {"role": "user", "content": []}
    for item in user_message["content"]:
        if item.get("type") in ("input_image", "image_url"):
            url = item.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            if url:
                pil_image = resolve_to_image(url)
                user_message_with_images["content"].append({"type": "image", "image": pil_image})
        elif item.get("type") in ("input_text", "text"):
            user_message_with_images["content"].append({"type": "text", "text": item.get("text", "")})
        else:
            user_message_with_images["content"].append(item)

    # Process user message with images
    message_both = processor.apply_chat_template(
        [user_message_with_images],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Keep the full HF processor token layout (with <img>/<image>×N/</img>) so that
    # collapse_multimodal_tokens() in Megatron can identify image regions and preserve
    # pixel_values for proper image embedding. Stripping wrappers here causes Megatron
    # to drop pixel_values and either crash or compute logprobs without image context.
    user_message["token_ids"] = message_both["input_ids"][0]

    # Extract multimodal keys (pixel_values, etc.). NanoV3-style processors
    # return flattened tiles under these extra keys, which are not always listed
    # in processor.model_input_names.
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    extra_multimodal_keys = {"pixel_values_flat", "image_num_patches"}
    all_multimodal_keys = list(
        dict.fromkeys(
            list(multimodal_keys)
            + [key for key in extra_multimodal_keys if key in message_both]
        )
    )
    for key in all_multimodal_keys:
        if key in message_both:
            user_message[key] = PackedTensor(
                message_both[key],
                dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    if "token_type_ids" in message_both:
        user_message["token_type_ids"] = message_both["token_type_ids"][0]

    message_log = [user_message]
    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    extra_env_info = (
        _strip_text_image_placeholders_in_example(nemo_gym_example)
        if _example_has_image_payload(nemo_gym_example)
        else copy.deepcopy(nemo_gym_example)
    )

    if "imgs_sizes" in message_both:
        imgs_sizes = message_both["imgs_sizes"].to(dtype=torch.int32)
        user_message["imgs_sizes"] = PackedTensor(imgs_sizes, dim_to_pack=0)

    if max_seq_length is not None and length >= max_seq_length:
        print(f"Discarding sample with length {length} >= {max_seq_length}")

        tokenizer = processor.tokenizer
        image_token_ids = [
            tokenizer.convert_tokens_to_ids(token)
            for token in ("<img>", "<image>", "</img>")
        ]
        image_token_ids = [token_id for token_id in image_token_ids if token_id is not None]

        for chat_message in message_log:
            token_ids = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
            for img_token_id in image_token_ids:
                token_ids = token_ids[token_ids != img_token_id]
            chat_message["token_ids"] = token_ids
            for key, value in chat_message.items():
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)

        loss_multiplier = 0.0
        length = sum(len(m["token_ids"]) for m in message_log)
        extra_env_info = _make_overlength_filtered_nemo_gym_example(nemo_gym_example)

    return DatumSpec(
        message_log=message_log,
        length=length,
        extra_env_info=extra_env_info,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="nemo_gym",
        stop_strings=None,
        token_ids=[],
    )
