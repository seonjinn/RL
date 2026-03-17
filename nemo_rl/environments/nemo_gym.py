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
import subprocess
import base64
import io
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
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        nemo_rl_message_log = []
        seen_token_ids = torch.tensor([], dtype=torch.int64)

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
                user_token_ids = torch.cat(
                    [msg["token_ids"] for msg in original_message_log], dim=0
                )
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
    nemo_gym_example: dict, idx: int, processor: Optional[Any] = None
) -> DatumSpec:
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
        elif item.get("type") == "input_text":
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

    # Extract multimodal keys (pixel_values, etc.)
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message_both:
            user_message[key] = PackedTensor(
                message_both[key],
                dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    if "imgs_sizes" in message_both:
        user_message["imgs_sizes"] = PackedTensor(message_both["imgs_sizes"], dim_to_pack=0)

    if "token_type_ids" in message_both:
        user_message["token_type_ids"] = message_both["token_type_ids"][0]

    message_log = [user_message]
    length = sum(len(m["token_ids"]) for m in message_log)

    return DatumSpec(
        message_log=message_log,
        length=length,
        extra_env_info=nemo_gym_example,
        loss_multiplier=1.0,
        idx=idx,
        task_name="nemo_gym",
        stop_strings=None,
        token_ids=[],
    )
