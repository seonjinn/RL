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

import argparse
import base64
import os
import pprint
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

from nemo_rl.algorithms.mpo import MasterConfig, mpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.response_datasets.mmpr import (
    MMPRDataset,
    format_mmpr_dataset,
)
from nemo_rl.data.datasets.response_datasets.omni_dataset import (
    OmniDataset,
    format_omni_mpo_dataset,
)
from nemo_rl.data.datasets.response_datasets.video_dataset import (
    VideoDataset,
    format_video_mpo_dataset,
)
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.data.llm_message_utils import (
    get_formatted_message_log,
    strip_image_tokens_from_text,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.vlm_environment import VLMEnvironment
from nemo_rl.models import nemotron_h_nano_vl
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm.utils import load_video_frames as _load_video_frames_np
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VLM MPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    # Parse known args for the script
    args, overrides = parser.parse_known_args()
    return args, overrides


# ===============================================================================
#                             VLM MPO Data Processor
# ===============================================================================


def resolve_to_image(image_path_or_image: str | Image.Image) -> Image.Image:
    """Resolve the image path to a PIL.Image object.

    image_path can be either:
    - path to local file
    - url to image
    - base64 encoded image
    """
    if isinstance(image_path_or_image, Image.Image):
        return image_path_or_image

    if image_path_or_image.startswith(("http://", "https://")):
        # Handle URL
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif image_path_or_image.startswith("data:"):
        # Handle base64 encoded image
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/vAAEEAQMCBAIGBgYFBwkICgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/v
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # Handle local file path
        return Image.open(image_path_or_image).convert("RGB")


def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> list[Image.Image]:
    """Load a video file and extract uniformly sampled frames as PIL images.

    Thin wrapper around :func:`nemo_rl.models.generation.vllm.utils.load_video_frames`
    that converts the returned numpy array to a list of PIL Images.
    """
    frames_nd = _load_video_frames_np(
        video_path, num_frames=num_frames,
        temporal_patch_size=temporal_patch_size,
    )
    return [Image.fromarray(f) for f in frames_nd]


def load_audio_from_video(video_path: str, target_sr: int = 16000) -> np.ndarray:
    """Extract audio waveform from a video file.

    Args:
        video_path: Path to video file (mp4, etc.)
        target_sr: Target sampling rate (default 16kHz for Parakeet).

    Returns:
        Audio waveform as a 1-D numpy float32 array.
    """
    try:
        import librosa
        waveform, _ = librosa.load(video_path, sr=target_sr, mono=True)
        return waveform
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio from {video_path}: {e}")


def process_multimodal_message(
    message: list[dict],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
) -> tuple[dict, list[Image.Image], list[str], list[str]]:
    """Process a multimodal message for MPO training.

    Args:
        message: Message dictionary containing multimodal content
        task_data_spec: Task specification
        processor: AutoProcessor for tokenization

    Returns:
        Tuple of (processed_message, images, image_paths, video_paths)
    """
    images = []
    image_paths: list[str] = []
    video_paths: list[str] = []
    _T = task_data_spec.video_temporal_patch_size
    content = message["content"]
    if isinstance(content, list):
        processed_content = []
        for item in content:
            if item["type"] == "image":
                images.append(resolve_to_image(item["image"]))
                processed_content.append({"type": "image", "image": resolve_to_image(item["image"])})
                if isinstance(item["image"], str):
                    image_paths.append(item["image"])
            elif item["type"] == "video":
                video_path = item["video"]
                video_paths.append(video_path)
                frames = load_video_frames(
                    video_path,
                    num_frames=task_data_spec.num_frames,
                    temporal_patch_size=_T,
                )
                for frame in frames:
                    images.append(frame)
                    processed_content.append({
                        "type": "image",
                        "image": frame,
                        "_is_video_frame": True,
                    })
            elif item["type"] == "audio":
                processed_content.append(item)
            elif item["type"] == "text":
                text = strip_image_tokens_from_text(item["text"])
                processed_content.append({
                    "type": "text",
                    "text": task_data_spec.prompt.format(text)
                    if task_data_spec.prompt
                    else text,
                })
            else:
                processed_content.append(item)
    else:
        text = strip_image_tokens_from_text(content)
        processed_content = (
            task_data_spec.prompt.format(text) if task_data_spec.prompt else text
        )

    message["content"] = processed_content
    return message, images, image_paths, video_paths


def vlm_mpo_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for VLM MPO training.

    Expected format:
        >>> # context can also contain multiple turns
        >>> datum = {
        ...     "context": [{"role": "user", "content": "I have a question."}, {"role": "assistant", "content": "Sure!"}, {"role": "user", "content": "What is 2+2?"}],
        ...     "completions": [
        ...         {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
        ...         {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
        ...     ]
        ... }
    """
    # Format the data based on task type
    include_audio = task_data_spec.use_audio
    if task_data_spec.task_name == "mmpr":
        datum_dict = format_mmpr_dataset(datum_dict)
    elif task_data_spec.task_name == "video_dataset":
        datum_dict = format_video_mpo_dataset(datum_dict, include_audio=include_audio)
    elif task_data_spec.task_name == "omni_dataset":
        datum_dict = format_omni_mpo_dataset(datum_dict, include_audio=include_audio)
    else:
        raise ValueError(f"No data processor for task {task_data_spec.task_name}")

    assert len(datum_dict["completions"]) == 2, (
        "MPO training supports only two completions"
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
            "Ties are not supported yet. You can use the following command to filter out ties: `cat <LocalPathToPreferenceDataset> | jq 'select(.completions[0].rank != .completions[1].rank)'`."
        )
    context_messages = datum_dict["context"]

    _T = task_data_spec.video_temporal_patch_size
    _DEBUG = os.environ.get("NRL_DEBUG", "0") == "1"

    # Process multimodal content in context (shared prompt)
    processed_context = []
    all_images = []
    all_image_paths: list[str] = []
    all_video_paths: list[str] = []
    for msg in context_messages:
        processed_msg, images, img_paths, vid_paths = process_multimodal_message(
            msg, task_data_spec, processor
        )
        processed_context.append(processed_msg)
        all_images.extend(images)
        all_image_paths.extend(img_paths)
        all_video_paths.extend(vid_paths)

    # Process chosen completion (text-only, no multimodal)
    processed_chosen = []
    for msg in chosen_completion["completion"]:
        processed_msg, _, _, _ = process_multimodal_message(
            msg, task_data_spec, processor
        )
        processed_chosen.append(processed_msg)

    # Process rejected completion (text-only, no multimodal)
    processed_rejected = []
    for msg in rejected_completion["completion"]:
        processed_msg, _, _, _ = process_multimodal_message(
            msg, task_data_spec, processor
        )
        processed_rejected.append(processed_msg)

    messages_chosen = processed_context + processed_chosen
    messages_rejected = processed_context + processed_rejected

    if _DEBUG and (idx < 3 or idx % 500 == 0):
        _n_video_frames = len(all_images) - len(all_image_paths)
        print(
            f"[MPO_PREPROC] idx={idx} num_videos={len(all_video_paths)} "
            f"num_images={len(all_image_paths)} "
            f"num_video_frames={_n_video_frames} "
            f"T={_T} config_num_frames={task_data_spec.num_frames} "
            f"use_audio={task_data_spec.use_audio}",
            flush=True,
        )

    message_log_chosen = get_formatted_message_log(
        messages_chosen, processor, task_data_spec,
        max_seq_length=max_seq_length,
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, processor, task_data_spec,
        max_seq_length=max_seq_length,
    )

    # -- Conv3d: inject num_frames and reduce secondary frame token groups --
    # Both chosen and rejected share the same user message (index 1 after system),
    # so we apply the same conv3d fixup to both branches identically.
    def _apply_conv3d_fixup(message_log: list, label: str) -> list:
        """Inject num_frames PackedTensor and remove secondary <img> groups."""
        if _T <= 1 or not all_video_paths:
            return message_log

        user_msg = None
        for m in message_log:
            if m.get("role") == "user":
                user_msg = m
                break
        if user_msg is None:
            return message_log

        # Build per-media num_frames: [1] per regular image, [N] per video
        if "imgs_sizes" in user_msg and isinstance(user_msg["imgs_sizes"], PackedTensor):
            _is = user_msg["imgs_sizes"].as_tensor()
            _n_imgs = _is.shape[0] if _is is not None else 0
        elif "pixel_values_flat" in user_msg and isinstance(user_msg["pixel_values_flat"], PackedTensor):
            _pv = user_msg["pixel_values_flat"].as_tensor()
            _n_imgs = _pv.shape[0] if _pv is not None else 0
        else:
            _n_imgs = 0

        if _n_imgs > 0:
            _n_regular = len(all_image_paths)
            _n_video_frames_total = _n_imgs - _n_regular
            _nf_list: list[int] = [1] * _n_regular
            if _n_video_frames_total > 0 and len(all_video_paths) > 0:
                _frames_per_vid = _n_video_frames_total // len(all_video_paths)
                for _ in range(len(all_video_paths)):
                    _nf_list.append(_frames_per_vid)
            elif _n_video_frames_total > 0:
                _nf_list.append(_n_video_frames_total)
            user_msg["num_frames"] = PackedTensor(
                torch.tensor(_nf_list, dtype=torch.int32), dim_to_pack=0
            )

        # Remove secondary <img>...<image>...</img> token groups (keep every T-th)
        _img_start_id = processor.tokenizer.convert_tokens_to_ids("<img>")
        _img_end_id = processor.tokenizer.convert_tokens_to_ids("</img>")
        _user_ids = user_msg["token_ids"]
        _starts = (_user_ids == _img_start_id).nonzero(as_tuple=True)[0].tolist()

        _group_idx = 0
        _keep = torch.ones(len(_user_ids), dtype=torch.bool)
        _n_secondary = 0

        for _s in _starts:
            _ends = (_user_ids[_s:] == _img_end_id).nonzero(as_tuple=True)[0]
            if len(_ends) == 0:
                break
            _e = _s + _ends[0].item()
            if _group_idx % _T != 0:
                _keep[_s : _e + 1] = False
                _n_secondary += 1
            _group_idx += 1

        if _n_secondary > 0:
            user_msg["token_ids"] = _user_ids[_keep]

        if _DEBUG and (idx < 3 or idx % 500 == 0):
            _nf_summary = _nf_list if _n_imgs > 0 else "N/A"
            _new_len = len(user_msg["token_ids"])
            print(
                f"[MPO_CONV3D] idx={idx} {label}: T={_T} "
                f"total_img_groups={_group_idx} secondary_removed={_n_secondary} "
                f"num_frames_list={_nf_summary} "
                f"token_len_after={_new_len}",
                flush=True,
            )

        return message_log

    message_log_chosen = _apply_conv3d_fixup(message_log_chosen, "chosen")
    message_log_rejected = _apply_conv3d_fixup(message_log_rejected, "rejected")

    # Pairwise sanity check: prompt-side tokens must match across branches
    if _DEBUG and (idx < 3 or idx % 500 == 0):
        _ch_user = next((m for m in message_log_chosen if m.get("role") == "user"), None)
        _rj_user = next((m for m in message_log_rejected if m.get("role") == "user"), None)
        if _ch_user is not None and _rj_user is not None:
            _match = torch.equal(_ch_user["token_ids"], _rj_user["token_ids"])
            print(
                f"[MPO_PAIR_CHECK] idx={idx} "
                f"chosen_user_len={len(_ch_user['token_ids'])} "
                f"rejected_user_len={len(_rj_user['token_ids'])} "
                f"user_tokens_identical={_match}",
                flush=True,
            )

    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    # Discard overlong samples by zeroing the loss multiplier. Token stubs and
    # empty PackedTensors are kept as structural placeholders for batching.
    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        print(f"Discarding overlong sample: chosen={length_chosen}, rejected={length_rejected}, max={max_seq_length}")
        tokenizer = processor.tokenizer
        image_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<img>", "<image>", "</img>")]
        for message in message_log_chosen:
            token_ids = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
            # Filter out all image tokens (<img>, <image>, </img>) since we're discarding images
            for img_token_id in image_token_ids:
                token_ids = token_ids[token_ids != img_token_id]
            message["token_ids"] = token_ids
            for key, value in message.items():
                if isinstance(value, PackedTensor):
                    message[key] = PackedTensor.empty_like(value)
        for message in message_log_rejected:
            token_ids = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
            # Filter out all image tokens (<img>, <image>, </img>) since we're discarding images
            for img_token_id in image_token_ids:
                token_ids = token_ids[token_ids != img_token_id]
            message["token_ids"] = token_ids
            for key, value in message.items():
                if isinstance(value, PackedTensor):
                    message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
        length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
        length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    output = {
        "message_log_chosen": message_log_chosen,
        "length_chosen": length_chosen,
        "message_log_rejected": message_log_rejected,
        "length_rejected": length_rejected,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name,
    }
    return output


def setup_data(
    processor: AutoProcessor,
    data_config: DataConfig,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset]
]:
    """This function will create a TaskSpec, DatumSpec, and connect the two for VLM MPO.

    task_spec contains the task name as well as prompt and system prompt modifiers that can be used by data processor
    """
    print("\n▶ Setting up VLM MPO data...")

    # Load appropriate VLM dataset
    if data_config["dataset_name"] == "mmpr":
        data: Any = MMPRDataset(
            data_path=data_config["data_path"],
            split=data_config["split"]
        )
    elif data_config["dataset_name"] == "video_dataset":
        data = VideoDataset(
            train_data_path=data_config["train_data_path"],
            prompt_file=data_config.get("prompt_file"),
            val_size=data_config.get("val_size", 0),
        )
    elif data_config["dataset_name"] == "omni_dataset":
        data = OmniDataset(
            train_data_path=data_config["train_data_path"],
            prompt_file=data_config.get("prompt_file"),
            val_size=data_config.get("val_size", 0),
        )
    else:
        raise ValueError(f"No processor for VLM dataset {data_config['dataset_name']}.")

    mpo_task_spec = data.task_name

    _vision_cfg = getattr(getattr(processor, "config", None), "vision_config", None)
    _auto_T = getattr(_vision_cfg, "video_temporal_patch_size", 1)
    _video_temporal_patch_size = data_config.get("video_temporal_patch_size", _auto_T)
    _auto_aspect = getattr(_vision_cfg, "video_maintain_aspect_ratio", True)
    _video_maintain_aspect_ratio = data_config.get("video_maintain_aspect_ratio", _auto_aspect)

    vlm_task_spec = TaskDataSpec(
        task_name=mpo_task_spec,
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
        num_frames=data_config.get("num_frames", 8),
        use_tiling=data_config.get("use_tiling", False),
        use_dynamic_resolution=data_config.get("use_dynamic_resolution", None),
        max_num_tiles=data_config.get("max_num_tiles", None),
        max_num_patches=data_config.get("max_num_patches", None),
        video_target_num_patches=data_config.get("video_target_num_patches", None),
        use_audio=data_config.get("use_audio", False),
        max_audio_duration=data_config.get("max_audio_duration", None),
        sound_clip_duration=float(data_config.get("sound_clip_duration", 30.0)),
        sound_clip_min_duration=float(data_config.get("sound_clip_min_duration", 0.1)),
        video_temporal_patch_size=_video_temporal_patch_size,
        video_maintain_aspect_ratio=_video_maintain_aspect_ratio,
        min_generation_tokens=data_config.get("min_generation_tokens", 2000),
    )
    print(
        f"[MPO VLM Data] task={mpo_task_spec} num_frames={vlm_task_spec.num_frames} "
        f"use_tiling={vlm_task_spec.use_tiling} "
        f"max_num_tiles={vlm_task_spec.max_num_tiles} "
        f"max_num_patches={vlm_task_spec.max_num_patches} "
        f"video_target_num_patches={vlm_task_spec.video_target_num_patches} "
        f"use_audio={vlm_task_spec.use_audio} "
        f"max_audio_duration={vlm_task_spec.max_audio_duration} "
        f"sound_clip_duration={vlm_task_spec.sound_clip_duration} "
        f"sound_clip_min_duration={vlm_task_spec.sound_clip_min_duration} "
        f"video_temporal_patch_size={vlm_task_spec.video_temporal_patch_size} "
        f"video_maintain_aspect_ratio={vlm_task_spec.video_maintain_aspect_ratio}"
    )

    # Create datasets
    train_dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        processor,
        vlm_task_spec,
        vlm_mpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds.get("validation"):
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            processor,
            vlm_task_spec,
            vlm_mpo_preprocessor,
            max_seq_length=data_config["max_input_seq_length"],
        )
    if not isinstance(val_dataset, dict):
        val_dataset = {} if val_dataset is None else {"default": val_dataset}
    # Set up task-to-environment mapping
    # task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: vlm_env)
    # task_to_env[task_name] = vlm_env

    return train_dataset, val_dataset


def main() -> None:
    """Main entry point for VLM MPO training."""
    nemotron_h_nano_vl.register()

    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "vlm_mpo_mmpr.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # Initialize processor for multimodal processing
    processor = get_tokenizer(config["policy"]["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer

    # Configure generation if specified
    if config["policy"].get("generation"):
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], processor.tokenizer
        )

    # When megatron_cfg.dynamic_resolution is explicitly false, tell vLLM
    # to use static tiling by injecting max_num_tiles (the Bridge default
    # is 12).  This keeps vLLM and Megatron in sync.
    _megatron_cfg = config["policy"].get("megatron_cfg", {}) or {}
    if "dynamic_resolution" in _megatron_cfg and not _megatron_cfg["dynamic_resolution"]:
        gen_cfg = config["policy"].setdefault("generation", {})
        if gen_cfg:
            vllm_kwargs = gen_cfg.setdefault("vllm_kwargs", {})
            mm_kwargs = vllm_kwargs.setdefault("mm_processor_kwargs", {})
            mm_kwargs.setdefault("max_num_tiles", 12)

    # Setup data with multimodal processing
    (
        train_dataset,
        val_dataset,
    ) = setup_data(processor, config["data"])

    # Setup MPO training
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        mpo_save_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # Run MPO training
    mpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        mpo_save_state,
    )


if __name__ == "__main__":
    main()
