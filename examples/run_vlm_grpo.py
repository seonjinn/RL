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
import os
import pprint
import time

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models import nemotron_h_nano_vl
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    # Parse known args for the script
    args, overrides = parser.parse_known_args()
    return args, overrides


def _first_train_data_config(data_cfg: dict) -> dict:
    train_cfg = data_cfg.get("train", {})
    if isinstance(train_cfg, list):
        return train_cfg[0] if train_cfg else {}
    return train_cfg or {}


def _setdefault_nested(root: dict, key: str) -> dict:
    value = root.setdefault(key, {})
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


_MEGATRON_VLM_FREEZE_ALIASES = {
    "freeze_vision_encoder": "freeze_vision_model",
    "freeze_vision_projector": "freeze_vision_projection",
    "freeze_audio_encoder": "freeze_sound_encoder",
    "freeze_audio_projector": "freeze_sound_projection",
}


def _translate_megatron_vlm_aliases(megatron_cfg: dict) -> None:
    for old_key, new_key in _MEGATRON_VLM_FREEZE_ALIASES.items():
        if old_key in megatron_cfg and new_key not in megatron_cfg:
            megatron_cfg[new_key] = megatron_cfg[old_key]


def _propagate_omni_runtime_config(config: dict) -> None:
    """Bridge current-schema Omni data keys into policy and generation config."""
    data_cfg = config.get("data", {}) or {}
    data_default = data_cfg.get("default", {}) or {}
    data_train = _first_train_data_config(data_cfg)
    data_settings = {**data_default, **data_train}

    grpo_cfg = config.get("grpo", {}) or {}
    if grpo_cfg.get("dynamic_format_reward") is not None:
        for env_cfg in (config.get("env", {}) or {}).values():
            if not isinstance(env_cfg, dict):
                continue
            for reward_cfg in env_cfg.get("reward_functions", []) or []:
                if reward_cfg.get("name") == "mmpr_filtered":
                    kwargs = reward_cfg.setdefault("kwargs", {})
                    kwargs.setdefault(
                        "dynamic_format_reward",
                        bool(grpo_cfg["dynamic_format_reward"]),
                    )

    policy_cfg = config.get("policy", {}) or {}
    _translate_megatron_vlm_aliases(_setdefault_nested(policy_cfg, "megatron_cfg"))
    generation_cfg = policy_cfg.get("generation", {}) or {}
    if generation_cfg.get("backend") == "vllm":
        vllm_kwargs = _setdefault_nested(generation_cfg, "vllm_kwargs")
        # vLLM passes top-level mm_processor_kwargs to the HF processor
        # constructor at engine startup. NanoNemotron's processor only accepts
        # constructor-level max_num_tiles; video sizing lives in hf_overrides
        # and row-specific image budgets are attached to each prompt.
        if data_settings.get("max_num_tiles") is not None:
            mm_processor_kwargs = _setdefault_nested(
                vllm_kwargs, "mm_processor_kwargs"
            )
            mm_processor_kwargs.setdefault(
                "max_num_tiles", data_settings["max_num_tiles"]
            )

        limit_mm_per_prompt = _setdefault_nested(vllm_kwargs, "limit_mm_per_prompt")
        if data_settings.get("num_frames") is not None:
            num_frames = int(data_settings["num_frames"])
            limit_mm_per_prompt.setdefault(
                "video", {"count": 1, "num_frames": num_frames}
            )
        if data_settings.get("max_images_per_prompt") is not None:
            limit_mm_per_prompt.setdefault(
                "image", int(data_settings["max_images_per_prompt"])
            )
        if data_settings.get("use_audio"):
            limit_mm_per_prompt.setdefault("audio", 1)

    use_dynamic_resolution = data_settings.get("use_dynamic_resolution")
    if use_dynamic_resolution is not None:
        megatron_cfg = _setdefault_nested(policy_cfg, "megatron_cfg")
        megatron_cfg.setdefault("dynamic_resolution", bool(use_dynamic_resolution))

    hf_overrides = _setdefault_nested(policy_cfg, "hf_config_overrides")
    if use_dynamic_resolution is not None:
        hf_overrides.setdefault("dynamic_resolution", bool(use_dynamic_resolution))
    vision_config = _setdefault_nested(hf_overrides, "vision_config")
    sound_config = _setdefault_nested(hf_overrides, "sound_config")

    for key in ("video_temporal_patch_size", "video_target_num_patches"):
        value = data_settings.get(key)
        if value is not None:
            vision_config.setdefault(key, value)

    for key in (
        "max_audio_duration",
        "sound_clip_duration",
        "sound_clip_min_duration",
    ):
        value = data_settings.get(key)
        if value is not None:
            sound_config.setdefault(key, value)


def main() -> None:
    """Main entry point."""
    main_start = time.perf_counter()
    log_container_init_timing()
    nemotron_h_nano_vl.register()

    rl_init_timer = Timer(context={"worker": "driver"})

    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "vlm_grpo_3B.yaml"
        )

    with rl_init_timer.time("config"):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

        if overrides:
            print(f"Overrides: {overrides}")
            config = parse_hydra_overrides(config, overrides)

        config: MasterConfig = OmegaConf.to_container(config, resolve=True)
        _propagate_omni_runtime_config(config)
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

    with rl_init_timer.time("ray_connect"):
        init_ray()

    # init processor
    with rl_init_timer.time("tokenizer"):
        processor = get_tokenizer(config["policy"]["tokenizer"], get_processor=True)
        tokenizer = processor.tokenizer

        assert config["policy"]["generation"] is not None, (
            "A generation config is required for GRPO"
        )
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], processor.tokenizer
        )
        if "vllm_cfg" in config["policy"]["generation"]:
            assert (
                config["policy"]["generation"]["vllm_cfg"]["skip_tokenizer_init"]
                == False
            ), (
                "VLMs require tokenizer to be initialized before generation, so skip_tokenizer_init must be set to False."
            )

    # setup data
    # this function is local to this script, and can be extended to other VLM datasets
    with rl_init_timer.time("data"):
        dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
            processor, config["data"], config["env"], is_vlm=True
        )

    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            _nemo_gym_actor,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        ) = setup(config, tokenizer, dataset, val_dataset, processor=processor)

    rl_init_timer.record("total", time.perf_counter() - main_start)

    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        processor=processor,
    )


if __name__ == "__main__":
    main()
