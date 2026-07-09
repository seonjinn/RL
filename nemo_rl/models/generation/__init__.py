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
import json
import warnings
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmConfig

TokenizerType = PreTrainedTokenizerBase

_ONLINE_REFIT_SPEC_METHODS = {"eagle", "eagle3"}
_EMBEDDED_MTP_SPEC_METHODS = {"deepseek_mtp", "mtp"}
_MODEL_FREE_SPEC_METHODS = {"suffix", "ngram"}
_STATIC_NEURAL_EXTERNAL_DRAFT_METHODS = {
    "draft_model",
    "eagle",
    "eagle3",
    "dflash",
    "medusa",
    "mlp_speculator",
}


def _get_speculative_config(config: VllmConfig) -> dict[str, Any] | None:
    speculative_config = config.get("vllm_kwargs", {}).get("speculative_config")
    if not speculative_config:
        return None
    return cast(dict[str, Any], speculative_config)


def _uses_static_neural_external_drafter(
    speculative_config: dict[str, Any],
) -> bool:
    if speculative_config.get("model") is None:
        return False
    method = speculative_config.get("method")
    return method not in _EMBEDDED_MTP_SPEC_METHODS.union(_MODEL_FREE_SPEC_METHODS)


def validate_vllm_speculative_config(
    config: VllmConfig,
    *,
    has_refit_draft_weights: bool,
) -> None:
    speculative_config = _get_speculative_config(config)
    if speculative_config is None:
        return

    method = speculative_config.get("method")
    rejection_sample_method = speculative_config.get(
        "rejection_sample_method", "standard"
    )
    if rejection_sample_method != "standard":
        raise ValueError(
            "NeMo-RL requires speculative_config.rejection_sample_method="
            f"'standard' for target-distribution correctness. Got "
            f"{rejection_sample_method!r}."
        )

    draft_sample_method = speculative_config.get("draft_sample_method", "greedy")
    if draft_sample_method not in ("greedy", "probabilistic"):
        raise ValueError(
            "speculative_config.draft_sample_method must be 'greedy' or "
            f"'probabilistic'. Got {draft_sample_method!r}."
        )

    if method == "pard2":
        raise ValueError("speculative_config.method='pard2' is not supported")

    if has_refit_draft_weights and method not in _ONLINE_REFIT_SPEC_METHODS:
        raise ValueError(
            "Online draft refit only supports speculative methods 'eagle' and "
            f"'eagle3'. Got method={method!r}."
        )

    if method in _STATIC_NEURAL_EXTERNAL_DRAFT_METHODS and not speculative_config.get(
        "model"
    ):
        raise ValueError(
            f"speculative_config.model is required for speculative method {method!r}."
        )

    if method == "draft_model":
        target_tp = config["vllm_cfg"]["tensor_parallel_size"]
        draft_tp = speculative_config.get("draft_tensor_parallel_size")
        if draft_tp is None:
            draft_tp = target_tp
            speculative_config["draft_tensor_parallel_size"] = target_tp
        if draft_tp != target_tp:
            raise ValueError(
                "draft_model requires draft_tensor_parallel_size to match the "
                "target tensor_parallel_size"
            )


def get_vllm_specdec_runtime_contract(
    config: VllmConfig,
) -> dict[str, Any] | None:
    """Return the effective NeMo-side SpecDec and sampling startup contract."""
    speculative_config = _get_speculative_config(config)
    if speculative_config is None:
        return None

    method = speculative_config.get("method")
    target_tp = config["vllm_cfg"]["tensor_parallel_size"]
    draft_tp: int | str | None = speculative_config.get("draft_tensor_parallel_size")
    if draft_tp is None:
        draft_tp = "vllm_auto" if method in (None, "mlp_speculator") else target_tp

    draft_load_config = speculative_config.get("draft_load_config") or {}
    return {
        "method": method or "vllm_auto",
        "model": speculative_config.get("model"),
        "num_speculative_tokens": speculative_config.get("num_speculative_tokens"),
        "target_tp": target_tp,
        "draft_tp": draft_tp,
        "target_load_format": config["vllm_cfg"].get("load_format", "auto"),
        "draft_load_format": draft_load_config.get(
            "load_format", config["vllm_cfg"].get("load_format", "auto")
        ),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "rejection_sample_method": speculative_config.get(
            "rejection_sample_method", "standard"
        ),
        "draft_sample_method": speculative_config.get("draft_sample_method", "greedy"),
        "cuda_graph_enabled": not config["vllm_cfg"].get("enforce_eager", False),
    }


def configure_generation_config(
    config: GenerationConfig,
    tokenizer: TokenizerType,
    is_eval: bool = False,
    has_refit_draft_weights: bool = False,
    trains_mtp: bool = False,
) -> GenerationConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    if "_pad_token_id" in config:
        warnings.warn(
            "'_pad_token_id' found in generation config and will be overridden with tokenizer.pad_token_id. "
            "Note: '_pad_token_id' is intended for internal use and has no effect when set in user-provided configs.",
            UserWarning,
        )
    config["_pad_token_id"] = tokenizer.pad_token_id
    if config["stop_token_ids"] is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # vllm setting
    if config["backend"] == "vllm":
        config = cast(VllmConfig, config)
        # set load_format
        config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"
        validate_vllm_speculative_config(
            config,
            has_refit_draft_weights=has_refit_draft_weights,
        )
        speculative_config = _get_speculative_config(config)
        if (
            speculative_config is not None
            and not is_eval
            and not has_refit_draft_weights
            and _uses_static_neural_external_drafter(speculative_config)
        ):
            if "draft_load_config" not in speculative_config:
                warnings.warn(
                    "Speculative decoding is enabled without draft refit sync. "
                    "Keeping vllm_cfg['load_format'] at 'dummy' and setting "
                    "speculative_config['draft_load_config'] to load the static "
                    "drafter from disk."
                )
                speculative_config["draft_load_config"] = {"load_format": "auto"}
            else:
                warnings.warn(
                    "Speculative decoding is enabled without draft refit sync. "
                    "Keeping vllm_cfg['load_format'] at 'dummy' and preserving "
                    "the user-provided speculative_config['draft_load_config'] "
                    "for the static drafter."
                )

        # MTP draft weights arrive via refit if the trainer trains the MTP layer.
        # If the trainer does not train the MTP layer, the weights need to be
        # loaded from the checkpoint.
        config["_mtp_weights_from_refit"] = trains_mtp

        # Respect the skip_tokenizer_init setting from the config. VLMs for example, require this to be False.
        if "skip_tokenizer_init" not in config["vllm_cfg"]:
            # set skip_tokenizer_init
            if (
                is_eval
                or config["stop_strings"] is not None
                or config["vllm_cfg"].get("expose_http_server", None)
            ):
                config["vllm_cfg"]["skip_tokenizer_init"] = False
            else:
                config["vllm_cfg"]["skip_tokenizer_init"] = True

        runtime_contract = get_vllm_specdec_runtime_contract(config)
        if runtime_contract is not None:
            print(
                "[specdec] runtime_contract="
                + json.dumps(runtime_contract, sort_keys=True),
                flush=True,
            )

    return config
