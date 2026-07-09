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
from collections.abc import Mapping
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm.config import (
    MTP_SPECULATIVE_METHODS,
    VllmConfig,
)

TokenizerType = PreTrainedTokenizerBase

_ONLINE_REFIT_SPEC_METHODS = {"eagle", "eagle3"}
_MODEL_FREE_SPEC_METHODS = {
    "suffix",
    "ngram",
    "ngram_gpu",
    "custom_class",
    "extract_hidden_states",
}
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
    return method not in MTP_SPECULATIVE_METHODS.union(_MODEL_FREE_SPEC_METHODS)


def validate_vllm_speculative_config(
    config: VllmConfig,
    *,
    has_refit_draft_weights: bool,
    is_eval: bool,
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

    draft_load_config = speculative_config.get("draft_load_config") or {}
    if (
        not has_refit_draft_weights
        and _uses_static_neural_external_drafter(speculative_config)
        and str(draft_load_config.get("load_format", "auto")).lower() == "dummy"
    ):
        raise ValueError(
            "Static external speculative drafter cannot use "
            "draft_load_config.load_format='dummy' without online draft refit; "
            "load real draft checkpoint weights instead."
        )

    if (
        method in MTP_SPECULATIVE_METHODS
        and speculative_config.get("model")
        and not is_eval
    ):
        raise ValueError(
            "Explicit MTP methods with an external speculative_config.model are "
            "not supported by NeMo-RL's dummy-target refit path. Omit method to "
            "use vLLM model auto-detection or use an embedded target MTP module."
        )

    if has_refit_draft_weights and method not in _ONLINE_REFIT_SPEC_METHODS:
        raise ValueError(
            "Online draft refit only supports speculative methods 'eagle' and "
            f"'eagle3'. Got method={method!r}."
        )
    if (
        has_refit_draft_weights
        and method in _ONLINE_REFIT_SPEC_METHODS
        and config["vllm_cfg"].get("pipeline_parallel_size", 1) != 1
    ):
        raise ValueError(
            "Online Eagle refit requires vLLM pipeline parallelism PP=1 because "
            "vLLM 0.24 does not share target embeddings with the draft across PP ranks."
        )
    if (
        method in MTP_SPECULATIVE_METHODS
        and config["vllm_cfg"].get("pipeline_parallel_size", 1) != 1
    ):
        raise ValueError(
            "MTP speculative decoding requires vLLM pipeline parallelism PP=1 "
            "until sampler-count, draft-token broadcast, and rollback fixes land."
        )
    if (
        method is None
        and speculative_config.get("model") is not None
        and config["vllm_cfg"].get("pipeline_parallel_size", 1) != 1
    ):
        raise ValueError(
            "Auto-detected speculative models require PP=1 in NeMo-RL. Set an "
            "explicit speculative_config.method so method-specific PP safety can "
            "be validated before engine construction."
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
    worker_env = config["vllm_cfg"].get("env_vars", {})
    draft_model_cudagraph_patch_requested = (
        str(
            worker_env.get(
                "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH",
                "false",
            )
        ).lower()
        == "true"
    )
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
        "draft_model_cudagraph_patch_requested": (
            draft_model_cudagraph_patch_requested
        ),
    }


def resolve_vllm_refit_draft_flags(
    policy_config: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Derive online-draft and MTP refit ownership from a policy config."""
    generation_config = policy_config.get("generation") or {}
    if generation_config.get("backend") != "vllm":
        return False, False

    draft_config = policy_config.get("draft") or {}
    has_refit_draft_weights = bool(draft_config.get("enabled", False))
    if has_refit_draft_weights:
        speculative_config = generation_config.get("vllm_kwargs", {}).get(
            "speculative_config"
        )
        if not speculative_config:
            raise ValueError(
                "policy.draft.enabled=true requires "
                "policy.generation.vllm_kwargs.speculative_config"
            )
        method = speculative_config.get("method")
        if method not in _ONLINE_REFIT_SPEC_METHODS:
            raise ValueError(
                "policy.draft.enabled=true only supports speculative methods "
                f"'eagle' and 'eagle3'. Got method={method!r}."
            )

    megatron_config = policy_config.get("megatron_cfg") or {}
    if has_refit_draft_weights and (
        megatron_config.get("pipeline_model_parallel_size", 1) != 1
    ):
        raise ValueError(
            "NeMo-RL online Eagle refit requires policy PP=1 because draft "
            "parameters and hidden-state ownership are not distributed safely "
            "across Megatron pipeline stages."
        )
    trains_mtp = bool(megatron_config.get("mtp_num_layers"))
    return has_refit_draft_weights, trains_mtp


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
            is_eval=is_eval,
        )
        speculative_config = _get_speculative_config(config)
        if speculative_config is not None and has_refit_draft_weights:
            draft_load_config = speculative_config.setdefault(
                "draft_load_config", {"load_format": "dummy"}
            )
            if draft_load_config.get("load_format") != "dummy":
                raise ValueError(
                    "Online Eagle refit requires draft_load_config.load_format="
                    "'dummy' so target and draft LM-head ownership is established "
                    "before the first refit."
                )
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
