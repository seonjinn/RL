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

import os
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any


def _get_vllm_file(relative_path: str) -> str:
    """Return absolute path to a vLLM file or raise if it cannot be found.

    The relative_path should be a POSIX-style path under the vllm
    package root, e.g. "v1/executor/ray_executor.py" or
    "attention/layer.py".
    """
    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "vLLM package not found while attempting to patch "
            f"'{relative_path}'. Ensure vLLM is installed and "
            "available in this environment."
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Failed to locate expected vLLM file to patch. "
            f"Looked for '{relative_path}' at '{file_path}'. "
            "This likely indicates an unexpected vLLM installation "
            "layout or version mismatch."
        )

    return file_path


def _get_optional_vllm_file(relative_path: str, logger, component: str) -> str | None:
    """Return an optional model-specific vLLM file when that feature is installed."""
    try:
        return _get_vllm_file(relative_path)
    except RuntimeError:
        logger.info(
            "%s is not installed in this vLLM build; skipping its patch.",
            component,
        )
        return None


@contextmanager
def _locked_file_patch(file_path: str):
    """Yield (content, writer) under an exclusive file lock."""
    import fcntl

    lock_path = file_path + ".patch_lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with open(file_path, "r") as f:
            content = f.read()

        def write_back(new_content: str):
            with open(file_path, "w") as f:
                f.write(new_content)

        yield content, write_back
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _patch_vllm_init_workers_ray(
    py_executable: str, extra_env_vars: list[str] | None
) -> None:
    """Patch the vLLM ray_distributed_executor.py file.

    1. Pass custom runtime_env in _init_workers_ray call.
        - This allows passing custom py_executable to worker initialization.
    2. Register required variables through vLLM's additive Ray env-copy API.
    """
    file_to_patch = _get_vllm_file("v1/executor/ray_executor.py")

    old_lines = ["self._init_workers_ray(placement_group)"]
    additional_env_vars = [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NCCL_CUMEM_ENABLE",
        "NCCL_NVLS_ENABLE",
        "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        *(extra_env_vars or []),
    ]
    existing_extra_env_vars = {
        name.strip()
        for name in os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "").split(",")
        if name.strip()
    }
    os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(
        sorted(existing_extra_env_vars | set(additional_env_vars))
    )
    new_lines = [
        (
            "self._init_workers_ray(placement_group, "
            f'runtime_env={{"py_executable": "{py_executable}"}})'
        ),
    ]

    with _locked_file_patch(file_to_patch) as (content, write_back):
        need_replace = False
        for old_line, new_line in zip(old_lines, new_lines):
            if new_line in content:
                continue
            if content.count(old_line) != 1:
                raise RuntimeError(
                    "Could not apply the Ray worker environment patch to "
                    f"{file_to_patch}; the vLLM source layout changed."
                )
            content = content.replace(old_line, new_line, 1)
            need_replace = True

        if need_replace:
            write_back(content)


def _patch_vllm_llama_eagle3_own_lm_head(logger) -> None:
    """Patch LlamaEagle3 to keep truncated draft lm_head ownership."""
    try:
        file_to_patch = _get_vllm_file("model_executor/models/llama_eagle3.py")
    except RuntimeError:
        logger.warning("Could not locate llama_eagle3.py for lm_head ownership patch.")
        return

    old_snippet = (
        "        self.lm_head = ParallelLMHead(\n"
        "            self.config.draft_vocab_size,\n"
        "            self.config.hidden_size,\n"
        "            quant_config=get_draft_quant_config(vllm_config),\n"
        '            prefix=maybe_prefix(prefix, "lm_head"),\n'
        "        )\n"
        "        self.logits_processor = LogitsProcessor(\n"
    )

    new_snippet = (
        "        self.lm_head = ParallelLMHead(\n"
        "            self.config.draft_vocab_size,\n"
        "            self.config.hidden_size,\n"
        "            quant_config=get_draft_quant_config(vllm_config),\n"
        '            prefix=maybe_prefix(prefix, "lm_head"),\n'
        "        )\n"
        "        self.has_own_lm_head = (\n"
        "            self.config.draft_vocab_size != self.config.vocab_size\n"
        "        )\n"
        "        self.logits_processor = LogitsProcessor(\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "self.has_own_lm_head = (" in content:
            logger.info("llama_eagle3 lm_head ownership patch already applied.")
            return

        if old_snippet not in content:
            logger.warning(
                "Could not apply llama_eagle3 lm_head ownership patch: "
                "expected code snippet not found in %s. "
                "The vLLM version may have changed.",
                file_to_patch,
            )
            return

        content = content.replace(old_snippet, new_snippet, 1)
        write_back(content)

    logger.info("Successfully patched llama_eagle3 lm_head ownership.")


def _patch_vllm_online_eagle_head_ownership(logger) -> None:
    """Keep an online-refit Eagle head distinct before CUDA graph capture."""
    file_to_patch = _get_vllm_file("v1/spec_decode/llm_base_proposer.py")
    marker = "online_refit_uses_dummy_drafter"
    old_snippet = "        self.model = self._get_model()\n\n"
    new_snippet = (
        "        self.model = self._get_model()\n\n"
        "        draft_load_config = self.speculative_config.draft_load_config\n"
        "        draft_load_format = getattr(draft_load_config, 'load_format', None)\n"
        "        draft_load_format = getattr(draft_load_format, 'value', draft_load_format)\n"
        "        online_refit_uses_dummy_drafter = (\n"
        "            self.speculative_config.method in ('eagle', 'eagle3')\n"
        "            and draft_load_config is not None\n"
        "            and str(draft_load_format).lower() == 'dummy'\n"
        "        )\n"
        "        if online_refit_uses_dummy_drafter:\n"
        "            self.model.has_own_lm_head = True\n"
        "            self.model.has_own_embed_tokens = False\n\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("Online Eagle head ownership patch already applied.")
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the online Eagle head ownership patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched online Eagle head ownership.")


def _patch_vllm_v2_eagle_load_config_and_ownership(logger) -> None:
    """Give Model Runner V2 Eagle/MTP an independent draft load contract."""
    file_to_patch = _get_vllm_file("v1/worker/gpu/spec_decode/eagle/utils.py")
    old_snippet = (
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        '    with set_model_tag("eagle_head"):\n'
        "        eagle_model = get_model(\n"
        "            vllm_config=vllm_config, model_config=draft_model_config\n"
        "        )\n"
    )
    new_snippet = (
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        "    draft_load_config = speculative_config.draft_load_config\n"
        '    with set_model_tag("eagle_head"):\n'
        "        eagle_model = get_model(\n"
        "            vllm_config=vllm_config,\n"
        "            model_config=draft_model_config,\n"
        "            load_config=draft_load_config or vllm_config.load_config,\n"
        "        )\n"
        "\n"
        "    draft_load_format = getattr(draft_load_config, 'load_format', None)\n"
        "    draft_load_format = getattr(draft_load_format, 'value', draft_load_format)\n"
        "    online_refit_uses_dummy_drafter = (\n"
        "        speculative_config.method in ('eagle', 'eagle3')\n"
        "        and draft_load_config is not None\n"
        "        and str(draft_load_format).lower() == 'dummy'\n"
        "    )\n"
        "    if online_refit_uses_dummy_drafter:\n"
        "        eagle_model.has_own_lm_head = True\n"
        "        eagle_model.has_own_embed_tokens = False\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "online_refit_uses_dummy_drafter" in content:
            logger.info("Model Runner V2 Eagle load/ownership patch already applied.")
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the Model Runner V2 Eagle load/ownership patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched Model Runner V2 Eagle load/ownership.")


def _patch_vllm_v2_dflash_load_config(logger) -> None:
    """Give Model Runner V2 DFlash an independent draft load contract."""
    file_to_patch = _get_optional_vllm_file(
        "v1/worker/gpu/spec_decode/dflash/utils.py", logger, "DFlash"
    )
    if file_to_patch is None:
        return
    old_snippet = (
        '    with set_model_tag("dflash_head"):\n'
        "        dflash_model = get_model(\n"
        "            vllm_config=draft_vllm_config, model_config=draft_model_config\n"
        "        )\n"
    )
    new_snippet = (
        '    with set_model_tag("dflash_head"):\n'
        "        dflash_model = get_model(\n"
        "            vllm_config=draft_vllm_config,\n"
        "            model_config=draft_model_config,\n"
        "            load_config=speculative_config.draft_load_config\n"
        "            or vllm_config.load_config,\n"
        "        )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "load_config=speculative_config.draft_load_config" in content:
            logger.info("Model Runner V2 DFlash load-config patch already applied.")
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the Model Runner V2 DFlash load-config patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched Model Runner V2 DFlash draft load config.")


def _patch_vllm_qwen3_draft_loader_results(logger) -> None:
    """Return loaded parameter names from Qwen3 Eagle-3 and DFlash loaders."""
    patches = (
        (
            "model_executor/models/qwen3_eagle3.py",
            "        loader.load_weights(model_weights.items())\n",
            "        self.has_own_lm_head = any(\n"
            '            name.startswith("lm_head.") for name in model_weights\n'
            "        )\n"
            "        return loader.load_weights(model_weights.items())\n",
        ),
        (
            "model_executor/models/qwen3_dflash.py",
            "        loader.load_weights(model_weights.items())\n"
            "        self.model._build_fused_kv_buffers()\n",
            "        loaded_weights = loader.load_weights(model_weights.items())\n"
            "        self.model._build_fused_kv_buffers()\n"
            "        return loaded_weights\n",
        ),
    )

    for relative_path, old_snippet, new_snippet in patches:
        component = (
            "Qwen3 DFlash"
            if relative_path.endswith("qwen3_dflash.py")
            else "Qwen3 Eagle-3"
        )
        file_to_patch = _get_optional_vllm_file(relative_path, logger, component)
        if file_to_patch is None:
            continue
        with _locked_file_patch(file_to_patch) as (content, write_back):
            if new_snippet in content:
                logger.info("Qwen3 draft loader result patch already applied.")
                continue
            legacy_receipt_snippet = (
                "        return loader.load_weights(model_weights.items())\n"
            )
            if (
                relative_path.endswith("qwen3_eagle3.py")
                and content.count(legacy_receipt_snippet) == 1
            ):
                write_back(content.replace(legacy_receipt_snippet, new_snippet, 1))
                logger.info(
                    "Upgraded legacy Qwen3 Eagle-3 loader receipt patch with "
                    "LM-head ownership reporting."
                )
                continue
            if content.count(old_snippet) != 1:
                raise RuntimeError(
                    "Could not apply the Qwen3 draft loader result patch to "
                    f"{file_to_patch}; the vLLM source layout changed."
                )
            write_back(content.replace(old_snippet, new_snippet, 1))
        logger.info("Successfully patched Qwen3 draft loader result reporting.")


def _patch_vllm_llama_draft_loader_result(logger) -> None:
    """Return an auditable receipt from the Llama Eagle-3 loader."""
    file_to_patch = _get_optional_vllm_file(
        "model_executor/models/llama_eagle3.py", logger, "Llama Eagle-3"
    )
    if file_to_patch is None:
        return
    old_snippet = "        loader.load_weights(model_weights.items())\n"
    legacy_snippet = (
        "        self.has_own_lm_head = any(\n"
        '            name.startswith("lm_head.") for name in model_weights\n'
        "        )\n"
        "        return loader.load_weights(model_weights.items())\n"
    )
    legacy_receipt_snippet = (
        "        return loader.load_weights(model_weights.items())\n"
    )
    new_snippet = (
        "        self.has_own_embed_tokens = includes_embed_tokens\n"
        "        self.has_own_lm_head = any(\n"
        '            name.startswith("lm_head.") for name in model_weights\n'
        "        )\n"
        "        loaded_weights = loader.load_weights(model_weights.items())\n"
        "        intentional_default_params = set()\n"
        "        if (\n"
        "            not includes_draft_id_mapping\n"
        '            and "draft_id_to_target_id" in skip_substrs\n'
        "        ):\n"
        '            intentional_default_params.add("draft_id_to_target_id")\n'
        "        if (\n"
        "            not self.has_own_embed_tokens\n"
        '            and "embed_tokens" in skip_substrs\n'
        "        ):\n"
        '            intentional_default_params.add("model.embed_tokens.weight")\n'
        "        if not self.has_own_lm_head:\n"
        '            intentional_default_params.add("lm_head.weight")\n'
        "        return loaded_weights | intentional_default_params\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
            logger.info("Llama draft loader result patch already applied.")
            return
        if "intentional_default_params" in content:
            raise RuntimeError(
                "Found an incomplete Llama draft loader result patch in "
                f"{file_to_patch}; refusing to continue."
            )
        if content.count("loader.load_weights(model_weights.items())") != 1:
            raise RuntimeError(
                "Could not apply the Llama draft loader result patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )

        if content.count(legacy_snippet) == 1:
            matched_snippet = legacy_snippet
        elif content.count(legacy_receipt_snippet) == 1:
            matched_snippet = legacy_receipt_snippet
        elif content.count(old_snippet) == 1:
            matched_snippet = old_snippet
        else:
            raise RuntimeError(
                "Could not apply the Llama draft loader result patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(matched_snippet, new_snippet, 1))
    logger.info("Successfully patched Llama draft loader result reporting.")


def _patch_vllm_missing_draft_probs_fail_closed(logger) -> None:
    """Reject a partial probabilistic-draft cache instead of changing sampling."""
    file_to_patch = _get_vllm_file("v1/worker/gpu_model_runner.py")
    old_snippet = (
        "            if row_idx is None:\n"
        "                logger.warning(\n"
        '                    "Missing cached draft probabilities for request %s; "\n'
        '                    "falling back to legacy speculative rejection behavior.",\n'
        "                    req_id,\n"
        "                )\n"
        "                return None\n"
    )
    new_snippet = (
        "            if row_idx is None:\n"
        "                raise RuntimeError(\n"
        '                    "Probabilistic speculative decoding is missing q(token) "\n'
        '                    f"for request {req_id}; refusing to fall back to legacy "\n'
        '                    "rejection behavior because it changes the sampling contract."\n'
        "                )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "missing q(token)" in content:
            logger.info(
                "Missing probabilistic draft-row fail-closed patch already applied."
            )
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the missing probabilistic draft-row fail-closed "
                f"patch to {file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched missing probabilistic draft rows to fail closed.")


def _patch_vllm_parallel_probabilistic_draft_temperature(logger) -> None:
    """Expand per-request temperatures for parallel probabilistic drafts."""
    file_to_patch = _get_vllm_file("v1/spec_decode/llm_base_proposer.py")
    old_snippet = (
        "    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)\n"
        "    # consistent with sampler.py's _SAMPLING_EPS threshold\n"
        "    temperature = sampling_metadata.temperature\n"
        "    # Avoid division by zero if there are greedy requests.\n"
        "    if not sampling_metadata.all_random:\n"
        "        is_greedy = temperature < _SAMPLING_EPS\n"
        "        temperature = torch.where(is_greedy, 1.0, temperature)\n"
        "    logits.div_(temperature.view(-1, 1))\n"
    )
    new_snippet = (
        "    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)\n"
        "    # consistent with sampler.py's _SAMPLING_EPS threshold\n"
        "    temperature = sampling_metadata.temperature\n"
        "    temperature_count = temperature.numel()\n"
        "    logits_count = logits.shape[0]\n"
        "    if logits_count <= 0 or temperature_count <= 0:\n"
        "        raise RuntimeError(\n"
        '            "parallel draft logits and sampling temperature counts must be "\n'
        '            f"positive: logits={logits_count}, temperatures={temperature_count}"\n'
        "        )\n"
        "    if temperature_count != logits_count:\n"
        "        if logits_count % temperature_count != 0:\n"
        "            raise RuntimeError(\n"
        '                "parallel draft logits count is not divisible by the sampling "\n'
        '                f"temperature count: logits={logits_count}, "\n'
        '                f"temperatures={temperature_count}"\n'
        "            )\n"
        "        temperature = temperature.repeat_interleave(logits_count // temperature_count)\n"
        "    # Avoid division by zero if there are greedy requests.\n"
        "    if not sampling_metadata.all_random:\n"
        "        is_greedy = temperature < _SAMPLING_EPS\n"
        "        temperature = torch.where(is_greedy, 1.0, temperature)\n"
        "    logits.div_(temperature.view(-1, 1))\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if content.count(new_snippet) == 1:
            logger.info(
                "Parallel probabilistic draft temperature patch already applied."
            )
            return
        if "temperature_count = temperature.numel()" in content:
            raise RuntimeError(
                "Found an incomplete parallel probabilistic draft temperature patch "
                f"in {file_to_patch}; refusing to continue."
            )
        if content.count(old_snippet) != 1:
            raise RuntimeError(
                "Could not apply the parallel probabilistic draft temperature patch "
                f"to {file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched parallel probabilistic draft temperatures.")


def _patch_vllm_draft_model_load_config(logger) -> None:
    """Make the generic draft-model proposer honor draft_load_config."""
    file_to_patch = _get_vllm_file("v1/spec_decode/draft_model.py")
    new_line = "            load_config=spec.draft_load_config or base.load_config,\n"
    old_snippet = (
        "        return replace(\n            base,\n            quant_config=None,\n"
    )
    new_snippet = (
        "        return replace(\n"
        "            base,\n"
        "            load_config=spec.draft_load_config or base.load_config,\n"
        "            quant_config=None,\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_line in content:
            logger.info("Generic draft-model load-config patch already applied.")
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the generic draft-model load-config patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched generic draft-model load config.")


def _patch_vllm_medusa_load_config(logger) -> None:
    """Make the Medusa proposer honor draft_load_config."""
    file_to_patch = _get_vllm_file("v1/spec_decode/medusa.py")
    new_line = (
        "                load_config=self.spec_config.draft_load_config "
        "or self.vllm_config.load_config,\n"
    )
    old_snippet = (
        "            self.model = get_model(\n"
        "                vllm_config=self.vllm_config,\n"
        "                model_config=self.spec_config.draft_model_config,\n"
        "            )\n"
    )
    new_snippet = (
        "            self.model = get_model(\n"
        "                vllm_config=self.vllm_config,\n"
        "                model_config=self.spec_config.draft_model_config,\n"
        "                load_config=self.spec_config.draft_load_config "
        "or self.vllm_config.load_config,\n"
        "            )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_line in content:
            logger.info("Medusa load-config patch already applied.")
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the Medusa load-config patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully patched Medusa draft load config.")


def _patch_vllm_draft_model_cudagraph_keys(logger) -> None:
    """Initialize CUDA-graph keys for generic draft-model proposers.

    vLLM 0.24 initializes the draft proposer attention backend for both Eagle
    and generic draft models, but only initializes the proposer's CUDA-graph
    dispatcher for Eagle/extract-hidden-state methods. Parallel draft models
    such as PARD therefore miss their captured dispatcher path.

    The source mutation is installed once, but the inserted branch checks
    ``NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH`` at runtime. This keeps
    enabled and disabled runs isolated even when they share a worker venv.
    """
    file_to_patch = _get_vllm_file("v1/worker/gpu_model_runner.py")
    old_snippet_with_gemma4 = (
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "        ):\n"
        "            assert isinstance(\n"
        "                self.drafter,\n"
        "                EagleProposer\n"
        "                | DFlashProposer\n"
        "                | ExtractHiddenStatesProposer\n"
        "                | Gemma4Proposer,\n"
        "            )\n"
        "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
    )
    new_snippet_with_gemma4 = (
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "            or (\n"
        "                self.speculative_config.uses_draft_model()\n"
        "                and __import__('os').environ.get(\n"
        "                    'NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH',\n"
        "                    'false',\n"
        "                ).lower() == 'true'\n"
        "            )\n"
        "        ):\n"
        "            assert isinstance(\n"
        "                self.drafter,\n"
        "                EagleProposer\n"
        "                | DFlashProposer\n"
        "                | DraftModelProposer\n"
        "                | ExtractHiddenStatesProposer\n"
        "                | Gemma4Proposer,\n"
        "            )\n"
        "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
    )
    old_snippet_without_gemma4 = (
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "        ):\n"
        "            assert isinstance(\n"
        "                self.drafter,\n"
        "                EagleProposer | DFlashProposer | ExtractHiddenStatesProposer,\n"
        "            )\n"
        "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
    )
    new_snippet_without_gemma4 = (
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "            or (\n"
        "                self.speculative_config.uses_draft_model()\n"
        "                and __import__('os').environ.get(\n"
        "                    'NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH',\n"
        "                    'false',\n"
        "                ).lower() == 'true'\n"
        "            )\n"
        "        ):\n"
        "            assert isinstance(\n"
        "                self.drafter,\n"
        "                EagleProposer\n"
        "                | DFlashProposer\n"
        "                | DraftModelProposer\n"
        "                | ExtractHiddenStatesProposer,\n"
        "            )\n"
        "            self.drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH" in content:
            logger.info("Generic draft-model CUDA-graph patch already applied.")
            return
        for old_snippet, new_snippet in (
            (old_snippet_with_gemma4, new_snippet_with_gemma4),
            (old_snippet_without_gemma4, new_snippet_without_gemma4),
        ):
            if old_snippet in content:
                write_back(content.replace(old_snippet, new_snippet, 1))
                break
        else:
            raise RuntimeError(
                "Could not apply the generic draft-model CUDA-graph patch to "
                f"{file_to_patch}; the vLLM source layout changed."
            )
    logger.info("Installed runtime-guarded generic draft-model CUDA-graph keys.")


def _patch_vllm_hermes_tool_parser_thread_safety(logger) -> None:
    """Patch Hermes2ProToolParser.__init__ to cache tokenizer calls.

    The HuggingFace tokenizer's Rust backend does not support concurrent
    access. When multiple async requests call _preprocess_chat concurrently,
    each one constructs a new Hermes2ProToolParser which calls
    tokenizer.encode() and tokenizer.decode() in __init__, causing
    "RuntimeError: Already borrowed".

    A lock alone is insufficient because the tool parser's encode() can
    race with render_chat_async() in another concurrent request - two
    different codepaths sharing the same tokenizer instance.

    This patch caches the encode/decode results so only the first
    instantiation (protected by a lock) touches the tokenizer. All
    subsequent instantiations read from cache without any tokenizer
    access.

    Related:
    - https://github.com/vllm-project/vllm/pull/30264
    - https://github.com/huggingface/tokenizers/issues/537
    - https://github.com/PrimeIntellect-ai/prime-rl/pull/1837
    """
    file_to_patch = _get_vllm_file("tool_parsers/hermes_tool_parser.py")

    old_import = "import json\nfrom collections.abc import Sequence"
    new_import = "import json\nimport threading\nfrom collections.abc import Sequence"

    old_class_line = "class Hermes2ProToolParser(ToolParser):"
    new_class_line = (
        "class Hermes2ProToolParser(ToolParser):\n"
        "    _tokenizer_lock = threading.Lock()\n"
        "    _tokenizer_cache = {}"
    )

    old_init_snippet = (
        "        self.tool_call_start_token_ids = self.model_tokenizer.encode(\n"
        "            self.tool_call_start_token, add_special_tokens=False\n"
        "        )\n"
        "        self.tool_call_end_token_ids = self.model_tokenizer.encode(\n"
        "            self.tool_call_end_token, add_special_tokens=False\n"
        "        )\n"
        "\n"
        "        self.tool_call_start_token_array = [\n"
        "            self.model_tokenizer.decode([token_id])\n"
        "            for token_id in self.tool_call_start_token_ids\n"
        "        ]\n"
        "\n"
        "        self.tool_call_end_token_array = [\n"
        "            self.model_tokenizer.decode([token_id])\n"
        "            for token_id in self.tool_call_end_token_ids\n"
        "        ]"
    )

    new_init_snippet = (
        "        _tid = id(self.model_tokenizer)\n"
        "        if _tid in Hermes2ProToolParser._tokenizer_cache:\n"
        "            _cached = Hermes2ProToolParser._tokenizer_cache[_tid]\n"
        "            self.tool_call_start_token_ids = _cached['start_ids']\n"
        "            self.tool_call_end_token_ids = _cached['end_ids']\n"
        "            self.tool_call_start_token_array = _cached['start_array']\n"
        "            self.tool_call_end_token_array = _cached['end_array']\n"
        "        else:\n"
        "            with Hermes2ProToolParser._tokenizer_lock:\n"
        "                if _tid in Hermes2ProToolParser._tokenizer_cache:\n"
        "                    _cached = Hermes2ProToolParser._tokenizer_cache[_tid]\n"
        "                    self.tool_call_start_token_ids = _cached['start_ids']\n"
        "                    self.tool_call_end_token_ids = _cached['end_ids']\n"
        "                    self.tool_call_start_token_array = _cached['start_array']\n"
        "                    self.tool_call_end_token_array = _cached['end_array']\n"
        "                else:\n"
        "                    self.tool_call_start_token_ids = self.model_tokenizer.encode(\n"
        "                        self.tool_call_start_token, add_special_tokens=False\n"
        "                    )\n"
        "                    self.tool_call_end_token_ids = self.model_tokenizer.encode(\n"
        "                        self.tool_call_end_token, add_special_tokens=False\n"
        "                    )\n"
        "                    self.tool_call_start_token_array = [\n"
        "                        self.model_tokenizer.decode([token_id])\n"
        "                        for token_id in self.tool_call_start_token_ids\n"
        "                    ]\n"
        "                    self.tool_call_end_token_array = [\n"
        "                        self.model_tokenizer.decode([token_id])\n"
        "                        for token_id in self.tool_call_end_token_ids\n"
        "                    ]\n"
        "                    Hermes2ProToolParser._tokenizer_cache[_tid] = {\n"
        "                        'start_ids': self.tool_call_start_token_ids,\n"
        "                        'end_ids': self.tool_call_end_token_ids,\n"
        "                        'start_array': self.tool_call_start_token_array,\n"
        "                        'end_array': self.tool_call_end_token_array,\n"
        "                    }"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "_tokenizer_cache" in content:
            logger.info("Hermes tool parser thread-safety patch already applied.")
            return

        if old_init_snippet not in content:
            if (
                "self.model_tokenizer.encode(" not in content
                and "self.model_tokenizer.decode(" not in content
            ):
                logger.info(
                    "Hermes tool parser thread-safety patch is not required: "
                    "the parser no longer calls tokenizer encode/decode during "
                    "initialization."
                )
                return
            logger.warning(
                "Could not apply hermes tool parser thread-safety patch: "
                "expected code snippet not found in %s. "
                "The vLLM version may have changed.",
                file_to_patch,
            )
            return

        content = content.replace(old_import, new_import, 1)
        content = content.replace(old_class_line, new_class_line, 1)
        content = content.replace(old_init_snippet, new_init_snippet, 1)
        write_back(content)

    logger.info("Successfully patched hermes tool parser for thread-safety.")


def _apply_vllm_patches(
    py_executable: str,
    *,
    extra_env_vars: list[str] | None = None,
    speculative_config: dict[str, Any] | None,
) -> None:
    # Import lazily so importing the worker module does not import vLLM.
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")

    _patch_vllm_init_workers_ray(py_executable, extra_env_vars)
    patch_logger.info("Successfully patched vllm _init_workers_ray.")

    if speculative_config:
        _patch_vllm_llama_eagle3_own_lm_head(patch_logger)
        _patch_vllm_online_eagle_head_ownership(patch_logger)
        _patch_vllm_v2_eagle_load_config_and_ownership(patch_logger)
        _patch_vllm_v2_dflash_load_config(patch_logger)
        _patch_vllm_qwen3_draft_loader_results(patch_logger)
        _patch_vllm_llama_draft_loader_result(patch_logger)
        if (
            speculative_config.get("rejection_sample_method", "standard") == "standard"
            and speculative_config.get("draft_sample_method", "greedy")
            == "probabilistic"
        ):
            _patch_vllm_parallel_probabilistic_draft_temperature(patch_logger)
            _patch_vllm_missing_draft_probs_fail_closed(patch_logger)
        _patch_vllm_draft_model_load_config(patch_logger)
        _patch_vllm_medusa_load_config(patch_logger)
        if (
            speculative_config.get("method") == "draft_model"
            and os.environ.get(
                "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH", "false"
            ).lower()
            == "true"
        ):
            _patch_vllm_draft_model_cudagraph_keys(patch_logger)
    _patch_vllm_hermes_tool_parser_thread_safety(patch_logger)
