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
from functools import wraps
from importlib.util import find_spec
from typing import Any

type _RuntimePatchReplacement = tuple[str, str, tuple[str, ...]]
type _RuntimePatchSpec = tuple[str, tuple[_RuntimePatchReplacement, ...]]


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


@contextmanager
def _locked_file_patches(file_paths: list[str]):
    """Yield file contents and a writer while holding every file lock."""
    import fcntl

    ordered_paths = list(dict.fromkeys(sorted(file_paths)))
    lock_fds = []
    try:
        for file_path in ordered_paths:
            lock_fd = open(file_path + ".patch_lock", "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            lock_fds.append(lock_fd)

        contents = {}
        for file_path in ordered_paths:
            with open(file_path, "r") as f:
                contents[file_path] = f.read()

        def write_back(file_path: str, new_content: str):
            with open(file_path, "w") as f:
                f.write(new_content)

        yield contents, write_back
    finally:
        for lock_fd in reversed(lock_fds):
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
        "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS",
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
    """Reject missing probabilistic-draft probabilities instead of changing sampling."""
    file_to_patch = _get_vllm_file("v1/worker/gpu_model_runner.py")
    old_cache_snippet = (
        "        if self._draft_probs is None or self._draft_prob_req_ids is None:\n"
        "            return None\n"
    )
    new_cache_snippet = (
        "        if self._draft_probs is None or self._draft_prob_req_ids is None:\n"
        "            if (\n"
        "                any(spec_decode_metadata.num_draft_tokens)\n"
        "                and not self.input_batch.sampling_metadata.all_greedy\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        '                    "Probabilistic speculative decoding has no cached q(token) "\n'
        '                    "for a batch with draft tokens; refusing to fall back to "\n'
        '                    "legacy rejection behavior because it changes the sampling "\n'
        '                    "contract."\n'
        "                )\n"
        "            return None\n"
    )
    old_row_snippet = (
        "            if row_idx is None:\n"
        "                logger.warning(\n"
        '                    "Missing cached draft probabilities for request %s; "\n'
        '                    "falling back to legacy speculative rejection behavior.",\n'
        "                    req_id,\n"
        "                )\n"
        "                return None\n"
    )
    new_row_snippet = (
        "            if row_idx is None:\n"
        "                raise RuntimeError(\n"
        '                    "Probabilistic speculative decoding is missing q(token) "\n'
        '                    f"for request {req_id}; refusing to fall back to legacy "\n'
        '                    "rejection behavior because it changes the sampling contract."\n'
        "                )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if (
            content.count(new_cache_snippet) == 1
            and content.count(new_row_snippet) == 1
        ):
            logger.info(
                "Missing probabilistic draft-probability fail-closed patch already applied."
            )
            return
        if "has no cached q(token)" in content or "missing q(token)" in content:
            raise RuntimeError(
                "Found an incomplete probabilistic draft-probability fail-closed "
                f"patch in {file_to_patch}; refusing to continue."
            )
        if content.count(old_cache_snippet) != 1 or content.count(old_row_snippet) != 1:
            raise RuntimeError(
                "Could not apply the missing probabilistic draft-probability fail-closed "
                f"patch to {file_to_patch}; the vLLM source layout changed."
            )
        content = content.replace(old_cache_snippet, new_cache_snippet, 1)
        write_back(content.replace(old_row_snippet, new_row_snippet, 1))
    logger.info("Successfully patched missing probabilistic draft probabilities.")


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


def _patch_vllm_piecewise_specdec_cudagraph_alignment(logger) -> None:
    """Align PIECEWISE capture sizes to the speculative decode query length.

    vLLM 0.24 only performs this alignment when the decode graph mode is FULL.
    PIECEWISE graphs use the same uniform K+1 decode layout, so unaligned
    captures can fall back to eager execution or use invalid slot offsets.
    """
    file_to_patch = _get_vllm_file("config/compilation.py")
    old_snippet = (
        "        if (\n"
        "            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
        "            and uniform_decode_query_len > 1\n"
        "        ):\n"
        "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
        "                uniform_decode_query_len,\n"
        "                tensor_parallel_size,\n"
        "            )\n"
    )
    new_snippet = (
        "        if (\n"
        "            cudagraph_mode != CUDAGraphMode.NONE\n"
        "            and uniform_decode_query_len > 1\n"
        "        ):\n"
        "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
        "                uniform_decode_query_len,\n"
        "                tensor_parallel_size,\n"
        "            )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
            logger.info("PIECEWISE SpecDec CUDA-graph alignment already applied.")
            return
        if old_snippet not in content:
            raise RuntimeError(
                "Could not apply the PIECEWISE SpecDec CUDA-graph alignment "
                f"patch to {file_to_patch}; the vLLM source layout changed."
            )
        write_back(content.replace(old_snippet, new_snippet, 1))
    logger.info("Successfully aligned PIECEWISE SpecDec CUDA-graph sizes.")


def _patch_vllm_runtime_tail_gating(logger) -> None:
    """Patch the pinned vLLM 0.24 binary runtime-K tail-gating contract."""
    config_old = (
        "    # dynamic speculative decoding control\n"
        "    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None\n"
        '    """Batch-size schedule used to dynamically choose speculative-token count.\n'
        "\n"
        "    Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an\n"
        "    inclusive batch-size range.\n"
        '    """\n'
        "\n"
        "    # params generated in the post-init stage\n"
    )
    config_new = (
        "    # dynamic speculative decoding control\n"
        "    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None\n"
        '    """Batch-size schedule used to dynamically choose speculative-token count.\n'
        "\n"
        "    Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an\n"
        "    inclusive batch-size range.\n"
        '    """\n'
        "\n"
        "    # NeMo-RL binary tail-gate runtime control.\n"
        '    sd_tail_gate_mode: str = "off"\n'
        "    sd_tail_gate_threshold: int | None = None\n"
        "    sd_tail_gate_consecutive_checks: int = 10\n"
        "    sd_tail_gate_margin: float = 0.05\n"
        "    sd_tail_gate_config_path: str | None = None\n"
        '    sd_tail_gate_off_mode: str = "advance_only"\n'
        "\n"
        "    # params generated in the post-init stage\n"
    )

    scheduler_output_old = (
        "    # Dynamic speculative decoding: optimal K chosen by scheduler.\n"
        "    # Number of spec tokens to schedule for the next step.\n"
        "    num_spec_tokens_to_schedule: int = 0\n"
        "\n"
        "    @classmethod\n"
        '    def make_empty(cls) -> "SchedulerOutput":\n'
    )
    scheduler_output_new = (
        "    # Dynamic speculative decoding: optimal K chosen by scheduler.\n"
        "    # Number of spec tokens to schedule for the next step.\n"
        "    num_spec_tokens_to_schedule: int = 0\n"
        "\n"
        "    # NeMo-RL tail-gate scheduler telemetry.\n"
        '    tail_gate_state: str = "OFF"\n'
        "    tail_gate_tick: int = 0\n"
        "    tail_gate_active_requests: int = 0\n"
        "    tail_gate_decode_active_requests: int = 0\n"
        "    tail_gate_mean_sequence_length: float = 0.0\n"
        "    tail_gate_predicted_speedup_sum: float = 0.0\n"
        "    tail_gate_predicted_speedup_count: int = 0\n"
        "    tail_gate_expected_accept_length: float = 0.0\n"
        "    tail_gate_just_activated: bool = False\n"
        "\n"
        "    @classmethod\n"
        '    def make_empty(cls) -> "SchedulerOutput":\n'
    )

    tail_gate_telemetry_old = (
        '                "vllm:spec_decode_tail_gate_disabled_steps": float(\n'
        "                    effective_runtime_k == 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_decode_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_decode_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_sum": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_count": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_count\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_sum": float(\n'
        "                    scheduler_output.tail_gate_expected_accept_length\n"
        "                ),\n"
        '                f"vllm:spec_decode_tail_gate_k_{effective_runtime_k}_steps": 1.0,\n'
    )
    tail_gate_telemetry_new = (
        '                "vllm:spec_decode_tail_gate_disabled_steps": float(\n'
        "                    effective_runtime_k == 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_active_requests_count": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_decode_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_decode_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_decode_active_requests_count": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_mean_sequence_length_sum": float(\n'
        "                    scheduler_output.tail_gate_mean_sequence_length\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_mean_sequence_length_count": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_predicted_speedup_sum": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_count": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_count\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_sum": float(\n'
        "                    scheduler_output.tail_gate_expected_accept_length\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_count": 1.0,\n'
        '                f"vllm:spec_decode_tail_gate_k_{effective_runtime_k}_steps": 1.0,\n'
    )

    v2_execute_old = (
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        "        scheduler_output: SchedulerOutput,\n"
        "        intermediate_tensors: IntermediateTensors | None = None,\n"
        "        dummy_run: bool = False,\n"
        "        skip_attn_for_dummy_run: bool = False,\n"
        "        is_profile: bool = False,\n"
        "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n"
        "        if not dummy_run:\n"
    )
    v2_execute_new = (
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        "        scheduler_output: SchedulerOutput,\n"
        "        intermediate_tensors: IntermediateTensors | None = None,\n"
        "        dummy_run: bool = False,\n"
        "        skip_attn_for_dummy_run: bool = False,\n"
        "        is_profile: bool = False,\n"
        "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n"
        "        runtime_num_spec_tokens = None\n"
        "        tail_gate_mode = getattr(\n"
        '            self.speculative_config, "sd_tail_gate_mode", "off"\n'
        "        )\n"
        '        if tail_gate_mode != "off" and not dummy_run:\n'
        "            runtime_num_spec_tokens = (\n"
        "                scheduler_output.num_spec_tokens_to_schedule\n"
        "            )\n"
        "            if runtime_num_spec_tokens is not None and (\n"
        "                isinstance(runtime_num_spec_tokens, bool)\n"
        "                or runtime_num_spec_tokens not in (0, self.num_speculative_steps)\n"
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Tail-gated Model Runner V2 runtime K must be None, 0, "\n'
        '                    f"or the configured maximum {self.num_speculative_steps}; "\n'
        '                    f"got {runtime_num_spec_tokens}."\n'
        "                )\n"
        "            if (\n"
        "                self.speculative_config is None\n"
        '                or self.speculative_config.method not in ("eagle", "eagle3")\n'
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Tail-gated Model Runner V2 requires an external "\n'
        '                    "Eagle or Eagle-3 speculator."\n'
        "                )\n"
        "            if (\n"
        "                getattr(\n"
        "                    self.speculative_config,\n"
        '                    "sd_tail_gate_off_mode",\n'
        '                    "advance_only",\n'
        "                )\n"
        '                != "advance_only"\n'
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Model Runner V2 binary tail gating only supports "\n'
        '                    "sd_tail_gate_off_mode=advance_only."\n'
        "                )\n"
        "\n"
        "            tail_gate_metrics = getattr(\n"
        '                self, "_nrl_tail_gate_metrics", None\n'
        "            )\n"
        "            if tail_gate_metrics is None:\n"
        "                tail_gate_metrics = {}\n"
        "                self._nrl_tail_gate_metrics = tail_gate_metrics\n"
        "            effective_runtime_k = (\n"
        "                self.num_speculative_steps\n"
        "                if runtime_num_spec_tokens is None\n"
        "                else runtime_num_spec_tokens\n"
        "            )\n"
        "            tail_gate_updates = {\n"
        '                "vllm:spec_decode_tail_gate_decisions": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_enabled_steps": float(\n'
        "                    effective_runtime_k > 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_disabled_steps": float(\n'
        "                    effective_runtime_k == 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_decode_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_decode_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_sum": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_count": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_count\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_sum": float(\n'
        "                    scheduler_output.tail_gate_expected_accept_length\n"
        "                ),\n"
        '                f"vllm:spec_decode_tail_gate_k_{effective_runtime_k}_steps": 1.0,\n'
        "            }\n"
        "            if scheduler_output.tail_gate_just_activated:\n"
        "                tail_gate_updates.update(\n"
        "                    {\n"
        '                        "vllm:spec_decode_tail_gate_activations": 1.0,\n'
        '                        "vllm:spec_decode_tail_gate_activation_batch_sum": float(\n'
        "                            scheduler_output.tail_gate_active_requests\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": float(\n'
        "                            scheduler_output.tail_gate_mean_sequence_length\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_tick_sum": float(\n'
        "                            scheduler_output.tail_gate_tick\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_tick_count": 1.0,\n'
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_sum": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_count": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_count\n"
        "                        ),\n"
        "                    }\n"
        "                )\n"
        "            tail_gate_state = scheduler_output.tail_gate_state.lower()\n"
        "            if tail_gate_state in (\n"
        '                "ramping_off",\n'
        '                "armed_off",\n'
        '                "on_latched",\n'
        "            ):\n"
        "                tail_gate_updates[\n"
        '                    f"vllm:spec_decode_tail_gate_{tail_gate_state}_steps"\n'
        "                ] = 1.0\n"
        "            for metric_name, metric_value in tail_gate_updates.items():\n"
        "                tail_gate_metrics[metric_name] = (\n"
        "                    tail_gate_metrics.get(metric_name, 0.0) + metric_value\n"
        "                )\n"
        "\n"
        "        if not dummy_run:\n"
    )
    if v2_execute_new.count(tail_gate_telemetry_old) != 1:
        raise RuntimeError("Internal V2 tail-gate telemetry anchor changed.")
    v2_execute_activation_tick_legacy = v2_execute_new
    v2_execute_new = v2_execute_new.replace(
        tail_gate_telemetry_old, tail_gate_telemetry_new, 1
    )
    v2_execute_legacy = (
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        "        scheduler_output: SchedulerOutput,\n"
        "        intermediate_tensors: IntermediateTensors | None = None,\n"
        "        dummy_run: bool = False,\n"
        "        skip_attn_for_dummy_run: bool = False,\n"
        "        is_profile: bool = False,\n"
        "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n"
        "        runtime_num_spec_tokens = None\n"
        "        tail_gate_mode = getattr(\n"
        '            self.speculative_config, "sd_tail_gate_mode", "off"\n'
        "        )\n"
        '        if tail_gate_mode != "off" and not dummy_run:\n'
        "            runtime_num_spec_tokens = (\n"
        "                scheduler_output.num_spec_tokens_to_schedule\n"
        "            )\n"
        "            if runtime_num_spec_tokens is not None and (\n"
        "                isinstance(runtime_num_spec_tokens, bool)\n"
        "                or runtime_num_spec_tokens not in (0, self.num_speculative_steps)\n"
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Tail-gated Model Runner V2 runtime K must be None, 0, "\n'
        '                    f"or the configured maximum {self.num_speculative_steps}; "\n'
        '                    f"got {runtime_num_spec_tokens}."\n'
        "                )\n"
        "            if (\n"
        "                self.speculative_config is None\n"
        '                or self.speculative_config.method not in ("eagle", "eagle3")\n'
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Tail-gated Model Runner V2 requires an external "\n'
        '                    "Eagle or Eagle-3 speculator."\n'
        "                )\n"
        "            if (\n"
        "                getattr(\n"
        "                    self.speculative_config,\n"
        '                    "sd_tail_gate_off_mode",\n'
        '                    "advance_only",\n'
        "                )\n"
        '                != "advance_only"\n'
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Model Runner V2 binary tail gating only supports "\n'
        '                    "sd_tail_gate_off_mode=advance_only."\n'
        "                )\n"
        "\n"
        "            tail_gate_metrics = getattr(\n"
        '                self, "_nrl_tail_gate_metrics", None\n'
        "            )\n"
        "            if tail_gate_metrics is None:\n"
        "                tail_gate_metrics = {}\n"
        "                self._nrl_tail_gate_metrics = tail_gate_metrics\n"
        "            effective_runtime_k = (\n"
        "                self.num_speculative_steps\n"
        "                if runtime_num_spec_tokens is None\n"
        "                else runtime_num_spec_tokens\n"
        "            )\n"
        "            tail_gate_updates = {\n"
        '                "vllm:spec_decode_tail_gate_decisions": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_enabled_steps": float(\n'
        "                    effective_runtime_k > 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_disabled_steps": float(\n'
        "                    effective_runtime_k == 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_decode_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_decode_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_sum": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_count": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_count\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_sum": float(\n'
        "                    scheduler_output.tail_gate_expected_accept_length\n"
        "                ),\n"
        '                f"vllm:spec_decode_tail_gate_k_{effective_runtime_k}_steps": 1.0,\n'
        "            }\n"
        "            if scheduler_output.tail_gate_just_activated:\n"
        "                tail_gate_updates.update(\n"
        "                    {\n"
        '                        "vllm:spec_decode_tail_gate_activations": 1.0,\n'
        '                        "vllm:spec_decode_tail_gate_activation_batch_sum": float(\n'
        "                            scheduler_output.tail_gate_active_requests\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": float(\n'
        "                            scheduler_output.tail_gate_mean_sequence_length\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_sum": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_count": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_count\n"
        "                        ),\n"
        "                    }\n"
        "                )\n"
        "            tail_gate_state = scheduler_output.tail_gate_state.lower()\n"
        "            if tail_gate_state in (\n"
        '                "ramping_off",\n'
        '                "armed_off",\n'
        '                "on_latched",\n'
        "            ):\n"
        "                tail_gate_updates[\n"
        '                    f"vllm:spec_decode_tail_gate_{tail_gate_state}_steps"\n'
        "                ] = 1.0\n"
        "            for metric_name, metric_value in tail_gate_updates.items():\n"
        "                tail_gate_metrics[metric_name] = (\n"
        "                    tail_gate_metrics.get(metric_name, 0.0) + metric_value\n"
        "                )\n"
        "\n"
        "        if not dummy_run:\n"
    )

    v2_state_old = (
        "        finished_req_ids = scheduler_output.finished_req_ids\n"
        "        self.execute_model_state = ExecuteModelState(\n"
        "            input_batch=input_batch,\n"
        "            attn_metadata=attn_metadata,\n"
        "            slot_mappings_by_layer=slot_mappings_by_layer,\n"
        "            hidden_states=hidden_states,\n"
        "            aux_hidden_states=aux_hidden_states,\n"
        "            finished_req_ids=finished_req_ids,\n"
        "        )\n"
    )
    v2_state_new = (
        "        finished_req_ids = scheduler_output.finished_req_ids\n"
        "        self.execute_model_state = ExecuteModelState(\n"
        "            input_batch=input_batch,\n"
        "            attn_metadata=attn_metadata,\n"
        "            slot_mappings_by_layer=slot_mappings_by_layer,\n"
        "            hidden_states=hidden_states,\n"
        "            aux_hidden_states=aux_hidden_states,\n"
        "            num_spec_tokens_to_schedule=runtime_num_spec_tokens,\n"
        "            finished_req_ids=finished_req_ids,\n"
        "        )\n"
    )

    v2_sample_state_old = (
        "        input_batch = self.execute_model_state.input_batch\n"
        "        attn_metadata = self.execute_model_state.attn_metadata\n"
        "        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer\n"
        "        hidden_states = self.execute_model_state.hidden_states\n"
        "        aux_hidden_states = self.execute_model_state.aux_hidden_states\n"
        "        finished_req_ids = self.execute_model_state.finished_req_ids\n"
        "        self.execute_model_state = None\n"
    )
    v2_sample_state_new = (
        "        input_batch = self.execute_model_state.input_batch\n"
        "        attn_metadata = self.execute_model_state.attn_metadata\n"
        "        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer\n"
        "        hidden_states = self.execute_model_state.hidden_states\n"
        "        aux_hidden_states = self.execute_model_state.aux_hidden_states\n"
        "        finished_req_ids = self.execute_model_state.finished_req_ids\n"
        "        runtime_num_spec_tokens = (\n"
        "            self.execute_model_state.num_spec_tokens_to_schedule\n"
        "        )\n"
        "        self.execute_model_state = None\n"
    )

    v2_handler_init_old = (
        "        self.postprocess_sampled(\n"
        "            input_batch.idx_mapping,\n"
        "            sampler_output.sampled_token_ids,\n"
        "            num_sampled,\n"
        "            num_rejected,\n"
        "            input_batch.query_start_loc,\n"
        "        )\n"
        "\n"
        "        if self.speculator is not None:\n"
        "            assert self.sampler is not None\n"
    )
    v2_handler_init_new = (
        "        self.postprocess_sampled(\n"
        "            input_batch.idx_mapping,\n"
        "            sampler_output.sampled_token_ids,\n"
        "            num_sampled,\n"
        "            num_rejected,\n"
        "            input_batch.query_start_loc,\n"
        "        )\n"
        "\n"
        "        draft_tokens_for_handler = None\n"
        "        if self.speculator is not None:\n"
        "            assert self.sampler is not None\n"
    )

    v2_proposal_old = (
        "            draft_tokens = self.speculator.propose(\n"
        "                input_batch,\n"
        "                attn_metadata,\n"
        "                slot_mappings_by_layer,\n"
        "                spec_hidden_states,\n"
        "                aux_hidden_states,\n"
        "                num_sampled,\n"
        "                num_rejected,\n"
        "                self.req_states.last_sampled_tokens,\n"
        "                self.req_states.next_prefill_tokens,\n"
        "                self.sampler.sampling_states.temperature.gpu,\n"
        "                self.sampler.sampling_states.seeds.gpu,\n"
        "                mm_inputs=mm_inputs,\n"
        "            )\n"
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
        "\n"
        "        if self.num_speculative_steps > 0:\n"
        "            # Spec-decode and diffusion LLMs both use draft tokens but the latter does\n"
        "            # not have a speculator (i.e. self.speculator is None)\n"
        "            self.draft_tokens_handler.set_draft_tokens(\n"
        "                input_batch,\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
        "            )\n"
    )
    v2_proposal_new = (
        "            if runtime_num_spec_tokens is None:\n"
        "                draft_tokens = self.speculator.propose(\n"
        "                    input_batch,\n"
        "                    attn_metadata,\n"
        "                    slot_mappings_by_layer,\n"
        "                    spec_hidden_states,\n"
        "                    aux_hidden_states,\n"
        "                    num_sampled,\n"
        "                    num_rejected,\n"
        "                    self.req_states.last_sampled_tokens,\n"
        "                    self.req_states.next_prefill_tokens,\n"
        "                    self.sampler.sampling_states.temperature.gpu,\n"
        "                    self.sampler.sampling_states.seeds.gpu,\n"
        "                    mm_inputs=mm_inputs,\n"
        "                )\n"
        "            else:\n"
        "                draft_tokens = self.speculator.propose(\n"
        "                    input_batch,\n"
        "                    attn_metadata,\n"
        "                    slot_mappings_by_layer,\n"
        "                    spec_hidden_states,\n"
        "                    aux_hidden_states,\n"
        "                    num_sampled,\n"
        "                    num_rejected,\n"
        "                    self.req_states.last_sampled_tokens,\n"
        "                    self.req_states.next_prefill_tokens,\n"
        "                    self.sampler.sampling_states.temperature.gpu,\n"
        "                    self.sampler.sampling_states.seeds.gpu,\n"
        "                    num_speculative_tokens=runtime_num_spec_tokens,\n"
        "                    mm_inputs=mm_inputs,\n"
        "                )\n"
        "            if runtime_num_spec_tokens == 0:\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping] = 0\n"
        "                draft_tokens_for_handler = draft_tokens\n"
        "            else:\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping] = (\n"
        "                    draft_tokens\n"
        "                )\n"
        "\n"
        "        if self.num_speculative_steps > 0:\n"
        "            # Spec-decode and diffusion LLMs both use draft tokens but the latter does\n"
        "            # not have a speculator (i.e. self.speculator is None)\n"
        "            if self.speculator is None or draft_tokens_for_handler is None:\n"
        "                draft_tokens_for_handler = self.req_states.draft_tokens[\n"
        "                    input_batch.idx_mapping\n"
        "                ]\n"
        "            self.draft_tokens_handler.set_draft_tokens(\n"
        "                input_batch,\n"
        "                draft_tokens_for_handler,\n"
        "            )\n"
    )

    v2_execute_state_old = (
        "class ExecuteModelState(NamedTuple):\n"
        "    input_batch: InputBatch\n"
        "    attn_metadata: dict[str, Any] | None\n"
        "    slot_mappings_by_layer: dict[str, torch.Tensor] | None\n"
        "    hidden_states: torch.Tensor | None\n"
        "    aux_hidden_states: list[torch.Tensor] | None\n"
        "    finished_req_ids: set[str]\n"
    )
    v2_execute_state_new = (
        "class ExecuteModelState(NamedTuple):\n"
        "    input_batch: InputBatch\n"
        "    attn_metadata: dict[str, Any] | None\n"
        "    slot_mappings_by_layer: dict[str, torch.Tensor] | None\n"
        "    hidden_states: torch.Tensor | None\n"
        "    aux_hidden_states: list[torch.Tensor] | None\n"
        "    num_spec_tokens_to_schedule: int | None\n"
        "    finished_req_ids: set[str]\n"
    )

    speculator_signature_old = (
        "    def propose(\n"
        "        self,\n"
        "        input_batch: InputBatch,\n"
        "        attn_metadata: dict[str, Any],\n"
        "        slot_mappings: dict[str, torch.Tensor],\n"
        "        # [num_tokens, hidden_size]\n"
        "        last_hidden_states: torch.Tensor,\n"
        "        # num_layers x [num_tokens, hidden_size]\n"
        "        aux_hidden_states: list[torch.Tensor] | None,\n"
        "        # [num_reqs]\n"
        "        num_sampled: torch.Tensor,\n"
        "        # [num_reqs]\n"
        "        num_rejected: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        last_sampled: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        next_prefill_tokens: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        temperature: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        seeds: torch.Tensor,\n"
        "        num_tokens_across_dp: torch.Tensor | None = None,\n"
        "        dummy_run: bool = False,\n"
        "        skip_attn_for_dummy_run: bool = False,\n"
        "        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,\n"
        "        is_profile: bool = False,\n"
        "    ) -> torch.Tensor:\n"
        "        num_tokens = input_batch.num_tokens_after_padding\n"
    )
    speculator_signature_new = (
        "    def propose(\n"
        "        self,\n"
        "        input_batch: InputBatch,\n"
        "        attn_metadata: dict[str, Any],\n"
        "        slot_mappings: dict[str, torch.Tensor],\n"
        "        # [num_tokens, hidden_size]\n"
        "        last_hidden_states: torch.Tensor,\n"
        "        # num_layers x [num_tokens, hidden_size]\n"
        "        aux_hidden_states: list[torch.Tensor] | None,\n"
        "        # [num_reqs]\n"
        "        num_sampled: torch.Tensor,\n"
        "        # [num_reqs]\n"
        "        num_rejected: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        last_sampled: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        next_prefill_tokens: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        temperature: torch.Tensor,\n"
        "        # [max_num_reqs]\n"
        "        seeds: torch.Tensor,\n"
        "        num_tokens_across_dp: torch.Tensor | None = None,\n"
        "        dummy_run: bool = False,\n"
        "        skip_attn_for_dummy_run: bool = False,\n"
        "        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,\n"
        "        is_profile: bool = False,\n"
        "        num_speculative_tokens: int | None = None,\n"
        "    ) -> torch.Tensor:\n"
        "        runtime_num_spec_tokens = (\n"
        "            self.num_speculative_steps\n"
        "            if num_speculative_tokens is None\n"
        "            else num_speculative_tokens\n"
        "        )\n"
        "        if isinstance(runtime_num_spec_tokens, bool) or (\n"
        "            runtime_num_spec_tokens not in (0, self.num_speculative_steps)\n"
        "        ):\n"
        "            raise ValueError(\n"
        '                "Autoregressive speculator runtime K must be 0 or the "\n'
        '                f"configured maximum {self.num_speculative_steps}; "\n'
        '                f"got {runtime_num_spec_tokens}."\n'
        "            )\n"
        "\n"
        "        num_tokens = input_batch.num_tokens_after_padding\n"
    )

    speculator_k0_old = (
        "            self._prefill(\n"
        "                num_reqs,\n"
        "                prefill_batch_desc.num_tokens,\n"
        "                attn_metadata,\n"
        "                slot_mappings,\n"
        "                num_tokens_across_dp=num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,\n"
        "                mm_inputs=mm_inputs,\n"
        "            )\n"
        "\n"
        "        if self.num_speculative_steps == 1:\n"
        "            # Early exit.\n"
        "            return self.draft_tokens[:num_reqs, :1]\n"
        "\n"
        "        # Prepare the inputs for the decode steps.\n"
        "        prepare_decode_inputs(\n"
    )
    speculator_k0_new = (
        "            self._prefill(\n"
        "                num_reqs,\n"
        "                prefill_batch_desc.num_tokens,\n"
        "                attn_metadata,\n"
        "                slot_mappings,\n"
        "                num_tokens_across_dp=num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,\n"
        "                mm_inputs=mm_inputs,\n"
        "            )\n"
        "\n"
        "        if runtime_num_spec_tokens == 0:\n"
        "            # The first pass advances external Eagle state; publish no drafts.\n"
        "            return self.draft_tokens[:num_reqs, :0]\n"
        "\n"
        "        if self.num_speculative_steps == 1:\n"
        "            # Early exit.\n"
        "            return self.draft_tokens[:num_reqs, :1]\n"
        "\n"
        "        # Prepare the inputs for the decode steps.\n"
        "        prepare_decode_inputs(\n"
    )

    v1_execute_old = (
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        '        scheduler_output: "SchedulerOutput",\n'
        "        intermediate_tensors: IntermediateTensors | None = None,\n"
        "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:\n"
        "        if self.execute_model_state is not None:\n"
        "            raise RuntimeError(\n"
        '                "State error: sample_tokens() must be called "\n'
        '                "after execute_model() returns None."\n'
        "            )\n"
        "\n"
        "        if self.routed_experts_initialized:\n"
    )
    v1_execute_new = (
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        '        scheduler_output: "SchedulerOutput",\n'
        "        intermediate_tensors: IntermediateTensors | None = None,\n"
        "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:\n"
        "        if self.execute_model_state is not None:\n"
        "            raise RuntimeError(\n"
        '                "State error: sample_tokens() must be called "\n'
        '                "after execute_model() returns None."\n'
        "            )\n"
        "\n"
        "        tail_gate_mode = getattr(\n"
        '            self.speculative_config, "sd_tail_gate_mode", "off"\n'
        "        )\n"
        '        if tail_gate_mode != "off":\n'
        "            runtime_num_spec_tokens = (\n"
        "                scheduler_output.num_spec_tokens_to_schedule\n"
        "            )\n"
        "            tail_gate_metrics = getattr(\n"
        '                self, "_nrl_tail_gate_metrics", None\n'
        "            )\n"
        "            if tail_gate_metrics is None:\n"
        "                tail_gate_metrics = {}\n"
        "                self._nrl_tail_gate_metrics = tail_gate_metrics\n"
        "            effective_runtime_k = (\n"
        "                self.num_spec_tokens\n"
        "                if runtime_num_spec_tokens is None\n"
        "                else runtime_num_spec_tokens\n"
        "            )\n"
        "            tail_gate_updates = {\n"
        '                "vllm:spec_decode_tail_gate_decisions": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_enabled_steps": float(\n'
        "                    effective_runtime_k > 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_disabled_steps": float(\n'
        "                    effective_runtime_k == 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_decode_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_decode_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_sum": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_count": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_count\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_sum": float(\n'
        "                    scheduler_output.tail_gate_expected_accept_length\n"
        "                ),\n"
        '                f"vllm:spec_decode_tail_gate_k_{effective_runtime_k}_steps": 1.0,\n'
        "            }\n"
        "            if scheduler_output.tail_gate_just_activated:\n"
        "                tail_gate_updates.update(\n"
        "                    {\n"
        '                        "vllm:spec_decode_tail_gate_activations": 1.0,\n'
        '                        "vllm:spec_decode_tail_gate_activation_batch_sum": float(\n'
        "                            scheduler_output.tail_gate_active_requests\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": float(\n'
        "                            scheduler_output.tail_gate_mean_sequence_length\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_tick_sum": float(\n'
        "                            scheduler_output.tail_gate_tick\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_tick_count": 1.0,\n'
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_sum": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_count": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_count\n"
        "                        ),\n"
        "                    }\n"
        "                )\n"
        "            tail_gate_state = scheduler_output.tail_gate_state.lower()\n"
        "            if tail_gate_state in (\n"
        '                "ramping_off",\n'
        '                "armed_off",\n'
        '                "on_latched",\n'
        "            ):\n"
        "                tail_gate_updates[\n"
        '                    f"vllm:spec_decode_tail_gate_{tail_gate_state}_steps"\n'
        "                ] = 1.0\n"
        "            for metric_name, metric_value in tail_gate_updates.items():\n"
        "                tail_gate_metrics[metric_name] = (\n"
        "                    tail_gate_metrics.get(metric_name, 0.0) + metric_value\n"
        "                )\n"
        "\n"
        "        if self.routed_experts_initialized:\n"
    )
    if v1_execute_new.count(tail_gate_telemetry_old) != 1:
        raise RuntimeError("Internal V1 tail-gate telemetry anchor changed.")
    v1_execute_activation_tick_legacy = v1_execute_new
    v1_execute_new = v1_execute_new.replace(
        tail_gate_telemetry_old, tail_gate_telemetry_new, 1
    )
    v1_execute_legacy = (
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        '        scheduler_output: "SchedulerOutput",\n'
        "        intermediate_tensors: IntermediateTensors | None = None,\n"
        "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:\n"
        "        if self.execute_model_state is not None:\n"
        "            raise RuntimeError(\n"
        '                "State error: sample_tokens() must be called "\n'
        '                "after execute_model() returns None."\n'
        "            )\n"
        "\n"
        "        tail_gate_mode = getattr(\n"
        '            self.speculative_config, "sd_tail_gate_mode", "off"\n'
        "        )\n"
        '        if tail_gate_mode != "off":\n'
        "            runtime_num_spec_tokens = (\n"
        "                scheduler_output.num_spec_tokens_to_schedule\n"
        "            )\n"
        "            tail_gate_metrics = getattr(\n"
        '                self, "_nrl_tail_gate_metrics", None\n'
        "            )\n"
        "            if tail_gate_metrics is None:\n"
        "                tail_gate_metrics = {}\n"
        "                self._nrl_tail_gate_metrics = tail_gate_metrics\n"
        "            effective_runtime_k = (\n"
        "                self.num_spec_tokens\n"
        "                if runtime_num_spec_tokens is None\n"
        "                else runtime_num_spec_tokens\n"
        "            )\n"
        "            tail_gate_updates = {\n"
        '                "vllm:spec_decode_tail_gate_decisions": 1.0,\n'
        '                "vllm:spec_decode_tail_gate_enabled_steps": float(\n'
        "                    effective_runtime_k > 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_disabled_steps": float(\n'
        "                    effective_runtime_k == 0\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_decode_active_requests_sum": float(\n'
        "                    scheduler_output.tail_gate_decode_active_requests\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_sum": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_predicted_speedup_count": float(\n'
        "                    scheduler_output.tail_gate_predicted_speedup_count\n"
        "                ),\n"
        '                "vllm:spec_decode_tail_gate_expected_accept_length_sum": float(\n'
        "                    scheduler_output.tail_gate_expected_accept_length\n"
        "                ),\n"
        '                f"vllm:spec_decode_tail_gate_k_{effective_runtime_k}_steps": 1.0,\n'
        "            }\n"
        "            if scheduler_output.tail_gate_just_activated:\n"
        "                tail_gate_updates.update(\n"
        "                    {\n"
        '                        "vllm:spec_decode_tail_gate_activations": 1.0,\n'
        '                        "vllm:spec_decode_tail_gate_activation_batch_sum": float(\n'
        "                            scheduler_output.tail_gate_active_requests\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_sequence_length_sum": float(\n'
        "                            scheduler_output.tail_gate_mean_sequence_length\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_sum": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_sum\n"
        "                        ),\n"
        '                        "vllm:spec_decode_tail_gate_activation_predicted_speedup_count": float(\n'
        "                            scheduler_output.tail_gate_predicted_speedup_count\n"
        "                        ),\n"
        "                    }\n"
        "                )\n"
        "            tail_gate_state = scheduler_output.tail_gate_state.lower()\n"
        "            if tail_gate_state in (\n"
        '                "ramping_off",\n'
        '                "armed_off",\n'
        '                "on_latched",\n'
        "            ):\n"
        "                tail_gate_updates[\n"
        '                    f"vllm:spec_decode_tail_gate_{tail_gate_state}_steps"\n'
        "                ] = 1.0\n"
        "            for metric_name, metric_value in tail_gate_updates.items():\n"
        "                tail_gate_metrics[metric_name] = (\n"
        "                    tail_gate_metrics.get(metric_name, 0.0) + metric_value\n"
        "                )\n"
        "\n"
        "        if self.routed_experts_initialized:\n"
    )

    patch_specs: tuple[_RuntimePatchSpec, ...] = (
        ("config/speculative.py", ((config_old, config_new, ()),)),
        (
            "v1/core/sched/output.py",
            ((scheduler_output_old, scheduler_output_new, ()),),
        ),
        (
            "v1/worker/gpu/model_runner.py",
            (
                (
                    v2_execute_old,
                    v2_execute_new,
                    (v2_execute_legacy, v2_execute_activation_tick_legacy),
                ),
                (v2_state_old, v2_state_new, ()),
                (v2_sample_state_old, v2_sample_state_new, ()),
                (v2_handler_init_old, v2_handler_init_new, ()),
                (v2_proposal_old, v2_proposal_new, ()),
                (v2_execute_state_old, v2_execute_state_new, ()),
            ),
        ),
        (
            "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
            (
                (speculator_signature_old, speculator_signature_new, ()),
                (speculator_k0_old, speculator_k0_new, ()),
            ),
        ),
        (
            "v1/worker/gpu_model_runner.py",
            (
                (
                    v1_execute_old,
                    v1_execute_new,
                    (v1_execute_legacy, v1_execute_activation_tick_legacy),
                ),
            ),
        ),
    )
    file_paths = {
        relative_path: _get_vllm_file(relative_path) for relative_path, _ in patch_specs
    }

    with _locked_file_patches(list(file_paths.values())) as (contents, write_back):
        patched_contents = {}
        changed_paths = set()
        for relative_path, replacements in patch_specs:
            file_path = file_paths[relative_path]
            content = contents[file_path]
            for old_snippet, new_snippet, legacy_snippets in replacements:
                source_snippets = (old_snippet,) + legacy_snippets
                source_counts = [content.count(snippet) for snippet in source_snippets]
                new_count = content.count(new_snippet)
                if (
                    new_count == 0
                    and source_counts.count(1) == 1
                    and max(source_counts) == 1
                ):
                    matched_snippet = source_snippets[source_counts.index(1)]
                    content = content.replace(matched_snippet, new_snippet, 1)
                    changed_paths.add(file_path)
                elif new_count == 1 and all(count == 0 for count in source_counts):
                    continue
                else:
                    raise RuntimeError(
                        "Could not apply the runtime tail-gating patch to "
                        f"{file_path}; the vLLM 0.24 source layout changed."
                    )
            patched_contents[file_path] = content

        for file_path in sorted(changed_paths):
            write_back(file_path, patched_contents[file_path])

    if changed_paths:
        logger.info("Successfully patched vLLM 0.24 runtime tail gating.")
    else:
        logger.info("vLLM 0.24 runtime tail-gating patch already applied.")


def _install_vllm_cudagraph_dispatch_metrics(dispatcher_cls: type[Any]) -> None:
    """Wrap a vLLM CUDA-graph dispatcher with cumulative coverage counters."""
    if getattr(dispatcher_cls, "_nrl_cudagraph_dispatch_metrics_installed", False):
        return

    original_dispatch = dispatcher_cls.dispatch

    @wraps(original_dispatch)
    def dispatch_with_metrics(self, *args, **kwargs):
        dispatch_result = original_dispatch(self, *args, **kwargs)
        is_v1_dispatcher = isinstance(dispatch_result, tuple)
        if is_v1_dispatcher:
            runtime_mode, batch_descriptor = dispatch_result
            positional_num_tokens = args[0] if args else None
        else:
            batch_descriptor = dispatch_result
            runtime_mode = batch_descriptor.cg_mode
            positional_num_tokens = args[1] if len(args) > 1 else None
        num_tokens = kwargs.get("num_tokens", positional_num_tokens)
        if not isinstance(num_tokens, int):
            raise RuntimeError(
                "Could not determine num_tokens while recording CUDA-graph "
                "dispatch metrics."
            )

        runtime_mode_name = getattr(runtime_mode, "name", str(runtime_mode)).lower()
        counters = getattr(self, "_nrl_cudagraph_dispatch_metrics", None)
        if counters is None:
            counters = {}
            self._nrl_cudagraph_dispatch_metrics = counters

        def increment(name: str, value: int = 1) -> None:
            counters[name] = counters.get(name, 0) + value

        increment(f"calls_{runtime_mode_name}")
        increment(f"unpadded_tokens_{runtime_mode_name}", num_tokens)
        increment(
            f"padded_tokens_{runtime_mode_name}",
            int(getattr(batch_descriptor, "num_tokens", num_tokens)),
        )

        if runtime_mode_name == "none":
            configured_mode_name = getattr(
                getattr(self, "cudagraph_mode", None),
                "name",
                str(getattr(self, "cudagraph_mode", "none")),
            ).lower()
            if is_v1_dispatcher and not getattr(self, "keys_initialized", False):
                fallback_reason = "uninitialized"
            elif configured_mode_name == "none":
                fallback_reason = "disabled"
            elif not is_v1_dispatcher and not getattr(self, "_graphs_captured", False):
                fallback_reason = "uninitialized"
            elif not is_v1_dispatcher and num_tokens <= 0:
                fallback_reason = "empty"
            else:
                max_capture_size = getattr(
                    getattr(self, "compilation_config", None),
                    "max_cudagraph_capture_size",
                    None,
                )
                if max_capture_size is None:
                    fallback_reason = "missing_capture_limit"
                elif num_tokens > max_capture_size:
                    fallback_reason = "oversize"
                elif not is_v1_dispatcher:
                    num_active_loras = kwargs.get(
                        "num_active_loras", args[3] if len(args) > 3 else 0
                    )
                    effective_loras = (
                        self._resolve_effective_loras(num_active_loras)
                        if hasattr(self, "_resolve_effective_loras")
                        else num_active_loras
                    )
                    candidate_key = (num_tokens, effective_loras)
                    fallback_reason = (
                        "incompatible"
                        if candidate_key in getattr(self, "_candidates", {})
                        else "missing_key"
                    )
                else:
                    valid_modes = kwargs.get(
                        "valid_modes", args[4] if len(args) > 4 else None
                    )
                    invalid_modes = kwargs.get(
                        "invalid_modes", args[5] if len(args) > 5 else None
                    )
                    valid_mode_names = (
                        {
                            getattr(mode, "name", str(mode)).lower()
                            for mode in valid_modes
                        }
                        if valid_modes is not None
                        else {"none", "piecewise", "full"}
                    )
                    if invalid_modes is not None:
                        valid_mode_names -= {
                            getattr(mode, "name", str(mode)).lower()
                            for mode in invalid_modes
                        }
                    fallback_reason = (
                        "mode_restricted"
                        if valid_mode_names <= {"none"}
                        else "missing_key"
                    )
            increment(f"fallback_{fallback_reason}")

        return dispatch_result

    setattr(dispatcher_cls, "dispatch", dispatch_with_metrics)
    setattr(dispatcher_cls, "_nrl_cudagraph_dispatch_metrics_installed", True)


def _patch_vllm_cudagraph_dispatch_metrics(logger) -> None:
    """Install environment-gated dispatch counters in every vLLM worker."""
    marker = "# NRL_CUDAGRAPH_DISPATCH_METRICS_PATCH"
    for relative_path, class_name in (
        ("v1/cudagraph_dispatcher.py", "CudagraphDispatcher"),
        ("v1/worker/gpu/cudagraph_utils.py", "CudaGraphManager"),
    ):
        file_to_patch = _get_vllm_file(relative_path)
        source_suffix = (
            "\n\n"
            f"{marker}\n"
            "import os as _nrl_os\n"
            "if _nrl_os.environ.get(\n"
            "    'NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS', 'false'\n"
            ").lower() == 'true':\n"
            "    from nemo_rl.models.generation.vllm.patches import (\n"
            "        _install_vllm_cudagraph_dispatch_metrics,\n"
            "    )\n"
            f"    _install_vllm_cudagraph_dispatch_metrics({class_name})\n"
        )

        with _locked_file_patch(file_to_patch) as (content, write_back):
            if marker not in content:
                write_back(content + source_suffix)
                logger.info(
                    "Installed CUDA-graph dispatch metrics source hook in %s.",
                    relative_path,
                )
            else:
                logger.info(
                    "CUDA-graph dispatch metrics source hook already installed in %s.",
                    relative_path,
                )

    if (
        os.environ.get("NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS", "false").lower()
        == "true"
    ):
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        _install_vllm_cudagraph_dispatch_metrics(CudagraphDispatcher)
        _install_vllm_cudagraph_dispatch_metrics(CudaGraphManager)


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

    if (
        os.environ.get("NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS", "false").lower()
        == "true"
    ):
        _patch_vllm_cudagraph_dispatch_metrics(patch_logger)

    if speculative_config:
        if speculative_config.get("sd_tail_gate_mode", "off") != "off":
            _patch_vllm_runtime_tail_gating(patch_logger)
        _patch_vllm_piecewise_specdec_cudagraph_alignment(patch_logger)
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
