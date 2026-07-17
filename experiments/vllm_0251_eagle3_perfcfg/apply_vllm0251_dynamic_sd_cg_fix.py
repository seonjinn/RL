#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import site
from pathlib import Path


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected vLLM 0.25.1 source for {label}")
    return text.replace(old, new), True


def _find_site_packages() -> Path:
    for candidate in map(Path, site.getsitepackages()):
        if (candidate / "vllm").is_dir():
            return candidate
    raise RuntimeError("Could not find the installed vLLM package")


def apply_patch(site_packages: Path) -> bool:
    cudagraph_utils = site_packages / "vllm/v1/worker/gpu/cudagraph_utils.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    if not cudagraph_utils.is_file() or not speculator.is_file():
        raise FileNotFoundError(f"vLLM 0.25.1 sources not found under {site_packages}")

    cudagraph_text = cudagraph_utils.read_text(encoding="utf-8")
    cudagraph_text, signature_changed = _replace_once(
        cudagraph_text,
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "    ):\n",
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "        use_dynamic_decode_shapes: bool = True,\n"
        "    ):\n",
        "CudaGraphManager signature",
    )
    cudagraph_text, attribute_changed = _replace_once(
        cudagraph_text,
        "        self.decode_query_len = decode_query_len\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n",
        "        self.decode_query_len = decode_query_len\n"
        "        self.use_dynamic_decode_shapes = use_dynamic_decode_shapes\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n",
        "CudaGraphManager attribute",
    )
    cudagraph_text, condition_changed = _replace_once(
        cudagraph_text,
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "        ):\n",
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "            and self.use_dynamic_decode_shapes\n"
        "        ):\n",
        "DynamicSD graph condition",
    )

    speculator_text = speculator.read_text(encoding="utf-8")
    speculator_text, speculator_changed = _replace_once(
        speculator_text,
        "            cudagraph_mode,\n            decode_query_len=1,\n        )\n",
        "            cudagraph_mode,\n"
        "            decode_query_len=1,\n"
        "            use_dynamic_decode_shapes=False,\n"
        "        )\n",
        "autoregressive draft decode manager",
    )

    changed = any(
        (signature_changed, attribute_changed, condition_changed, speculator_changed)
    )
    if changed:
        cudagraph_utils.write_text(cudagraph_text, encoding="utf-8")
        speculator.write_text(speculator_text, encoding="utf-8")
    return changed


def apply_smoke_telemetry_patch(site_packages: Path) -> bool:
    """Apply run-scoped variable-width drafting and its smoke telemetry."""
    model_runner = site_packages / "vllm/v1/worker/gpu/model_runner.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    if not model_runner.is_file() or not speculator.is_file():
        raise FileNotFoundError(
            f"vLLM 0.25.1 DynamicSD sources not found under {site_packages}"
        )

    speculator_text = speculator.read_text(encoding="utf-8")
    speculator_text, signature_changed = _replace_once(
        speculator_text,
        "        is_profile: bool = False,\n"
        "    ) -> torch.Tensor:\n",
        "        is_profile: bool = False,\n"
        "        num_speculative_steps: int | None = None,\n"
        "    ) -> torch.Tensor:\n",
        "DynamicSD selected-K proposal signature",
    )
    speculator_text, selection_changed = _replace_once(
        speculator_text,
        "        num_reqs = input_batch.num_reqs\n"
        "        max_query_len = input_batch.num_scheduled_tokens.max()\n",
        "        num_reqs = input_batch.num_reqs\n"
        "        selected_num_speculative_steps = (\n"
        "            self.num_speculative_steps\n"
        "            if num_speculative_steps is None\n"
        "            else num_speculative_steps\n"
        "        )\n"
        "        if not 0 <= selected_num_speculative_steps <= self.num_speculative_steps:\n"
        "            raise ValueError(\n"
        '                "DynamicSD selected K must be between 0 and the configured "\n'
        "                f\"maximum {self.num_speculative_steps}, got \"\n"
        "                f\"{selected_num_speculative_steps}\"\n"
        "            )\n"
        "        if selected_num_speculative_steps == 0:\n"
        "            return self.draft_tokens[:num_reqs, :0]\n"
        "        max_query_len = input_batch.num_scheduled_tokens.max()\n",
        "DynamicSD selected-K validation",
    )
    speculator_text, sequence_length_changed = _replace_once(
        speculator_text,
        "            max_seq_len + self.num_speculative_steps, self.max_model_len\n",
        "            max_seq_len + selected_num_speculative_steps, self.max_model_len\n",
        "DynamicSD selected-K sequence length",
    )
    speculator_text, early_exit_changed = _replace_once(
        speculator_text,
        "        if self.num_speculative_steps == 1:\n"
        "            # Early exit.\n"
        "            return self.draft_tokens[:num_reqs, :1]\n",
        "        if selected_num_speculative_steps == 1:\n"
        "            # Early exit.\n"
        "            return self.draft_tokens[:num_reqs, :1]\n",
        "DynamicSD selected-K early exit",
    )
    speculator_text, call_changed = _replace_once(
        speculator_text,
        "            num_tokens_across_dp,\n"
        "        )\n",
        "            num_tokens_across_dp,\n"
        "            selected_num_speculative_steps,\n"
        "        )\n",
        "DynamicSD selected-K decode call",
    )
    speculator_text, return_changed = _replace_once(
        speculator_text,
        "        return self.draft_tokens[:num_reqs]\n",
        "        return self.draft_tokens[:num_reqs, :selected_num_speculative_steps]\n",
        "DynamicSD selected-K return width",
    )
    speculator_text, decode_signature_changed = _replace_once(
        speculator_text,
        "        num_tokens_across_dp: torch.Tensor | None,\n"
        "    ) -> None:\n",
        "        num_tokens_across_dp: torch.Tensor | None,\n"
        "        selected_num_speculative_steps: int,\n"
        "    ) -> None:\n",
        "DynamicSD selected-K decode signature",
    )
    speculator_text, loop_changed = _replace_once(
        speculator_text,
        "        for step in range(1, self.num_speculative_steps):\n",
        "        for step in range(1, selected_num_speculative_steps):\n",
        "DynamicSD selected-K decode loop",
    )

    text = model_runner.read_text(encoding="utf-8")
    text, state_changed = _replace_once(
        text,
        "        # For transferring state from execute_model to subsequent sample_tokens call.\n"
        "        self.execute_model_state: ExecuteModelState | None = None\n",
        "        # For transferring state from execute_model to subsequent sample_tokens call.\n"
        "        self.execute_model_state: ExecuteModelState | None = None\n"
        "        self.dynamic_sd_smoke_telemetry_seen: set[tuple[int, int, int]] = set()\n",
        "DynamicSD smoke telemetry state",
    )
    text, creation_changed = _replace_once(
        text,
        "            aux_hidden_states=aux_hidden_states,\n"
        "            finished_req_ids=finished_req_ids,\n"
        "        )\n",
        "            aux_hidden_states=aux_hidden_states,\n"
        "            finished_req_ids=finished_req_ids,\n"
        "            scheduler_batch_size=(\n"
        "                len(scheduler_output.num_scheduled_tokens)\n"
        "                if not dummy_run and not is_profile\n"
        "                else None\n"
        "            ),\n"
        "            requested_draft_width=(\n"
        "                scheduler_output.num_spec_tokens_to_schedule\n"
        "                if not dummy_run and not is_profile\n"
        "                else None\n"
        "            ),\n"
        "        )\n",
        "DynamicSD smoke telemetry state creation",
    )
    text, extraction_changed = _replace_once(
        text,
        "        aux_hidden_states = self.execute_model_state.aux_hidden_states\n"
        "        finished_req_ids = self.execute_model_state.finished_req_ids\n"
        "        self.execute_model_state = None\n",
        "        aux_hidden_states = self.execute_model_state.aux_hidden_states\n"
        "        finished_req_ids = self.execute_model_state.finished_req_ids\n"
        "        scheduler_batch_size = self.execute_model_state.scheduler_batch_size\n"
        "        requested_draft_width = self.execute_model_state.requested_draft_width\n"
        "        self.execute_model_state = None\n",
        "DynamicSD smoke telemetry state extraction",
    )
    text, default_width_changed = _replace_once(
        text,
        "        if self.speculator is not None:\n"
        "            assert self.sampler is not None\n"
        "            # Let the target override the hidden state fed to the drafter\n",
        "        actual_draft_width = self.num_speculative_steps\n"
        "        if self.speculator is not None:\n"
        "            assert self.sampler is not None\n"
        "            # Let the target override the hidden state fed to the drafter\n",
        "DynamicSD default draft width",
    )
    text, proposal_changed = _replace_once(
        text,
        "                mm_inputs=mm_inputs,\n"
        "            )\n",
        "                mm_inputs=mm_inputs,\n"
        "                num_speculative_steps=requested_draft_width,\n"
        "            )\n",
        "DynamicSD selected-K proposal",
    )
    text, logging_changed = _replace_once(
        text,
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
        "            actual_draft_width = draft_tokens.shape[1]\n"
        "            if (\n"
        "                scheduler_batch_size is not None\n"
        "                and requested_draft_width is not None\n"
        "            ):\n"
        "                telemetry_key = (\n"
        "                    scheduler_batch_size,\n"
        "                    requested_draft_width,\n"
        "                    actual_draft_width,\n"
        "                )\n"
        "                if telemetry_key not in self.dynamic_sd_smoke_telemetry_seen:\n"
        "                    logger.info(\n"
        '                        "DYNAMIC_SD_SMOKE_TELEMETRY batch_size=%d "\n'
        '                        "selected_k=%d requested_draft_width=%d "\n'
        '                        "actual_draft_width=%d",\n'
        "                        scheduler_batch_size,\n"
        "                        requested_draft_width,\n"
        "                        requested_draft_width,\n"
        "                        actual_draft_width,\n"
        "                    )\n"
        "                    self.dynamic_sd_smoke_telemetry_seen.add(telemetry_key)\n"
        "            self.req_states.draft_tokens[\n"
        "                input_batch.idx_mapping, :actual_draft_width\n"
        "            ] = draft_tokens\n",
        "DynamicSD smoke telemetry logging",
    )
    text, scheduler_width_changed = _replace_once(
        text,
        "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
        "            )\n",
        "                self.req_states.draft_tokens[\n"
        "                    input_batch.idx_mapping, :actual_draft_width\n"
        "                ],\n"
        "            )\n",
        "DynamicSD scheduler draft width",
    )
    text, tuple_changed = _replace_once(
        text,
        "    aux_hidden_states: list[torch.Tensor] | None\n"
        "    finished_req_ids: set[str]\n",
        "    aux_hidden_states: list[torch.Tensor] | None\n"
        "    finished_req_ids: set[str]\n"
        "    scheduler_batch_size: int | None\n"
        "    requested_draft_width: int | None\n",
        "DynamicSD smoke telemetry tuple",
    )
    changed = any(
        (
            signature_changed,
            selection_changed,
            sequence_length_changed,
            early_exit_changed,
            call_changed,
            return_changed,
            decode_signature_changed,
            loop_changed,
            state_changed,
            creation_changed,
            extraction_changed,
            default_width_changed,
            proposal_changed,
            logging_changed,
            scheduler_width_changed,
            tuple_changed,
        )
    )
    if changed:
        speculator.write_text(speculator_text, encoding="utf-8")
        model_runner.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path)
    args = parser.parse_args()
    site_packages = args.site_packages or _find_site_packages()
    changed = apply_patch(site_packages)
    state = "applied" if changed else "already applied"
    print(f"vLLM 0.25.1 DynamicSD CUDA graph fix: {state}")
    if os.environ.get("NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY") == "1":
        telemetry_changed = apply_smoke_telemetry_patch(site_packages)
        telemetry_state = "applied" if telemetry_changed else "already applied"
        print(f"vLLM 0.25.1 DynamicSD smoke telemetry: {telemetry_state}")


if __name__ == "__main__":
    main()
