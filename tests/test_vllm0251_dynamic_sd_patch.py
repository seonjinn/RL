import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "vllm_0251_eagle3_perfcfg"
    / "apply_vllm0251_dynamic_sd_cg_fix.py"
)


def _write_model_runner(path: Path) -> str:
    source = (
        "        # For transferring state from execute_model to subsequent sample_tokens call.\n"
        "        self.execute_model_state: ExecuteModelState | None = None\n"
        "\n"
        "        self.execute_model_state = ExecuteModelState(\n"
        "            input_batch=input_batch,\n"
        "            attn_metadata=attn_metadata,\n"
        "            slot_mappings_by_layer=slot_mappings_by_layer,\n"
        "            hidden_states=hidden_states,\n"
        "            aux_hidden_states=aux_hidden_states,\n"
        "            finished_req_ids=finished_req_ids,\n"
        "        )\n"
        "\n"
        "        input_batch = self.execute_model_state.input_batch\n"
        "        attn_metadata = self.execute_model_state.attn_metadata\n"
        "        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer\n"
        "        hidden_states = self.execute_model_state.hidden_states\n"
        "        aux_hidden_states = self.execute_model_state.aux_hidden_states\n"
        "        finished_req_ids = self.execute_model_state.finished_req_ids\n"
        "        self.execute_model_state = None\n"
        "\n"
        "        if self.speculator is not None:\n"
        "            assert self.sampler is not None\n"
        "            # Let the target override the hidden state fed to the drafter\n"
        "            draft_tokens = self.speculator.propose(\n"
        "                input_batch,\n"
        "                attn_metadata,\n"
        "                mm_inputs=mm_inputs,\n"
        "            )\n"
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
        "\n"
        "        if self.num_speculative_steps > 0:\n"
        "            self.draft_tokens_handler.set_draft_tokens(\n"
        "                input_batch,\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
        "            )\n"
        "\n"
        "class ExecuteModelState(NamedTuple):\n"
        "    input_batch: InputBatch\n"
        "    attn_metadata: dict[str, Any] | None\n"
        "    slot_mappings_by_layer: dict[str, torch.Tensor] | None\n"
        "    hidden_states: torch.Tensor | None\n"
        "    aux_hidden_states: list[torch.Tensor] | None\n"
        "    finished_req_ids: set[str]\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return source


def test_patch_updates_vllm0251_and_is_idempotent(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    cudagraph_utils = site_packages / "vllm/v1/worker/gpu/cudagraph_utils.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    cudagraph_utils.parent.mkdir(parents=True)
    speculator.parent.mkdir(parents=True)
    cudagraph_utils.write_text(
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "    ):\n"
        "        self.decode_query_len = decode_query_len\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n"
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "        ):\n",
        encoding="utf-8",
    )
    speculator.write_text(
        "            cudagraph_mode,\n            decode_query_len=1,\n        )\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(PATCH_SCRIPT),
        "--site-packages",
        str(site_packages),
    ]
    first = subprocess.run(command, check=True, text=True, capture_output=True)
    second = subprocess.run(command, check=True, text=True, capture_output=True)

    assert "applied" in first.stdout
    assert "already applied" in second.stdout
    assert "use_dynamic_decode_shapes: bool = True" in cudagraph_utils.read_text()
    assert "and self.use_dynamic_decode_shapes" in cudagraph_utils.read_text()
    assert "use_dynamic_decode_shapes=False" in speculator.read_text()


def test_default_patch_leaves_model_runner_byte_for_byte_unchanged(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    cudagraph_utils = site_packages / "vllm/v1/worker/gpu/cudagraph_utils.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    model_runner = site_packages / "vllm/v1/worker/gpu/model_runner.py"
    cudagraph_utils.parent.mkdir(parents=True)
    speculator.parent.mkdir(parents=True)
    cudagraph_utils.write_text(
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "    ):\n"
        "        self.decode_query_len = decode_query_len\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n"
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "        ):\n",
        encoding="utf-8",
    )
    speculator.write_text(
        "            cudagraph_mode,\n            decode_query_len=1,\n        )\n",
        encoding="utf-8",
    )
    original = _write_model_runner(model_runner)

    subprocess.run(
        [sys.executable, str(PATCH_SCRIPT), "--site-packages", str(site_packages)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert model_runner.read_text(encoding="utf-8") == original


def test_smoke_telemetry_patch_records_selected_and_actual_draft_width(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    cudagraph_utils = site_packages / "vllm/v1/worker/gpu/cudagraph_utils.py"
    speculator = (
        site_packages / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    model_runner = site_packages / "vllm/v1/worker/gpu/model_runner.py"
    cudagraph_utils.parent.mkdir(parents=True)
    speculator.parent.mkdir(parents=True)
    cudagraph_utils.write_text(
        "class CudaGraphManager:\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: VllmConfig,\n"
        "        device: torch.device,\n"
        "        cudagraph_mode: CUDAGraphMode,\n"
        "        decode_query_len: int,\n"
        "        lora_capture_cases: list[int] | None = None,\n"
        "    ):\n"
        "        self.decode_query_len = decode_query_len\n\n"
        "        self.dp_size = vllm_config.parallel_config.data_parallel_size\n"
        "            and speculative_config.uses_dynamic_speculative_decoding()\n"
        "        ):\n",
        encoding="utf-8",
    )
    speculator.write_text(
        "            cudagraph_mode,\n"
        "            decode_query_len=1,\n"
        "        )\n"
        "\n"
        "    @torch.inference_mode()\n"
        "    def propose(\n"
        "        self,\n"
        "        input_batch: InputBatch,\n"
        "        is_profile: bool = False,\n"
        "    ) -> torch.Tensor:\n"
        "        num_tokens = input_batch.num_tokens_after_padding\n"
        "        num_reqs = input_batch.num_reqs\n"
        "        max_query_len = input_batch.num_scheduled_tokens.max()\n"
        "        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()\n"
        "        self.draft_max_seq_len = min(\n"
        "            max_seq_len + self.num_speculative_steps, self.max_model_len\n"
        "        )\n"
        "        if self.num_speculative_steps == 1:\n"
        "            # Early exit.\n"
        "            return self.draft_tokens[:num_reqs, :1]\n"
        "        self._multi_step_decode(\n"
        "            num_reqs,\n"
        "            False,\n"
        "            decode_batch_desc,\n"
        "            num_tokens_across_dp,\n"
        "        )\n"
        "        return self.draft_tokens[:num_reqs]\n"
        "\n"
        "    def _multi_step_decode(\n"
        "        self,\n"
        "        num_reqs: int,\n"
        "        skip_attn: bool,\n"
        "        batch_desc: BatchExecutionDescriptor,\n"
        "        num_tokens_across_dp: torch.Tensor | None,\n"
        "    ) -> None:\n"
        "        for step in range(1, self.num_speculative_steps):\n"
        "            self.current_draft_step.fill_(step)\n",
        encoding="utf-8",
    )
    _write_model_runner(model_runner)
    environment = {
        **os.environ,
        "NRL_VLLM_DYNAMIC_SD_SMOKE_TELEMETRY": "1",
    }
    command = [
        sys.executable,
        str(PATCH_SCRIPT),
        "--site-packages",
        str(site_packages),
    ]

    first = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment
    )
    second = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment
    )
    patched = model_runner.read_text(encoding="utf-8")

    assert "smoke telemetry: applied" in first.stdout
    assert "smoke telemetry: already applied" in second.stdout
    assert "scheduler_batch_size: int | None" in patched
    assert "requested_draft_width: int | None" in patched
    assert "actual_draft_width = draft_tokens.shape[1]" in patched
    assert "DYNAMIC_SD_SMOKE_TELEMETRY" in patched
    assert "if not dummy_run and not is_profile" in patched
    assert "num_speculative_steps=requested_draft_width" in patched
    assert "input_batch.idx_mapping, :actual_draft_width" in patched
    assert "] = draft_tokens" in patched
    assert "draft_token_state.copy_" not in patched

    patched_speculator = speculator.read_text(encoding="utf-8")
    assert "num_speculative_steps: int | None = None" in patched_speculator
    assert "selected_num_speculative_steps == 0" in patched_speculator
    assert "range(1, selected_num_speculative_steps)" in patched_speculator
    assert (
        "self.draft_tokens[:num_reqs, :selected_num_speculative_steps]"
        in patched_speculator
    )
