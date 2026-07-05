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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


PATCHES_PATH = Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/patches.py"
PATCHES_SPEC = importlib.util.spec_from_file_location(
    "nemo_rl_vllm_patches_test_target", PATCHES_PATH
)
assert PATCHES_SPEC is not None and PATCHES_SPEC.loader is not None
patches = importlib.util.module_from_spec(PATCHES_SPEC)
PATCHES_SPEC.loader.exec_module(patches)


VLLM_020_DRAFT_INIT = """\
        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer | DFlashProposer | ExtractHiddenStatesProposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)
"""


@pytest.fixture
def draft_patch_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    vllm_module = ModuleType("vllm")
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = Mock(return_value=Mock())
    vllm_module.logger = logger_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    monkeypatch.setattr(patches, "_patch_vllm_init_workers_ray", Mock())
    monkeypatch.setattr(patches, "_patch_vllm_llama_eagle3_own_lm_head", Mock())
    monkeypatch.setattr(patches, "_patch_vllm_hermes_tool_parser_thread_safety", Mock())
    draft_patch = Mock()
    monkeypatch.setattr(patches, "_patch_vllm_draft_model_cudagraph_init", draft_patch)
    return draft_patch


@pytest.mark.parametrize(
    ("speculative_config", "enforce_eager", "expected_calls"),
    [
        ({"method": "draft_model"}, False, 1),
        ({"method": "pard2"}, False, 1),
        ({"model": "draft-checkpoint"}, False, 1),
        ({"method": "draft_model"}, True, 0),
        ({"method": "eagle3"}, False, 0),
        ({"method": "mtp"}, False, 0),
        ({"method": "suffix"}, False, 0),
        (None, False, 0),
    ],
)
def test_apply_vllm_patches_gates_generic_draft_cudagraph_patch(
    draft_patch_mock: Mock,
    speculative_config: dict[str, str] | None,
    enforce_eager: bool,
    expected_calls: int,
) -> None:
    patches._apply_vllm_patches(
        sys.executable,
        speculative_config=speculative_config,
        enforce_eager=enforce_eager,
    )

    assert draft_patch_mock.call_count == expected_calls


def test_draft_cudagraph_patch_updates_vllm_source_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "gpu_model_runner.py"
    source_path.write_text(VLLM_020_DRAFT_INIT)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(source_path))
    patch_logger = Mock()

    patches._patch_vllm_draft_model_cudagraph_init(patch_logger)
    patched_source = source_path.read_text()
    patches._patch_vllm_draft_model_cudagraph_init(patch_logger)

    assert source_path.read_text() == patched_source
    assert "NRL_DRAFT_MODEL_CUDAGRAPH_INIT_PATCH" in patched_source
    assert "self.speculative_config.uses_draft_model()" in patched_source


def test_worker_forwards_draft_config_to_patch_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.generation.vllm import vllm_worker

    apply_patches = Mock()
    monkeypatch.setattr(vllm_worker, "_apply_vllm_patches", apply_patches)
    worker = object.__new__(vllm_worker.BaseVllmGenerationWorker)
    speculative_config = {"method": "draft_model", "model": "draft-checkpoint"}
    config = {
        "model_name": "target-checkpoint",
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "precision": "bfloat16",
            "enforce_eager": False,
        },
        "vllm_kwargs": {"speculative_config": speculative_config},
    }

    worker._init_config(
        config,
        bundle_indices=None,
        fraction_of_gpus=1.0,
        seed=None,
        extra_env_vars=None,
    )

    apply_patches.assert_called_once_with(
        sys.executable,
        extra_env_vars=None,
        speculative_config=speculative_config,
        enforce_eager=False,
    )
