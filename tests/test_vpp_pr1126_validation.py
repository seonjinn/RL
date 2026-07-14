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

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
VALIDATION_JOB = PROJECT_ROOT / "scripts/validate_nemo_rl_vpp.sbatch"


def test_vpp_validation_uses_locked_mcore_runtime_and_exact_source() -> None:
    source = VALIDATION_JOB.read_text()

    assert '"${UV_BIN}" sync --locked --extra mcore --group test' in source
    assert '"${UV_BIN}" lock --check' in source
    assert "UV_NO_EDITABLE=1" in source
    assert 'git -C "${repo_root}" status --porcelain' in source
    assert "submodule status --recursive" in source
    assert "rev-parse '@{upstream}'" in source
    assert '"${source_sha}" != "${VPP_VALIDATION_GIT_SHA}"' in source
    assert '"${actual_image_sha256}" != "${VPP_VALIDATION_IMAGE_SHA256}"' in source


def test_vpp_validation_runs_only_pr1126_gpu_cases_without_soft_skips() -> None:
    source = VALIDATION_JOB.read_text()
    expected_node_ids = (
        "test_megatron_policy_training[2gpu_pp2_vpp2_llama]",
        "test_megatron_policy_logprobs[2gpu_pp2_vpp2_llama]",
        "test_megatron_checkpoint_save_kill_and_restore[2gpu_pp2_vpp2_save_restore]",
    )

    for node_id in expected_node_ids:
        assert node_id in source
    assert "--mcore-only" in source
    assert "--hf-gated" in source
    assert "--runxfail" in source
    assert "--maxfail" not in source
    assert "--no-header" not in source
    assert "|| true" not in source
    assert "pytest.skip" not in source


def test_vpp_validation_is_one_node_and_preserves_cluster_specific_gpu_request() -> (
    None
):
    source = VALIDATION_JOB.read_text()

    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --exclusive" in source
    assert "#SBATCH --time=01:00:00" in source
    assert 'if [[ -n "${VPP_VALIDATION_GRES:-}" ]]; then' in source
    assert 'container_run+=("--gres=${VPP_VALIDATION_GRES}")' in source
    assert "--container-mounts=" in source
    assert "HF_HOME=/hf_home" in source
    assert "NEMO_RL_VENV_DIR=/runtime/worker_venvs" in source
    assert "RAY_TMPDIR=/runtime/ray" in source
    assert "RAY_USAGE_STATS_ENABLED=0" in source
    assert "TORCH_CUDA_ARCH_LIST=10.0" in source
    assert "NRL_IGNORE_VERSION_MISMATCH" not in source


def test_vpp_fixture_does_not_require_a_gated_tokenizer() -> None:
    source = (PROJECT_ROOT / "tests/unit/conftest.py").read_text()
    fixture = source.split("def tiny_llama_4layer_model_path", 1)[1].split(
        "@pytest.fixture", 1
    )[0]

    assert 'AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")' in fixture
    assert "meta-llama" not in fixture
