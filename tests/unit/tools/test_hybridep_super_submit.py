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


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "cw-dfw"
    / "hybridep"
    / "submit_super_5515_ab.sh"
)
RAY_SUB = REPO_ROOT / "ray.sub"


def test_super_setup_avoids_pipefail_sigpipe_and_verifies_deepep() -> None:
    script = SUBMIT_SCRIPT.read_text()

    assert "gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)" in script
    assert 'grep -q H100 <<< "${gpu_names}"' in script
    assert (
        'nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100'
        not in script
    )
    assert "DEEPEP_SETUP_VERSION" in script
    assert "import hybrid_ep_cpp" in script
    assert script.count("import torch; import hybrid_ep_cpp") == 2


def test_super_setup_preserves_node_local_paths_through_pyxis() -> None:
    submit_script = SUBMIT_SCRIPT.read_text()
    ray_sub = RAY_SUB.read_text()

    assert "export CONTAINER_ENV_VARS=" in submit_script
    for name in (
        "NRL_NODE_LOCAL_UV_CACHE_DIR",
        "NEMO_RL_VENV_DIR",
        "DEEPEP_OVERLAY_DIR",
    ):
        assert name in submit_script.split("export CONTAINER_ENV_VARS=", 1)[1]
    assert '--container-env=${CONTAINER_ENV_VARS}' in ray_sub


def test_super_setup_uses_the_container_python_for_wheel_install() -> None:
    script = SUBMIT_SCRIPT.read_text()

    assert "unset CONDA_PREFIX VIRTUAL_ENV" in script
    assert "python_bin=$(command -v python)" in script
    assert 'uv pip install --no-config --python "${python_bin}"' in script


def test_super_setup_reports_failed_preflight_values() -> None:
    script = SUBMIT_SCRIPT.read_text()

    assert "Expected 8 visible H100 GPUs" in script
    assert "Expected node-local setup path under /raid/scratch" in script
    assert "DeepEP wheel is not readable inside the container" in script
    assert "DEEPEP_SETUP_INPUTS" in script


def test_super_setup_smoke_uses_one_node_and_skips_training() -> None:
    script = SUBMIT_SCRIPT.read_text()

    assert "SETUP_SMOKE_ONLY" in script
    assert "NUM_ACTOR_NODES=1" in script
    assert "TIME_LIMIT=00:20:00" in script
    assert 'COMMAND="${version_check}"' in script
    assert "import hybrid_ep_cpp" in script.split("printf -v version_check", 1)[1]


def test_ray_setup_failure_stops_head_and_worker_startup() -> None:
    script = RAY_SUB.read_text()

    assert script.count('if ! bash "$SETUP_COMMAND_FILE"; then') == 2
    assert script.count("Setup command failed; refusing to start Ray") == 2
