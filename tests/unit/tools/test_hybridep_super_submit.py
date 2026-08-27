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


def test_ray_setup_failure_stops_head_and_worker_startup() -> None:
    script = RAY_SUB.read_text()

    assert script.count('if ! bash "$SETUP_COMMAND_FILE"; then') == 2
    assert script.count("Setup command failed; refusing to start Ray") == 2
