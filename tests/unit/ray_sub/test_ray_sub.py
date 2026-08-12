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

import re
import subprocess
from pathlib import Path


RAY_SUB = Path(__file__).parents[3] / "ray.sub"


def test_ray_sub_routes_cli_commands_through_configurable_executable() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert "RAY_CLI=${RAY_CLI:-ray}" in source
    assert not re.search(r"^\s*ray (?:start|status|stop)\b", source, re.MULTILINE)
    assert r"\$(ray status" not in source
    assert source.count('"${RAY_CLI}"') == 6


def test_dedicated_head_excludes_head_from_model_resources() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert "DEDICATED_RAY_HEAD=${DEDICATED_RAY_HEAD:-false}" in source
    assert "MODEL_NUM_NODES=${MODEL_NUM_NODES:-}" in source
    assert "expected_allocated_nodes=$((MODEL_NUM_NODES + 1))" in source
    assert "NUM_ACTORS=$((GPUS_PER_NODE * MODEL_NUM_NODES))" in source
    assert 'HEAD_RAY_GPU_ARG="--num-gpus=0"' in source
    assert 'HEAD_CUDA_VISIBILITY_COMMAND=\'export CUDA_VISIBLE_DEVICES=""\'' in source
    assert "RAY_RESOURCES=" in source


def test_dedicated_head_resources_survive_nested_shell_parsing() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")
    assignment = next(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("RAY_RESOURCES=")
        and "slurm_managed_ray_cluster" in line
        and "worker_units" not in line
    )
    script = f"""{assignment}
cat <<EOF | bash
python3 -c 'import json, sys; json.loads(sys.argv[1])' "$RAY_RESOURCES"
EOF
"""

    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dedicated_head_driver_has_no_visible_cuda_devices() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "bash \"$DRIVER_COMMAND_FILE\"" in source


def test_default_topology_still_counts_every_allocated_node() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert "NUM_ACTORS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))" in source
