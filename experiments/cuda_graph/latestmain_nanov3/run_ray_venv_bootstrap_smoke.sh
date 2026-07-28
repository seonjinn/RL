#!/usr/bin/env bash
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

# Validate the actual Ray/MCore worker bootstrap before submitting model jobs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/profiles/${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}.env"

RUN_NAME=latestmain-nanov3-ray-venv-bootstrap-smoke
COMMAND='uv sync --frozen && NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore --frozen python - <<'"'"'PY'"'"'
import os
import sys
from pathlib import Path

import ray

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.utils.venvs import create_local_venv_on_each_node


venv_root = Path(os.environ["NEMO_RL_VENV_DIR"])
assert venv_root.name == "venvs", venv_root
assert venv_root.parent.name.endswith("-logs"), venv_root

worker_name = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
python_path = Path(create_local_venv_on_each_node(PY_EXECUTABLES.MCORE, worker_name))
assert python_path.is_file(), python_path
assert python_path.is_relative_to(venv_root), (python_path, venv_root)

worker_venv = python_path.parent.parent
runtime_env = {
    "py_executable": str(python_path),
    "env_vars": {
        "VIRTUAL_ENV": str(worker_venv),
        "UV_PROJECT_ENVIRONMENT": str(worker_venv),
    },
}


@ray.remote
def worker_runtime_probe() -> dict[str, str]:
    import mamba_ssm
    import megatron.core
    import transformer_engine.pytorch

    return {
        "python": sys.executable,
        "virtual_env": os.environ["VIRTUAL_ENV"],
        "mamba_ssm": str(Path(mamba_ssm.__file__).resolve()),
        "megatron_core": str(Path(megatron.core.__file__).resolve()),
        "transformer_engine": str(Path(transformer_engine.pytorch.__file__).resolve()),
    }


result = ray.get(worker_runtime_probe.options(runtime_env=runtime_env).remote())
assert Path(result["python"]) == python_path, result
assert Path(result["virtual_env"]) == worker_venv, result
assert all(result[module] for module in ("mamba_ssm", "megatron_core", "transformer_engine")), result
print(f"ray_venv_root={venv_root}")
print(f"ray_venv_python={python_path}")
print("ray_venv_bootstrap_smoke=passed")
PY'

if [[ -z "${CONTAINER:-}" ]]; then
  echo "CONTAINER must not be blank" >&2
  exit 2
fi

SBATCH_CMD=(sbatch)
if [[ "${TEST_ONLY:-0}" == "1" ]]; then
  SBATCH_CMD+=(--test-only)
fi
SBATCH_CMD+=(
  --nodes=1
  "${SBATCH_GPU_ARGS[@]+${SBATCH_GPU_ARGS[@]}}"
  "--account=${ACCOUNT}"
  "--job-name=${ACCOUNT}-sna.${RUN_NAME}"
  "--partition=${PARTITION}"
  --time=01:00:00
  "--segment=${SEGMENT_SIZE}"
  ray.sub
)

printf 'COMMAND:\n%s\n' "${COMMAND}"
printf 'SBATCH:'
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'

COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
MOUNTS="${MOUNTS}" \
BASE_LOG_DIR="exp_logs/${RUN_NAME}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
"${SBATCH_CMD[@]}"
