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

# Validate the TE #2898 runtime backport in the actual Linux MCore environment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/profiles/${CLUSTER:?Set CLUSTER to ptyche or oci-hsg}.env"
PARTITION="${PARTITION_OVERRIDE:-${PARTITION}}"
unset UV_CACHE_DIR_OVERRIDE

SOURCE_SHA=$(git rev-parse --short=9 HEAD)
RUN_NAME="latestmain-nanov3-thd-cp-patch-unit-${SOURCE_SHA}"
read -r -d '' COMMAND <<'COMMAND_EOF' || true
uv run --no-sync pytest -q \
  tests/unit/models/policy/test_patches.py \
  tests/unit/models/policy/test_megatron_worker.py \
  -k 'PatchThdContextParallelCudaGraph or ThdContextParallelPatchBootstrap or WeakRefFloat64 or te_cuda_graph_capture_uses_safe_forward_pre_hook_boundary or te_cuda_graph_first_replay_emits_visible_rank_zero_event'

NRL_FORCE_REBUILD_VENVS=true uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import ray

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.utils.venvs import create_local_venv_on_each_node


ray.init(address="auto")
worker_name = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
python_path = Path(create_local_venv_on_each_node(PY_EXECUTABLES.MCORE, worker_name))
worker_venv = python_path.parent.parent
runtime_env = {
    "py_executable": str(python_path),
    "env_vars": {
        "VIRTUAL_ENV": str(worker_venv),
        "UV_PROJECT_ENVIRONMENT": str(worker_venv),
    },
}


@ray.remote(num_gpus=1)
def worker_runtime_probe() -> dict[str, object]:
    import torch

    from nemo_rl.models.policy.workers.patches import (
        apply_transformer_engine_weak_ref_float64_patch,
    )

    apply_transformer_engine_weak_ref_float64_patch(required=True)

    from transformer_engine.pytorch.utils import make_weak_ref

    original = torch.tensor(
        [1.25, -2.5],
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.float64,
    )
    weak_reference = make_weak_ref(original)
    return {
        "dtype": str(weak_reference.dtype),
        "shape": tuple(weak_reference.shape),
        "same_storage": weak_reference.data_ptr() == original.data_ptr(),
        "equal": torch.equal(weak_reference, original),
    }


result = ray.get(worker_runtime_probe.options(runtime_env=runtime_env).remote())
assert result == {
    "dtype": "torch.float64",
    "shape": (2,),
    "same_storage": True,
    "equal": True,
}, result
print(f"mcore_venv_python={python_path}")
print("te_float64_weak_ref_cuda_smoke=passed")
PY
COMMAND_EOF

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
  --segment=1
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
