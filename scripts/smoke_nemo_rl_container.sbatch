#!/bin/bash
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

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --job-name=smoke_nemo_rl_container
#SBATCH --output=smoke_nemo_rl_container_%j.log

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to an immutable squashfs path}"
: "${CONTAINER_IMAGE_SHA256:?Set CONTAINER_IMAGE_SHA256 to the expected image digest}"
: "${CONTAINER_SMOKE_DIR:?Set CONTAINER_SMOKE_DIR to an absolute host result directory}"
: "${SLURM_SUBMIT_DIR:?Submit this job from the NeMo-RL checkout root}"
if [[ "${CONTAINER_SMOKE_DIR}" != /* ]]; then
    echo "[ERROR] CONTAINER_SMOKE_DIR must be an absolute path: ${CONTAINER_SMOKE_DIR}" >&2
    exit 1
fi
mkdir -p "${CONTAINER_SMOKE_DIR}"

repo_root=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
readonly repo_root
source_sha=$(git -C "${repo_root}" rev-parse HEAD)
readonly source_sha
actual_image_sha256=$(sha256sum "${CONTAINER_IMAGE}" | awk '{print $1}')
readonly actual_image_sha256
if [[ "${actual_image_sha256}" != "${CONTAINER_IMAGE_SHA256}" ]]; then
    echo "[ERROR] image SHA256 mismatch: ${actual_image_sha256}" >&2
    exit 1
fi
printf '%s\n' "${source_sha}" > "${CONTAINER_SMOKE_DIR}/source.sha"
printf '%s  %s\n' "${actual_image_sha256}" "${CONTAINER_IMAGE}" \
    > "${CONTAINER_SMOKE_DIR}/image.sha256"
mkdir -p "${CONTAINER_SMOKE_DIR}/runtime"

container_run=(
    srun
    --nodes=1
    --ntasks=1
    --container-image="${CONTAINER_IMAGE}"
    --container-mounts="${repo_root}:/workspace/nemo-rl,${CONTAINER_SMOKE_DIR}:/results,${CONTAINER_SMOKE_DIR}/runtime:/runtime"
    --container-workdir=/workspace/nemo-rl
)
if [[ -n "${CONTAINER_SMOKE_GRES:-}" ]]; then
    container_run+=("--gres=${CONTAINER_SMOKE_GRES}")
fi
readonly -a container_run

"${container_run[@]}" bash -s 2>&1 <<'BASH' | tee "${CONTAINER_SMOKE_DIR}/smoke.log"
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="/runtime/venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export UV_NO_EDITABLE=1
export CUDA_HOME="/usr/local/cuda"
export NVTE_CUDA_ARCHS="100"
readonly UV_BIN="/root/.local/bin/uv"
[[ -x "${UV_BIN}" ]]
cd /workspace/nemo-rl
"${UV_BIN}" sync --locked --extra mcore --python 3.13.13
[[ "$("${UV_PROJECT_ENVIRONMENT}/bin/python" --version)" == "Python 3.13.13" ]]
"${UV_BIN}" lock --check

"${UV_BIN}" run --active --no-sync python scripts/smoke_nemo_rl_image.py \
    | tee /results/runtime.json
BASH
