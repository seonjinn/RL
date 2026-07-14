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
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=nemo-rl-vpp-pr1126
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${VPP_VALIDATION_IMAGE:?Set VPP_VALIDATION_IMAGE to an immutable squashfs path}"
: "${VPP_VALIDATION_IMAGE_SHA256:?Set VPP_VALIDATION_IMAGE_SHA256}"
: "${VPP_VALIDATION_GIT_SHA:?Set VPP_VALIDATION_GIT_SHA to the pushed source SHA}"
: "${VPP_VALIDATION_DIR:?Set VPP_VALIDATION_DIR to an absolute result directory}"
: "${VPP_VALIDATION_HF_HOME:?Set VPP_VALIDATION_HF_HOME to a shared Hugging Face cache}"
: "${SLURM_SUBMIT_DIR:?Submit from the NeMo-RL checkout root}"
if [[ "${VPP_VALIDATION_DIR}" != /* || "${VPP_VALIDATION_HF_HOME}" != /* ]]; then
    echo "[ERROR] validation and Hugging Face paths must be absolute" >&2
    exit 1
fi

repo_root=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
readonly repo_root
source_status=$(git -C "${repo_root}" status --porcelain \
    --untracked-files=normal --ignore-submodules=none)
readonly source_status
if [[ -n "${source_status}" ]]; then
    echo "[ERROR] refusing to validate a dirty source tree" >&2
    printf '%s\n' "${source_status}" >&2
    exit 1
fi
submodule_status=$(git -C "${repo_root}" submodule status --recursive)
readonly submodule_status
while IFS= read -r line; do
    if [[ -n "${line}" && "${line:0:1}" != " " ]]; then
        echo "[ERROR] recursive submodule is not clean and initialized: ${line}" >&2
        exit 1
    fi
done <<< "${submodule_status}"
if ! git -C "${repo_root}" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=normal --ignore-submodules=none)"'; then
    echo "[ERROR] recursive submodule checkout is dirty" >&2
    exit 1
fi

source_sha=$(git -C "${repo_root}" rev-parse HEAD)
readonly source_sha
if [[ "${source_sha}" != "${VPP_VALIDATION_GIT_SHA}" ]]; then
    echo "[ERROR] source SHA ${source_sha} does not match ${VPP_VALIDATION_GIT_SHA}" >&2
    exit 1
fi
upstream_sha=$(git -C "${repo_root}" rev-parse '@{upstream}')
readonly upstream_sha
if [[ "${source_sha}" != "${upstream_sha}" ]]; then
    echo "[ERROR] source SHA ${source_sha} does not match upstream ${upstream_sha}" >&2
    exit 1
fi

actual_image_sha256=$(sha256sum "${VPP_VALIDATION_IMAGE}" | awk '{print $1}')
readonly actual_image_sha256
if [[ "${actual_image_sha256}" != "${VPP_VALIDATION_IMAGE_SHA256}" ]]; then
    echo "[ERROR] image SHA256 ${actual_image_sha256} does not match expected digest" >&2
    exit 1
fi

mkdir -p "${VPP_VALIDATION_DIR}/runtime/tmp"
printf '%s\n' "${source_sha}" > "${VPP_VALIDATION_DIR}/source.sha"
printf '%s  %s\n' "${actual_image_sha256}" "${VPP_VALIDATION_IMAGE}" \
    > "${VPP_VALIDATION_DIR}/image.sha256"
printf '%s\n' "${submodule_status}" > "${VPP_VALIDATION_DIR}/submodule_status.txt"

container_run=(
    srun
    --nodes=1
    --ntasks=1
    --cpus-per-task=72
    --mpi=pmix
    --container-image="${VPP_VALIDATION_IMAGE}"
    --container-mounts="${repo_root}:/workspace/nemo-rl,${VPP_VALIDATION_DIR}:/results,${VPP_VALIDATION_DIR}/runtime:/runtime,${VPP_VALIDATION_HF_HOME}:/hf_home"
    --container-workdir=/workspace/nemo-rl
)
if [[ -n "${VPP_VALIDATION_GRES:-}" ]]; then
    container_run+=("--gres=${VPP_VALIDATION_GRES}")
fi
readonly -a container_run

"${container_run[@]}" bash -s 2>&1 <<'BASH' | tee "${VPP_VALIDATION_DIR}/validation.log"
set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export HF_HOME=/hf_home
export NEMO_RL_VENV_DIR=/runtime/worker_venvs
export NRL_FORCE_REBUILD_VENVS=true
export NVTE_CUDA_ARCHS=100
export RAY_TMPDIR=/runtime/ray
export RAY_USAGE_STATS_ENABLED=0
export TMPDIR=/runtime/tmp
export TORCH_CUDA_ARCH_LIST=10.0
export TORCH_EXTENSIONS_DIR=/runtime/torch_extensions
export TRITON_CACHE_DIR=/runtime/triton_cache
export UV_NO_EDITABLE=1
export UV_PROJECT_ENVIRONMENT=/runtime/venv
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
readonly UV_BIN=/root/.local/bin/uv

mkdir -p \
    "${NEMO_RL_VENV_DIR}" \
    "${RAY_TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}"
cd /workspace/nemo-rl
"${UV_BIN}" sync --locked --extra mcore --group test --python 3.13.13
"${UV_BIN}" lock --check
"${UV_BIN}" run --active --no-sync pytest -vv -s \
    --mcore-only \
    --hf-gated \
    --runxfail \
    --junitxml=/results/junit.xml \
    'tests/unit/models/policy/test_megatron_worker.py::test_megatron_policy_training[2gpu_pp2_vpp2_llama]' \
    'tests/unit/models/policy/test_megatron_worker.py::test_megatron_policy_logprobs[2gpu_pp2_vpp2_llama]' \
    'tests/unit/models/policy/test_megatron_worker.py::test_megatron_checkpoint_save_kill_and_restore[2gpu_pp2_vpp2_save_restore]'

"${UV_BIN}" run --active --no-sync python - <<'PY'
from pathlib import Path
from xml.etree import ElementTree


root = ElementTree.parse(Path("/results/junit.xml")).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
totals = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
if totals != {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}:
    raise SystemExit(f"unexpected VPP test totals: {totals}")
print(f"VPP PR #1126 validation passed: {totals}")
PY
BASH
