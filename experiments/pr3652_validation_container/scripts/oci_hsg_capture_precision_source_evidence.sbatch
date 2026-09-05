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
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=nemotron_n3_post-capture_precision_evidence
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured/logs/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured/logs/slurm-%j.err

set -Eeuo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

: "${SCRIPT_ROOT:?Set SCRIPT_ROOT to the absolute clean staging worktree root}"
: "${EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA to the staging worktree HEAD}"
: "${SEMANTIC_WORKTREE:?Set SEMANTIC_WORKTREE to the absolute clean semantic policy worktree root}"
: "${EXPECTED_REPO_SHA:?Set EXPECTED_REPO_SHA to the semantic worktree HEAD}"
: "${COMPRESSED_TENSORS_SOURCE_ROOT:?Set COMPRESSED_TENSORS_SOURCE_ROOT to the pinned source tree}"
: "${MODELOPT_LIGHTNING_SOURCE_ROOT:?Set MODELOPT_LIGHTNING_SOURCE_ROOT to the pinned source tree}"
: "${TRANSFORMER_ENGINE_SOURCE_ROOT:?Set TRANSFORMER_ENGINE_SOURCE_ROOT to the pinned source tree}"
: "${STAGED_METADATA_ROOT:?Set STAGED_METADATA_ROOT to the verified raw receipt root}"

readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_CONTAINER_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly EXPECTED_RAW_MANIFEST_SHA256=d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b
readonly CAPTURED_BASE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured
readonly VALIDATOR_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/validate_transferred_nightly.sh
readonly CAPTURE_BODY_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sh
readonly VALIDATOR=${SCRIPT_ROOT}/${VALIDATOR_RELATIVE_PATH}
readonly CAPTURE_BODY=${SCRIPT_ROOT}/${CAPTURE_BODY_RELATIVE_PATH}
readonly SEMANTIC_WORKTREE EXPECTED_REPO_SHA
readonly COMPRESSED_TENSORS_SOURCE_ROOT MODELOPT_LIGHTNING_SOURCE_ROOT
readonly TRANSFORMER_ENGINE_SOURCE_ROOT STAGED_METADATA_ROOT

if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi

readonly SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-semantic-precision-evidence/oci-capture-${SLURM_JOB_ID}
readonly VALIDATOR_SNAPSHOT=${SCRATCH_DIRECTORY}/validate_transferred_nightly.sh
readonly CAPTURE_BODY_SNAPSHOT=${SCRATCH_DIRECTORY}/oci_hsg_capture_precision_source_evidence.sh

require_clean_git_tree() {
  local path=$1
  local expected_sha=$2
  local worktree_status

  [[ "${path}" = /* ]]
  test ! -L "${path}"
  test "$(git -C "${path}" rev-parse --is-inside-work-tree)" = true
  test "$(git -C "${path}" rev-parse --show-toplevel)" = "${path}"
  test "$(git -C "${path}" rev-parse HEAD)" = "${expected_sha}"
  worktree_status=$(git -C "${path}" status --porcelain)
  test -z "${worktree_status}"
}

require_pinned_source_tree() {
  local path=$1
  local expected_sha=$2
  local expected_origin=$3
  local origin_count
  local origin_urls

  require_clean_git_tree "${path}" "${expected_sha}"
  origin_urls=$(git -C "${path}" config --get-all remote.origin.url)
  origin_count=$(printf '%s\n' "${origin_urls}" | sed '/^$/d' | wc -l | tr -d ' ')
  test "${origin_count}" = 1
  test "${origin_urls}" = "${expected_origin}"
}

validate_tooling_root() {
  local worktree_status

  [[ "${SCRIPT_ROOT}" = /* ]]
  test -d "${SCRIPT_ROOT}"
  test "$(git -C "${SCRIPT_ROOT}" rev-parse --show-toplevel)" = "${SCRIPT_ROOT}"
  test "$(git -C "${SCRIPT_ROOT}" rev-parse HEAD)" = "${EXPECTED_TOOLING_SHA}"
  worktree_status=$(git -C "${SCRIPT_ROOT}" status --porcelain)
  test -z "${worktree_status}"
  test -x "${VALIDATOR}"
  test -x "${CAPTURE_BODY}"
}

snapshot_tooling_file() {
  local relative_path=$1
  local snapshot_path=$2
  local expected_blob
  local actual_blob

  expected_blob=$(git -C "${SCRIPT_ROOT}" rev-parse "${EXPECTED_TOOLING_SHA}:${relative_path}")
  git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${relative_path}" >"${snapshot_path}"
  actual_blob=$(git -C "${SCRIPT_ROOT}" hash-object "${snapshot_path}")
  test "${actual_blob}" = "${expected_blob}"
  chmod 755 "${snapshot_path}"
}

validate_raw_receipt() {
  local file_count
  local manifest_line_count
  local manifest_sha256
  local symlink_path

  [[ "${STAGED_METADATA_ROOT}" = /* ]]
  test -d "${STAGED_METADATA_ROOT}"
  test ! -L "${STAGED_METADATA_ROOT}"
  test -f "${STAGED_METADATA_ROOT}/SHA256SUMS"
  test ! -L "${STAGED_METADATA_ROOT}/SHA256SUMS"
  symlink_path=$(find "${STAGED_METADATA_ROOT}" -type l -print -quit)
  test -z "${symlink_path}"
  file_count=$(find "${STAGED_METADATA_ROOT}" -type f ! -path "${STAGED_METADATA_ROOT}/SHA256SUMS" -print | wc -l | tr -d ' ')
  test "${file_count}" = 19
  manifest_line_count=$(wc -l <"${STAGED_METADATA_ROOT}/SHA256SUMS" | tr -d ' ')
  test "${manifest_line_count}" = 19
  manifest_sha256=$(sha256sum "${STAGED_METADATA_ROOT}/SHA256SUMS" | awk '{print $1}')
  test "${manifest_sha256}" = "${EXPECTED_RAW_MANIFEST_SHA256}"
  (
    cd "${STAGED_METADATA_ROOT}"
    awk 'NF != 2 || $2 ~ /^\// || $2 ~ /(^|\/)\.\.($|\/)/ { exit 1 }' SHA256SUMS
    diff -u \
      <(awk '{print $2}' SHA256SUMS | LC_ALL=C sort) \
      <(find . -type f ! -name SHA256SUMS -print | sed 's#^\./##' | LC_ALL=C sort)
    sha256sum --check --strict SHA256SUMS
  )
}

cleanup() {
  local exit_status=$?

  trap - EXIT
  rm -rf -- "${SCRATCH_DIRECTORY}"
  exit "${exit_status}"
}

test -f "${CONTAINER}"
validate_tooling_root
require_clean_git_tree "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}"
require_pinned_source_tree "${COMPRESSED_TENSORS_SOURCE_ROOT}" f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0 https://github.com/vllm-project/compressed-tensors.git
require_pinned_source_tree "${MODELOPT_LIGHTNING_SOURCE_ROOT}" c897fbeaaff66d53d61033f107885b7c5432f235 https://github.com/NVIDIA/Model-Optimizer.git
require_pinned_source_tree "${TRANSFORMER_ENGINE_SOURCE_ROOT}" 42b840051647eef89761a16dfdff87e82bb253ab https://github.com/NVIDIA/TransformerEngine.git
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)" = b11414c71b15e54d333eb49346ed199f20fa9021
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Automodel-workspace/Automodel" rev-parse HEAD)" = 1814c6c93a66b9d59d254960ef6a99a64249b671
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)" = 7c9c3a027c503ae9ae1e8ad7b14397abb8269378
validate_raw_receipt
test -d "${CAPTURED_BASE}"
test -d "${CAPTURED_BASE}/runs"
test ! -e "${SCRATCH_DIRECTORY}"
mkdir -p "${SCRATCH_DIRECTORY}"
trap cleanup EXIT
snapshot_tooling_file "${VALIDATOR_RELATIVE_PATH}" "${VALIDATOR_SNAPSHOT}"
snapshot_tooling_file "${CAPTURE_BODY_RELATIVE_PATH}" "${CAPTURE_BODY_SNAPSHOT}"
validator_snapshot_sha256=$(sha256sum "${VALIDATOR_SNAPSHOT}" | awk '{print $1}')
validator_expected_image_sha256=$(sed -n 's/^readonly EXPECTED_SHA256=//p' "${VALIDATOR_SNAPSHOT}")
test "${validator_expected_image_sha256}" = "${EXPECTED_CONTAINER_SHA256}"
printf 'validator_snapshot_path=%s\n' "${VALIDATOR_SNAPSHOT}"
printf 'validator_snapshot_sha256=%s\n' "${validator_snapshot_sha256}"
printf 'expected_image_sha256=%s\n' "${EXPECTED_CONTAINER_SHA256}"

export TMPDIR=${SCRATCH_DIRECTORY}/tmp
export PYTHONPYCACHEPREFIX=${SCRATCH_DIRECTORY}/pycache
export XDG_CACHE_HOME=${SCRATCH_DIRECTORY}/xdg-cache
export UV_CACHE_DIR=${SCRATCH_DIRECTORY}/uv-cache
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH_DIRECTORY}/torchinductor-cache
export TRITON_CACHE_DIR=${SCRATCH_DIRECTORY}/triton-cache
export SLURM_EXPORT_ENV=ALL
mkdir -p \
  "${TMPDIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}"

"${VALIDATOR_SNAPSHOT}" "${CONTAINER}" "${CONTAINER}.metadata.txt" "${CONTAINER}.complete"

/cm/local/apps/slurm/current/bin/srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-remap-root \
  "${CAPTURE_BODY_SNAPSHOT}" \
  "${SEMANTIC_WORKTREE}" \
  "${EXPECTED_REPO_SHA}" \
  "${COMPRESSED_TENSORS_SOURCE_ROOT}" \
  "${MODELOPT_LIGHTNING_SOURCE_ROOT}" \
  "${TRANSFORMER_ENGINE_SOURCE_ROOT}" \
  "${STAGED_METADATA_ROOT}" \
  "${SCRATCH_DIRECTORY}" \
  "${CAPTURED_BASE}" \
  "${EXPECTED_TOOLING_SHA}"
