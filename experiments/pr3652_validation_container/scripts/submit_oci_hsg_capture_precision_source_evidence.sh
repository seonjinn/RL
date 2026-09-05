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

set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: ACTION=test-only|submit $0 [ACTION]" >&2
  exit 2
fi

readonly ACTION=${1:-test-only}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "${script_dir}" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
EXPECTED_TOOLING_SHA=$(git -C "${SCRIPT_ROOT}" rev-parse HEAD)
readonly EXPECTED_TOOLING_SHA
TOOLING_UPSTREAM=$(git -C "${SCRIPT_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
readonly TOOLING_UPSTREAM
test -n "${TOOLING_UPSTREAM}"
TOOLING_UPSTREAM_SHA=$(git -C "${SCRIPT_ROOT}" rev-parse '@{upstream}')
readonly TOOLING_UPSTREAM_SHA
test "${EXPECTED_TOOLING_SHA}" = "${TOOLING_UPSTREAM_SHA}"

readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sbatch
readonly BATCH_SCRIPT=${SCRIPT_ROOT}/${BATCH_RELATIVE_PATH}
readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch
readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf
readonly SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28
readonly COMPRESSED_TENSORS_SOURCE_ROOT=/home/sna/nemorl-source-evidence/checkouts/compressed-tensors/sha256-f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0
readonly MODELOPT_LIGHTNING_SOURCE_ROOT=/home/sna/nemorl-source-evidence/checkouts/model-optimizer/sha256-c897fbeaaff66d53d61033f107885b7c5432f235
readonly TRANSFORMER_ENGINE_SOURCE_ROOT=/home/sna/nemorl-source-evidence/checkouts/transformer-engine/sha256-42b840051647eef89761a16dfdff87e82bb253ab
readonly STAGED_METADATA_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw/sha256-d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b
readonly EXPECTED_RAW_MANIFEST_SHA256=d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b
readonly CAPTURED_BASE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured
readonly LOG_DIRECTORY=${CAPTURED_BASE}/logs
readonly RUN_DIRECTORY=${CAPTURED_BASE}/runs

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

require_pushed_git_tree() {
  local path=$1
  local expected_sha=$2
  local upstream
  local upstream_sha

  require_clean_git_tree "${path}" "${expected_sha}"
  upstream=$(git -C "${path}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
  test -n "${upstream}"
  upstream_sha=$(git -C "${path}" rev-parse '@{upstream}')
  test "${expected_sha}" = "${upstream_sha}"
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

[[ "${SCRIPT_ROOT}" = /* ]]
test -x "${BATCH_SCRIPT}"
test -x "${SBATCH_COMMAND}"
test -f "${OCI_SLURM_CONF}"
worktree_status=$(git -C "${SCRIPT_ROOT}" status --porcelain)
test -z "${worktree_status}"
EXPECTED_REPO_SHA=$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)
readonly EXPECTED_REPO_SHA
require_pushed_git_tree "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}"
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)" = b11414c71b15e54d333eb49346ed199f20fa9021
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Automodel-workspace/Automodel" rev-parse HEAD)" = 1814c6c93a66b9d59d254960ef6a99a64249b671
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)" = 7c9c3a027c503ae9ae1e8ad7b14397abb8269378
require_pinned_source_tree "${COMPRESSED_TENSORS_SOURCE_ROOT}" f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0 https://github.com/vllm-project/compressed-tensors.git
require_pinned_source_tree "${MODELOPT_LIGHTNING_SOURCE_ROOT}" c897fbeaaff66d53d61033f107885b7c5432f235 https://github.com/NVIDIA/Model-Optimizer.git
require_pinned_source_tree "${TRANSFORMER_ENGINE_SOURCE_ROOT}" 42b840051647eef89761a16dfdff87e82bb253ab https://github.com/NVIDIA/TransformerEngine.git
validate_raw_receipt
mkdir -p "${LOG_DIRECTORY}" "${RUN_DIRECTORY}"

readonly EXPORTS=SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA},SEMANTIC_WORKTREE=${SEMANTIC_WORKTREE},EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA},COMPRESSED_TENSORS_SOURCE_ROOT=${COMPRESSED_TENSORS_SOURCE_ROOT},MODELOPT_LIGHTNING_SOURCE_ROOT=${MODELOPT_LIGHTNING_SOURCE_ROOT},TRANSFORMER_ENGINE_SOURCE_ROOT=${TRANSFORMER_ENGINE_SOURCE_ROOT},STAGED_METADATA_ROOT=${STAGED_METADATA_ROOT}

case ${ACTION} in
  test-only)
    git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" | /usr/bin/env -i \
      PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin \
      SLURM_CONF="${OCI_SLURM_CONF}" \
      "${SBATCH_COMMAND}" \
      --test-only \
      --chdir="${SCRIPT_ROOT}" \
      --export="${EXPORTS}"
    ;;
  submit)
    git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" | /usr/bin/env -i \
      PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin \
      SLURM_CONF="${OCI_SLURM_CONF}" \
      "${SBATCH_COMMAND}" \
      --chdir="${SCRIPT_ROOT}" \
      --export="${EXPORTS}"
    ;;
  *)
    echo "Unsupported ACTION: ${ACTION}" >&2
    exit 2
    ;;
esac
