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
readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/oci_hsg_stage_precision_source_metadata.sbatch
readonly BATCH_SCRIPT=${SCRIPT_ROOT}/${BATCH_RELATIVE_PATH}
readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch
readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf
readonly SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28
readonly OUTPUT_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw
readonly LOG_DIRECTORY=${OUTPUT_ROOT}/logs
readonly -a REQUIRED_SEMANTIC_BLOBS=(
  tools/stage_precision_policy_source_metadata.py
  tools/capture_precision_policy_source_evidence.py
  nemo_rl/precision_policy/semantic.py
  nemo_rl/precision_policy/source_formats.py
)

require_clean_pushed_git_tree() {
  local path=$1
  local expected_sha=$2
  local status
  local upstream
  local upstream_sha

  [[ "${path}" = /* ]]
  test ! -L "${path}"
  test "$(git -C "${path}" rev-parse --is-inside-work-tree)" = true
  test "$(git -C "${path}" rev-parse --show-toplevel)" = "${path}"
  test "$(git -C "${path}" rev-parse HEAD)" = "${expected_sha}"
  status=$(git -C "${path}" status --porcelain)
  test -z "${status}"
  upstream=$(git -C "${path}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
  test -n "${upstream}"
  upstream_sha=$(git -C "${path}" rev-parse '@{upstream}')
  test "${expected_sha}" = "${upstream_sha}"
}

require_commit_blob() {
  local repository=$1
  local commit=$2
  local relative_path=$3

  git -C "${repository}" cat-file -e "${commit}:${relative_path}"
  test "$(git -C "${repository}" cat-file -t "${commit}:${relative_path}")" = blob
}

[[ "${SCRIPT_ROOT}" = /* ]]
test -x "${BATCH_SCRIPT}"
test -x "${SBATCH_COMMAND}"
test -f "${OCI_SLURM_CONF}"
require_clean_pushed_git_tree "${SCRIPT_ROOT}" "${EXPECTED_TOOLING_SHA}"
require_commit_blob "${SCRIPT_ROOT}" "${EXPECTED_TOOLING_SHA}" "${BATCH_RELATIVE_PATH}"

EXPECTED_REPO_SHA=$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)
readonly EXPECTED_REPO_SHA
require_clean_pushed_git_tree "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}"
for relative_path in "${REQUIRED_SEMANTIC_BLOBS[@]}"; do
  require_commit_blob "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}" "${relative_path}"
done

mkdir -p "${LOG_DIRECTORY}"
readonly EXPORTS=SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA},SEMANTIC_WORKTREE=${SEMANTIC_WORKTREE},EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA}

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
