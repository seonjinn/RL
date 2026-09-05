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
readonly BATCH_SCRIPT=${SCRIPT_ROOT}/experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch
readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch
readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch
readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf
readonly LOG_DIRECTORY=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-smoke
readonly SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28

[[ "${SCRIPT_ROOT}" = /* ]]
worktree_status=$(git -C "${SCRIPT_ROOT}" status --porcelain)
test -z "${worktree_status}"
test -x "${BATCH_SCRIPT}"
test -x "${SBATCH_COMMAND}"
test -f "${OCI_SLURM_CONF}"
[[ "${SEMANTIC_WORKTREE}" = /* ]]
test "$(git -C "${SEMANTIC_WORKTREE}" rev-parse --is-inside-work-tree)" = true
test "$(git -C "${SEMANTIC_WORKTREE}" rev-parse --show-toplevel)" = "${SEMANTIC_WORKTREE}"
semantic_worktree_status=$(git -C "${SEMANTIC_WORKTREE}" status --porcelain)
test -z "${semantic_worktree_status}"
SEMANTIC_UPSTREAM=$(git -C "${SEMANTIC_WORKTREE}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
readonly SEMANTIC_UPSTREAM
test -n "${SEMANTIC_UPSTREAM}"
EXPECTED_REPO_SHA=$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)
readonly EXPECTED_REPO_SHA
SEMANTIC_UPSTREAM_SHA=$(git -C "${SEMANTIC_WORKTREE}" rev-parse '@{upstream}')
readonly SEMANTIC_UPSTREAM_SHA
test "${EXPECTED_REPO_SHA}" = "${SEMANTIC_UPSTREAM_SHA}"
mkdir -p "${LOG_DIRECTORY}"

case ${ACTION} in
  test-only)
    git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" | /usr/bin/env -i \
      PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin \
      SLURM_CONF="${OCI_SLURM_CONF}" \
      "${SBATCH_COMMAND}" \
      --test-only \
      --chdir="${SCRIPT_ROOT}" \
      --export="SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA},SEMANTIC_WORKTREE=${SEMANTIC_WORKTREE},EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA}"
    ;;
  submit)
    git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" | /usr/bin/env -i \
      PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin \
      SLURM_CONF="${OCI_SLURM_CONF}" \
      "${SBATCH_COMMAND}" \
      --chdir="${SCRIPT_ROOT}" \
      --export="SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA},SEMANTIC_WORKTREE=${SEMANTIC_WORKTREE},EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA}"
    ;;
  *)
    echo "Unsupported ACTION: ${ACTION}" >&2
    exit 2
    ;;
esac
