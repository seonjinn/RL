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
  echo "Usage: $0 [test-only|submit]" >&2
  exit 2
fi

readonly ACTION=${1:-test-only}
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "${script_directory}" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
EXPECTED_TOOLING_SHA=$(git -C "${SCRIPT_ROOT}" rev-parse HEAD)
readonly EXPECTED_TOOLING_SHA
TOOLING_UPSTREAM=$(git -C "${SCRIPT_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
readonly TOOLING_UPSTREAM
TOOLING_UPSTREAM_SHA=$(git -C "${SCRIPT_ROOT}" rev-parse '@{upstream}')
readonly TOOLING_UPSTREAM_SHA
readonly WRAPPER_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/submit_lyris_download_validated_nightly.sh
readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/lyris_download_validated_nightly.sbatch
readonly BATCH_SCRIPT=${SCRIPT_ROOT}/${BATCH_RELATIVE_PATH}
readonly SBATCH_COMMAND=/usr/bin/sbatch
readonly HOME_DIRECTORY=/home/sna
readonly LOG_DIRECTORY=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/lyris-download

[[ ${SCRIPT_ROOT} = /* ]]
test ! -L "${SCRIPT_ROOT}"
[[ ${EXPECTED_TOOLING_SHA} =~ ^[0-9a-f]{40}$ ]]
test -n "${TOOLING_UPSTREAM}"
test "${EXPECTED_TOOLING_SHA}" = "${TOOLING_UPSTREAM_SHA}"
worktree_status=$(git -C "${SCRIPT_ROOT}" status --porcelain)
test -z "${worktree_status}"
test -x "${BATCH_SCRIPT}"
test -x "${SBATCH_COMMAND}"

committed_wrapper_blob=$(git -C "${SCRIPT_ROOT}" rev-parse --verify \
  "${EXPECTED_TOOLING_SHA}:${WRAPPER_RELATIVE_PATH}")
executed_wrapper_blob=$(git -C "${SCRIPT_ROOT}" hash-object "${BASH_SOURCE[0]}")
test "${executed_wrapper_blob}" = "${committed_wrapper_blob}"
git -C "${SCRIPT_ROOT}" cat-file -e "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}"
test "$(git -C "${SCRIPT_ROOT}" cat-file -t \
  "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}")" = blob

mkdir -p "${LOG_DIRECTORY}"
readonly EXPORTS=HOME=${HOME_DIRECTORY},PATH=/usr/bin:/bin,SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${EXPECTED_TOOLING_SHA}

SBATCH_ARGUMENTS=(
  --chdir="${SCRIPT_ROOT}"
  --export="${EXPORTS}"
)
case ${ACTION} in
  test-only)
    SBATCH_ARGUMENTS=(--test-only "${SBATCH_ARGUMENTS[@]}")
    ;;
  submit)
    ;;
  *)
    echo "Unsupported action: ${ACTION}" >&2
    exit 2
    ;;
esac
readonly SBATCH_ARGUMENTS

git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" |
  /usr/bin/env -i \
    HOME="${HOME_DIRECTORY}" \
    PATH=/usr/bin:/bin \
    "${SBATCH_COMMAND}" "${SBATCH_ARGUMENTS[@]}"
