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
unset BASH_ENV ENV CDPATH GLOBIGNORE
PATH=/usr/bin:/bin
readonly PATH
export PATH

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_sha256() {
  local variable_name=$1
  local value=$2

  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    die "${variable_name} must be exactly 64 lowercase hexadecimal characters"
  fi
}

readonly HOME_DIRECTORY=/home/sna

clean_git() {
  /usr/bin/env -i \
    HOME="${HOME_DIRECTORY}" \
    PATH=/usr/bin:/bin \
    /usr/bin/git "$@"
}

require_commit_blob() {
  local repository=$1
  local commit=$2
  local relative_path=$3
  local object_type

  object_type=$(clean_git -C "${repository}" cat-file -t \
    "${commit}:${relative_path}") ||
    die "committed path is missing: ${relative_path}"
  if [[ "${object_type}" != blob ]]; then
    die "committed path is not a blob: ${relative_path}"
  fi
}

require_canonical_directory() {
  local directory=$1
  local containment_root=$2
  local resolved_directory

  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    die "directory must exist and must not be a symlink: ${directory}"
  fi
  resolved_directory=$(cd "${directory}" && pwd -P) ||
    die "directory is not accessible: ${directory}"
  if [[ "${resolved_directory}" != "${directory}" ]]; then
    die "directory path must be canonical: ${directory}"
  fi
  case "${resolved_directory}" in
  "${containment_root}" | "${containment_root}"/*) ;;
  *) die "directory escapes containment root ${containment_root}: ${directory}" ;;
  esac
}

create_canonical_leaf_directory() {
  local parent=$1
  local leaf=$2
  local mode=$3
  local directory=${parent}/${leaf}

  if [[ ! "${leaf}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    die "unsafe directory leaf: ${leaf}"
  fi
  if [[ ! -e "${directory}" && ! -L "${directory}" ]]; then
    /bin/mkdir -m "${mode}" "${directory}" 2>/dev/null || true
  fi
  require_canonical_directory "${directory}" "${parent}"
}

require_private_owned_directory() {
  local directory=$1
  local mode

  if [[ ! -O "${directory}" ]]; then
    die "directory must be owned by the staging user: ${directory}"
  fi
  if mode=$(/usr/bin/stat -Lc '%a' "${directory}" 2>/dev/null); then
    :
  elif mode=$(/usr/bin/stat -f '%Lp' "${directory}" 2>/dev/null); then
    :
  else
    die "could not inspect directory permissions: ${directory}"
  fi
  if [[ ! "${mode}" =~ ^[0-7]{3,4}$ ]]; then
    die "directory permissions are malformed: ${directory}"
  fi
  if ((8#${mode} & 8#022)); then
    die "directory must not be group- or other-writable: ${directory}"
  fi
}

if (( $# > 1 )); then
  die "Usage: $0 [test-only|submit]"
fi
readonly ACTION=${1:-test-only}
case "${ACTION}" in
test-only | submit) ;;
*) die "unsupported action: ${ACTION}" ;;
esac

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to an explicit registry/repository:tag}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the expected NeMo-RL source commit}"
: "${SOURCE_LOCK_SHA256:?Set SOURCE_LOCK_SHA256 to the expected uv.lock SHA-256}"
: "${SOURCE_PYPROJECT_SHA256:?Set SOURCE_PYPROJECT_SHA256 to the expected pyproject.toml SHA-256}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe candidate name}"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  die 'SOURCE_COMMIT must be exactly 40 lowercase hexadecimal characters'
fi
require_sha256 SOURCE_LOCK_SHA256 "${SOURCE_LOCK_SHA256}"
require_sha256 SOURCE_PYPROJECT_SHA256 "${SOURCE_PYPROJECT_SHA256}"
if [[ ! "${OUTPUT_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  die 'OUTPUT_PREFIX is not filesystem-safe'
fi
if [[ "${SOURCE_IMAGE}" != */*:* || "${SOURCE_IMAGE}" == *://* ||
  "${SOURCE_IMAGE}" == *'@'* || "${SOURCE_IMAGE}" == *','* ||
  "${SOURCE_IMAGE}" == *'#'* || "${SOURCE_IMAGE}" =~ [[:space:]] ]]; then
  die 'SOURCE_IMAGE must be an uncredentialed registry/repository:tag reference'
fi
SOURCE_REGISTRY=${SOURCE_IMAGE%%/*}
REPOSITORY_AND_TAG=${SOURCE_IMAGE#*/}
SOURCE_REPOSITORY=${REPOSITORY_AND_TAG%:*}
SOURCE_TAG=${REPOSITORY_AND_TAG##*:}
readonly SOURCE_REGISTRY REPOSITORY_AND_TAG SOURCE_REPOSITORY SOURCE_TAG
if [[ ! "${SOURCE_REGISTRY}" =~ ^[a-z0-9][a-z0-9.-]*(:[0-9]+)?$ ||
  ! "${SOURCE_REPOSITORY}" =~ ^[a-z0-9]+([-._][a-z0-9]+)*(/[a-z0-9]+([-._][a-z0-9]+)*)*$ ||
  ! "${SOURCE_TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  die 'SOURCE_IMAGE registry, repository, or tag is malformed'
fi

SCRIPT_DIRECTORY=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIRECTORY
readonly WRAPPER_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/submit_oci_hsg_stage_enroot_image.sh
readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/stage_enroot_image.sbatch
EXECUTED_WRAPPER=${SCRIPT_DIRECTORY}/$(/usr/bin/basename "${BASH_SOURCE[0]}")
readonly EXECUTED_WRAPPER
SCRIPT_ROOT=$(clean_git -C "${SCRIPT_DIRECTORY}" rev-parse --show-toplevel) ||
  die 'submit wrapper is not inside a Git checkout'
SCRIPT_ROOT=$(cd "${SCRIPT_ROOT}" && pwd -P) ||
  die 'tooling Git checkout is not accessible'
readonly SCRIPT_ROOT
if [[ "${EXECUTED_WRAPPER}" != "${SCRIPT_ROOT}/${WRAPPER_RELATIVE_PATH}" ||
  ! -f "${EXECUTED_WRAPPER}" || -L "${EXECUTED_WRAPPER}" ]]; then
  die 'submit wrapper must be executed from its canonical committed path'
fi
EXPECTED_TOOLING_SHA=$(clean_git -C "${SCRIPT_ROOT}" rev-parse --verify HEAD) ||
  die 'could not resolve tooling HEAD'
readonly EXPECTED_TOOLING_SHA
if [[ ! "${EXPECTED_TOOLING_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  die 'tooling HEAD is not an exact SHA-1 commit'
fi
TOOLING_STATUS=$(clean_git -C "${SCRIPT_ROOT}" status --porcelain=v1 \
  --untracked-files=all --ignore-submodules=none) ||
  die 'could not inspect tooling checkout status'
readonly TOOLING_STATUS
if [[ -n "${TOOLING_STATUS}" ]]; then
  die 'tooling checkout must be clean, including untracked files'
fi
TOOLING_UPSTREAM=$(clean_git -C "${SCRIPT_ROOT}" rev-parse \
  --abbrev-ref --symbolic-full-name '@{upstream}') ||
  die 'tooling branch must have an upstream'
readonly TOOLING_UPSTREAM
if [[ -z "${TOOLING_UPSTREAM}" ]]; then
  die 'tooling upstream name is empty'
fi
TOOLING_UPSTREAM_SHA=$(clean_git -C "${SCRIPT_ROOT}" rev-parse --verify \
  '@{upstream}') || die 'could not resolve tooling upstream'
readonly TOOLING_UPSTREAM_SHA
if [[ "${EXPECTED_TOOLING_SHA}" != "${TOOLING_UPSTREAM_SHA}" ]]; then
  die 'tooling HEAD must exactly match its pushed upstream'
fi

require_commit_blob "${SCRIPT_ROOT}" "${EXPECTED_TOOLING_SHA}" \
  "${WRAPPER_RELATIVE_PATH}"
require_commit_blob "${SCRIPT_ROOT}" "${EXPECTED_TOOLING_SHA}" \
  "${BATCH_RELATIVE_PATH}"
COMMITTED_WRAPPER_BLOB=$(clean_git -C "${SCRIPT_ROOT}" rev-parse --verify \
  "${EXPECTED_TOOLING_SHA}:${WRAPPER_RELATIVE_PATH}") ||
  die 'could not resolve committed submit wrapper blob'
readonly COMMITTED_WRAPPER_BLOB
EXECUTED_WRAPPER_BLOB=$(clean_git -C "${SCRIPT_ROOT}" hash-object \
  "${EXECUTED_WRAPPER}") || die 'could not hash executed submit wrapper'
readonly EXECUTED_WRAPPER_BLOB
if [[ "${EXECUTED_WRAPPER_BLOB}" != "${COMMITTED_WRAPPER_BLOB}" ]]; then
  die 'executed submit wrapper bytes do not match the committed blob'
fi

readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch
readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf
readonly PUBLISH_ROOT=/lustre
readonly CONTAINER_VALIDATION_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/container-validation
readonly STAGE_ROOT=${CONTAINER_VALIDATION_ROOT}/nightly-stage
readonly CONTAINER_DIR=${STAGE_ROOT}/candidates
readonly LOG_DIRECTORY=${STAGE_ROOT}/logs

[[ -x "${SBATCH_COMMAND}" ]] || die "sbatch is not executable: ${SBATCH_COMMAND}"
[[ -f "${OCI_SLURM_CONF}" && ! -L "${OCI_SLURM_CONF}" ]] ||
  die "Slurm configuration is not a regular file: ${OCI_SLURM_CONF}"
require_canonical_directory "${PUBLISH_ROOT}" "${PUBLISH_ROOT}"
require_canonical_directory "${CONTAINER_VALIDATION_ROOT}" "${PUBLISH_ROOT}"
create_canonical_leaf_directory "${CONTAINER_VALIDATION_ROOT}" nightly-stage 2770
create_canonical_leaf_directory "${STAGE_ROOT}" candidates 2700
create_canonical_leaf_directory "${STAGE_ROOT}" logs 2770
require_private_owned_directory "${CONTAINER_DIR}"

readonly EXPORT_NAMES=HOME,PATH,SOURCE_IMAGE,SOURCE_COMMIT,SOURCE_LOCK_SHA256,SOURCE_PYPROJECT_SHA256,OUTPUT_PREFIX,CONTAINER_DIR,TOOLING_WORKTREE,TOOLING_COMMIT

SBATCH_ARGUMENTS=(
  --nodes=1
  --ntasks-per-node=1
  --gres=gpu:1
  --mem=0
  --time=02:00:00
  --job-name=stage_nemo_rl_candidate
  "--output=${LOG_DIRECTORY}/slurm-%j.out"
  "--error=${LOG_DIRECTORY}/slurm-%j.err"
  "--chdir=${SCRIPT_ROOT}"
  "--export=${EXPORT_NAMES}"
)
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_ARGUMENTS=(--test-only "${SBATCH_ARGUMENTS[@]}")
fi
readonly SBATCH_ARGUMENTS

clean_git -C "${SCRIPT_ROOT}" show \
  "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}" |
  /usr/bin/env -i \
    HOME="${HOME_DIRECTORY}" \
    PATH=/usr/bin:/bin \
    SOURCE_IMAGE="${SOURCE_IMAGE}" \
    SOURCE_COMMIT="${SOURCE_COMMIT}" \
    SOURCE_LOCK_SHA256="${SOURCE_LOCK_SHA256}" \
    SOURCE_PYPROJECT_SHA256="${SOURCE_PYPROJECT_SHA256}" \
    OUTPUT_PREFIX="${OUTPUT_PREFIX}" \
    CONTAINER_DIR="${CONTAINER_DIR}" \
    TOOLING_WORKTREE="${SCRIPT_ROOT}" \
    TOOLING_COMMIT="${EXPECTED_TOOLING_SHA}" \
    SLURM_CONF="${OCI_SLURM_CONF}" \
    "${SBATCH_COMMAND}" "${SBATCH_ARGUMENTS[@]}"
