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
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_nemo_rl_candidate
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/container-validation/nightly-stage/logs/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/container-validation/nightly-stage/logs/slurm-%j.err

set -euo pipefail

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

hash_file() {
  local file=$1
  local hash_output
  local hash_value

  hash_output=$(sha256sum "${file}")
  hash_value=${hash_output%%[[:space:]]*}
  require_sha256 'computed SHA-256' "${hash_value}"
  echo "${hash_value}"
}

file_identity() {
  local file=$1
  local identity

  if identity=$(stat -Lc '%d:%i' "${file}" 2>/dev/null); then
    :
  elif identity=$(stat -f '%d:%i' "${file}" 2>/dev/null); then
    :
  else
    return 1
  fi
  if [[ ! "${identity}" =~ ^[0-9]+:[0-9]+$ ]]; then
    return 1
  fi
  echo "${identity}"
}

file_mode() {
  local file=$1
  local mode

  if mode=$(stat -Lc '%a' "${file}" 2>/dev/null); then
    :
  elif mode=$(stat -f '%Lp' "${file}" 2>/dev/null); then
    :
  else
    return 1
  fi
  if [[ ! "${mode}" =~ ^[0-7]{3,4}$ ]]; then
    return 1
  fi
  echo "${mode}"
}

receipt_value() {
  local key=$1
  local receipt=$2

  awk -v expected_key="${key}" '
    index($0, expected_key "=") == 1 {
      count += 1
      value = substr($0, length(expected_key) + 2)
    }
    END {
      if (count != 1) {
        exit 3
      }
      print value
    }
  ' "${receipt}"
}

require_receipt_value() {
  local receipt=$1
  local key=$2
  local expected=$3
  local actual

  actual=$(receipt_value "${key}" "${receipt}") ||
    die "candidate receipt has a missing or duplicate ${key} field: ${receipt}"
  if [[ "${actual}" != "${expected}" ]]; then
    die "candidate receipt ${key} mismatch: ${receipt}"
  fi
}

require_receipt_regex() {
  local receipt=$1
  local key=$2
  local expected_regex=$3
  local actual

  actual=$(receipt_value "${key}" "${receipt}") ||
    die "candidate receipt has a missing or duplicate ${key} field: ${receipt}"
  if [[ ! "${actual}" =~ ${expected_regex} ]]; then
    die "candidate receipt ${key} is malformed: ${receipt}"
  fi
}

validate_receipt_schema() {
  local receipt=$1

  awk -F= '
    BEGIN {
      allowed["receipt_version"] = 1
      allowed["artifact_kind"] = 1
      allowed["validation_state"] = 1
      allowed["requested_source_ref"] = 1
      allowed["requested_source_registry"] = 1
      allowed["requested_source_repository"] = 1
      allowed["requested_source_reference"] = 1
      allowed["requested_source_reference_kind"] = 1
      allowed["expected_source_commit"] = 1
      allowed["expected_source_lock_sha256"] = 1
      allowed["expected_source_pyproject_sha256"] = 1
      allowed["imported_enroot_uri"] = 1
      allowed["image_sha256"] = 1
      allowed["image_filename"] = 1
      allowed["tooling_blob_sha256"] = 1
      allowed["tooling_commit"] = 1
      allowed["tooling_worktree"] = 1
      allowed["slurm_job_id"] = 1
      allowed["slurm_job_name"] = 1
      allowed["slurm_node"] = 1
      allowed["stage_host"] = 1
      allowed["stage_started_at_utc"] = 1
      allowed["stage_finished_at_utc"] = 1
      allowed["enroot_version"] = 1
    }
    !($1 in allowed) {
      exit 3
    }
    {
      count[$1] += 1
      lines += 1
    }
    END {
      if (lines != 24) {
        exit 3
      }
      for (key in allowed) {
        if (count[key] != 1) {
          exit 3
        }
      }
    }
  ' "${receipt}" || die "candidate receipt schema is invalid: ${receipt}"
}

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to an explicit registry/repository:tag; Enroot 3.5 digest references are unsupported}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the exact NeMo-RL source commit}"
: "${SOURCE_LOCK_SHA256:?Set SOURCE_LOCK_SHA256 to the exact uv.lock SHA-256}"
: "${SOURCE_PYPROJECT_SHA256:?Set SOURCE_PYPROJECT_SHA256 to the exact pyproject.toml SHA-256}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe candidate name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to an absolute shared-storage path}"
: "${SLURM_JOB_ID:?This script must run as a Slurm job}"
: "${TOOLING_WORKTREE:?Set TOOLING_WORKTREE to the clean staging-tool checkout}"
: "${TOOLING_COMMIT:?Set TOOLING_COMMIT to the exact staging-tool commit}"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  die 'SOURCE_COMMIT must be exactly 40 lowercase hexadecimal characters'
fi
require_sha256 'SOURCE_LOCK_SHA256' "${SOURCE_LOCK_SHA256}"
require_sha256 'SOURCE_PYPROJECT_SHA256' "${SOURCE_PYPROJECT_SHA256}"
if [[ ! "${OUTPUT_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  die 'OUTPUT_PREFIX is not filesystem-safe'
fi
if [[ "${CONTAINER_DIR}" != /* ]]; then
  die 'CONTAINER_DIR must be absolute'
fi
if [[ ! "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
  die 'SLURM_JOB_ID must contain only decimal digits'
fi
if [[ "${TOOLING_WORKTREE}" != /* || "${TOOLING_WORKTREE}" == *$'\n'* ||
  "${TOOLING_WORKTREE}" == *$'\r'* ]]; then
  die 'TOOLING_WORKTREE must be an absolute single-line path'
fi
if [[ ! "${TOOLING_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  die 'TOOLING_COMMIT must be exactly 40 lowercase hexadecimal characters'
fi
readonly TOOLING_COMMIT

if [[ "${SOURCE_IMAGE}" != */* || "${SOURCE_IMAGE}" == *://* ||
  "${SOURCE_IMAGE}" == *'#'* || "${SOURCE_IMAGE}" =~ [[:space:]] ]]; then
  die 'SOURCE_IMAGE must be an uncredentialed registry/repository reference'
fi
readonly SOURCE_REGISTRY=${SOURCE_IMAGE%%/*}
readonly REPOSITORY_AND_REFERENCE=${SOURCE_IMAGE#*/}
if [[ ! "${SOURCE_REGISTRY}" =~ ^[a-z0-9][a-z0-9.-]*(:[0-9]+)?$ ]]; then
  die 'SOURCE_IMAGE registry is malformed'
fi

SOURCE_REFERENCE_KIND=
SOURCE_REPOSITORY=
SOURCE_REFERENCE=
if [[ "${REPOSITORY_AND_REFERENCE}" == *@sha256:* ]]; then
  SOURCE_REFERENCE_KIND=digest
  SOURCE_REPOSITORY=${REPOSITORY_AND_REFERENCE%@sha256:*}
  SOURCE_REFERENCE=${REPOSITORY_AND_REFERENCE##*@sha256:}
  require_sha256 'SOURCE_IMAGE digest' "${SOURCE_REFERENCE}"
elif [[ "${REPOSITORY_AND_REFERENCE}" == *:* ]]; then
  SOURCE_REFERENCE_KIND=tag
  SOURCE_REPOSITORY=${REPOSITORY_AND_REFERENCE%:*}
  SOURCE_REFERENCE=${REPOSITORY_AND_REFERENCE##*:}
  if [[ ! "${SOURCE_REFERENCE}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    die 'SOURCE_IMAGE tag is malformed'
  fi
else
  die 'SOURCE_IMAGE must include an explicit tag or sha256 digest'
fi
readonly SOURCE_REFERENCE_KIND SOURCE_REPOSITORY SOURCE_REFERENCE
if [[ ! "${SOURCE_REPOSITORY}" =~ ^[a-z0-9]+([-._][a-z0-9]+)*(/[a-z0-9]+([-._][a-z0-9]+)*)*$ ]]; then
  die 'SOURCE_IMAGE repository is malformed'
fi

readonly REGISTRY_HOST=${SOURCE_REGISTRY%%:*}
readonly ENROOT_IMAGE_URI="docker://${SOURCE_REGISTRY}#${REPOSITORY_AND_REFERENCE}"
export NO_PROXY="${REGISTRY_HOST},${SOURCE_REGISTRY}"
export no_proxy="${NO_PROXY}"

readonly PUBLISH_ROOT=/lustre
readonly SCRATCH_BASE=/raid/scratch/nemo-rl-image-stage
readonly SCRATCH_ROOT=${SCRATCH_BASE%/*}
readonly SCRATCH_DIRECTORY_NAME=job-${SLURM_JOB_ID}
readonly SCRATCH_DIRECTORY=${SCRATCH_BASE}/${SCRATCH_DIRECTORY_NAME}
readonly SCRATCH_IMAGE=${SCRATCH_DIRECTORY}/candidate.sqsh.partial
readonly SCRATCH_OWNER_FILE=${SCRATCH_DIRECTORY}/.stage-owner
export ENROOT_CACHE_PATH=${SCRATCH_DIRECTORY}/enroot-cache
export ENROOT_DATA_PATH=${SCRATCH_DIRECTORY}/enroot-data
export ENROOT_RUNTIME_PATH=${SCRATCH_DIRECTORY}/enroot-runtime
export TMPDIR=${SCRATCH_DIRECTORY}/tmp
export TMP=${TMPDIR}
export TEMP=${TMPDIR}

CONTAINER_PARTIAL_FILE=
METADATA_PARTIAL_FILE=
OUTPUT_FILE=
METADATA_FILE=
OUTPUT_ABSOLUTE_FILE=
METADATA_ABSOLUTE_FILE=
PUBLISHED_IMAGE=0
PUBLISHED_RECEIPT=0
SCRATCH_OWNED=0
SCRATCH_OWNER_TOKEN=
CONTAINER_PARTIAL_IDENTITY=
METADATA_PARTIAL_IDENTITY=
ORPHAN_IMAGE_IDENTITY=
SCRATCH_DIRECTORY_IDENTITY=
CONTAINER_DIRECTORY_IDENTITY=
CONTAINER_DIRECTORY_PINNED=0

require_container_directory_path_identity() {
  local actual_identity
  local actual_path

  if [[ ! -d "${RESOLVED_CONTAINER_DIR}" || -L "${RESOLVED_CONTAINER_DIR}" ]]; then
    die "container directory identity changed: ${RESOLVED_CONTAINER_DIR}"
  fi
  actual_identity=$(file_identity "${RESOLVED_CONTAINER_DIR}") ||
    die "could not identify container directory: ${RESOLVED_CONTAINER_DIR}"
  actual_path=$(cd "${RESOLVED_CONTAINER_DIR}" && pwd -P) ||
    die "container directory is no longer accessible: ${RESOLVED_CONTAINER_DIR}"
  if [[ "${actual_identity}" != "${CONTAINER_DIRECTORY_IDENTITY}" ||
    "${actual_path}" != "${RESOLVED_CONTAINER_DIR}" ]]; then
    die "container directory identity changed: ${RESOLVED_CONTAINER_DIR}"
  fi
}

require_pinned_container_directory_identity() {
  local actual_identity

  if ((CONTAINER_DIRECTORY_PINNED != 1)); then
    die 'container directory is not pinned as the process working directory'
  fi
  actual_identity=$(file_identity .) ||
    die 'could not identify the pinned container directory'
  if [[ "${actual_identity}" != "${CONTAINER_DIRECTORY_IDENTITY}" ]]; then
    die 'pinned container directory identity changed'
  fi
}

cleanup() {
  local exit_status=$?
  local cleanup_status=0
  local actual_owner_token=
  local actual_identity=
  local actual_scratch_identity=
  local actual_quarantine_identity=
  local published_receipt_identity=
  local rollback_hash=
  local scratch_quarantine_directory=
  local scratch_quarantine_identity=
  local quarantined_scratch_directory=

  trap - EXIT
  set +e
  if ((PUBLISHED_IMAGE == 1 && PUBLISHED_RECEIPT == 0)) &&
    [[ -n "${METADATA_PARTIAL_IDENTITY}" ]]; then
    published_receipt_identity=$(file_identity "${METADATA_FILE}")
    if [[ "${published_receipt_identity}" == "${METADATA_PARTIAL_IDENTITY}" &&
      -f "${METADATA_FILE}" && ! -L "${METADATA_FILE}" ]]; then
      PUBLISHED_RECEIPT=1
    fi
  fi
  if ((PUBLISHED_IMAGE == 1 && PUBLISHED_RECEIPT == 0)); then
    actual_identity=$(file_identity "${OUTPUT_FILE}")
    if [[ -n "${CONTAINER_PARTIAL_IDENTITY}" &&
      "${actual_identity}" == "${CONTAINER_PARTIAL_IDENTITY}" &&
      -f "${OUTPUT_FILE}" && ! -L "${OUTPUT_FILE}" ]]; then
      rollback_hash=$(hash_file "${OUTPUT_FILE}")
      if [[ "${rollback_hash}" == "${IMAGE_SHA256}" ]]; then
        rm -f "${OUTPUT_FILE}" || cleanup_status=$?
      else
        echo "ERROR: refusing to remove changed published image: ${OUTPUT_FILE}" >&2
        cleanup_status=96
      fi
    else
      echo "ERROR: refusing to remove replaced published image: ${OUTPUT_FILE}" >&2
      cleanup_status=96
    fi
  fi
  if [[ -n "${CONTAINER_PARTIAL_FILE}" ]]; then
    actual_identity=$(file_identity "${CONTAINER_PARTIAL_FILE}")
    if [[ -n "${CONTAINER_PARTIAL_IDENTITY}" &&
      "${actual_identity}" == "${CONTAINER_PARTIAL_IDENTITY}" &&
      -f "${CONTAINER_PARTIAL_FILE}" && ! -L "${CONTAINER_PARTIAL_FILE}" ]]; then
      rm -f "${CONTAINER_PARTIAL_FILE}" || cleanup_status=$?
    elif [[ -e "${CONTAINER_PARTIAL_FILE}" || -L "${CONTAINER_PARTIAL_FILE}" ]]; then
      echo "ERROR: refusing to remove replaced image partial: ${CONTAINER_PARTIAL_FILE}" >&2
      cleanup_status=95
    fi
  fi
  if [[ -n "${METADATA_PARTIAL_FILE}" ]]; then
    actual_identity=$(file_identity "${METADATA_PARTIAL_FILE}")
    if [[ -n "${METADATA_PARTIAL_IDENTITY}" &&
      "${actual_identity}" == "${METADATA_PARTIAL_IDENTITY}" &&
      -f "${METADATA_PARTIAL_FILE}" && ! -L "${METADATA_PARTIAL_FILE}" ]]; then
      rm -f "${METADATA_PARTIAL_FILE}" || cleanup_status=$?
    elif [[ -e "${METADATA_PARTIAL_FILE}" || -L "${METADATA_PARTIAL_FILE}" ]]; then
      echo "ERROR: refusing to remove replaced receipt partial: ${METADATA_PARTIAL_FILE}" >&2
      cleanup_status=95
    fi
  fi
  if ((SCRATCH_OWNED == 1)); then
    scratch_quarantine_directory=$(
      mktemp -d "${SCRATCH_BASE}/.cleanup-${SLURM_JOB_ID}.XXXXXX"
    )
    if [[ -n "${scratch_quarantine_directory}" &&
      -d "${scratch_quarantine_directory}" &&
      ! -L "${scratch_quarantine_directory}" ]]; then
      chmod 700 "${scratch_quarantine_directory}" || cleanup_status=97
      scratch_quarantine_identity=$(file_identity "${scratch_quarantine_directory}")
      quarantined_scratch_directory=${scratch_quarantine_directory}/${SCRATCH_DIRECTORY_NAME}
      if mv "${SCRATCH_DIRECTORY}" "${scratch_quarantine_directory}/"; then
        actual_scratch_identity=$(file_identity "${quarantined_scratch_directory}")
        if [[ -f "${quarantined_scratch_directory}/.stage-owner" &&
          ! -L "${quarantined_scratch_directory}/.stage-owner" ]]; then
          IFS= read -r actual_owner_token \
            <"${quarantined_scratch_directory}/.stage-owner" || true
        fi
        if [[ -n "${SCRATCH_OWNER_TOKEN}" &&
          -n "${SCRATCH_DIRECTORY_IDENTITY}" &&
          -d "${quarantined_scratch_directory}" &&
          ! -L "${quarantined_scratch_directory}" &&
          "${actual_scratch_identity}" == "${SCRATCH_DIRECTORY_IDENTITY}" &&
          "${actual_owner_token}" == "${SCRATCH_OWNER_TOKEN}" ]]; then
          rm -rf "${quarantined_scratch_directory}" || cleanup_status=97
        else
          echo "ERROR: preserving unverified quarantined scratch: ${quarantined_scratch_directory}" >&2
          cleanup_status=97
        fi
      else
        echo "ERROR: could not quarantine job scratch: ${SCRATCH_DIRECTORY}" >&2
        cleanup_status=97
      fi
      actual_quarantine_identity=$(file_identity "${scratch_quarantine_directory}")
      if [[ -n "${scratch_quarantine_identity}" &&
        "${actual_quarantine_identity}" == "${scratch_quarantine_identity}" ]]; then
        if ! rmdir "${scratch_quarantine_directory}" 2>/dev/null &&
          ((cleanup_status == 0)); then
          cleanup_status=97
        fi
      else
        echo "ERROR: refusing to remove replaced scratch quarantine: ${scratch_quarantine_directory}" >&2
        cleanup_status=97
      fi
    else
      echo "ERROR: could not allocate private scratch quarantine under ${SCRATCH_BASE}" >&2
      cleanup_status=97
    fi
  fi
  if ((exit_status == 0 && cleanup_status != 0)); then
    exit_status=${cleanup_status}
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

command -v enroot >/dev/null || die 'enroot is not available'
command -v sha256sum >/dev/null || die 'sha256sum is not available'
command -v awk >/dev/null || die 'awk is not available'
command -v hostname >/dev/null || die 'hostname is not available'
command -v mktemp >/dev/null || die 'mktemp is not available'
command -v git >/dev/null || die 'git is not available'
command -v stat >/dev/null || die 'stat is not available'

readonly TOOLING_SCRIPT_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/stage_enroot_image.sbatch
RESOLVED_TOOLING_WORKTREE=$(cd "${TOOLING_WORKTREE}" && pwd -P) ||
  die "TOOLING_WORKTREE is not accessible: ${TOOLING_WORKTREE}"
readonly RESOLVED_TOOLING_WORKTREE
TOOLING_TOPLEVEL=$(git -C "${RESOLVED_TOOLING_WORKTREE}" rev-parse --show-toplevel) ||
  die "TOOLING_WORKTREE is not a Git checkout: ${RESOLVED_TOOLING_WORKTREE}"
TOOLING_TOPLEVEL=$(cd "${TOOLING_TOPLEVEL}" && pwd -P) ||
  die "tooling Git top-level is not accessible: ${TOOLING_TOPLEVEL}"
readonly TOOLING_TOPLEVEL
if [[ "${TOOLING_TOPLEVEL}" != "${RESOLVED_TOOLING_WORKTREE}" ]]; then
  die 'TOOLING_WORKTREE must name the Git checkout root'
fi
ACTUAL_TOOLING_COMMIT=$(
  git -C "${RESOLVED_TOOLING_WORKTREE}" rev-parse --verify HEAD
) || die 'could not resolve tooling checkout HEAD'
readonly ACTUAL_TOOLING_COMMIT
if [[ "${ACTUAL_TOOLING_COMMIT}" != "${TOOLING_COMMIT}" ]]; then
  die 'TOOLING_COMMIT does not match the tooling checkout HEAD'
fi
TOOLING_STATUS=$(
  git -C "${RESOLVED_TOOLING_WORKTREE}" status --porcelain=v1 \
    --untracked-files=all --ignore-submodules=none
) || die 'could not inspect tooling checkout status'
readonly TOOLING_STATUS
if [[ -n "${TOOLING_STATUS}" ]]; then
  die 'TOOLING_WORKTREE must be clean, including untracked files'
fi
readonly TOOLING_SCRIPT_FILE=${RESOLVED_TOOLING_WORKTREE}/${TOOLING_SCRIPT_RELATIVE_PATH}
if [[ ! -f "${TOOLING_SCRIPT_FILE}" || -L "${TOOLING_SCRIPT_FILE}" ]]; then
  die "committed tooling script is not a regular file: ${TOOLING_SCRIPT_FILE}"
fi
COMMITTED_TOOLING_GIT_BLOB=$(
  git -C "${RESOLVED_TOOLING_WORKTREE}" rev-parse --verify \
    "${TOOLING_COMMIT}:${TOOLING_SCRIPT_RELATIVE_PATH}"
) || die 'could not resolve the committed tooling script blob'
readonly COMMITTED_TOOLING_GIT_BLOB
COMMITTED_TOOLING_OBJECT_TYPE=$(
  git -C "${RESOLVED_TOOLING_WORKTREE}" cat-file -t \
    "${COMMITTED_TOOLING_GIT_BLOB}"
) || die 'could not inspect the committed tooling object type'
readonly COMMITTED_TOOLING_OBJECT_TYPE
if [[ "${COMMITTED_TOOLING_OBJECT_TYPE}" != blob ]]; then
  die 'committed tooling path does not resolve to a Git blob'
fi
WORKTREE_TOOLING_GIT_BLOB=$(
  git -C "${RESOLVED_TOOLING_WORKTREE}" hash-object "${TOOLING_SCRIPT_FILE}"
) || die 'could not hash the tooling worktree script'
readonly WORKTREE_TOOLING_GIT_BLOB
if [[ "${WORKTREE_TOOLING_GIT_BLOB}" != "${COMMITTED_TOOLING_GIT_BLOB}" ]]; then
  die 'tooling worktree script bytes do not match the committed blob'
fi
EXECUTED_TOOLING_GIT_BLOB=$(
  git -C "${RESOLVED_TOOLING_WORKTREE}" hash-object "$0"
) || die 'could not hash the executed Slurm script'
readonly EXECUTED_TOOLING_GIT_BLOB
if [[ "${EXECUTED_TOOLING_GIT_BLOB}" != "${COMMITTED_TOOLING_GIT_BLOB}" ]]; then
  die 'executed Slurm script bytes do not match the committed tooling script'
fi
TOOLING_BLOB_SHA256=$(hash_file "$0")
readonly TOOLING_BLOB_SHA256

if [[ ! -d "${PUBLISH_ROOT}" || -L "${PUBLISH_ROOT}" ]]; then
  die "publish root must be an existing non-symlink directory: ${PUBLISH_ROOT}"
fi
RESOLVED_PUBLISH_ROOT=$(cd "${PUBLISH_ROOT}" && pwd -P) ||
  die "publish root is not accessible: ${PUBLISH_ROOT}"
readonly RESOLVED_PUBLISH_ROOT
if [[ "${RESOLVED_PUBLISH_ROOT}" != "${PUBLISH_ROOT}" ]]; then
  die "publish root must be canonical: ${PUBLISH_ROOT}"
fi
if [[ ! -d "${CONTAINER_DIR}" || -L "${CONTAINER_DIR}" ]]; then
  die 'CONTAINER_DIR must be an existing non-symlink directory'
fi
RESOLVED_CONTAINER_DIR=$(cd "${CONTAINER_DIR}" && pwd -P) ||
  die "CONTAINER_DIR is not accessible: ${CONTAINER_DIR}"
readonly RESOLVED_CONTAINER_DIR
if [[ "${RESOLVED_CONTAINER_DIR}" != "${CONTAINER_DIR}" ]]; then
  die 'CONTAINER_DIR must be a canonical path without symlinks or traversal'
fi
case "${RESOLVED_CONTAINER_DIR}" in
"${RESOLVED_PUBLISH_ROOT}"/*) ;;
*) die "CONTAINER_DIR must resolve under ${RESOLVED_PUBLISH_ROOT}" ;;
esac
if [[ ! -O "${RESOLVED_CONTAINER_DIR}" ]]; then
  die "CONTAINER_DIR must be owned by the staging user: ${RESOLVED_CONTAINER_DIR}"
fi
CONTAINER_DIRECTORY_MODE=$(file_mode "${RESOLVED_CONTAINER_DIR}") ||
  die "could not inspect CONTAINER_DIR permissions: ${RESOLVED_CONTAINER_DIR}"
readonly CONTAINER_DIRECTORY_MODE
if ((8#${CONTAINER_DIRECTORY_MODE} & 8#022)); then
  die "CONTAINER_DIR must not be group- or other-writable: ${RESOLVED_CONTAINER_DIR}"
fi
CONTAINER_DIRECTORY_IDENTITY=$(file_identity "${RESOLVED_CONTAINER_DIR}") ||
  die "could not identify container directory: ${RESOLVED_CONTAINER_DIR}"
readonly CONTAINER_DIRECTORY_IDENTITY

STAGE_STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
readonly STAGE_STARTED_AT_UTC
ENROOT_VERSION=$(enroot version)
readonly ENROOT_VERSION
if [[ ! "${ENROOT_VERSION}" =~ ^3\.5\.[0-9]+$ ]]; then
  die "unsupported Enroot version '${ENROOT_VERSION}'; this stage requires 3.5.x import semantics"
fi
if [[ "${SOURCE_REFERENCE_KIND}" != tag ]]; then
  die 'Enroot 3.5.x cannot truthfully import this digest reference; use an explicit unique build tag and enforce registry immutability where available'
fi

if [[ ! -d "${SCRATCH_ROOT}" || -L "${SCRATCH_ROOT}" ]]; then
  die "scratch root must be an existing non-symlink directory: ${SCRATCH_ROOT}"
fi
RESOLVED_SCRATCH_ROOT=$(cd "${SCRATCH_ROOT}" && pwd -P) ||
  die "scratch root is not accessible: ${SCRATCH_ROOT}"
readonly RESOLVED_SCRATCH_ROOT
if [[ "${RESOLVED_SCRATCH_ROOT}" != "${SCRATCH_ROOT}" ]]; then
  die "scratch root must be canonical: ${SCRATCH_ROOT}"
fi
if [[ ! -e "${SCRATCH_BASE}" && ! -L "${SCRATCH_BASE}" ]]; then
  mkdir -m 700 "${SCRATCH_BASE}" 2>/dev/null || true
fi
if [[ ! -d "${SCRATCH_BASE}" || -L "${SCRATCH_BASE}" || ! -O "${SCRATCH_BASE}" ]]; then
  die "scratch base must be an owned non-symlink directory: ${SCRATCH_BASE}"
fi
RESOLVED_SCRATCH_BASE=$(cd "${SCRATCH_BASE}" && pwd -P) ||
  die "scratch base is not accessible: ${SCRATCH_BASE}"
readonly RESOLVED_SCRATCH_BASE
if [[ "${RESOLVED_SCRATCH_BASE}" != "${SCRATCH_BASE}" ]]; then
  die "scratch base must be canonical: ${SCRATCH_BASE}"
fi
if ! mkdir -m 700 "${SCRATCH_DIRECTORY}"; then
  die "could not atomically claim job scratch: ${SCRATCH_DIRECTORY}"
fi
SCRATCH_OWNED=1
SCRATCH_DIRECTORY_IDENTITY=$(file_identity "${SCRATCH_DIRECTORY}") ||
  die "could not identify claimed job scratch: ${SCRATCH_DIRECTORY}"
readonly SCRATCH_DIRECTORY_IDENTITY
SCRATCH_OWNER_TOKEN="job-${SLURM_JOB_ID}:pid-$$:tooling-${TOOLING_BLOB_SHA256}"
readonly SCRATCH_OWNER_TOKEN
(set -o noclobber; printf '%s\n' "${SCRATCH_OWNER_TOKEN}" >"${SCRATCH_OWNER_FILE}") ||
  die "could not create scratch ownership marker: ${SCRATCH_OWNER_FILE}"
chmod 600 "${SCRATCH_OWNER_FILE}"
mkdir "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" \
  "${ENROOT_RUNTIME_PATH}" "${TMPDIR}"
chmod 700 "${ENROOT_CACHE_PATH}" \
  "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${TMPDIR}"

echo 'candidate_status=importing'
echo "requested_source_ref=${SOURCE_IMAGE}"
echo "expected_source_commit=${SOURCE_COMMIT}"
echo "scratch_directory=${SCRATCH_DIRECTORY}"
enroot import -o "${SCRATCH_IMAGE}" "${ENROOT_IMAGE_URI}"

IMAGE_SHA256=$(hash_file "${SCRATCH_IMAGE}")
readonly IMAGE_SHA256
readonly IMAGE_FILENAME=${OUTPUT_PREFIX}_sha256-${IMAGE_SHA256}.sqsh
require_container_directory_path_identity
cd "${RESOLVED_CONTAINER_DIR}" ||
  die "could not pin container directory: ${RESOLVED_CONTAINER_DIR}"
CONTAINER_DIRECTORY_PINNED=1
require_pinned_container_directory_identity
OUTPUT_FILE=./${IMAGE_FILENAME}
METADATA_FILE=${OUTPUT_FILE}.metadata.txt
OUTPUT_ABSOLUTE_FILE=${RESOLVED_CONTAINER_DIR}/${IMAGE_FILENAME}
METADATA_ABSOLUTE_FILE=${OUTPUT_ABSOLUTE_FILE}.metadata.txt
readonly OUTPUT_FILE METADATA_FILE OUTPUT_ABSOLUTE_FILE METADATA_ABSOLUTE_FILE

validate_candidate_receipt() {
  local receipt=$1

  validate_receipt_schema "${receipt}"
  require_receipt_value "${receipt}" receipt_version 1
  require_receipt_value "${receipt}" artifact_kind nemo_rl_enroot_candidate
  require_receipt_value "${receipt}" validation_state unvalidated_candidate
  require_receipt_value "${receipt}" requested_source_ref "${SOURCE_IMAGE}"
  require_receipt_value "${receipt}" requested_source_registry "${SOURCE_REGISTRY}"
  require_receipt_value "${receipt}" requested_source_repository "${SOURCE_REPOSITORY}"
  require_receipt_value "${receipt}" requested_source_reference "${SOURCE_REFERENCE}"
  require_receipt_value "${receipt}" requested_source_reference_kind \
    "${SOURCE_REFERENCE_KIND}"
  require_receipt_value "${receipt}" expected_source_commit "${SOURCE_COMMIT}"
  require_receipt_value "${receipt}" expected_source_lock_sha256 \
    "${SOURCE_LOCK_SHA256}"
  require_receipt_value "${receipt}" expected_source_pyproject_sha256 \
    "${SOURCE_PYPROJECT_SHA256}"
  require_receipt_value "${receipt}" imported_enroot_uri "${ENROOT_IMAGE_URI}"
  require_receipt_value "${receipt}" image_sha256 "${IMAGE_SHA256}"
  require_receipt_value "${receipt}" image_filename "${IMAGE_FILENAME}"
  require_receipt_value "${receipt}" tooling_blob_sha256 \
    "${TOOLING_BLOB_SHA256}"
  require_receipt_value "${receipt}" tooling_commit "${TOOLING_COMMIT}"
  require_receipt_value "${receipt}" tooling_worktree \
    "${RESOLVED_TOOLING_WORKTREE}"
  require_receipt_regex "${receipt}" slurm_job_id '^[0-9]+$'
  require_receipt_regex "${receipt}" slurm_job_name '^.+$'
  require_receipt_regex "${receipt}" slurm_node '^.+$'
  require_receipt_regex "${receipt}" stage_host '^.+$'
  require_receipt_regex "${receipt}" stage_started_at_utc \
    '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
  require_receipt_regex "${receipt}" stage_finished_at_utc \
    '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
  require_receipt_regex "${receipt}" enroot_version '^.+$'
}

validate_generated_receipt() {
  local receipt=$1

  validate_candidate_receipt "${receipt}"
  require_receipt_value "${receipt}" slurm_job_id "${SLURM_JOB_ID}"
  require_receipt_value "${receipt}" slurm_job_name "${SLURM_JOB_NAME_VALUE}"
  require_receipt_value "${receipt}" slurm_node "${SLURM_NODE}"
  require_receipt_value "${receipt}" stage_host "${STAGE_HOST}"
  require_receipt_value "${receipt}" stage_started_at_utc "${STAGE_STARTED_AT_UTC}"
  require_receipt_value "${receipt}" stage_finished_at_utc "${STAGE_FINISHED_AT_UTC}"
  require_receipt_value "${receipt}" enroot_version "${ENROOT_VERSION}"
}

validate_existing_candidate() {
  local existing_hash

  if [[ ! -f "${OUTPUT_FILE}" || -L "${OUTPUT_FILE}" ||
    ! -f "${METADATA_FILE}" || -L "${METADATA_FILE}" ]]; then
    die "immutable candidate collision is incomplete or not a regular file: ${OUTPUT_FILE}"
  fi
  existing_hash=$(hash_file "${OUTPUT_FILE}")
  if [[ "${existing_hash}" != "${IMAGE_SHA256}" ]]; then
    die "immutable candidate bytes do not match their content address: ${OUTPUT_FILE}"
  fi
  validate_candidate_receipt "${METADATA_FILE}"
}

RECOVERING_ORPHAN_IMAGE=0
if [[ -e "${OUTPUT_FILE}" || -L "${OUTPUT_FILE}" ]] &&
  [[ -e "${METADATA_FILE}" || -L "${METADATA_FILE}" ]]; then
  validate_existing_candidate
  require_container_directory_path_identity
  echo 'candidate_status=reused_verified'
  echo "container=${OUTPUT_ABSOLUTE_FILE}"
  echo "metadata=${METADATA_ABSOLUTE_FILE}"
  exit 0
fi
if [[ -e "${METADATA_FILE}" || -L "${METADATA_FILE}" ]]; then
  die "immutable candidate receipt exists without its image: ${METADATA_FILE}"
fi
if [[ -e "${OUTPUT_FILE}" || -L "${OUTPUT_FILE}" ]]; then
  if [[ ! -f "${OUTPUT_FILE}" || -L "${OUTPUT_FILE}" ]]; then
    die "orphan candidate image is not a regular file: ${OUTPUT_FILE}"
  fi
  ORPHAN_IMAGE_IDENTITY=$(file_identity "${OUTPUT_FILE}") ||
    die "could not identify orphan candidate image: ${OUTPUT_FILE}"
  readonly ORPHAN_IMAGE_IDENTITY
  ORPHAN_IMAGE_SHA256=$(hash_file "${OUTPUT_FILE}")
  readonly ORPHAN_IMAGE_SHA256
  if [[ "${ORPHAN_IMAGE_SHA256}" != "${IMAGE_SHA256}" ||
    "$(file_identity "${OUTPUT_FILE}")" != "${ORPHAN_IMAGE_IDENTITY}" ]]; then
    die "orphan candidate bytes do not match their content address: ${OUTPUT_FILE}"
  fi
  RECOVERING_ORPHAN_IMAGE=1
fi

if ((RECOVERING_ORPHAN_IMAGE == 0)); then
  require_pinned_container_directory_identity
  CONTAINER_PARTIAL_FILE=$(
    mktemp "./.${IMAGE_FILENAME}.partial.job-${SLURM_JOB_ID}.XXXXXX"
  )
  CONTAINER_PARTIAL_IDENTITY=$(file_identity "${CONTAINER_PARTIAL_FILE}") ||
    die 'could not identify the exclusive candidate image partial'
  readonly CONTAINER_PARTIAL_IDENTITY
  require_pinned_container_directory_identity
  cp "${SCRATCH_IMAGE}" "${CONTAINER_PARTIAL_FILE}"
  if [[ "$(file_identity "${CONTAINER_PARTIAL_FILE}")" != "${CONTAINER_PARTIAL_IDENTITY}" ]]; then
    die 'candidate image partial was replaced while copying to shared storage'
  fi
  COPIED_IMAGE_SHA256=$(hash_file "${CONTAINER_PARTIAL_FILE}")
  readonly COPIED_IMAGE_SHA256
  if [[ "${COPIED_IMAGE_SHA256}" != "${IMAGE_SHA256}" ]]; then
    die 'candidate image changed while copying to shared storage'
  fi
  chmod a-w "${CONTAINER_PARTIAL_FILE}"
fi
require_pinned_container_directory_identity
METADATA_PARTIAL_FILE=$(
  mktemp "./.${IMAGE_FILENAME}.metadata.txt.partial.job-${SLURM_JOB_ID}.XXXXXX"
)
METADATA_PARTIAL_IDENTITY=$(file_identity "${METADATA_PARTIAL_FILE}") ||
  die 'could not identify the exclusive candidate receipt partial'
readonly METADATA_PARTIAL_IDENTITY
require_pinned_container_directory_identity

STAGE_HOST=$(hostname)
readonly STAGE_HOST
readonly SLURM_NODE=${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-unavailable}}
readonly SLURM_JOB_NAME_VALUE=${SLURM_JOB_NAME:-unavailable}
STAGE_FINISHED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
readonly STAGE_FINISHED_AT_UTC
{
  echo 'receipt_version=1'
  echo 'artifact_kind=nemo_rl_enroot_candidate'
  echo 'validation_state=unvalidated_candidate'
  echo "requested_source_ref=${SOURCE_IMAGE}"
  echo "requested_source_registry=${SOURCE_REGISTRY}"
  echo "requested_source_repository=${SOURCE_REPOSITORY}"
  echo "requested_source_reference=${SOURCE_REFERENCE}"
  echo "requested_source_reference_kind=${SOURCE_REFERENCE_KIND}"
  echo "expected_source_commit=${SOURCE_COMMIT}"
  echo "expected_source_lock_sha256=${SOURCE_LOCK_SHA256}"
  echo "expected_source_pyproject_sha256=${SOURCE_PYPROJECT_SHA256}"
  echo "imported_enroot_uri=${ENROOT_IMAGE_URI}"
  echo "image_sha256=${IMAGE_SHA256}"
  echo "image_filename=${IMAGE_FILENAME}"
  echo "tooling_blob_sha256=${TOOLING_BLOB_SHA256}"
  echo "tooling_commit=${TOOLING_COMMIT}"
  echo "tooling_worktree=${RESOLVED_TOOLING_WORKTREE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_job_name=${SLURM_JOB_NAME_VALUE}"
  echo "slurm_node=${SLURM_NODE}"
  echo "stage_host=${STAGE_HOST}"
  echo "stage_started_at_utc=${STAGE_STARTED_AT_UTC}"
  echo "stage_finished_at_utc=${STAGE_FINISHED_AT_UTC}"
  echo "enroot_version=${ENROOT_VERSION}"
} >"${METADATA_PARTIAL_FILE}"
if [[ "$(file_identity "${METADATA_PARTIAL_FILE}")" != "${METADATA_PARTIAL_IDENTITY}" ]]; then
  die 'candidate receipt partial was replaced while it was generated'
fi
validate_generated_receipt "${METADATA_PARTIAL_FILE}"
chmod a-w "${METADATA_PARTIAL_FILE}"

if ((RECOVERING_ORPHAN_IMAGE == 0)); then
  require_pinned_container_directory_identity
  ln "${CONTAINER_PARTIAL_FILE}" "${OUTPUT_FILE}" ||
    die "immutable candidate image appeared during publish: ${OUTPUT_FILE}"
  PUBLISHED_IMAGE=1
  if [[ "$(file_identity "${OUTPUT_FILE}")" != "${CONTAINER_PARTIAL_IDENTITY}" ]]; then
    die "published candidate image identity changed: ${OUTPUT_FILE}"
  fi
  require_pinned_container_directory_identity
elif [[ "$(file_identity "${OUTPUT_FILE}")" != "${ORPHAN_IMAGE_IDENTITY}" ]]; then
  die "orphan candidate image identity changed during recovery: ${OUTPUT_FILE}"
fi

require_pinned_container_directory_identity
ln "${METADATA_PARTIAL_FILE}" "${METADATA_FILE}" ||
  die "immutable candidate receipt appeared during publish: ${METADATA_FILE}"
PUBLISHED_RECEIPT=1
require_pinned_container_directory_identity
if [[ "$(file_identity "${METADATA_FILE}")" != "${METADATA_PARTIAL_IDENTITY}" ]]; then
  die "published candidate receipt identity changed: ${METADATA_FILE}"
fi
require_container_directory_path_identity

if ((RECOVERING_ORPHAN_IMAGE == 1)); then
  echo 'candidate_status=recovered_orphan_image'
else
  echo 'candidate_status=staged_unvalidated'
fi
echo "container=${OUTPUT_ABSOLUTE_FILE}"
echo "metadata=${METADATA_ABSOLUTE_FILE}"
ls -lh "${OUTPUT_FILE}" "${METADATA_FILE}"
