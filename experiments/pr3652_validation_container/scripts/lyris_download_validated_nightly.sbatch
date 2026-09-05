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

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --time=02:00:00
#SBATCH --job-name=coreai_dlalgo_llm-container_transfer.download_validated_nightly
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/lyris-download/slurm-%j.out
#SBATCH --error=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/lyris-download/slurm-%j.err

set -Eeuo pipefail

: "${SCRIPT_ROOT:?Set SCRIPT_ROOT to the absolute clean staging worktree root}"
: "${EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA to the staging worktree HEAD}"
readonly SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly DESTINATION=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly RCLONE_SOURCE=/home/sna/.local/libexec/nemo-rl/rclone/sha256-a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a/rclone
readonly EXPECTED_RCLONE_SHA256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a
readonly EXPECTED_COMPUTE_ARCHITECTURE=aarch64
readonly RCLONE_CONFIG=/home/sna/.config/rclone/rclone.conf
readonly PARTIAL=${DESTINATION}.partial.${SLURM_JOB_ID:-invalid}
readonly METADATA=${DESTINATION}.metadata.txt
readonly PARTIAL_METADATA=${METADATA}.partial.${SLURM_JOB_ID:-invalid}
readonly COMPLETION_MARKER=${DESTINATION}.complete
readonly PARTIAL_COMPLETION_MARKER=${COMPLETION_MARKER}.partial.${SLURM_JOB_ID:-invalid}
readonly VALIDATOR_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/validate_lyris_transferred_nightly.sh
readonly VALIDATOR=${SCRIPT_ROOT}/${VALIDATOR_RELATIVE_PATH}
readonly JOB_SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-validated-nightly/lyris-download-${SLURM_JOB_ID:-invalid}
readonly VALIDATOR_SNAPSHOT=${JOB_SCRATCH_DIRECTORY}/validate_lyris_transferred_nightly.sh
readonly RCLONE_DIRECTORY=${JOB_SCRATCH_DIRECTORY}/rclone
readonly RCLONE=${RCLONE_DIRECTORY}/rclone

CURRENT_STEP=validate-slurm-job-id

report_error() {
  local exit_status=$?

  trap - ERR
  printf 'Lyris validated-nightly download failed: step=%s exit=%s line=%s job=%s node=%s\n' \
    "${CURRENT_STEP}" \
    "${exit_status}" \
    "${BASH_LINENO[0]:-unknown}" \
    "${SLURM_JOB_ID:-unknown}" \
    "${SLURMD_NODENAME:-unknown}" >&2
}

cleanup() {
  local exit_status=$?
  local cleanup_exit_status

  trap - EXIT ERR
  if rm -f -- "${PARTIAL}" "${PARTIAL_METADATA}" "${PARTIAL_COMPLETION_MARKER}"; then
    :
  else
    cleanup_exit_status=$?
    echo 'Lyris validated-nightly cleanup failed: could not remove exact partial files' >&2
    if (( exit_status == 0 )); then
      exit_status=${cleanup_exit_status}
    fi
  fi
  if rm -rf -- "${JOB_SCRATCH_DIRECTORY}"; then
    :
  else
    cleanup_exit_status=$?
    echo 'Lyris validated-nightly cleanup failed: could not remove exact job scratch' >&2
    if (( exit_status == 0 )); then
      exit_status=${cleanup_exit_status}
    fi
  fi
  exit "${exit_status}"
}

require_command() {
  local command_name=$1

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Lyris validated-nightly preflight failed: required command is unavailable: %s\n' \
      "${command_name}" >&2
    return 127
  fi
}

require_regular_file() {
  local path=$1

  if [[ ! -f ${path} || -L ${path} ]]; then
    printf 'Lyris validated-nightly preflight failed: regular non-symlink file is required: %s\n' \
      "${path}" >&2
    return 1
  fi
}

require_regular_executable() {
  local path=$1

  if [[ ! -f ${path} || -L ${path} || ! -x ${path} ]]; then
    printf 'Lyris validated-nightly preflight failed: regular non-symlink executable is required: %s\n' \
      "${path}" >&2
    return 1
  fi
}

validate_tooling_root() {
  local committed_validator_blob
  local worktree_status
  local upstream_sha

  [[ ${SCRIPT_ROOT} = /* ]]
  test ! -L "${SCRIPT_ROOT}"
  [[ ${EXPECTED_TOOLING_SHA} =~ ^[0-9a-f]{40}$ ]]
  test "$(git -C "${SCRIPT_ROOT}" rev-parse --is-inside-work-tree)" = true
  test "$(git -C "${SCRIPT_ROOT}" rev-parse --show-toplevel)" = "${SCRIPT_ROOT}"
  test "$(git -C "${SCRIPT_ROOT}" rev-parse HEAD)" = "${EXPECTED_TOOLING_SHA}"
  worktree_status=$(git -C "${SCRIPT_ROOT}" status --porcelain)
  test -z "${worktree_status}"
  upstream_sha=$(git -C "${SCRIPT_ROOT}" rev-parse '@{upstream}')
  test "${upstream_sha}" = "${EXPECTED_TOOLING_SHA}"
  require_regular_executable "${VALIDATOR}"
  committed_validator_blob=$(git -C "${SCRIPT_ROOT}" rev-parse --verify \
    "${EXPECTED_TOOLING_SHA}:${VALIDATOR_RELATIVE_PATH}")
  test -n "${committed_validator_blob}"
}

snapshot_validator() {
  local actual_blob
  local expected_blob

  expected_blob=$(git -C "${SCRIPT_ROOT}" rev-parse \
    "${EXPECTED_TOOLING_SHA}:${VALIDATOR_RELATIVE_PATH}")
  git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${VALIDATOR_RELATIVE_PATH}" \
    >"${VALIDATOR_SNAPSHOT}"
  actual_blob=$(git -C "${SCRIPT_ROOT}" hash-object "${VALIDATOR_SNAPSHOT}")
  test "${actual_blob}" = "${expected_blob}"
  chmod 500 "${VALIDATOR_SNAPSHOT}"
}

image_matches_expected() {
  local image_path=$1
  local actual_sha256

  require_regular_file "${image_path}"
  actual_sha256=$(sha256sum "${image_path}" | awk '{print $1}')
  test "${actual_sha256}" = "${EXPECTED_SHA256}"
}

metadata_matches_expected() {
  local metadata_path=$1

  require_regular_file "${metadata_path}"
  test "$(wc -l <"${metadata_path}" | tr -d ' ')" -eq 10
  grep -Fx -- "source=${SOURCE}" "${metadata_path}" >/dev/null
  grep -Fx -- 'source_cluster=ptyche' "${metadata_path}" >/dev/null
  grep -Fx -- 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh' "${metadata_path}" >/dev/null
  grep -Fx -- 'source_smoke_job=2721177' "${metadata_path}" >/dev/null
  grep -Fx -- 'destination_cluster=lyris' "${metadata_path}" >/dev/null
  grep -Ex -- 'download_job=[0-9]+' "${metadata_path}" >/dev/null
  grep -Ex -- 'downloaded_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${metadata_path}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_SHA256}" "${metadata_path}" >/dev/null
  grep -Fx -- "tooling_commit=${EXPECTED_TOOLING_SHA}" "${metadata_path}" >/dev/null
  grep -Fx -- "rclone_sha256=${EXPECTED_RCLONE_SHA256}" "${metadata_path}" >/dev/null
}

completion_marker_matches_expected() {
  local marker_path=$1

  require_regular_file "${marker_path}"
  test "$(wc -l <"${marker_path}" | tr -d ' ')" -eq 6
  grep -Fx -- "destination=${DESTINATION}" "${marker_path}" >/dev/null
  grep -Fx -- "source=${SOURCE}" "${marker_path}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_SHA256}" "${marker_path}" >/dev/null
  grep -Fx -- "tooling_commit=${EXPECTED_TOOLING_SHA}" "${marker_path}" >/dev/null
  grep -Fx -- "rclone_sha256=${EXPECTED_RCLONE_SHA256}" "${marker_path}" >/dev/null
  grep -Ex -- 'completed_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${marker_path}" >/dev/null
}

trap report_error ERR

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
[[ ${JOB_SCRATCH_DIRECTORY} == /raid/scratch/nemo-rl-validated-nightly/lyris-download-${SLURM_JOB_ID} ]]
for command_name in awk chmod cp git grep ln mkdir rm sha256sum srun tr uname wc; do
  require_command "${command_name}"
done

CURRENT_STEP=validate-compute-architecture
test "$(uname -m)" = "${EXPECTED_COMPUTE_ARCHITECTURE}"

CURRENT_STEP=validate-pinned-rclone
require_regular_executable "${RCLONE_SOURCE}"
source_rclone_sha256=$(sha256sum "${RCLONE_SOURCE}" | awk '{print $1}')
test "${source_rclone_sha256}" = "${EXPECTED_RCLONE_SHA256}"
require_regular_file "${RCLONE_CONFIG}"

CURRENT_STEP=validate-tooling-root
validate_tooling_root
test -d "${DESTINATION%/*}"
test ! -L "${DESTINATION%/*}"
test ! -e "${JOB_SCRATCH_DIRECTORY}"
test ! -e "${PARTIAL}"
test ! -e "${PARTIAL_METADATA}"
test ! -e "${PARTIAL_COMPLETION_MARKER}"

CURRENT_STEP=create-job-scratch
mkdir -p "${RCLONE_DIRECTORY}"
trap cleanup EXIT
snapshot_validator
cp -- "${RCLONE_SOURCE}" "${RCLONE}"
chmod 500 "${RCLONE}"
runtime_rclone_sha256=$(sha256sum "${RCLONE}" | awk '{print $1}')
test "${runtime_rclone_sha256}" = "${EXPECTED_RCLONE_SHA256}"
"${RCLONE}" version >/dev/null
GOMAXPROCS=2 "${RCLONE}" listremotes | grep -Fx 'pbss-team-nemo-ci-s3:' >/dev/null

CURRENT_STEP=download-or-validate-image
if [[ -e ${DESTINATION} || -L ${DESTINATION} ]]; then
  image_matches_expected "${DESTINATION}"
else
  export GOMAXPROCS=8
  srun -J download-validated-nightly \
    "${RCLONE}" copyto \
    --verbose \
    --checksum \
    --checkers=8 \
    --transfers=4 \
    --multi-thread-streams=32 \
    --buffer-size=128M \
    "${SOURCE}" \
    "${PARTIAL}"
  image_matches_expected "${PARTIAL}"
  if ln -T "${PARTIAL}" "${DESTINATION}"; then
    :
  elif [[ -e ${DESTINATION} || -L ${DESTINATION} ]]; then
    image_matches_expected "${DESTINATION}"
  else
    echo "Unable to publish immutable destination: ${DESTINATION}" >&2
    exit 1
  fi
fi

CURRENT_STEP=publish-provenance-metadata
{
  echo "source=${SOURCE}"
  echo 'source_cluster=ptyche'
  echo 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh'
  echo 'source_smoke_job=2721177'
  echo 'destination_cluster=lyris'
  echo "download_job=${SLURM_JOB_ID}"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=${EXPECTED_SHA256}"
  echo "tooling_commit=${EXPECTED_TOOLING_SHA}"
  echo "rclone_sha256=${EXPECTED_RCLONE_SHA256}"
} >"${PARTIAL_METADATA}"
if ln -T "${PARTIAL_METADATA}" "${METADATA}"; then
  :
elif [[ -e ${METADATA} || -L ${METADATA} ]]; then
  metadata_matches_expected "${METADATA}"
else
  echo "Unable to publish immutable metadata: ${METADATA}" >&2
  exit 1
fi

CURRENT_STEP=publish-completion-marker
{
  echo "destination=${DESTINATION}"
  echo "source=${SOURCE}"
  echo "sha256=${EXPECTED_SHA256}"
  echo "tooling_commit=${EXPECTED_TOOLING_SHA}"
  echo "rclone_sha256=${EXPECTED_RCLONE_SHA256}"
  echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${PARTIAL_COMPLETION_MARKER}"
if ln -T "${PARTIAL_COMPLETION_MARKER}" "${COMPLETION_MARKER}"; then
  :
elif [[ -e ${COMPLETION_MARKER} || -L ${COMPLETION_MARKER} ]]; then
  completion_marker_matches_expected "${COMPLETION_MARKER}"
else
  echo "Unable to publish immutable completion marker: ${COMPLETION_MARKER}" >&2
  exit 1
fi

CURRENT_STEP=validate-final-receipt
"${VALIDATOR_SNAPSHOT}" \
  "${DESTINATION}" \
  "${METADATA}" \
  "${COMPLETION_MARKER}" \
  "${EXPECTED_TOOLING_SHA}"

CURRENT_STEP=cleanup-success
rm -f -- "${PARTIAL}" "${PARTIAL_METADATA}" "${PARTIAL_COMPLETION_MARKER}"
rm -rf -- "${JOB_SCRATCH_DIRECTORY}"
trap - EXIT ERR

ls -lh "${DESTINATION}" "${METADATA}" "${COMPLETION_MARKER}"
