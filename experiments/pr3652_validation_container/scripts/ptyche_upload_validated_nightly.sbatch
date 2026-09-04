#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=tcpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --time=02:00:00
#SBATCH --job-name=coreai_dlalgo_llm-container_transfer.upload_validated_nightly
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/ptyche-upload/slurm-%j.out
#SBATCH --error=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/ptyche-upload/slurm-%j.err

set -Eeuo pipefail

readonly RCLONE=/home/sna/.local/libexec/nemo-rl/rclone/sha256-dc1ec3109000e4d36c8d14efac6d4c4158d1b860853cb75c09dcc9f6dded420b/rclone
readonly EXPECTED_RCLONE_SHA256=dc1ec3109000e4d36c8d14efac6d4c4158d1b860853cb75c09dcc9f6dded420b
readonly SOURCE=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh
readonly SOURCE_HASH_FILE=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh.sha256
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly BASENAME=nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly FINAL_DIRECTORY=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers
readonly DESTINATION=${FINAL_DIRECTORY}/${BASENAME}

CURRENT_STEP=validate-slurm-job-id
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi

readonly TEMP_DIRECTORY=${FINAL_DIRECTORY}/.temporary-${BASENAME}.${SLURM_JOB_ID}
readonly TEMP_OBJECT=${TEMP_DIRECTORY}/${BASENAME}
readonly SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-validated-nightly/ptyche-upload-${SLURM_JOB_ID}
readonly TEMP_DOWNLOAD_DIRECTORY=${SCRATCH_DIRECTORY}/temporary
readonly FINAL_DOWNLOAD_DIRECTORY=${SCRATCH_DIRECTORY}/final
readonly TEMP_DOWNLOAD=${TEMP_DOWNLOAD_DIRECTORY}/${BASENAME}
readonly FINAL_DOWNLOAD=${FINAL_DOWNLOAD_DIRECTORY}/${BASENAME}

CURRENT_STEP=initialize

report_error() {
  local exit_status=$?

  trap - ERR
  printf 'ptyche upload failed: step=%s exit=%s line=%s job=%s node=%s\n' \
    "${CURRENT_STEP}" \
    "${exit_status}" \
    "${BASH_LINENO[0]:-unknown}" \
    "${SLURM_JOB_ID:-unknown}" \
    "${SLURMD_NODENAME:-unknown}" >&2
}

report_cleanup_error() {
  local step=$1
  local exit_status=$2
  local line=$3

  printf 'ptyche upload cleanup failed: step=%s exit=%s line=%s job=%s node=%s\n' \
    "${step}" \
    "${exit_status}" \
    "${line}" \
    "${SLURM_JOB_ID:-unknown}" \
    "${SLURMD_NODENAME:-unknown}" >&2
}

require_command() {
  local command_name=$1

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'ptyche upload preflight failed: required command is unavailable: %s\n' \
      "${command_name}" >&2
    return 127
  fi
}

require_regular_file() {
  local path=$1

  if [[ ! -f "${path}" ]]; then
    printf 'ptyche upload preflight failed: required regular file is missing: %s\n' \
      "${path}" >&2
    return 1
  fi
}

require_regular_executable() {
  local path=$1

  if [[ ! -f "${path}" || -L "${path}" || ! -x "${path}" ]]; then
    printf 'ptyche upload preflight failed: required regular executable is missing: %s\n' \
      "${path}" >&2
    return 1
  fi
}

require_compatible_rclone() {
  local path=$1
  local exit_status

  if "${path}" version >/dev/null 2>&1; then
    return 0
  else
    exit_status=$?
  fi
  echo 'ptyche upload preflight failed: rclone binary is incompatible with this node' >&2
  return "${exit_status}"
}

require_absent_path() {
  local path=$1

  if [[ -e "${path}" || -L "${path}" ]]; then
    printf 'ptyche upload preflight failed: job scratch path already exists: %s\n' \
      "${path}" >&2
    return 1
  fi
}

require_equal() {
  local actual=$1
  local expected=$2
  local description=$3

  if [[ "${actual}" != "${expected}" ]]; then
    printf 'ptyche upload integrity check failed: %s did not match the expected value\n' \
      "${description}" >&2
    return 1
  fi
}

trap report_error ERR

cleanup() {
  local exit_status=$?
  local cleanup_exit_status
  local cleanup_line

  trap - EXIT ERR
  if "${RCLONE}" purge "${TEMP_DIRECTORY}"; then
    :
  else
    cleanup_exit_status=$?
    cleanup_line=$LINENO
    report_cleanup_error \
      cleanup-job-temporary-remote \
      "${cleanup_exit_status}" \
      "${cleanup_line}"
    if (( exit_status == 0 )); then
      exit_status=${cleanup_exit_status}
    fi
  fi
  if rm -rf -- "${SCRATCH_DIRECTORY}"; then
    :
  else
    cleanup_exit_status=$?
    cleanup_line=$LINENO
    report_cleanup_error \
      cleanup-job-scratch \
      "${cleanup_exit_status}" \
      "${cleanup_line}"
    if (( exit_status == 0 )); then
      exit_status=${cleanup_exit_status}
    fi
  fi
  exit "${exit_status}"
}

CURRENT_STEP=preflight-required-commands
require_command sha256sum
require_command stat
require_command awk
require_command srun

CURRENT_STEP=preflight-rclone-binary
require_regular_executable "${RCLONE}"

CURRENT_STEP=preflight-rclone-integrity
rclone_sha256=$(sha256sum "${RCLONE}" | awk '{print $1}')
require_equal "${rclone_sha256}" "${EXPECTED_RCLONE_SHA256}" 'rclone SHA256'

CURRENT_STEP=preflight-rclone-compatibility
require_compatible_rclone "${RCLONE}"

CURRENT_STEP=preflight-source-files
require_regular_file "${SOURCE}"
require_regular_file "${SOURCE_HASH_FILE}"

CURRENT_STEP=preflight-job-scratch
require_absent_path "${SCRATCH_DIRECTORY}"

CURRENT_STEP=create-job-scratch
mkdir -p "${TEMP_DOWNLOAD_DIRECTORY}" "${FINAL_DOWNLOAD_DIRECTORY}"
trap cleanup EXIT

export GOMAXPROCS=8
CURRENT_STEP=hash-source-before-upload
source_size=$(stat --format='%s' "${SOURCE}")
source_sha256=$(sha256sum "${SOURCE}" | awk '{print $1}')
require_equal "${source_sha256}" "${EXPECTED_SHA256}" 'source SHA256 before upload'
sidecar_sha256=$(awk '{print $1}' "${SOURCE_HASH_FILE}")
require_equal "${sidecar_sha256}" "${source_sha256}" 'source SHA256 sidecar'

CURRENT_STEP=upload-job-temporary-directory
srun -J upload-validated-nightly-temporary \
  "${RCLONE}" copy \
  --verbose \
  --checksum \
  --checkers=8 \
  --transfers=4 \
  --multi-thread-streams=32 \
  --s3-upload-concurrency=32 \
  --s3-chunk-size=128M \
  --buffer-size=128M \
  "${SOURCE}" \
  "${TEMP_DIRECTORY}"

CURRENT_STEP=download-and-hash-job-temporary-object
srun -J verify-validated-nightly-temporary \
  "${RCLONE}" copy \
  --verbose \
  --checksum \
  "${TEMP_OBJECT}" \
  "${TEMP_DOWNLOAD_DIRECTORY}"
temporary_download_sha256=$(sha256sum "${TEMP_DOWNLOAD}" | awk '{print $1}')
require_equal "${temporary_download_sha256}" "${EXPECTED_SHA256}" 'temporary remote download SHA256'

CURRENT_STEP=rehash-source-after-temporary-upload
source_sha256_after_upload=$(sha256sum "${SOURCE}" | awk '{print $1}')
require_equal "${source_sha256_after_upload}" "${EXPECTED_SHA256}" 'source SHA256 after upload'

CURRENT_STEP=promote-job-temporary-directory
if ! srun -J publish-validated-nightly \
  "${RCLONE}" copy \
  --verbose \
  --immutable \
  --checksum \
  "${TEMP_DIRECTORY}" \
  "${FINAL_DIRECTORY}"; then
  echo 'Final directory promotion returned nonzero; verifying immutable final bytes' >&2
fi

CURRENT_STEP=download-and-hash-final-object
srun -J verify-validated-nightly-final \
  "${RCLONE}" copy \
  --verbose \
  --checksum \
  "${DESTINATION}" \
  "${FINAL_DOWNLOAD_DIRECTORY}"
final_download_sha256=$(sha256sum "${FINAL_DOWNLOAD}" | awk '{print $1}')
require_equal "${final_download_sha256}" "${EXPECTED_SHA256}" 'final remote download SHA256'
final_download_size=$(stat --format='%s' "${FINAL_DOWNLOAD}")
require_equal "${final_download_size}" "${source_size}" 'final remote download size'

CURRENT_STEP=inspect-final-remote-object
"${RCLONE}" lsjson --stat "${DESTINATION}"
