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

set -euo pipefail

readonly SOURCE=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh
readonly SOURCE_HASH_FILE=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh.sha256
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly BASENAME=nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly FINAL_DIRECTORY=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers
readonly DESTINATION=${FINAL_DIRECTORY}/${BASENAME}

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

cleanup() {
  local exit_status=$?

  trap - EXIT
  if ! rclone purge "${TEMP_DIRECTORY}"; then
    echo "Failed to purge job-scoped temporary remote: ${TEMP_DIRECTORY}" >&2
    if (( exit_status == 0 )); then
      exit_status=1
    fi
  fi
  rm -rf -- "${SCRATCH_DIRECTORY}"
  exit "${exit_status}"
}

command -v rclone >/dev/null
command -v sha256sum >/dev/null
command -v stat >/dev/null
test -f "${SOURCE}"
test -f "${SOURCE_HASH_FILE}"
test ! -e "${SCRATCH_DIRECTORY}"
mkdir -p "${TEMP_DOWNLOAD_DIRECTORY}" "${FINAL_DOWNLOAD_DIRECTORY}"
trap cleanup EXIT

export GOMAXPROCS=8
source_size=$(stat --format='%s' "${SOURCE}")
source_sha256=$(sha256sum "${SOURCE}" | awk '{print $1}')
test "${source_sha256}" = "${EXPECTED_SHA256}"
sidecar_sha256=$(awk '{print $1}' "${SOURCE_HASH_FILE}")
test "${sidecar_sha256}" = "${source_sha256}"

srun -J upload-validated-nightly-temporary \
  rclone copy \
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

srun -J verify-validated-nightly-temporary \
  rclone copy \
  --verbose \
  --checksum \
  "${TEMP_OBJECT}" \
  "${TEMP_DOWNLOAD_DIRECTORY}"
test "$(sha256sum "${TEMP_DOWNLOAD}" | awk '{print $1}')" = "${EXPECTED_SHA256}"

source_sha256_after_upload=$(sha256sum "${SOURCE}" | awk '{print $1}')
test "${source_sha256_after_upload}" = "${EXPECTED_SHA256}"

if ! srun -J publish-validated-nightly \
  rclone copy \
  --verbose \
  --immutable \
  --checksum \
  "${TEMP_DIRECTORY}" \
  "${FINAL_DIRECTORY}"; then
  echo "Final directory promotion reported a conflict; verifying final bytes" >&2
fi

srun -J verify-validated-nightly-final \
  rclone copy \
  --verbose \
  --checksum \
  "${DESTINATION}" \
  "${FINAL_DOWNLOAD_DIRECTORY}"
test "$(sha256sum "${FINAL_DOWNLOAD}" | awk '{print $1}')" = "${EXPECTED_SHA256}"
test "$(stat --format='%s' "${FINAL_DOWNLOAD}")" = "${source_size}"

rclone lsjson --stat "${DESTINATION}"
