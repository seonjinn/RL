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
readonly DESTINATION=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh

command -v rclone >/dev/null
command -v sha256sum >/dev/null
command -v stat >/dev/null
test -f "${SOURCE}"
test -f "${SOURCE_HASH_FILE}"

export GOMAXPROCS=8
source_size=$(stat --format='%s' "${SOURCE}")
actual_sha256=$(sha256sum "${SOURCE}" | awk '{print $1}')
test "${actual_sha256}" = "${EXPECTED_SHA256}"
sidecar_sha256=$(awk '{print $1}' "${SOURCE_HASH_FILE}")
test "${sidecar_sha256}" = "${actual_sha256}"

srun -J upload-validated-nightly \
  rclone copyto \
  --verbose \
  --immutable \
  --checksum \
  --checkers=8 \
  --transfers=4 \
  --multi-thread-streams=32 \
  --s3-upload-concurrency=32 \
  --s3-chunk-size=128M \
  --buffer-size=128M \
  "${SOURCE}" \
  "${DESTINATION}"

rclone lsjson --stat "${DESTINATION}"
remote_size=$(rclone lsl "${DESTINATION}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')
test "${remote_size}" = "${source_size}"
