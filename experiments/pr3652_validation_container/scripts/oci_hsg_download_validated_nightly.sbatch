#!/bin/bash
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --time=02:00:00
#SBATCH --job-name=nemotron_n3_post-download_validated_nightly
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-download/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-download/slurm-%j.err

set -euo pipefail

readonly SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly DESTINATION=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly PARTIAL=${DESTINATION}.partial.${SLURM_JOB_ID}
readonly METADATA=${DESTINATION}.metadata.txt
readonly PARTIAL_METADATA=${METADATA}.partial.${SLURM_JOB_ID}

cleanup() {
  rm -f -- "${PARTIAL}" "${PARTIAL_METADATA}"
}
trap cleanup EXIT

command -v rclone >/dev/null
test ! -e "${DESTINATION}"
test ! -e "${METADATA}"

export GOMAXPROCS=8
/cm/local/apps/slurm/current/bin/srun -J download-validated-nightly \
  rclone copyto \
  --verbose \
  --checksum \
  --checkers=8 \
  --transfers=4 \
  --multi-thread-streams=32 \
  --buffer-size=128M \
  "${SOURCE}" \
  "${PARTIAL}"

actual_sha256=$(sha256sum "${PARTIAL}" | awk '{print $1}')
test "${actual_sha256}" = "${EXPECTED_SHA256}"

{
  echo "source=${SOURCE}"
  echo "source_cluster=ptyche"
  echo "source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh"
  echo "source_smoke_job=2721177"
  echo "download_job=${SLURM_JOB_ID}"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=${actual_sha256}"
} >"${PARTIAL_METADATA}"

mv "${PARTIAL_METADATA}" "${METADATA}"
mv "${PARTIAL}" "${DESTINATION}"
cleanup
trap - EXIT

ls -lh "${DESTINATION}" "${METADATA}"
