#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

# PBSS transfer pattern: one data-mover task, bounded assets, immutable destination.
readonly BUCKET=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/q30-dapo-parity-20260914/oci-7023221-c9775f03
readonly IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly SOURCE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa
scratch_parent=/raid/scratch
if [[ ! -d "${scratch_parent}" ]]; then scratch_parent=/tmp; fi
readonly STAGE=${scratch_parent}/sna/q30-parity-upload-${SLURM_JOB_ID}
mkdir -p "${STAGE}/payload"
ln -s "${IMAGE}" "${STAGE}/payload/nemo_rl_nightly_20260909_7023221.sqsh"
ln -s "${IMAGE}.metadata.txt" "${STAGE}/payload/image.metadata.txt"
ln -s /lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B "${STAGE}/payload/target"
ln -s "${SOURCE}/sd2p3swa-q30-base-ptv3swe-dflash-b8-16n/exported-checkpoint-44000" "${STAGE}/payload/dflash"
ln -s "${SOURCE}/sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/exported-checkpoint-44000" "${STAGE}/payload/dspark"
cd "${STAGE}/payload"
test -s target/config.json
test -s dflash/model.safetensors
test -s dspark/model.safetensors
find -L . -maxdepth 3 -type f ! -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum > "${STAGE}/SHA256SUMS"
grep -q '^60cfa5cb28834f1a4c154103a993fb87fdf5d38a0fa29ebd3ae37b63b492093c  ./nemo_rl_nightly_20260909_7023221.sqsh$' "${STAGE}/SHA256SUMS"
rclone copy -L --checksum --immutable --exclude '.cache/**' --transfers=4 --checkers=4 --s3-upload-concurrency=8 --buffer-size=32M --stats=60s . "${BUCKET}"
# The manifest is published last and is also the readiness marker.
rclone copyto --immutable "${STAGE}/SHA256SUMS" "${BUCKET}/SHA256SUMS"
echo "UPLOAD_COMPLETE ${BUCKET}"
