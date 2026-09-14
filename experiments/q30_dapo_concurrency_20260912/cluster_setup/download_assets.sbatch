#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

readonly BUCKET=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/q30-dapo-parity-20260914/oci-7023221-c9775f03
readonly DEST=/lustre/fsw/coreai_dlalgo_llm/users/sna/q30-dapo-parity-20260914
readonly RCLONE=/home/sna/.local/lib/rclone/rclone-arm64
test -x "${RCLONE}"
mkdir -p "${DEST}"
ready=false
for ((attempt=0; attempt<90; attempt++)); do
  if "${RCLONE}" lsf "${BUCKET}/SHA256SUMS" --retries=1 --low-level-retries=1 2>/dev/null | grep -qx SHA256SUMS; then
    ready=true
    break
  fi
  echo "Waiting for OCI asset manifest: attempt ${attempt}"
  sleep 30
done
test "${ready}" = true
"${RCLONE}" copy --checksum --immutable --transfers=4 --checkers=4 --multi-thread-streams=4 --buffer-size=32M --stats=60s "${BUCKET}" "${DEST}"
cd "${DEST}"
sha256sum --check SHA256SUMS
echo "ASSETS_VERIFIED ${DEST}"
