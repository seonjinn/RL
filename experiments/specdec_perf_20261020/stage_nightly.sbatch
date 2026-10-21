#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --job-name=coreai_dlalgo_nemorl.stage_nemo_rl_nightly
#SBATCH --output=stage_nemo_rl_nightly_%j.log

# Import nvcr.io/nvidian/nemo-rl:nightly to an immutable sqsh and repoint the
# nemo_rl_nightly.sqsh symlink. Runs on a GB200 node so the aarch64 layers match
# (batch QOS rejects jobs with fewer than 4 GPUs).
set -euo pipefail

SOURCE_IMAGE=${SOURCE_IMAGE:-nvcr.io/nvidian/nemo-rl:nightly}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-nemo_rl_nightly}
CONTAINER_DIR=${CONTAINER_DIR:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers}
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}

registry=${SOURCE_IMAGE%%/*}
repository_and_tag=${SOURCE_IMAGE#*/}
registry_host=${registry%%:*}
enroot_image_uri="docker://${registry}#${repository_and_tag}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
export no_proxy="${NO_PROXY}"

date_stamp=$(date +%Y%m%d)
output_file="${CONTAINER_DIR}/${OUTPUT_PREFIX}_${date_stamp}_${SLURM_JOB_ID}.sqsh"
metadata_file="${output_file}.metadata.txt"
current_link="${CONTAINER_DIR}/${OUTPUT_PREFIX}.sqsh"
partial_file="${output_file}.partial.${SLURM_JOB_ID}"
partial_metadata_file="${metadata_file}.partial.${SLURM_JOB_ID}"
partial_current_link="${current_link}.partial.${SLURM_JOB_ID}"

mkdir -p "${CONTAINER_DIR}"
# layer cache on node-local disk, never on lustre
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-/raid/scratch/${USER}/enroot-cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-/raid/scratch/${USER}/enroot-data}
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}"

cleanup() {
  rm -f "${partial_file}" "${partial_metadata_file}" "${partial_current_link}"
}
trap cleanup EXIT

if [[ -e "${output_file}" || -e "${metadata_file}" ]]; then
  echo "Immutable output already exists: ${output_file}" >&2
  exit 1
fi

echo "source_image=${SOURCE_IMAGE}"
echo "output_file=${output_file}"

enroot import -o "${partial_file}" "${enroot_image_uri}"
image_sha256=$(sha256sum "${partial_file}" | awk '{print $1}')

{
  echo "source_image=${SOURCE_IMAGE}"
  echo "imported_enroot_uri=${enroot_image_uri}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "download_host=$(hostname)"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=${image_sha256}"
} >"${partial_metadata_file}"

mv "${partial_metadata_file}" "${metadata_file}"
mv "${partial_file}" "${output_file}"
ln -s "${output_file}" "${partial_current_link}"
mv -Tf "${partial_current_link}" "${current_link}"
cleanup
trap - EXIT

echo "container=${output_file}"
ls -lh "${output_file}" "${metadata_file}" "${current_link}"