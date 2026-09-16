#!/usr/bin/env bash
# Adapted from the stage-training-containers skill template for MARS storage.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
set -euo pipefail

: "${SOURCE_IMAGE:?Set a digest-pinned registry image}"
: "${CONTAINER_DIR:?Set the dedicated Lustre output directory}"
: "${SOURCE_COMMIT:?Set the launcher source commit}"
: "${SETUP_REPO:?Set the source checkout under /home}"
: "${SLURM_JOB_ID:?Run in a SLURM allocation}"
[[ "${SOURCE_IMAGE}" =~ ^nvcr.io/nvidian/nemo-rl@sha256:[a-f0-9]{64}$ ]]
[[ "${CONTAINER_DIR}" == /lustre/*/nemo-rl-20260916 ]]
[[ "${SETUP_REPO}" == /home/sna/nemorl-q30-dapo-* ]]
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
test "$(git -C "${SETUP_REPO}" rev-parse HEAD)" = "${SOURCE_COMMIT}"

readonly scratch=/raid/scratch/sna/nemo-nightly-stage-${SLURM_JOB_ID}
readonly output=${CONTAINER_DIR}/nemo_rl_nightly_20260916_${SLURM_JOB_ID}.sqsh
readonly partial=${output}.partial
readonly stage_image=${scratch}/image.sqsh
readonly stage_uri="docker://nvcr.io#${SOURCE_IMAGE#nvcr.io/}"
test ! -e "${output}"
test ! -e "${partial}"
mkdir -p "${scratch}" "${CONTAINER_DIR}"
export ENROOT_CACHE_PATH=${scratch}/cache
export ENROOT_DATA_PATH=${scratch}/data
export ENROOT_TEMP_PATH=${scratch}/tmp
export ENROOT_RUNTIME_PATH=${scratch}/runtime
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_TEMP_PATH}" "${ENROOT_RUNTIME_PATH}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}nvcr.io"
export no_proxy="${NO_PROXY}"

echo "SOURCE_IMAGE=${SOURCE_IMAGE}"
echo "LAUNCHER_SOURCE_COMMIT=${SOURCE_COMMIT}"
echo "OUTPUT=${output}"
enroot import -o "${stage_image}" "${stage_uri}"
sha=$(sha256sum "${stage_image}" | awk '{print $1}')
cp "${stage_image}" "${partial}"
echo "${sha}  ${partial}" | sha256sum -c -
mv "${partial}" "${output}"
{
  echo "source_tag=nvcr.io/nvidian/nemo-rl:nightly"
  echo "source_image=${SOURCE_IMAGE}"
  echo "image_created_utc=2026-09-16T09:10:22.91877945Z"
  echo "launcher_source_commit=${SOURCE_COMMIT}"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "download_host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "sha256=${sha}"
} >"${output}.metadata.txt"
echo "${sha}  $(basename "${output}")" >"${output}.sha256"
echo "IMAGE_STAGED ${output}"

srun --container-image="${output}" --no-container-mount-home \
  --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid \
  --container-workdir=/opt \
  /bin/bash "${SETUP_REPO}/experiments/q30_dapo_concurrency_20260912/cluster_setup/nightly_smoke_inside.sh"
touch "${output}.smoke-passed"
ln -s "$(basename "${output}")" "${CONTAINER_DIR}/validated.sqsh.${SLURM_JOB_ID}"
mv -T "${CONTAINER_DIR}/validated.sqsh.${SLURM_JOB_ID}" "${CONTAINER_DIR}/validated.sqsh"
echo "IMAGE_GPU_SMOKE_COMPLETE ${output}"

# Remove only this successful job's large node-local import output.
rm -- "${stage_image}"
