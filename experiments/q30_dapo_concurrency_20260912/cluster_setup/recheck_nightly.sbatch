#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -euo pipefail
readonly site=${1:?lyris or ptyche}
case "${site}" in
  lyris) readonly image_job=3078480 ;;
  ptyche) readonly image_job=2837270 ;;
  *) exit 2 ;;
esac
readonly repo=/home/sna/nemorl-q30-dapo-${site}-20260914
readonly image_dir=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-20260916
readonly image=${image_dir}/nemo_rl_nightly_20260916_${image_job}.sqsh
git -C "${repo}" rev-parse HEAD
cd "${image_dir}"
sha256sum --check "${image}.sha256"
srun --container-image="${image}" --no-container-mount-home \
  --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid \
  --container-workdir=/opt \
  /bin/bash "${repo}/experiments/q30_dapo_concurrency_20260912/cluster_setup/nightly_smoke_inside.sh"
touch "${image}.smoke-passed"
ln -s "$(basename "${image}")" "${image_dir}/validated.sqsh.${SLURM_JOB_ID}"
mv -T "${image_dir}/validated.sqsh.${SLURM_JOB_ID}" "${image_dir}/validated.sqsh"
echo "IMAGE_GPU_SMOKE_COMPLETE ${image}"
