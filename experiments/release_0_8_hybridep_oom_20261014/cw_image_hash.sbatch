#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-image-hash
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
readonly OUT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/image.sha256
test -r "$IMAGE"
sha256sum "$IMAGE" > "${OUT}.tmp.${SLURM_JOB_ID}"
mv "${OUT}.tmp.${SLURM_JOB_ID}" "$OUT"
cat "$OUT"
