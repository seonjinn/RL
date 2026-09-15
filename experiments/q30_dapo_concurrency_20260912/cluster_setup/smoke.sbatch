#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -euo pipefail

readonly SITE="${1:?lyris or ptyche}"
case "${SITE}" in lyris|ptyche) ;; *) exit 2 ;; esac
readonly REPO=/home/sna/nemorl-q30-dapo-${SITE}-20260914
readonly ASSETS=/lustre/fsw/coreai_dlalgo_llm/users/sna/q30-dapo-parity-20260914
test -s "${ASSETS}/SHA256SUMS"
git -C "${REPO}" rev-parse HEAD
git -C "${REPO}" submodule status --recursive
export Q30_SETUP_REPO="${REPO}" Q30_SETUP_ASSETS="${ASSETS}"
srun --container-image="${ASSETS}/nemo_rl_nightly_20260909_7023221.sqsh" \
  --container-mounts=/home:/home,/lustre:/lustre \
  --no-container-mount-home \
  --container-workdir="${REPO}" \
  bash experiments/q30_dapo_concurrency_20260912/cluster_setup/smoke_inside.sh
