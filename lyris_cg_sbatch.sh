#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-cg.llama_cg
#SBATCH --partition=gb200
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cgfix/experiments/lyris_llama_cg/slurm-%j.out

set -euo pipefail

mkdir -p /lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cgfix/experiments/lyris_llama_cg/logs

srun --nodes=1 --ntasks=1 \
  --no-container-mount-home \
  --container-image=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh \
  --container-mounts=/lustre:/lustre \
  --container-workdir=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cgfix \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cgfix/lyris_setup_cg.sh
