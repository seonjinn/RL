#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-minimax-prefetch
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
readonly CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
test -r "$IMAGE"
test -r "$CACHE/token"

export HF_HOME=$CACHE
export HF_HUB_CACHE=$CACHE/hub
srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/lustre:/lustre \
  bash -lc 'python -c "from huggingface_hub import HfApi, snapshot_download; model=\"MiniMaxAI/MiniMax-M2.7\"; info=HfApi().model_info(model, files_metadata=True); print(\"model_sha\", info.sha, flush=True); print(\"model_bytes\", sum(item.size or 0 for item in info.siblings), flush=True); print(\"cache_path\", snapshot_download(model, max_workers=8), flush=True)"'
