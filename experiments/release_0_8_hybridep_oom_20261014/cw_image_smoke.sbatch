#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-image-smoke
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
test -r "$IMAGE"

srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/lustre:/lustre,/home:/home \
  bash -lc 'python -c "import torch; import deep_ep; import nemo_rl; from deep_ep import HybridEPBuffer; print(\"CUDA_VISIBLE\", torch.cuda.is_available(), torch.cuda.device_count()); print(\"TORCH\", torch.__version__); print(\"DEEP_EP\", deep_ep.__file__); print(\"NEMO_RL\", nemo_rl.__file__)"'
