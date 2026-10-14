#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-hf-offline-smoke
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1

test -r "$IMAGE"

srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/lustre:/lustre,/home:/home \
  bash -lc 'set -euo pipefail; cd /opt/nemo-rl; uv run --no-sync python -c "import os; from transformers import AutoConfig, AutoTokenizer; models = (\"Qwen/Qwen3.5-35B-A3B-Base\", \"MiniMaxAI/MiniMax-M2.7\"); print(\"HF_HUB_OFFLINE=\", os.environ[\"HF_HUB_OFFLINE\"]); print([(model, type(AutoTokenizer.from_pretrained(model, trust_remote_code=True)).__name__, type(AutoConfig.from_pretrained(model, trust_remote_code=True)).__name__) for model in models])"; uv run --no-sync python -c "from datasets import load_dataset; repos = (\"BytedTsinghua-SIA/DAPO-Math-17k\", \"BytedTsinghua-SIA/AIME-2024\"); print([(repo, len(load_dataset(repo, split=\"train\"))) for repo in repos])"'
