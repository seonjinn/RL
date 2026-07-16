#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=coreai_dlalgo_llm-prefetch.qwen30
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5783-ptyche-runtime-20260716/experiments/cuda_graph/logs/prefetch-qwen30-%j.out

set -euo pipefail

WORKTREE=${WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5783-ptyche-runtime-20260716}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf}
export HF_HOME
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}

if [[ ! -s "${HF_HOME}/token" ]]; then
  echo "Missing Hugging Face token at ${HF_HOME}/token" >&2
  exit 2
fi

mkdir -p "${WORKTREE}/experiments/cuda_graph/logs"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${WORKTREE}" \
  python - Qwen/Qwen3-30B-A3B <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
path = snapshot_download(repo_id=repo_id)
print(f"PREFETCH_OK repo_id={repo_id} path={path}")
PY
