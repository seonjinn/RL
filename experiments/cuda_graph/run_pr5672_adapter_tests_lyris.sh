#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_dlalgo_llm-cg.pr5672-tests
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cudagraph-seqpack-lyris-pr5672-adapter-20260716/experiments/cuda_graph/logs/pr5672-tests-%j.out

set -euo pipefail

WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cudagraph-seqpack-lyris-pr5672-adapter-20260716
CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_2346595.sqsh
VENV=/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/nemo-rl-cgseqpack-lyris-20260715

mkdir -p "${WORKTREE}/experiments/cuda_graph/logs"

export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKTREE}" \
    "${VENV}/bin/python" -m pytest -q tests/unit/models/megatron/test_pr5672_cuda_graph_adapter.py
