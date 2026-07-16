#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=coreai_dlalgo_llm-cg.llama8b
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cudagraph-seqpack-lyris-pr5672-vs-pr5783-20260716/experiments/cuda_graph/logs/llama8b-cg-comparison-%j.out

set -euo pipefail

CONDITION=${CONDITION:?Set CONDITION to nocg, current-attn, current-attn-mlp, pr5672-attn, or pr5672-attn-mlp.}
STEPS=${STEPS:-20}
RUN_TAG=${RUN_TAG:-${CONDITION}-steps${STEPS}}
BASELINE_WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cudagraph-seqpack-lyris-pr5783-baseline-20260716
PR5672_WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cudagraph-seqpack-lyris-pr5672-vs-pr5783-20260716
CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_2346595.sqsh
VENV=/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/nemo-rl-cgseqpack-lyris-20260715

case "${CONDITION}" in
  nocg)
    WORKTREE=${BASELINE_WORKTREE}
    RECIPE=grpo-llama3.1-8b-instruct-1n4g-nocg.yaml
    ;;
  current-attn)
    WORKTREE=${BASELINE_WORKTREE}
    RECIPE=grpo-llama3.1-8b-instruct-1n4g-cg.yaml
    ;;
  current-attn-mlp)
    WORKTREE=${BASELINE_WORKTREE}
    RECIPE=grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w6.yaml
    ;;
  pr5672-attn)
    WORKTREE=${PR5672_WORKTREE}
    RECIPE=grpo-llama3.1-8b-instruct-1n4g-cg-pr5672-attn.yaml
    ;;
  pr5672-attn-mlp)
    WORKTREE=${PR5672_WORKTREE}
    RECIPE=grpo-llama3.1-8b-instruct-1n4g-cg-pr5672-attn-mlp.yaml
    ;;
  *)
    echo "Unknown CONDITION: ${CONDITION}" >&2
    exit 2
    ;;
esac

CONFIG=${WORKTREE}/examples/configs/recipes/llm/performance/${RECIPE}
LOG_DIR=logs/llama8b-pr5672-vs-pr5783/${RUN_TAG}

mkdir -p "${PR5672_WORKTREE}/experiments/cuda_graph/logs"

echo "condition=${CONDITION} steps=${STEPS} worktree=${WORKTREE} recipe=${RECIPE}"
git -C "${WORKTREE}" rev-parse HEAD
git -C "${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM" rev-parse HEAD

export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKTREE}" \
    "${VENV}/bin/python" "${WORKTREE}/examples/run_grpo.py" \
    --config "${CONFIG}" \
    "grpo.max_num_steps=${STEPS}" \
    "logger.log_dir=${LOG_DIR}" \
    "logger.wandb.name=${RUN_TAG}"
