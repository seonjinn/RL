#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment=metrics

set -euo pipefail

: "${ARM:?ARM is required}"
: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${REFIT_OPT:?REFIT_OPT is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"

BASE=${BASE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna}
REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}
WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-pr3294-ab}
CONTAINER=${CONTAINER:-$BASE/containers/nemo_rl_nightly.sqsh}
CONFIG_PATH=examples/configs/recipes/llm/performance/${CONFIG_NAME}.yaml
EXP_DIR=$WORK/results/$RUN_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log

case "$REFIT_OPT" in
  0)
    PREQUANTIZE=false
    PERSISTENT_IPC=false
    SLIM_OFFLOAD=false
    export NRL_MXFP8_BATCHED_SHUFFLE=0
    export NRL_REFIT_CACHED_LOADERS=0
    ;;
  1)
    PREQUANTIZE=true
    PERSISTENT_IPC=true
    SLIM_OFFLOAD=true
    export NRL_MXFP8_BATCHED_SHUFFLE=1
    export NRL_REFIT_CACHED_LOADERS=1
    ;;
  *)
    echo "REFIT_OPT must be 0 or 1" >&2
    exit 2
    ;;
esac

test -d "$REPO/.git"
test -f "$REPO/$CONFIG_PATH"
test -f "$CONTAINER"
mkdir -p "$WORK/slurm" "$LOG_DIR" "$BASE/uv_cache/mxfp8-refit-pr3294"

if [[ -f "$HOME/.nemo_rl_tokens" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.nemo_rl_tokens"
fi

export CONTAINER
export MOUNTS=/lustre:/lustre,/scratch:/scratch
export GPUS_PER_NODE=4
export HF_HOME=$BASE/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export UV_CACHE_DIR=$BASE/uv_cache/mxfp8-refit-pr3294
export UV_HTTP_TIMEOUT=300
export UV_LOCK_TIMEOUT=3600
export BASE_LOG_DIR=$WORK/slurm
export XDG_CACHE_HOME=/tmp/nemo_rl_pr3294_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
export CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export FLASHINFER_CACHE_DIR=$XDG_CACHE_HOME/flashinfer
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export NVTE_CUDA_ARCHS=100
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray
export NRL_MXFP8_SHUFFLE_VERIFY=0

if [[ -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME/token")"
  export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-sna-mxfp8-qkvo-refit-pr3294}

{
  echo "arm=$ARM"
  echo "config=$CONFIG_PATH"
  echo "repo_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "max_steps=$MAX_STEPS"
  echo "train_global_batch_size=2048"
  echo "seed=42"
  echo "real_importance_sampling=true"
  echo "sleep_level=1"
  echo "refit_buffer_size_gb=4"
  echo "pinned_reference_swap=false"
  echo "refit_prequantize=$PREQUANTIZE"
  echo "refit_persistent_ipc_buffers=$PERSISTENT_IPC"
  echo "refit_slim_offload_after=$SLIM_OFFLOAD"
  echo "NRL_MXFP8_BATCHED_SHUFFLE=$NRL_MXFP8_BATCHED_SHUFFLE"
  echo "NRL_REFIT_CACHED_LOADERS=$NRL_REFIT_CACHED_LOADERS"
} >"$EXP_DIR/metadata.env"

export COMMAND="set -euo pipefail
cd '$REPO'
if [[ -f \"\$HOME/.nemo_rl_tokens\" ]]; then source \"\$HOME/.nemo_rl_tokens\"; fi
mkdir -p '$LOG_DIR'
export HF_HOME='$HF_HOME'
export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
export HF_HUB_CACHE='$HF_HUB_CACHE'
export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
export UV_CACHE_DIR='$UV_CACHE_DIR'
export UV_HTTP_TIMEOUT='$UV_HTTP_TIMEOUT'
export UV_LOCK_TIMEOUT='$UV_LOCK_TIMEOUT'
export XDG_CACHE_HOME='$XDG_CACHE_HOME'
export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'
export TORCHINDUCTOR_CACHE_DIR='$TORCHINDUCTOR_CACHE_DIR'
export CUDA_CACHE_PATH='$CUDA_CACHE_PATH'
export FLASHINFER_CACHE_DIR='$FLASHINFER_CACHE_DIR'
export VLLM_CACHE_ROOT='$VLLM_CACHE_ROOT'
export NVTE_CUDA_ARCHS='$NVTE_CUDA_ARCHS'
export WANDB_MODE='$WANDB_MODE'
export WANDB_PROJECT='$WANDB_PROJECT'
export NRL_MXFP8_BATCHED_SHUFFLE='$NRL_MXFP8_BATCHED_SHUFFLE'
export NRL_REFIT_CACHED_LOADERS='$NRL_REFIT_CACHED_LOADERS'
export NRL_MXFP8_SHUFFLE_VERIFY=0
echo RUN_NAME='$RUN_NAME'
echo CONFIG_PATH='$CONFIG_PATH'
echo REPO_SHA=\$(git rev-parse HEAD)
git submodule status --recursive
NRL_FORCE_REBUILD_VENVS='${NRL_FORCE_REBUILD_VENVS:-true}' \
  uv run examples/run_grpo.py \
  --config '$CONFIG_PATH' \
  grpo.max_num_steps='$MAX_STEPS' \
  grpo.seed=42 \
  policy.train_global_batch_size=2048 \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  ++policy.generation.vllm_cfg.sleep_level=1 \
  ++policy.generation.vllm_cfg.refit_prequantize='$PREQUANTIZE' \
  policy.refit_persistent_ipc_buffers='$PERSISTENT_IPC' \
  ++policy.refit_buffer_size_gb=4 \
  policy.megatron_cfg.refit_slim_offload_after='$SLIM_OFFLOAD' \
  policy.megatron_cfg.pinned_reference_swap=false \
  checkpointing.enabled=false \
  logger.log_dir='$LOG_DIR' \
  logger.wandb_enabled=True \
  logger.wandb.project='$WANDB_PROJECT' \
  logger.wandb.name='$RUN_NAME' \
  logger.monitor_gpus=True \
  logger.tensorboard_enabled=True \
  2>&1 | tee '$RUN_LOG'"

cd "$REPO"
exec bash ray.sub
