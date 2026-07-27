#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment=metrics

set -euo pipefail

: "${ARM:?ARM is required}"
: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${REFIT_OPT:?REFIT_OPT is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${NUM_NODES:?NUM_NODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
: "${EXPECTED_REPO_SHA:?EXPECTED_REPO_SHA is required}"

BASE=${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}
WORK=${WORK:-$BASE/experiments/mxfp8-qkvo-nano}
CONTAINER=${CONTAINER:-$BASE/containers/nemo_rl_nightly.sqsh}
NANO_MODEL_PATH=${NANO_MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/models/nemotron-nano3/Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf}
CONFIG_PATH=examples/configs/recipes/llm/${CONFIG_NAME}.yaml
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

case "$ARM" in
  bf16)
    ROLLOUT_PRECISION=bf16
    QUANTIZATION_SCOPE=none
    VLLM_BACKEND_OVERRIDE="++policy.generation.vllm_kwargs.moe_backend=triton"
    ;;
  moe-baseline | moe-optimized)
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=moe-only
    VLLM_BACKEND_OVERRIDE=
    ;;
  qkvo-baseline | qkvo-optimized)
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=moe+qkvo
    VLLM_BACKEND_OVERRIDE=
    ;;
  *)
    echo "Unsupported arm: $ARM" >&2
    exit 2
    ;;
esac

git -C "$REPO" rev-parse --git-dir >/dev/null
test -f "$REPO/$CONFIG_PATH"
test -f "$NANO_MODEL_PATH/config.json"
test -f "$CONTAINER"
ACTUAL_REPO_SHA=$(git -C "$REPO" rev-parse HEAD)
if [[ "$ACTUAL_REPO_SHA" != "$EXPECTED_REPO_SHA" ]]; then
  echo "Repository changed after submission: expected $EXPECTED_REPO_SHA, got $ACTUAL_REPO_SHA" >&2
  exit 3
fi
mkdir -p "$WORK/slurm" "$LOG_DIR" "$BASE/uv_cache/mxfp8-qkvo-nano"

if [[ -f "$HOME/.nemo_rl_tokens" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.nemo_rl_tokens"
fi

WANDB_AUTH_SOURCE=environment
if [[ -f "$HOME/.netrc" ]]; then
  WANDB_NETRC_KEY=$(awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "machine") {
          wandb_machine = ($(i + 1) == "api.wandb.ai")
        } else if (wandb_machine && $i == "password") {
          print $(i + 1)
          exit
        }
      }
    }
  ' "$HOME/.netrc")
  if [[ -n "$WANDB_NETRC_KEY" ]]; then
    export WANDB_API_KEY=$WANDB_NETRC_KEY
    WANDB_AUTH_SOURCE=netrc-host
  fi
fi

export CONTAINER
export MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre,/project:/project}
export NUM_NODES
export GPUS_PER_NODE
export HF_HOME=${NRL_HF_HOME:-$BASE/hf_home}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export UV_CACHE_DIR=$BASE/uv_cache/mxfp8-qkvo-nano
export UV_HTTP_TIMEOUT=300
export UV_LOCK_TIMEOUT=3600
export BASE_LOG_DIR=$WORK/slurm
export XDG_CACHE_HOME=/tmp/nemo_rl_mxfp8_qkvo_nano_${SLURM_JOB_ID}
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
export WANDB_PROJECT=${WANDB_PROJECT:-sna-mxfp8-qkvo-nano}
export WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
export EXPERIMENT_CLUSTER=${EXPERIMENT_CLUSTER:-unknown}
export TENSORBOARD_ENABLED=${TENSORBOARD_ENABLED:-True}

{
  echo "arm=$ARM"
  echo "config=$CONFIG_PATH"
  echo "repo_sha=$ACTUAL_REPO_SHA"
  echo "container=$CONTAINER"
  echo "model_path=$NANO_MODEL_PATH"
  echo "cluster=$EXPERIMENT_CLUSTER"
  echo "nodes=$NUM_NODES"
  echo "gpus_per_node=$GPUS_PER_NODE"
  echo "total_gpus=$((NUM_NODES * GPUS_PER_NODE))"
  echo "max_steps=$MAX_STEPS"
  echo "train_global_batch_size=16"
  echo "seed=42"
  echo "real_importance_sampling=true"
  echo "rollout_precision=$ROLLOUT_PRECISION"
  echo "quantization_scope=$QUANTIZATION_SCOPE"
  echo "vllm_tensor_parallel_size=1"
  echo "vllm_gpu_memory_utilization=0.5"
  echo "vllm_backend_override=$VLLM_BACKEND_OVERRIDE"
  echo "megatron_tensor_parallel_size=2"
  echo "megatron_pipeline_parallel_size=2"
  echo "megatron_context_parallel_size=2"
  echo "megatron_expert_parallel_size=8"
  echo "wandb_entity=$WANDB_ENTITY"
  echo "wandb_project=$WANDB_PROJECT"
  echo "tensorboard_enabled=$TENSORBOARD_ENABLED"
  echo "pinned_reference_swap=false"
  echo "refit_prequantize=$PREQUANTIZE"
  echo "refit_persistent_ipc_buffers=$PERSISTENT_IPC"
  echo "refit_slim_offload_after=$SLIM_OFFLOAD"
  echo "NRL_MXFP8_BATCHED_SHUFFLE=$NRL_MXFP8_BATCHED_SHUFFLE"
  echo "NRL_REFIT_CACHED_LOADERS=$NRL_REFIT_CACHED_LOADERS"
} >"$EXP_DIR/metadata.env"

export COMMAND="set -euo pipefail
cd '$REPO'
echo WANDB_AUTH_SOURCE='$WANDB_AUTH_SOURCE'
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
export WANDB_ENTITY='$WANDB_ENTITY'
export NRL_MXFP8_BATCHED_SHUFFLE='$NRL_MXFP8_BATCHED_SHUFFLE'
export NRL_REFIT_CACHED_LOADERS='$NRL_REFIT_CACHED_LOADERS'
export NRL_MXFP8_SHUFFLE_VERIFY=0
export PYTHONPATH='$REPO':\${PYTHONPATH:-}
export NRL_FORCE_REBUILD_VENVS=false
echo RUN_NAME='$RUN_NAME'
echo CONFIG_PATH='$CONFIG_PATH'
echo MODEL_PATH='$NANO_MODEL_PATH'
echo REPO_SHA=\$(git rev-parse HEAD)
git submodule status --recursive
RAY_CLI_VERSION=\$(ray --version | sed -n 's/^ray, version //p')
DRIVER_RAY_VERSION=\$(/opt/nemo_rl_venv/bin/python -c 'import ray; print(ray.__version__)')
DRIVER_NRL_SOURCE=\$(/opt/nemo_rl_venv/bin/python -c 'import nemo_rl; print(nemo_rl.__file__)')
VLLM_WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
test -x \"\$VLLM_WORKER_PYTHON\"
VLLM_WORKER_VERSION=\$(\"\$VLLM_WORKER_PYTHON\" -c 'import vllm; print(vllm.__version__)')
echo RAY_CLI_VERSION=\$RAY_CLI_VERSION
echo DRIVER_RAY_VERSION=\$DRIVER_RAY_VERSION
echo DRIVER_NRL_SOURCE=\$DRIVER_NRL_SOURCE
echo VLLM_WORKER_VERSION=\$VLLM_WORKER_VERSION
if [[ \"\$RAY_CLI_VERSION\" != \"\$DRIVER_RAY_VERSION\" ]]; then
  echo 'Ray version mismatch before driver launch' >&2
  exit 3
fi
case \"\$DRIVER_NRL_SOURCE\" in
  '$REPO'/nemo_rl/*) ;;
  *)
    echo 'Driver did not import NeMo-RL from the experiment checkout' >&2
    exit 3
    ;;
esac
\"\$VLLM_WORKER_PYTHON\" -c 'from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8FusedMoE, ModelOptMxFp8LinearMethod; assert all(hasattr(ModelOptMxFp8FusedMoE, name) for name in (\"create_weights\", \"process_weights_after_loading\", \"apply_monolithic\")); assert hasattr(ModelOptMxFp8LinearMethod, \"process_weights_after_loading\")'
if [[ '$WANDB_MODE' != offline && '$WANDB_MODE' != disabled ]]; then
  /opt/nemo_rl_venv/bin/python -c 'import wandb; assert wandb.login(verify=True), \"W&B login verification failed\"'
fi
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config '$CONFIG_PATH' \
  grpo.max_num_steps='$MAX_STEPS' \
  grpo.seed=42 \
  policy.model_name='$NANO_MODEL_PATH' \
  policy.tokenizer.name='$NANO_MODEL_PATH' \
  cluster.num_nodes='$NUM_NODES' \
  cluster.gpus_per_node='$GPUS_PER_NODE' \
  cluster.segment_size='$NUM_NODES' \
  policy.train_global_batch_size=16 \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
  $VLLM_BACKEND_OVERRIDE \
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
  +logger.wandb.entity='$WANDB_ENTITY' \
  logger.wandb.name='$RUN_NAME' \
  logger.monitor_gpus=True \
  logger.tensorboard_enabled='$TENSORBOARD_ENABLED' \
  2>&1 | tee '$RUN_LOG'"

cd "$REPO"
exec bash ray.sub
