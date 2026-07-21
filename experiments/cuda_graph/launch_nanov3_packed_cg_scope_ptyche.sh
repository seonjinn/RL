#!/usr/bin/env bash

# Validate or submit one packed-sequence CUDA Graph functional run for
# Nemotron-3 Nano 30B-A3B on Ptyche. The initial `nocg` run creates only the
# reusable Megatron conversion cache; training checkpoint saves stay disabled.
#
# Examples:
#   SCOPE_CASE=nocg STEPS=1 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh
#   SCOPE_CASE=attn-moe-router STEPS=5 SUBMIT=1 ./experiments/cuda_graph/launch_nanov3_packed_cg_scope_ptyche.sh

set -euo pipefail

SCOPE_CASE=${SCOPE_CASE:?Set SCOPE_CASE.}
STEPS=${STEPS:-5}
RUN_TAG=${RUN_TAG:-nanov3-packed-cg-${SCOPE_CASE}-steps${STEPS}}
ADAPTER_WORKTREE=${ADAPTER_WORKTREE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-adapter-ptyche-20260719}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf}
MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/checkpoints/nanov3-30b-a3b-pr5672-20260720}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}

CG_ARGS=()
case "${SCOPE_CASE}" in
  nocg)
    CUDA_GRAPH_SCOPE=none
    ;;
  attn)
    CUDA_GRAPH_SCOPE='[attn]'
    ;;
  mamba)
    CUDA_GRAPH_SCOPE='[mamba]'
    ;;
  mlp)
    CUDA_GRAPH_SCOPE='[mlp]'
    ;;
  moe-router)
    CUDA_GRAPH_SCOPE='[moe_router]'
    ;;
  attn-moe-router)
    CUDA_GRAPH_SCOPE='[attn,moe_router]'
    ;;
  router-preprocess)
    CUDA_GRAPH_SCOPE='[moe_router,moe_preprocess]'
    ;;
  all-safe)
    CUDA_GRAPH_SCOPE='[attn,mamba,moe_router,moe_preprocess]'
    ;;
  moe-act)
    echo 'moe_act is an activation-recompute module, not a CUDA Graph scope' >&2
    exit 2
    ;;
  moe)
    echo 'full moe CUDA Graph requires drop-and-pad MoE and is excluded from this packed all-to-all workload' >&2
    exit 2
    ;;
  *)
    echo "Unknown SCOPE_CASE: ${SCOPE_CASE}" >&2
    exit 2
    ;;
esac

if [[ "${SCOPE_CASE}" != "nocg" ]]; then
  CG_ARGS=(
    "+policy.megatron_cfg.cuda_graph_impl=transformer_engine"
    "+policy.megatron_cfg.cuda_graph_scope=${CUDA_GRAPH_SCOPE}"
    "+policy.megatron_cfg.cuda_graph_warmup_steps=3"
    "+policy.megatron_cfg.cuda_graph_packed_seq=true"
    "+policy.megatron_cfg.cuda_graph_pr5672_thd=true"
    "+policy.megatron_cfg.cuda_graph_max_packed_seqs=512"
    "+policy.megatron_cfg.cuda_graph_buckets=[8192]"
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "scope_case=${SCOPE_CASE}"
  echo "cuda_graph_scope=${CUDA_GRAPH_SCOPE}"
  echo "cuda_graph_packed_seq=true"
  echo "cuda_graph_warmup_steps=3"
  exit 0
fi

if [[ ! -s "${HF_HOME}/token" ]]; then
  echo "Missing Hugging Face token at ${HF_HOME}/token" >&2
  exit 2
fi

WORKTREE=${ADAPTER_WORKTREE}
CONFIG="${WORKTREE}/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml"
if [[ ! -f "${WORKTREE}/ray.sub" || ! -f "${CONFIG}" ]]; then
  echo "Missing worktree, ray.sub, or NanoV3 recipe under ${WORKTREE}" >&2
  exit 2
fi

LOG_BASE="${WORKTREE}/experiments/cuda_graph/logs"
mkdir -p "${LOG_BASE}"

echo "scope_case=${SCOPE_CASE} steps=${STEPS}"
echo "worktree=${WORKTREE} config=${CONFIG}"
echo "cuda_graph_scope=${CUDA_GRAPH_SCOPE}"
git -C "${WORKTREE}" rev-parse HEAD
git -C "${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM" rev-parse HEAD

BASE_ARGS=(
  "grpo.max_num_steps=${STEPS}"
  "grpo.val_period=10"
  "checkpointing.enabled=false"
  "logger.wandb_enabled=false"
  "logger.tensorboard_enabled=false"
  "logger.log_dir=logs/nanov3-30b-a3b-cg/${RUN_TAG}"
  "logger.wandb.name=${RUN_TAG}"
  "${CG_ARGS[@]}"
)
printf -v RUN_ARGS ' %q' "${BASE_ARGS[@]}"
COMMAND=$(printf '%s\n' \
  "cd ${WORKTREE}" \
  'export NRL_IGNORE_VERSION_MISMATCH=1' \
  "export NRL_MEGATRON_CHECKPOINT_DIR=${MODEL_CACHE_ROOT}" \
  "export PYTHONPATH=${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:\${PYTHONPATH:-}" \
  "uv run --locked --extra mcore --directory ${WORKTREE} python ${WORKTREE}/examples/run_grpo.py --config ${CONFIG}${RUN_ARGS}")

submit() {
  (
    export COMMAND CONTAINER HF_HOME RAY_LOG_SYNC_FREQUENCY
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export MOUNTS="/lustre:/lustre"
    export GPUS_PER_NODE=4
    export BASE_LOG_DIR="${LOG_BASE}"
    sbatch "$@" --nodes=4 --segment=4 --exclusive --account="${ACCOUNT}" --partition="${PARTITION}" --time=04:00:00 --job-name="${ACCOUNT}-nanov3.${SCOPE_CASE}" "${WORKTREE}/ray.sub"
  )
}

submit --test-only
echo "Submission validated. Set SUBMIT=1 to submit this scope."
if [[ "${SUBMIT:-0}" == "1" ]]; then
  submit
fi
