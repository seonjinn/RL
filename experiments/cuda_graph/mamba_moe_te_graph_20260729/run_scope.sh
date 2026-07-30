#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "$*" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

PHASE=${PHASE:-smoke}
case "${PHASE}" in
  smoke) DEFAULT_STEPS=5 ;;
  performance) DEFAULT_STEPS=20 ;;
  accuracy) DEFAULT_STEPS=100 ;;
  *) fail "PHASE must be smoke, performance, or accuracy" ;;
esac

: "${SCOPE:?SCOPE must be set by a persistent scope or variant script}"
: "${SCOPE_NAME:?SCOPE_NAME must be set by a persistent scope or variant script}"
: "${CUDA_GRAPH_IMPL:?CUDA_GRAPH_IMPL must be set by a persistent scope or variant script}"
: "${CLUSTER:?CLUSTER must be ptyche or oci-hsg}"

case "${CUDA_GRAPH_IMPL}:${SCOPE}" in
  "none:[no_cg]") ;;
  transformer_engine:*)
    valid_scope=false
    for attn in "" attn; do
      for mlp in "" mlp; do
        for mamba in "" mamba; do
          dense_scope="${attn}"
          if [[ -n "${mlp}" ]]; then
            [[ -n "${dense_scope}" ]] && dense_scope+=","
            dense_scope+="${mlp}"
          fi
          if [[ -n "${mamba}" ]]; then
            [[ -n "${dense_scope}" ]] && dense_scope+=","
            dense_scope+="${mamba}"
          fi
          for moe_axis in "" moe moe_router "moe_router,moe_preprocess"; do
            candidate_scope="${dense_scope}"
            if [[ -n "${moe_axis}" ]]; then
              [[ -n "${candidate_scope}" ]] && candidate_scope+=","
              candidate_scope+="${moe_axis}"
            fi
            candidate="[${candidate_scope}]"
            if [[ "${SCOPE}" == "${candidate}" ]]; then
              valid_scope=true
            fi
          done
        done
      done
    done
    [[ "${valid_scope}" == true ]] || fail "Unsupported TE graph SCOPE: ${SCOPE}"
    ;;
  *) fail "Unsupported CUDA graph implementation/scope pair: ${CUDA_GRAPH_IMPL}:${SCOPE}" ;;
esac

case "${CLUSTER}" in
  ptyche|oci-hsg) ;;
  *) fail "CLUSTER must be ptyche or oci-hsg" ;;
esac
PROFILE="${SCRIPT_DIR}/profiles/${CLUSTER}.env"
[[ -f "${PROFILE}" ]] || fail "Missing cluster profile: ${PROFILE}"
source "${PROFILE}"

MODEL=${MODEL:-nano-hybrid}
case "${MODEL}" in
  nano-hybrid)
    CONFIG=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml
    MODEL_SNAPSHOT=${NANOV3_MODEL_SNAPSHOT:-}
    TOKENIZER_SNAPSHOT=${NANOV3_TOKENIZER_SNAPSHOT:-}
    PRETRAINED_CHECKPOINT=${NANOV3_PRETRAINED_CHECKPOINT:-}
    TOTAL_NODES=${NANOV3_TOTAL_NODES:-}
    INFERENCE_NODES=${NANOV3_INFERENCE_NODES:-}
    ;;
  qwen3-30b-a3b)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
    MODEL_SNAPSHOT=${QWEN3_30BA3B_SNAPSHOT:-}
    TOKENIZER_SNAPSHOT=${QWEN3_30BA3B_SNAPSHOT:-}
    PRETRAINED_CHECKPOINT=${QWEN3_30BA3B_PRETRAINED_CHECKPOINT:-}
    TOTAL_NODES=${QWEN3_30BA3B_TOTAL_NODES:-}
    INFERENCE_NODES=${QWEN3_30BA3B_INFERENCE_NODES:-}
    ;;
  qwen3-235b-a22b)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml
    [[ -f "${REPO_ROOT}/${CONFIG}" ]] || fail "Qwen3-235B profile requires existing recipe: ${CONFIG}"
    MODEL_SNAPSHOT=${QWEN3_235BA22B_SNAPSHOT:-}
    TOKENIZER_SNAPSHOT=${QWEN3_235BA22B_SNAPSHOT:-}
    PRETRAINED_CHECKPOINT=${QWEN3_235BA22B_PRETRAINED_CHECKPOINT:-}
    TOTAL_NODES=${QWEN3_235BA22B_TOTAL_NODES:-}
    INFERENCE_NODES=${QWEN3_235BA22B_INFERENCE_NODES:-}
    ;;
  *) fail "MODEL must be nano-hybrid, qwen3-30b-a3b, or qwen3-235b-a22b" ;;
esac

if [[ "${MODEL}" == qwen3-* && "${SCOPE}" == *mamba* ]]; then
  fail "${MODEL} recipe has no Mamba layers; Mamba graph scopes are invalid"
fi

[[ -f "${REPO_ROOT}/${CONFIG}" ]] || fail "Missing immutable base recipe: ${CONFIG}"
[[ "${WARMUP_STEPS:-}" == 3 ]] || fail "WARMUP_STEPS must be 3"
[[ "${CACHE_CAPACITY:-}" == 2 ]] || fail "CACHE_CAPACITY must be 2"
[[ "${MAX_PACKED_SEQS:-}" == 16 ]] || fail "MAX_PACKED_SEQS must be 16"
[[ "${CHECKPOINTING_ENABLED:-}" == false ]] || fail "CHECKPOINTING_ENABLED must be false"
[[ "${WANDB_PROJECT:-}" == sna-cg-study ]] || fail "WANDB_PROJECT must be sna-cg-study"

STEPS=${STEPS:-${DEFAULT_STEPS}}
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "STEPS must be a positive integer"

MOE_SHARED_EXPERT_OVERLAP=${MOE_SHARED_EXPERT_OVERLAP:-}
case "${MOE_SHARED_EXPERT_OVERLAP}" in
  ""|false|true) ;;
  *) fail "MOE_SHARED_EXPERT_OVERLAP must be false or true" ;;
esac
MOE_ACT_RECOMPUTE=${MOE_ACT_RECOMPUTE:-false}
case "${MOE_ACT_RECOMPUTE}" in
  false|true) ;;
  *) fail "MOE_ACT_RECOMPUTE must be false or true" ;;
esac

RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}-${MODEL}-${CLUSTER}-${PHASE}-${RUN_TAG}"
LOG_ROOT=${LOG_ROOT_OVERRIDE:-exp_logs/mamba_moe_te_graph_20260729}
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION}}
TIME_LIMIT=${TIME_LIMIT_OVERRIDE:-${TIME_LIMIT}}

COMMAND_ARGS=(
  env
  NRL_FORCE_REBUILD_VENVS=true
  uv
  run
  --frozen
  --extra
  mcore
  examples/run_grpo.py
  --config
  "${CONFIG}"
  "policy.model_name=${MODEL_SNAPSHOT}"
  "policy.tokenizer.name=${TOKENIZER_SNAPSHOT}"
  "cluster.num_nodes=${TOTAL_NODES}"
  "cluster.gpus_per_node=${GPUS_PER_NODE}"
  "cluster.segment_size=${SEGMENT_SIZE}"
  policy.sequence_packing.enabled=true
  policy.generation.colocated.enabled=false
  "policy.generation.colocated.resources.num_nodes=${INFERENCE_NODES}"
  "policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE}"
  "grpo.max_num_steps=${STEPS}"
  checkpointing.enabled=false
  "logger.log_dir=${RUN_LOG_DIR}"
  logger.wandb_enabled=true
  logger.tensorboard_enabled=true
  "logger.wandb.project=${WANDB_PROJECT}"
  "logger.wandb.name=${RUN_NAME}"
  "+checkpointing.pretrained_checkpoint.path=${PRETRAINED_CHECKPOINT}"
  +checkpointing.pretrained_checkpoint.format=megatron_lm
)

if [[ "${CUDA_GRAPH_IMPL}" == none ]]; then
  COMMAND_ARGS+=(
    policy.megatron_cfg.cuda_graph_impl=none
    "policy.megatron_cfg.cuda_graph_modules=[]"
    policy.megatron_cfg.activation_checkpointing=false
  )
else
  COMMAND_ARGS+=(
    policy.megatron_cfg.cuda_graph_impl=transformer_engine
    "policy.megatron_cfg.cuda_graph_modules=${SCOPE}"
    policy.megatron_cfg.cuda_graph_packed_seq=true
    "policy.megatron_cfg.cuda_graph_max_packed_seqs=${MAX_PACKED_SEQS}"
    "policy.megatron_cfg.cuda_graph_warmup_steps=${WARMUP_STEPS}"
    "policy.megatron_cfg.cuda_graph_max_cached_schedules=${CACHE_CAPACITY}"
  )
  if [[ "${MOE_ACT_RECOMPUTE}" == true ]]; then
    COMMAND_ARGS+=(
      policy.megatron_cfg.activation_checkpointing=true
      policy.megatron_cfg.recompute_granularity=selective
      "policy.megatron_cfg.recompute_modules=[moe_act]"
    )
  else
    COMMAND_ARGS+=(policy.megatron_cfg.activation_checkpointing=false)
  fi
  if [[ -n "${MOE_SHARED_EXPERT_OVERLAP}" ]]; then
    COMMAND_ARGS+=(
      "policy.megatron_cfg.moe_shared_expert_overlap=${MOE_SHARED_EXPERT_OVERLAP}"
    )
  fi
fi

printf -v COMMAND '%q ' "${COMMAND_ARGS[@]}"
COMMAND=${COMMAND% }
SBATCH_CMD=(
  sbatch
  "--nodes=${TOTAL_NODES}"
)
set +u
SBATCH_CMD+=("${SBATCH_GPU_ARGS[@]}")
set -u
SBATCH_CMD+=(
  "--account=${ACCOUNT}"
  "--job-name=${ACCOUNT}-sna.${RUN_NAME}"
  "--partition=${PARTITION}"
  "--time=${TIME_LIMIT}"
  "--segment=${SEGMENT_SIZE}"
  "--output=${RUN_LOG_DIR}/slurm-%j.out"
  "--error=${RUN_LOG_DIR}/slurm-%j.out"
  ray.sub
)

unresolved=()
for field in \
  ACCOUNT \
  PARTITION \
  CONTAINER \
  CONTAINER_SHA256 \
  HF_HOME \
  HF_DATASETS_CACHE \
  MOUNTS \
  NVTE_CUDA_ARCHS \
  UV_CACHE_DIR_OVERRIDE \
  SETUP_COMMAND \
  RAY_CLIENT_SERVER_ENABLED \
  MODEL_SNAPSHOT \
  TOKENIZER_SNAPSHOT \
  PRETRAINED_CHECKPOINT \
  TOTAL_NODES \
  INFERENCE_NODES; do
  value=${!field:-}
  case "${value}" in
    ""|__REQUIRED_*__) unresolved+=("${field}") ;;
  esac
done

if ((${#unresolved[@]})); then
  printf 'UNRESOLVED:'
  printf ' %s' "${unresolved[@]}"
  printf '\n'
else
  printf 'UNRESOLVED: none\n'
fi
printf 'PROFILE: %s\n' "${PROFILE_ID}"
printf 'CONTAINER_SHA256: %s\n' "${CONTAINER_SHA256}"
printf 'COMMAND:\n%s\n' "${COMMAND}"
printf 'SBATCH:'
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'

if [[ "${TEST_ONLY:-0}" == 1 ]]; then
  printf 'TEST_ONLY: no submission performed\n'
  exit 0
fi
if ((${#unresolved[@]})); then
  fail "Refusing submission with unresolved fields: ${unresolved[*]}"
fi

mkdir -p "${RUN_LOG_DIR}"
COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
CONTAINER_SHA256="${CONTAINER_SHA256}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" \
UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE}" \
SETUP_COMMAND="${SETUP_COMMAND}" \
RAY_CLIENT_SERVER_ENABLED="${RAY_CLIENT_SERVER_ENABLED}" \
WANDB_MODE="${WANDB_MODE_OVERRIDE:-offline}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
MOUNTS="${MOUNTS}" \
BASE_LOG_DIR="${RUN_LOG_DIR}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
"${SBATCH_CMD[@]}"
