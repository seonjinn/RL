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

NATIVE_TE_RUNTIME=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/runtimes/transformer-engine/transformer-engine-pr2898-4a18653fc7274b10e33cd786b91be6261c523dc0-wheel-029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e
NATIVE_TE_SITE_PACKAGES=${NATIVE_TE_RUNTIME}/site-packages
TASK7_RUNTIME_ARCHIVE_PREFIX=${NATIVE_TE_SITE_PACKAGES}:/root/.cache/uv/archive-v0/26H_iFoUOK00pyG5:/root/.cache/uv/archive-v0/ymbKBYrUysuiERDQ:/root/.cache/uv/archive-v0/Lp_mVBWGrC-sLPL6:/root/.cache/uv/archive-v0/kIpfdwf26Al4-BTb:/root/.cache/uv/archive-v0/i7-d_jifMXRoKKrY
EXPECTED_RUNTIME_ARCHIVE_PREFIX="${TASK7_RUNTIME_ARCHIVE_PREFIX}"
if [[ "${LAUNCHER_TEST_CONTRACT_OVERRIDE:-0}" == 1 ]]; then
  [[ "${SBATCH_TEST_ONLY:-0}" == 1 ]] || fail "LAUNCHER_TEST_CONTRACT_OVERRIDE is only allowed with SBATCH_TEST_ONLY=1"
  : "${MCORE_DRIVER_PYTHON_OVERRIDE:?LAUNCHER_TEST_CONTRACT_OVERRIDE requires MCORE_DRIVER_PYTHON_OVERRIDE}"
  : "${MCORE_LOCK_BLOB_OVERRIDE:?LAUNCHER_TEST_CONTRACT_OVERRIDE requires MCORE_LOCK_BLOB_OVERRIDE}"
  : "${RUNTIME_ARCHIVE_PREFIX_OVERRIDE:?LAUNCHER_TEST_CONTRACT_OVERRIDE requires RUNTIME_ARCHIVE_PREFIX_OVERRIDE}"
  : "${TE_NATIVE_PROVENANCE_OVERRIDE:?LAUNCHER_TEST_CONTRACT_OVERRIDE requires TE_NATIVE_PROVENANCE_OVERRIDE}"
  : "${CONTAINER_OVERRIDE:?LAUNCHER_TEST_CONTRACT_OVERRIDE requires CONTAINER_OVERRIDE}"
  MCORE_DRIVER_PYTHON="${MCORE_DRIVER_PYTHON_OVERRIDE}"
  MCORE_LOCK_BLOB="${MCORE_LOCK_BLOB_OVERRIDE}"
  RUNTIME_ARCHIVE_PREFIX="${RUNTIME_ARCHIVE_PREFIX_OVERRIDE}"
  EXPECTED_RUNTIME_ARCHIVE_PREFIX="${RUNTIME_ARCHIVE_PREFIX_OVERRIDE}"
  TE_NATIVE_SITE_PACKAGES="${RUNTIME_ARCHIVE_PREFIX%%:*}"
  TE_NATIVE_RUNTIME=$(dirname -- "${TE_NATIVE_SITE_PACKAGES}")
  TE_NATIVE_PROVENANCE="${TE_NATIVE_PROVENANCE_OVERRIDE}"
  CONTAINER="${CONTAINER_OVERRIDE}"
  MOUNTS="/lustre:/lustre"
fi

: "${MCORE_DRIVER_PYTHON:=}"
: "${MCORE_LOCK_BLOB:=}"
: "${RUNTIME_ARCHIVE_PREFIX:=}"
: "${TE_NATIVE_COMMIT:=}"
: "${TE_NATIVE_WHEEL_SHA256:=}"
: "${TE_NATIVE_RUNTIME:=}"
: "${TE_NATIVE_PROVENANCE:=}"
: "${TE_NATIVE_SITE_PACKAGES:=}"
IMMUTABLE_RUNTIME_PYTHONPATH="${RUNTIME_ARCHIVE_PREFIX}:${REPO_ROOT}:${REPO_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

MODEL=${MODEL:-nano-hybrid}
DROP_PAD_MOE_PAIR=${DROP_PAD_MOE_PAIR:-false}
case "${DROP_PAD_MOE_PAIR}" in
  false|true) ;;
  *) fail "DROP_PAD_MOE_PAIR must be false or true" ;;
esac

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

if [[ "${DROP_PAD_MOE_PAIR}" == true && "${MODEL}" != nano-hybrid ]]; then
  fail "Drop-pad MoE pair requires MODEL=nano-hybrid; Qwen Flex/HybridEP recipes are unsupported"
fi

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
MOE_EXPERT_CAPACITY_FACTOR=${MOE_EXPERT_CAPACITY_FACTOR:-}
MOE_PAD_EXPERT_INPUT_TO_CAPACITY=${MOE_PAD_EXPERT_INPUT_TO_CAPACITY:-}
case "${MOE_PAD_EXPERT_INPUT_TO_CAPACITY}" in
  ""|false|true) ;;
  *) fail "MOE_PAD_EXPERT_INPUT_TO_CAPACITY must be false or true" ;;
esac

RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}-${MODEL}-${CLUSTER}-${PHASE}-${RUN_TAG}"
LOG_ROOT=${LOG_ROOT_OVERRIDE:-exp_logs/mamba_moe_te_graph_20260729}
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION}}
TIME_LIMIT=${TIME_LIMIT_OVERRIDE:-${TIME_LIMIT}}

COMMAND_ARGS=(
  env
  NEMO_RL_REQUIRE_SYSTEM_MCORE=1
  "NEMO_RL_MCORE_SYSTEM_PYTHON=${MCORE_DRIVER_PYTHON}"
  NRL_FORCE_REBUILD_VENVS=true
  "PYTHONPATH=${IMMUTABLE_RUNTIME_PYTHONPATH}"
  "${MCORE_DRIVER_PYTHON}"
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
    ++policy.megatron_cfg.cuda_graph_impl=none
    "++policy.megatron_cfg.cuda_graph_modules=[]"
    ++policy.megatron_cfg.activation_checkpointing=false
  )
else
  COMMAND_ARGS+=(
    ++policy.megatron_cfg.cuda_graph_impl=transformer_engine
    "++policy.megatron_cfg.cuda_graph_modules=${SCOPE}"
    ++policy.megatron_cfg.cuda_graph_packed_seq=true
    "++policy.megatron_cfg.cuda_graph_max_packed_seqs=${MAX_PACKED_SEQS}"
    "++policy.megatron_cfg.cuda_graph_warmup_steps=${WARMUP_STEPS}"
    "++policy.megatron_cfg.cuda_graph_max_cached_schedules=${CACHE_CAPACITY}"
  )
  if [[ "${MOE_ACT_RECOMPUTE}" == true ]]; then
    COMMAND_ARGS+=(
      ++policy.megatron_cfg.activation_checkpointing=true
      ++policy.megatron_cfg.recompute_granularity=selective
      "++policy.megatron_cfg.recompute_modules=[moe_act]"
    )
  else
    COMMAND_ARGS+=(++policy.megatron_cfg.activation_checkpointing=false)
  fi
  if [[ -n "${MOE_SHARED_EXPERT_OVERLAP}" ]]; then
    COMMAND_ARGS+=(
      "++policy.megatron_cfg.moe_shared_expert_overlap=${MOE_SHARED_EXPERT_OVERLAP}"
    )
  fi
fi

if [[ -n "${MOE_EXPERT_CAPACITY_FACTOR}" ]]; then
  COMMAND_ARGS+=(
    "++policy.megatron_cfg.moe_expert_capacity_factor=${MOE_EXPERT_CAPACITY_FACTOR}"
  )
fi
if [[ -n "${MOE_PAD_EXPERT_INPUT_TO_CAPACITY}" ]]; then
  COMMAND_ARGS+=(
    "++policy.megatron_cfg.moe_pad_expert_input_to_capacity=${MOE_PAD_EXPERT_INPUT_TO_CAPACITY}"
  )
fi

printf -v COMMAND '%q ' "${COMMAND_ARGS[@]}"
COMMAND=${COMMAND% }
SBATCH_CMD=(
  sbatch
  --parsable
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
if [[ "${SBATCH_TEST_ONLY:-0}" == 1 ]]; then
  SBATCH_CMD=(sbatch --parsable --test-only "${SBATCH_CMD[@]:2}")
fi

unresolved=()
for field in \
  ACCOUNT \
  PARTITION \
  CONTAINER \
  CONTAINER_SHA256 \
  HF_HOME \
  HF_DATASETS_CACHE \
  MOUNTS \
  MCORE_DRIVER_PYTHON \
  MCORE_LOCK_BLOB \
  RUNTIME_ARCHIVE_PREFIX \
  TE_NATIVE_COMMIT \
  TE_NATIVE_WHEEL_SHA256 \
  TE_NATIVE_RUNTIME \
  TE_NATIVE_PROVENANCE \
  TE_NATIVE_SITE_PACKAGES \
  TE_EXPECTED_VERSION \
  NVTE_CUDA_ARCHS \
  UV_CACHE_DIR_OVERRIDE \
  SETUP_COMMAND \
  SETUP_COMMAND_ON_WORKERS \
  RAY_CLIENT_SERVER_ENABLED \
  RAY_DASHBOARD_ENABLED \
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
printf 'TE_NATIVE_COMMIT: %s\n' "${TE_NATIVE_COMMIT:-}"
printf 'TE_NATIVE_WHEEL_SHA256: %s\n' "${TE_NATIVE_WHEEL_SHA256:-}"
printf 'TE_NATIVE_RUNTIME: %s\n' "${TE_NATIVE_RUNTIME:-}"
printf 'TE_NATIVE_PROVENANCE: %s\n' "${TE_NATIVE_PROVENANCE:-}"
printf 'TE_NATIVE_SITE_PACKAGES: %s\n' "${TE_NATIVE_SITE_PACKAGES:-}"
printf 'TE_EXPECTED_VERSION: %s\n' "${TE_EXPECTED_VERSION:-}"
printf 'MCORE_DRIVER_PYTHON: %s\n' "${MCORE_DRIVER_PYTHON}"
printf 'MCORE_LOCK_BLOB: %s\n' "${MCORE_LOCK_BLOB}"
printf 'IMMUTABLE_RUNTIME_PYTHONPATH: %s\n' "${IMMUTABLE_RUNTIME_PYTHONPATH}"
printf 'MOUNTS: %s\n' "${MOUNTS}"
printf 'SETUP_COMMAND: %s\n' "${SETUP_COMMAND}"
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

preflight_runtime_contract() {
  [[ "$(git hash-object uv.lock)" == "${MCORE_LOCK_BLOB}" ]] || fail "uv.lock hash mismatch for locked MCore environment"
  [[ "${RUNTIME_ARCHIVE_PREFIX}" == "${EXPECTED_RUNTIME_ARCHIVE_PREFIX}" ]] || fail "immutable runtime archive prefix mismatch"
  [[ "${TE_NATIVE_SITE_PACKAGES}" == "${RUNTIME_ARCHIVE_PREFIX%%:*}" ]] || fail "Transformer Engine native runtime must be first on PYTHONPATH"
  [[ "${TE_NATIVE_RUNTIME}" == "$(dirname -- "${TE_NATIVE_SITE_PACKAGES}")" ]] || fail "Transformer Engine native runtime path mismatch"
  [[ "${TE_NATIVE_PROVENANCE}" == "${TE_NATIVE_RUNTIME}/provenance.json" ]] || fail "Transformer Engine native provenance path mismatch"
  [[ "${MOUNTS}" == "/lustre:/lustre" ]] || fail "Transformer Engine native runtime mount mismatch"
  [[ -r "${CONTAINER}" ]] || fail "pinned container is not readable: ${CONTAINER}"
  [[ -r "${REPO_ROOT}" ]] || fail "repository source is not readable: ${REPO_ROOT}"
  [[ -r "${REPO_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src" ]] || fail "Bridge source is not readable"
  [[ -r "${REPO_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" ]] || fail "Megatron-LM source is not readable"

  if [[ "${LAUNCHER_TEST_CONTRACT_OVERRIDE:-0}" != 1 ]]; then
    [[ "${MCORE_DRIVER_PYTHON}" == /lustre/* ]] || fail "locked MCore interpreter must be on Lustre"
  fi
  if [[ -L "${MCORE_DRIVER_PYTHON}" ]]; then
    local target
    target=$(readlink "${MCORE_DRIVER_PYTHON}")
    case "${target}" in
      /root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python|/root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13) ;;
      *) fail "locked MCore interpreter has an unusable symlink target: ${target}" ;;
    esac
  elif [[ ! -x "${MCORE_DRIVER_PYTHON}" ]]; then
    fail "locked MCore interpreter is not executable: ${MCORE_DRIVER_PYTHON}"
  fi

  [[ -d "${TE_NATIVE_SITE_PACKAGES}" && -r "${TE_NATIVE_SITE_PACKAGES}" ]] || fail "Transformer Engine native site-packages is not readable"
  [[ -f "${TE_NATIVE_PROVENANCE}" && -r "${TE_NATIVE_PROVENANCE}" ]] || fail "Transformer Engine native provenance is not readable"
  if ! python3 \
    experiments/cuda_graph/mamba_moe_te_graph_20260729/validate_te_native_runtime.py \
    --provenance "${TE_NATIVE_PROVENANCE}" \
    --site-packages "${TE_NATIVE_SITE_PACKAGES}" \
    --expected-commit "${TE_NATIVE_COMMIT}" \
    --expected-wheel-sha256 "${TE_NATIVE_WHEEL_SHA256}" \
    --expected-image "${CONTAINER}" \
    --expected-image-sha256 "${CONTAINER_SHA256}" \
    --expected-version "${TE_EXPECTED_VERSION}"; then
    fail "Transformer Engine native runtime provenance mismatch"
  fi
  printf 'RUNTIME_PREFLIGHT: passed\n'
}

preflight_runtime_contract

mkdir -p "${RUN_LOG_DIR}"
job_id="$(
COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
CONTAINER_SHA256="${CONTAINER_SHA256}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="${IMMUTABLE_RUNTIME_PYTHONPATH}" \
RUNTIME_ARCHIVE_PREFIX="${RUNTIME_ARCHIVE_PREFIX}" \
NEMO_RL_REQUIRE_SYSTEM_MCORE=1 \
NEMO_RL_MCORE_SYSTEM_PYTHON="${MCORE_DRIVER_PYTHON}" \
NRL_FORCE_REBUILD_VENVS=true \
NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" \
UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE}" \
SETUP_COMMAND="${SETUP_COMMAND}" \
SETUP_COMMAND_ON_WORKERS="${SETUP_COMMAND_ON_WORKERS}" \
RAY_CLIENT_SERVER_ENABLED="${RAY_CLIENT_SERVER_ENABLED}" \
RAY_DASHBOARD_ENABLED="${RAY_DASHBOARD_ENABLED}" \
WANDB_MODE="${WANDB_MODE_OVERRIDE:-offline}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
MOUNTS="${MOUNTS}" \
BASE_LOG_DIR="${RUN_LOG_DIR}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
"${SBATCH_CMD[@]}"
)"
printf 'SLURM_JOB_ID: %s\n' "${job_id}"
