#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-test-only}
CLUSTER=${CLUSTER:?Set CLUSTER to ptyche or lyris}
MODEL=${MODEL:?Set MODEL to qwen30, qwen35, or nano35}
MODE=${MODE:?Set MODE to sync or async}
FP8_PARAM=${FP8_PARAM:?Set FP8_PARAM to false or true}

EXPECTED_HEAD=${EXPECTED_HEAD:?Set EXPECTED_HEAD to the pushed experiment commit}
REPO=${REPO:-/home/sna/worktrees/RL-mxfp8-training-matrix-20260907}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
BASE=${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
RESULT_ROOT=${RESULT_ROOT:-${BASE}/results/mxfp8-training-matrix-20260907}
MAX_STEPS=${MAX_STEPS:-20}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WALLTIME=${WALLTIME:-04:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

case "${ACTION}" in
  submit) SBATCH_ACTION=(--parsable) ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
esac

case "${CLUSTER}" in
  ptyche)
    PARTITION=${PARTITION:-batch}
    CONTAINER=${CONTAINER:-${BASE}/containers/nemo_rl_nightly_candidate_20260907_2750074.sqsh}
    ;;
  lyris)
    PARTITION=${PARTITION:-gb200}
    CONTAINER=${CONTAINER:-${BASE}/containers/nemo_rl_nightly_20260905_2928135.sqsh}
    ;;
  *) echo "CLUSTER must be ptyche or lyris" >&2; exit 2 ;;
esac

case "${MODEL}:${MODE}" in
  qwen30:sync)
    CONFIG=experiments/mxfp8_training_matrix_20260907/configs/qwen30-sync.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    ;;
  qwen30:async)
    CONFIG=experiments/mxfp8_training_matrix_20260907/configs/qwen30-async.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    ;;
  qwen35:sync)
    CONFIG=experiments/mxfp8_training_matrix_20260907/configs/qwen35-sync.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    ;;
  qwen35:async)
    CONFIG=experiments/mxfp8_training_matrix_20260907/configs/qwen35-async.yaml
    NUM_NODES=8
    SEGMENT_SIZE=2
    ;;
  nano35:sync)
    CONFIG=experiments/mxfp8_training_matrix_20260907/configs/nano35-sync.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    ;;
  nano35:async)
    CONFIG=experiments/mxfp8_training_matrix_20260907/configs/nano35-async.yaml
    NUM_NODES=8
    SEGMENT_SIZE=2
    ;;
  *) echo "Unsupported MODEL:MODE pair: ${MODEL}:${MODE}" >&2; exit 2 ;;
esac

case "${FP8_PARAM}" in
  false)
    TE_CONFIG=experiments/mxfp8_training_matrix_20260907/te_mxfp8_moe_fp8param_false.yaml
    ;;
  true)
    TE_CONFIG=experiments/mxfp8_training_matrix_20260907/te_mxfp8_moe_fp8param_true.yaml
    ;;
  *) echo "FP8_PARAM must be false or true" >&2; exit 2 ;;
esac

if [[ "${MODE}" == "sync" && "${FP8_PARAM}" == "false" ]]; then
  REFIT_PREQUANTIZE=true
else
  REFIT_PREQUANTIZE=false
fi

test -f "${REPO}/${CONFIG}"
test -f "${REPO}/${TE_CONFIG}"
test -f "${REPO}/ray.sub"
test -f "${CONTAINER}"
test -d "${HF_HOME}"

LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
if [[ "${LOCAL_HEAD}" != "${EXPECTED_HEAD}" ]]; then
  echo "Expected ${EXPECTED_HEAD}; found ${LOCAL_HEAD}" >&2
  exit 2
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned submodules must be initialized" >&2
  exit 2
fi

RUN_NAME="${MODEL}-${MODE}-mxfp8-train-fp8param-${FP8_PARAM}-mxfp8-rollout-${RUN_SUFFIX}"
BASE_LOG_DIR="${RESULT_ROOT}/${RUN_NAME}"
mkdir -p "${BASE_LOG_DIR}"

WANDB_API_KEY=${WANDB_API_KEY:-$(awk '
  $1 == "machine" && $2 == "api.wandb.ai" { found = 1 }
  found && $1 == "password" { print $2; exit }
' "${HOME}/.netrc" 2>/dev/null || true)}
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WANDB_API_KEY is required" >&2
  exit 2
fi

cat >"${BASE_LOG_DIR}/metadata.env" <<EOF
source_sha=${LOCAL_HEAD}
cluster=${CLUSTER}
hardware=GB200
container=${CONTAINER}
config=${CONFIG}
te_config=${TE_CONFIG}
model=${MODEL}
mode=${MODE}
training_precision=mxfp8
fp8_param=${FP8_PARAM}
rollout_precision=mxfp8
refit_prequantize=${REFIT_PREQUANTIZE}
max_steps=${MAX_STEPS}
num_nodes=${NUM_NODES}
gpus_per_node=${GPUS_PER_NODE}
EOF

JOB_CACHE_ROOT="/raid/scratch/${USER}/mxfp8-training-matrix/${RUN_NAME}"
# shellcheck disable=SC2089
COMMAND="set -euo pipefail
cd ${REPO}
export NRL_FORCE_REBUILD_VENVS=true
export NRL_VLLM_USE_V1=1
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800
export FLA_TILELANG=0
export FLA_DISABLE_BACKEND_DISPATCH=0
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export XDG_CACHE_HOME=${JOB_CACHE_ROOT}/xdg
export UV_CACHE_DIR=${JOB_CACHE_ROOT}/uv
export PYTHONPYCACHEPREFIX=${JOB_CACHE_ROOT}/pycache
export TORCHINDUCTOR_CACHE_DIR=${JOB_CACHE_ROOT}/inductor
export TRITON_CACHE_DIR=${JOB_CACHE_ROOT}/triton
export VLLM_CACHE_ROOT=${JOB_CACHE_ROOT}/vllm
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
printf 'NEMO_RL_SOURCE_COMMIT=%s\\n' \"\$(git rev-parse HEAD)\"
uv run --active --frozen examples/run_grpo.py \\
  --config ${CONFIG} \\
  policy.megatron_cfg.fp8_cfg.fp8_param=${FP8_PARAM} \\
  policy.megatron_cfg.te_precision_config_file=${TE_CONFIG} \\
  policy.generation.vllm_cfg.refit_prequantize=${REFIT_PREQUANTIZE} \\
  grpo.max_num_steps=${MAX_STEPS} \\
  grpo.val_at_start=false \\
  ++grpo.val_at_end=false \\
  checkpointing.enabled=false \\
  logger.log_dir=${BASE_LOG_DIR}/app \\
  logger.wandb_enabled=true \\
  ++logger.wandb.entity=nvidia \\
  logger.wandb.project=nemo-rl-mxfp8-training-matrix \\
  logger.wandb.name=${RUN_NAME}"

SETUP_COMMAND="mkdir -p ${JOB_CACHE_ROOT}/{xdg,uv,pycache,inductor,triton,vllm}"
MOUNTS="/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch"

# shellcheck disable=SC2090
export BASE_LOG_DIR COMMAND CONTAINER GPUS_PER_NODE MOUNTS SETUP_COMMAND WANDB_API_KEY
export CONTAINER_REMAP_ROOT=1

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment="${SEGMENT_SIZE}"
  --job-name="${ACCOUNT}-mxfp8.${RUN_NAME}"
  --output="${BASE_LOG_DIR}/slurm-%j.out"
)

printf 'action=%s cluster=%s model=%s mode=%s fp8_param=%s sha=%s\n' \
  "${ACTION}" "${CLUSTER}" "${MODEL}" "${MODE}" "${FP8_PARAM}" "${LOCAL_HEAD}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
