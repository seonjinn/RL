#!/usr/bin/env bash
set -euo pipefail

ARM=${ARM:?Set ARM to triton, trtllm, or trtllm-main}
PAIR_ID=${PAIR_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
STEPS=${STEPS:-20}
TEST_ONLY=${TEST_ONLY:-0}

ROOT=${ROOT:-/project/coreai_dlalgo_llm/sna/deepseek-pr3669-20261013}
MAIN_SOURCE=${MAIN_SOURCE:-${ROOT}/main}
PR_SOURCE=${PR_SOURCE:-${ROOT}/pr}
IMAGE=${IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/mxfp8_backends_20261007/nemo_rl_nightly_vllm029_20261007_2977681.sqsh}
MODEL=${MODEL:-/lustre/fsw/coreai_dlalgo_llm/nemo-rl/custom_hf_checkpoints/DeepSeek-V3-BF16}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/deepseek-pr3669-20261013}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-36x2-a01r}
WALLTIME=${WALLTIME:-04:00:00}
CONFIG=examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n4g.yaml

case "${ARM}" in
  triton) SOURCE=${PR_SOURCE}; MOE_BACKEND=triton ;;
  trtllm) SOURCE=${PR_SOURCE}; MOE_BACKEND=flashinfer_trtllm ;;
  trtllm-main) SOURCE=${MAIN_SOURCE}; MOE_BACKEND=flashinfer_trtllm ;;
  *) echo "Unknown ARM: ${ARM}" >&2; exit 2 ;;
esac

[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be positive" >&2; exit 2; }
[[ "${PAIR_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "PAIR_ID contains unsupported characters" >&2; exit 2; }
test -f "${SOURCE}/${CONFIG}"
test -f "${SOURCE}/ray.sub"
test -f "${IMAGE}"
test -f "${MODEL}/config.json"
test -d "${HF_HOME}/datasets/nvidia___open_math_instruct-2"
test -n "${WANDB_API_KEY:-}"
test -z "$(git -C "${SOURCE}" status --porcelain)"

RUN_NAME="deepseek-v3-${ARM}-${STEPS}step-${PAIR_ID}"
RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

TRAIN_CMD="cd /opt/nemo-rl && \
if [[ -f /opt/nemo-rl/3rdparty/vllm/nemo-rl.env ]]; then source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env; fi && \
NRL_IGNORE_VERSION_MISMATCH=1 \
NRL_VLLM_USE_V1=1 \
HF_HOME=${HF_HOME} \
HF_DATASETS_CACHE=${HF_HOME}/datasets \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
uv run --no-sync examples/run_grpo.py \
--config ${CONFIG} \
grpo.max_num_steps=${STEPS} \
checkpointing.enabled=false \
logger.log_dir=${RUN_DIR}/app-\${SLURM_JOB_ID} \
logger.wandb_enabled=true \
logger.wandb.project=nemo-rl-deepseek-pr3669-vllm029 \
logger.wandb.name=${RUN_NAME} \
policy.model_name=${MODEL} \
cluster.num_nodes=48 \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=16 \
policy.generation.colocated.resources.gpus_per_node=4 \
policy.generation.vllm_cfg.tensor_parallel_size=16 \
policy.generation.vllm_cfg.pipeline_parallel_size=1 \
policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
policy.generation.vllm_kwargs.moe_backend=${MOE_BACKEND}"

printf 'source=%s\ncommit=%s\nimage=%s\nmodel=%s\nmodel_config_sha256=%s\nconfig=%s\narm=%s\nsteps=%s\nnodes=48\ntraining_nodes=32\ngeneration_nodes=16\nbackend=%s\ncommand=%s\n' \
  "${SOURCE}" "$(git -C "${SOURCE}" rev-parse HEAD)" "${IMAGE}" "${MODEL}" \
  "$(sha256sum "${MODEL}/config.json" | cut -d ' ' -f 1)" "${CONFIG}" \
  "${ARM}" "${STEPS}" "${MOE_BACKEND}" "${TRAIN_CMD}" > "${RUN_DIR}/manifest.txt"

export CONTAINER=${IMAGE}
export GPUS_PER_NODE=4
export MOUNTS="${SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl,${SOURCE}/examples:/opt/nemo-rl/examples,/lustre:/lustre"
export BASE_LOG_DIR=${RUN_DIR}
export COMMAND=${TRAIN_CMD}

cd "${RUN_DIR}"
SBATCH_ARGS=(
  --nodes=48
  --account="${ACCOUNT}"
  --job-name="${ACCOUNT}-pr3669.${RUN_NAME}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --exclusive
  --mem=0
  --segment=16
  --output="${RUN_DIR}/slurm-%j.out"
)
sbatch --test-only "${SBATCH_ARGS[@]}" "${SOURCE}/ray.sub"
if [[ "${TEST_ONLY}" == 1 ]]; then
  exit 0
fi
sbatch --parsable "${SBATCH_ARGS[@]}" "${SOURCE}/ray.sub"
