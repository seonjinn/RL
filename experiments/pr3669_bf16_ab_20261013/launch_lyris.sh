#!/usr/bin/env bash
set -euo pipefail

# Compare the exact PR base and head with identical BF16 TRTLLM refit settings.
MODEL=${MODEL:?Set MODEL to qwen30, qwen235, super3, or qwen35}
ARM=${ARM:?Set ARM to base or pr}
STEPS=${STEPS:-2}
PAIR_ID=${PAIR_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
TEST_ONLY=${TEST_ONLY:-0}

BASE_SHA=432050b6106b00eab37a8adc149e30ce0380d1c5
PR_SHA=cb9e2ae1d26523ac1131b16c5f73009c3db52ee3
BASE_SOURCE=${BASE_SOURCE:-/home/sna/rl-pr3669-base-20261013}
PR_SOURCE=${PR_SOURCE:-/home/sna/rl-pr3669-20261013}
IMAGE=${IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_pr3294_20261007_3270601.sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3669-bf16-ab-20261013}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
WALLTIME=${WALLTIME:-04:00:00}

case "${MODEL}" in
  qwen30)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml
    NODES=4; SEGMENT=2; VLLM_EP=1 ;;
  qwen235)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml
    NODES=32; SEGMENT=16; VLLM_EP=8 ;;
  super3)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-async-1off.yaml
    NODES=32; SEGMENT=8; VLLM_EP=4 ;;
  qwen35)
    CONFIG=examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-6n4g-async-1off-bf16-trtllm.yaml
    NODES=6; SEGMENT=2; VLLM_EP=4 ;;
  *) echo "Unknown MODEL: ${MODEL}" >&2; exit 2 ;;
esac

case "${ARM}" in
  base) SOURCE=${BASE_SOURCE}; EXPECTED_SHA=${BASE_SHA} ;;
  pr) SOURCE=${PR_SOURCE}; EXPECTED_SHA=${PR_SHA} ;;
  *) echo "Unknown ARM: ${ARM}" >&2; exit 2 ;;
esac

[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be positive" >&2; exit 2; }
[[ "${PAIR_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "PAIR_ID contains unsupported characters" >&2; exit 2; }
test -f "${SOURCE}/${CONFIG}"
test -f "${SOURCE}/ray.sub"
test -f "${IMAGE}"
test -d "${HF_HOME}"
test -n "${WANDB_API_KEY:-}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_SHA}"
test -z "$(git -C "${SOURCE}" status --porcelain)"

RUN_NAME="${MODEL}-${ARM}-${STEPS}step-${PAIR_ID}"
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
logger.wandb.project=nemo-rl-pr3669-bf16-ab \
logger.wandb.name=${RUN_NAME} \
policy.generation.refit_transport=nccl_reshard \
policy.generation.vllm_cfg.precision=bfloat16 \
policy.generation.vllm_cfg.expert_parallel_size=${VLLM_EP} \
policy.generation.vllm_cfg.refit_with_reload_api=false \
++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
++policy.generation.vllm_kwargs.expert_placement_strategy=linear"

printf 'source=%s\ncommit=%s\nimage=%s\nconfig=%s\nmodel=%s\narm=%s\nsteps=%s\nnodes=%s\nsegment=%s\nvllm_ep=%s\ncommand=%s\n' \
  "${SOURCE}" "${EXPECTED_SHA}" "${IMAGE}" "${CONFIG}" "${MODEL}" "${ARM}" \
  "${STEPS}" "${NODES}" "${SEGMENT}" "${VLLM_EP}" "${TRAIN_CMD}" \
  > "${RUN_DIR}/manifest.txt"

export CONTAINER=${IMAGE}
export GPUS_PER_NODE=4
export MOUNTS="${SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl,${SOURCE}/examples:/opt/nemo-rl/examples,/lustre:/lustre"
export BASE_LOG_DIR=${RUN_DIR}
export COMMAND=${TRAIN_CMD}

cd "${RUN_DIR}"
SBATCH_ARGS=(
  --nodes="${NODES}"
  --account="${ACCOUNT}"
  --job-name="${ACCOUNT}-pr3669.${RUN_NAME}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --exclusive
  --mem=0
  --segment="${SEGMENT}"
  --output="${RUN_DIR}/slurm-%j.out"
)
sbatch --test-only "${SBATCH_ARGS[@]}" "${SOURCE}/ray.sub"
if [[ "${TEST_ONLY}" == 1 ]]; then
  exit 0
fi
sbatch --parsable "${SBATCH_ARGS[@]}" "${SOURCE}/ray.sub"
