#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}
source "${SCRIPT_DIR}/provenance.sh"

ACTION=${ACTION:-dry-run}
ARM=${ARM:-candidate}
MAX_STEPS=${MAX_STEPS:-2}
case "${ACTION}" in
    test-only|dry-run|submit) ;;
    *) echo "Unsupported ACTION: ${ACTION}" >&2; exit 2 ;;
esac
case "${ARM}" in
    stock|candidate) ;;
    *) echo "ARM must be stock or candidate" >&2; exit 2 ;;
esac
case "${MAX_STEPS}" in
    2|8) ;;
    *) echo "MAX_STEPS must be 2 or 8" >&2; exit 2 ;;
esac

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-a76062edee3a3ac23d47a93c7ce466f06a19111f}
MODEL=Qwen/Qwen3-30B-A3B
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
RUN_ID=${RUN_ID:-validation-moe-audit}
RUN_ROOT=${RUN_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/validation/${ARM}/${RUN_ID}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
STOCK_CACHE_ROOT=${STOCK_CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/cache/stock}
CANDIDATE_CACHE_ROOT=${CANDIDATE_CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/cache/candidate}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
GSM8K_EVALUATOR=${GSM8K_EVALUATOR:-${WORK_ROOT}/vllm-benchmark/experiments/eval/gsm8k_vllm_eval.py}
GSM8K_DATASET=${GSM8K_DATASET:-${WORK_ROOT}/vllm-benchmark/experiments/eval/data/gsm8k_test_openai_1319.jsonl}
VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:8000}
case "${ARM}" in
    stock) CACHE_ROOT=${STOCK_CACHE_ROOT} ;;
    candidate) CACHE_ROOT=${CANDIDATE_CACHE_ROOT} ;;
esac
if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)

POST_SMOKE_COMMAND=''
if [[ "${MAX_STEPS}" == 8 ]]; then
    POST_SMOKE_COMMAND=$(cat <<EOF
python experiments/mxfp8_moe_tactic_audit/validate_correctness.py generation \\
  --stock ${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/validation/stock/${RUN_ID}/generation.jsonl \\
  --candidate ${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/validation/candidate/${RUN_ID}/generation.jsonl
python ${GSM8K_EVALUATOR} \\
  --endpoint ${VLLM_ENDPOINT} \\
  --model ${MODEL} \\
  --dataset ${GSM8K_DATASET} \\
  --limit 1319 \\
  --seed 20260807 \\
  --concurrency 1 \\
  --output-dir ${RUN_ROOT}/gsm8k \\
  --provenance-json ${RUN_ROOT}/run_manifest.json
EOF
)
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
source ${CUSTOM_VLLM_ROOT}/nemo-rl.env
runtime_nemo_rl_commit=\$(git rev-parse HEAD)
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_nemo_rl_commit}" == "${NEMO_RL_COMMIT}" ]]
[[ "\${runtime_vllm_commit}" == "${EXPECTED_VLLM_COMMIT}" ]]
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${CACHE_ROOT}
export MXFP8_MOE_CUDA_GRAPH_REPLAY=required
mkdir -p ${RUN_ROOT} ${CACHE_ROOT}
python examples/run_grpo.py \\
  --config ${CONFIG} \\
  cluster.num_nodes=4 \\
  cluster.gpus_per_node=4 \\
  cluster.segment_size=4 \\
  policy.generation.vllm_cfg.enforce_eager=false \\
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \\
  grpo.max_num_steps=${MAX_STEPS} \\
  grpo.val_at_start=false \\
  checkpointing.enabled=false \\
  checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints \\
  logger.log_dir=${RUN_ROOT}/logs \\
  logger.wandb_enabled=false
if [[ ${MAX_STEPS} -eq 2 ]]; then
  touch ${RUN_ROOT}/smoke_complete
fi
if [[ ${MAX_STEPS} -eq 8 ]]; then
  [[ -f ${RUN_ROOT}/smoke_complete ]] || {
    echo "Two-step ${ARM} smoke is required before deterministic validation" >&2
    exit 1
  }
${POST_SMOKE_COMMAND}
fi
EOF
)

SBATCH_ARGS=(
    --nodes=4
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time=05:00:00
    --job-name="mx-moe-${ARM}-${MAX_STEPS}s-${RUN_ID}"
    --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'action=%s\n' "${ACTION}"
printf 'arm=%s\n' "${ARM}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'cache_root=%s\n' "${CACHE_ROOT}"
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run) ;;
    test-only)
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
    submit)
        audit_write_manifest "${RUN_ROOT}" "validation-${ARM}" "${REPO_DIR}" \
            "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}" "${CONTAINER}" "${CONFIG}" \
            "${MODEL_SNAPSHOT}" "${CACHE_ROOT}" "${SCRIPT_DIR}"
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
esac
