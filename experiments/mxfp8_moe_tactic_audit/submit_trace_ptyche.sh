#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}
source "${SCRIPT_DIR}/provenance.sh"

ACTION=${ACTION:-dry-run}
case "${ACTION}" in
    test-only|dry-run|submit) ;;
    *) echo "Unsupported ACTION: ${ACTION}" >&2; exit 2 ;;
esac

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-a76062edee3a3ac23d47a93c7ce466f06a19111f}
MODEL=Qwen/Qwen3-30B-A3B
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
RUN_ID=${RUN_ID:-trace-moe-audit}
RUN_ROOT=${RUN_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/trace/${RUN_ID}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
MODEL_REVISION_FILE=${MODEL_REVISION_FILE:-${MODEL_SNAPSHOT}/refs/main}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/trace}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
WANDB_ENABLED=${WANDB_ENABLED:-false}
case "${WANDB_ENABLED}" in
    true|false) ;;
    *) echo "WANDB_ENABLED must be true or false" >&2; exit 2 ;;
esac

if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi

TRACE_DIR=${RUN_ROOT}/trace
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)
if [[ "${ACTION}" == submit ]]; then
    [[ -s "${MODEL_REVISION_FILE}" ]] || {
        echo "Missing local model revision: ${MODEL_REVISION_FILE}" >&2
        exit 1
    }
    MODEL_REVISION=$(tr -d '[:space:]' < "${MODEL_REVISION_FILE}")
else
    MODEL_REVISION=${MODEL_REVISION:-dry-run-not-validated}
fi
COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
source ${CUSTOM_VLLM_ROOT}/nemo-rl.env
runtime_nemo_rl_commit=\$(git rev-parse HEAD)
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_nemo_rl_commit}" == "${NEMO_RL_COMMIT}" ]]
[[ "\${runtime_vllm_commit}" == "${EXPECTED_VLLM_COMMIT}" ]]
unset VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
export VLLM_MXFP8_MOE_TRACE_DIR=${TRACE_DIR}
export VLLM_MXFP8_MOE_MODEL_REVISION=${MODEL_REVISION}
export VLLM_MXFP8_MOE_RUNTIME_FINGERPRINT=${NEMO_RL_COMMIT}-${EXPECTED_VLLM_COMMIT}
export VLLM_MXFP8_MOE_DP_SIZE=16
mkdir -p ${TRACE_DIR}
python examples/run_grpo.py \\
  --config ${CONFIG} \\
  cluster.num_nodes=4 \\
  cluster.gpus_per_node=4 \\
  cluster.segment_size=4 \\
  policy.generation.vllm_cfg.enforce_eager=true \\
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \\
  grpo.max_num_steps=2 \\
  grpo.val_at_start=false \\
  checkpointing.enabled=false \\
  checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints \\
  logger.log_dir=${RUN_ROOT}/logs \\
  logger.wandb_enabled=${WANDB_ENABLED}
touch ${RUN_ROOT}/trace_complete
EOF
)

SBATCH_ARGS=(
    --nodes=4
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time="${WALLTIME}"
    --job-name="mx-moe-trace-${RUN_ID}"
    --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'action=%s\n' "${ACTION}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'trace_is_metadata_only=true\n'
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run) ;;
    test-only)
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
    submit)
        audit_write_manifest "${RUN_ROOT}" trace "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" \
            "${EXPECTED_VLLM_COMMIT}" "${CONTAINER}" "${CONFIG}" "${MODEL_SNAPSHOT}" \
            "${CACHE_ROOT}" "${SCRIPT_DIR}"
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
esac
