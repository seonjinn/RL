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
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
RUN_ID=${RUN_ID:-shmoo-moe-audit}
RUN_ROOT=${RUN_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/shmoo/${RUN_ID}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
SELECTED_PROFILES=${SELECTED_PROFILES:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/selected_profiles.json}
MOE_WEIGHTS=${MOE_WEIGHTS:-${MODEL_SNAPSHOT}/model.safetensors.index.json}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/shmoo}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
NSYS_CAPTURE_TACTICS=${NSYS_CAPTURE_TACTICS:-stock,winners}
if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)

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
export MXFP8_MOE_NSYS_CAPTURE_TACTICS=${NSYS_CAPTURE_TACTICS}
mkdir -p ${RUN_ROOT} ${CACHE_ROOT}
printf 'cuda_graph_replay=required\\n'
printf 'crash_rows_preserved=true\\n'
printf 'nsys_capture_tactics=%s\\n' "\${MXFP8_MOE_NSYS_CAPTURE_TACTICS}"
nsys profile --trace=cuda,nvtx --force-overwrite=true --output ${RUN_ROOT}/nsys-selected \\
  python experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \\
  --profiles ${SELECTED_PROFILES} \\
  --weights ${MOE_WEIGHTS} \\
  --warmups 3 \\
  --repetitions 10 \\
  --output ${RUN_ROOT}/measurements.jsonl
EOF
)

SBATCH_ARGS=(
    --nodes=1
    --ntasks=1
    --gpus=1
    --gpus-per-task=1
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time=05:00:00
    --job-name="mx-moe-shmoo-${RUN_ID}"
    --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'action=%s\n' "${ACTION}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'CUDA Graph replay required\n'
printf 'NSys captures selected winners plus stock\n'
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run) ;;
    test-only)
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=1 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
    submit)
        audit_write_manifest "${RUN_ROOT}" shmoo "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" \
            "${EXPECTED_VLLM_COMMIT}" "${CONTAINER}" \
            examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml \
            "${MODEL_SNAPSHOT}" "${CACHE_ROOT}" "${SCRIPT_DIR}"
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=1 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
esac
