#!/usr/bin/env bash
#SBATCH --job-name=dflash2-grpo-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE to an immutable NeMo-RL + f946 vLLM image}"
: "${REPO_ROOT:?set REPO_ROOT to the /home checkout path}"
: "${TARGET_SNAPSHOT:?set TARGET_SNAPSHOT to the pinned /lustre target snapshot}"
: "${DRAFT_SNAPSHOT:?set DRAFT_SNAPSHOT to the pinned /lustre draft snapshot}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a /lustre result directory}"
: "${SMOKE_STEPS:?set SMOKE_STEPS to 1 or 20}"

if [[ "${REPO_ROOT}" != /home/* ]]; then
    echo "REPO_ROOT must be under /home" >&2
    exit 2
fi
for shared_path in "${TARGET_SNAPSHOT}" "${DRAFT_SNAPSHOT}" "${OUTPUT_ROOT}"; do
    if [[ "${shared_path}" != /lustre/* ]]; then
        echo "model snapshots and OUTPUT_ROOT must be under /lustre" >&2
        exit 2
    fi
done
if [[ "${SMOKE_STEPS}" != 1 && "${SMOKE_STEPS}" != 20 ]]; then
    echo "SMOKE_STEPS must be exactly 1 or 20" >&2
    exit 2
fi

JOB_ID="${SLURM_JOB_ID:-dry-run}"
NODE_SCRATCH="/raid/scratch/${USER}/dflash2-grpo-${JOB_ID}"
RESULT_DIR="${OUTPUT_ROOT}/steps-${SMOKE_STEPS}/${JOB_ID}"
RECIPE="${REPO_ROOT}/research/qwen3_8_27b_dflash2_static/grpo.yaml"
CONTRACT="${REPO_ROOT}/research/qwen3_8_27b_dflash2_static/training_contract.py"
PREFLIGHT="${REPO_ROOT}/research/qwen3_8_27b_dflash2_static/preflight.py"
if [[ "${DRY_RUN}" == true ]]; then
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    IMAGE_CONTRACT_HELPER="${SCRIPT_DIR}/image_contract.py"
else
    IMAGE_CONTRACT_HELPER="${REPO_ROOT}/research/qwen3_8_27b_dflash2_static/image_contract.py"
fi
SOURCE_COMMIT=$(python3 "${IMAGE_CONTRACT_HELPER}" get source_commit)
SOURCE_INDEX_DIGEST=$(python3 "${IMAGE_CONTRACT_HELPER}" get source_index_digest)
SOURCE_ARM64_DIGEST=$(python3 "${IMAGE_CONTRACT_HELPER}" get source_arm64_digest)
DFLASH2_MERGE_ANCESTOR=$(
    python3 "${IMAGE_CONTRACT_HELPER}" get dflash2_merge_ancestor
)

container_command=$(printf '%s ' \
    "set -euo pipefail;" \
    "export PYTHONPATH='${REPO_ROOT}';" \
    "export NRL_TRAINING_ENTRYPOINT='${REPO_ROOT}/examples/run_grpo.py';" \
    "export NRL_TRAINING_STEPS_OVERRIDE='grpo.max_num_steps=${SMOKE_STEPS}';" \
    "export NEMO_RL_PY_EXECUTABLES_SYSTEM=1;" \
    "export VLLM_USE_V2_MODEL_RUNNER=1;" \
    "export NRL_VLLM_SOURCE_COMMIT='${SOURCE_COMMIT}';" \
    "export NRL_VLLM_SOURCE_INDEX_DIGEST='${SOURCE_INDEX_DIGEST}';" \
    "export NRL_VLLM_SOURCE_ARM64_DIGEST='${SOURCE_ARM64_DIGEST}';" \
    "export NRL_DFLASH2_MERGE_ANCESTOR='${DFLASH2_MERGE_ANCESTOR}';" \
    "export HF_HOME='${NODE_SCRATCH}/huggingface';" \
    "export UV_CACHE_DIR='${NODE_SCRATCH}/uv-cache';" \
    "export UV_PROJECT_ENVIRONMENT='${NODE_SCRATCH}/uv-project';" \
    "export TRITON_CACHE_DIR='${NODE_SCRATCH}/triton';" \
    "export VLLM_CACHE_ROOT='${NODE_SCRATCH}/vllm';" \
    "mkdir -p '${NODE_SCRATCH}/huggingface' '${NODE_SCRATCH}/uv-cache' '${NODE_SCRATCH}/triton' '${NODE_SCRATCH}/vllm' '${RESULT_DIR}';" \
    "python3 '${PREFLIGHT}' --nemo-actor-json > '${RESULT_DIR}/runtime_fingerprint.json';" \
    "python3 '${CONTRACT}' --repo-root '${REPO_ROOT}' --recipe '${RECIPE}' --target-snapshot '${TARGET_SNAPSHOT}' --draft-snapshot '${DRAFT_SNAPSHOT}' --output-dir '${RESULT_DIR}/training' --steps '${SMOKE_STEPS}'")

command=(
    srun
    --ntasks=1
    --gpus-per-task=4
    "--container-image=${CONTAINER_IMAGE}"
    "--container-mounts=${REPO_ROOT}:${REPO_ROOT},${TARGET_SNAPSHOT}:${TARGET_SNAPSHOT},${DRAFT_SNAPSHOT}:${DRAFT_SNAPSHOT},${OUTPUT_ROOT}:${OUTPUT_ROOT},${NODE_SCRATCH}:${NODE_SCRATCH}"
    bash -lc "${container_command}"
)

if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

: "${SLURM_JOB_ID:?run_grpo.slurm must execute inside an sbatch allocation}"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "DFlash2 feasibility spike requires an aarch64 host" >&2
    exit 1
fi
for required_path in "${CONTAINER_IMAGE}" "${TARGET_SNAPSHOT}" "${DRAFT_SNAPSHOT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
mkdir -p "${NODE_SCRATCH}" "${OUTPUT_ROOT}/steps-${SMOKE_STEPS}"
"${command[@]}"
