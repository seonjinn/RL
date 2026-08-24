#!/usr/bin/env bash
#SBATCH --job-name=dflash2-actor-preflight
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# OCI-HSG QOS allocates a minimum four-GPU node; the srun below exposes one GPU.
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE to the candidate runtime .sqsh}"
: "${REPO_ROOT:?set REPO_ROOT to the /home checkout path}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a /lustre result directory}"
if [[ "${REPO_ROOT}" != /home/* || "${OUTPUT_ROOT}" != /lustre/* ]]; then
    echo "REPO_ROOT must be under /home and OUTPUT_ROOT under /lustre" >&2
    exit 2
fi

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

JOB_ID="${SLURM_JOB_ID:-dry-run}"
NODE_SCRATCH="/raid/scratch/${USER}/dflash2-actor-preflight-${JOB_ID}"
RESULT_DIR="${OUTPUT_ROOT}/${JOB_ID}"
PREFLIGHT="${REPO_ROOT}/research/qwen3_8_27b_dflash2_static/preflight.py"
container_command=$(printf '%s ' \
    "set -euo pipefail;" \
    "export PYTHONPATH='${REPO_ROOT}';" \
    "export NEMO_RL_PY_EXECUTABLES_SYSTEM=1;" \
    "export VLLM_USE_V2_MODEL_RUNNER=1;" \
    "export NRL_VLLM_SOURCE_COMMIT='${SOURCE_COMMIT}';" \
    "export NRL_VLLM_SOURCE_INDEX_DIGEST='${SOURCE_INDEX_DIGEST}';" \
    "export NRL_VLLM_SOURCE_ARM64_DIGEST='${SOURCE_ARM64_DIGEST}';" \
    "export NRL_DFLASH2_MERGE_ANCESTOR='${DFLASH2_MERGE_ANCESTOR}';" \
    "export HF_HOME='${NODE_SCRATCH}/huggingface';" \
    "mkdir -p '${NODE_SCRATCH}/huggingface' '${RESULT_DIR}';" \
    "python '${PREFLIGHT}' --nemo-actor-json | tee '${RESULT_DIR}/runtime_fingerprint.json'")
command=(
    srun
    --ntasks=1
    --gpus-per-task=1
    "--container-image=${CONTAINER_IMAGE}"
    "--container-mounts=${REPO_ROOT}:${REPO_ROOT},${OUTPUT_ROOT}:${OUTPUT_ROOT},${NODE_SCRATCH}:${NODE_SCRATCH}"
    bash -lc "${container_command}"
)

if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

: "${SLURM_JOB_ID:?runtime_preflight.slurm must execute inside sbatch}"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "DFlash2 runtime preflight requires an aarch64 host" >&2
    exit 1
fi
if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "CONTAINER_IMAGE does not exist: ${CONTAINER_IMAGE}" >&2
    exit 1
fi
mkdir -p "${NODE_SCRATCH}" "${RESULT_DIR}"
"${command[@]}"
