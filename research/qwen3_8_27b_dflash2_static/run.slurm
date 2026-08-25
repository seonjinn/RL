#!/usr/bin/env bash
#SBATCH --job-name=dflash2-static-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

: "${ARM:?set ARM to baseline or dflash2}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE to an immutable DFlash2-capable .sqsh}"
: "${REPO_ROOT:?set REPO_ROOT to the /home checkout path}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a /lustre result directory}"

if [[ "${ARM}" != "baseline" && "${ARM}" != "dflash2" ]]; then
    echo "ARM must be baseline or dflash2" >&2
    exit 2
fi
if [[ "${REPO_ROOT}" != /home/* ]]; then
    echo "REPO_ROOT must be under /home" >&2
    exit 2
fi
if [[ "${OUTPUT_ROOT}" != /lustre/* ]]; then
    echo "OUTPUT_ROOT must be under /lustre" >&2
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
NODE_SCRATCH="/raid/scratch/${USER}/dflash2-static-${JOB_ID}"
RESULT_DIR="/results/${ARM}/${JOB_ID}"
MANIFEST="/workspace/nemo-rl/research/qwen3_8_27b_dflash2_static/${ARM}.yaml"
LAUNCHER="/workspace/nemo-rl/research/qwen3_8_27b_dflash2_static/launch.py"

container_command=$(printf '%s ' \
    "export VLLM_USE_V2_MODEL_RUNNER=1;" \
    "export NRL_CONTAINER_IMAGE='${CONTAINER_IMAGE}';" \
    "export NRL_VLLM_SOURCE_COMMIT='${SOURCE_COMMIT}';" \
    "export NRL_VLLM_SOURCE_INDEX_DIGEST='${SOURCE_INDEX_DIGEST}';" \
    "export NRL_VLLM_SOURCE_ARM64_DIGEST='${SOURCE_ARM64_DIGEST}';" \
    "export NRL_DFLASH2_MERGE_ANCESTOR='${DFLASH2_MERGE_ANCESTOR}';" \
    "export HF_HOME=/scratch/huggingface;" \
    "export TRITON_CACHE_DIR=/scratch/triton;" \
    "export VLLM_CACHE_ROOT=/scratch/vllm;" \
    "mkdir -p /scratch/huggingface /scratch/triton /scratch/vllm '${RESULT_DIR}';" \
    "python3 '${LAUNCHER}' --manifest '${MANIFEST}' --output-dir '${RESULT_DIR}/run' --request-count 20")

command=(
    srun
    --ntasks=1
    --gpus-per-task=1
    "--container-image=${CONTAINER_IMAGE}"
    "--container-mounts=${REPO_ROOT}:/workspace/nemo-rl,${OUTPUT_ROOT}:/results,${NODE_SCRATCH}:/scratch"
    bash -lc "${container_command}"
)

if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

: "${SLURM_JOB_ID:?run.slurm must execute inside an sbatch allocation}"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "DFlash2 image contract requires an aarch64 host" >&2
    exit 1
fi
if [[ -L "${CONTAINER_IMAGE}" ]]; then
    echo "CONTAINER_IMAGE must be a concrete immutable file, not a symlink" >&2
    exit 1
fi
if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "CONTAINER_IMAGE does not exist: ${CONTAINER_IMAGE}" >&2
    exit 1
fi
python3 "${IMAGE_CONTRACT_HELPER}" validate-metadata \
    "${CONTAINER_IMAGE}.metadata.txt" >/dev/null
mkdir -p "${NODE_SCRATCH}" "${OUTPUT_ROOT}/${ARM}/${SLURM_JOB_ID}"
"${command[@]}"
