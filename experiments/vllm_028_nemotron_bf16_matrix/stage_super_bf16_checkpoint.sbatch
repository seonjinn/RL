#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=stage_super_bf16_checkpoint
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/stage-super-bf16-%j.out

set -euo pipefail

readonly REPO_ID="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
readonly REVISION="d51eab0d1f979ebc26b546e634a04f450d99158e"
readonly CONTAINER_IMAGE="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm-openai-v0.28.0-aarch64-ubuntu2404.sqsh"
readonly TARGET="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
readonly PARTIAL_TARGET="${TARGET}.partial.${SLURM_JOB_ID}"
readonly NODE_STAGE="/raid/scratch/${USER}/nemotron-super-bf16-${SLURM_JOB_ID}"
readonly LOCAL_SNAPSHOT="${NODE_STAGE}/snapshot"
export HF_HOME="${NODE_STAGE}/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"

cleanup_node_stage() {
  if [[ "${NODE_STAGE}" == "/raid/scratch/${USER}/nemotron-super-bf16-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${NODE_STAGE}"
  fi
}
trap cleanup_node_stage EXIT

verify_snapshot() {
  local snapshot=$1
  [[ -s "${snapshot}/config.json" ]] || return 1
  local shard_count
  shard_count=$(find "${snapshot}" -maxdepth 1 -type f -size +0c -name 'model-*-of-00050.safetensors' | wc -l)
  [[ "${shard_count}" -eq 50 ]]
}

if verify_snapshot "${TARGET}"; then
  echo "Verified existing Super BF16 checkpoint: ${TARGET}"
  exit 0
fi
if [[ -e "${TARGET}" ]]; then
  echo "Target exists but failed the 50-shard validation: ${TARGET}" >&2
  exit 1
fi
if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
  echo "Missing staged vLLM container: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

mkdir -p "${LOCAL_SNAPSHOT}" "${HF_HUB_CACHE}" "${PARTIAL_TARGET}"
export REPO_ID REVISION LOCAL_SNAPSHOT
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch" \
  --no-container-mount-home \
  --container-remap-root \
  python3 -c \
  'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["REPO_ID"], revision=os.environ["REVISION"], local_dir=os.environ["LOCAL_SNAPSHOT"])'

if ! verify_snapshot "${LOCAL_SNAPSHOT}"; then
  echo "Downloaded Super BF16 checkpoint failed config.json/50-shard validation" >&2
  exit 1
fi
rsync -a --info=progress2 "${LOCAL_SNAPSHOT}/" "${PARTIAL_TARGET}/"
if ! verify_snapshot "${PARTIAL_TARGET}"; then
  echo "Durable Super BF16 checkpoint copy failed validation" >&2
  exit 1
fi
mv "${PARTIAL_TARGET}" "${TARGET}"
echo "Staged and verified ${TARGET}"
