#!/usr/bin/env bash
#SBATCH --job-name=coreai_dlalgo_llm-specdec.q235-eagle3-stage
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-rp25-perf-20260917/eagle3-stage/slurm-%j.out
#SBATCH --error=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-rp25-perf-20260917/eagle3-stage/slurm-%j.err

set -euo pipefail

readonly CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-20260916/nemo_rl_nightly_20260916_2837270.sqsh
readonly HF_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
readonly REVISION=33f3c01ce807376d1171301b9a148b1b28f239ba
readonly SNAPSHOT="${HF_ROOT}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/${REVISION}"

test -f "${CONTAINER}"
mkdir -p "${HF_ROOT}/hub" /raid/scratch/sna/q235-eagle3-stage

srun --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,/raid:/raid \
  /opt/nemo_rl_venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="nvidia/Qwen3-235B-A22B-Eagle3",
    revision="33f3c01ce807376d1171301b9a148b1b28f239ba",
    cache_dir="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub",
    allow_patterns=("config.json", "model.safetensors"),
)
print(path)
PY

test -f "${SNAPSHOT}/config.json"
test -f "${SNAPSHOT}/model.safetensors"
sha256sum "${SNAPSHOT}/config.json" "${SNAPSHOT}/model.safetensors"
