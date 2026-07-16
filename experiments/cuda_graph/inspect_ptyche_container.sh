#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --job-name=coreai_dlalgo_llm-cg.runtime-inspect
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-vs-pr5783-ptyche-20260716/experiments/cuda_graph/logs/runtime-inspect-%j.out

set -euo pipefail

WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-vs-pr5783-ptyche-20260716
CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh

mkdir -p "${WORKTREE}/experiments/cuda_graph/logs"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKTREE}" \
    bash -lc '
        set -euo pipefail
        echo "PATH=${PATH}"
        command -v python || true
        for candidate in python /opt/nemo_rl_venv/bin/python /opt/venv/bin/python; do
            if command -v "${candidate}" >/dev/null 2>&1 || [ -x "${candidate}" ]; then
                echo "== ${candidate} =="
                "${candidate}" --version
                "${candidate}" - <<"PY"
import importlib.util

for module in ("ray", "torch", "transformer_engine", "pytest", "nemo_rl"):
    print(f"{module}={importlib.util.find_spec(module) is not None}")
PY
            fi
        done
        echo "== Hugging Face Qwen cache candidates =="
        for cache_dir in /root/.cache/huggingface/hub /opt/.cache/huggingface/hub; do
            if [ -d "${cache_dir}" ]; then
                find "${cache_dir}" -maxdepth 1 -type d -iname "*qwen*" -printf "%f\\n"
            fi
        done
        echo "== CUDA =="
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    '
