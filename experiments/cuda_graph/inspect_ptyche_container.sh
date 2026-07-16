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
HF_HOME=${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf}
export HF_HOME

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
        python - <<"PY"
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="meta-llama/Llama-3.1-8B-Instruct", filename="config.json")
print("LLAMA_HF_ACCESS_OK")
PY
        echo "== CUDA =="
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    '
