#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_dlalgo_llm-cg.pr5672-tests
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-vs-pr5783-ptyche-20260716/experiments/cuda_graph/logs/pr5672-tests-%j.out

set -euo pipefail

WORKTREE=/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-cgseqpack-pr5672-vs-pr5783-ptyche-20260716
CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh
export WORKTREE

mkdir -p "${WORKTREE}/experiments/cuda_graph/logs"

export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKTREE}" \
    bash -lc '
        set -euo pipefail
        python --version
        python -m pytest -q \
            tests/unit/test_ray_sub_submission.py
        uv run --locked --extra mcore --directory "${WORKTREE}" python -m pytest -q \
            tests/unit/models/policy/test_pr5672_cuda_graph_adapter.py
        uv run --locked --extra mcore --directory "${WORKTREE}" python -c \
            "import torch, transformer_engine; print(\"MCORE_RUNTIME_OK\", torch.__version__, transformer_engine.__version__)"
        cd "${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM"
        uv run --locked --extra mcore --directory "${WORKTREE}" python -m pytest -q \
            tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
    '
