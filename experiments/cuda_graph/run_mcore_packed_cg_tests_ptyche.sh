#!/usr/bin/env bash
# Run the packed-THD Transformer Engine CUDA Graph unit tests on Ptyche.
# Submit with:
#   WORKTREE=/lustre/... sbatch --output=/lustre/.../mcore-packed-cg-%j.out \
#     experiments/cuda_graph/run_mcore_packed_cg_tests_ptyche.sh

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=coreai_dlalgo_llm-cg.packed-tests

set -euo pipefail

WORKTREE=${WORKTREE:?Set WORKTREE to the remote NeMo-RL worktree.}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}
TEST=${TEST:-tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py}

export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${WORKTREE}" \
  uv run --locked --extra mcore --directory "${WORKTREE}" bash -lc \
  "cd '${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM' && pytest -q '${TEST}'"
