#!/usr/bin/env bash
# Run the focused CUDA-graph scope tests inside the Ptyche NeMo-RL container.
# Submit with an explicit output path, for example:
#   WORKTREE=/lustre/... sbatch --output=/lustre/.../scope-tests-%j.out \
#     experiments/cuda_graph/run_qwen30_scope_tests_ptyche.sh

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=coreai_dlalgo_llm-cg.scope-tests

set -euo pipefail

WORKTREE=${WORKTREE:?Set WORKTREE to the remote NeMo-RL worktree.}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}
STATIC_THD_TEST=${STATIC_THD_TEST:-0}

tests=(
  "3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/rl/test_rl_utils.py::TestRLUtils::test_megatron_rl_inference_mode_restores_training_cuda_graph_state"
  "3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/rl/test_rl_utils.py::TestRLUtils::test_megatron_rl_inference_mode_preserves_requested_moe_cuda_graph_modules"
)

if [[ "${STATIC_THD_TEST}" == "1" ]]; then
  tests+=(
    "tests/unit/models/megatron/test_megatron_setup.py::test_static_thd_cuda_graph_preserves_transformer_engine_packing_mode"
  )
fi

export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${WORKTREE}" \
  uv run --locked --directory "${WORKTREE}" pytest -q "${tests[@]}"
