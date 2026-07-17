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
STATIC_THD_LOSS_TEST=${STATIC_THD_LOSS_TEST:-0}
STATIC_THD_OPTIONAL_KV_TEST=${STATIC_THD_OPTIONAL_KV_TEST:-0}

tests=(
  "tests/unit_tests/rl/test_rl_utils.py::TestRLUtils::test_megatron_rl_inference_mode_restores_training_cuda_graph_state"
  "tests/unit_tests/rl/test_rl_utils.py::TestRLUtils::test_megatron_rl_inference_mode_preserves_requested_moe_cuda_graph_modules"
)

static_thd_suffix=""
if [[ "${STATIC_THD_TEST}" == "1" ]]; then
  static_thd_suffix=" && cd '${WORKTREE}' && pytest -q \\
    --confcutdir='${WORKTREE}/tests/unit/models/megatron' \\
    '${WORKTREE}/tests/unit/models/megatron/test_megatron_setup.py::test_static_thd_cuda_graph_preserves_transformer_engine_packing_mode'"
fi

if [[ "${STATIC_THD_LOSS_TEST}" == "1" ]]; then
  static_thd_suffix+=" && cd '${WORKTREE}' && pytest -q \\
    --confcutdir='${WORKTREE}/tests/unit/models/megatron' \\
    '${WORKTREE}/tests/unit/models/megatron/test_train.py::TestForwardWithPostProcessingFn::test_forward_with_loss_post_processor_uses_real_packed_loss_metadata' \\
    '${WORKTREE}/tests/unit/models/megatron/test_train.py::TestLossPostProcessor::test_loss_post_processor_with_packing'"
fi

if [[ "${STATIC_THD_OPTIONAL_KV_TEST}" == "1" ]]; then
  static_thd_suffix+=" && cd '${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM' && pytest -q \\
    'tests/unit_tests/transformer/test_thd_cuda_graph.py::TestDecomposeReconstruct::test_round_trip_canonicalizes_optional_kv_inputs'"
fi

export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${WORKTREE}:${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${PYTHONPATH:-}"

srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${WORKTREE}" \
  uv run --locked --extra mcore --directory "${WORKTREE}" bash -lc \
  "cd '${WORKTREE}/3rdparty/Megatron-LM-workspace/Megatron-LM' && pytest -q ${tests[*]}${static_thd_suffix}"
