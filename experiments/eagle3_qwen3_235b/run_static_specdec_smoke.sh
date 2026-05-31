#!/usr/bin/env bash
set -euo pipefail

# Submit a short NeMo-RL SWE job with static Eagle3 speculative decoding enabled.
# This is meant to test vLLM/NeMo-RL integration and acceptance/perf signals
# before training a Thinking-2507-specific draft.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EAGLE3_DRAFT_MODEL="${EAGLE3_DRAFT_MODEL:-nvidia/Qwen3-235B-A22B-Eagle3}"
EAGLE3_NUM_SPEC_TOKENS="${EAGLE3_NUM_SPEC_TOKENS:-3}"
EAGLE3_DRAFT_TP="${EAGLE3_DRAFT_TP:-1}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
WANDB_NAME="${WANDB_NAME:-qwen3-235b-swe-eagle3-smoke}"
EXP_SUFFIX_OVERRIDE="${EXP_SUFFIX_OVERRIDE:-$WANDB_NAME}"
DRY_RUN="${DRY_RUN:-false}"

export MAX_NUM_STEPS
export WANDB_NAME
export EXP_SUFFIX_OVERRIDE
export SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
export EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-} \
++policy.generation.vllm_kwargs.speculative_config.method=eagle3 \
++policy.generation.vllm_kwargs.speculative_config.model=${EAGLE3_DRAFT_MODEL} \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${EAGLE3_NUM_SPEC_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${EAGLE3_DRAFT_TP}"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "MAX_NUM_STEPS=$MAX_NUM_STEPS"
  echo "WANDB_NAME=$WANDB_NAME"
  echo "EXP_SUFFIX_OVERRIDE=$EXP_SUFFIX_OVERRIDE"
  echo "SBATCH_DEPENDENCY=$SBATCH_DEPENDENCY"
  echo "EXTRA_HYDRA_OVERRIDES=$EXTRA_HYDRA_OVERRIDES"
  printf '%q\n' "$ROOT_DIR/run_grpo_qwen3_235b_swe.sh"
  exit 0
fi

exec "$ROOT_DIR/run_grpo_qwen3_235b_swe.sh"
