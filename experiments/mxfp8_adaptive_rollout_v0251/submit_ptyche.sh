#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-submit}
if [[ "$ACTION" != submit && "$ACTION" != test-only ]]; then
  echo "usage: submit_ptyche.sh [submit|test-only]" >&2
  exit 2
fi

export NEMO_RL_REPO_ROOT=${NEMO_RL_REPO_ROOT:-/home/sna/nemorl-v0251-mxfp8-safe-adaptive-canary}
export CUSTOM_VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:-/home/sna/mxfp8-safe-backend/vllm-v0251-safe-backend}
export EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-658d7b1571a914bee7df48f717c2a428ee7c45ad}
export MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
export TACTIC_FILE=${TACTIC_FILE:-/home/sna/mxfp8-safe-backend/vllm-benchmark-v0251-safe/experiments/sweep/data/microbench/mxfp8_v0251_safe_backend_artifacts_20260801_r6_robust/exact_tactics.json}
export TACTIC_SHA256=${TACTIC_SHA256:-d5681371ea2476c3732d58089148e13123165b9e740d3e32ddec98d6eca40a1d}
export LAYER_ALLOWLIST_B64=${LAYER_ALLOWLIST_B64:-MTI4MCw4MTkyCjIwNDgsODE5Mgo0Mzg0LDgxOTIKODE5MiwxMDI0CjgxOTIsMTI4MAo4MTkyLDIwNDgK}
export CANARY_RESULT_ROOT=${CANARY_RESULT_ROOT:-/home/sna/results/nemorl-v0251-mxfp8-safe-adaptive/$(date +%Y%m%d_%H%M%S)}
export CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
export MOUNTS=${MOUNTS:-/lustre:/lustre,/home/sna:/home/sna}
export GPUS_PER_NODE=4
export BASE_LOG_DIR="$CANARY_RESULT_ROOT/slurm"
export COMMAND="bash $NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_ab.sh"
export UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-/home/sna/.cache/uv-canary}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/home/sna/.cache/nemorl-venvs-v0251-canary}
mkdir -p "$BASE_LOG_DIR" "$UV_CACHE_DIR_OVERRIDE" "$NEMO_RL_VENV_DIR"

sha256sum --check <(printf '%s  %s\n' "$TACTIC_SHA256" "$TACTIC_FILE")
git -C "$NEMO_RL_REPO_ROOT" diff --quiet
git -C "$NEMO_RL_REPO_ROOT" diff --cached --quiet
git -C "$NEMO_RL_REPO_ROOT" pull --ff-only
test "$(git -C "$CUSTOM_VLLM_SOURCE" rev-parse HEAD)" = "$EXPECTED_VLLM_COMMIT"
git -C "$CUSTOM_VLLM_SOURCE" diff --quiet
git -C "$CUSTOM_VLLM_SOURCE" diff --cached --quiet

args=(
  --account=coreai_dlalgo_llm
  --partition=36x2-a01r
  --nodes=2
  --time=05:00:00
  --segment=2
  --job-name=coreai_dlalgo_llm-nemorl.mxfp8-adaptive-canary
  --dependency=
  --export=ALL
)
if [[ "$ACTION" == test-only ]]; then
  args+=(--test-only)
fi

cd "$NEMO_RL_REPO_ROOT"
sbatch "${args[@]}" ray.sub
