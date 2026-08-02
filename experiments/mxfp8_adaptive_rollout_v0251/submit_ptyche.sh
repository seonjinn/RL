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

for name in \
  NEMORL_MXFP8_LINEAR_BACKEND \
  VLLM_MXFP8_DENSE_SHAPE_TRACE \
  VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR \
  VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX \
  VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK \
  VLLM_MXFP8_DENSE_TRTLLM_LAYOUT \
  VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M \
  VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE \
  VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256 \
  VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST \
  VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64 \
  VLLM_MXFP8_DENSE_TRTLLM_TACTIC \
  VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4; do
  unset "$name"
done

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
lock_sha=$(sha256sum "$NEMO_RL_REPO_ROOT/uv.lock" | awk '{print $1}')
venv_key=${lock_sha:0:16}-${EXPECTED_VLLM_COMMIT:0:12}
export NEMO_RL_DRIVER_VENV_DIR=${NEMO_RL_DRIVER_VENV_DIR:-/home/sna/.cache/nemorl-driver-v0251-canary/$venv_key}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/home/sna/.cache/nemorl-venvs-v0251-canary/$venv_key}
export CUSTOM_VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/cache/vllm-runtime-overlays/$venv_key}
mkdir -p \
  "$BASE_LOG_DIR" \
  "$UV_CACHE_DIR_OVERRIDE" \
  "$NEMO_RL_DRIVER_VENV_DIR" \
  "$NEMO_RL_VENV_DIR" \
  "$CUSTOM_VLLM_RUNTIME_BASE"

if [[ ! -x "$NEMO_RL_DRIVER_VENV_DIR/bin/ray" ]]; then
  echo "locked driver venv is not prepared: $NEMO_RL_DRIVER_VENV_DIR" >&2
  echo "build it in a matching container before submitting the canary" >&2
  exit 2
fi
export RAY_CLI="$NEMO_RL_DRIVER_VENV_DIR/bin/ray"

sha256sum --check <(printf '%s  %s\n' "$TACTIC_SHA256" "$TACTIC_FILE")
git -C "$NEMO_RL_REPO_ROOT" diff --quiet
git -C "$NEMO_RL_REPO_ROOT" diff --cached --quiet
git -C "$NEMO_RL_REPO_ROOT" pull --ff-only
git -C "$NEMO_RL_REPO_ROOT" submodule update --init --recursive --depth 1
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
