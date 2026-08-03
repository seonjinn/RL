#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-submit}
if [[ "$ACTION" != submit && "$ACTION" != test-only ]]; then
  echo "usage: submit_qwen235_32k_trace_ptyche.sh [submit|test-only]" >&2
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
  VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64; do
  unset "$name"
done

timestamp=$(date +%Y%m%d_%H%M%S)
export CANARY_RESULT_ROOT=${CANARY_RESULT_ROOT:-/home/sna/results/nemorl-qwen235-mxfp8-32k-shape-trace/$timestamp}
export SHAPE_TRACE_DIR=${SHAPE_TRACE_DIR:-$CANARY_RESULT_ROOT/trace/raw}
export SHAPE_TRACE_MAX=${SHAPE_TRACE_MAX:-16384}
export CANARY_CONFIG=${CANARY_CONFIG:-$NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout_v0251/configs/eval_qwen3_235ba22b_32k_cuda_graph_trace.yaml}
export CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
export MOUNTS=${MOUNTS:-/lustre:/lustre,/home/sna:/home/sna}
export GPUS_PER_NODE=4
export BASE_LOG_DIR="$CANARY_RESULT_ROOT/slurm"
export COMMAND="bash $NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_trace.sh"
export UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-/home/sna/.cache/uv-canary}

git -C "$NEMO_RL_REPO_ROOT" diff --quiet
git -C "$NEMO_RL_REPO_ROOT" diff --cached --quiet
git -C "$NEMO_RL_REPO_ROOT" pull --ff-only
git -C "$NEMO_RL_REPO_ROOT" submodule update --init --recursive --depth 1
test "$(git -C "$CUSTOM_VLLM_SOURCE" rev-parse HEAD)" = "$EXPECTED_VLLM_COMMIT"
git -C "$CUSTOM_VLLM_SOURCE" diff --quiet
git -C "$CUSTOM_VLLM_SOURCE" diff --cached --quiet

lock_sha=$(sha256sum "$NEMO_RL_REPO_ROOT/uv.lock" | awk '{print $1}')
venv_key=${lock_sha:0:16}-${EXPECTED_VLLM_COMMIT:0:12}
export NEMO_RL_DRIVER_VENV_DIR=${NEMO_RL_DRIVER_VENV_DIR:-/home/sna/.cache/nemorl-driver-v0251-canary/$venv_key}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/home/sna/.cache/nemorl-venvs-v0251-canary/$venv_key}
export CUSTOM_VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:-/home/sna/.cache/vllm-runtime-overlays/$venv_key}
mkdir -p "$BASE_LOG_DIR" "$SHAPE_TRACE_DIR" "$UV_CACHE_DIR_OVERRIDE" \
  "$NEMO_RL_DRIVER_VENV_DIR" "$NEMO_RL_VENV_DIR" "$CUSTOM_VLLM_RUNTIME_BASE"
test -x "$NEMO_RL_DRIVER_VENV_DIR/bin/ray"
export RAY_CLI="$NEMO_RL_DRIVER_VENV_DIR/bin/ray"

args=(
  --account=coreai_dlalgo_llm
  --partition=36x2-a01r
  --nodes=2
  --time=05:00:00
  --segment=2
  --job-name=coreai_dlalgo_llm-nemorl.qwen235-mxfp8-32k-trace
  --dependency=
  --export=ALL
)
if [[ "$ACTION" == test-only ]]; then
  args+=(--test-only)
fi

cd "$NEMO_RL_REPO_ROOT"
sbatch "${args[@]}" ray.sub
