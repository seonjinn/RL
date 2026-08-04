#!/usr/bin/env bash
set -euo pipefail

action=${1:-submit}
scope=${2:-moe}
if [[ "$action" != submit && "$action" != test-only ]]; then
  echo "usage: submit_qwen235_refit_token_smoke_ptyche.sh [submit|test-only] [moe|qkvo]" >&2
  exit 2
fi
if [[ "$scope" != moe && "$scope" != qkvo ]]; then
  echo "unsupported Qwen235 refit scope: $scope" >&2
  exit 2
fi

export NEMO_RL_REPO_ROOT=${NEMO_RL_REPO_ROOT:-/home/sna/nemorl-v0251-mxfp8-safe-adaptive-canary}
export CUSTOM_VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:-/home/sna/mxfp8-safe-backend/vllm-v0251-safe-backend}
export EXPECTED_NEMO_RL_COMMIT=${EXPECTED_NEMO_RL_COMMIT:-$(git -C "$NEMO_RL_REPO_ROOT" rev-parse HEAD)}
export EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-658d7b1571a914bee7df48f717c2a428ee7c45ad}
export CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
export MOUNTS=${MOUNTS:-/lustre:/lustre,/home/sna:/home/sna}
export HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE=/home/sna/.cache/hf-datasets-refit-canary
export UV_CACHE_DIR_OVERRIDE=/home/sna/.cache/uv-refit-canary
export NEMORL_QWEN235_REFIT_SCOPE=$scope
export NEMORL_MXFP8_MOE_BACKEND=${NEMORL_MXFP8_MOE_BACKEND:-flashinfer_trtllm}
export GPUS_PER_NODE=4

timestamp=$(date +%Y%m%d_%H%M%S)
export CANARY_RESULT_ROOT=${CANARY_RESULT_ROOT:-/home/sna/results/nemorl-qwen235-mxfp8-$scope-refit-token-smoke/$timestamp}
export BASE_LOG_DIR="$CANARY_RESULT_ROOT/slurm"
export COMMAND="bash $NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_qwen235_refit_token_smoke.sh"

if [[ -n "$(git -C "$NEMO_RL_REPO_ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "NeMo-RL repository is not clean" >&2
  exit 2
fi
git -C "$NEMO_RL_REPO_ROOT" pull --ff-only
git -C "$NEMO_RL_REPO_ROOT" submodule update --init --recursive --depth 1
actual_nemo_rl_commit=$(git -C "$NEMO_RL_REPO_ROOT" rev-parse HEAD)
if [[ "$actual_nemo_rl_commit" != "$EXPECTED_NEMO_RL_COMMIT" ]]; then
  echo "NeMo-RL commit mismatch: expected $EXPECTED_NEMO_RL_COMMIT, got $actual_nemo_rl_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$CUSTOM_VLLM_SOURCE" status --porcelain --untracked-files=all)" ]]; then
  echo "custom vLLM repository is not clean: $CUSTOM_VLLM_SOURCE" >&2
  exit 2
fi
actual_vllm_commit=$(git -C "$CUSTOM_VLLM_SOURCE" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "$EXPECTED_VLLM_COMMIT" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi

lock_sha=$(sha256sum "$NEMO_RL_REPO_ROOT/uv.lock" | awk '{print $1}')
venv_key=${lock_sha:0:16}-${EXPECTED_VLLM_COMMIT:0:12}
export NEMO_RL_DRIVER_VENV_DIR=${NEMO_RL_DRIVER_VENV_DIR:-/home/sna/.cache/nemorl-driver-v0251-canary/$venv_key}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/home/sna/.cache/nemorl-venvs-v0251-canary/$venv_key}
export CUSTOM_VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:-/home/sna/.cache/vllm-runtime-overlays/$venv_key}
export RAY_CLI="$NEMO_RL_DRIVER_VENV_DIR/bin/ray"
mkdir -p "$CANARY_RESULT_ROOT" "$BASE_LOG_DIR" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR_OVERRIDE"
printf 'nemo_rl_commit=%s\ncustom_vllm_commit=%s\nscope=%s\n' \
  "$actual_nemo_rl_commit" "$actual_vllm_commit" "$scope" \
  > "$CANARY_RESULT_ROOT/provenance.txt"

args=(
  --account=coreai_dlalgo_llm
  --partition=36x2-a01r
  --nodes=16
  --time=05:00:00
  --segment=16
  --job-name=coreai_dlalgo_llm-nemorl.qwen235-mxfp8-$scope-refit-smoke
  --dependency=
  --export=ALL
)
if [[ "$action" == test-only ]]; then
  args+=(--test-only)
fi

cd "$NEMO_RL_REPO_ROOT"
sbatch "${args[@]}" ray.sub
