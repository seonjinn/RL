#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-submit}
if [[ "$ACTION" != submit && "$ACTION" != test-only ]]; then
  echo "usage: submit_qwen235_gsm8k_correctness_ptyche.sh [submit|test-only]" >&2
  exit 2
fi

export NEMO_RL_REPO_ROOT=${NEMO_RL_REPO_ROOT:-/home/sna/nemorl-v0251-mxfp8-safe-adaptive-canary}
export CUSTOM_VLLM_SOURCE=${CUSTOM_VLLM_SOURCE:-/home/sna/mxfp8-safe-backend/vllm-v0251-safe-backend}
export EXPECTED_NEMO_RL_COMMIT=${EXPECTED_NEMO_RL_COMMIT:-$(git -C "$NEMO_RL_REPO_ROOT" rev-parse HEAD)}
export EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-658d7b1571a914bee7df48f717c2a428ee7c45ad}

require_clean_repo() {
  local repository=$1
  if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "repository is not clean: $repository" >&2
    exit 2
  fi
}

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

artifact_dir=${TACTIC_ARTIFACT_DIR:-/home/sna/mxfp8-safe-backend/vllm-benchmark-qwen235-shmoo/experiments/sweep/data/microbench/qwen235_tp4ep4_8x4_fix1_20260802}
export TACTIC_FILE="$artifact_dir/exact_tactics.json"
export TACTIC_SHA256=${TACTIC_SHA256:-bf1630d7327d58b6742ab0359c5993b59dc28c4ec96d9008c9fe0a1e399c189e}
LAYER_ALLOWLIST_FILE="$artifact_dir/layer_allowlist.txt"
export LAYER_ALLOWLIST_B64
LAYER_ALLOWLIST_B64=$(base64 < "$LAYER_ALLOWLIST_FILE")

timestamp=$(date +%Y%m%d_%H%M%S)
export CANARY_RESULT_ROOT=${CANARY_RESULT_ROOT:-/home/sna/results/nemorl-qwen235-mxfp8-gsm8k-correctness/$timestamp}
export CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
export MOUNTS=${MOUNTS:-/lustre:/lustre,/home/sna:/home/sna}
export HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE=/home/sna/.cache/hf-datasets-canary
export GPUS_PER_NODE=4
export BASE_LOG_DIR="$CANARY_RESULT_ROOT/slurm"
export NEMORL_ENABLE_QWEN235_GSM8K_CORRECTNESS=1
export COMMAND="bash $NEMO_RL_REPO_ROOT/experiments/mxfp8_adaptive_rollout_v0251/run_qwen235_gsm8k_correctness.sh"
export UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-/home/sna/.cache/uv-canary}

if [[ ! -d "$HF_HUB_CACHE/models--Qwen--Qwen3-235B-A22B" ]]; then
  echo "missing Qwen3-235B Hub cache: $HF_HUB_CACHE/models--Qwen--Qwen3-235B-A22B" >&2
  exit 2
fi
mkdir -p "$HF_DATASETS_CACHE"

require_clean_repo "$NEMO_RL_REPO_ROOT"
git -C "$NEMO_RL_REPO_ROOT" pull --ff-only
git -C "$NEMO_RL_REPO_ROOT" submodule update --init --recursive --depth 1
actual_nemo_rl_commit=$(git -C "$NEMO_RL_REPO_ROOT" rev-parse HEAD)
if [[ "$actual_nemo_rl_commit" != "$EXPECTED_NEMO_RL_COMMIT" ]]; then
  echo "NeMo-RL commit mismatch: expected $EXPECTED_NEMO_RL_COMMIT, got $actual_nemo_rl_commit" >&2
  exit 2
fi
require_clean_repo "$NEMO_RL_REPO_ROOT"

require_clean_repo "$CUSTOM_VLLM_SOURCE"
actual_vllm_commit=$(git -C "$CUSTOM_VLLM_SOURCE" rev-parse HEAD)
if [[ "$actual_vllm_commit" != "$EXPECTED_VLLM_COMMIT" ]]; then
  echo "custom vLLM commit mismatch: expected $EXPECTED_VLLM_COMMIT, got $actual_vllm_commit" >&2
  exit 2
fi

sha256sum --check <(printf '%s  %s\n' "$TACTIC_SHA256" "$TACTIC_FILE")

mkdir -p "$CANARY_RESULT_ROOT"
printf 'nemo_rl_commit=%s\ncustom_vllm_commit=%s\ntactic_sha256=%s\n' \
  "$actual_nemo_rl_commit" "$actual_vllm_commit" "$TACTIC_SHA256" \
  > "$CANARY_RESULT_ROOT/provenance.txt"

lock_sha=$(sha256sum "$NEMO_RL_REPO_ROOT/uv.lock" | awk '{print $1}')
venv_key=${lock_sha:0:16}-${EXPECTED_VLLM_COMMIT:0:12}
export NEMO_RL_DRIVER_VENV_DIR=${NEMO_RL_DRIVER_VENV_DIR:-/home/sna/.cache/nemorl-driver-v0251-canary/$venv_key}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/home/sna/.cache/nemorl-venvs-v0251-canary/$venv_key}
export CUSTOM_VLLM_RUNTIME_BASE=${CUSTOM_VLLM_RUNTIME_BASE:-/home/sna/.cache/vllm-runtime-overlays/$venv_key}
mkdir -p "$BASE_LOG_DIR" "$UV_CACHE_DIR_OVERRIDE" "$NEMO_RL_DRIVER_VENV_DIR" \
  "$NEMO_RL_VENV_DIR" "$CUSTOM_VLLM_RUNTIME_BASE"
test -x "$NEMO_RL_DRIVER_VENV_DIR/bin/ray"
export RAY_CLI="$NEMO_RL_DRIVER_VENV_DIR/bin/ray"

args=(
  --account=coreai_dlalgo_llm
  --partition=36x2-a01r
  --nodes=2
  --time=05:00:00
  --segment=2
  --job-name=coreai_dlalgo_llm-nemorl.qwen235-mxfp8-gsm8k
  --dependency=
  --export=ALL
)
if [[ "$ACTION" == test-only ]]; then
  args+=(--test-only)
fi

cd "$NEMO_RL_REPO_ROOT"
sbatch "${args[@]}" ray.sub
