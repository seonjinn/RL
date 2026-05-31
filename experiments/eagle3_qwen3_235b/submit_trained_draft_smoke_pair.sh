#!/usr/bin/env bash
set -euo pipefail

# Submit or print an apples-to-apples baseline vs trained-Eagle3 smoke pair.
# Use this after modelopt_qwen3_235b_export_vllm.sh has produced VLLM_DRAFT_DIR.
#
# Dry-run:
#   VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
#     bash experiments/eagle3_qwen3_235b/submit_trained_draft_smoke_pair.sh
#
# Submit:
#   SUBMIT=true VLLM_DRAFT_DIR=/path/to/qwen3_235b_eagle3_vllm \
#     bash experiments/eagle3_qwen3_235b/submit_trained_draft_smoke_pair.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:?set VLLM_DRAFT_DIR to the converted vLLM Eagle3 draft directory}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_trained_draft_specdec_smoke_jobs.txt}"

MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
BASELINE_WANDB_NAME="${BASELINE_WANDB_NAME:-qwen3-235b-swe-baseline-trained-draft-smoke}"
SPECDEC_WANDB_NAME="${SPECDEC_WANDB_NAME:-qwen3-235b-swe-eagle3-trained-smoke}"
EAGLE3_NUM_SPEC_TOKENS="${EAGLE3_NUM_SPEC_TOKENS:-3}"
EAGLE3_DRAFT_TP="${EAGLE3_DRAFT_TP:-1}"
ALLOW_MISSING_DRAFT_FOR_DEPENDENCY="${ALLOW_MISSING_DRAFT_FOR_DEPENDENCY:-false}"

draft_has_safetensors() {
  compgen -G "$VLLM_DRAFT_DIR/*.safetensors" >/dev/null
}

if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
  if [[ ! -f "$VLLM_DRAFT_DIR/config.json" ]] || ! draft_has_safetensors; then
    if [[ "$ALLOW_MISSING_DRAFT_FOR_DEPENDENCY" == "true" || "$ALLOW_MISSING_DRAFT_FOR_DEPENDENCY" == "True" ]]; then
      echo "WARN: VLLM_DRAFT_DIR is not complete yet; relying on Slurm dependency before smoke runs: $VLLM_DRAFT_DIR" >&2
    else
      echo "VLLM_DRAFT_DIR must contain config.json and at least one .safetensors weight before submit: $VLLM_DRAFT_DIR" >&2
      exit 1
    fi
  fi
fi

export SUBMIT
export JOB_FILE
export MAX_NUM_STEPS
export BASELINE_WANDB_NAME
export SPECDEC_WANDB_NAME
export EAGLE3_DRAFT_MODEL="$VLLM_DRAFT_DIR"
export EAGLE3_NUM_SPEC_TOKENS
export EAGLE3_DRAFT_TP

bash "$SCRIPT_DIR/submit_static_specdec_smoke_pair.sh"

cat <<EOF

# Trained draft smoke analysis:
python3 experiments/eagle3_qwen3_235b/analyze_static_specdec_smoke_pair.py \\
  --job-file $JOB_FILE \\
  --repo-root "${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --markdown-out /path/to/trained_draft_specdec_smoke.md \\
  --json-out /path/to/trained_draft_specdec_smoke.json \\
  --fail-on-missing-spec-metrics
EOF
