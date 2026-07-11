#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

MODE="${1:-dry-run}"
case "${MODE}" in
  dry-run|test-only|submit) ;;
  *)
    echo "ERROR: mode must be dry-run, test-only, or submit" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/submit_tail_gated_specdec_step20.sh"
RUN_TAG="${RUN_TAG:-qwen32b-cudagraph-off-step20-20260710}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
ABLATION_ROOT="${LYRIS_ROOT}/experiments/vllm024-cudagraph-off/${RUN_TAG}"
ABLATION_PROJECT="nemorl-vllm024-cudagraph-off-step20-lyris"

variants=(
  baseline_v2
  always_on_v2_k5
  fastrl_threshold_v2_k5
  baseline_v1
  always_on_v1_k5
  stock_dynamic_v1
)

for variant in "${variants[@]}"; do
  env \
    ABLATION_BEHAVIOR_REVISION=539cfb96f3944ea6e32616ec43e10f4d1cf20491 \
    CUDA_GRAPH_MODE=off \
    EXPERIMENT_ROOT="${ABLATION_ROOT}" \
    WANDB_PROJECT="${ABLATION_PROJECT}" \
    MAX_STEPS=20 \
    DRAFT_SAMPLE_METHOD=probabilistic \
    TAIL_GATE_THRESHOLD=64 \
    TAIL_GATE_CONSECUTIVE_CHECKS=3 \
    RUN_TAG="${RUN_TAG}" \
    bash "${LAUNCHER}" "${MODE}" qwen32b "${variant}"
done
