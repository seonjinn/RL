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
TARGET_REVISION="9216db5781bf21249d130ec9da846c4624c16137"
DRAFT_REVISION="dc84fe7ff1db31efa824776f49c141fc8195eb47"
HF_HOME="${LYRIS_ROOT}/hf_home"
TARGET_MODEL="${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/${TARGET_REVISION}"
DRAFT_MODEL="${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/${DRAFT_REVISION}"
CONTAINER="${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh"

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
    CLUSTER_NUM_NODES=4 \
    CLUSTER_GPUS_PER_NODE=4 \
    CLUSTER_NAME=lyris-gb200 \
    RUNTIME_NAME=nemo-rl \
    RUNTIME_VERSION=nightly-20260707 \
    VLLM_VERSION=0.24.0 \
    VLLM_COMMIT=ee0da84a \
    ACCOUNT=coreai_dlalgo_llm \
    PARTITION=gb200 \
    WALLTIME=04:00:00 \
    WANDB_ENTITY=nvidia \
    GENERATION_EP=1 \
    CONTAINER="${CONTAINER}" \
    EXPERIMENT_ROOT="${ABLATION_ROOT}" \
    WANDB_PROJECT="${ABLATION_PROJECT}" \
    HF_HOME="${HF_HOME}" \
    QWEN32_TARGET_CHECKPOINT_REVISION="${TARGET_REVISION}" \
    QWEN32_DRAFT_CHECKPOINT_REVISION="${DRAFT_REVISION}" \
    QWEN32_TARGET_MODEL="${TARGET_MODEL}" \
    QWEN32_DRAFT_MODEL="${DRAFT_MODEL}" \
    MAX_STEPS=20 \
    NUM_PROMPTS=64 \
    NUM_GENERATIONS=32 \
    TRAIN_GBS=512 \
    MAX_OSL=4096 \
    SPECDEC_CONTEXT_HEADROOM_TOKENS=32 \
    MAX_SEQUENCE_LENGTH=4096 \
    TEMPERATURE=1.0 \
    TOP_P=1.0 \
    SAMPLING=standard \
    DRAFT_SAMPLE_METHOD=probabilistic \
    MAX_NUM_BATCHED_TOKENS=16384 \
    MAX_NUM_SEQS=1024 \
    TAIL_GATE_THRESHOLD=64 \
    TAIL_GATE_CONSECUTIVE_CHECKS=3 \
    RUN_TAG="${RUN_TAG}" \
    bash "${LAUNCHER}" "${MODE}" qwen32b "${variant}"
done
