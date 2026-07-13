#!/usr/bin/env bash
# Per-model presets mirroring one vLLM DP-worker shard of the NeMo-RL GB200
# SyncRL recipes (grpo-qwen3-30ba3b-4n4g / grpo-qwen3-32b-4n4g /
# grpo-qwen3-235b-16n4g). Usage:
#   MODE=profile ./submit_matrix_lyris.sh <qwen3_30ba3b|qwen3_32b|qwen3_235b> <math|swe> [extra env]
#   MODE=rollout DYNAMIC_SPEC_JSON=... ./submit_matrix_lyris.sh qwen3_30ba3b math
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:?preset: qwen3_30ba3b | qwen3_32b | qwen3_235b}"
BENCH_SEL="${2:?bench: math | swe}"

REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark}"

case "${PRESET}" in
  qwen3_30ba3b)
    # NeMo-RL recipe target (base, hybrid-thinking); drafter is the Thinking speculator
    export MODEL="Qwen/Qwen3-30B-A3B"
    export MODEL_LABEL="qwen3_30ba3b"
    export DRAFT_MODEL="RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3"
    export TP=1
    # recipe: 64 prompts x 32 gens over 16 TP1 engines -> 128 seqs/engine
    export NUM_PROMPTS_PER_STEP=4
    export NUM_GENERATIONS_PER_PROMPT=32
    export MAX_NUM_SEQS=128
    export MAX_TOKENS=4096
    export BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128}"
    ;;
  qwen3_32b)
    export MODEL="Qwen/Qwen3-32B"
    export MODEL_LABEL="qwen3_32b"
    export DRAFT_MODEL="RedHatAI/Qwen3-32B-Thinking-speculator.eagle3"
    export TP=2
    # recipe: 64 prompts x 32 gens over 8 TP2 engines -> 256 seqs/engine
    export NUM_PROMPTS_PER_STEP=8
    export NUM_GENERATIONS_PER_PROMPT=32
    export MAX_NUM_SEQS=256
    export MAX_TOKENS=4096
    export BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128 256}"
    ;;
  qwen3_235b)
    # NeMo-RL recipe target (base); Thinking speculator's verifier is this base model
    export MODEL="Qwen/Qwen3-235B-A22B"
    export MODEL_LABEL="qwen3_235b"
    export DRAFT_MODEL="RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3"
    # recipe uses vLLM TP8 across 4-GPU nodes; standalone single node caps TP=4
    export TP=4
    # recipe: 16 prompts x 32 gens over 8 TP8 engines -> 64 seqs/engine
    export NUM_PROMPTS_PER_STEP=2
    export NUM_GENERATIONS_PER_PROMPT=32
    export MAX_NUM_SEQS=64
    export MAX_TOKENS=8192
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
    export CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-1 2 4 8 16 32 64}"
    export BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64}"
    export TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
    ;;
  *)
    echo "ERROR: unknown preset '${PRESET}'" >&2
    exit 2
    ;;
esac

case "${BENCH_SEL}" in
  math|math500)
    export BENCH="math500"
    export PROMPT_JSONL="${REMOTE_REPO}/data/math500_prompts_full.jsonl"
    export ISL_CAP="${ISL_CAP:-1024}"
    ;;
  openmath)
    export BENCH="openmath"
    export PROMPT_JSONL="${REMOTE_REPO}/data/openmath2_prompts_2048.jsonl"
    export ISL_CAP="${ISL_CAP:-1024}"
    ;;
  dapo)
    export BENCH="dapo"
    export PROMPT_JSONL="${REMOTE_REPO}/data/dapo_math_prompts_2048.jsonl"
    export ISL_CAP="${ISL_CAP:-1024}"
    ;;
  swe|swe_verified)
    export BENCH="swe_verified"
    export PROMPT_JSONL="${REMOTE_REPO}/data/swebench_verified_prompts_all.jsonl"
    export ISL_CAP="${ISL_CAP:-4096}"
    ;;
  swe_full)
    export BENCH="swe_full"
    export PROMPT_JSONL="${REMOTE_REPO}/data/swebench_full_test_prompts_all.jsonl"
    export ISL_CAP="${ISL_CAP:-4096}"
    ;;
  *)
    echo "ERROR: unknown bench '${BENCH_SEL}'" >&2
    exit 2
    ;;
esac

export JOB_FILE="${JOB_FILE:-${SCRIPT_DIR}/latest_lyris_${PRESET}_${BENCH_SEL}_${MODE:-profile}_jobs.txt}"
bash "${SCRIPT_DIR}/submit_lyris_dynamic_sd.sh"
