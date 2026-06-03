#!/bin/bash

# ops/eval.sh <checkpoint> <benchmark> [benchmark ...]
#
# 16K reasoning mode evaluation (apples-to-apples with smohs's v6 baseline evals).
# For 1024-token NORMAL mode, use ops/eval_normal.sh instead.
#
# Eval mode: --reasoning --temperature 0.6 --top-p 0.95 --vllm-max-tokens 16384
# Output dir: benchmarks-reasoning-16384/vllm_local/
#
# Default benchmark suite:
#   MMLongBench_DOC AI2D_TEST ChartQA_TEST DocVQA_VAL MathVista_MINI
#   MMMU_DEV_VAL InfoVQA_VAL OCRBench OCRBenchV2 TextVQA_VAL
#   OCR_Reasoning WeMath MathVerse_MINI_Vision_Only MathVision LogicVista
#   CharXiv_reasoning_val ScreenSpot_Pro
#
# Checkpoint formats:
#   /abs/path/to/step_N          - direct path (NeMo-RL, HF, or Megatron SFT)
#   run-name[@step]              - NeMo-RL run under results/
#   sft:run-name[@iter]          - Megatron SFT run under workspace/output/
#
# Examples:
#   ops/eval.sh grpo-nanov3vl-sft783-unans-16k@66 MMLongBench_DOC
#   ops/eval.sh grpo-nanov3vl-sft783-unans-16k MMLongBench_DOC MMMU_DEV_VAL
#   ops/eval.sh /abs/path/to/step_66 MMLongBench_DOC

set -euo pipefail

ROOT=$(realpath "$(dirname "$0")/..")
NEMORL="$ROOT"
WORKSPACE=/lustre/fsw/portfolios/llmservice/users/smohsenitahe/workspace/output

set -a; source "$ROOT/.env"; set +a

# Use ikarmanov's VLMEvalKitMcore (verified working with our 16K reasoning evals)
VLMEVALKIT=/lustre/fsw/portfolios/llmservice/users/smohsenitahe/VLMEvalKitMcore_super/VLMEvalKitMcore
MEGATRON_SRC=/lustre/fsw/portfolios/llmservice/users/smohsenitahe/megatron_super/megatron-lm
SERVE_BIN=
VLLM_CHAT_TMPL=
CONTAINER_IMAGE=/lustre/fsw/portfolios/llmservice/users/cmccarthy/vllm_containers/vllm-openai-v0.20.0-vlmeval-super-omni.sqsh
SLURM_ACCOUNT=${SLURM_ACCOUNT:-llmservice_fm_vision}
PARTITIONS=${PARTITIONS:-batch,batch_long}
LONG_PARTITIONS=${LONG_PARTITIONS:-batch_long}
CONVERT_PARTITIONS=${CONVERT_PARTITIONS:-$PARTITIONS}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-8}
VLLM_LOCAL_BS=${VLLM_LOCAL_BS:-64}
OPENAI_API_BASE=${OPENAI_API_BASE:-${OPENAI_BASE_URL:-https://inference-api.nvidia.com/v1/chat/completions}}
NLTK_DATA=${NLTK_DATA:-/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/smohsenitahe/nltk_data}
VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}

# Output subdir = mode signature; matches smohs v6 evals for apples-to-apples comparison
EVAL_MODE_SUBDIR="benchmarks-reasoning-16384"
# Eval args: 16K reasoning mode (matches what was used for v6 + the targeted ScreenSpot/MMLongBench evals)
EVAL_ARGS="--reasoning --temperature 0.6 --top-p 0.95 --vllm-max-tokens 16384 --fps 2 --nframe -1 --nframe-max 128 --retry 10 --api-nproc 1 --reuse"

DEFAULT_BENCHMARKS=(
    "MMLongBench_DOC" "AI2D_TEST" "ChartQA_TEST" "DocVQA_VAL" "MathVista_MINI"
    "MMMU_DEV_VAL" "InfoVQA_VAL" "OCRBench" "OCRBenchV2" "TextVQA_VAL"
    "OCR_Reasoning" "WeMath" "MathVerse_MINI_Vision_Only" "MathVision" "LogicVista"
    "CharXiv_reasoning_val" "ScreenSpot_Pro"
)

if [[ $# -lt 1 ]]; then
    echo "Usage: ops/eval.sh <checkpoint> [benchmark] [benchmark] ..."
    echo ""
    echo "If no benchmarks are provided, the default benchmark suite is used."
    echo ""
    echo "Checkpoint formats:"
    echo "  /path/to/checkpoint       - direct path to checkpoint dir"
    echo "  run-name[@step]           - NeMo-RL run (from results/)"
    echo "  sft:run-name[@iter]       - Megatron SFT run (from workspace/output/)"
    exit 1
fi

CKPT_ARG="$1"

BENCHMARKS=("${@:2}")
if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    BENCHMARKS=("${DEFAULT_BENCHMARKS[@]}")
    echo "No benchmarks provided; using default benchmark suite (${#BENCHMARKS[@]} benchmarks)."
fi

# Normalize CKPT_ARG to a directory path
if [[ -d "$CKPT_ARG" ]]; then
    if [[ "$(basename "$CKPT_ARG")" == "mcore_to_hf" ]]; then
        CKPT_DIR=$(realpath "$(dirname "$CKPT_ARG")")
    else
        CKPT_DIR=$(realpath "$CKPT_ARG")
    fi
elif [[ "$CKPT_ARG" == sft:* ]]; then
    SFT_ARG="${CKPT_ARG#sft:}"
    RUN_NAME="${SFT_ARG%@*}"
    if [[ "$SFT_ARG" == *"@"* ]]; then
        ITER_NUM="${SFT_ARG##*@}"
        FOLDER_NAME=$(printf "iter_%07d" "$ITER_NUM")
    else
        FOLDER_NAME=$(ls -1d "$WORKSPACE/${RUN_NAME}/checkpoints/iter_"* 2>/dev/null | sort -t '_' -k2 -n | tail -n 1 | xargs basename)
    fi
    CKPT_DIR="$WORKSPACE/${RUN_NAME}/checkpoints/${FOLDER_NAME}"
elif [[ -d "$NEMORL/results/${CKPT_ARG%@*}" ]]; then
    RUN_NAME="${CKPT_ARG%@*}"
    if [[ "$CKPT_ARG" == *"@"* ]]; then
        CKPT_STEP="${CKPT_ARG##*@}"
    else
        CKPT_STEP=$(ls -1 "$NEMORL/results/${RUN_NAME}" | sed 's/step_//' | sort -n -r | head -n 1)
    fi
    CKPT_DIR="$NEMORL/results/${RUN_NAME}/step_${CKPT_STEP}"
else
    echo "Checkpoint not found: $CKPT_ARG" >&2
    exit 1
fi

echo "=== Checkpoint: $CKPT_DIR ==="

DEPENDENCY_ARG=""
if [[ -d "${CKPT_DIR}/policy/weights" ]]; then
    # NeMo-RL checkpoint (Torch DCP format)
    CKPT_STEP=$(echo "$CKPT_DIR" | sed 's/.*step_\([0-9]*\)/\1/')
    CKPT_NAME=$(basename "$(dirname "$CKPT_DIR")")_step_${CKPT_STEP}
    MODEL_HF_ROOT=$(dirname "$CKPT_DIR")/tp_1_hf/iter_${CKPT_STEP}
    FOLDER_NAME=.
    if [[ ! -f "${MODEL_HF_ROOT}/mcore_to_hf/config.json" || \
          ! -f "${MODEL_HF_ROOT}/mcore_to_hf/model.safetensors.index.json" ]] || \
       ! ls "${MODEL_HF_ROOT}/mcore_to_hf"/*.safetensors >/dev/null 2>&1; then
        echo "Complete Super HF checkpoint not found — submitting DCP→HF conversion job..."
        cd "$NEMORL"
        CONVERT_JOB_ID=$(sbatch --parsable \
            -p "$CONVERT_PARTITIONS" \
            --account "$SLURM_ACCOUNT" \
            --job-name "convert-${CKPT_NAME}" \
            ops/convert_dcp.sh "$CKPT_DIR" "$MODEL_HF_ROOT/mcore_to_hf")
        echo "Conversion job: $CONVERT_JOB_ID"
        echo "${CONVERT_JOB_ID},convert-${CKPT_NAME},$(date -Iseconds)" >> "${NEMORL}/eval_jobs.csv"
        DEPENDENCY_ARG="--dependency=afterok:${CONVERT_JOB_ID}"
    else
        echo "Super HF checkpoint exists at $MODEL_HF_ROOT/mcore_to_hf — skipping conversion."
    fi
elif [[ -f "${CKPT_DIR}/config.json" || -f "${CKPT_DIR}/mcore_to_hf/config.json" ]]; then
    # Already an HF checkpoint
    if [[ "$(basename "${CKPT_DIR}")" == "mcore_to_hf" ]]; then
        MODEL_HF_ROOT="$(dirname "$CKPT_DIR")"
        CKPT_NAME="$(basename "$MODEL_HF_ROOT")"
    else
        MODEL_HF_ROOT="$CKPT_DIR"
        CKPT_NAME="$(basename "$MODEL_HF_ROOT")"
        ln -s . "${MODEL_HF_ROOT}/mcore_to_hf" 2>/dev/null || true
    fi
    FOLDER_NAME=.
elif [[ "$(basename "$CKPT_DIR")" == iter_* ]]; then
    # Legacy Megatron SFT checkpoint
    ITER_FOLDER_NAME=$(basename "$CKPT_DIR")
    MCORE_PATH=$(dirname "$CKPT_DIR")
    RUN_DIR=$(dirname "$MCORE_PATH")
    RUN_NAME=$(basename "$RUN_DIR")
    CKPT_NAME="${RUN_NAME}_${ITER_FOLDER_NAME}"
    MODEL_HF_ROOT="${MCORE_PATH}/tp_1_hf/${ITER_FOLDER_NAME}"
    FOLDER_NAME=.
    if [[ ! -f "${MODEL_HF_ROOT}/mcore_to_hf/config.json" ]]; then
        echo "HF checkpoint not found — submitting Megatron→HF conversion job..."
        cd "$VLMEVALKIT"
        CONVERT_JOB_ID=$(sbatch --parsable \
            -p ${PARTITIONS} \
            --gres=gpu:1 \
            --mem=300000M \
            --time=1:00:00 \
            --job-name "convert-${CKPT_NAME}" \
            shell/convert_to_hf.sh \
            --model-ckpt-dir "$MCORE_PATH" \
            --folder-name "$ITER_FOLDER_NAME" \
            --model-hf-ckpt-dir "${MCORE_PATH}/tp_1_hf" \
            --output-dir "$MODEL_HF_ROOT" \
            --megatron-src "$MEGATRON_SRC" \
            --model-size "30_3b")
        echo "Conversion job: $CONVERT_JOB_ID"
        DEPENDENCY_ARG="--dependency=afterok:${CONVERT_JOB_ID}"
    fi
else
    echo "Unknown checkpoint format at $CKPT_DIR" >&2
    exit 1
fi

for BENCHMARK in "${BENCHMARKS[@]}"; do

OUTPUT_DIR="$MODEL_HF_ROOT/$FOLDER_NAME/${EVAL_MODE_SUBDIR}/vllm_local"
if [[ -d "$OUTPUT_DIR" ]] && \
   find "$OUTPUT_DIR" -maxdepth 2 -name "vllm_local_${BENCHMARK}*" \
       \( -name "*_acc.csv" -o -name "*_score.csv" -o -name "*_score.json" -o -name "*_results.json" \) \
       2>/dev/null | grep -q .; then
    echo "Skipping $BENCHMARK — already evaluated at $OUTPUT_DIR"
    continue
fi

BENCH_EVAL_ARGS="$EVAL_ARGS"
BENCH_GPUS_PER_NODE=8
BENCH_VLLM_TP_SIZE="$VLLM_TP_SIZE"
BENCH_MEM_ARGS=()
if [[ -d "$OUTPUT_DIR" ]] && \
   find "$OUTPUT_DIR" -maxdepth 2 -name "vllm_local_${BENCHMARK}.xlsx" \
       2>/dev/null | grep -q .; then
    echo "Found cached predictions for $BENCHMARK — submitting eval-only scoring job."
    BENCH_EVAL_ARGS="$BENCH_EVAL_ARGS --mode eval"
    BENCH_GPUS_PER_NODE=1
    BENCH_VLLM_TP_SIZE=1
    BENCH_MEM_ARGS=(--mem=251740M)
fi

# Long-running benchmarks need 4hr partitions; others can use 2hr partitions
JOB_PARTITIONS="$PARTITIONS"
if [[ $BENCHMARK == "MathVision" || \
      $BENCHMARK == "AI2D_TEST" || \
      $BENCHMARK == "MMLongBench_DOC" || \
      $BENCHMARK == "MathVista_MINI" || \
      $BENCHMARK == "OCRBenchV2" || \
      $BENCHMARK == "OCR_Reasoning" || \
      $BENCHMARK == "RefCOCO" || \
      $BENCHMARK == "SLIDEVQA" || \
      $BENCHMARK == "TextVQA_VAL" || \
      $BENCHMARK == "Video-MME" || \
      $BENCHMARK == "ScreenSpotV2" || \
      $BENCHMARK == "ScreenSpot_Pro" || \
      $BENCHMARK == "HallusionBench" || \
      $BENCHMARK == "CharXiv_reasoning_val" || \
      $BENCHMARK == "MMVet" || \
      $BENCHMARK == "WeMath" || \
      $BENCHMARK == "LogicVista" || \
      $BENCHMARK == "MathVerse_MINI_Vision_Only" || \
      $BENCHMARK == "MMMU_DEV_VAL" ]]; then
    TIME_LIMIT="4:00:00"
    JOB_PARTITIONS="$LONG_PARTITIONS"
else
    TIME_LIMIT="2:00:00"
fi

export VLLM_CACHE_ROOT=/tmp/vllm_cache
export VLLM_CLIENT_WORKERS=64
export VLLM_LOCAL_BS="$VLLM_LOCAL_BS"
export VLLM_MAX_NUM_SEQS=128
export VLLM_NO_USAGE_STATS
export NLTK_DATA
export NCCL_TIMEOUT=14400
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
unset PYTHONPATH

mkdir -p "$MODEL_HF_ROOT/$FOLDER_NAME/${EVAL_MODE_SUBDIR}/benchmark_logs"
echo "Submitting $BENCHMARK eval (output: $OUTPUT_DIR)..."
cd "$VLMEVALKIT"
JOB_OUTPUT=$(sbatch \
    -p "$JOB_PARTITIONS" \
    --job-name "eval-${CKPT_NAME}-${BENCHMARK}" \
    --time "$TIME_LIMIT" \
    --nodes 1 \
    --gpus-per-node "$BENCH_GPUS_PER_NODE" \
    --ntasks 1 \
    --ntasks-per-node 1 \
    --exclusive \
    "${BENCH_MEM_ARGS[@]}" \
    --account "$SLURM_ACCOUNT" \
    --export="ALL,NLTK_DATA=${NLTK_DATA},VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS}" \
    ${DEPENDENCY_ARG} \
    shell/run_one_benchmark_vllm_auto.sh \
    --benchmark "$BENCHMARK" \
    --folder-name "$FOLDER_NAME" \
    --model-hf-ckpt-dir "$MODEL_HF_ROOT" \
    --output-dir "$MODEL_HF_ROOT/$FOLDER_NAME/${EVAL_MODE_SUBDIR}" \
    --openai-api-key "$OPENAI_API_KEY" \
    --openai-api-base "$OPENAI_API_BASE" \
    --megatron-src "$MEGATRON_SRC" \
    --vlmevalkit-src "$VLMEVALKIT" \
    --vllm-chat-tpl "$VLLM_CHAT_TMPL" \
    --serve-bin "$SERVE_BIN" \
    --container-image "$CONTAINER_IMAGE" \
    --eval-args "$BENCH_EVAL_ARGS" \
    --vllm-tp-size "$BENCH_VLLM_TP_SIZE")
echo "$JOB_OUTPUT"
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+$')
echo "${JOB_ID},eval-${CKPT_NAME}-${BENCHMARK},$(date -Iseconds)" >> "${NEMORL}/eval_jobs.csv"

done
