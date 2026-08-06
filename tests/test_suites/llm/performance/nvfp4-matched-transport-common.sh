#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

NUM_NODES=2
GPUS_PER_NODE=8
SEGMENT_SIZE=1
MAX_STEPS=${MAX_STEPS:-2}
STEPS_PER_RUN=$MAX_STEPS
NUM_RUNS=1
NUM_MINUTES=240
SNAPSHOT_MEGATRON_BRIDGE=1

if [[ ! "$MAX_STEPS" =~ ^[0-9]+$ ]] || ((MAX_STEPS < 2)); then
  echo "[ERROR] MAX_STEPS must be an integer >= 2, got '$MAX_STEPS'"
  exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB_API_KEY must be exported"
  exit 2
fi

REFIT_TRANSPORT=${REFIT_TRANSPORT:-null}
case "$REFIT_TRANSPORT" in
  null)
    TRANSPORT_TAG=collective
    ;;
  nccl_reshard)
    TRANSPORT_TAG=nccl-reshard
    ;;
  *)
    echo "[ERROR] REFIT_TRANSPORT must be null or nccl_reshard, got '$REFIT_TRANSPORT'"
    exit 2
    ;;
esac

case "$QUANT_MODE" in
  w4a16)
    QUANT_PATTERN='quantization=nemo_modelopt_w4a16_nvfp4|quant_algo[=: '\''"]+W4A16_NVFP4'
    QUANT_OVERRIDES=()
    ;;
  w4a4)
    if [[ -z "${NVFP4_CALIBRATION_ARTIFACT:-}" ]]; then
      echo "[ERROR] NVFP4_CALIBRATION_ARTIFACT must name a W4A4 calibration artifact"
      exit 2
    fi
    if [[ ! -f "$NVFP4_CALIBRATION_ARTIFACT" ]]; then
      echo "[ERROR] W4A4 calibration artifact not found: $NVFP4_CALIBRATION_ARTIFACT"
      exit 2
    fi
    QUANT_PATTERN='quantization=nemo_modelopt_nvfp4([^_[:alnum:]]|$)|quant_algo[=: '\''"]+NVFP4([^_[:alnum:]]|$)'
    QUANT_OVERRIDES=(
      "policy.generation.real_quant_calibration_path=$NVFP4_CALIBRATION_ARTIFACT"
    )
    ;;
  *)
    echo "[ERROR] QUANT_MODE must be w4a16 or w4a4, got '$QUANT_MODE'"
    exit 2
    ;;
esac

EXP_DIR="$SCRIPT_DIR/${EXP_NAME}-${TRANSPORT_TAG}-${MAX_STEPS}step"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
WANDB_PROJECT_OVERRIDE=${WANDB_PROJECT_OVERRIDE:-sna-bf16-nvfp4-matched-transport}
WANDB_NAME_OVERRIDE=${WANDB_NAME_OVERRIDE:-${EXP_NAME}-${TRANSPORT_TAG}-${MAX_STEPS}step}
mkdir -p "$EXP_DIR" "$LOG_DIR"

assert_grep() {
  local pattern=$1
  local file=$2
  local message=$3
  if ! grep -Eq -- "$pattern" "$file"; then
    echo "[ERROR] $message"
    exit 1
  fi
}

assert_not_grep() {
  local pattern=$1
  local file=$2
  local message=$3
  if grep -Eiq -- "$pattern" "$file"; then
    echo "[ERROR] $message"
    exit 1
  fi
}

cd "$PROJECT_ROOT"
uv run --no-sync examples/run_grpo.py \
  --config "$CONFIG_PATH" \
  "$@" \
  grpo.max_num_steps=$MAX_STEPS \
  cluster.num_nodes=$NUM_NODES \
  cluster.gpus_per_node=$GPUS_PER_NODE \
  cluster.segment_size=$SEGMENT_SIZE \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=$GPUS_PER_NODE \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.generation.refit_transport=$REFIT_TRANSPORT \
  logger.log_dir="$LOG_DIR" \
  logger.wandb_enabled=True \
  logger.wandb.project="$WANDB_PROJECT_OVERRIDE" \
  logger.wandb.name="$WANDB_NAME_OVERRIDE" \
  logger.monitor_gpus=True \
  logger.tensorboard_enabled=True \
  checkpointing.enabled=false \
  "${QUANT_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

assert_grep "VllmQuantInternalWorkerExtension" "$RUN_LOG" \
  "Real ModelOpt vLLM worker extension was not selected"
assert_grep "Detected ModelOpt NVFP4 checkpoint" "$RUN_LOG" \
  "vLLM did not detect a real ModelOpt NVFP4 checkpoint"
assert_grep "$QUANT_PATTERN" "$RUN_LOG" \
  "Expected ModelOpt $QUANT_MODE quantization method was not detected"
assert_grep "MegatronPolicyWorker" "$RUN_LOG" \
  "Plain BF16 Megatron policy worker was not selected"

assert_not_grep "MegatronQuantPolicyWorker|FakeQuantWorker|VLLM_QUANT_CFG" "$RUN_LOG" \
  "Run selected a QARL or fake-quant policy path"
assert_not_grep "Traceback \(most recent call last\)|Policy generation refit failed|ModelOpt real-quant.*failed|Exception during (collective_rpc|nccl_reshard_refit)|Error: Worker failed" "$RUN_LOG" \
  "Weight refit raised an exception"
assert_not_grep "(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)" "$RUN_LOG" \
  "Run log contains a NaN or infinity"

expected_refits=$((MAX_STEPS - 1))
uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
  "len(data[\"train/loss\"]) == $MAX_STEPS" \
  "len(data[\"timing/train/prepare_for_generation/transfer_and_update_weights\"]) >= $expected_refits"

echo "[PASS] BF16 to NVFP4 $QUANT_MODE $TRANSPORT_TAG completed $MAX_STEPS GRPO steps"
