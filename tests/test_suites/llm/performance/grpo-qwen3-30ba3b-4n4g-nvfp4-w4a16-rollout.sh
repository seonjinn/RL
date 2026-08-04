#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=8
SEGMENT_SIZE=${SCHEDULER_SEGMENT_SIZE:-2}
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=240
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB_API_KEY must be exported for this smoke run"
  exit 2
fi

REFIT_TRANSPORT=${REFIT_TRANSPORT:-null}
case "$REFIT_TRANSPORT" in
  null)
    if [[ "$SEGMENT_SIZE" -ne 2 ]]; then
      echo "[ERROR] Legacy refit requires SCHEDULER_SEGMENT_SIZE=2"
      exit 2
    fi
    TRANSPORT_TAG=legacy
    TRANSPORT_OVERRIDES=("policy.generation.refit_transport=null")
    ;;
  nccl_reshard)
    if [[ "$SEGMENT_SIZE" -ne 1 ]]; then
      echo "[ERROR] NCCL-Reshard requires SCHEDULER_SEGMENT_SIZE=1"
      exit 2
    fi
    TRANSPORT_TAG=nccl-reshard
    TRANSPORT_OVERRIDES=(
      "policy.generation.refit_transport=nccl_reshard"
      "cluster.segment_size=1"
      "policy.generation.colocated.enabled=false"
      "policy.generation.colocated.resources.num_nodes=1"
      "policy.generation.colocated.resources.gpus_per_node=$GPUS_PER_NODE"
      "policy.megatron_cfg.expert_model_parallel_size=8"
    )
    ;;
  *)
    echo "[ERROR] REFIT_TRANSPORT must be null or nccl_reshard, got '$REFIT_TRANSPORT'"
    exit 2
    ;;
esac

EXP_DIR="$SCRIPT_DIR/${EXP_NAME}-${TRANSPORT_TAG}"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
WANDB_PROJECT_OVERRIDE=${WANDB_PROJECT_OVERRIDE:-sna-bf16-nvfp4-rollout}
WANDB_NAME_OVERRIDE=${WANDB_NAME_OVERRIDE:-${EXP_NAME}-${TRANSPORT_TAG}}
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

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    "$@" \
    grpo.max_num_steps=$MAX_STEPS \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.segment_size=$SEGMENT_SIZE \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project="$WANDB_PROJECT_OVERRIDE" \
    logger.wandb.name="$WANDB_NAME_OVERRIDE" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=false \
    "${TRANSPORT_OVERRIDES[@]}" \
    2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

assert_grep "VllmQuantInternalWorkerExtension" "$RUN_LOG" \
  "Real ModelOpt vLLM worker extension was not selected"
assert_grep "Detected ModelOpt NVFP4 checkpoint" "$RUN_LOG" \
  "vLLM did not detect a real ModelOpt NVFP4 checkpoint"
assert_grep "quantization=nemo_modelopt_w4a16_nvfp4|quant_algo[=: '\"]+W4A16_NVFP4" "$RUN_LOG" \
  "Expected ModelOpt W4A16 NVFP4 quantization method was not detected"
assert_grep "MegatronPolicyWorker" "$RUN_LOG" \
  "Plain Megatron policy worker was not selected"

assert_not_grep "MegatronQuantPolicyWorker|FakeQuantWorker|VLLM_QUANT_CFG" "$RUN_LOG" \
  "Rollout-only smoke selected a QARL or fake-quant path"
assert_not_grep "layerwise reload is incomplete|BF16 NVFP4 refit is incomplete|reload is missing" "$RUN_LOG" \
  "ModelOpt reload was incomplete"
assert_not_grep "IPCWeightManifestError|manifest (mismatch|incomplete|rejected|error)|mixed BF16 and ModelOpt real-quant manifest" "$RUN_LOG" \
  "Refit manifest validation failed"
assert_not_grep "agreement mismatch|Serialized refit plan agreement does not match|No refit plan agreement" "$RUN_LOG" \
  "NCCL refit plan agreement failed"
assert_not_grep "(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)" "$RUN_LOG" \
  "Run log contains a NaN or infinity"
assert_not_grep "Traceback \(most recent call last\)|Policy generation refit failed|ModelOpt real-quant.*failed|Exception during (collective_rpc|nccl_reshard_refit)|Error: Worker failed" "$RUN_LOG" \
  "Weight refit raised an exception"

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) == 2' \
    'len(data["timing/train/prepare_for_generation/transfer_and_update_weights"]) == 2'

assert_not_grep "(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)" "$JSON_METRICS" \
  "Metrics contain a NaN or infinity"

echo "[PASS] BF16 to NVFP4 W4A16 $TRANSPORT_TAG smoke completed two refits and two GRPO steps"
