#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=8
SEGMENT_SIZE=2
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB_API_KEY must be exported for this control run"
  exit 2
fi

CONFIG_PATH="$PROJECT_ROOT/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
WANDB_PROJECT_OVERRIDE=${WANDB_PROJECT_OVERRIDE:-sna-bf16-nvfp4-rollout}
WANDB_NAME_OVERRIDE=${WANDB_NAME_OVERRIDE:-${EXP_NAME}}
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
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  policy.generation.refit_transport=null \
  policy.generation.vllm_kwargs.revision=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  logger.log_dir="$LOG_DIR" \
  logger.wandb_enabled=True \
  logger.wandb.project="$WANDB_PROJECT_OVERRIDE" \
  logger.wandb.name="$WANDB_NAME_OVERRIDE" \
  logger.monitor_gpus=True \
  logger.tensorboard_enabled=True \
  checkpointing.enabled=false \
  2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

assert_grep "MegatronPolicyWorker" "$RUN_LOG" \
  "Plain Megatron policy worker was not selected"
assert_not_grep "VllmQuantInternalWorkerExtension|Detected ModelOpt NVFP4 checkpoint|nemo_modelopt_(nvfp4|w4a16_nvfp4)|W4A16_NVFP4" "$RUN_LOG" \
  "BF16 control selected a ModelOpt real-quant path"
assert_not_grep "Traceback \(most recent call last\)|Error: Worker failed" "$RUN_LOG" \
  "BF16 control raised an exception"
assert_not_grep "(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)" "$RUN_LOG" \
  "Run log contains a NaN or infinity"

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
  'len(data["train/loss"]) == 2' \
  'len(data["timing/train/prepare_for_generation/transfer_and_update_weights"]) == 2'

assert_not_grep "(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)" "$JSON_METRICS" \
  "Metrics contain a NaN or infinity"

echo "[PASS] Matched BF16 control completed two refits and two GRPO steps"
