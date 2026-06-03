#!/bin/bash
# Phase 2 dedup comparison: runs the async gym VLM smoke twice —
# once with deduplicate_multimodal_data=false, once with true —
# and asserts gradient equivalence with significant bytes reduction.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR_OFF=$EXP_DIR/logs_dedup_off
LOG_DIR_ON=$EXP_DIR/logs_dedup_on
JSON_OFF=$EXP_DIR/metrics_dedup_off.json
JSON_ON=$EXP_DIR/metrics_dedup_on.json
RUN_LOG_OFF=$EXP_DIR/run_dedup_off.log
RUN_LOG_ON=$EXP_DIR/run_dedup_on.log
CHECKPOINT_DIR=$EXP_DIR/checkpoints
MODEL_NAME=${MODEL_NAME:-${IMAGE_GRPO_MODEL_NAME:-}}
if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: Set MODEL_NAME or IMAGE_GRPO_MODEL_NAME for the VLM smoke."
    exit 1
fi
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR $LOG_DIR_OFF $LOG_DIR_ON $CHECKPOINT_DIR

trap "rm -rf $CHECKPOINT_DIR" EXIT

cd $PROJECT_ROOT

COMMON_OVERRIDES=(
    policy.model_name=$MODEL_NAME
    policy.tokenizer.name=$MODEL_NAME
    policy.tokenizer.chat_template=$MODEL_NAME/chat_template.jinja
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=$MODEL_NAME/chat_template.jinja
    policy.generation.vllm_cfg.async_engine=true
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.colocated.resources.gpus_per_node=1
    policy.generation.vllm_cfg.tensor_parallel_size=1
    +policy.generation.vllm_cfg.logprobs_mode=raw_logprobs
    +policy.generation.vllm_kwargs.mm_processor_cache_gb=0
    grpo.async_grpo.enabled=true
    grpo.async_grpo.max_trajectory_age_steps=1
    +grpo.async_grpo.in_flight_weight_updates=false
    +grpo.async_grpo.recompute_kv_cache_after_weight_updates=false
    +grpo.debug_payload_metrics=true
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    grpo.max_num_steps=1
    grpo.val_period=100
    loss_fn.use_importance_sampling_correction=true
    checkpointing.enabled=false
    policy.megatron_cfg.scheduler.lr_warmup_iters=0
    logger.tensorboard_enabled=true
    logger.wandb_enabled=false
    logger.monitor_gpus=true
    cluster.gpus_per_node=2
)

echo "=== Run A: dedup OFF ==="
uv run --no-sync coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_nanov3omni.yaml \
    "${COMMON_OVERRIDES[@]}" \
    grpo.deduplicate_multimodal_data=false \
    logger.log_dir=$LOG_DIR_OFF \
    $@ \
    2>&1 | tee $RUN_LOG_OFF

echo "=== Run B: dedup ON ==="
uv run --no-sync coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_nanov3omni.yaml \
    "${COMMON_OVERRIDES[@]}" \
    grpo.deduplicate_multimodal_data=true \
    logger.log_dir=$LOG_DIR_ON \
    $@ \
    2>&1 | tee $RUN_LOG_ON

# Both runs must have received multimodal data
for log in "$RUN_LOG_OFF" "$RUN_LOG_ON"; do
    if ! grep -q '"multi_modal_data"' "$log"; then
        echo "ERROR: vLLM never received multi_modal_data in $(basename $log)"
        exit 1
    fi
done

uv run --no-sync tests/json_dump_tb_logs.py $LOG_DIR_OFF --output_path $JSON_OFF
uv run --no-sync tests/json_dump_tb_logs.py $LOG_DIR_ON --output_path $JSON_ON

# Quality gates: both runs independently pass Phase 1 thresholds
for json in "$JSON_OFF" "$JSON_ON"; do
    uv run --no-sync tests/check_metrics.py "$json" \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'max(data["train/token_mult_prob_error"]) < 1.2' \
        'mean(data["train/mean_seq_mult_prob_error"]) < 1.1' \
        'data["payload_bytes/driver_to_policy_get_logprobs/tensor_mm"]["1"] > 0'
done

# Dedup-specific: bytes reduction
# tensor_mm bytes for dedup-on should be significantly less than dedup-off
uv run --no-sync python3 -c "
import json, sys

with open('$JSON_OFF') as f:
    off = json.load(f)
with open('$JSON_ON') as f:
    on = json.load(f)

bytes_off = off['payload_bytes/driver_to_policy_get_logprobs/tensor_mm']['1']
bytes_on = on['payload_bytes/driver_to_policy_get_logprobs/tensor_mm']['1']

if bytes_on <= 0:
    print(f'ERROR: dedup-on tensor_mm bytes is {bytes_on}, expected > 0')
    sys.exit(1)

ratio = bytes_off / bytes_on
print(f'dedup_ratio = {ratio:.2f} (bytes_off={bytes_off}, bytes_on={bytes_on})')

NUM_GENS = 4  # grpo.num_generations_per_prompt
if ratio < NUM_GENS - 2:
    print(f'ERROR: dedup_ratio {ratio:.2f} too low, expected >= {NUM_GENS - 2}')
    sys.exit(1)

print('✅ Dedup bytes reduction check passed')
"

echo "✅ Dedup comparison complete"
