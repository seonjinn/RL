#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
CHECKPOINT_DIR=$EXP_DIR/checkpoints
MODEL_NAME=${MODEL_NAME:-${IMAGE_GRPO_MODEL_NAME:-}}
if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: Set MODEL_NAME or IMAGE_GRPO_MODEL_NAME for the VLM smoke."
    exit 1
fi
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CHECKPOINT_DIR

trap "rm -rf $CHECKPOINT_DIR" EXIT

cd $PROJECT_ROOT

uv run --no-sync coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_nanov3omni.yaml \
    policy.model_name=$MODEL_NAME \
    policy.tokenizer.name=$MODEL_NAME \
    policy.tokenizer.chat_template=$MODEL_NAME/chat_template.jinja \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=$MODEL_NAME/chat_template.jinja \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    +policy.generation.vllm_cfg.logprobs_mode=raw_logprobs \
    +policy.generation.vllm_kwargs.mm_processor_cache_gb=0 \
    grpo.async_grpo.enabled=true \
    grpo.async_grpo.max_trajectory_age_steps=1 \
    +grpo.async_grpo.in_flight_weight_updates=false \
    +grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
    grpo.deduplicate_multimodal_data=false \
    +grpo.debug_payload_metrics=true \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=1 \
    grpo.val_period=100 \
    loss_fn.use_importance_sampling_correction=true \
    checkpointing.enabled=false \
    policy.megatron_cfg.scheduler.lr_warmup_iters=0 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    cluster.gpus_per_node=2 \
    $@ \
    2>&1 | tee $RUN_LOG

if ! grep -q '"multi_modal_data"' "$RUN_LOG"; then
    echo "ERROR: vLLM never received multi_modal_data; images may have been dropped."
    exit 1
fi

uv run --no-sync tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run --no-sync tests/check_metrics.py $JSON_METRICS \
    'median(data["train/token_mult_prob_error"]) < 1.1' \
    'max(data["train/token_mult_prob_error"]) < 1.2' \
    'mean(data["train/mean_seq_mult_prob_error"]) < 1.1' \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'data["payload_bytes/driver_to_policy_get_logprobs/tensor_mm"]["1"] > 0'
