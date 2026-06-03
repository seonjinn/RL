NEMORL=/scratch/fsw/portfolios/llmservice/users/smohsenitahe/NemoRL/super-v3-omni-vlm20/nemo-rl
set -a
source /scratch/fsw/portfolios/llmservice/users/smohsenitahe/.env
set +a
NUM_NODES=16
JOB_NAME=grpo-super-v3-omni-vllm20-super-16n-steps100
SEED=$(echo -n train:${JOB_NAME} | openssl dgst -md5 -binary | od -An -tu4 -N4 | xargs)
MODEL_NAME="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/repos_super_eval/artifacts/hf_conversions/sft_super_final_ckpt_conv3d_radiov4_1377_newcontainer_0422_clonecache_fix/iter_0058764/mcore_to_hf"
export CONTAINER=/scratch/fsw/portfolios/llmservice/users/smohsenitahe/sqsh/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh
# export NRL_FORCE_REBUILD_VENVS=false
export COMMAND="\
uv run examples/run_vlm_grpo.py --config examples/configs/vlm_grpo_super_blend_random.yaml \
cluster.num_nodes=$NUM_NODES \
policy.model_name=${MODEL_NAME} \
grpo.seed=$SEED \
grpo.max_num_steps=100 \
checkpointing.checkpoint_dir='results/${JOB_NAME}' \
checkpointing.checkpoint_must_save_by='00:03:45:00' \
checkpointing.save_period=20 \
logger.wandb_enabled=true \
logger.wandb.name='${JOB_NAME}' \
policy.tokenizer.chat_template=${MODEL_NAME}/chat_template.jinja \
policy.max_total_sequence_length=16384 \
policy.sequence_packing.train_mb_tokens=16384 \
policy.sequence_packing.logprob_mb_tokens=16384 \
policy.megatron_cfg.context_parallel_size=2 \
policy.megatron_cfg.moe_shared_expert_overlap=false \
policy.generation.vllm_cfg.gpu_memory_utilization=0.55 \
+policy.generation.vllm_cfg.logprobs_mode=raw_logprobs \
+policy.generation.vllm_kwargs.disable_custom_all_reduce=true \
+policy.megatron_cfg.scheduler.override_opt_param_scheduler=true \
data.train.data_path='/scratch/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/random_blend_v6.jsonl'"
export NCCL_DEBUG=INFO
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NEMO_RL_LOG_GPU_MEMORY=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NRL_IGNORE_VERSION_MISMATCH=true

GPUS_PER_NODE=8 \
MOUNTS="/lustre:/lustre,/scratch:/scratch,${NEMORL}:/opt/nemo-rl" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=llmservice_fm_vision \
    --job-name=nemo-rl-${JOB_NAME} \
    --partition=batch \
    --dependency=singleton \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub
