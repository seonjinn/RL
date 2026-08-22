#!/usr/bin/env bash

#SBATCH --partition=batch
#SBATCH --account=nemotron_n3_post
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --segment=4

set -euo pipefail

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-/home/sna/RL-sync-refit-phase-profile}
PROFILE_COMMIT=${PROFILE_COMMIT:-8796aac0e6119847147ca00719b458060be20fa1}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${ROOT}/experiments/mxfp8-sync-refit-phase-profile/${RUN_GROUP}}
WANDB_HOME=${WANDB_HOME:-${ROOT}/wandb_netrc_home}
ARM_ROOT=${RUN_ROOT}/${SLURM_JOB_ID}

ACTUAL_COMMIT=$(git -C "${REPO}" rev-parse HEAD)
if ! git -C "${REPO}" merge-base --is-ancestor "${PROFILE_COMMIT}" "${ACTUAL_COMMIT}"; then
  echo "${ACTUAL_COMMIT} does not include profile commit ${PROFILE_COMMIT}" >&2
  exit 2
fi
git -C "${REPO}" diff --quiet
git -C "${REPO}" diff --cached --quiet
if git -C "${REPO}" submodule status --recursive | grep -q '^[+-U]'; then
  echo "All pinned submodules must be initialized at their pinned revisions" >&2
  exit 2
fi
test -f "${CONTAINER}"
test -f "${WANDB_HOME}/.netrc"
mkdir -p "${ARM_ROOT}/logs"

cat >"${ARM_ROOT}/provenance.env" <<EOF
model=Qwen3-30B-A3B
source_commit=${ACTUAL_COMMIT}
container=${CONTAINER}
hardware=GB200
execution=sync_colocated_cuda_ipc
nodes=4
gpus_per_node=4
cuda_graph=true
profile_step=2
max_steps=3
rollout_precision=mxfp8
quantization_scope=routed_expert_fc1_fc2_only
moe_backend=flashinfer_trtllm
refit_prequantize=true
refit_buffer_size_gb=4
refit_persistent_ipc_buffers=true
refit_slim_offload_after=true
refit_batched_moe_shuffle=true
refit_cache_loader_routes=true
EOF

export GPUS_PER_NODE=4
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export MOUNTS=/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,${WANDB_HOME}/.netrc:/root/.netrc
export BASE_LOG_DIR=${ARM_ROOT}
export RAY_LOG_SYNC_FREQUENCY=30
export SETUP_COMMAND='mkdir -p /raid/scratch/sna/nemo-rl-worker-cache /raid/scratch/sna/uv-cache /raid/scratch/sna/uv-python'
export COMMAND
COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HOME=/root
export HF_HOME=${ROOT}/hf_home
export HF_DATASETS_CACHE=\${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu
export NCCL_NVLS_ENABLE=0
export NRL_MXFP8_BATCHED_SHUFFLE=1
export NRL_MXFP8_SHUFFLE_VERIFY=0
export NRL_NSYS_WORKER_PATTERNS=megatron_policy_worker,vllm_generation_worker
export NRL_NSYS_PROFILE_STEP_RANGE=2:3
export NRL_NSYS_EXTRA_OPTIONS='{"cpuctxsw":"none"}'
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=/raid/scratch/sna/nemo-rl-worker-cache/sync-refit-profile-${PROFILE_COMMIT}
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=/raid/scratch/sna/uv-cache
export UV_PYTHON_INSTALL_DIR=/raid/scratch/sna/uv-python
export UV_LOCK_TIMEOUT=7200
unset UV_PROJECT_ENVIRONMENT WANDB_API_KEY
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=4 \
  cluster.segment_size=4 \
  policy.precision=bfloat16 \
  grpo.async_grpo.enabled=false \
  data_plane.enabled=false \
  policy.generation.colocated.enabled=true \
  policy.generation.colocated.resources.num_nodes=4 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  policy.generation.refit_transport=null \
  policy.generation.real_quant_export_cpu_offload=false \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.generation.vllm_cfg.async_engine=false \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
  policy.generation.vllm_cfg.use_tqdm=false \
  policy.generation.vllm_cfg.precision=fp8 \
  ++policy.generation.vllm_cfg.is_mx=true \
  ++policy.generation.vllm_cfg.refit_prequantize=true \
  ++policy.generation.vllm_cfg.refit_cache_loader_routes=true \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.train_global_batch_size=16 \
  ++policy.refit_buffer_size_gb=4 \
  ++policy.refit_persistent_ipc_buffers=true \
  ++policy.megatron_cfg.refit_slim_offload_after=true \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  ++grpo.skip_reference_policy_logprobs_calculation=false \
  grpo.max_num_steps=3 \
  grpo.seed=42 \
  grpo.val_period=0 \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  data.train.dataset_name=DAPOMath17K \
  grpo.num_prompts_per_step=8 \
  grpo.num_generations_per_prompt=2 \
  policy.max_total_sequence_length=4096 \
  policy.logprob_batch_size=1 \
  policy.megatron_cfg.defer_fp32_logits=true \
  policy.logprob_chunk_size=128 \
  policy.megatron_cfg.use_fused_linear_logprobs=true \
  policy.megatron_cfg.fused_linear_logprobs_chunk_size=128 \
  policy.megatron_cfg.activation_checkpointing=true \
  policy.sequence_packing.enabled=false \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
  loss_fn.reference_policy_kl_penalty=0.01 \
  logger.log_dir=${ARM_ROOT}/logs \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=false \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=sna-mxfp8-sync-refit-phase-profile \
  logger.wandb.name=qwen30-sync-refit-profile-${RUN_GROUP}
/opt/nemo_rl_venv/bin/python tests/json_dump_tb_logs.py ${ARM_ROOT}/logs \
  --output_path ${ARM_ROOT}/metrics.json || true
EOF
)

exec bash "${REPO}/ray.sub"
