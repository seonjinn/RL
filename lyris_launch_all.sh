#!/bin/bash
set -euo pipefail

REMOTE="sna-mfa@login-lyris"
BASE="/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-feature-cuda-graph-training"
CFG_DIR="$BASE/examples/configs/recipes/llm/performance"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-nightly-ultra.sqsh"
UV_CACHE="/lustre/fsw/coreai_dlalgo_llm/users/sna/job_cache/uv"
HF_HOME="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"

# Copy local-only recipe files that are not committed to the branch.
scp \
  /Users/sna/CudaGraph_PR/RL/examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-nocg.yaml \
  /Users/sna/CudaGraph_PR/RL/examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml \
  /Users/sna/CudaGraph_PR/RL/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml \
  /Users/sna/CudaGraph_PR/RL/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nocg.yaml \
  /Users/sna/CudaGraph_PR/RL/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router.yaml \
  "${REMOTE}:${CFG_DIR}/"

ssh "${REMOTE}" bash <<EOF
set -euo pipefail

BASE="${BASE}"
CFG_DIR="${CFG_DIR}"
CONTAINER="${CONTAINER}"
UV_CACHE="${UV_CACHE}"
HF_HOME="${HF_HOME}"

COMMON_ENV='export CUDA_HOME=/usr/local/cuda; export HF_HOME='"${HF_HOME}"'; export UV_CACHE_DIR='"${UV_CACHE}"'; export NRL_FORCE_REBUILD_VENVS=true; export PYTHONPATH='"${BASE}"'/3rdparty/Megatron-LM-workspace/Megatron-LM:\$PYTHONPATH; export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}; cd '"${BASE}"

mkdir -p \
  "${BASE}/experiments/lyris_llama_nocg/logs" \
  "${BASE}/experiments/lyris_llama_cg/logs" \
  "${BASE}/experiments/lyris_qwen8_nocg/logs" \
  "${BASE}/experiments/lyris_qwen8_cg/logs" \
  "${BASE}/experiments/lyris_qwen30_nocg" \
  "${BASE}/experiments/lyris_qwen30_cg"

scancel 1643613 1643614 2>/dev/null || true

sbatch --account=coreai_dlalgo_llm --partition=gb200 --nodes=1 --exclusive \
  --job-name=lyr-ll8-nocg \
  --output="${BASE}/experiments/lyris_llama_nocg/slurm-%j.out" \
  --wrap="srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --container-workdir=${BASE} bash -lc '${COMMON_ENV}; uv run examples/run_grpo.py --config ${CFG_DIR}/grpo-llama3.1-8b-instruct-1n4g-nocg.yaml grpo.max_num_steps=20 logger.log_dir=${BASE}/experiments/lyris_llama_nocg/logs logger.wandb_enabled=true logger.tensorboard_enabled=false checkpointing.enabled=false cluster.num_nodes=1 cluster.gpus_per_node=4'"

sbatch --account=coreai_dlalgo_llm --partition=gb200 --nodes=1 --exclusive \
  --job-name=lyr-ll8-cg \
  --output="${BASE}/experiments/lyris_llama_cg/slurm-%j.out" \
  --wrap="srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --container-workdir=${BASE} bash -lc '${COMMON_ENV}; uv run examples/run_grpo.py --config ${CFG_DIR}/grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w6.yaml grpo.max_num_steps=20 logger.log_dir=${BASE}/experiments/lyris_llama_cg/logs logger.wandb_enabled=true logger.tensorboard_enabled=false checkpointing.enabled=false cluster.num_nodes=1 cluster.gpus_per_node=4'"

sbatch --account=coreai_dlalgo_llm --partition=gb200 --nodes=1 --exclusive \
  --job-name=lyr-qw8-nocg \
  --output="${BASE}/experiments/lyris_qwen8_nocg/slurm-%j.out" \
  --wrap="srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --container-workdir=${BASE} bash -lc '${COMMON_ENV}; uv run examples/run_grpo.py --config ${CFG_DIR}/grpo-qwen3-8b-1n4g-nocg.yaml grpo.max_num_steps=20 logger.log_dir=${BASE}/experiments/lyris_qwen8_nocg/logs logger.wandb_enabled=true logger.tensorboard_enabled=false checkpointing.enabled=false cluster.num_nodes=1 cluster.gpus_per_node=4'"

sbatch --account=coreai_dlalgo_llm --partition=gb200 --nodes=1 --exclusive \
  --job-name=lyr-qw8-cg \
  --output="${BASE}/experiments/lyris_qwen8_cg/slurm-%j.out" \
  --wrap="srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --container-workdir=${BASE} bash -lc '${COMMON_ENV}; uv run examples/run_grpo.py --config ${CFG_DIR}/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml grpo.max_num_steps=20 logger.log_dir=${BASE}/experiments/lyris_qwen8_cg/logs logger.wandb_enabled=true logger.tensorboard_enabled=false checkpointing.enabled=false cluster.num_nodes=1 cluster.gpus_per_node=4'"

CONTAINER="${CONTAINER}" MOUNTS=/lustre:/lustre GPUS_PER_NODE=4 UV_CACHE_DIR_OVERRIDE="${UV_CACHE}" BASE_LOG_DIR="${BASE}/experiments/lyris_qwen30_nocg" \
COMMAND="uv run ./examples/run_grpo.py --config ${CFG_DIR}/grpo-qwen3-30ba3b-4n4g-nocg.yaml logger.log_dir=${BASE}/experiments/lyris_qwen30_nocg logger.wandb_enabled=true logger.tensorboard_enabled=false checkpointing.enabled=false" \
sbatch --account=coreai_dlalgo_llm --partition=gb200 --nodes=4 --job-name=lyr-qw30-nocg ray.sub

CONTAINER="${CONTAINER}" MOUNTS=/lustre:/lustre GPUS_PER_NODE=4 UV_CACHE_DIR_OVERRIDE="${UV_CACHE}" BASE_LOG_DIR="${BASE}/experiments/lyris_qwen30_cg" \
COMMAND="uv run ./examples/run_grpo.py --config ${CFG_DIR}/grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router.yaml logger.log_dir=${BASE}/experiments/lyris_qwen30_cg logger.wandb_enabled=true logger.tensorboard_enabled=false checkpointing.enabled=false" \
sbatch --account=coreai_dlalgo_llm --partition=gb200 --nodes=4 --job-name=lyr-qw30-cg ray.sub

squeue -u sna -o "%i %T %M %R %j" | egrep 'lyr-(ll8|qw8|qw30)|coreai_dlalgo_llm-cg' | sed -n '1,40p'
EOF
