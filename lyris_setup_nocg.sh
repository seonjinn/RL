#!/bin/bash
set -euo pipefail

BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg-test
MEGATRON_BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna/Megatron-LM-Cudagraph-Test

export CUDA_HOME=/usr/local/cuda
export HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
export UV_CACHE_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/job_cache/uv
export NRL_FORCE_REBUILD_VENVS=true
export PYTHONPATH="${MEGATRON_BASE}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

cd "${BASE}"

echo "=== Regenerating uv.lock ==="
uv lock 2>&1 | tail -5

echo "=== Llama3.1-8B NO CG baseline (Lyris GB200) ==="
uv run examples/run_grpo.py \
  --config "${BASE}/examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-nocg.yaml" \
  grpo.max_num_steps=20 \
  logger.log_dir="${BASE}/experiments/lyris_llama_nocg/logs" \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=false \
  checkpointing.enabled=false \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=4

echo "Exit: $?"
