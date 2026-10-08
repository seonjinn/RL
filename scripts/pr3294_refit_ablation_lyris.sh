#!/usr/bin/env bash
set -euo pipefail

arm=${1:?Usage: pr3294_refit_ablation_lyris.sh ARM [--test-only]}
mode=${2:-submit}

case "$arm" in
  control) prequant=false; persistent=false; slim=false; cache=false ;;
  prequant) prequant=true; persistent=false; slim=false; cache=false ;;
  persistent) prequant=true; persistent=true; slim=false; cache=false ;;
  slim) prequant=true; persistent=false; slim=true; cache=false ;;
  cache) prequant=true; persistent=false; slim=false; cache=true ;;
  full) prequant=true; persistent=true; slim=true; cache=true ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac

: "${REPO_DIR:?Set the remote worktree path}"
: "${EXPECTED_SHA:?Set the immutable experiment commit}"
: "${CONTAINER:?Set an immutable nightly .sqsh path}"

account=coreai_dlalgo_llm
result_root=/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3294-ablation-20261007
hf_home=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
run_name="pr3294-${arm}-${EXPECTED_SHA:0:8}"
result_dir="${result_root}/${run_name}"
scratch="/raid/scratch/${USER}/${run_name}"
config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml

test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_SHA"
test -f "$CONTAINER"
mkdir -p "${result_dir}/ray" "${result_dir}/logs"

export GPUS_PER_NODE=4
export CPUS_PER_WORKER=144
export CONTAINER
export MOUNTS="${REPO_DIR}:${REPO_DIR},${REPO_DIR}:/opt/nemo-rl,/lustre:/lustre,/raid/scratch:/raid/scratch"
export BASE_LOG_DIR="${result_dir}/ray"
export RAY_LOG_SYNC_FREQUENCY=60
export SETUP_COMMAND="mkdir -p ${scratch}/venvs ${scratch}/uv ${scratch}/xdg ${scratch}/triton ${scratch}/torchinductor ${scratch}/vllm"
export COMMAND="set -euo pipefail
cd ${REPO_DIR}
test \"\$(git rev-parse HEAD)\" = ${EXPECTED_SHA}
export NEMO_RL_VENV_DIR=${scratch}/venvs
export UV_CACHE_DIR=${scratch}/uv
export XDG_CACHE_HOME=${scratch}/xdg
export TRITON_CACHE_DIR=${scratch}/triton
export TORCHINDUCTOR_CACHE_DIR=${scratch}/torchinductor
export VLLM_CACHE_ROOT=${scratch}/vllm
export HF_HOME=${hf_home}
export HF_HUB_CACHE=${hf_home}/hub
export NRL_FORCE_REBUILD_VENVS=true
uv run --locked examples/run_grpo.py --config ${config} \
  grpo.max_num_steps=20 \
  checkpointing.enabled=false \
  policy.refit_buffer_size_gb=4 \
  policy.refit_persistent_ipc_buffers=${persistent} \
  policy.megatron_cfg.refit_slim_offload_after=${slim} \
  policy.generation.vllm_cfg.refit_prequantize=${prequant} \
  policy.generation.vllm_cfg.refit_cache_loader_routes=${cache} \
  logger.log_dir=${result_dir}/logs \
  logger.wandb_enabled=true \
  logger.wandb.project=sna-pr3294-ablation \
  logger.wandb.name=${run_name} \
  logger.monitor_gpus=true \
  logger.tensorboard_enabled=false"

args=(
  --nodes=4
  --account="$account"
  --job-name="${account}-refit.${run_name}"
  --partition=gb200
  --time=04:00:00
  --exclusive
  --segment=4
  --output="${result_dir}/slurm-%j.out"
)

if [[ "$mode" == --test-only ]]; then
  sbatch --test-only "${args[@]}" "${REPO_DIR}/ray.sub"
elif [[ "$mode" == submit ]]; then
  sbatch "${args[@]}" "${REPO_DIR}/ray.sub"
else
  echo "Unknown mode: $mode" >&2
  exit 2
fi
