#!/usr/bin/env bash
set -euo pipefail

model=${1:?Usage: pr3294_refit_ablation_lyris.sh MODEL ARM [--test-only]}
arm=${2:?Usage: pr3294_refit_ablation_lyris.sh MODEL ARM [--test-only]}
mode=${3:-submit}

case "$model" in
  q30)
    nodes=4; segment=4; buffer_gib=4; recipe_prequant=true
    config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
    ;;
  q235)
    nodes=16; segment=16; buffer_gib=4; recipe_prequant=true
    config=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml
    ;;
  super)
    nodes=32; segment=8; buffer_gib=0.5; recipe_prequant=false
    config=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-mxfp8-rollout.yaml
    ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac

case "$arm" in
  control) prequant=false; persistent=false; slim=false; cache=false ;;
  prequant) prequant=true; persistent=false; slim=false; cache=false ;;
  persistent) prequant=$recipe_prequant; persistent=true; slim=false; cache=false ;;
  slim) prequant=$recipe_prequant; persistent=false; slim=true; cache=false ;;
  cache) prequant=$recipe_prequant; persistent=false; slim=false; cache=true ;;
  full) prequant=true; persistent=true; slim=true; cache=true ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac

: "${REPO_DIR:?Set the remote worktree path}"
: "${EXPECTED_SHA:?Set the immutable experiment commit}"
: "${CONTAINER:?Set an immutable nightly .sqsh path}"

account=coreai_dlalgo_llm
result_root=/lustre/fsw/coreai_dlalgo_llm/users/sna/results/pr3294-ablation-20261007
hf_home=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
run_name="pr3294-${model}-${arm}-${EXPECTED_SHA:0:8}"
result_dir="${result_root}/${run_name}"
scratch="/raid/scratch/${USER}/${run_name}"

test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_SHA"
test -f "$CONTAINER"
mkdir -p "${result_dir}/ray" "${result_dir}/logs"

export GPUS_PER_NODE=4
export CPUS_PER_WORKER=144
export CONTAINER
export MOUNTS="/home:/home,${REPO_DIR}:/opt/nemo-rl,/lustre:/lustre,/raid/scratch:/raid/scratch"
export BASE_LOG_DIR="${result_dir}/ray"
export RAY_LOG_SYNC_FREQUENCY=60
export SETUP_COMMAND="mkdir -p ${scratch}/venvs ${scratch}/uv ${scratch}/xdg ${scratch}/triton ${scratch}/torchinductor ${scratch}/vllm ${scratch}/tmp"
export COMMAND="set -euo pipefail
cd ${REPO_DIR}
test \"\$(git rev-parse HEAD)\" = ${EXPECTED_SHA}
export NEMO_RL_VENV_DIR=${scratch}/venvs
export UV_PROJECT_ENVIRONMENT=${scratch}/driver-venv
export UV_CACHE_DIR=${scratch}/uv
export TMPDIR=${scratch}/tmp
export XDG_CACHE_HOME=${scratch}/xdg
export TRITON_CACHE_DIR=${scratch}/triton
export TORCHINDUCTOR_CACHE_DIR=${scratch}/torchinductor
export VLLM_CACHE_ROOT=${scratch}/vllm
export HF_HOME=${hf_home}
export HF_HUB_CACHE=${hf_home}/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NRL_FORCE_REBUILD_VENVS=true
uv run --frozen examples/run_grpo.py --config ${config} \
  grpo.max_num_steps=20 \
  checkpointing.enabled=false \
  +policy.refit_buffer_size_gb=${buffer_gib} \
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
  --nodes="$nodes"
  --account="$account"
  --job-name="${account}-refit.${run_name}"
  --partition=gb200
  --time=04:00:00
  --exclusive
  --segment="$segment"
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
