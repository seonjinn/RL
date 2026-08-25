#!/bin/bash

set -euo pipefail

readonly EXPECTED_MEGATRON_LM_COMMIT=377ad24cd05f41686fafe2e6747f47678b8581c8
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}
readonly PARTITION=${PARTITION:-batch}
readonly GPUS_PER_NODE=8
readonly MAX_STEPS=${MAX_STEPS:-3}

case_name=${1:-}
if [[ -z ${case_name} ]]; then
  echo "Usage: $0 CASE" >&2
  exit 2
fi

: "${CONTAINER:?Set CONTAINER to the staged immutable .sqsh path}"
: "${HF_HOME:?Set HF_HOME to the shared Hugging Face cache}"
: "${RESULTS_ROOT:?Set RESULTS_ROOT to the shared validation-results directory}"

project_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
megatron_lm_root="${project_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
source_commit=$(git -C "${project_root}" rev-parse HEAD)
megatron_lm_commit=$(git -C "${megatron_lm_root}" rev-parse HEAD)

if [[ ${megatron_lm_commit} != "${EXPECTED_MEGATRON_LM_COMMIT}" ]]; then
  echo "Expected Megatron-LM ${EXPECTED_MEGATRON_LM_COMMIT}, got ${megatron_lm_commit}" >&2
  exit 1
fi

config=
nodes=
run_suffix=
overrides=()

case "${case_name}" in
  qwen30_pp1_cp1)
    config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
    nodes=4
    run_suffix=qwen30-pp1-cp1
    overrides+=(
      policy.megatron_cfg.pipeline_model_parallel_size=1
      policy.megatron_cfg.context_parallel_size=1
      policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true
    )
    ;;
  qwen30_pp2_cp2)
    config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
    nodes=4
    run_suffix=qwen30-pp2-cp2
    overrides+=(
      policy.megatron_cfg.pipeline_model_parallel_size=2
      policy.megatron_cfg.context_parallel_size=2
      policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=false
    )
    ;;
  qwen235_pp8_cp2)
    config=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml
    nodes=16
    run_suffix=qwen235-pp8-cp2
    overrides+=(
      policy.megatron_cfg.pipeline_model_parallel_size=8
      policy.megatron_cfg.context_parallel_size=2
      policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=false
    )
    ;;
  super_pp1_cp1)
    config=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n8g.yaml
    nodes=32
    run_suffix=super-pp1-cp1
    overrides+=(
      policy.megatron_cfg.pipeline_model_parallel_size=1
      policy.megatron_cfg.context_parallel_size=1
      policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true
      policy.megatron_cfg.moe_router_enable_expert_bias=true
    )
    ;;
  *)
    echo "Unknown validation case: ${case_name}" >&2
    exit 2
    ;;
esac

source_short=${source_commit:0:9}
megatron_lm_short=${megatron_lm_commit:0:9}
run_name="hybridep-6114-${run_suffix}-${source_short}-${megatron_lm_short}"
case_dir="${RESULTS_ROOT}/${run_name}"
node_local_root="/raid/scratch/${USER}/nemo-rl-hybridep-6114-${source_short}-${megatron_lm_short}"
main_venv="${node_local_root}/main-venv"
worker_venvs="${node_local_root}/worker-venvs"
uv_cache="${node_local_root}/uv-cache"

mkdir -p "${case_dir}/ray" "${case_dir}/metrics"

command_args=(
  uv run --locked examples/run_grpo.py
  --config "${config}"
  "grpo.max_num_steps=${MAX_STEPS}"
  checkpointing.enabled=false
  "logger.log_dir=${case_dir}/metrics"
  logger.wandb_enabled=true
  logger.tensorboard_enabled=true
  logger.monitor_gpus=true
  logger.wandb.project=nemo-rl-hybridep-validation
  "logger.wandb.name=${run_name}"
  "${overrides[@]}"
)
printf -v driver_command '%q ' "${command_args[@]}"

setup_command="mkdir -p ${main_venv} ${worker_venvs} ${uv_cache}"
mounts="${project_root}:${project_root},${RESULTS_ROOT}:${RESULTS_ROOT},${HF_HOME}:${HF_HOME},/raid/scratch:/raid/scratch"
job_name="coreai_dlalgo_nemorl:${run_name}"

sbatch_args=(
  --nodes="${nodes}"
  --account="${ACCOUNT}"
  --job-name="${job_name}"
  --partition="${PARTITION}"
  --time=04:00:00
  --gpus-per-node="${GPUS_PER_NODE}"
  --output="${case_dir}/slurm-%j.out"
  "${project_root}/ray.sub"
)

echo "case=${case_name}"
echo "source_commit=${source_commit}"
echo "megatron_lm_commit=${megatron_lm_commit}"
echo "container=${CONTAINER}"
echo "nodes=${nodes}"
echo "command=${driver_command}"
echo "environment=NRL_FORCE_REBUILD_VENVS=true NEMO_RL_VENV_DIR=${worker_venvs} UV_PROJECT_ENVIRONMENT=${main_venv} UV_CACHE_DIR_OVERRIDE=${uv_cache} HYBRID_EP_MULTINODE=1"
printf 'sbatch='
printf ' %q' sbatch "${sbatch_args[@]}"
printf '\n'

if [[ ${VALIDATION_DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

common_env=(
  "CONTAINER=${CONTAINER}"
  "MOUNTS=${mounts}"
  "COMMAND=${driver_command}"
  "SETUP_COMMAND=${setup_command}"
  "BASE_LOG_DIR=${case_dir}/ray"
  "GPUS_PER_NODE=${GPUS_PER_NODE}"
  "HF_HOME=${HF_HOME}"
  "HF_DATASETS_CACHE=${HF_HOME}/cache"
  "NEMO_RL_VENV_DIR=${worker_venvs}"
  "UV_PROJECT_ENVIRONMENT=${main_venv}"
  "UV_CACHE_DIR_OVERRIDE=${uv_cache}"
  NRL_FORCE_REBUILD_VENVS=true
  HYBRID_EP_MULTINODE=1
  TORCH_CUDA_ARCH_LIST=9.0
  NVTE_CUDA_ARCHS=90
  RDMA_CORE_HOME=/usr
  USE_NIXL=0
  MAX_JOBS=8
  NCCL_NVLS_ENABLE=0
)

env "${common_env[@]}" sbatch --test-only "${sbatch_args[@]}"
env "${common_env[@]}" sbatch --parsable "${sbatch_args[@]}"
