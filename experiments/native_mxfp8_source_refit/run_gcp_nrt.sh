#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  REPO=${REPO:-$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)}
  ACTION=${ACTION:-test-only}
  MAX_STEPS=${MAX_STEPS:-2}
  RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
  ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
  PARTITION=${PARTITION:-batch}
  WALLTIME=${WALLTIME:-04:00:00}
  WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
  CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh}
  RESULT_ROOT=${RESULT_ROOT:-${WORK_ROOT}/experiments/native-mxfp8-source-refit/gcp-b200}
  RUN_NAME=${RUN_NAME:-native-mxfp8-source-qwen30b-${MAX_STEPS}step-${RUN_SUFFIX}}
  EXPERIMENT_ROOT=${RESULT_ROOT}/results/${RUN_NAME}

  case "${ACTION}" in
    test-only) ACTION_ARG=--test-only ;;
    submit) ACTION_ARG= ;;
    *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
  esac

  git -C "${REPO}" pull --ff-only
  test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
  if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
    echo "All pinned submodules must be initialized before submission" >&2
    exit 2
  fi
  REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
  test -f "${CONTAINER}"
  mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/manifests"

  args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes=4
    --ntasks-per-node=1
    --gpus-per-node=8
    --exclusive
    --time="${WALLTIME}"
    --job-name="native-mxfp8-source-${MAX_STEPS}step-${RUN_SUFFIX}"
    --output="${RESULT_ROOT}/slurm/%x-%j.out"
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"native_mxfp8_refit","description":"venv setup and model initialization"}}'
    --export="ALL,REPO=${REPO},EXPECTED_REPO_SHA=${REPO_SHA},CONTAINER=${CONTAINER},MAX_STEPS=${MAX_STEPS},RUN_NAME=${RUN_NAME},EXPERIMENT_ROOT=${EXPERIMENT_ROOT},WORK_ROOT=${WORK_ROOT}"
  )
  if [[ -n "${ACTION_ARG}" ]]; then
    args+=("${ACTION_ARG}")
  fi

  output=$(sbatch "${args[@]}" "${BASH_SOURCE[0]}")
  job_id=$(sed -n 's/^Submitted batch job //p' <<<"${output}")
  manifest=${RESULT_ROOT}/manifests/submission-${RUN_SUFFIX}.tsv
  printf 'action\tjob_id\trepo_sha\trun_name\tcontainer\n' >"${manifest}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${ACTION}" "${job_id:-n/a}" "${REPO_SHA}" "${RUN_NAME}" "${CONTAINER}" \
    | tee -a "${manifest}"
  echo "manifest=${manifest}"
  exit 0
fi

: "${REPO:?REPO is required}"
: "${EXPECTED_REPO_SHA:?EXPECTED_REPO_SHA is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${WORK_ROOT:?WORK_ROOT is required}"

test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_SHA}"
test -f "${CONTAINER}"
test -f "${REPO}/ray.sub"
mkdir -p "${EXPERIMENT_ROOT}/logs"

HF_HOME=${HF_HOME:-${WORK_ROOT}/.cache/huggingface}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/native-mxfp8-source/${EXPECTED_REPO_SHA}}
SHARED_UV_CACHE=${SHARED_UV_CACHE:-${WORK_ROOT}/mopd_nano_fast/.cache/native-mxfp8-source/shared-vllm025/uv}
RAY_BOOTSTRAP_ARCHIVE=${RAY_BOOTSTRAP_ARCHIVE:-${WORK_ROOT}/mopd_nano_fast/.cache/nccl-reshard-pr3294/bootstrap/ray-2.56.1-py31313.tar.gz}
RAY_BOOTSTRAP_LOCAL_ROOT=${RAY_BOOTSTRAP_LOCAL_ROOT:-/tmp/nrl-ray-bootstrap-${SLURM_JOB_ID}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-native-mxfp8-source-refit}
WANDB_KEY_FILE=${WANDB_KEY_FILE:-${WORK_ROOT}/mopd_nano_fast/.cache/native-mxfp8-source/.wandb_key}

test -f "${RAY_BOOTSTRAP_ARCHIVE}"
mkdir -p "${HF_HOME}" "${SHARED_UV_CACHE}" "${CACHE_ROOT}/venvs"
if [[ ! -s "${WANDB_KEY_FILE}" && -f "${HOME}/.netrc" ]]; then
  mkdir -p "$(dirname "${WANDB_KEY_FILE}")"
  (umask 077; awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' \
    "${HOME}/.netrc" >"${WANDB_KEY_FILE}")
fi

export SETUP_COMMAND="
set -euo pipefail
if [[ ! -x '${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/ray' ]]; then
  rm -rf '${RAY_BOOTSTRAP_LOCAL_ROOT}' '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp'
  mkdir -p '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp'
  tar -xzf '${RAY_BOOTSTRAP_ARCHIVE}' \\
    -C '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp' --strip-components=1
  mv '${RAY_BOOTSTRAP_LOCAL_ROOT}.tmp' '${RAY_BOOTSTRAP_LOCAL_ROOT}'
fi
test -x '${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/ray'
test -x '${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/uv'
'${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/python' -c 'import ray, requests, urllib3'
"

MCORE_ACTOR_VENV=${CACHE_ROOT}/venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
MCORE_NVIDIA_ROOT=${MCORE_ACTOR_VENV}/lib/python3.13/site-packages/nvidia
MCORE_LD_LIBRARY_PATH=${MCORE_NVIDIA_ROOT}/cudnn/lib:${MCORE_NVIDIA_ROOT}/cu13/lib:${MCORE_NVIDIA_ROOT}/nccl/lib:${MCORE_NVIDIA_ROOT}/nvshmem/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo=${REPO}
repo_sha=${EXPECTED_REPO_SHA}
container=${CONTAINER}
max_steps=${MAX_STEPS}
total_nodes=4
trainer_nodes=2
generation_nodes=2
gpus_per_node=8
train_precision=native_mxfp8
generation_precision=mxfp8
refit_transport=nccl_reshard
cache_root=${CACHE_ROOT}
shared_uv_cache=${SHARED_UV_CACHE}
wandb_project=${WANDB_PROJECT}
wandb_name=${RUN_NAME}
EOF

export CONTAINER
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR=${EXPERIMENT_ROOT}
export PATH=${RAY_BOOTSTRAP_LOCAL_ROOT}/bin:${PATH}
export PYTHONPATH=${RAY_BOOTSTRAP_LOCAL_ROOT}/lib/python3.13/site-packages${PYTHONPATH:+:${PYTHONPATH}}
export RAY_CLI=${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/ray
export UV_CACHE_DIR_OVERRIDE=${SHARED_UV_CACHE}
export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}

export COMMAND="
set -euo pipefail
cd '${REPO}'
export PYTHONPATH='${REPO}:${RAY_BOOTSTRAP_LOCAL_ROOT}/lib/python3.13/site-packages'
export HF_HOME='${HF_HOME}'
export NEMO_RL_VENV_DIR='${CACHE_ROOT}/venvs'
export UV_CACHE_DIR='${SHARED_UV_CACHE}'
export UV_PROJECT_ENVIRONMENT='${CACHE_ROOT}/driver-venv'
export UV_PYTHON='${RAY_BOOTSTRAP_LOCAL_ROOT}/bin/python3.13'
export UV_LOCK_TIMEOUT=7200
export NRL_FORCE_REBUILD_VENVS=false
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
if [[ -s '${WANDB_KEY_FILE}' ]]; then
  export WANDB_API_KEY=\$(cat '${WANDB_KEY_FILE}')
fi
uv run --frozen examples/run_grpo.py \\
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml \\
  cluster.num_nodes=4 \\
  cluster.gpus_per_node=8 \\
  cluster.segment_size=1 \\
  policy.generation.colocated.enabled=false \\
  policy.generation.colocated.resources.num_nodes=2 \\
  policy.generation.colocated.resources.gpus_per_node=8 \\
  policy.generation.refit_transport=nccl_reshard \\
  policy.megatron_cfg.tensor_model_parallel_size=1 \\
  policy.megatron_cfg.pipeline_model_parallel_size=1 \\
  policy.megatron_cfg.expert_model_parallel_size=16 \\
  policy.megatron_cfg.expert_tensor_parallel_size=1 \\
  policy.megatron_cfg.fp8_cfg.enabled=true \\
  policy.megatron_cfg.fp8_cfg.fp8=e4m3 \\
  policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8 \\
  policy.megatron_cfg.fp8_cfg.fp8_param=true \\
  policy.generation.vllm_cfg.tensor_parallel_size=1 \\
  ++policy.generation.vllm_cfg.pipeline_parallel_size=1 \\
  ++policy.generation.vllm_cfg.expert_parallel_size=1 \\
  policy.generation.vllm_cfg.precision=fp8 \\
  policy.generation.vllm_cfg.is_mx=true \\
  grpo.num_prompts_per_step=64 \\
  grpo.num_generations_per_prompt=32 \\
  policy.train_global_batch_size=2048 \\
  policy.max_total_sequence_length=4096 \\
  loss_fn.force_on_policy_ratio=false \\
  loss_fn.use_importance_sampling_correction=true \\
  ++grpo.skip_reference_policy_logprobs_calculation=false \\
  grpo.max_num_steps='${MAX_STEPS}' \\
  grpo.val_at_start=false \\
  ++grpo.val_at_end=false \\
  grpo.val_period=1000000 \\
  checkpointing.enabled=false \\
  logger.log_dir='${EXPERIMENT_ROOT}/logs' \\
  logger.wandb_enabled=true \\
  logger.wandb.project='${WANDB_PROJECT}' \\
  logger.wandb.name='${RUN_NAME}' \\
  ++policy.megatron_cfg.env_vars.CUDNN_HOME='${MCORE_NVIDIA_ROOT}/cudnn' \\
  ++policy.megatron_cfg.env_vars.CUDNN_PATH='${MCORE_NVIDIA_ROOT}/cudnn' \\
  ++policy.megatron_cfg.env_vars.LD_LIBRARY_PATH='${MCORE_LD_LIBRARY_PATH}'
"

exec bash "${REPO}/ray.sub"
