#!/bin/bash

set -euo pipefail

model=${1:?Usage: submit_20step_ab.sh qwen30|qwen235|super|nano baseline|hybridep}
arm=${2:?Usage: submit_20step_ab.sh qwen30|qwen235|super|nano baseline|hybridep}
project_root=$(git rev-parse --show-toplevel)

case "$model" in
  qwen30)
    config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
    nodes=4
    model_id=qwen3-30ba3b-4n8g
    ;;
  qwen235)
    config=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml
    nodes=16
    model_id=qwen3-235b-16n8g
    ;;
  super)
    config=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n8g.yaml
    nodes=32
    model_id=nemotron3-super-120ba12b-32n8g
    ;;
  nano)
    config=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml
    nodes=2
    model_id=nemotron3-nano-30ba3b-2n8g-pack-cp
    ;;
  *)
    printf 'Unknown model: %s\n' "$model" >&2
    exit 2
    ;;
esac

case "$arm" in
  baseline|hybridep) ;;
  *)
    printf 'Unknown arm: %s\n' "$arm" >&2
    exit 2
    ;;
esac

: "${ACCOUNT:?ACCOUNT is required}"
: "${PARTITION:?PARTITION is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_RL_COMMIT:?EXPECTED_RL_COMMIT is required}"

expected_rl_commit=$EXPECTED_RL_COMMIT
cluster_tag=${CLUSTER_TAG:-h100}
wandb_project=${WANDB_PROJECT:-sna-hybridep-h100}
segment=${SEGMENT:-}

if [[ ! -f "$CONTAINER" ]]; then
  if [[ -z "${DEPENDENCY_JOB_ID:-}" || ! "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]]; then
    printf 'Container does not exist and no valid staging dependency was provided: %s\n' "$CONTAINER" >&2
    exit 2
  fi
  test -d "$(dirname "$CONTAINER")"
fi
test -d "$HF_HOME"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(git rev-parse HEAD)" = "$expected_rl_commit"
git merge-base --is-ancestor 9b25508a3340bffdd8e3a2245ada72279fbc15d6 HEAD

bridge_dir=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mcore_dir=$bridge_dir/3rdparty/Megatron-LM
rl_commit=$(git rev-parse HEAD)
bridge_commit=$(git -C "$bridge_dir" rev-parse HEAD)
mcore_commit=$(git -C "$mcore_dir" rev-parse HEAD)

grep -Fq 'DeepEP.git@17cfb817bccec3a9c247013360cc550c2bac441e' pyproject.toml
if [[ "$model" == nano ]]; then
  grep -Fq 'expert_model_parallel_size: 8' "$config"
else
  grep -Fq 'moe_flex_dispatcher_backend: hybridep' "$config"
  grep -Fq 'NVLINK_DOMAIN_SIZE: "8"' "$config"
fi

run_name=${RUN_NAME:-"${cluster_tag}-${model_id}-${arm}-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$RUN_ROOT"

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/cache"}
export NCCL_NVLS_ENABLE=0
export NRL_FORCE_REBUILD_VENVS=true
export NRL_NODE_LOCAL_UV_CACHE_DIR="/raid/scratch/nemo-rl-uv-cache-${USER}-${run_name}"
export NEMO_RL_VENV_DIR="/raid/scratch/nemo-rl-venvs-${USER}-${run_name}"
export UV_CACHE_DIR="${NRL_NODE_LOCAL_UV_CACHE_DIR}/driver"
export UV_PROJECT_ENVIRONMENT="/raid/scratch/nemo-rl-driver-venv-${USER}-${run_name}"

unset CUDNN_HOME CUDNN_PATH
mcore_venv_name=nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
mcore_cudnn_lib="${NEMO_RL_VENV_DIR}/${mcore_venv_name}/lib/python3.13/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${mcore_cudnn_lib}:${LD_LIBRARY_PATH:-}"

if [[ "$arm" == hybridep ]]; then
  export HYBRID_EP_MULTINODE=1
  export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
  export NVLINK_DOMAIN_SIZE=8
  export USE_MNNVL=0
fi

# shellcheck disable=SC2016
printf -v setup_command '%q ' bash -lc \
  'set -euo pipefail
   test "$(nvidia-smi --query-gpu=name --format=csv,noheader | sed "/^$/d" | wc -l)" = 8
   test -d /raid/scratch
   test "$(git -C '"$project_root"' rev-parse HEAD)" = '"$expected_rl_commit"''
export SETUP_COMMAND=$setup_command

driver_args=(
  uv run --no-sync examples/run_grpo.py
  --config "$config"
  grpo.max_num_steps=20
  checkpointing.enabled=false
  "logger.log_dir=${RUN_ROOT}/training"
  logger.wandb_enabled=true
  "logger.wandb.project=${wandb_project}"
  "logger.wandb.name=${run_name}"
)

if [[ "$arm" == baseline ]]; then
  driver_args+=(
    '~policy.megatron_cfg.moe_token_dispatcher_type'
    '~policy.megatron_cfg.moe_flex_dispatcher_backend'
    '~policy.megatron_cfg.moe_hybridep_num_sms'
    '~policy.megatron_cfg.env_vars.NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN'
    '~policy.megatron_cfg.env_vars.NUM_OF_TOKENS_PER_CHUNK_COMBINE_API'
    '~policy.megatron_cfg.env_vars.NVLINK_DOMAIN_SIZE'
    '~policy.megatron_cfg.env_vars.USE_MNNVL'
  )
  if grep -Fq 'moe_hybridep_prepad_packed_inputs:' "$config"; then
    driver_args+=('~policy.megatron_cfg.moe_hybridep_prepad_packed_inputs')
  fi
fi
if [[ "$model" == nano && "$arm" == hybridep ]]; then
  driver_args+=(
    policy.megatron_cfg.moe_token_dispatcher_type=flex
    +policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
    +policy.megatron_cfg.moe_hybridep_num_sms=32
    policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=false
  )
fi
printf -v driver_command '%q ' "${driver_args[@]}"

mcore_dataset_dir=$project_root/$mcore_dir/megatron/core/datasets
# shellcheck disable=SC2016
printf -v helper_build \
  'suffix=$(python -c %q); make -B -C %q "LIBEXT=${suffix}"; test -s %q/helpers_cpp${suffix}' \
  'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' \
  "$mcore_dataset_dir" \
  "$mcore_dataset_dir"
export COMMAND="uv sync --locked --no-install-project && ${helper_build} && ${driver_command}"
export MOUNTS="$project_root:$project_root,/lustre:/lustre,/raid/scratch:/raid/scratch"
export BASE_LOG_DIR="$RUN_ROOT/ray"
export GPUS_PER_NODE=8
export PYTHONPATH="$project_root:$project_root/$bridge_dir/src:$project_root/$mcore_dir${PYTHONPATH:+:$PYTHONPATH}"

{
  printf 'run_name=%q\n' "$run_name"
  printf 'arm=%q\n' "$arm"
  printf 'config=%q\n' "$config"
  printf 'cluster_tag=%q\n' "$cluster_tag"
  printf 'account=%q\n' "$ACCOUNT"
  printf 'container=%q\n' "$CONTAINER"
  printf 'rl_commit=%q\n' "$rl_commit"
  printf 'bridge_commit=%q\n' "$bridge_commit"
  printf 'mcore_commit=%q\n' "$mcore_commit"
  printf 'deep_ep_commit=17cfb817bccec3a9c247013360cc550c2bac441e\n'
  printf 'nodes=%q\n' "$nodes"
  printf 'gpus_per_node=8\n'
  printf 'max_steps=20\n'
} >"$RUN_ROOT/submission.env"

sbatch_args=(
  --export=ALL
  --nodes="$nodes"
  --gpus-per-node=8
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --time=04:00:00
  --job-name="${ACCOUNT}.${run_name}"
  --output="$RUN_ROOT/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"45","reason":"other","description":"NeMo-RL model initialization before HybridEP training"}}'
)
if [[ -n "$segment" ]]; then
  sbatch_args+=(--segment="$segment")
fi
if [[ -n "${DEPENDENCY_JOB_ID:-}" ]]; then
  if [[ ! "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]]; then
    printf 'DEPENDENCY_JOB_ID must be numeric: %s\n' "$DEPENDENCY_JOB_ID" >&2
    exit 2
  fi
  sbatch_args+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
fi
sbatch_args+=(ray.sub)

sbatch --test-only "${sbatch_args[@]}"
if [[ "${TEST_ONLY:-0}" == 1 ]]; then
  exit 0
fi

job_id=$(sbatch --parsable "${sbatch_args[@]}")
printf 'job_id=%q\n' "$job_id" >>"$RUN_ROOT/submission.env"
printf '%s\n' "$job_id"
