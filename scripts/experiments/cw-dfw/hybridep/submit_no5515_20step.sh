#!/bin/bash

set -euo pipefail

model=${1:?Usage: submit_no5515_20step.sh qwen30|super}
project_root=$(git rev-parse --show-toplevel)

case "${model}" in
  qwen30)
    config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
    nodes=4
    model_id=qwen3-30ba3b-4n8g-no5515
    ;;
  super)
    config=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n8g.yaml
    nodes=32
    model_id=nemotron3-super-120ba12b-32n8g-no5515
    ;;
  *)
    printf 'Unknown model: %s\n' "${model}" >&2
    exit 2
    ;;
esac

: "${ACCOUNT:?ACCOUNT is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

test -f "${CONTAINER}"
test -d "${HF_HOME}"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"

bridge_dir=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mcore_dir=${bridge_dir}/3rdparty/Megatron-LM
rl_commit=$(git rev-parse HEAD)
bridge_commit=$(git -C "${bridge_dir}" rev-parse HEAD)
mcore_commit=$(git -C "${mcore_dir}" rev-parse HEAD)

git -C "${mcore_dir}" merge-base --is-ancestor \
  81770cb015eab05785ecd540ba929d1400a52f67 HEAD
git -C "${mcore_dir}" merge-base --is-ancestor \
  723db5a72790aefc02f5a0228e6607eef70c0533 HEAD
if grep -Fq 'use_dropless_hybridep = (' \
  "${mcore_dir}/megatron/core/transformer/moe/router.py"; then
  printf 'MCore unexpectedly contains the #5515 routing exclusion.\n' >&2
  exit 2
fi

grep -Fq 'DeepEP.git@17cfb817bccec3a9c247013360cc550c2bac441e' pyproject.toml
grep -Fq 'moe_hybridep_prepad_packed_inputs: true' "${config}"

run_name=${RUN_NAME:-"${model_id}-$(date +%Y%m%d-%H%M%S)"}
partition=${PARTITION:-batch}
mkdir -p "${RUN_ROOT}"

export HYBRID_EP_MULTINODE=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NVLINK_DOMAIN_SIZE=8
export USE_MNNVL=0
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/cache"}
export NRL_NODE_LOCAL_UV_CACHE_DIR="/raid/scratch/nemo-rl-uv-cache-${USER}-${SLURM_JOB_ID:-submit}-${model}"
export NEMO_RL_VENV_DIR="/raid/scratch/nemo-rl-venvs-${USER}-${SLURM_JOB_ID:-submit}-${model}"
export CONTAINER_ENV_VARS=HYBRID_EP_MULTINODE,NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN,NVLINK_DOMAIN_SIZE,USE_MNNVL,NRL_NODE_LOCAL_UV_CACHE_DIR,NEMO_RL_VENV_DIR

# shellcheck disable=SC2016
printf -v setup_command '%q ' bash -lc \
  'set -euo pipefail
   test "$(nvidia-smi --query-gpu=name --format=csv,noheader | sed "/^$/d" | wc -l)" = 8
   test "${HYBRID_EP_MULTINODE}" = 1
   test "${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN}" = 8
   python -c "import importlib.metadata as m; import torch; import hybrid_ep_cpp; print(\"DEEPEP_RUNTIME_VERSION\", m.version(\"deep_ep\"))"'
export SETUP_COMMAND=${setup_command}

driver_args=(
  uv run --no-sync examples/run_grpo.py
  --config "${config}"
  grpo.max_num_steps=20
  checkpointing.enabled=false
  policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=false
  "logger.log_dir=${RUN_ROOT}/training"
  logger.wandb_enabled=True
  logger.wandb.project=sna-hybridep-no5515-validation
  "logger.wandb.name=${run_name}"
  logger.tensorboard_enabled=True
)
printf -v driver_command '%q ' "${driver_args[@]}"

mcore_dataset_dir=${project_root}/${mcore_dir}/megatron/core/datasets
# shellcheck disable=SC2016
printf -v helper_build \
  'suffix=$(python -c %q); make -B -C %q "LIBEXT=${suffix}"; test -s %q/helpers_cpp${suffix}' \
  'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' \
  "${mcore_dataset_dir}" \
  "${mcore_dataset_dir}"
export COMMAND="${helper_build} && ${driver_command}"
export MOUNTS="${project_root}:${project_root},/lustre:/lustre,/raid/scratch:/raid/scratch"
export BASE_LOG_DIR="${RUN_ROOT}/ray"
export GPUS_PER_NODE=8
export PYTHONPATH="${project_root}:${project_root}/${bridge_dir}/src:${project_root}/${mcore_dir}${PYTHONPATH:+:${PYTHONPATH}}"

{
  printf 'run_name=%q\n' "${run_name}"
  printf 'config=%q\n' "${config}"
  printf 'rl_commit=%q\n' "${rl_commit}"
  printf 'bridge_commit=%q\n' "${bridge_commit}"
  printf 'mcore_commit=%q\n' "${mcore_commit}"
  printf 'mcore_pr_5008=present\n'
  printf 'mcore_pr_6114=present\n'
  printf 'mcore_pr_5515=absent\n'
  printf 'nemo_prepad=false\n'
  printf 'mcore_pad_uneven_dispatch_inputs=true\n'
  printf 'nodes=%q\n' "${nodes}"
  printf 'gpus_per_node=8\n'
  printf 'partition=%q\n' "${partition}"
  printf 'max_steps=20\n'
  printf 'container=%q\n' "${CONTAINER}"
} > "${RUN_ROOT}/submission.env"

sbatch_args=(
  --export=ALL
  --nodes="${nodes}"
  --gpus-per-node=8
  --segment=16
  --account="${ACCOUNT}"
  --partition="${partition}"
  --time=04:00:00
  --job-name="${ACCOUNT}.${run_name}"
  --output="${RUN_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"45","reason":"other","description":"NeMo-RL model initialization before HybridEP training"}}'
  ray.sub
)

sbatch --test-only "${sbatch_args[@]}"
if [[ "${TEST_ONLY:-0}" == 1 ]]; then
  exit 0
fi

job_id=$(sbatch --parsable "${sbatch_args[@]}")
printf 'job_id=%q\n' "${job_id}" >> "${RUN_ROOT}/submission.env"
printf '%s\n' "${job_id}"
