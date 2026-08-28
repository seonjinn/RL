#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PROFILE=${1:-"${SCRIPT_DIR}/models/nemotron3-super-120ba12b-32n8g-no5515.env"}

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${PROFILE}"

unset CONDA_PREFIX VIRTUAL_ENV

SETUP_SMOKE_ONLY=${SETUP_SMOKE_ONLY:-0}
case "${SETUP_SMOKE_ONLY}" in
  0) ;;
  1)
    NUM_ACTOR_NODES=1
    TIME_LIMIT=00:20:00
    ;;
  *)
    printf 'SETUP_SMOKE_ONLY must be 0 or 1.\n' >&2
    exit 2
    ;;
esac

: "${ACCOUNT:?Set ACCOUNT after checking FairShare.}"
: "${CONTAINER:?Set CONTAINER to an immutable x86 NeMo-RL image under /lustre.}"
: "${DEEPEP_WHEEL:?Set DEEPEP_WHEEL to the expected DeepEP x86_64 wheel under /lustre.}"
: "${UV_GIT_CACHE_SEED:?Set UV_GIT_CACHE_SEED to the immutable offline Git seed under /lustre.}"
: "${HF_HOME:?Set HF_HOME to the staged model/cache root under /lustre.}"
: "${RUN_ROOT:?Set RUN_ROOT to a durable experiment directory under /lustre.}"
: "${MODEL_ID:?MODEL_ID is required.}"
: "${CONFIG_PATH:?CONFIG_PATH is required.}"
: "${NUM_ACTOR_NODES:?NUM_ACTOR_NODES is required.}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required.}"
: "${SEGMENT_SIZE:?SEGMENT_SIZE is required.}"
: "${MAX_STEPS:?MAX_STEPS is required.}"
: "${TIME_LIMIT:?TIME_LIMIT is required.}"
: "${EXPECTED_BRIDGE_COMMIT:?EXPECTED_BRIDGE_COMMIT is required.}"
: "${EXPECTED_MEGATRON_LM_COMMIT:?EXPECTED_MEGATRON_LM_COMMIT is required.}"
: "${EXPECTED_DEEPEP_COMMIT:?EXPECTED_DEEPEP_COMMIT is required.}"
: "${EXPECTED_DEEPEP_VERSION:?EXPECTED_DEEPEP_VERSION is required.}"
: "${EXPECTED_5515_STATE:?EXPECTED_5515_STATE must be present or absent.}"

PRETRAINED_CHECKPOINT_FORMAT=${PRETRAINED_CHECKPOINT_FORMAT:-megatron_lm}
if [[ -n "${PRETRAINED_CHECKPOINT_PATH:-}" ]]; then
  [[ "${PRETRAINED_CHECKPOINT_PATH}" == /lustre/* ]]
  [[ -d "${PRETRAINED_CHECKPOINT_PATH}" ]]
fi

for path in "${CONTAINER}" "${DEEPEP_WHEEL}" "${UV_GIT_CACHE_SEED}"; do
  [[ "${path}" == /lustre/* && -f "${path}" ]]
done
for path in "${HF_HOME}" "${RUN_ROOT}"; do
  [[ "${path}" == /lustre/* ]]
done
[[ "${GPUS_PER_NODE}" == 8 ]]
if [[ "${SETUP_SMOKE_ONLY}" == 1 ]]; then
  [[ "${NUM_ACTOR_NODES}" == 1 ]]
else
  [[ "${NUM_ACTOR_NODES}" == 32 ]]
fi

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to submit from a dirty checkout.\n' >&2
  git status --short >&2
  exit 2
fi

current_branch=$(git branch --show-current)
upstream_remote=$(git config --get "branch.${current_branch}.remote")
upstream_merge=$(git config --get "branch.${current_branch}.merge")
git fetch --no-recurse-submodules "${upstream_remote}" "${upstream_merge}"
git merge --ff-only FETCH_HEAD
git submodule sync --recursive
submodule_state=$(git submodule status --recursive)
if grep -q '^-' <<< "${submodule_state}"; then
  printf 'Initialize the pinned submodules before running this launcher.\n' >&2
  exit 2
fi
git submodule update --recursive --force --no-fetch

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Submodule update left a dirty checkout.\n' >&2
  exit 2
fi
if [[ -n "$(git rev-list '@{upstream}..HEAD')" ]]; then
  printf 'The validation commit has not been pushed.\n' >&2
  exit 2
fi

BRIDGE_DIR=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE_DIR=${BRIDGE_DIR}/3rdparty/Megatron-LM
RL_COMMIT=$(git rev-parse HEAD)
BRIDGE_COMMIT=$(git -C "${BRIDGE_DIR}" rev-parse HEAD)
MCORE_COMMIT=$(git -C "${MCORE_DIR}" rev-parse HEAD)

[[ "${BRIDGE_COMMIT}" == "${EXPECTED_BRIDGE_COMMIT}" ]]
[[ "${MCORE_COMMIT}" == "${EXPECTED_MEGATRON_LM_COMMIT}" ]]
git -C "${MCORE_DIR}" merge-base --is-ancestor \
  81770cb015eab05785ecd540ba929d1400a52f67 HEAD
grep -Fq 'routing_map = routing_map & (~padding_mask).unsqueeze(-1)' \
  "${MCORE_DIR}/megatron/core/transformer/moe/router.py"

case "${EXPECTED_5515_STATE}" in
  present)
    grep -Fq 'use_dropless_hybridep = (' \
      "${MCORE_DIR}/megatron/core/transformer/moe/router.py"
    grep -Fq 'routing_map = routing_map & valid_tokens' \
      "${MCORE_DIR}/megatron/core/transformer/moe/router.py"
    ;;
  absent)
    if grep -Fq 'use_dropless_hybridep = (' \
      "${MCORE_DIR}/megatron/core/transformer/moe/router.py"; then
      printf 'MCore unexpectedly contains the #5515 implementation.\n' >&2
      exit 2
    fi
    ;;
  *)
    printf 'EXPECTED_5515_STATE must be present or absent.\n' >&2
    exit 2
    ;;
esac

grep -Fq "DeepEP.git@${EXPECTED_DEEPEP_COMMIT}" pyproject.toml
grep -Fq "rev=${EXPECTED_DEEPEP_COMMIT}#${EXPECTED_DEEPEP_COMMIT}" uv.lock
grep -Fq 'expert_model_parallel_size: 32' "${CONFIG_PATH}"
grep -Fq 'moe_hybridep_prepad_packed_inputs: true' "${CONFIG_PATH}"
grep -Fq 'NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: "8"' "${CONFIG_PATH}"
grep -Fq 'NVLINK_DOMAIN_SIZE: "8"' "${CONFIG_PATH}"
grep -Fq 'USE_MNNVL: "0"' "${CONFIG_PATH}"
grep -Fq 'gpus_per_node: 8' "${CONFIG_PATH}"
grep -Fq 'num_nodes: 32' "${CONFIG_PATH}"

RUN_NAME=${RUN_NAME:-"${MODEL_ID}-$(date +%Y%m%d-%H%M%S)"}
PARTITION=${PARTITION:-batch}
WANDB_ENABLED=${WANDB_ENABLED:-True}
mkdir -p "${RUN_ROOT}"

cache_tag=$(printf '%s' "${RUN_NAME}" | tr -c '[:alnum:]_.-' '-')
export NRL_NODE_LOCAL_UV_CACHE_DIR="/raid/scratch/nemo-rl-uv-cache-${USER}-${cache_tag}"
export NEMO_RL_VENV_DIR="/raid/scratch/nemo-rl-venvs-${USER}-${cache_tag}"
export DEEPEP_OVERLAY_DIR="/raid/scratch/nemo-rl-deepep-${USER}-${cache_tag}"
export CONTAINER_ENV_VARS=NRL_NODE_LOCAL_UV_CACHE_DIR,NEMO_RL_VENV_DIR,DEEPEP_OVERLAY_DIR
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0="url.file://${NRL_NODE_LOCAL_UV_CACHE_DIR}/offline-git-mirrors/github.com/.insteadOf"
export GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=protocol.file.allow
export GIT_CONFIG_VALUE_1=always
export GIT_CONFIG_KEY_2=http.proxy
export GIT_CONFIG_VALUE_2=http://127.0.0.1:9
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=/bin/false

# shellcheck disable=SC2016,SC2089
printf -v setup_command '%q ' bash -lc \
  'set -euo pipefail
   gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
   gpu_count=$(sed "/^$/d" <<< "${gpu_names}" | wc -l)
   if [[ "${gpu_count}" != 8 ]] || ! grep -q H100 <<< "${gpu_names}"; then
     printf "Expected 8 visible H100 GPUs; observed count=%s names=%q\n" "${gpu_count}" "${gpu_names}" >&2
     exit 2
   fi
   for path in "${NRL_NODE_LOCAL_UV_CACHE_DIR}" "${NEMO_RL_VENV_DIR}" "${DEEPEP_OVERLAY_DIR}"; do
     if [[ "${path}" != /raid/scratch/* ]]; then
       printf "Expected node-local setup path under /raid/scratch; observed=%q\n" "${path}" >&2
       exit 2
     fi
     rm -rf -- "${path}"
     mkdir -p "${path}"
   done
   if [[ ! -r "${DEEPEP_WHEEL}" ]]; then
     printf "DeepEP wheel is not readable inside the container: %q\n" "${DEEPEP_WHEEL}" >&2
     exit 2
   fi
   python_bin=$(command -v python)
   printf "DEEPEP_SETUP_INPUTS wheel=%q expected=%q python=%q uv=%q\n" \
     "${DEEPEP_WHEEL}" "${DEEPEP_EXPECTED_VERSION}" "${python_bin}" "$(command -v uv)"
   local_seed="/raid/scratch/$(basename "${UV_GIT_CACHE_SEED}")"
   cp "${UV_GIT_CACHE_SEED}" "${local_seed}"
   tar -xf "${local_seed}" -C "${NRL_NODE_LOCAL_UV_CACHE_DIR}"
   rm -f -- "${local_seed}"
   uv pip install --no-config --python "${python_bin}" --target "${DEEPEP_OVERLAY_DIR}" --no-deps --reinstall "${DEEPEP_WHEEL}"
   PYTHONPATH="${DEEPEP_OVERLAY_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python -c "import importlib.metadata as m; import os; import torch; import hybrid_ep_cpp; v=m.version(\"deep_ep\"); print(\"DEEPEP_SETUP_VERSION\", v); assert v == os.environ[\"DEEPEP_EXPECTED_VERSION\"], v"'
SETUP_COMMAND=${setup_command}

driver_args=(
  uv run --no-sync examples/run_grpo.py
  --config "${CONFIG_PATH}"
  "grpo.max_num_steps=${MAX_STEPS}"
  checkpointing.enabled=false
  "cluster.segment_size=${SEGMENT_SIZE}"
  "logger.log_dir=${RUN_ROOT}/training"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  logger.wandb.project=sna-hybridep-h100-5515-ab
  "logger.wandb.name=${RUN_NAME}"
  logger.tensorboard_enabled=True
)
if [[ -n "${PRETRAINED_CHECKPOINT_PATH:-}" ]]; then
  driver_args+=(
    "policy.pretrained_checkpoint.format=${PRETRAINED_CHECKPOINT_FORMAT}"
    "policy.pretrained_checkpoint.path=${PRETRAINED_CHECKPOINT_PATH}"
  )
fi
printf -v driver_command '%q ' "${driver_args[@]}"
# shellcheck disable=SC2089
printf -v version_check 'python -c %q' \
  "import importlib.metadata as m; import torch; import hybrid_ep_cpp; v=m.version('deep_ep'); print('DEEPEP_RUNTIME_VERSION', v); assert v == '${EXPECTED_DEEPEP_VERSION}', v"
if [[ "${SETUP_SMOKE_ONLY}" == 1 ]]; then
  COMMAND="${version_check}"
else
  COMMAND="${version_check} && ${driver_command}"
fi

metadata_path=${RUN_ROOT}/submission.env
{
  printf 'run_name=%q\n' "${RUN_NAME}"
  printf 'config_path=%q\n' "${CONFIG_PATH}"
  printf 'rl_commit=%q\n' "${RL_COMMIT}"
  printf 'bridge_commit=%q\n' "${BRIDGE_COMMIT}"
  printf 'megatron_lm_commit=%q\n' "${MCORE_COMMIT}"
  printf 'pr_5515_state=%q\n' "${EXPECTED_5515_STATE}"
  printf 'deepep_commit=%q\n' "${EXPECTED_DEEPEP_COMMIT}"
  printf 'nodes=%q\n' "${NUM_ACTOR_NODES}"
  printf 'gpus_per_node=%q\n' "${GPUS_PER_NODE}"
  printf 'ep_size=32\n'
  printf 'max_steps=%q\n' "${MAX_STEPS}"
  printf 'setup_smoke_only=%q\n' "${SETUP_SMOKE_ONLY}"
  printf 'pretrained_checkpoint_format=%q\n' "${PRETRAINED_CHECKPOINT_FORMAT}"
  printf 'pretrained_checkpoint_path=%q\n' "${PRETRAINED_CHECKPOINT_PATH:-}"
  printf 'submitted_at=%q\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "${metadata_path}"

# shellcheck disable=SC2090
export SETUP_COMMAND COMMAND CONTAINER HF_HOME DEEPEP_WHEEL UV_GIT_CACHE_SEED
export DEEPEP_EXPECTED_VERSION="${EXPECTED_DEEPEP_VERSION}"
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/cache"}
export MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre,/raid/scratch:/raid/scratch"
export BASE_LOG_DIR="${RUN_ROOT}/ray"
export GPUS_PER_NODE
export NRL_FORCE_REBUILD_VENVS=true
export NCCL_NVLS_ENABLE=0
export PYTHONPATH="${DEEPEP_OVERLAY_DIR}:${PROJECT_ROOT}:${PROJECT_ROOT}/${BRIDGE_DIR}/src:${PROJECT_ROOT}/${MCORE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

sbatch_args=(
  --export=ALL
  --nodes="${NUM_ACTOR_NODES}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --segment="${SEGMENT_SIZE}"
  --account="${ACCOUNT}"
  --job-name="${ACCOUNT}.${RUN_NAME}"
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --output="${RUN_ROOT}/slurm-%j.out"
  ray.sub
)

sbatch --test-only "${sbatch_args[@]}"
if [[ "${TEST_ONLY:-0}" == 1 ]]; then
  printf 'Scheduler validation passed; TEST_ONLY=1, not submitting.\n'
  exit 0
fi

job_id=$(sbatch --parsable "${sbatch_args[@]}")
printf 'job_id=%q\n' "${job_id}" >> "${metadata_path}"
printf 'Submitted job %s\nMetadata: %s\n' "${job_id}" "${metadata_path}"
