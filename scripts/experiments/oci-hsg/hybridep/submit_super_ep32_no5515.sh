#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PROFILE=${1:-"${SCRIPT_DIR}/models/nemotron3-super-120ba12b-32n4g-ep32-no5515.env"}

cd "${PROJECT_ROOT}"

if [[ ! -f "${PROFILE}" ]]; then
  printf 'Profile does not exist: %s\n' "${PROFILE}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${PROFILE}"

: "${ACCOUNT:?Set ACCOUNT to a valid SLURM account.}"
: "${CONTAINER:?Set CONTAINER to an existing .sqsh file under /lustre.}"
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

case "${CONTAINER}" in
  /lustre/*.sqsh) ;;
  *) printf 'CONTAINER must be an immutable .sqsh path under /lustre.\n' >&2; exit 2 ;;
esac
case "${HF_HOME}" in
  /lustre/*) ;;
  *) printf 'HF_HOME must be under /lustre.\n' >&2; exit 2 ;;
esac
case "${RUN_ROOT}" in
  /lustre/*) ;;
  *) printf 'RUN_ROOT must be under /lustre.\n' >&2; exit 2 ;;
esac

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
  printf '%s\n' "${submodule_state}" >&2
  exit 2
fi
git submodule update --recursive --force --no-fetch

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Submodule update left a dirty checkout.\n' >&2
  git status --short >&2
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
git -C "${MCORE_DIR}" merge-base --is-ancestor \
  723db5a72790aefc02f5a0228e6607eef70c0533 HEAD
case "${EXPECTED_5515_STATE}" in
  present)
    git -C "${MCORE_DIR}" merge-base --is-ancestor \
      278cc9128c233a38ea9fa8ac7cf9de22e434efa6 HEAD
    ;;
  absent)
    if git -C "${MCORE_DIR}" cat-file -e \
      "278cc9128c233a38ea9fa8ac7cf9de22e434efa6^{commit}" 2>/dev/null && \
      git -C "${MCORE_DIR}" merge-base --is-ancestor \
        278cc9128c233a38ea9fa8ac7cf9de22e434efa6 HEAD; then
      printf 'MCore unexpectedly contains the #5515 padding-exclusion commit.\n' >&2
      exit 2
    fi
    ;;
  *)
    printf 'EXPECTED_5515_STATE must be present or absent, got: %s\n' \
      "${EXPECTED_5515_STATE}" >&2
    exit 2
    ;;
esac

grep -Fq "DeepEP.git@${EXPECTED_DEEPEP_COMMIT}" pyproject.toml
grep -Fq "rev=${EXPECTED_DEEPEP_COMMIT}#${EXPECTED_DEEPEP_COMMIT}" uv.lock
grep -Fq 'expert_model_parallel_size: 32' "${CONFIG_PATH}"
grep -Fq 'moe_hybridep_prepad_packed_inputs: false' "${CONFIG_PATH}"

RUN_NAME=${RUN_NAME:-"${MODEL_ID}-$(date +%Y%m%d-%H%M%S)"}
PARTITION=${PARTITION:-batch}
WANDB_ENABLED=${WANDB_ENABLED:-False}
mkdir -p "${RUN_ROOT}"

driver_args=(
  uv run examples/run_grpo.py
  --config "${CONFIG_PATH}"
  "grpo.max_num_steps=${MAX_STEPS}"
  checkpointing.enabled=false
  "logger.log_dir=${RUN_ROOT}/training"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.wandb.name=${RUN_NAME}"
  logger.tensorboard_enabled=True
)
printf -v driver_command '%q ' "${driver_args[@]}"
# shellcheck disable=SC2089
printf -v version_check \
  'uv run python -c %q' \
  "import importlib.metadata as m; v=m.version('deep_ep'); print('DEEPEP_RUNTIME_VERSION', v); assert v == '${EXPECTED_DEEPEP_VERSION}', v"
COMMAND="${version_check} && ${driver_command}"

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
  printf 'submitted_at=%q\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "${metadata_path}"

# shellcheck disable=SC2090
export COMMAND CONTAINER HF_HOME
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/cache"}
export MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre"
export BASE_LOG_DIR="${RUN_ROOT}/ray"
export GPUS_PER_NODE
export NRL_FORCE_REBUILD_VENVS=true
export NCCL_NVLS_ENABLE=0
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/${BRIDGE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

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
