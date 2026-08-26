#!/bin/bash

set -euo pipefail

ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}
PARTITION=${PARTITION:-batch}
NUM_NODES=${NUM_NODES:-24}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MAX_STEPS=${MAX_STEPS:-200}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MODE=${MODE:-submit}

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${ROOT}/RL_worktrees/qwen3-30ba3b-async8off-logprob-skip-20260825}
IMAGE=${IMAGE:-${ROOT}/containers/nemo-rl-nightly-20260817/nemo_rl_nightly_20260817_15967993.sqsh}
HF_HOME=${HF_HOME:-${ROOT}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-${ROOT}/experiments/qwen3-30ba3b-async8off-accuracy-20260825}
CONFIG_REL=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-24n8g-async-8off.yaml
CONFIG=${REPO}/${CONFIG_REL}
PROJECT=${WANDB_PROJECT:-sna-qwen3-async8off-accuracy}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed experiment commit}"

if [[ ${MODE} != submit && ${MODE} != test ]]; then
  echo "MODE must be submit or test" >&2
  exit 2
fi

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)"
test -f "${IMAGE}"
test -f "${CONFIG}"
test -f "${REPO}/ray.sub"

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/manifests"
manifest=${EXPERIMENT_DIR}/manifests/${RUN_ID}.txt
test ! -e "${manifest}"

config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')
if [[ -f ${IMAGE}.metadata.txt ]]; then
  image_sha256=$(awk -F= '$1 == "sha256" {print $2}' "${IMAGE}.metadata.txt")
else
  image_sha256=$(sha256sum "${IMAGE}" | awk '{print $1}')
fi
test -n "${image_sha256}"

{
  echo "run_id=${RUN_ID}"
  echo "mode=${MODE}"
  echo "commit=${actual_commit}"
  echo "container=${IMAGE}"
  echo "container_sha256=${image_sha256}"
  echo "config=${CONFIG}"
  echo "config_sha256=${config_sha256}"
  echo "topology=${NUM_NODES}_nodes,$((NUM_NODES * GPUS_PER_NODE))_H100"
  echo "steps=${MAX_STEPS}"
  echo "seed=42"
  echo "rollout_batch=2048"
  echo "reference_policy_kl_penalty=0.01"
  echo "seq_logprob_error_threshold=null"
  echo "baseline=train_gbs_512,force_on_policy_ratio_false"
  echo "variant=train_gbs_2048,force_on_policy_ratio_true"
} | tee "${manifest}"

submit_case() {
  local case_name=$1
  local train_gbs=$2
  local force_on_policy=$3
  local run_name="qwen3-30ba3b-8off-accuracy-${case_name}-${RUN_ID}"
  local log_dir="${EXPERIMENT_DIR}/logs/${run_name}"
  local command
  local output

  mkdir -p "${log_dir}"

  command="NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py \
    --config ${CONFIG_REL} \
    grpo.max_num_steps=${MAX_STEPS} \
    grpo.seed=42 \
    grpo.seq_logprob_error_threshold=null \
    loss_fn.reference_policy_kl_penalty=0.01 \
    loss_fn.force_on_policy_ratio=${force_on_policy} \
    policy.train_global_batch_size=${train_gbs} \
    checkpointing.enabled=false \
    logger.log_dir=${log_dir} \
    logger.wandb_enabled=true \
    logger.wandb.project=${PROJECT} \
    logger.wandb.name=${run_name}"

  export CONTAINER="${IMAGE}"
  export HF_HOME
  export HF_DATASETS_CACHE
  export MOUNTS="/lustre:/lustre"
  export COMMAND="${command}"
  export GPUS_PER_NODE

  if [[ ${MODE} == test ]]; then
    output=$(
      cd "${REPO}"
      sbatch --test-only \
        --nodes="${NUM_NODES}" \
        --account="${ACCOUNT}" \
        --job-name="${run_name}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --gres="gpu:${GPUS_PER_NODE}" \
        ray.sub
    )
  else
    output=$(
      cd "${REPO}"
      sbatch --parsable \
        --nodes="${NUM_NODES}" \
        --account="${ACCOUNT}" \
        --job-name="${run_name}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --gres="gpu:${GPUS_PER_NODE}" \
        --output="${log_dir}/slurm-%j.out" \
        ray.sub
    )
  fi

  printf '%s\n' "${output}"
  printf '%s\n' "job_${case_name}=${output}" | tee -a "${manifest}"
}

submit_case baseline 512 false
submit_case gbs2048_force_on_policy 2048 true
