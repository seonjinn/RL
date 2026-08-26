#!/bin/bash

set -euo pipefail

ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}
PARTITION=${PARTITION:-batch}
NUM_NODES=${NUM_NODES:-24}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MAX_STEPS=${MAX_STEPS:-200}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MODE=${MODE:-submit}
VERIFY_IMAGE_SHA256=${VERIFY_IMAGE_SHA256:-true}

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${ROOT}/RL_worktrees/qwen3-30ba3b-async8off-logprob-skip-20260825}
IMAGE=${IMAGE:-${ROOT}/containers/nemo-rl-nightly-20260817/nemo_rl_nightly_20260817_15967993.sqsh}
HF_HOME=${HF_HOME:-${ROOT}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-${ROOT}/uv_cache}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-${ROOT}/experiments/qwen3-30ba3b-async8off-accuracy-20260825}
CONFIG_REL=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-24n8g-async-8off.yaml
CONFIG=${REPO}/${CONFIG_REL}
PROJECT=${WANDB_PROJECT:-sna-qwen3-async8off-accuracy}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
MODEL_REF_FILE=${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/refs/main
DATASET_CACHE_ROOT=${HF_DATASETS_CACHE}/nvidia___open_math_instruct-2
SBATCH_EXPORTS=CONTAINER,MOUNTS,COMMAND,GPUS_PER_NODE,BASE_LOG_DIR,HF_HOME,HF_DATASETS_CACHE,HF_HUB_OFFLINE,HF_DATASETS_OFFLINE,UV_CACHE_DIR_OVERRIDE,SLURM_EXPORT_ENV,PATH,HOME,USER,LOGNAME,LD_LIBRARY_PATH,HTTP_PROXY,HTTPS_PROXY,NO_PROXY,http_proxy,https_proxy,no_proxy

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed experiment commit}"
if [[ ${MODE} != submit && ${MODE} != test ]]; then
  echo "MODE must be submit or test" >&2
  exit 2
fi

if [[ ${MODE} == submit ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY must be set in submit mode}"
  SBATCH_EXPORTS+=,WANDB_API_KEY
fi

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)"
test -f "${IMAGE}"
test -f "${CONFIG}"
test -f "${REPO}/ray.sub"
test -f "${MODEL_REF_FILE}"
test -d "${DATASET_CACHE_ROOT}"

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/manifests" "${UV_CACHE_DIR_OVERRIDE}"
manifest=${EXPERIMENT_DIR}/manifests/${RUN_ID}.txt
test ! -e "${manifest}"

config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')
if [[ -f ${IMAGE}.metadata.txt ]]; then
  image_sha256=$(awk -F= '$1 == "sha256" {print $2}' "${IMAGE}.metadata.txt")
  image_source_commit=$(awk -F= '$1 == "source_commit" {print $2}' "${IMAGE}.metadata.txt")
else
  image_sha256=$(sha256sum "${IMAGE}" | awk '{print $1}')
  image_source_commit=unknown
fi
test -n "${image_sha256}"
if [[ ${VERIFY_IMAGE_SHA256} == true ]]; then
  printf '%s  %s\n' "${image_sha256}" "${IMAGE}" | sha256sum --check
fi

model_revision=$(tr -d '\n' < "${MODEL_REF_FILE}")
model_snapshot_dir=${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/${model_revision}
mapfile -t dataset_cache_dirs < <(
  find "${DATASET_CACHE_ROOT}" -mindepth 3 -maxdepth 3 -type d -print | sort
)
test -n "${model_revision}"
test -d "${model_snapshot_dir}"
test -s "${model_snapshot_dir}/config.json"
test "${#dataset_cache_dirs[@]}" -eq 1
dataset_cache_dir=${dataset_cache_dirs[0]}
test -d "${dataset_cache_dir}"
dataset_cache_marker=$(find "${dataset_cache_dir}" -type f -size +0c -print -quit)
test -n "${dataset_cache_marker}"
dataset_fingerprint=$(basename "${dataset_cache_dir}")

{
  echo "run_id=${RUN_ID}"
  echo "mode=${MODE}"
  echo "commit=${actual_commit}"
  echo "container=${IMAGE}"
  echo "container_sha256=${image_sha256}"
  echo "container_source_commit=${image_source_commit}"
  echo "config=${CONFIG}"
  echo "config_sha256=${config_sha256}"
  echo "model_revision=${model_revision}"
  echo "model_snapshot_dir=${model_snapshot_dir}"
  echo "dataset_cache_dir=${dataset_cache_dir}"
  echo "dataset_cache_marker=${dataset_cache_marker}"
  echo "dataset_fingerprint=${dataset_fingerprint}"
  echo "topology=${NUM_NODES}_nodes,$((NUM_NODES * GPUS_PER_NODE))_H100"
  echo "steps=${MAX_STEPS}"
  echo "seed=42"
  echo "rollout_batch=2048"
  echo "reference_policy_kl_penalty=0.01"
  echo "seq_logprob_error_threshold=null"
  echo "baseline=train_gbs_512,force_on_policy_ratio_false"
  echo "gbs_only=train_gbs_2048,force_on_policy_ratio_false"
  echo "gbs_and_skip=train_gbs_2048,force_on_policy_ratio_true"
} | tee "${manifest}"

submitted_jobs=()
release_complete=false

cancel_held_jobs_on_exit() {
  local exit_code=$?

  trap - EXIT INT TERM
  if [[ ${MODE} == submit && ${release_complete} == false && ${#submitted_jobs[@]} -gt 0 ]]; then
    scancel "${submitted_jobs[@]}"
  fi
  exit "${exit_code}"
}

trap cancel_held_jobs_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

submit_case() {
  local case_name=$1
  local train_gbs=$2
  local force_on_policy=$3
  local run_name="qwen3-30ba3b-8off-accuracy-${case_name}-${RUN_ID}"
  local log_dir="${EXPERIMENT_DIR}/logs/${run_name}"
  local command
  local output

  mkdir -p "${log_dir}"

  printf -v command "test \"\$(git rev-parse HEAD)\" = %q && \
    test -z \"\$(git status --porcelain --untracked-files=all)\" && \
    test \"\$(sha256sum %q | awk '{print \$1}')\" = %q && \
    test \"\$(tr -d '\\n' < %q)\" = %q && \
    test -s %q && \
    test -s %q && \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NRL_FORCE_REBUILD_VENVS=true \
    uv run examples/run_grpo.py \
    --config %q \
    %q \
    grpo.seed=42 \
    grpo.seq_logprob_error_threshold=null \
    loss_fn.reference_policy_kl_penalty=0.01 \
    %q \
    %q \
    checkpointing.enabled=false \
    %q \
    logger.wandb_enabled=true \
    %q \
    %q" \
    "${EXPECTED_COMMIT}" \
    "${CONFIG_REL}" \
    "${config_sha256}" \
    "${MODEL_REF_FILE}" \
    "${model_revision}" \
    "${model_snapshot_dir}/config.json" \
    "${dataset_cache_marker}" \
    "${CONFIG_REL}" \
    "grpo.max_num_steps=${MAX_STEPS}" \
    "loss_fn.force_on_policy_ratio=${force_on_policy}" \
    "policy.train_global_batch_size=${train_gbs}" \
    "logger.log_dir=${log_dir}" \
    "logger.wandb.project=${PROJECT}" \
    "logger.wandb.name=${run_name}"

  export CONTAINER="${IMAGE}"
  export HF_HOME
  export HF_DATASETS_CACHE
  export UV_CACHE_DIR_OVERRIDE
  export MOUNTS="/lustre:/lustre"
  export COMMAND="${command}"
  export GPUS_PER_NODE
  export BASE_LOG_DIR="${log_dir}"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export SLURM_EXPORT_ENV=ALL

  if [[ ${MODE} == test ]]; then
    output=$(
      cd "${REPO}"
      sbatch --test-only \
        --export="${SBATCH_EXPORTS}" \
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
        --hold \
        --export="${SBATCH_EXPORTS}" \
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

  if [[ ${MODE} == submit ]]; then
    job_id=${output%%;*}
    [[ ${job_id} =~ ^[0-9]+$ ]]
    submitted_jobs+=("${job_id}")
  fi
  printf '%s\n' "${output}"
  printf '%s\n' "job_${case_name}=${output}" | tee -a "${manifest}"
}

submit_case baseline 512 false
submit_case gbs2048_no_force 2048 false
submit_case gbs2048_force_on_policy 2048 true

if [[ ${MODE} == submit ]]; then
  job_list=$(IFS=,; printf '%s' "${submitted_jobs[*]}")
  scontrol release "${job_list}"
  release_complete=true
  printf 'released_jobs=%s\n' "${job_list}" | tee -a "${manifest}"
fi

trap - EXIT INT TERM
