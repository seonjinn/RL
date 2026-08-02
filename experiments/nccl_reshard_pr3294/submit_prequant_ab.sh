#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
MAX_STEPS=${MAX_STEPS:-5}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
REPO=${REPO:-${DEFAULT_REPO}}
CONTAINER=${CONTAINER:?CONTAINER is required}
RESULT_ROOT=${RESULT_ROOT:-${WORK_ROOT}/experiments/pr3294-nccl-mxfp8-prequant/gcp-b200}

TOTAL_NODES=${TOTAL_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GEN_NODES=${GEN_NODES:-2}
SEGMENT_SIZE=${SEGMENT_SIZE:-1}
USE_CONTAINER_VENVS=${USE_CONTAINER_VENVS:-false}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3294-nccl-mxfp8-prequant}
DEPENDENCY=${DEPENDENCY:-}
MODES=${MODES:-"bf16 mxfp8-nccl-prequant mxfp8-rollout"}
ARMS=${ARMS:-optimized}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/nccl-reshard-pr3294/v2-vllm025-py31313}
SHARED_UV_CACHE=${SHARED_UV_CACHE:-${WORK_ROOT}/mopd_nano_fast/.cache/nccl-reshard-pr3294/v2-vllm025-shared/uv}
RAY_BOOTSTRAP_ARCHIVE=${RAY_BOOTSTRAP_ARCHIVE:-${WORK_ROOT}/mopd_nano_fast/.cache/nccl-reshard-pr3294/bootstrap/ray-2.56.1-py31313.tar.gz}

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
BATCH_SCRIPT=${REPO}/experiments/nccl_reshard_pr3294/run_arm.sbatch
test -x "${BATCH_SCRIPT}"
test -f "${CONTAINER}"
if [[ "${USE_CONTAINER_VENVS}" != true ]]; then
  test -f "${RAY_BOOTSTRAP_ARCHIVE}"
fi
mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/manifests"

MANIFEST=${RESULT_ROOT}/manifests/submission-${RUN_SUFFIX}.tsv
printf 'mode\tarm\taction\tjob_id\trepo_sha\trun_name\n' >"${MANIFEST}"

for mode in ${MODES}; do
  for arm in ${ARMS}; do
    case "${arm}" in
      baseline|optimized) ;;
      *) echo "ARMS entries must be baseline or optimized" >&2; exit 2 ;;
    esac

    run_name="pr3294-nccl-gcp-b200-${mode}-${arm}-${MAX_STEPS}step-${RUN_SUFFIX}"
    experiment_root="${RESULT_ROOT}/results/${run_name}"
    args=(
      --account="${ACCOUNT}"
      --partition="${PARTITION}"
      --nodes="${TOTAL_NODES}"
      --ntasks-per-node=1
      --gpus-per-node="${GPUS_PER_NODE}"
      --exclusive
      --time="${WALLTIME}"
      --job-name="pr3294-${mode}-${arm}-${MAX_STEPS}step-${RUN_SUFFIX}"
      --output="${RESULT_ROOT}/slurm/%x-%j.out"
      --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"nccl_reshard_refit","description":"venv setup and model initialization"}}'
      --export="ALL,ARM=${arm},MODE=${mode},RUN_NAME=${run_name},EXPERIMENT_ROOT=${experiment_root},REPO=${REPO},EXPECTED_REPO_SHA=${REPO_SHA},CONTAINER=${CONTAINER},TOTAL_NODES=${TOTAL_NODES},GPUS_PER_NODE=${GPUS_PER_NODE},GEN_NODES=${GEN_NODES},SEGMENT_SIZE=${SEGMENT_SIZE},MAX_STEPS=${MAX_STEPS},WORK_ROOT=${WORK_ROOT},RAY_SUB_PATH=${REPO}/ray.sub,USE_CONTAINER_VENVS=${USE_CONTAINER_VENVS},CACHE_ROOT=${CACHE_ROOT},SHARED_UV_CACHE=${SHARED_UV_CACHE},RAY_BOOTSTRAP_ARCHIVE=${RAY_BOOTSTRAP_ARCHIVE},WANDB_PROJECT=${WANDB_PROJECT}"
    )
    if [[ -n "${DEPENDENCY}" ]]; then
      args+=(--dependency="${DEPENDENCY}" --kill-on-invalid-dep=yes)
    fi
    if [[ -n "${ACTION_ARG}" ]]; then
      args+=("${ACTION_ARG}")
    fi
    output=$(sbatch "${args[@]}" "${BATCH_SCRIPT}")
    job_id=$(sed -n 's/^Submitted batch job //p' <<<"${output}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${mode}" "${arm}" "${ACTION}" "${job_id:-n/a}" "${REPO_SHA}" "${run_name}" \
      | tee -a "${MANIFEST}"
  done
done

echo "manifest=${MANIFEST}"
