#!/usr/bin/env bash

set -euo pipefail

PLATFORM=${PLATFORM:-gcp-b200}
ACTION=${ACTION:-test-only}
MODE_FILTER=${MODE_FILTER:-bf16,mxfp8-rollout}
ARM_FILTER=${ARM_FILTER:-baseline,optimized}
MAX_STEPS=${MAX_STEPS:-2}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}

case "${PLATFORM}" in
  gcp-b200)
    WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
    TOTAL_NODES=${TOTAL_NODES:-4}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    GEN_NODES=${GEN_NODES:-2}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    ;;
  cw-h100)
    WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
    TOTAL_NODES=${TOTAL_NODES:-3}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    GEN_NODES=${GEN_NODES:-1}
    SEGMENT_SIZE=${SEGMENT_SIZE:-4}
    RAY_BOOTSTRAP_VENV=${RAY_BOOTSTRAP_VENV:-${WORK_ROOT}/experiments/pr3294-nccl-reshard/cw-h100-quick/venvs/ray-2.56.1}
    ;;
  gb200)
    WORK_ROOT=${WORK_ROOT:?Set WORK_ROOT to the GB200 user root}
    TOTAL_NODES=${TOTAL_NODES:-5}
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    GEN_NODES=${GEN_NODES:-1}
    SEGMENT_SIZE=${SEGMENT_SIZE:-1}
    ;;
  *)
    echo "PLATFORM must be gcp-b200, cw-h100, or gb200" >&2
    exit 2
    ;;
esac

OPT_REPO=${OPT_REPO:-${WORK_ROOT}/RL-pr3294-nccl-benchmark}
BASELINE_REPO=${BASELINE_REPO:-${WORK_ROOT}/RL-e40-nccl-reshard-baseline}
REFIT_TRANSPORT=${REFIT_TRANSPORT:-}
CONTAINER=${CONTAINER:?CONTAINER is required}
RESULT_ROOT=${RESULT_ROOT:-${WORK_ROOT}/experiments/pr3294-nccl-reshard/${PLATFORM}}
BATCH_SCRIPT=${OPT_REPO}/experiments/nccl_reshard_pr3294/run_arm.sbatch
CONTAINER_ENV_VARS=${CONTAINER_ENV_VARS:-}
if [[ -n "${RAY_BOOTSTRAP_VENV:-}${RAY_BOOTSTRAP_ARCHIVE:-}" && \
  -z "${CONTAINER_ENV_VARS}" ]]; then
  CONTAINER_ENV_VARS=PATH
fi

case "${ACTION}" in
  test-only) ACTION_ARG=--test-only ;;
  submit) ACTION_ARG= ;;
  *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
esac

git -C "${OPT_REPO}" pull --ff-only
if [[ "$(git -C "${BASELINE_REPO}" rev-parse HEAD)" != \
  "$(git -C "${OPT_REPO}" rev-parse HEAD)" ]]; then
  test "$(git -C "${BASELINE_REPO}" rev-parse HEAD)" = \
    "e40aa046e5fd4af30f93c27acdcdb9cc748670ab"
fi
test -x "${BATCH_SCRIPT}"
test -f "${CONTAINER}"
mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/manifests"

selected() {
  local needle=$1
  local csv=$2
  [[ ",${csv}," == *",${needle},"* ]]
}

MANIFEST=${RESULT_ROOT}/manifests/submission-${RUN_SUFFIX}.tsv
printf 'platform\tmode\tarm\taction\tjob_id\trepo_sha\trun_name\n' >"${MANIFEST}"

for MODE in bf16 blockwise-fp8 mxfp8-rollout mxfp8-probe; do
  selected "${MODE}" "${MODE_FILTER}" || continue
  for ARM in baseline optimized; do
    selected "${ARM}" "${ARM_FILTER}" || continue
    if [[ "${ARM}" == baseline ]]; then
      REPO=${BASELINE_REPO}
    else
      REPO=${OPT_REPO}
    fi
    RUN_NAME="pr3294-nccl-${PLATFORM}-${MODE}-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}"
    EXPERIMENT_ROOT=${RESULT_ROOT}/results/${RUN_NAME}
    args=(
      --account="${ACCOUNT}"
      --partition="${PARTITION}"
      --nodes="${TOTAL_NODES}"
      --ntasks-per-node=1
      --exclusive
      --time="${WALLTIME}"
      --job-name="${RUN_NAME}"
      --output="${RESULT_ROOT}/slurm/%x-%j.out"
      --export="ALL,ARM=${ARM},MODE=${MODE},REPO=${REPO},CONTAINER=${CONTAINER},TOTAL_NODES=${TOTAL_NODES},GPUS_PER_NODE=${GPUS_PER_NODE},GEN_NODES=${GEN_NODES},SEGMENT_SIZE=${SEGMENT_SIZE},MAX_STEPS=${MAX_STEPS},RUN_NAME=${RUN_NAME},EXPERIMENT_ROOT=${EXPERIMENT_ROOT},WORK_ROOT=${WORK_ROOT},RAY_SUB_PATH=${OPT_REPO}/ray.sub,RAY_BOOTSTRAP_VENV=${RAY_BOOTSTRAP_VENV:-},RAY_BOOTSTRAP_ARCHIVE=${RAY_BOOTSTRAP_ARCHIVE:-},RAY_BOOTSTRAP_LOCAL_ROOT=${RAY_BOOTSTRAP_LOCAL_ROOT:-},REFIT_TRANSPORT=${REFIT_TRANSPORT},CONTAINER_ENV_VARS=${CONTAINER_ENV_VARS}"
    )
    if [[ "${PLATFORM}" == gcp-b200 ]]; then
      args+=(--gpus-per-node="${GPUS_PER_NODE}")
    else
      args+=(--gres="gpu:${GPUS_PER_NODE}")
    fi
    if [[ "${PLATFORM}" == gcp-b200 ]]; then
      args+=(--comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"nccl_reshard_refit","description":"venv setup and model initialization"}}')
    fi
    if [[ -n "${ACTION_ARG}" ]]; then
      args+=("${ACTION_ARG}")
    fi
    output=$(sbatch "${args[@]}" "${BATCH_SCRIPT}")
    job_id=$(sed -n 's/^Submitted batch job //p' <<<"${output}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${PLATFORM}" "${MODE}" "${ARM}" "${ACTION}" "${job_id:-n/a}" \
      "$(git -C "${REPO}" rev-parse HEAD)" "${RUN_NAME}" | tee -a "${MANIFEST}"
  done
done

echo "manifest=${MANIFEST}"
