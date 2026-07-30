#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-test-only}
MAX_STEPS=${MAX_STEPS:-5}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
REPO=${REPO:-${WORK_ROOT}/RL-pr3294-nccl-benchmark}
CONTAINER=${CONTAINER:?CONTAINER is required}
RESULT_ROOT=${RESULT_ROOT:-${WORK_ROOT}/experiments/pr3294-nccl-mxfp8-prequant/gcp-b200}

TOTAL_NODES=${TOTAL_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GEN_NODES=${GEN_NODES:-2}
SEGMENT_SIZE=${SEGMENT_SIZE:-1}
USE_CONTAINER_VENVS=${USE_CONTAINER_VENVS:-true}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3294-nccl-mxfp8-prequant}

case "${ACTION}" in
  test-only) ACTION_ARG=--test-only ;;
  submit) ACTION_ARG= ;;
  *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
esac

git -C "${REPO}" pull --ff-only
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
BATCH_SCRIPT=${REPO}/experiments/nccl_reshard_pr3294/run_prequant_ab.sbatch
test -x "${BATCH_SCRIPT}"
test -f "${CONTAINER}"
mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/manifests"

MANIFEST=${RESULT_ROOT}/manifests/submission-${RUN_SUFFIX}.tsv
printf 'modes\taction\tjob_id\trepo_sha\trun_suffix\n' >"${MANIFEST}"

args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes="${TOTAL_NODES}"
  --ntasks-per-node=1
  --gpus-per-node="${GPUS_PER_NODE}"
  --exclusive
  --time="${WALLTIME}"
  --job-name="pr3294-nccl-prequant-ab-${MAX_STEPS}step-${RUN_SUFFIX}"
  --output="${RESULT_ROOT}/slurm/%x-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"nccl_reshard_refit","description":"venv setup and model initialization"}}'
  --export="ALL,REPO=${REPO},EXPECTED_REPO_SHA=${REPO_SHA},CONTAINER=${CONTAINER},TOTAL_NODES=${TOTAL_NODES},GPUS_PER_NODE=${GPUS_PER_NODE},GEN_NODES=${GEN_NODES},SEGMENT_SIZE=${SEGMENT_SIZE},MAX_STEPS=${MAX_STEPS},RUN_SUFFIX=${RUN_SUFFIX},RESULT_ROOT=${RESULT_ROOT},WORK_ROOT=${WORK_ROOT},RAY_SUB_PATH=${REPO}/ray.sub,USE_CONTAINER_VENVS=${USE_CONTAINER_VENVS},WANDB_PROJECT=${WANDB_PROJECT}"
)
if [[ -n "${ACTION_ARG}" ]]; then
  args+=("${ACTION_ARG}")
fi
output=$(sbatch "${args[@]}" "${BATCH_SCRIPT}")
job_id=$(sed -n 's/^Submitted batch job //p' <<<"${output}")
printf '%s\t%s\t%s\t%s\t%s\n' \
  "bf16,mxfp8-nccl-prequant,mxfp8-rollout" "${ACTION}" "${job_id:-n/a}" \
  "${REPO_SHA}" "${RUN_SUFFIX}" | tee -a "${MANIFEST}"

echo "manifest=${MANIFEST}"
