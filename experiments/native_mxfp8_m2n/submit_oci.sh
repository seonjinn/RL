#!/usr/bin/env bash
set -euo pipefail

ACTION=${ACTION:-test-only}
REPO=${REPO:-$(git rev-parse --show-toplevel)}
NODES=${NODES:-1}
: "${CONTAINER:?Set an immutable NeMo-RL image path}"
: "${SLURM_ACCOUNT:?Check FairShare first}"
: "${RESULT_ROOT:?Set the durable result directory}"
case "${ACTION}" in test-only|submit) ;; *) exit 2 ;; esac
case "${NODES}" in 1|2) ;; *) echo "Use one or two nodes for this probe" >&2; exit 2 ;; esac
if [[ "${ACTION}" == submit ]]; then
  git -C "${REPO}" pull --ff-only
fi
if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=none)" ]]; then
  echo "Source must be clean and pushed" >&2
  exit 2
fi
SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
if [[ "${SOURCE_SHA}" != "$(git -C "${REPO}" rev-parse '@{upstream}')" ]]; then
  echo "Push this commit before submitting" >&2
  exit 2
fi
RUN_NAME=${RUN_NAME:-m2n-native-mxfp8-${SOURCE_SHA:0:8}-${NODES}n}
RESULT_DIR=${RESULT_ROOT}/${RUN_NAME}
mkdir -p "${RESULT_DIR}"
export REPO SOURCE_SHA CONTAINER RESULT_DIR
args=(--parsable --nodes="${NODES}" --account="${SLURM_ACCOUNT}"
      --job-name="${SLURM_ACCOUNT}.${RUN_NAME}" --partition="${PARTITION:-batch}"
      --time=04:00:00 --gres=gpu:4 --ntasks-per-node=1 --exclusive --mem=0
      --output="${RESULT_DIR}/slurm-%j.log")
if [[ "${ACTION}" == test-only ]]; then
  args+=(--test-only)
fi
sbatch "${args[@]}" "${REPO}/experiments/native_mxfp8_m2n/job.sh"
