#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-pr3294-fused-refit-spike-20260823}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-/home/sna/RL-pr3294-fused-refit-spike-20260823}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/experiments/mxfp8-fused-refit-spike}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/microbench}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}

if [[ "${ACTION}" == render ]]; then
  printf 'model=Qwen3-30B-A3B\nshape=E128-I768-K2048\nbenchmark=current-vs-direct-layout\n'
  exit 0
fi
case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

git -C "${REPO}" fetch origin "${BRANCH}"
git -C "${REPO}" pull --ff-only origin "${BRANCH}"
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
test -e "${CONTAINER}"
mkdir -p "${RUN_ROOT}"

COMMAND="cd ${REPO} && /opt/nemo_rl_venv/bin/python experiments/mxfp8_fused_refit_spike/benchmark_direct_expert_layout.py | tee ${RUN_ROOT}/result.json"
export COMMAND CONTAINER
export MOUNTS=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch

exec sbatch "${SBATCH_ACTION[@]}" \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --time=00:30:00 \
  --job-name="${ACCOUNT}-mxfp8-layout-microbench" \
  --output="${RUN_ROOT}/slurm-%j.out" \
  --wrap='srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" bash -lc "$COMMAND"'
