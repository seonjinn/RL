#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
REPO=${REPO:-/home/sna/RL-pr3294-fused-refit-spike-ptyche-20260823}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/mxfp8-fused-refit-spike}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/microbench}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
BENCHMARK=${BENCHMARK:-benchmark_batched_expert_copy.py}

if [[ "${ACTION}" == render ]]; then
  printf 'benchmark=%s\nrepo=%s\ncontainer=%s\nresult=%s\n' \
    "${BENCHMARK}" "${REPO}" "${CONTAINER}" "${RUN_ROOT}"
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

test -n "${EXPECTED_HEAD}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)"
test -e "${CONTAINER}"
test -e "${REPO}/experiments/mxfp8_fused_refit_spike/${BENCHMARK}"
mkdir -p "${RUN_ROOT}"

COMMAND=$(cat <<EOF
set -euo pipefail
test "\$(git -C ${REPO} rev-parse HEAD)" = "${EXPECTED_HEAD}"
cd ${REPO}
/opt/nemo_rl_venv/bin/python experiments/mxfp8_fused_refit_spike/${BENCHMARK} \
  | tee ${RUN_ROOT}/result.json
EOF
)

exec sbatch "${SBATCH_ACTION[@]}" \
  --nodes=1 \
  --ntasks=1 \
  --segment=1 \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --time=00:30:00 \
  --job-name="${ACCOUNT}-mxfp8-batched-copy" \
  --output="${RUN_ROOT}/slurm-%j.out" \
  --wrap="srun --container-image=${CONTAINER} --container-mounts=/home:/home,/lustre:/lustre bash -lc '$COMMAND'"
