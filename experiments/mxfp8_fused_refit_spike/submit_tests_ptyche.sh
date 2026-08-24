#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
REPO=${REPO:-/home/sna/RL-pr3294-fused-refit-spike-ptyche-20260823}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/mxfp8-fused-refit-spike}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/tests}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}

if [[ "${ACTION}" == render ]]; then
  printf 'tests=MXFP8-layout,batched-expert-replay\nrepo=%s\nresult=%s\n' \
    "${REPO}" "${RUN_ROOT}"
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
mkdir -p "${RUN_ROOT}"

COMMAND=$(cat <<EOF
set -euo pipefail
test "\$(git -C ${REPO} rev-parse HEAD)" = "${EXPECTED_HEAD}"
cd ${REPO}
PYTHONPATH=${REPO} /opt/nemo_rl_venv/bin/python -m pytest -q \
  tests/unit/models/generation/test_mxfp8_prequant.py::test_batched_moe_shuffle_matches_per_expert \
  tests/unit/models/generation/test_vllm_fp8_quantization.py::test_batched_moe_shuffle_matches_per_expert \
  tests/unit/models/generation/test_vllm_fp8_quantization.py::test_process_mxfp8_moe_refit_uses_batched_flashinfer_shuffle \
  tests/unit/models/generation/test_vllm_fp8_quantization.py::test_process_weights_after_loading_copies_in_place_on_refit \
  tests/unit/models/generation/test_vllm_refit_loader.py::test_refit_loader_cache_batches_mxfp8_expert_replay \
  --vllm-only \
  | tee ${RUN_ROOT}/pytest.txt
EOF
)

exec sbatch "${SBATCH_ACTION[@]}" \
  --nodes=1 \
  --ntasks=1 \
  --segment=1 \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --time=00:30:00 \
  --job-name="${ACCOUNT}-mxfp8.refit-tests" \
  --output="${RUN_ROOT}/slurm-%j.out" \
  --wrap="srun --container-image=${CONTAINER} --container-mounts=/home:/home,/lustre:/lustre bash -lc '$COMMAND'"
