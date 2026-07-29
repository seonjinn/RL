#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

touch "${TMP_DIR}/container.sqsh"
cat >"${TMP_DIR}/capture_command.sh" <<'EOF'
#!/usr/bin/env bash
printf 'SETUP_COMMAND=%s\n' "${SETUP_COMMAND:-}"
printf 'PATH=%s\n' "${PATH}"
printf '%s\n' "${COMMAND}"
EOF
chmod +x "${TMP_DIR}/capture_command.sh"

render_command() {
  local mode=$1
  ARM=optimized \
  MODE="${mode}" \
  REPO="${REPO_ROOT}" \
  CONTAINER="${TMP_DIR}/container.sqsh" \
  TOTAL_NODES=3 \
  GPUS_PER_NODE=8 \
  GEN_NODES=1 \
  SEGMENT_SIZE=4 \
  MAX_STEPS=5 \
  RUN_NAME="test-${mode}" \
  EXPERIMENT_ROOT="${TMP_DIR}/${mode}" \
  WORK_ROOT="${TMP_DIR}/work" \
  CACHE_ROOT="${TMP_DIR}/cache-${mode}" \
  SHARED_UV_CACHE="${TMP_DIR}/uv" \
  RAY_SUB_PATH="${TMP_DIR}/capture_command.sh" \
  REFIT_TRANSPORT=null \
  bash "${REPO_ROOT}/experiments/nccl_reshard_pr3294/run_arm.sbatch"
}

rollout_command=$(render_command mxfp8-rollout)
grep -q "policy.generation.refit_transport='null'" <<<"${rollout_command}"
grep -q "policy.generation.vllm_cfg.precision=fp8" <<<"${rollout_command}"
grep -q "policy.generation.vllm_cfg.is_mx=true" <<<"${rollout_command}"
if grep -q "policy.megatron_cfg.fp8_cfg.fp8_param=true" <<<"${rollout_command}"; then
  echo "mxfp8-rollout must retain BF16 trainer parameter storage" >&2
  exit 1
fi

probe_command=$(render_command mxfp8-probe)
uv_line=$(grep '^uv run ' <<<"${probe_command}")
grep -q "policy.megatron_cfg.fp8_cfg.fp8_param=true" <<<"${uv_line}"
grep -q "policy.generation.vllm_cfg.is_mx=true" <<<"${uv_line}"

touch "${TMP_DIR}/ray-bootstrap.tar.gz"
archive_command=$(
  ARM=optimized \
  MODE=mxfp8-rollout \
  REPO="${REPO_ROOT}" \
  CONTAINER="${TMP_DIR}/container.sqsh" \
  TOTAL_NODES=3 \
  GPUS_PER_NODE=8 \
  GEN_NODES=1 \
  SEGMENT_SIZE=2 \
  MAX_STEPS=5 \
  RUN_NAME=test-archive \
  EXPERIMENT_ROOT="${TMP_DIR}/archive" \
  WORK_ROOT="${TMP_DIR}/work" \
  CACHE_ROOT="${TMP_DIR}/cache-archive" \
  SHARED_UV_CACHE="${TMP_DIR}/uv" \
  RAY_SUB_PATH="${TMP_DIR}/capture_command.sh" \
  RAY_BOOTSTRAP_ARCHIVE="${TMP_DIR}/ray-bootstrap.tar.gz" \
  RAY_BOOTSTRAP_LOCAL_ROOT=/tmp/test-ray-bootstrap \
  REFIT_TRANSPORT=null \
  bash "${REPO_ROOT}/experiments/nccl_reshard_pr3294/run_arm.sbatch"
)
grep -q "tar -xzf '${TMP_DIR}/ray-bootstrap.tar.gz'" <<<"${archive_command}"
grep -q "^PATH=/tmp/test-ray-bootstrap/bin:" <<<"${archive_command}"
grep -q "export UV_PYTHON='/tmp/test-ray-bootstrap/bin/python3.13'" \
  <<<"${archive_command}"

cat >"${TMP_DIR}/capture_arm.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s|%s|%s\n' "${ARM}" "${RUN_NAME}" "${EXPERIMENT_ROOT}"
EOF
chmod +x "${TMP_DIR}/capture_arm.sh"

pair_output=$(
  MODE=mxfp8-rollout \
  REPO="${REPO_ROOT}" \
  CONTAINER="${TMP_DIR}/container.sqsh" \
  TOTAL_NODES=3 \
  GPUS_PER_NODE=8 \
  GEN_NODES=1 \
  SEGMENT_SIZE=2 \
  MAX_STEPS=5 \
  RUN_PREFIX=test-pair \
  RESULT_ROOT="${TMP_DIR}/pair-results" \
  WORK_ROOT="${TMP_DIR}/work" \
  ARM_RUNNER="${TMP_DIR}/capture_arm.sh" \
  REFIT_TRANSPORT=null \
  bash "${REPO_ROOT}/experiments/nccl_reshard_pr3294/run_pair.sbatch"
)
grep -q "^baseline|test-pair-baseline|${TMP_DIR}/pair-results/results/test-pair-baseline$" \
  <<<"${pair_output}"
grep -q "^optimized|test-pair-optimized|${TMP_DIR}/pair-results/results/test-pair-optimized$" \
  <<<"${pair_output}"

bash -n "${REPO_ROOT}/experiments/nccl_reshard_pr3294/run_arm.sbatch"
bash -n "${REPO_ROOT}/experiments/nccl_reshard_pr3294/run_pair.sbatch"
bash -n "${REPO_ROOT}/experiments/nccl_reshard_pr3294/submit_suite.sh"

gb200_defaults=$(
  sed -n '/^  gb200)/,/^    ;;/p' \
    "${REPO_ROOT}/experiments/nccl_reshard_pr3294/submit_suite.sh"
)
grep -Fq 'SEGMENT_SIZE=${SEGMENT_SIZE:-1}' <<<"${gb200_defaults}"

if grep -Fq 'REFIT_TRANSPORT=${REFIT_TRANSPORT},CONTAINER_ENV_VARS=PATH"' \
  "${REPO_ROOT}/experiments/nccl_reshard_pr3294/submit_suite.sh"; then
  echo "submit_suite must not overwrite the container PATH without a bootstrap venv" >&2
  exit 1
fi
