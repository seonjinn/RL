#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?}"
: "${PRODUCT_SHA:?}"
: "${HARNESS_SHA:?}"
: "${ALLOWED_SIGNERS:?}"
: "${ARTIFACT_DIR:?}"
: "${RESULT_DIR:?}"
: "${CONFIG:?}"
: "${DATA_SOURCE:?}"
: "${DATASET:?}"
: "${TARGET:?}"
: "${VARIANT:?}"
: "${SEGMENT_STOP_STEP:?}"
: "${MAX_NUM_STEPS:?}"
: "${WANDB_RUN_ID:?}"
: "${WANDB_RESUME:?}"

readonly CONTAINER_SHA256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly ALLOWED_SIGNERS_SHA256=e17123da460679f323f85ac201a9826738cc6b16bb54411aa8b0adc3aa072561
train_log="${ARTIFACT_DIR}/train.log"
gates="${RESULT_DIR}/runtime-gates/step_${SEGMENT_STOP_STEP}.json"

die() {
  echo "Q8_DAPO32K_100STEP_FAIL_CLOSED: $*" >&2
  exit 1
}

source_guard() {
  test -r "${ALLOWED_SIGNERS}" || die "missing allowed-signers file"
  test "$(sha256sum "${ALLOWED_SIGNERS}" | awk '{print $1}')" = "${ALLOWED_SIGNERS_SHA256}" || die "allowed-signers SHA256 drift"
  test -e "${SOURCE_ROOT}/.git" || die "missing product source"
  test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${PRODUCT_SHA}" || die "product source SHA drift"
  test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || die "product source is dirty"
  if git -C "${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "product source has unresolved submodule gitlinks"
  fi
  test -z "$(git -C "${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "product source submodule is dirty"
  git -C "${SOURCE_ROOT}" verify-commit HEAD >/dev/null || die "product commit signature is invalid"
}

source_guard
python3 "${ARTIFACT_DIR}/harness.py" segment-preflight \
  --arm "${VARIANT}" \
  --endpoint "${SEGMENT_STOP_STEP}" \
  --result-dir "${RESULT_DIR}" \
  --harness-sha "${HARNESS_SHA}" \
  --product-sha "${PRODUCT_SHA}"

test -r "${CONTAINER}" || die "missing immutable container"
if [[ "${SEGMENT_STOP_STEP}" == 25 ]]; then
  test "$(sha256sum "${CONTAINER}" | awk '{print $1}')" = "${CONTAINER_SHA256}" || die "container SHA256 drift"
fi

python3 "${ARTIFACT_DIR}/verify_dapo_slice.py" \
  --source "${DATA_SOURCE}" \
  --output "${DATASET}" \
  --identity-file "${ARTIFACT_DIR}/dataset_identity.json" \
  --verify-only
model_sha_args=()
if [[ "${SEGMENT_STOP_STEP}" == 25 ]]; then
  model_sha_args+=(--verify-content-sha)
fi
python3 "${ARTIFACT_DIR}/verify_model_identity.py" \
  --artifact target \
  --root "${TARGET}" \
  --identity-file "${ARTIFACT_DIR}/checkpoint_identity.json" \
  "${model_sha_args[@]}"
if [[ "${VARIANT}" != baseline-k0 ]]; then
  : "${METHOD:?}"
  : "${CHECKPOINT:?}"
  python3 "${ARTIFACT_DIR}/check_checkpoint_state_dict.py" \
    --variant "${METHOD}" \
    --checkpoint "${CHECKPOINT}" \
    --identity-file "${ARTIFACT_DIR}/checkpoint_identity.json" \
    "${model_sha_args[@]}"
fi

touch "${train_log}"
set +e
(
  set -o pipefail
  cd "${SOURCE_ROOT}"
  uv run examples/run_grpo.py \
    --config "${CONFIG}" \
    "grpo.segment_stop_step=${SEGMENT_STOP_STEP}" \
    logger.log_dir="${RESULT_DIR}/logs" \
    logger.wandb_enabled=true \
    logger.wandb.project=sna-specdec \
    logger.wandb.name="${WANDB_RUN_ID}" \
    2>&1 | tee "${train_log}"
)
training_status=$?
set -e
test "${training_status}" -eq 0 || die "training exited ${training_status}"

python3 "${ARTIFACT_DIR}/harness.py" runtime-gates \
  --arm "${VARIANT}" \
  --endpoint "${SEGMENT_STOP_STEP}" \
  --log "${train_log}" \
  --output "${gates}"

segment_start=1
if [[ "${SEGMENT_STOP_STEP}" != 25 ]]; then
  segment_start=$((SEGMENT_STOP_STEP - 24))
fi
mapfile -t expected_steps < <(seq "${segment_start}" "${SEGMENT_STOP_STEP}")
python3 "${ARTIFACT_DIR}/summarize_output_lengths.py" \
  --log-root "${RESULT_DIR}/logs" \
  --output "${ARTIFACT_DIR}/output-length-metrics.json" \
  --max-output-length 32768 \
  --expected-steps "${expected_steps[@]}" \
  --expected-samples-per-step 8

python3 "${ARTIFACT_DIR}/harness.py" segment-finalize \
  --arm "${VARIANT}" \
  --endpoint "${SEGMENT_STOP_STEP}" \
  --result-dir "${RESULT_DIR}" \
  --harness-sha "${HARNESS_SHA}" \
  --product-sha "${PRODUCT_SHA}"
