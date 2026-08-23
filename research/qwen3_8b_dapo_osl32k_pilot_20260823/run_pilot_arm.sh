#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?}"
: "${SOURCE_SHA:?}"
: "${ARTIFACT_DIR:?}"
: "${CONFIG:?}"
: "${DATA_SOURCE:?}"
: "${DATASET:?}"
: "${VARIANT:?}"
: "${WANDB_ID:?}"

readonly CAPTURE_SIZES='[1,2,4,6,8,12,16,18,24,30,32,36,40,42,48,56,64]'
train_log="${ARTIFACT_DIR}/train.log"

die() { echo "Q8_DAPO32K_FAIL_CLOSED: $*" >&2; exit 1; }

source_guard() {
  test -e "${SOURCE_ROOT}/.git" || die "missing product source"
  test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}" || die "product source SHA drift"
  test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || die "product source is dirty"
  if git -C "${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "product source has unresolved submodule gitlinks"
  fi
  test -z "$(git -C "${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "product source submodule is dirty"
}

check_fatal() {
  if grep -qEi 'CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "${train_log}"; then
    kill -- "-${train_pid}" 2>/dev/null || true
    wait "${train_pid}" || true
    die "fatal pattern found: out of memory, nan, or traceback"
  fi
}

wait_for_gate() {
  local pattern="$1" marker="$2" deadline="$((SECONDS + 21600))"
  while kill -0 "${train_pid}" 2>/dev/null; do
    check_fatal
    if grep -qE "${pattern}" "${train_log}"; then
      echo "${marker}" | tee -a "${ARTIFACT_DIR}/gates.log"
      return
    fi
    if (( SECONDS >= deadline )); then
      kill -- "-${train_pid}" 2>/dev/null || true
      wait "${train_pid}" || true
      die "timed out waiting for ${marker}"
    fi
    sleep 10
  done
  wait "${train_pid}" || die "training ended before ${marker}"
  grep -qE "${pattern}" "${train_log}" || die "missing ${marker}"
  echo "${marker}" | tee -a "${ARTIFACT_DIR}/gates.log"
}

require_count() {
  local pattern="$1" expected="$2" marker="$3" count
  count="$(grep -Ec "${pattern}" "${train_log}" || true)"
  (( count >= expected )) || die "${marker} count ${count} < ${expected}"
}

source_guard
echo SOURCE_CLEAN_GATE_PASS | tee "${ARTIFACT_DIR}/gates.log"
python3 "${ARTIFACT_DIR}/verify_dapo_slice.py" \
  --source "${DATA_SOURCE}" \
  --output "${DATASET}" \
  --identity-file "${ARTIFACT_DIR}/dataset_identity.json" \
  --verify-only | tee -a "${ARTIFACT_DIR}/gates.log"
echo DATA_IDENTITY_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
python3 "${ARTIFACT_DIR}/verify_pilot_config.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${CONFIG}" \
  --capture-sizes "${CAPTURE_SIZES}" | tee "${ARTIFACT_DIR}/compose.json"
echo CONFIG_COMPOSE_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
if [[ "${VARIANT}" != baseline-k0 ]]; then
  : "${METHOD:?}"
  : "${CHECKPOINT:?}"
  : "${CHECKPOINT_IDENTITY:?}"
  python3 "${ARTIFACT_DIR}/check_checkpoint_state_dict.py" \
    --variant "${METHOD}" \
    --checkpoint "${CHECKPOINT}" \
    --identity-file "${CHECKPOINT_IDENTITY}" \
    --verify-content-sha | tee -a "${ARTIFACT_DIR}/gates.log"
fi

export WANDB_RUN_ID="${WANDB_ID}"
touch "${train_log}"
setsid bash -c "set -o pipefail; cd '${SOURCE_ROOT}'; NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py --config '${CONFIG}' ++policy.generation.vllm_kwargs.max_num_seqs=8 ++policy.generation.vllm_kwargs.compilation_config.backend=eager ++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE ++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[1,2,4,6,8,12,16,18,24,30,32,36,40,42,48,56,64] logger.log_dir='${ARTIFACT_DIR}/logs' logger.wandb_enabled=True logger.wandb.project=sna-specdec logger.wandb.name='${WANDB_ID}' 2>&1 | tee '${train_log}'" &
train_pid=$!
wait_for_gate 'Capturing CUDA graphs.*100%|Graph capturing finished' CUDAGRAPH_GATE_PASS
wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*2' STEP1_GATE_PASS
wait_for_gate 'Step[[:space:]]+2[[:space:]]*/[[:space:]]*2' STEP2_GATE_PASS
wait "${train_pid}"
check_fatal
require_count 'GPU Memory after refit complete' 2 REFIT
require_count "wake up tags \['weights'\]" 2 WAKE_WEIGHTS
require_count "wake up tags \['kv_cache'\]" 2 WAKE_KV
echo WAKE_REFIT_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
python3 "${ARTIFACT_DIR}/summarize_output_lengths.py" \
  --log-root "${ARTIFACT_DIR}/logs" \
  --output "${ARTIFACT_DIR}/output-length-metrics.json" \
  --max-output-length 32768 \
  --expected-steps 1 2 \
  --expected-samples-per-step 8 | tee -a "${ARTIFACT_DIR}/gates.log"
echo OUTPUT_LENGTH_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
echo NO_FATAL_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
