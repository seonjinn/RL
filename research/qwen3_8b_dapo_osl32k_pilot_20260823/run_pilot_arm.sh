#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?}"
: "${SOURCE_SHA:?}"
: "${ARTIFACT_DIR:?}"
: "${CONFIG:?}"
: "${DATA_SOURCE:?}"
: "${DATASET:?}"
: "${TARGET:?}"
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

assert_step2_refit_window() {
  python3 - "${train_log}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
start = re.search(r"Step\s+2\s*/\s*2", text)
end = re.search(r"Logged data to .*train_data_step2\.jsonl", text)
if start is None or end is None or end.start() <= start.end():
    raise SystemExit("missing durable Step-2 log window")
window = text[start.end():end.start()]
required = {
    "refit": r"GPU Memory after refit complete",
    "wake_weights": r"wake up tags \['weights'\]",
    "wake_kv": r"wake up tags \['kv_cache'\]",
}
missing = [name for name, pattern in required.items() if re.search(pattern, window) is None]
if missing:
    raise SystemExit(f"missing Step-2 refit/wake evidence: {missing}")
print("STEP2_WAKE_REFIT_WINDOW_GATE_PASS")
PY
}

source_guard
echo SOURCE_CLEAN_GATE_PASS | tee "${ARTIFACT_DIR}/gates.log"
python3 "${ARTIFACT_DIR}/verify_dapo_slice.py" \
  --source "${DATA_SOURCE}" \
  --output "${DATASET}" \
  --identity-file "${ARTIFACT_DIR}/dataset_identity.json" \
  --verify-only | tee -a "${ARTIFACT_DIR}/gates.log"
echo DATA_IDENTITY_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
python3 "${ARTIFACT_DIR}/verify_model_identity.py" \
  --artifact target \
  --root "${TARGET}" \
  --identity-file "${CHECKPOINT_IDENTITY}" \
  --verify-content-sha | tee -a "${ARTIFACT_DIR}/gates.log"
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
wait "${train_pid}"
check_fatal
python3 "${ARTIFACT_DIR}/summarize_output_lengths.py" \
  --log-root "${ARTIFACT_DIR}/logs" \
  --output "${ARTIFACT_DIR}/output-length-metrics.json" \
  --max-output-length 32768 \
  --expected-steps 1 2 \
  --expected-samples-per-step 8 | tee -a "${ARTIFACT_DIR}/gates.log"
echo OUTPUT_LENGTH_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
grep -qE 'Logged data to .*train_data_step1\.jsonl' "${train_log}" || die "missing durable Step-1 completion evidence"
echo STEP1_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
grep -qE 'Logged data to .*train_data_step2\.jsonl' "${train_log}" || die "missing durable Step-2 completion evidence"
echo STEP2_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
assert_step2_refit_window | tee -a "${ARTIFACT_DIR}/gates.log"
echo WAKE_REFIT_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
echo NO_FATAL_GATE_PASS | tee -a "${ARTIFACT_DIR}/gates.log"
