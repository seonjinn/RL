#!/usr/bin/env bash
set -euo pipefail

# Poll a rollout-capture Slurm job. After it leaves squeue, analyze the
# train_data_step artifacts, materialize conversations when possible, then
# refresh corpus/training-scale/operator reports. This watcher is generic and
# does not submit the hidden-state/train/export pipeline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
SWE_REPO_ROOT="${SWE_REPO_ROOT:-${REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}}"
JOB_ID="${JOB_ID:?set JOB_ID}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:?set ROLLOUT_LOG_DIR}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:?set OUTPUT_CONVERSATIONS}"
REPORT_PREFIX="${REPORT_PREFIX:-rollout_capture_${JOB_ID}}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_POLLS="${MAX_POLLS:-240}"
WAIT_FOR_LOCK="${WAIT_FOR_LOCK:-false}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
DUMP_GPUS_PER_NODE="${DUMP_GPUS_PER_NODE:-$TRAIN_GPUS_PER_NODE}"
EXPORT_GPUS_PER_NODE="${EXPORT_GPUS_PER_NODE:-1}"
TP="${TP:-4}"
PROMOTE_TO_CANONICAL="${PROMOTE_TO_CANONICAL:-true}"
RUN_PIPELINE_PREFLIGHT="${RUN_PIPELINE_PREFLIGHT:-true}"
AUTO_SUBMIT_PIPELINE="${AUTO_SUBMIT_PIPELINE:-false}"
RUN_OPERATOR_REFRESH="${RUN_OPERATOR_REFRESH:-true}"
RUN_FULL_ROLLOUT_GATE="${RUN_FULL_ROLLOUT_GATE:-true}"
AUTO_SUBMIT_FULL_ROLLOUT="${AUTO_SUBMIT_FULL_ROLLOUT:-false}"
ALLOW_FULL_ROLLOUT_HEAVY_GPU="${ALLOW_FULL_ROLLOUT_HEAVY_GPU:-${ALLOW_HEAVY_GPU:-false}}"
START_FULL_ROLLOUT_WATCHER="${START_FULL_ROLLOUT_WATCHER:-true}"
ALLOW_FULL_ROLLOUT_BACKGROUND="${ALLOW_FULL_ROLLOUT_BACKGROUND:-true}"
RUN_PENDING_STATE_REFRESH="${RUN_PENDING_STATE_REFRESH:-true}"
PENDING_STATE_REFRESH_POLLS="${PENDING_STATE_REFRESH_POLLS:-5}"
INFER_FLAT_CONTENT_ROLES="${INFER_FLAT_CONTENT_ROLES:-false}"
COMPACT_CURRENT_TURN="${COMPACT_CURRENT_TURN:-false}"
INCLUDE_REASONING_CONTENT="${INCLUDE_REASONING_CONTENT:-false}"
MIN_ASSISTANT_CHARS="${MIN_ASSISTANT_CHARS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-modelopt}"
TARGET_CONTEXT="${TARGET_CONTEXT:-${EAGLE3_TARGET_CONTEXT:-swe_rl}}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
TRAINED_CKPT="${TRAINED_CKPT:-$OUTPUT_DIR}"
EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}"
CONTAINER_PREFLIGHT_JSON="${CONTAINER_PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/container_preflight_analysis.json}"
DEFAULT_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
CONTAINER="${CONTAINER:-$DEFAULT_CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"

REPORT_DIR="$ARTIFACT_ROOT/reports"
JOB_JSON="$REPORT_DIR/${REPORT_PREFIX}_job_analysis.json"
JOB_MD="$REPORT_DIR/${REPORT_PREFIX}_job_analysis.md"
ARTIFACT_JSON="$REPORT_DIR/${REPORT_PREFIX}_analysis.json"
ARTIFACT_MD="$REPORT_DIR/${REPORT_PREFIX}_analysis.md"
CORPUS_JSON="$REPORT_DIR/${REPORT_PREFIX}_corpus_strategy.json"
CORPUS_MD="$REPORT_DIR/${REPORT_PREFIX}_corpus_strategy.md"
SCALE_JSON="$REPORT_DIR/${REPORT_PREFIX}_training_scale.json"
SCALE_MD="$REPORT_DIR/${REPORT_PREFIX}_training_scale.md"
PIPELINE_PREFLIGHT_JSON="$REPORT_DIR/${REPORT_PREFIX}_pipeline_submit_preflight.json"
PIPELINE_PREFLIGHT_MD="$REPORT_DIR/${REPORT_PREFIX}_pipeline_submit_preflight.md"
CANONICAL_SCALE_JSON="$REPORT_DIR/eagle3_training_scale.json"
CANONICAL_SCALE_MD="$REPORT_DIR/eagle3_training_scale.md"
if [[ -z "${CANONICAL_OUTPUT_CONVERSATIONS:-}" ]]; then
  if [[ "$TARGET_CONTEXT" == "math" ]]; then
    CANONICAL_OUTPUT_CONVERSATIONS="$ARTIFACT_ROOT/data/qwen3_235b_math_rollout_conversations.jsonl"
  else
    CANONICAL_OUTPUT_CONVERSATIONS="$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl"
  fi
fi
if [[ -z "${STATE_JSON:-}" ]]; then
  if [[ "$OUTPUT_CONVERSATIONS" == "$CANONICAL_OUTPUT_CONVERSATIONS" ]]; then
    STATE_JSON="$REPORT_DIR/rollout_capture_state_advance.json"
  else
    STATE_JSON="$REPORT_DIR/${REPORT_PREFIX}_state_advance.json"
  fi
fi
if [[ -z "${STATE_MD:-}" ]]; then
  STATE_MD="${STATE_JSON%.json}.md"
fi
if [[ -z "${PROMOTION_MARKER:-}" ]]; then
  if [[ "$TARGET_CONTEXT" == "math" ]]; then
    PROMOTION_MARKER="$REPORT_DIR/rollout_capture_math_canonical_promotion.json"
  else
    PROMOTION_MARKER="$REPORT_DIR/rollout_capture_canonical_promotion.json"
  fi
fi
if [[ -z "${PROMOTION_LOCK:-}" ]]; then
  PROMOTION_LOCK="${PROMOTION_MARKER%.json}.lock"
fi
LOCK_FILE="$REPORT_DIR/${REPORT_PREFIX}_watch.lock"
WATCH_PID_FILE="${WATCH_PID_FILE:-$REPORT_DIR/${REPORT_PREFIX}_watch.pid}"

STATE_REPORT_PREFIX_ARGS=()
if [[ "$STATE_JSON" != "$REPORT_DIR/rollout_capture_state_advance.json" ]]; then
  STATE_REPORT_PREFIX_ARGS=(--report-prefix "$REPORT_PREFIX")
fi

mkdir -p "$REPORT_DIR"

exec 9>"$LOCK_FILE"
if [[ "$WAIT_FOR_LOCK" == "true" || "$WAIT_FOR_LOCK" == "True" || "$WAIT_FOR_LOCK" == "1" ]]; then
  echo "[$(date)] waiting for watcher lock: $LOCK_FILE"
  flock 9
else
  if ! flock -n 9; then
    echo "[$(date)] another watcher holds lock: $LOCK_FILE"
    exit 0
  fi
fi
printf '%s\n' "$$" > "$WATCH_PID_FILE"
cleanup_watch_pid() {
  if [[ -f "$WATCH_PID_FILE" ]] && [[ "$(cat "$WATCH_PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$WATCH_PID_FILE"
  fi
}
trap cleanup_watch_pid EXIT

cd "$ROOT_DIR"

json_value() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
path, key = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, encoding="utf-8"))
    print(data.get(key, ""))
except Exception:
    print("")
PY
}

run_job_analysis() {
  python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --repo-root "$SWE_REPO_ROOT" \
    --job-id "$JOB_ID" \
    --rollout-log-dir "$ROLLOUT_LOG_DIR" \
    --output-data "$OUTPUT_CONVERSATIONS" \
    --json-out "$JOB_JSON" \
    --markdown-out "$JOB_MD"
}

run_artifact_analysis() {
  python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --rollout-log-dir "$ROLLOUT_LOG_DIR" \
    --output-data "$OUTPUT_CONVERSATIONS" \
    --json-out "$ARTIFACT_JSON" \
    --markdown-out "$ARTIFACT_MD"
}

run_post_materialize_reports() {
  python3 experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --target-context "$TARGET_CONTEXT" \
    --input-data "$OUTPUT_CONVERSATIONS" \
    --rollout-capture-analysis-json "$ARTIFACT_JSON" \
    --json-out "$CORPUS_JSON" \
    --markdown-out "$CORPUS_MD"

  TRAIN_GPUS_PER_NODE="$TRAIN_GPUS_PER_NODE" \
    INPUT_DATA="$OUTPUT_CONVERSATIONS" \
    python3 experiments/eagle3_qwen3_235b/estimate_eagle3_training_scale.py \
      --artifact-root "$ARTIFACT_ROOT" \
      --input-data "$OUTPUT_CONVERSATIONS" \
      --target-context "$TARGET_CONTEXT" \
      --corpus-strategy-json "$CORPUS_JSON" \
      --json-out "$SCALE_JSON" \
      --markdown-out "$SCALE_MD"
}

promote_canonical_reports() {
  python3 experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --repo-root "$SWE_REPO_ROOT" \
    --job-id "$JOB_ID" \
    --rollout-log-dir "$ROLLOUT_LOG_DIR" \
    --output-data "$OUTPUT_CONVERSATIONS" \
    --target-context "$TARGET_CONTEXT" \
    --json-out "$REPORT_DIR/rollout_capture_state_advance.json" \
    --markdown-out "$REPORT_DIR/rollout_capture_state_advance.md"
}

write_promotion_marker() {
  python3 - "$PROMOTION_MARKER" "$JOB_ID" "$REPORT_PREFIX" "$ROLLOUT_LOG_DIR" "$OUTPUT_CONVERSATIONS" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
data = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "job_id": sys.argv[2],
    "report_prefix": sys.argv[3],
    "rollout_log_dir": sys.argv[4],
    "output_data": sys.argv[5],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

promote_canonical_if_unclaimed() {
  exec 8>"$PROMOTION_LOCK"
  flock 8

  if [[ -f "$PROMOTION_MARKER" ]]; then
    existing_output="$(json_value "$PROMOTION_MARKER" output_data)"
    existing_job="$(json_value "$PROMOTION_MARKER" job_id)"
    if [[ "$existing_output" == "$OUTPUT_CONVERSATIONS" ]]; then
      echo "[$(date)] canonical rollout already promoted by this output: job=${existing_job:-unknown}"
      return 0
    fi
    echo "[$(date)] canonical rollout already claimed by job=${existing_job:-unknown} output=${existing_output:-unknown}; skipping promotion for job=$JOB_ID"
    return 1
  fi

  echo "[$(date)] promoting rollout corpus to canonical reports"
  promote_canonical_reports
  write_promotion_marker
}

run_pipeline_submit_preflight() {
  python3 experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --input-data "$OUTPUT_CONVERSATIONS" \
    --hidden-states-dir "$HIDDEN_STATES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --trained-ckpt "$TRAINED_CKPT" \
    --export-dir "$EXPORT_DIR" \
    --vllm-draft-dir "$VLLM_DRAFT_DIR" \
    --verifier-config-dir "$VERIFIER_CONFIG_DIR" \
    --chat-template "$CHAT_TEMPLATE" \
    --modelopt-dir "$MODELOPT_DIR" \
    --reference-arch "$REFERENCE_ARCH" \
    --arch-env-file "$ARCH_ENV_FILE" \
    --container-preflight-json "$CONTAINER_PREFLIGHT_JSON" \
    --corpus-strategy-json "$REPORT_DIR/corpus_strategy.json" \
    --rollout-state-json "$REPORT_DIR/rollout_capture_state_advance.json" \
    --sbatch-account "$SBATCH_ACCOUNT" \
    --sbatch-partition "$SBATCH_PARTITION" \
    --container "$CONTAINER" \
    --mounts "$MOUNTS" \
    --run-pilot true \
    --dump-gpus-per-node "$DUMP_GPUS_PER_NODE" \
    --train-gpus-per-node "$TRAIN_GPUS_PER_NODE" \
    --export-gpus-per-node "$EXPORT_GPUS_PER_NODE" \
    --tp "$TP" \
    --target-context "$TARGET_CONTEXT" \
    --json-out "$PIPELINE_PREFLIGHT_JSON" \
    --markdown-out "$PIPELINE_PREFLIGHT_MD"

  cp "$PIPELINE_PREFLIGHT_JSON" "$REPORT_DIR/eagle3_pipeline_submit_preflight.json"
  cp "$PIPELINE_PREFLIGHT_MD" "$REPORT_DIR/eagle3_pipeline_submit_preflight.md"
}

submit_pipeline_if_ready() {
  python3 experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --preflight-json "$REPORT_DIR/eagle3_pipeline_submit_preflight.json" \
    --json-out "$REPORT_DIR/eagle3_pipeline_gated_submit.json" \
    --markdown-out "$REPORT_DIR/eagle3_pipeline_gated_submit.md" \
    --execute \
    --allow-heavy-gpu
}

run_canonical_training_scale() {
  TRAIN_GPUS_PER_NODE="$TRAIN_GPUS_PER_NODE" \
    INPUT_DATA="$OUTPUT_CONVERSATIONS" \
    python3 experiments/eagle3_qwen3_235b/estimate_eagle3_training_scale.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --input-data "$OUTPUT_CONVERSATIONS" \
    --target-context "$TARGET_CONTEXT" \
      --corpus-strategy-json "$REPORT_DIR/corpus_strategy.json" \
      --pipeline-submit-preflight-json "$REPORT_DIR/eagle3_pipeline_submit_preflight.json" \
      --json-out "$CANONICAL_SCALE_JSON" \
      --markdown-out "$CANONICAL_SCALE_MD"
}

run_operator_refresh() {
  python3 experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --json-out "$REPORT_DIR/eagle3_operator_state_refresh.json" \
    --markdown-out "$REPORT_DIR/eagle3_operator_state_refresh.md"
}

run_full_rollout_gate() {
  full_gate_cmd=(
    python3 experiments/eagle3_qwen3_235b/submit_full_rollout_after_smoke_if_ready.py
    --artifact-root "$ARTIFACT_ROOT" \
    --smoke-job-id "$JOB_ID" \
    --smoke-report-prefix "$REPORT_PREFIX" \
    --json-out "$REPORT_DIR/full_swegym_after_smoke_gate.json" \
    --markdown-out "$REPORT_DIR/full_swegym_after_smoke_gate.md"
  )
  if [[ "$AUTO_SUBMIT_FULL_ROLLOUT" == "true" || "$AUTO_SUBMIT_FULL_ROLLOUT" == "True" ]]; then
    full_gate_cmd+=(--execute)
    if [[ "$ALLOW_FULL_ROLLOUT_HEAVY_GPU" == "true" || "$ALLOW_FULL_ROLLOUT_HEAVY_GPU" == "True" ]]; then
      full_gate_cmd+=(--allow-heavy-gpu)
    fi
    if [[ "$START_FULL_ROLLOUT_WATCHER" == "true" || "$START_FULL_ROLLOUT_WATCHER" == "True" ]]; then
      full_gate_cmd+=(--start-watcher)
      if [[ "$ALLOW_FULL_ROLLOUT_BACKGROUND" == "true" || "$ALLOW_FULL_ROLLOUT_BACKGROUND" == "True" ]]; then
        full_gate_cmd+=(--allow-background)
      fi
    fi
  fi
  "${full_gate_cmd[@]}"
}

run_pending_state_refresh() {
  python3 experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --repo-root "$SWE_REPO_ROOT" \
    --job-id "$JOB_ID" \
    --rollout-log-dir "$ROLLOUT_LOG_DIR" \
    --output-data "$OUTPUT_CONVERSATIONS" \
    "${STATE_REPORT_PREFIX_ARGS[@]}" \
    --json-out "$STATE_JSON" \
    --markdown-out "$STATE_MD"
}

echo "[$(date)] watcher start job=$JOB_ID prefix=$REPORT_PREFIX poll_seconds=$POLL_SECONDS max_polls=$MAX_POLLS"
poll_count=0
for _ in $(seq 1 "$MAX_POLLS"); do
  poll_count=$((poll_count + 1))
  state="$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)"
  if [[ -n "$state" ]]; then
    start_time="$(squeue -j "$JOB_ID" -h -o "%S" 2>/dev/null || true)"
    reason="$(squeue -j "$JOB_ID" -h -o "%R" 2>/dev/null || true)"
    echo "[$(date)] job=$JOB_ID active state=$state start=${start_time:-unknown} reason=${reason:-unknown}"
    if [[ "$RUN_PENDING_STATE_REFRESH" == "true" || "$RUN_PENDING_STATE_REFRESH" == "True" ]]; then
      if (( PENDING_STATE_REFRESH_POLLS > 0 && (poll_count == 1 || poll_count % PENDING_STATE_REFRESH_POLLS == 0) )); then
        echo "[$(date)] refreshing pending rollout state report: $STATE_JSON"
        run_pending_state_refresh || echo "[$(date)] pending state refresh failed for job=$JOB_ID"
      fi
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "[$(date)] job=$JOB_ID no longer in squeue; analyzing artifacts"
  run_job_analysis
  run_artifact_analysis

  artifact_status="$(json_value "$ARTIFACT_JSON" overall_status)"
  if [[ "$artifact_status" == "needs_materialize" ]]; then
    echo "[$(date)] materializing rollout corpus: $OUTPUT_CONVERSATIONS"
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
      ROLLOUT_LOG_DIR="$ROLLOUT_LOG_DIR" \
      OUTPUT_DATA="$OUTPUT_CONVERSATIONS" \
      INFER_FLAT_CONTENT_ROLES="$INFER_FLAT_CONTENT_ROLES" \
      COMPACT_CURRENT_TURN="$COMPACT_CURRENT_TURN" \
      INCLUDE_REASONING_CONTENT="$INCLUDE_REASONING_CONTENT" \
      MIN_ASSISTANT_CHARS="$MIN_ASSISTANT_CHARS" \
      MAX_SEQ_LEN="$MAX_SEQ_LEN" \
      OUTPUT_SCHEMA="$OUTPUT_SCHEMA" \
      bash experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh
    run_job_analysis
    run_artifact_analysis
    artifact_status="$(json_value "$ARTIFACT_JSON" overall_status)"
  fi

  if [[ "$artifact_status" == "pass" ]]; then
    echo "[$(date)] rollout corpus pass; refreshing corpus and scale reports"
    run_post_materialize_reports
    promoted_canonical=false
    if [[ "$PROMOTE_TO_CANONICAL" == "true" || "$PROMOTE_TO_CANONICAL" == "True" ]]; then
      if promote_canonical_if_unclaimed; then
        promoted_canonical=true
      fi
    fi
    if [[ "$RUN_PIPELINE_PREFLIGHT" == "true" || "$RUN_PIPELINE_PREFLIGHT" == "True" ]] && [[ "$promoted_canonical" == "true" ]]; then
      echo "[$(date)] running no-submit pipeline submit preflight"
      run_pipeline_submit_preflight
      if [[ "$AUTO_SUBMIT_PIPELINE" == "true" || "$AUTO_SUBMIT_PIPELINE" == "True" ]]; then
        echo "[$(date)] AUTO_SUBMIT_PIPELINE=true; submitting pilot pipeline through gated helper"
        submit_pipeline_if_ready
      fi
    elif [[ "$RUN_PIPELINE_PREFLIGHT" == "true" || "$RUN_PIPELINE_PREFLIGHT" == "True" ]]; then
      echo "[$(date)] skipping pipeline submit preflight because this rollout did not claim canonical promotion"
    fi
    if [[ "$PROMOTE_TO_CANONICAL" == "true" || "$PROMOTE_TO_CANONICAL" == "True" ]] && [[ "$promoted_canonical" == "true" ]]; then
      echo "[$(date)] refreshing canonical training scale"
      run_canonical_training_scale
    fi
  else
    echo "[$(date)] rollout artifact status after analysis: ${artifact_status:-unknown}"
  fi

  if [[ "$RUN_OPERATOR_REFRESH" == "true" || "$RUN_OPERATOR_REFRESH" == "True" ]]; then
    echo "[$(date)] refreshing operator next-action state"
    run_operator_refresh
  fi

  if [[ "$RUN_FULL_ROLLOUT_GATE" == "true" || "$RUN_FULL_ROLLOUT_GATE" == "True" ]]; then
    echo "[$(date)] refreshing full SWE-Gym rollout gate"
    if [[ "$AUTO_SUBMIT_FULL_ROLLOUT" == "true" || "$AUTO_SUBMIT_FULL_ROLLOUT" == "True" ]]; then
      echo "[$(date)] AUTO_SUBMIT_FULL_ROLLOUT=true; gate may submit only if smoke PASS, full preflight PASS, and allow flag is set"
    fi
    run_full_rollout_gate || echo "[$(date)] full SWE-Gym rollout gate refresh failed"
    if [[ "$AUTO_SUBMIT_FULL_ROLLOUT" == "true" || "$AUTO_SUBMIT_FULL_ROLLOUT" == "True" ]] && [[ "$RUN_OPERATOR_REFRESH" == "true" || "$RUN_OPERATOR_REFRESH" == "True" ]]; then
      echo "[$(date)] refreshing operator next-action state after full rollout gate"
      run_operator_refresh || echo "[$(date)] post-full-gate operator refresh failed"
    fi
  fi

  echo "[$(date)] watcher completed job=$JOB_ID"
  exit 0
done

echo "[$(date)] watcher timeout before terminal state: job=$JOB_ID"
exit 2
