#!/usr/bin/env bash
set -euo pipefail

# Poll a submitted Eagle3 hidden-state/train/export pipeline and refresh the
# no-submit reports after all recorded Slurm job ids leave the queue. This
# script does not submit any jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_eagle3_pipeline_jobs.txt}"
LOGS_DIR="${LOGS_DIR:-$ROOT_DIR/logs}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_POLLS="${MAX_POLLS:-240}"
RUN_OPERATOR_REFRESH="${RUN_OPERATOR_REFRESH:-true}"
RUN_COMPLETION_AUDIT="${RUN_COMPLETION_AUDIT:-true}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
DEFAULT_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
CONTAINER="${CONTAINER:-$DEFAULT_CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
INPUT_DATA="${INPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
HIDDEN_STATES_VALIDATION_JSON="${HIDDEN_STATES_VALIDATION_JSON:-$HIDDEN_STATES_DIR/validation_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
TRAINING_CKPT_VALIDATION_JSON="${TRAINING_CKPT_VALIDATION_JSON:-$REPORT_DIR/eagle3_training_checkpoint.json}"
EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
EXPORT_ARTIFACTS_JSON="${EXPORT_ARTIFACTS_JSON:-$REPORT_DIR/eagle3_export_artifacts.json}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
RUN_PILOT="${RUN_PILOT:-true}"

PIPELINE_ANALYSIS_JSON="${PIPELINE_ANALYSIS_JSON:-$REPORT_DIR/eagle3_pipeline_analysis.json}"
PIPELINE_ANALYSIS_MD="${PIPELINE_ANALYSIS_MD:-$REPORT_DIR/eagle3_pipeline_analysis.md}"
COMPLETION_AUDIT_JSON="${COMPLETION_AUDIT_JSON:-$REPORT_DIR/eagle3_completion_audit.json}"
COMPLETION_AUDIT_MD="${COMPLETION_AUDIT_MD:-$REPORT_DIR/eagle3_completion_audit.md}"
LOCK_FILE="${LOCK_FILE:-$REPORT_DIR/eagle3_pipeline_watch.lock}"

mkdir -p "$REPORT_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date)] another pipeline watcher holds lock: $LOCK_FILE"
  exit 0
fi

cd "$ROOT_DIR"

job_ids() {
  if [[ ! -f "$JOB_FILE" ]]; then
    return 0
  fi
  awk -F= '/_job=/ {print $2}' "$JOB_FILE" | sed -E 's/[^0-9].*$//' | awk 'NF' | sort -u
}

run_pipeline_analysis() {
  python3 "$SCRIPT_DIR/analyze_eagle3_pipeline.py" \
    --job-file "$JOB_FILE" \
    --logs-dir "$LOGS_DIR" \
    --base-model "$BASE_MODEL" \
    --modelopt-dir "$MODELOPT_DIR" \
    --verifier-config-dir "$VERIFIER_CONFIG_DIR" \
    --reference-arch "$REFERENCE_ARCH" \
    --arch-env-file "$ARCH_ENV_FILE" \
    --chat-template "$CHAT_TEMPLATE" \
    --container "$CONTAINER" \
    --mounts "$MOUNTS" \
    --input-data "$INPUT_DATA" \
    --hidden-states-dir "$HIDDEN_STATES_DIR" \
    --hidden-validation-json "$HIDDEN_STATES_VALIDATION_JSON" \
    --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --export-dir "$EXPORT_DIR" \
    --vllm-draft-dir "$VLLM_DRAFT_DIR" \
    --export-artifacts-json "$EXPORT_ARTIFACTS_JSON" \
    --sbatch-account "$SBATCH_ACCOUNT" \
    --sbatch-partition "$SBATCH_PARTITION" \
    --run-pilot "$RUN_PILOT" \
    --markdown-out "$PIPELINE_ANALYSIS_MD" \
    --json-out "$PIPELINE_ANALYSIS_JSON"
}

run_completion_audit() {
  python3 "$SCRIPT_DIR/audit_eagle3_completion.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --markdown-out "$COMPLETION_AUDIT_MD" \
    --json-out "$COMPLETION_AUDIT_JSON"
}

run_operator_refresh() {
  python3 "$SCRIPT_DIR/refresh_eagle3_operator_state.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --json-out "$REPORT_DIR/eagle3_operator_state_refresh.json" \
    --markdown-out "$REPORT_DIR/eagle3_operator_state_refresh.md"
}

echo "[$(date)] pipeline watcher start job_file=$JOB_FILE"
for _ in $(seq 1 "$MAX_POLLS"); do
  mapfile -t ids < <(job_ids)
  if [[ "${#ids[@]}" -eq 0 ]]; then
    echo "[$(date)] no numeric pipeline job ids visible yet"
    sleep "$POLL_SECONDS"
    continue
  fi

  joined="$(IFS=,; echo "${ids[*]}")"
  active="$(squeue -j "$joined" -h -o "%i|%T|%R" 2>/dev/null || true)"
  if [[ -n "$active" ]]; then
    echo "[$(date)] pipeline jobs still active:"
    printf '%s\n' "$active"
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "[$(date)] pipeline jobs no longer in squeue; refreshing analysis"
  run_pipeline_analysis
  if [[ "$RUN_COMPLETION_AUDIT" == "true" || "$RUN_COMPLETION_AUDIT" == "True" ]]; then
    run_completion_audit
  fi
  if [[ "$RUN_OPERATOR_REFRESH" == "true" || "$RUN_OPERATOR_REFRESH" == "True" ]]; then
    run_operator_refresh
  fi
  echo "[$(date)] pipeline watcher completed"
  exit 0
done

echo "[$(date)] pipeline watcher timeout before terminal state"
exit 2
