#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the Qwen3-235B Eagle3 path without doing expensive work by default.
#
# Default behavior:
#   - print/run local data/template prep commands in DRY_RUN mode,
#   - run local preflight with path existence checks skipped,
#   - print the Slurm pipeline plan with SUBMIT=false,
#   - write a readiness audit under ARTIFACT_ROOT/reports.
#
# Minimal dry-run:
#   ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \
#     bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh
#
# Real local data/template prep, followed by Slurm pipeline dry-run:
#   PREP_DRY_RUN=false MODE=rollout INPUT_PATHS="/path/rollouts.jsonl" \
#   TOKENIZER_CONFIG=/path/to/tokenizer_config.json \
#   VERIFIER_CONFIG_DIR=/path/to/Qwen3-235B-A22B-Thinking-2507 \
#   SBATCH_ACCOUNT=<account> \
#     bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh
#
# Actual Slurm submission requires SUBMIT=true and concrete data/template/
# verifier paths. This wrapper intentionally fails early if those are missing.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

step() {
  printf '\n## %s\n' "$1"
}

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/outputs/qwen3_235b_eagle3}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
REVISION="${REVISION:-main}"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"

SUBMIT="${SUBMIT:-false}"
RUN_PILOT="${RUN_PILOT:-false}"
PREP_DRY_RUN="${PREP_DRY_RUN:-true}"
RUN_TEMPLATE_PREP="${RUN_TEMPLATE_PREP:-true}"
RUN_STATIC_INPUT_PREP="${RUN_STATIC_INPUT_PREP:-auto}"
RUN_ARCH_DERIVE="${RUN_ARCH_DERIVE:-auto}"
RUN_DATA_PREP="${RUN_DATA_PREP:-true}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
RUN_PIPELINE="${RUN_PIPELINE:-true}"
RUN_PIPELINE_SUBMIT_PREFLIGHT="${RUN_PIPELINE_SUBMIT_PREFLIGHT:-true}"
RUN_TRAINING_SCALE_PLAN="${RUN_TRAINING_SCALE_PLAN:-true}"
RUN_NEXT_ACTION_PLAN="${RUN_NEXT_ACTION_PLAN:-true}"
RUN_AUDIT="${RUN_AUDIT:-true}"
RUN_PROVENANCE="${RUN_PROVENANCE:-true}"
RUN_TRAINED_DRAFT_SMOKE="${RUN_TRAINED_DRAFT_SMOKE:-false}"
RUN_TRAINED_DRAFT_SWEEP="${RUN_TRAINED_DRAFT_SWEEP:-false}"

DATA_MODE="${DATA_MODE:-${MODE:-discover}}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"

INPUT_DATA="${INPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
TRAINED_CKPT="${TRAINED_CKPT:-$OUTPUT_DIR}"
EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
ARCH_DIR="${ARCH_DIR:-$ARTIFACT_ROOT/architecture}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$ARCH_DIR/eagle3_architecture.json}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-$ARCH_DIR/eagle3_architecture.env}"
ARCH_DOTLIST="${ARCH_DOTLIST:-$ARCH_DIR/eagle3_architecture.dotlist}"

REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
CONVERSATION_VALIDATION_JSON="${CONVERSATION_VALIDATION_JSON:-${INPUT_DATA%.jsonl}.validation.json}"
DISCOVERY_JSON="${DISCOVERY_JSON:-${INPUT_DATA%.jsonl}.discovery.json}"
DISCOVERY_MARKDOWN="${DISCOVERY_MARKDOWN:-${INPUT_DATA%.jsonl}.discovery.md}"
CHAT_TEMPLATE_VALIDATION_JSON="${CHAT_TEMPLATE_VALIDATION_JSON:-${CHAT_TEMPLATE%.jinja2}.mask_validation.json}"
READINESS_JSON="${READINESS_JSON:-$REPORT_DIR/eagle3_readiness.json}"
READINESS_MARKDOWN="${READINESS_MARKDOWN:-$REPORT_DIR/eagle3_readiness.md}"
CONTAINER_PREFLIGHT_JSON="${CONTAINER_PREFLIGHT_JSON:-$REPORT_DIR/container_preflight_analysis.json}"
MODELOPT_LOSS_MASK_JSON="${MODELOPT_LOSS_MASK_JSON:-$REPORT_DIR/modelopt_loss_mask_patch.json}"
NEMO_RL_SPECDEC_JSON="${NEMO_RL_SPECDEC_JSON:-$REPORT_DIR/nemo_rl_specdec_integration.json}"
NEMO_RL_DRIFT_JSON="${NEMO_RL_DRIFT_JSON:-$REPORT_DIR/nemo_rl_eagle3_drift.json}"
ROLLOUT_CAPTURE_JSON="${ROLLOUT_CAPTURE_JSON:-$REPORT_DIR/rollout_capture_validation.json}"
ROLLOUT_CAPTURE_ANALYSIS_JSON="${ROLLOUT_CAPTURE_ANALYSIS_JSON:-$REPORT_DIR/rollout_capture_analysis.json}"
ROLLOUT_CAPTURE_JOB_JSON="${ROLLOUT_CAPTURE_JOB_JSON:-$REPORT_DIR/rollout_capture_job_analysis.json}"
ROLLOUT_CAPTURE_JOB_MARKDOWN="${ROLLOUT_CAPTURE_JOB_MARKDOWN:-$REPORT_DIR/rollout_capture_job_analysis.md}"
ROLLOUT_SUBMIT_PREFLIGHT_JSON="${ROLLOUT_SUBMIT_PREFLIGHT_JSON:-$REPORT_DIR/rollout_capture_submit_preflight.json}"
ROLLOUT_STATE_ADVANCE_JSON="${ROLLOUT_STATE_ADVANCE_JSON:-$REPORT_DIR/rollout_capture_state_advance.json}"
PIPELINE_SUBMIT_PREFLIGHT_JSON="${PIPELINE_SUBMIT_PREFLIGHT_JSON:-$REPORT_DIR/eagle3_pipeline_submit_preflight.json}"
PIPELINE_SUBMIT_PREFLIGHT_MARKDOWN="${PIPELINE_SUBMIT_PREFLIGHT_MARKDOWN:-$REPORT_DIR/eagle3_pipeline_submit_preflight.md}"
PIPELINE_ANALYSIS_JSON="${PIPELINE_ANALYSIS_JSON:-$REPORT_DIR/eagle3_pipeline_analysis.json}"
SWEEP_JSON="${SWEEP_JSON:-$REPORT_DIR/trained_draft_spec_tokens_sweep.json}"
CORPUS_STRATEGY_JSON="${CORPUS_STRATEGY_JSON:-$REPORT_DIR/corpus_strategy.json}"
CORPUS_STRATEGY_MARKDOWN="${CORPUS_STRATEGY_MARKDOWN:-$REPORT_DIR/corpus_strategy.md}"
TRAINING_SCALE_JSON="${TRAINING_SCALE_JSON:-$REPORT_DIR/eagle3_training_scale.json}"
TRAINING_CKPT_VALIDATION_JSON="${TRAINING_CKPT_VALIDATION_JSON:-$REPORT_DIR/eagle3_training_checkpoint.json}"
TRAINING_CKPT_VALIDATION_MARKDOWN="${TRAINING_CKPT_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_training_checkpoint.md}"
NEXT_ACTION_PLAN_JSON="${NEXT_ACTION_PLAN_JSON:-$REPORT_DIR/eagle3_next_actions.json}"
NEXT_ACTION_PLAN_MARKDOWN="${NEXT_ACTION_PLAN_MARKDOWN:-$REPORT_DIR/eagle3_next_actions.md}"
NEXT_ACTION_PLAN_VALIDATION_JSON="${NEXT_ACTION_PLAN_VALIDATION_JSON:-$REPORT_DIR/eagle3_next_actions_validation.json}"
NEXT_ACTION_PLAN_VALIDATION_MARKDOWN="${NEXT_ACTION_PLAN_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_next_actions_validation.md}"
NEXT_ACTION_TRANSITIONS_JSON="${NEXT_ACTION_TRANSITIONS_JSON:-$REPORT_DIR/eagle3_next_action_transitions.json}"
NEXT_ACTION_TRANSITIONS_MARKDOWN="${NEXT_ACTION_TRANSITIONS_MARKDOWN:-$REPORT_DIR/eagle3_next_action_transitions.md}"
PROVENANCE_JSON="${PROVENANCE_JSON:-$REPORT_DIR/eagle3_provenance.json}"
PROVENANCE_MARKDOWN="${PROVENANCE_MARKDOWN:-$REPORT_DIR/eagle3_provenance.md}"

ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}"
ROLLOUT_CONVERSATIONS="${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
SWE_REPO_ROOT="${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
ROLLOUT_ROOTS="${ROLLOUT_ROOTS:-}"
INPUT_PATHS="${INPUT_PATHS:-}"
INPUT_DATA_SOURCE="${INPUT_DATA_SOURCE:-}"
PROMPT_DATA="${PROMPT_DATA:-}"

TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-}"
TEMPLATE="${TEMPLATE:-}"
MODEL_OR_TOKENIZER="${MODEL_OR_TOKENIZER:-$BASE_MODEL}"
TOKENIZER="${TOKENIZER:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
NUM_RESPONSES="${NUM_RESPONSES:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-$MAX_SEQ_LEN}"
INCLUDE_METADATA="${INCLUDE_METADATA:-true}"
EAGLE3_TARGET_CONTEXT="${EAGLE3_TARGET_CONTEXT:-swe_rl}"
ALLOW_MISSING_TRANSFORMERS="${ALLOW_MISSING_TRANSFORMERS:-}"
if [[ -z "$ALLOW_MISSING_TRANSFORMERS" ]]; then
  if is_true "$PREP_DRY_RUN"; then
    ALLOW_MISSING_TRANSFORMERS=true
  else
    ALLOW_MISSING_TRANSFORMERS=false
  fi
fi
STATIC_INPUT_SOURCE_DIR="${STATIC_INPUT_SOURCE_DIR:-}"
STATIC_INPUT_FORCE="${STATIC_INPUT_FORCE:-false}"
STATIC_INPUT_SKIP_TEMPLATE_VALIDATION="${STATIC_INPUT_SKIP_TEMPLATE_VALIDATION:-false}"
STATIC_INPUT_MODEL_OR_TOKENIZER="${STATIC_INPUT_MODEL_OR_TOKENIZER:-}"
STATIC_INPUT_JSON="${STATIC_INPUT_JSON:-$REPORT_DIR/qwen3_static_inputs.json}"
STATIC_INPUT_MARKDOWN="${STATIC_INPUT_MARKDOWN:-$REPORT_DIR/qwen3_static_inputs.md}"

PREFLIGHT_SKIP_EXISTING_PATH_CHECKS="${PREFLIGHT_SKIP_EXISTING_PATH_CHECKS:-}"
if [[ -z "$PREFLIGHT_SKIP_EXISTING_PATH_CHECKS" ]]; then
  if is_true "$PREP_DRY_RUN"; then
    PREFLIGHT_SKIP_EXISTING_PATH_CHECKS=true
  else
    PREFLIGHT_SKIP_EXISTING_PATH_CHECKS=false
  fi
fi
PREFLIGHT_REQUIRE_MODELOPT_IMPORT="${PREFLIGHT_REQUIRE_MODELOPT_IMPORT:-false}"
PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK="${PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK:-false}"

DEFAULT_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
CONTAINER="${CONTAINER:-$DEFAULT_CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"

mkdir -p \
  "$ARTIFACT_ROOT/data" \
  "$ARTIFACT_ROOT/templates" \
  "$ARTIFACT_ROOT/hidden_states" \
  "$ARCH_DIR" \
  "$REPORT_DIR"

step "Resolved plan"
cat <<EOF
ARTIFACT_ROOT=$ARTIFACT_ROOT
BASE_MODEL=$BASE_MODEL
MODELOPT_DIR=$MODELOPT_DIR
PREP_DRY_RUN=$PREP_DRY_RUN
SUBMIT=$SUBMIT
DATA_MODE=$DATA_MODE
INPUT_DATA=$INPUT_DATA
CHAT_TEMPLATE=$CHAT_TEMPLATE
HIDDEN_STATES_DIR=$HIDDEN_STATES_DIR
OUTPUT_DIR=$OUTPUT_DIR
EXPORT_DIR=$EXPORT_DIR
VLLM_DRAFT_DIR=$VLLM_DRAFT_DIR
VERIFIER_CONFIG_DIR=$VERIFIER_CONFIG_DIR
REFERENCE_ARCH=$REFERENCE_ARCH
ARCH_ENV_FILE=$ARCH_ENV_FILE
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
RUN_TRAINED_DRAFT_SMOKE=$RUN_TRAINED_DRAFT_SMOKE
RUN_TRAINED_DRAFT_SWEEP=$RUN_TRAINED_DRAFT_SWEEP
RUN_PILOT=$RUN_PILOT
RUN_PROVENANCE=$RUN_PROVENANCE
RUN_STATIC_INPUT_PREP=$RUN_STATIC_INPUT_PREP
EOF

should_static_input_prep=false
case "$RUN_STATIC_INPUT_PREP" in
  true|True|TRUE|1|yes|Yes|YES)
    should_static_input_prep=true
    ;;
  auto|Auto|AUTO)
    if ! is_true "$PREP_DRY_RUN" && [[ ! -s "$VERIFIER_CONFIG_DIR/config.json" && ! -s "$VERIFIER_CONFIG_DIR" ]]; then
      should_static_input_prep=true
    fi
    ;;
  false|False|FALSE|0|no|No|NO)
    should_static_input_prep=false
    ;;
  *)
    echo "Unsupported RUN_STATIC_INPUT_PREP=$RUN_STATIC_INPUT_PREP. Use auto, true, or false." >&2
    exit 1
    ;;
esac

if [[ "$should_static_input_prep" == "true" ]]; then
  step "Qwen3 static input materialization"
  static_input_cmd=(
    python3 "$EXP_DIR/materialize_qwen3_static_inputs.py"
    --artifact-root "$ARTIFACT_ROOT"
    --model "$BASE_MODEL"
    --revision "$REVISION"
    --verifier-config-dir "$VERIFIER_CONFIG_DIR"
    --architecture-dir "$ARCH_DIR"
    --template-out "$CHAT_TEMPLATE"
    --report-json "$STATIC_INPUT_JSON"
    --report-markdown "$STATIC_INPUT_MARKDOWN"
  )
  if [[ -n "$STATIC_INPUT_SOURCE_DIR" ]]; then
    static_input_cmd+=(--source-dir "$STATIC_INPUT_SOURCE_DIR")
  fi
  if is_true "$STATIC_INPUT_FORCE"; then
    static_input_cmd+=(--force)
  fi
  if is_true "$STATIC_INPUT_SKIP_TEMPLATE_VALIDATION"; then
    static_input_cmd+=(--skip-template-validation)
  fi
  if is_true "$ALLOW_MISSING_TRANSFORMERS"; then
    static_input_cmd+=(--allow-missing-transformers)
  fi
  if [[ -n "$STATIC_INPUT_MODEL_OR_TOKENIZER" ]]; then
    static_input_cmd+=(--model-or-tokenizer "$STATIC_INPUT_MODEL_OR_TOKENIZER")
  fi
  print_cmd "${static_input_cmd[@]}"
  if ! "${static_input_cmd[@]}"; then
    echo "WARN: Qwen3 static input materialization failed; inspect $STATIC_INPUT_MARKDOWN" >&2
    if is_true "$SUBMIT" || is_true "$RUN_STATIC_INPUT_PREP"; then
      exit 1
    fi
  fi
  if [[ -z "$TOKENIZER_CONFIG" && -s "$VERIFIER_CONFIG_DIR/tokenizer_config.json" ]]; then
    TOKENIZER_CONFIG="$VERIFIER_CONFIG_DIR/tokenizer_config.json"
  fi
else
  step "Qwen3 static input materialization"
  echo "Skipped because RUN_STATIC_INPUT_PREP=$RUN_STATIC_INPUT_PREP"
fi

if is_true "$RUN_PROVENANCE"; then
  step "Provenance capture"
  provenance_args=(
    python3 "$EXP_DIR/collect_eagle3_provenance.py"
    --artifact-root "$ARTIFACT_ROOT"
    --modelopt-dir "$MODELOPT_DIR"
    --verifier-config-dir "$VERIFIER_CONFIG_DIR"
    --input-data "$INPUT_DATA"
    --hidden-states-dir "$HIDDEN_STATES_DIR"
    --output-dir "$OUTPUT_DIR"
    --export-dir "$EXPORT_DIR"
    --vllm-draft-dir "$VLLM_DRAFT_DIR"
    --json-out "$PROVENANCE_JSON"
    --markdown-out "$PROVENANCE_MARKDOWN"
  )
  print_cmd "${provenance_args[@]}"
  "${provenance_args[@]}"
else
  step "Provenance capture"
  echo "Skipped because RUN_PROVENANCE=$RUN_PROVENANCE"
fi

data_source_available() {
  case "$DATA_MODE" in
    discover) [[ -n "$ROLLOUT_ROOTS" ]] ;;
    rollout) [[ -n "$INPUT_PATHS" ]] ;;
    existing) [[ -n "$INPUT_DATA_SOURCE" ]] ;;
    generate) [[ -n "$PROMPT_DATA" ]] ;;
    *) return 1 ;;
  esac
}

explain_missing_data_source() {
  case "$DATA_MODE" in
    discover) echo "Set ROLLOUT_ROOTS to one or more NeMo-RL output/log directories." ;;
    rollout) echo "Set INPUT_PATHS to one or more rollout JSONL files or directories." ;;
    existing) echo "Set INPUT_DATA_SOURCE to JSONL that already contains assistant responses." ;;
    generate) echo "Set PROMPT_DATA and OPENAI_BASE_URL for fresh Thinking-2507 generations." ;;
    *) echo "Unsupported DATA_MODE=$DATA_MODE. Use discover, rollout, existing, or generate." ;;
  esac
}

should_derive_arch=false
case "$RUN_ARCH_DERIVE" in
  true|True|TRUE|1|yes|Yes|YES)
    should_derive_arch=true
    ;;
  auto|Auto|AUTO)
    if [[ -f "$VERIFIER_CONFIG_DIR/config.json" || -f "$VERIFIER_CONFIG_DIR" ]]; then
      should_derive_arch=true
    fi
    ;;
  false|False|FALSE|0|no|No|NO)
    should_derive_arch=false
    ;;
  *)
    echo "Unsupported RUN_ARCH_DERIVE=$RUN_ARCH_DERIVE. Use auto, true, or false." >&2
    exit 1
    ;;
esac

if [[ "$should_derive_arch" == "true" ]]; then
  step "Eagle3 architecture derivation"
  if [[ ! -f "$VERIFIER_CONFIG_DIR/config.json" && ! -f "$VERIFIER_CONFIG_DIR" ]]; then
    echo "RUN_ARCH_DERIVE=true requires VERIFIER_CONFIG_DIR/config.json or a direct config path: $VERIFIER_CONFIG_DIR" >&2
    exit 1
  fi
  derive_cmd=(
    python3 "$EXP_DIR/derive_eagle3_architecture.py"
    --verifier-config "$VERIFIER_CONFIG_DIR"
    --json-out "$REFERENCE_ARCH"
    --env-out "$ARCH_ENV_FILE"
    --dotlist-out "$ARCH_DOTLIST"
  )
  print_cmd "${derive_cmd[@]}"
  "${derive_cmd[@]}"
else
  step "Eagle3 architecture derivation"
  echo "Skipped because RUN_ARCH_DERIVE=$RUN_ARCH_DERIVE and verifier config is not visible."
fi

pipeline_arch_env_file=""
if [[ -f "$ARCH_ENV_FILE" ]]; then
  pipeline_arch_env_file="$ARCH_ENV_FILE"
fi
pipeline_reference_arch="$EXP_DIR/qwen3_235b_thinking_eagle3_architecture.json"
if [[ -f "$REFERENCE_ARCH" ]]; then
  pipeline_reference_arch="$REFERENCE_ARCH"
fi

if is_true "$RUN_TEMPLATE_PREP"; then
  step "Chat-template prep"
  template_env=(
    env
    DRY_RUN="$PREP_DRY_RUN"
    BASE_MODEL="$BASE_MODEL"
    MODEL_OR_TOKENIZER="$MODEL_OR_TOKENIZER"
    TEMPLATE="$TEMPLATE"
    TOKENIZER_CONFIG="$TOKENIZER_CONFIG"
    OUTPUT_TEMPLATE="$CHAT_TEMPLATE"
    VALIDATION_JSON="$CHAT_TEMPLATE_VALIDATION_JSON"
    ALLOW_MISSING_TRANSFORMERS="$ALLOW_MISSING_TRANSFORMERS"
  )
  print_cmd "${template_env[@]}" bash "$EXP_DIR/prepare_qwen3_chat_template.sh"
  "${template_env[@]}" bash "$EXP_DIR/prepare_qwen3_chat_template.sh"
else
  step "Chat-template prep"
  echo "Skipped because RUN_TEMPLATE_PREP=$RUN_TEMPLATE_PREP"
fi

if is_true "$RUN_DATA_PREP"; then
  step "Training-conversation prep"
  if data_source_available; then
    data_env=(
      env
      DRY_RUN="$PREP_DRY_RUN"
      MODE="$DATA_MODE"
      BASE_MODEL="$BASE_MODEL"
      MODEL_PATH="$BASE_MODEL"
      OUTPUT_DATA="$INPUT_DATA"
      VALIDATION_JSON="$CONVERSATION_VALIDATION_JSON"
      DISCOVERY_JSON="$DISCOVERY_JSON"
      DISCOVERY_MARKDOWN="$DISCOVERY_MARKDOWN"
      ROLLOUT_ROOTS="$ROLLOUT_ROOTS"
      INPUT_PATHS="$INPUT_PATHS"
      INPUT_DATA_SOURCE="$INPUT_DATA_SOURCE"
      PROMPT_DATA="$PROMPT_DATA"
      OPENAI_BASE_URL="$OPENAI_BASE_URL"
      OPENAI_API_KEY="$OPENAI_API_KEY"
      NUM_RESPONSES="$NUM_RESPONSES"
      TEMPERATURE="$TEMPERATURE"
      TOP_P="$TOP_P"
      MAX_SEQ_LEN="$MAX_SEQ_LEN"
      MAX_TOKENS="$MAX_TOKENS"
      CHAT_TEMPLATE="$CHAT_TEMPLATE"
      TOKENIZER="$TOKENIZER"
      INCLUDE_METADATA="$INCLUDE_METADATA"
    )
    print_cmd "${data_env[@]}" bash "$EXP_DIR/prepare_training_conversations.sh"
    "${data_env[@]}" bash "$EXP_DIR/prepare_training_conversations.sh"
  else
    explain_missing_data_source
    echo "Skipping data prep; existing INPUT_DATA can still be used if it already exists."
  fi
else
  step "Training-conversation prep"
  echo "Skipped because RUN_DATA_PREP=$RUN_DATA_PREP"
fi

preflight_args=(
  python3 "$EXP_DIR/preflight_eagle3_pipeline.py"
  --input-data "$INPUT_DATA"
  --hidden-states-dir "$HIDDEN_STATES_DIR"
  --output-dir "$OUTPUT_DIR"
  --trained-ckpt "$TRAINED_CKPT"
  --export-dir "$EXPORT_DIR"
  --vllm-draft-dir "$VLLM_DRAFT_DIR"
  --verifier-config-dir "$VERIFIER_CONFIG_DIR"
  --sbatch-account "$SBATCH_ACCOUNT"
  --chat-template "$CHAT_TEMPLATE"
  --base-model "$BASE_MODEL"
  --modelopt-dir "$MODELOPT_DIR"
  --reference-arch "$pipeline_reference_arch"
)
if is_true "$PREFLIGHT_SKIP_EXISTING_PATH_CHECKS"; then
  preflight_args+=(--skip-existing-path-checks)
fi
if is_true "$PREFLIGHT_REQUIRE_MODELOPT_IMPORT"; then
  preflight_args+=(--require-modelopt-import)
fi
if is_true "$PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK"; then
  preflight_args+=(--require-chat-template-mask)
fi

if is_true "$RUN_PREFLIGHT"; then
  step "Local preflight"
  print_cmd env ARCH_ENV_FILE="$pipeline_arch_env_file" "${preflight_args[@]}"
  env ARCH_ENV_FILE="$pipeline_arch_env_file" "${preflight_args[@]}"
else
  step "Local preflight"
  echo "Skipped because RUN_PREFLIGHT=$RUN_PREFLIGHT"
fi

if is_true "$SUBMIT"; then
  submit_failures=0
  if [[ "$SBATCH_ACCOUNT" == "dummy" ]]; then
    echo "SUBMIT=true requires a real SBATCH_ACCOUNT." >&2
    submit_failures=1
  fi
  if [[ ! -s "$INPUT_DATA" ]]; then
    echo "SUBMIT=true requires existing non-empty INPUT_DATA: $INPUT_DATA" >&2
    submit_failures=1
  fi
  if [[ ! -s "$CHAT_TEMPLATE" ]]; then
    echo "SUBMIT=true requires existing non-empty CHAT_TEMPLATE: $CHAT_TEMPLATE" >&2
    submit_failures=1
  fi
  if [[ ! -f "$VERIFIER_CONFIG_DIR/config.json" ]]; then
    echo "SUBMIT=true requires VERIFIER_CONFIG_DIR/config.json: $VERIFIER_CONFIG_DIR" >&2
    submit_failures=1
  fi
  if [[ "$submit_failures" -ne 0 ]]; then
    exit 1
  fi
fi

if is_true "$RUN_PIPELINE"; then
  step "Slurm pipeline plan"
  pipeline_env=(
    env
    ARTIFACT_ROOT="$ARTIFACT_ROOT"
    SUBMIT="$SUBMIT"
    SBATCH_ACCOUNT="$SBATCH_ACCOUNT"
    SBATCH_PARTITION="$SBATCH_PARTITION"
    INPUT_DATA="$INPUT_DATA"
    HIDDEN_STATES_DIR="$HIDDEN_STATES_DIR"
    OUTPUT_DIR="$OUTPUT_DIR"
    TRAINED_CKPT="$TRAINED_CKPT"
    EXPORT_DIR="$EXPORT_DIR"
    VLLM_DRAFT_DIR="$VLLM_DRAFT_DIR"
    TRAINING_CKPT_VALIDATION_JSON="$TRAINING_CKPT_VALIDATION_JSON"
    TRAINING_CKPT_VALIDATION_MARKDOWN="$TRAINING_CKPT_VALIDATION_MARKDOWN"
    EXPORT_ARTIFACTS_JSON="$REPORT_DIR/eagle3_export_artifacts.json"
    EXPORT_ARTIFACTS_MARKDOWN="$REPORT_DIR/eagle3_export_artifacts.md"
    VERIFIER_CONFIG_DIR="$VERIFIER_CONFIG_DIR"
    BASE_MODEL="$BASE_MODEL"
    CHAT_TEMPLATE="$CHAT_TEMPLATE"
    CONTAINER="$CONTAINER"
    MOUNTS="$MOUNTS"
    MODELOPT_DIR="$MODELOPT_DIR"
    ARCH_ENV_FILE="$pipeline_arch_env_file"
    REFERENCE_ARCH="$pipeline_reference_arch"
    RUN_PILOT="$RUN_PILOT"
    RUN_TRAINED_DRAFT_SMOKE="$RUN_TRAINED_DRAFT_SMOKE"
    RUN_TRAINED_DRAFT_SWEEP="$RUN_TRAINED_DRAFT_SWEEP"
  )
  print_cmd "${pipeline_env[@]}" bash "$EXP_DIR/submit_eagle3_pipeline.sh"
  "${pipeline_env[@]}" bash "$EXP_DIR/submit_eagle3_pipeline.sh"
else
  step "Slurm pipeline plan"
  echo "Skipped because RUN_PIPELINE=$RUN_PIPELINE"
fi

step "Rollout capture job analysis"
rollout_job_cmd=(
  python3 "$EXP_DIR/analyze_rollout_capture_job.py"
  --artifact-root "$ARTIFACT_ROOT"
  --repo-root "$SWE_REPO_ROOT"
  --rollout-log-dir "$ROLLOUT_LOG_DIR"
  --output-data "$ROLLOUT_CONVERSATIONS"
  --markdown-out "$ROLLOUT_CAPTURE_JOB_MARKDOWN"
  --json-out "$ROLLOUT_CAPTURE_JOB_JSON"
)
print_cmd "${rollout_job_cmd[@]}"
"${rollout_job_cmd[@]}"

step "Corpus strategy"
corpus_strategy_cmd=(
  python3 "$EXP_DIR/analyze_corpus_strategy.py"
  --artifact-root "$ARTIFACT_ROOT"
  --target-context "$EAGLE3_TARGET_CONTEXT"
  --input-data "$INPUT_DATA"
  --rollout-capture-analysis-json "$ROLLOUT_CAPTURE_ANALYSIS_JSON"
  --markdown-out "$CORPUS_STRATEGY_MARKDOWN"
  --json-out "$CORPUS_STRATEGY_JSON"
)
print_cmd "${corpus_strategy_cmd[@]}"
"${corpus_strategy_cmd[@]}"

if is_true "$RUN_PIPELINE_SUBMIT_PREFLIGHT"; then
  step "Eagle3 pipeline submit preflight"
  pipeline_submit_preflight_cmd=(
    python3 "$EXP_DIR/preflight_eagle3_pipeline_submit.py"
    --artifact-root "$ARTIFACT_ROOT"
    --input-data "$INPUT_DATA"
    --hidden-states-dir "$HIDDEN_STATES_DIR"
    --output-dir "$OUTPUT_DIR"
    --trained-ckpt "$TRAINED_CKPT"
    --export-dir "$EXPORT_DIR"
    --vllm-draft-dir "$VLLM_DRAFT_DIR"
    --verifier-config-dir "$VERIFIER_CONFIG_DIR"
    --chat-template "$CHAT_TEMPLATE"
    --modelopt-dir "$MODELOPT_DIR"
    --reference-arch "$pipeline_reference_arch"
    --arch-env-file "$pipeline_arch_env_file"
    --container-preflight-json "$CONTAINER_PREFLIGHT_JSON"
    --corpus-strategy-json "$CORPUS_STRATEGY_JSON"
    --rollout-state-json "$ROLLOUT_STATE_ADVANCE_JSON"
    --sbatch-account "$SBATCH_ACCOUNT"
    --sbatch-partition "$SBATCH_PARTITION"
    --container "$CONTAINER"
    --mounts "$MOUNTS"
    --run-pilot "$RUN_PILOT"
    --target-context "$EAGLE3_TARGET_CONTEXT"
    --markdown-out "$PIPELINE_SUBMIT_PREFLIGHT_MARKDOWN"
    --json-out "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
  )
  if is_true "$SUBMIT"; then
    pipeline_submit_preflight_cmd+=(--fail-if-not-ready)
  fi
  print_cmd "${pipeline_submit_preflight_cmd[@]}"
  if ! "${pipeline_submit_preflight_cmd[@]}"; then
    echo "WARN: Eagle3 pipeline submit preflight returned nonzero; inspect $PIPELINE_SUBMIT_PREFLIGHT_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "Eagle3 pipeline submit preflight"
  echo "Skipped because RUN_PIPELINE_SUBMIT_PREFLIGHT=$RUN_PIPELINE_SUBMIT_PREFLIGHT"
fi

if is_true "$RUN_TRAINING_SCALE_PLAN"; then
  step "Eagle3 training scale plan"
  scale_cmd=(
    python3 "$EXP_DIR/estimate_eagle3_training_scale.py"
    --artifact-root "$ARTIFACT_ROOT"
    --input-data "$INPUT_DATA"
    --corpus-strategy-json "$CORPUS_STRATEGY_JSON"
    --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
    --target-context "$EAGLE3_TARGET_CONTEXT"
    --gpus "${TRAIN_GPUS_PER_NODE:-8}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --epochs "${NUM_TRAIN_EPOCHS:-1}"
    --markdown-out "$REPORT_DIR/eagle3_training_scale.md"
    --json-out "$TRAINING_SCALE_JSON"
  )
  print_cmd "${scale_cmd[@]}"
  "${scale_cmd[@]}"
else
  step "Eagle3 training scale plan"
  echo "Skipped because RUN_TRAINING_SCALE_PLAN=$RUN_TRAINING_SCALE_PLAN"
fi

if is_true "$RUN_AUDIT"; then
  step "Readiness audit"
  audit_args=(
    python3 "$EXP_DIR/audit_eagle3_readiness.py"
    --input-data "$INPUT_DATA"
    --hidden-states-dir "$HIDDEN_STATES_DIR"
    --output-dir "$OUTPUT_DIR"
    --trained-ckpt "$TRAINED_CKPT"
    --export-dir "$EXPORT_DIR"
    --vllm-draft-dir "$VLLM_DRAFT_DIR"
    --verifier-config-dir "$VERIFIER_CONFIG_DIR"
    --chat-template "$CHAT_TEMPLATE"
    --modelopt-dir "$MODELOPT_DIR"
    --reference-arch "$pipeline_reference_arch"
    --arch-env-file "$pipeline_arch_env_file"
    --sbatch-account "$SBATCH_ACCOUNT"
    --container-preflight-json "$CONTAINER_PREFLIGHT_JSON"
    --modelopt-loss-mask-json "$MODELOPT_LOSS_MASK_JSON"
    --nemo-rl-specdec-json "$NEMO_RL_SPECDEC_JSON"
    --nemo-rl-drift-json "$NEMO_RL_DRIFT_JSON"
    --rollout-capture-json "$ROLLOUT_CAPTURE_JSON"
    --rollout-capture-analysis-json "$ROLLOUT_CAPTURE_ANALYSIS_JSON"
    --rollout-capture-job-json "$ROLLOUT_CAPTURE_JOB_JSON"
    --rollout-submit-preflight-json "$ROLLOUT_SUBMIT_PREFLIGHT_JSON"
    --corpus-strategy-json "$CORPUS_STRATEGY_JSON"
    --training-scale-json "$TRAINING_SCALE_JSON"
    --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
    --markdown-out "$READINESS_MARKDOWN"
    --json-out "$READINESS_JSON"
  )
  print_cmd "${audit_args[@]}"
  "${audit_args[@]}"
else
  step "Readiness audit"
  echo "Skipped because RUN_AUDIT=$RUN_AUDIT"
fi

if is_true "$RUN_NEXT_ACTION_PLAN"; then
  step "Eagle3 next-action plan"
  next_action_cmd=(
    python3 "$EXP_DIR/plan_eagle3_next_actions.py"
    --artifact-root "$ARTIFACT_ROOT"
    --container-preflight-json "$CONTAINER_PREFLIGHT_JSON"
    --rollout-submit-preflight-json "$ROLLOUT_SUBMIT_PREFLIGHT_JSON"
    --rollout-state-json "$ROLLOUT_STATE_ADVANCE_JSON"
    --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
    --pipeline-analysis-json "$PIPELINE_ANALYSIS_JSON"
    --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON"
    --export-artifacts-json "$REPORT_DIR/eagle3_export_artifacts.json"
    --sweep-json "$SWEEP_JSON"
    --training-scale-json "$TRAINING_SCALE_JSON"
    --modelopt-loss-mask-json "$MODELOPT_LOSS_MASK_JSON"
    --nemo-rl-drift-json "$NEMO_RL_DRIFT_JSON"
    --readiness-json "$READINESS_JSON"
    --json-out "$NEXT_ACTION_PLAN_JSON"
    --markdown-out "$NEXT_ACTION_PLAN_MARKDOWN"
  )
  print_cmd "${next_action_cmd[@]}"
  if ! "${next_action_cmd[@]}"; then
    echo "WARN: Eagle3 next-action plan returned nonzero; inspect $NEXT_ACTION_PLAN_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
  next_action_validation_cmd=(
    python3 "$EXP_DIR/validate_eagle3_next_action_plan.py"
    --plan-json "$NEXT_ACTION_PLAN_JSON"
    --json-out "$NEXT_ACTION_PLAN_VALIDATION_JSON"
    --markdown-out "$NEXT_ACTION_PLAN_VALIDATION_MARKDOWN"
  )
  print_cmd "${next_action_validation_cmd[@]}"
  if ! "${next_action_validation_cmd[@]}"; then
    echo "WARN: Eagle3 next-action validation returned nonzero; inspect $NEXT_ACTION_PLAN_VALIDATION_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
  next_action_transition_cmd=(
    python3 "$EXP_DIR/validate_eagle3_next_action_transitions.py"
    --json-out "$NEXT_ACTION_TRANSITIONS_JSON"
    --markdown-out "$NEXT_ACTION_TRANSITIONS_MARKDOWN"
  )
  print_cmd "${next_action_transition_cmd[@]}"
  if ! "${next_action_transition_cmd[@]}"; then
    echo "WARN: Eagle3 next-action transition validation returned nonzero; inspect $NEXT_ACTION_TRANSITIONS_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "Eagle3 next-action plan"
  echo "Skipped because RUN_NEXT_ACTION_PLAN=$RUN_NEXT_ACTION_PLAN"
fi

step "Next concrete inputs"
cat <<EOF
1. Provide one data source:
   - MODE=discover ROLLOUT_ROOTS="/path/to/nemo_rl_outputs ..."
   - MODE=rollout INPUT_PATHS="/path/to/rollouts.jsonl ..."
   - MODE=generate PROMPT_DATA=/path/to/swe_prompts.jsonl OPENAI_BASE_URL=http://host:8000/v1
2. Set PREP_DRY_RUN=false to materialize INPUT_DATA and CHAT_TEMPLATE.
3. Set VERIFIER_CONFIG_DIR to a local Qwen3-235B-A22B-Thinking-2507 config directory.
4. For non-Qwen3 verifiers, keep RUN_ARCH_DERIVE=auto or set RUN_ARCH_DERIVE=true and inspect $REFERENCE_ARCH.
5. Use RUN_PILOT=true for the first GPU attempt; it limits dump/train defaults before a full run.
6. Keep SUBMIT=false until the readiness audit only reports the expected missing heavy artifacts.
7. Set SUBMIT=true to submit preflight -> dump -> validate -> train -> export, optionally with RUN_TRAINED_DRAFT_SMOKE=true.
EOF
