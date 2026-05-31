#!/usr/bin/env bash
set -euo pipefail

# Submit the Qwen3-235B Eagle3 draft-model pipeline with Slurm dependencies:
# preflight -> dump hidden states -> validate hidden states -> train -> export.
#
# Default mode prints the commands only:
#
#   SUBMIT=false SBATCH_ACCOUNT=<account> INPUT_DATA=... HIDDEN_STATES_DIR=... \
#     OUTPUT_DIR=... EXPORT_DIR=... VLLM_DRAFT_DIR=... VERIFIER_CONFIG_DIR=... \
#     bash experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh
#
# To actually submit, set SUBMIT=true.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

SUBMIT="${SUBMIT:-false}"
RUN_PILOT="${RUN_PILOT:-false}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
RUN_DUMP="${RUN_DUMP:-true}"
RUN_VALIDATE_HIDDENS="${RUN_VALIDATE_HIDDENS:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EXPORT="${RUN_EXPORT:-true}"
RUN_TRAINED_DRAFT_SMOKE="${RUN_TRAINED_DRAFT_SMOKE:-false}"
RUN_TRAINED_DRAFT_SWEEP="${RUN_TRAINED_DRAFT_SWEEP:-false}"
START_PIPELINE_WATCHER="${START_PIPELINE_WATCHER:-true}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"

INPUT_DATA="${INPUT_DATA:?set INPUT_DATA}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:?set HIDDEN_STATES_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
TRAINED_CKPT="${TRAINED_CKPT:-$OUTPUT_DIR}"
EXPORT_DIR="${EXPORT_DIR:?set EXPORT_DIR}"
VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:?set VLLM_DRAFT_DIR}"
VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:?set VERIFIER_CONFIG_DIR}"
EXPORT_CONFIG_COMPARE_JSON="${EXPORT_CONFIG_COMPARE_JSON:-$EXPORT_DIR/config_compare.json}"
VLLM_CONFIG_COMPARE_JSON="${VLLM_CONFIG_COMPARE_JSON:-$VLLM_DRAFT_DIR/config_compare.json}"
if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
  DEFAULT_TRAINING_CKPT_VALIDATION_JSON="$ARTIFACT_ROOT/reports/eagle3_training_checkpoint.json"
  DEFAULT_TRAINING_CKPT_VALIDATION_MARKDOWN="$ARTIFACT_ROOT/reports/eagle3_training_checkpoint.md"
  DEFAULT_EXPORT_ARTIFACTS_JSON="$ARTIFACT_ROOT/reports/eagle3_export_artifacts.json"
  DEFAULT_EXPORT_ARTIFACTS_MARKDOWN="$ARTIFACT_ROOT/reports/eagle3_export_artifacts.md"
else
  DEFAULT_TRAINING_CKPT_VALIDATION_JSON="$TRAINED_CKPT/training_checkpoint_validation.json"
  DEFAULT_TRAINING_CKPT_VALIDATION_MARKDOWN="$TRAINED_CKPT/training_checkpoint_validation.md"
  DEFAULT_EXPORT_ARTIFACTS_JSON="$VLLM_DRAFT_DIR/export_artifacts.json"
  DEFAULT_EXPORT_ARTIFACTS_MARKDOWN="$VLLM_DRAFT_DIR/export_artifacts.md"
fi
TRAINING_CKPT_VALIDATION_JSON="${TRAINING_CKPT_VALIDATION_JSON:-$DEFAULT_TRAINING_CKPT_VALIDATION_JSON}"
TRAINING_CKPT_VALIDATION_MARKDOWN="${TRAINING_CKPT_VALIDATION_MARKDOWN:-$DEFAULT_TRAINING_CKPT_VALIDATION_MARKDOWN}"
EXPORT_ARTIFACTS_JSON="${EXPORT_ARTIFACTS_JSON:-$DEFAULT_EXPORT_ARTIFACTS_JSON}"
EXPORT_ARTIFACTS_MARKDOWN="${EXPORT_ARTIFACTS_MARKDOWN:-$DEFAULT_EXPORT_ARTIFACTS_MARKDOWN}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
TRAINING_SEQ_LEN="${TRAINING_SEQ_LEN:-16384}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$TRAINING_SEQ_LEN}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
DEFAULT_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
CONTAINER="${CONTAINER:-$DEFAULT_CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR}"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$SCRIPT_DIR/qwen3_235b_thinking_eagle3_architecture.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"
RESOURCE_PROFILE_ENV="${RESOURCE_PROFILE_ENV:-${ARTIFACT_ROOT:+$ARTIFACT_ROOT/reports/eagle3_resource_profile.env}}"

if [[ -n "$ARCH_ENV_FILE" ]]; then
  if [[ ! -f "$ARCH_ENV_FILE" ]]; then
    echo "ARCH_ENV_FILE does not exist: $ARCH_ENV_FILE" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$ARCH_ENV_FILE"
fi

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

profile_value() {
  local key="$1"
  local file="${2:-}"
  [[ -n "$file" && -f "$file" ]] || return 0
  awk -v key="$key" '
    $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "=" {
      sub(/^[[:space:]]*export[[:space:]]+/, "")
      sub("^[^=]*=", "")
      gsub(/^["'\'']|["'\'']$/, "")
      print
      exit
    }
  ' "$file"
}

if [[ -n "$RESOURCE_PROFILE_ENV" && -f "$RESOURCE_PROFILE_ENV" ]]; then
  if [[ -z "${DUMP_GPUS_PER_NODE+x}" ]]; then
    value="$(profile_value DUMP_GPUS_PER_NODE "$RESOURCE_PROFILE_ENV")"
    [[ -n "$value" ]] && DUMP_GPUS_PER_NODE="$value"
  fi
  if [[ -z "${TRAIN_GPUS_PER_NODE+x}" ]]; then
    value="$(profile_value TRAIN_GPUS_PER_NODE "$RESOURCE_PROFILE_ENV")"
    [[ -n "$value" ]] && TRAIN_GPUS_PER_NODE="$value"
  fi
  if [[ -z "${EXPORT_GPUS_PER_NODE+x}" ]]; then
    value="$(profile_value EXPORT_GPUS_PER_NODE "$RESOURCE_PROFILE_ENV")"
    [[ -n "$value" ]] && EXPORT_GPUS_PER_NODE="$value"
  fi
  if [[ -z "${TP+x}" ]]; then
    value="$(profile_value TP "$RESOURCE_PROFILE_ENV")"
    [[ -n "$value" ]] && TP="$value"
  fi
fi

if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
  if [[ -z "$CONTAINER" ]]; then
    echo "SUBMIT=true requires CONTAINER so the Eagle3 pipeline runs in the proven runtime image." >&2
    exit 1
  fi
  if [[ ! -e "$CONTAINER" ]]; then
    echo "CONTAINER is not visible from this host: $CONTAINER" >&2
    exit 1
  fi
fi

PREFLIGHT_TIME="${PREFLIGHT_TIME:-00:30:00}"
PREFLIGHT_REQUIRE_MODELOPT_IMPORT="${PREFLIGHT_REQUIRE_MODELOPT_IMPORT:-true}"
PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK="${PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK:-true}"
PREFLIGHT_SKIP_EXISTING_PATH_CHECKS="${PREFLIGHT_SKIP_EXISTING_PATH_CHECKS:-false}"

DUMP_NODES="${DUMP_NODES:-1}"
DUMP_NTASKS_PER_NODE="${DUMP_NTASKS_PER_NODE:-1}"
DUMP_GPUS_PER_NODE="${DUMP_GPUS_PER_NODE:-8}"
TP="${TP:-8}"
if is_true "$RUN_PILOT"; then
  DEBUG_MAX_NUM_CONVERSATIONS="${DEBUG_MAX_NUM_CONVERSATIONS:-8}"
  DATA_SAMPLE_SIZE="${DATA_SAMPLE_SIZE:-8}"
  MAX_STEPS="${MAX_STEPS:-20}"
  SAVE_STEPS="${SAVE_STEPS:-20}"
  DUMP_TIME="${DUMP_TIME:-02:00:00}"
  TRAIN_TIME="${TRAIN_TIME:-02:00:00}"
  EXPORT_TIME="${EXPORT_TIME:-01:00:00}"
else
  DEBUG_MAX_NUM_CONVERSATIONS="${DEBUG_MAX_NUM_CONVERSATIONS:-}"
  DATA_SAMPLE_SIZE="${DATA_SAMPLE_SIZE:-}"
  MAX_STEPS="${MAX_STEPS:-}"
  SAVE_STEPS="${SAVE_STEPS:-512}"
  DUMP_TIME="${DUMP_TIME:-12:00:00}"
  TRAIN_TIME="${TRAIN_TIME:-12:00:00}"
  EXPORT_TIME="${EXPORT_TIME:-02:00:00}"
fi

VALIDATE_HIDDENS_TIME="${VALIDATE_HIDDENS_TIME:-00:30:00}"
if is_true "$RUN_PILOT"; then
  HIDDEN_STATES_VALIDATE_LIMIT="${HIDDEN_STATES_VALIDATE_LIMIT:-${DEBUG_MAX_NUM_CONVERSATIONS:-8}}"
else
  HIDDEN_STATES_VALIDATE_LIMIT="${HIDDEN_STATES_VALIDATE_LIMIT:-64}"
fi
EXPECTED_HIDDEN_SIZE="${EXPECTED_HIDDEN_SIZE:-4096}"
EXPECTED_AUX_COUNT="${EXPECTED_AUX_COUNT:-3}"
REQUIRE_LOSS_MASK="${REQUIRE_LOSS_MASK:-true}"
REQUIRE_POSITIVE_LOSS_MASK="${REQUIRE_POSITIVE_LOSS_MASK:-true}"
VALIDATE_MODELOPT_LOADER="${VALIDATE_MODELOPT_LOADER:-true}"
HIDDEN_STATES_VALIDATION_JSON="${HIDDEN_STATES_VALIDATION_JSON:-$HIDDEN_STATES_DIR/validation_summary.json}"

TRAIN_NODES="${TRAIN_NODES:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-8}"
EXPORT_GPUS_PER_NODE="${EXPORT_GPUS_PER_NODE:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
EAGLE_TTT_STEPS="${EAGLE_TTT_STEPS:-3}"
EAGLE_LOSS_DECAY_FACTOR="${EAGLE_LOSS_DECAY_FACTOR:-0.9}"
USE_FAKE_BASE_FOR_OFFLINE="${USE_FAKE_BASE_FOR_OFFLINE:-true}"
SMOKE_MAX_NUM_STEPS="${SMOKE_MAX_NUM_STEPS:-1}"
SMOKE_EAGLE3_NUM_SPEC_TOKENS="${SMOKE_EAGLE3_NUM_SPEC_TOKENS:-3}"
SMOKE_EAGLE3_DRAFT_TP="${SMOKE_EAGLE3_DRAFT_TP:-1}"
SMOKE_JOB_FILE="${SMOKE_JOB_FILE:-$ROOT_DIR/latest_trained_draft_specdec_smoke_jobs.txt}"
SWEEP_SPEC_TOKENS_LIST="${SWEEP_SPEC_TOKENS_LIST:-2 3 4}"
SWEEP_MAX_NUM_STEPS="${SWEEP_MAX_NUM_STEPS:-2}"
SWEEP_JOB_FILE="${SWEEP_JOB_FILE:-$ROOT_DIR/latest_trained_draft_spec_tokens_sweep_jobs.txt}"

export REPO_ROOT="$ROOT_DIR"
export ARTIFACT_ROOT
export INPUT_DATA HIDDEN_STATES_DIR OUTPUT_DIR TRAINED_CKPT EXPORT_DIR VLLM_DRAFT_DIR VERIFIER_CONFIG_DIR
export EXPORT_CONFIG_COMPARE_JSON VLLM_CONFIG_COMPARE_JSON
export TRAINING_CKPT_VALIDATION_JSON TRAINING_CKPT_VALIDATION_MARKDOWN
export EXPORT_ARTIFACTS_JSON EXPORT_ARTIFACTS_MARKDOWN
export BASE_MODEL CHAT_TEMPLATE CONTAINER MOUNTS MODELOPT_DIR ARCH_ENV_FILE REFERENCE_ARCH
export TRAINING_SEQ_LEN MAX_SEQ_LEN ANSWER_ONLY_LOSS TRUST_REMOTE_CODE
export RUN_PILOT DEBUG_MAX_NUM_CONVERSATIONS DATA_SAMPLE_SIZE MAX_STEPS
export NUM_TRAIN_EPOCHS PER_DEVICE_TRAIN_BATCH_SIZE LEARNING_RATE SAVE_STEPS
export EAGLE_TTT_STEPS EAGLE_LOSS_DECAY_FACTOR
export USE_FAKE_BASE_FOR_OFFLINE
export DUMP_GPUS_PER_NODE TRAIN_GPUS_PER_NODE EXPORT_GPUS_PER_NODE TP
export PREFLIGHT_REQUIRE_MODELOPT_IMPORT PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK
export PREFLIGHT_SKIP_EXISTING_PATH_CHECKS
export HIDDEN_STATES_VALIDATE_LIMIT EXPECTED_HIDDEN_SIZE EXPECTED_AUX_COUNT
export REQUIRE_LOSS_MASK REQUIRE_POSITIVE_LOSS_MASK VALIDATE_MODELOPT_LOADER
export HIDDEN_STATES_VALIDATION_JSON

mkdir -p "$ROOT_DIR/logs"

print_export_summary() {
  cat <<EOF
# pipeline env
ARTIFACT_ROOT=$ARTIFACT_ROOT
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
INPUT_DATA=$INPUT_DATA
HIDDEN_STATES_DIR=$HIDDEN_STATES_DIR
OUTPUT_DIR=$OUTPUT_DIR
TRAINED_CKPT=$TRAINED_CKPT
EXPORT_DIR=$EXPORT_DIR
VLLM_DRAFT_DIR=$VLLM_DRAFT_DIR
VERIFIER_CONFIG_DIR=$VERIFIER_CONFIG_DIR
EXPORT_CONFIG_COMPARE_JSON=$EXPORT_CONFIG_COMPARE_JSON
VLLM_CONFIG_COMPARE_JSON=$VLLM_CONFIG_COMPARE_JSON
TRAINING_CKPT_VALIDATION_JSON=$TRAINING_CKPT_VALIDATION_JSON
TRAINING_CKPT_VALIDATION_MARKDOWN=$TRAINING_CKPT_VALIDATION_MARKDOWN
EXPORT_ARTIFACTS_JSON=$EXPORT_ARTIFACTS_JSON
EXPORT_ARTIFACTS_MARKDOWN=$EXPORT_ARTIFACTS_MARKDOWN
HIDDEN_STATES_VALIDATION_JSON=$HIDDEN_STATES_VALIDATION_JSON
BASE_MODEL=$BASE_MODEL
TRAINING_SEQ_LEN=$TRAINING_SEQ_LEN
MAX_SEQ_LEN=$MAX_SEQ_LEN
ANSWER_ONLY_LOSS=$ANSWER_ONLY_LOSS
TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE
CHAT_TEMPLATE=$CHAT_TEMPLATE
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
MODELOPT_DIR=$MODELOPT_DIR
ARCH_ENV_FILE=$ARCH_ENV_FILE
RESOURCE_PROFILE_ENV=$RESOURCE_PROFILE_ENV
REFERENCE_ARCH=$REFERENCE_ARCH
RUN_PILOT=$RUN_PILOT
DEBUG_MAX_NUM_CONVERSATIONS=$DEBUG_MAX_NUM_CONVERSATIONS
DATA_SAMPLE_SIZE=$DATA_SAMPLE_SIZE
MAX_STEPS=$MAX_STEPS
SAVE_STEPS=$SAVE_STEPS
USE_FAKE_BASE_FOR_OFFLINE=$USE_FAKE_BASE_FOR_OFFLINE
DUMP_GPUS_PER_NODE=$DUMP_GPUS_PER_NODE
TRAIN_GPUS_PER_NODE=$TRAIN_GPUS_PER_NODE
EXPORT_GPUS_PER_NODE=$EXPORT_GPUS_PER_NODE
TP=$TP
RUN_PREFLIGHT=$RUN_PREFLIGHT
RUN_DUMP=$RUN_DUMP
RUN_VALIDATE_HIDDENS=$RUN_VALIDATE_HIDDENS
RUN_TRAIN=$RUN_TRAIN
RUN_EXPORT=$RUN_EXPORT
RUN_TRAINED_DRAFT_SMOKE=$RUN_TRAINED_DRAFT_SMOKE
RUN_TRAINED_DRAFT_SWEEP=$RUN_TRAINED_DRAFT_SWEEP
START_PIPELINE_WATCHER=$START_PIPELINE_WATCHER
PREFLIGHT_REQUIRE_MODELOPT_IMPORT=$PREFLIGHT_REQUIRE_MODELOPT_IMPORT
PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK=$PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK
VALIDATE_MODELOPT_LOADER=$VALIDATE_MODELOPT_LOADER
EOF
}

run_or_print() {
  local label="$1"
  shift
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    "$@"
  else
    printf '# %s\n' "$label" >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
  fi
}

job_file="$ROOT_DIR/latest_eagle3_pipeline_jobs.txt"
: > "$job_file"

print_export_summary

preflight_job=""
if [[ "$RUN_PREFLIGHT" == "true" || "$RUN_PREFLIGHT" == "True" ]]; then
  preflight_cmd=(
    sbatch --parsable
    --account="$SBATCH_ACCOUNT"
    --partition="$SBATCH_PARTITION"
    --time="$PREFLIGHT_TIME"
    --export=ALL
    "$SCRIPT_DIR/slurm_preflight.sbatch"
  )
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    preflight_job="$("${preflight_cmd[@]}")"
    echo "preflight_job=$preflight_job" | tee -a "$job_file"
  else
    run_or_print "preflight" "${preflight_cmd[@]}"
    preflight_job="PREFLIGHT_JOB_ID"
    echo "preflight_job=$preflight_job" >> "$job_file"
  fi
fi

dump_job=""
if [[ "$RUN_DUMP" == "true" || "$RUN_DUMP" == "True" ]]; then
  dump_cmd=(
    sbatch --parsable
    --account="$SBATCH_ACCOUNT"
    --partition="$SBATCH_PARTITION"
    --nodes="$DUMP_NODES"
    --ntasks-per-node="$DUMP_NTASKS_PER_NODE"
    --gres="gpu:$DUMP_GPUS_PER_NODE"
    --time="$DUMP_TIME"
    --export=ALL
  )
  [[ -n "$preflight_job" ]] && dump_cmd+=(--dependency="afterok:$preflight_job")
  dump_cmd+=("$SCRIPT_DIR/slurm_dump_hidden_states.sbatch")
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    dump_job="$("${dump_cmd[@]}")"
    echo "dump_job=$dump_job" | tee -a "$job_file"
  else
    run_or_print "dump" "${dump_cmd[@]}"
    dump_job="DUMP_JOB_ID"
    echo "dump_job=$dump_job" >> "$job_file"
  fi
fi

validate_hiddens_job=""
if [[ "$RUN_VALIDATE_HIDDENS" == "true" || "$RUN_VALIDATE_HIDDENS" == "True" ]]; then
  validate_hiddens_cmd=(
    sbatch --parsable
    --account="$SBATCH_ACCOUNT"
    --partition="$SBATCH_PARTITION"
    --time="$VALIDATE_HIDDENS_TIME"
    --export=ALL
  )
  if [[ -n "$dump_job" ]]; then
    validate_hiddens_cmd+=(--dependency="afterok:$dump_job")
  elif [[ -n "$preflight_job" ]]; then
    validate_hiddens_cmd+=(--dependency="afterok:$preflight_job")
  fi
  validate_hiddens_cmd+=("$SCRIPT_DIR/slurm_validate_hidden_states.sbatch")
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    validate_hiddens_job="$("${validate_hiddens_cmd[@]}")"
    echo "validate_hiddens_job=$validate_hiddens_job" | tee -a "$job_file"
  else
    run_or_print "validate_hiddens" "${validate_hiddens_cmd[@]}"
    validate_hiddens_job="VALIDATE_HIDDENS_JOB_ID"
    echo "validate_hiddens_job=$validate_hiddens_job" >> "$job_file"
  fi
fi

train_job=""
if [[ "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "True" ]]; then
  train_cmd=(
    sbatch --parsable
    --account="$SBATCH_ACCOUNT"
    --partition="$SBATCH_PARTITION"
    --nodes="$TRAIN_NODES"
    --gres="gpu:$TRAIN_GPUS_PER_NODE"
    --time="$TRAIN_TIME"
    --export=ALL
  )
  if [[ -n "$validate_hiddens_job" ]]; then
    train_cmd+=(--dependency="afterok:$validate_hiddens_job")
  elif [[ -n "$dump_job" ]]; then
    train_cmd+=(--dependency="afterok:$dump_job")
  elif [[ -n "$preflight_job" ]]; then
    train_cmd+=(--dependency="afterok:$preflight_job")
  fi
  train_cmd+=("$SCRIPT_DIR/slurm_offline_train.sbatch")
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    train_job="$("${train_cmd[@]}")"
    echo "train_job=$train_job" | tee -a "$job_file"
  else
    run_or_print "train" "${train_cmd[@]}"
    train_job="TRAIN_JOB_ID"
    echo "train_job=$train_job" >> "$job_file"
  fi
fi

export_job=""
if [[ "$RUN_EXPORT" == "true" || "$RUN_EXPORT" == "True" ]]; then
  export_cmd=(
    sbatch --parsable
    --account="$SBATCH_ACCOUNT"
    --partition="$SBATCH_PARTITION"
    --gres="gpu:$EXPORT_GPUS_PER_NODE"
    --time="$EXPORT_TIME"
    --export=ALL
  )
  if [[ -n "$train_job" ]]; then
    export_cmd+=(--dependency="afterok:$train_job")
  elif [[ -n "$preflight_job" ]]; then
    export_cmd+=(--dependency="afterok:$preflight_job")
  fi
  export_cmd+=("$SCRIPT_DIR/slurm_export_vllm.sbatch")
  if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
    export_job="$("${export_cmd[@]}")"
    echo "export_job=$export_job" | tee -a "$job_file"
  else
    run_or_print "export" "${export_cmd[@]}"
    export_job="EXPORT_JOB_ID"
    echo "export_job=$export_job" >> "$job_file"
  fi
fi

if [[ "$RUN_TRAINED_DRAFT_SMOKE" == "true" || "$RUN_TRAINED_DRAFT_SMOKE" == "True" ]]; then
  smoke_dependency=""
  if [[ -n "$export_job" ]]; then
    smoke_dependency="afterok:$export_job"
  elif [[ -n "$train_job" ]]; then
    smoke_dependency="afterok:$train_job"
  fi
  smoke_cmd=(
    env
    SUBMIT="$SUBMIT"
    VLLM_DRAFT_DIR="$VLLM_DRAFT_DIR"
    MAX_NUM_STEPS="$SMOKE_MAX_NUM_STEPS"
    EAGLE3_NUM_SPEC_TOKENS="$SMOKE_EAGLE3_NUM_SPEC_TOKENS"
    EAGLE3_DRAFT_TP="$SMOKE_EAGLE3_DRAFT_TP"
    JOB_FILE="$SMOKE_JOB_FILE"
  )
  if [[ -n "$smoke_dependency" ]]; then
    smoke_cmd+=(
      BASELINE_SBATCH_DEPENDENCY="$smoke_dependency"
      ALLOW_MISSING_DRAFT_FOR_DEPENDENCY=true
    )
  fi
  smoke_cmd+=(bash "$SCRIPT_DIR/submit_trained_draft_smoke_pair.sh")
  run_or_print "trained_draft_smoke" "${smoke_cmd[@]}"
fi

if [[ "$RUN_TRAINED_DRAFT_SWEEP" == "true" || "$RUN_TRAINED_DRAFT_SWEEP" == "True" ]]; then
  sweep_dependency=""
  if [[ -n "$export_job" ]]; then
    sweep_dependency="afterok:$export_job"
  elif [[ -n "$train_job" ]]; then
    sweep_dependency="afterok:$train_job"
  fi
  sweep_cmd=(
    env
    SUBMIT="$SUBMIT"
    VLLM_DRAFT_DIR="$VLLM_DRAFT_DIR"
    MAX_NUM_STEPS="$SWEEP_MAX_NUM_STEPS"
    SPEC_TOKENS_LIST="$SWEEP_SPEC_TOKENS_LIST"
    EAGLE3_DRAFT_TP="$SMOKE_EAGLE3_DRAFT_TP"
    JOB_FILE="$SWEEP_JOB_FILE"
  )
  if [[ -n "$sweep_dependency" ]]; then
    sweep_cmd+=(
      BASELINE_SBATCH_DEPENDENCY="$sweep_dependency"
      ALLOW_MISSING_DRAFT_FOR_DEPENDENCY=true
    )
  fi
  sweep_cmd+=(bash "$SCRIPT_DIR/submit_trained_draft_spec_tokens_sweep.sh")
  run_or_print "trained_draft_spec_tokens_sweep" "${sweep_cmd[@]}"
fi

if [[ "$SUBMIT" == "true" || "$SUBMIT" == "True" ]]; then
  if [[ "$START_PIPELINE_WATCHER" == "true" || "$START_PIPELINE_WATCHER" == "True" ]]; then
    if [[ -n "$ARTIFACT_ROOT" ]]; then
      watcher_log="$ARTIFACT_ROOT/reports/eagle3_pipeline_watch.log"
      watcher_pid="$ARTIFACT_ROOT/reports/eagle3_pipeline_watch.pid"
      mkdir -p "$(dirname "$watcher_log")"
      nohup env \
        ARTIFACT_ROOT="$ARTIFACT_ROOT" \
        JOB_FILE="$job_file" \
        LOGS_DIR="$ROOT_DIR/logs" \
        BASE_MODEL="$BASE_MODEL" \
        MODELOPT_DIR="$MODELOPT_DIR" \
        VERIFIER_CONFIG_DIR="$VERIFIER_CONFIG_DIR" \
        REFERENCE_ARCH="$REFERENCE_ARCH" \
        ARCH_ENV_FILE="$ARCH_ENV_FILE" \
        CHAT_TEMPLATE="$CHAT_TEMPLATE" \
        CONTAINER="$CONTAINER" \
        MOUNTS="$MOUNTS" \
        INPUT_DATA="$INPUT_DATA" \
        HIDDEN_STATES_DIR="$HIDDEN_STATES_DIR" \
        HIDDEN_STATES_VALIDATION_JSON="$HIDDEN_STATES_VALIDATION_JSON" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        TRAINING_CKPT_VALIDATION_JSON="$TRAINING_CKPT_VALIDATION_JSON" \
        EXPORT_DIR="$EXPORT_DIR" \
        VLLM_DRAFT_DIR="$VLLM_DRAFT_DIR" \
        EXPORT_ARTIFACTS_JSON="$EXPORT_ARTIFACTS_JSON" \
        SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
        SBATCH_PARTITION="$SBATCH_PARTITION" \
        RUN_PILOT="$RUN_PILOT" \
        bash "$SCRIPT_DIR/watch_eagle3_pipeline_followup.sh" >> "$watcher_log" 2>&1 &
      echo $! > "$watcher_pid"
      echo "pipeline_watcher_pid=$(cat "$watcher_pid")"
      echo "pipeline_watcher_log=$watcher_log"
    else
      echo "WARN: START_PIPELINE_WATCHER=true but ARTIFACT_ROOT is empty; watcher not started" >&2
    fi
  fi
fi

if [[ "$SUBMIT" != "true" && "$SUBMIT" != "True" ]]; then
  echo "# dry run only. Set SUBMIT=true to submit jobs." >&2
fi
