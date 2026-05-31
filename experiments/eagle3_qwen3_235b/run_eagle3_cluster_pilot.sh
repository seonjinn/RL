#!/usr/bin/env bash
set -euo pipefail

# One-command cluster entrypoint for the Qwen3-235B Eagle3 pilot path.
#
# Default behavior is safe: discover inputs, capture provenance/reference
# reports, run bootstrap in dry-run mode, and create a handoff bundle. It does
# not submit Slurm jobs unless SUBMIT=true is set.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  ARTIFACT_ROOT=/path/to/qwen3_235b_eagle3 \\
  SBATCH_ACCOUNT=<account> \\
  bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh

Important env:
  SUBMIT=false|true                  Submit Slurm jobs only when true.
  RUN_PILOT=true|false               Use small pilot limits before full run.
  PREP_DRY_RUN=true|false            Materialize data/template only when false.
  DISCOVERY_ROOTS="path path ..."    Roots scanned for verifier/data/drafts.
  RUN_INPUT_DISCOVERY=true|false     Run discover_eagle3_run_inputs.py.
  SOURCE_DISCOVERY_ENV=true|false    Source generated eagle3_inputs.env.
  RUN_CLUSTER_PROBE=true|false       Probe Slurm/container/path substrate.
  RUN_CORPUS_STRATEGY=true|false     Report whether rollout/math/bootstrap corpus is appropriate.
  RUN_TRAINING_SCALE_PLAN=true|false Estimate pilot/calibration/production Eagle3 training scale.
  RUN_NEXT_ACTION_PLAN=true|false    Summarize exact next operator actions from generated reports.
  RUN_PIPELINE_SUBMIT_PREFLIGHT=true|false Check hidden-state/train/export submit readiness.
  RUN_HAYATE_WORKFLOW=true|false     Classify accessible Hayate/Hiso ModelOpt Eagle3 workflow files.
  RUN_UPSTREAM_DRIFT=true|false      Report local/upstream/Hayate ModelOpt drift.
  RUN_NEMO_RL_DRIFT=true|false       Report fixed/online Eagle3 support in SpecDec-RL.
  RUN_MODELOPT_LOSS_MASK_CHECK=true|false Validate ModelOpt TRT-LLM loss-mask patch.
  RUN_MODELOPT_PATCH=true|false      Export local ModelOpt patch bundle.
  COMPAT_MODELOPT_DIR=/path          Optional checkout for patch apply checks.
  RUN_BOOTSTRAP=true|false           Run bootstrap_eagle3_path.sh.
  RUN_HANDOFF=true|false             Create handoff bundle.

Typical cluster dry-run:
  ARTIFACT_ROOT=/lustre/.../qwen3_235b_eagle3 \\
  SBATCH_ACCOUNT=<account> \\
  bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh

Pilot submit after inspecting reports:
  SUBMIT=true PREP_DRY_RUN=false RUN_PILOT=true \\
  ARTIFACT_ROOT=/lustre/.../qwen3_235b_eagle3 \\
  SBATCH_ACCOUNT=<account> \\
  bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh
EOF
  exit 0
fi

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
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SUBMIT="${SUBMIT:-false}"
RUN_PILOT="${RUN_PILOT:-true}"
PREP_DRY_RUN="${PREP_DRY_RUN:-true}"

RUN_INPUT_DISCOVERY="${RUN_INPUT_DISCOVERY:-true}"
SOURCE_DISCOVERY_ENV="${SOURCE_DISCOVERY_ENV:-true}"
RUN_CLUSTER_PROBE="${RUN_CLUSTER_PROBE:-true}"
RUN_HAYATE_INVENTORY="${RUN_HAYATE_INVENTORY:-true}"
RUN_HAYATE_WORKFLOW="${RUN_HAYATE_WORKFLOW:-true}"
RUN_DRAFT_INVENTORY="${RUN_DRAFT_INVENTORY:-true}"
RUN_SPECFORGE_REFERENCE="${RUN_SPECFORGE_REFERENCE:-true}"
RUN_UPSTREAM_DRIFT="${RUN_UPSTREAM_DRIFT:-true}"
PROBE_UPSTREAM="${PROBE_UPSTREAM:-true}"
RUN_MODELOPT_LOSS_MASK_CHECK="${RUN_MODELOPT_LOSS_MASK_CHECK:-true}"
RUN_MODELOPT_PATCH="${RUN_MODELOPT_PATCH:-true}"
RUN_PROVENANCE="${RUN_PROVENANCE:-true}"
RUN_RL_SPECDEC_VALIDATION="${RUN_RL_SPECDEC_VALIDATION:-true}"
RUN_NEMO_RL_DRIFT="${RUN_NEMO_RL_DRIFT:-true}"
RUN_ROLLOUT_CAPTURE_VALIDATION="${RUN_ROLLOUT_CAPTURE_VALIDATION:-true}"
RUN_ROLLOUT_SUBMIT_PREFLIGHT="${RUN_ROLLOUT_SUBMIT_PREFLIGHT:-true}"
RUN_ROLLOUT_CAPTURE_JOB_ANALYSIS="${RUN_ROLLOUT_CAPTURE_JOB_ANALYSIS:-true}"
RUN_ROLLOUT_STATE_ADVANCE="${RUN_ROLLOUT_STATE_ADVANCE:-true}"
RUN_PIPELINE_SUBMIT_PREFLIGHT="${RUN_PIPELINE_SUBMIT_PREFLIGHT:-true}"
RUN_CORPUS_STRATEGY="${RUN_CORPUS_STRATEGY:-true}"
RUN_TRAINING_SCALE_PLAN="${RUN_TRAINING_SCALE_PLAN:-true}"
RUN_NEXT_ACTION_PLAN="${RUN_NEXT_ACTION_PLAN:-true}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-true}"
RUN_HANDOFF="${RUN_HANDOFF:-true}"
BOOTSTRAP_RUN_PROVENANCE="${BOOTSTRAP_RUN_PROVENANCE:-false}"

DISCOVERY_ROOTS="${DISCOVERY_ROOTS:-$ROOT_DIR /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso /lustre/fsw/portfolios/coreai/users/sna}"
NEMO_RL_CONFIG="${NEMO_RL_CONFIG:-$ROOT_DIR/grpo_qwen3_235b_swe.yaml}"
SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
EAGLE3_DRAFT_MODEL="${EAGLE3_DRAFT_MODEL:-nvidia/Qwen3-235B-A22B-Eagle3}"
EAGLE3_NUM_SPEC_TOKENS="${EAGLE3_NUM_SPEC_TOKENS:-3}"
EAGLE3_DRAFT_TP="${EAGLE3_DRAFT_TP:-1}"
HAYATE_MODEL_OPT_DIR="${HAYATE_MODEL_OPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer}"
HAYATE_NEMO_RL_DIR="${HAYATE_NEMO_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec}"
HAYATE_DRAFT_MODELS_DIR="${HAYATE_DRAFT_MODELS_DIR:-$HAYATE_NEMO_RL_DIR/models}"

INPUT_DISCOVERY_JSON="${INPUT_DISCOVERY_JSON:-$ARTIFACT_ROOT/eagle3_input_discovery.json}"
INPUT_DISCOVERY_MARKDOWN="${INPUT_DISCOVERY_MARKDOWN:-$ARTIFACT_ROOT/eagle3_input_discovery.md}"
INPUT_DISCOVERY_ENV="${INPUT_DISCOVERY_ENV:-$ARTIFACT_ROOT/eagle3_inputs.env}"
PROVENANCE_JSON="${PROVENANCE_JSON:-$REPORT_DIR/eagle3_provenance.json}"
PROVENANCE_MARKDOWN="${PROVENANCE_MARKDOWN:-$REPORT_DIR/eagle3_provenance.md}"
CLUSTER_PROBE_JSON="${CLUSTER_PROBE_JSON:-$REPORT_DIR/cluster_environment_probe.json}"
CLUSTER_PROBE_MARKDOWN="${CLUSTER_PROBE_MARKDOWN:-$REPORT_DIR/cluster_environment_probe.md}"
HAYATE_INVENTORY="${HAYATE_INVENTORY:-$REPORT_DIR/hayate_inventory.txt}"
HAYATE_WORKFLOW_JSON="${HAYATE_WORKFLOW_JSON:-$REPORT_DIR/hayate_modelopt_workflow.json}"
HAYATE_WORKFLOW_MARKDOWN="${HAYATE_WORKFLOW_MARKDOWN:-$REPORT_DIR/hayate_modelopt_workflow.md}"
DRAFT_INVENTORY_JSON="${DRAFT_INVENTORY_JSON:-$REPORT_DIR/eagle3_draft_config_inventory.json}"
DRAFT_INVENTORY_MARKDOWN="${DRAFT_INVENTORY_MARKDOWN:-$REPORT_DIR/eagle3_draft_config_inventory.md}"
SPECFORGE_REFERENCE_JSON="${SPECFORGE_REFERENCE_JSON:-$REPORT_DIR/specforge_reference.json}"
SPECFORGE_REFERENCE_MARKDOWN="${SPECFORGE_REFERENCE_MARKDOWN:-$REPORT_DIR/specforge_reference.md}"
UPSTREAM_DRIFT_JSON="${UPSTREAM_DRIFT_JSON:-$REPORT_DIR/modelopt_upstream_drift.json}"
UPSTREAM_DRIFT_MARKDOWN="${UPSTREAM_DRIFT_MARKDOWN:-$REPORT_DIR/modelopt_upstream_drift.md}"
MODELOPT_LOSS_MASK_JSON="${MODELOPT_LOSS_MASK_JSON:-$REPORT_DIR/modelopt_loss_mask_patch.json}"
MODELOPT_LOSS_MASK_MARKDOWN="${MODELOPT_LOSS_MASK_MARKDOWN:-$REPORT_DIR/modelopt_loss_mask_patch.md}"
MODELOPT_PATCH_DIR="${MODELOPT_PATCH_DIR:-$ARTIFACT_ROOT/patches/modelopt_eagle3_qwen3}"
MODELOPT_PATCH_MANIFEST="${MODELOPT_PATCH_MANIFEST:-$MODELOPT_PATCH_DIR/manifest.json}"
COMPAT_MODELOPT_DIR="${COMPAT_MODELOPT_DIR:-}"
READINESS_JSON="${READINESS_JSON:-$REPORT_DIR/eagle3_readiness.json}"
PIPELINE_ANALYSIS_JSON="${PIPELINE_ANALYSIS_JSON:-$REPORT_DIR/eagle3_pipeline_analysis.json}"
SWEEP_JSON="${SWEEP_JSON:-$REPORT_DIR/trained_draft_spec_tokens_sweep.json}"
COMPLETION_JSON="${COMPLETION_JSON:-$REPORT_DIR/eagle3_completion_audit.json}"
CONTAINER_PREFLIGHT_JSON="${CONTAINER_PREFLIGHT_JSON:-$REPORT_DIR/container_preflight_analysis.json}"
NEMO_RL_SPECDEC_JSON="${NEMO_RL_SPECDEC_JSON:-$REPORT_DIR/nemo_rl_specdec_integration.json}"
NEMO_RL_SPECDEC_MARKDOWN="${NEMO_RL_SPECDEC_MARKDOWN:-$REPORT_DIR/nemo_rl_specdec_integration.md}"
NEMO_RL_SPECDEC_ENV="${NEMO_RL_SPECDEC_ENV:-$REPORT_DIR/nemo_rl_specdec_overrides.env}"
NEMO_RL_DRIFT_JSON="${NEMO_RL_DRIFT_JSON:-$REPORT_DIR/nemo_rl_eagle3_drift.json}"
NEMO_RL_DRIFT_MARKDOWN="${NEMO_RL_DRIFT_MARKDOWN:-$REPORT_DIR/nemo_rl_eagle3_drift.md}"
ROLLOUT_CAPTURE_JSON="${ROLLOUT_CAPTURE_JSON:-$REPORT_DIR/rollout_capture_validation.json}"
ROLLOUT_CAPTURE_MARKDOWN="${ROLLOUT_CAPTURE_MARKDOWN:-$REPORT_DIR/rollout_capture_validation.md}"
ROLLOUT_CAPTURE_ENV="${ROLLOUT_CAPTURE_ENV:-$REPORT_DIR/rollout_capture.env}"
ROLLOUT_CAPTURE_ANALYSIS_JSON="${ROLLOUT_CAPTURE_ANALYSIS_JSON:-$REPORT_DIR/rollout_capture_analysis.json}"
ROLLOUT_CAPTURE_ANALYSIS_MARKDOWN="${ROLLOUT_CAPTURE_ANALYSIS_MARKDOWN:-$REPORT_DIR/rollout_capture_analysis.md}"
ROLLOUT_CAPTURE_JOB_JSON="${ROLLOUT_CAPTURE_JOB_JSON:-$REPORT_DIR/rollout_capture_job_analysis.json}"
ROLLOUT_CAPTURE_JOB_MARKDOWN="${ROLLOUT_CAPTURE_JOB_MARKDOWN:-$REPORT_DIR/rollout_capture_job_analysis.md}"
ROLLOUT_SUBMIT_PREFLIGHT_JSON="${ROLLOUT_SUBMIT_PREFLIGHT_JSON:-$REPORT_DIR/rollout_capture_submit_preflight.json}"
ROLLOUT_SUBMIT_PREFLIGHT_MARKDOWN="${ROLLOUT_SUBMIT_PREFLIGHT_MARKDOWN:-$REPORT_DIR/rollout_capture_submit_preflight.md}"
ROLLOUT_STATE_ADVANCE_JSON="${ROLLOUT_STATE_ADVANCE_JSON:-$REPORT_DIR/rollout_capture_state_advance.json}"
ROLLOUT_STATE_ADVANCE_MARKDOWN="${ROLLOUT_STATE_ADVANCE_MARKDOWN:-$REPORT_DIR/rollout_capture_state_advance.md}"
PIPELINE_SUBMIT_PREFLIGHT_JSON="${PIPELINE_SUBMIT_PREFLIGHT_JSON:-$REPORT_DIR/eagle3_pipeline_submit_preflight.json}"
PIPELINE_SUBMIT_PREFLIGHT_MARKDOWN="${PIPELINE_SUBMIT_PREFLIGHT_MARKDOWN:-$REPORT_DIR/eagle3_pipeline_submit_preflight.md}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}"
ROLLOUT_CONVERSATIONS="${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
CORPUS_STRATEGY_JSON="${CORPUS_STRATEGY_JSON:-$REPORT_DIR/corpus_strategy.json}"
CORPUS_STRATEGY_MARKDOWN="${CORPUS_STRATEGY_MARKDOWN:-$REPORT_DIR/corpus_strategy.md}"
TRAINING_SCALE_JSON="${TRAINING_SCALE_JSON:-$REPORT_DIR/eagle3_training_scale.json}"
TRAINING_SCALE_MARKDOWN="${TRAINING_SCALE_MARKDOWN:-$REPORT_DIR/eagle3_training_scale.md}"
TRAINING_CKPT_VALIDATION_JSON="${TRAINING_CKPT_VALIDATION_JSON:-$REPORT_DIR/eagle3_training_checkpoint.json}"
TRAINING_CKPT_VALIDATION_MARKDOWN="${TRAINING_CKPT_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_training_checkpoint.md}"
NEXT_ACTION_PLAN_JSON="${NEXT_ACTION_PLAN_JSON:-$REPORT_DIR/eagle3_next_actions.json}"
NEXT_ACTION_PLAN_MARKDOWN="${NEXT_ACTION_PLAN_MARKDOWN:-$REPORT_DIR/eagle3_next_actions.md}"
NEXT_ACTION_PLAN_VALIDATION_JSON="${NEXT_ACTION_PLAN_VALIDATION_JSON:-$REPORT_DIR/eagle3_next_actions_validation.json}"
NEXT_ACTION_PLAN_VALIDATION_MARKDOWN="${NEXT_ACTION_PLAN_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_next_actions_validation.md}"
NEXT_ACTION_TRANSITIONS_JSON="${NEXT_ACTION_TRANSITIONS_JSON:-$REPORT_DIR/eagle3_next_action_transitions.json}"
NEXT_ACTION_TRANSITIONS_MARKDOWN="${NEXT_ACTION_TRANSITIONS_MARKDOWN:-$REPORT_DIR/eagle3_next_action_transitions.md}"
EAGLE3_TARGET_CONTEXT="${EAGLE3_TARGET_CONTEXT:-swe_rl}"
SWE_REPO_ROOT="${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
HANDOFF_DIR="${HANDOFF_DIR:-$ARTIFACT_ROOT/handoff}"

mkdir -p "$ARTIFACT_ROOT" "$REPORT_DIR"

step "Cluster pilot entrypoint"
cat <<EOF
ARTIFACT_ROOT=$ARTIFACT_ROOT
MODELOPT_DIR=$MODELOPT_DIR
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
SUBMIT=$SUBMIT
RUN_PILOT=$RUN_PILOT
PREP_DRY_RUN=$PREP_DRY_RUN
RUN_INPUT_DISCOVERY=$RUN_INPUT_DISCOVERY
SOURCE_DISCOVERY_ENV=$SOURCE_DISCOVERY_ENV
RUN_CLUSTER_PROBE=$RUN_CLUSTER_PROBE
RUN_RL_SPECDEC_VALIDATION=$RUN_RL_SPECDEC_VALIDATION
RUN_NEMO_RL_DRIFT=$RUN_NEMO_RL_DRIFT
RUN_ROLLOUT_SUBMIT_PREFLIGHT=$RUN_ROLLOUT_SUBMIT_PREFLIGHT
RUN_ROLLOUT_STATE_ADVANCE=$RUN_ROLLOUT_STATE_ADVANCE
RUN_PIPELINE_SUBMIT_PREFLIGHT=$RUN_PIPELINE_SUBMIT_PREFLIGHT
RUN_TRAINING_SCALE_PLAN=$RUN_TRAINING_SCALE_PLAN
RUN_NEXT_ACTION_PLAN=$RUN_NEXT_ACTION_PLAN
RUN_HAYATE_WORKFLOW=$RUN_HAYATE_WORKFLOW
RUN_SPECFORGE_REFERENCE=$RUN_SPECFORGE_REFERENCE
RUN_UPSTREAM_DRIFT=$RUN_UPSTREAM_DRIFT
RUN_MODELOPT_LOSS_MASK_CHECK=$RUN_MODELOPT_LOSS_MASK_CHECK
RUN_MODELOPT_PATCH=$RUN_MODELOPT_PATCH
RUN_BOOTSTRAP=$RUN_BOOTSTRAP
RUN_HANDOFF=$RUN_HANDOFF
EOF

preserve_user_overrides() {
  USER_BASE_MODEL="${BASE_MODEL:-}"
  USER_VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-}"
  USER_TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-}"
  USER_MODE="${MODE:-}"
  USER_DATA_MODE="${DATA_MODE:-}"
  USER_INPUT_PATHS="${INPUT_PATHS:-}"
  USER_REFERENCE_ARCH="${REFERENCE_ARCH:-}"
  USER_ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"
  USER_CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
  USER_INPUT_DATA="${INPUT_DATA:-}"
  USER_HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-}"
  USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
  USER_EXPORT_DIR="${EXPORT_DIR:-}"
  USER_VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-}"
}

restore_user_overrides() {
  [[ -n "$USER_BASE_MODEL" ]] && BASE_MODEL="$USER_BASE_MODEL"
  [[ -n "$USER_VERIFIER_CONFIG_DIR" ]] && VERIFIER_CONFIG_DIR="$USER_VERIFIER_CONFIG_DIR"
  [[ -n "$USER_TOKENIZER_CONFIG" ]] && TOKENIZER_CONFIG="$USER_TOKENIZER_CONFIG"
  [[ -n "$USER_MODE" ]] && MODE="$USER_MODE"
  [[ -n "$USER_DATA_MODE" ]] && DATA_MODE="$USER_DATA_MODE"
  [[ -n "$USER_INPUT_PATHS" ]] && INPUT_PATHS="$USER_INPUT_PATHS"
  [[ -n "$USER_REFERENCE_ARCH" ]] && REFERENCE_ARCH="$USER_REFERENCE_ARCH"
  [[ -n "$USER_ARCH_ENV_FILE" ]] && ARCH_ENV_FILE="$USER_ARCH_ENV_FILE"
  [[ -n "$USER_CHAT_TEMPLATE" ]] && CHAT_TEMPLATE="$USER_CHAT_TEMPLATE"
  [[ -n "$USER_INPUT_DATA" ]] && INPUT_DATA="$USER_INPUT_DATA"
  [[ -n "$USER_HIDDEN_STATES_DIR" ]] && HIDDEN_STATES_DIR="$USER_HIDDEN_STATES_DIR"
  [[ -n "$USER_OUTPUT_DIR" ]] && OUTPUT_DIR="$USER_OUTPUT_DIR"
  [[ -n "$USER_EXPORT_DIR" ]] && EXPORT_DIR="$USER_EXPORT_DIR"
  [[ -n "$USER_VLLM_DRAFT_DIR" ]] && VLLM_DRAFT_DIR="$USER_VLLM_DRAFT_DIR"
}

if is_true "$RUN_INPUT_DISCOVERY"; then
  step "Input discovery"
  read -r -a discovery_roots <<< "$DISCOVERY_ROOTS"
  discover_cmd=(
    python3 "$EXP_DIR/discover_eagle3_run_inputs.py"
    "${discovery_roots[@]}"
    --artifact-root "$ARTIFACT_ROOT"
    --env-out "$INPUT_DISCOVERY_ENV"
    --markdown-out "$INPUT_DISCOVERY_MARKDOWN"
    --json-out "$INPUT_DISCOVERY_JSON"
  )
  print_cmd "${discover_cmd[@]}"
  set +e
  "${discover_cmd[@]}"
  discover_rc=$?
  set -e
  if [[ "$discover_rc" -ne 0 ]]; then
    echo "WARN: input discovery exited with $discover_rc. Inspect $INPUT_DISCOVERY_MARKDOWN." >&2
    if is_true "$SUBMIT"; then
      echo "SUBMIT=true requires successful input discovery or explicit env overrides." >&2
      exit "$discover_rc"
    fi
  fi
  if is_true "$SOURCE_DISCOVERY_ENV" && [[ -f "$INPUT_DISCOVERY_ENV" ]]; then
    preserve_user_overrides
    # shellcheck source=/dev/null
    source "$INPUT_DISCOVERY_ENV"
    restore_user_overrides
  fi
else
  step "Input discovery"
  echo "Skipped because RUN_INPUT_DISCOVERY=$RUN_INPUT_DISCOVERY"
fi

if is_true "$RUN_CLUSTER_PROBE"; then
  step "Cluster environment probe"
  probe_cmd=(
    python3 "$EXP_DIR/probe_cluster_environment.py"
    --artifact-root "$ARTIFACT_ROOT"
    --modelopt-dir "$MODELOPT_DIR"
    --sbatch-account "$SBATCH_ACCOUNT"
    --sbatch-partition "$SBATCH_PARTITION"
    --json-out "$CLUSTER_PROBE_JSON"
    --markdown-out "$CLUSTER_PROBE_MARKDOWN"
  )
  if [[ -n "${VERIFIER_CONFIG_DIR:-}" ]]; then
    probe_cmd+=(--verifier-config-dir "$VERIFIER_CONFIG_DIR")
  fi
  if [[ -n "${INPUT_DATA:-}" ]]; then
    probe_cmd+=(--input-data "$INPUT_DATA")
  fi
  if [[ -n "${CONTAINER:-}" ]]; then
    probe_cmd+=(--container "$CONTAINER")
  fi
  if [[ -n "${MOUNTS:-}" ]]; then
    probe_cmd+=(--mounts "$MOUNTS")
  fi
  if is_true "$SUBMIT"; then
    probe_cmd+=(--strict)
  fi
  print_cmd "${probe_cmd[@]}"
  if ! "${probe_cmd[@]}"; then
    echo "WARN: cluster environment probe returned nonzero; inspect $CLUSTER_PROBE_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "Cluster environment probe"
  echo "Skipped because RUN_CLUSTER_PROBE=$RUN_CLUSTER_PROBE"
fi

if is_true "$RUN_RL_SPECDEC_VALIDATION"; then
  step "NeMo-RL SpecDec integration validation"
  rl_specdec_cmd=(
    python3 "$EXP_DIR/validate_nemo_rl_specdec_integration.py"
    --config "$NEMO_RL_CONFIG"
    --draft-model "$EAGLE3_DRAFT_MODEL"
    --num-speculative-tokens "$EAGLE3_NUM_SPEC_TOKENS"
    --draft-tensor-parallel-size "$EAGLE3_DRAFT_TP"
    --specdec-rl-dir "$SPECDEC_RL_DIR"
    --markdown-out "$NEMO_RL_SPECDEC_MARKDOWN"
    --json-out "$NEMO_RL_SPECDEC_JSON"
    --env-out "$NEMO_RL_SPECDEC_ENV"
  )
  print_cmd "${rl_specdec_cmd[@]}"
  if ! "${rl_specdec_cmd[@]}"; then
    echo "WARN: NeMo-RL SpecDec integration validation returned nonzero; inspect $NEMO_RL_SPECDEC_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "NeMo-RL SpecDec integration validation"
  echo "Skipped because RUN_RL_SPECDEC_VALIDATION=$RUN_RL_SPECDEC_VALIDATION"
fi

if is_true "$RUN_NEMO_RL_DRIFT"; then
  step "NeMo-RL Eagle3 drift/support"
  nemo_rl_drift_cmd=(
    python3 "$EXP_DIR/check_nemo_rl_eagle3_drift.py"
    --nemo-rl-dir "$SPECDEC_RL_DIR"
    --markdown-out "$NEMO_RL_DRIFT_MARKDOWN"
    --json-out "$NEMO_RL_DRIFT_JSON"
  )
  if ! is_true "$PROBE_UPSTREAM"; then
    nemo_rl_drift_cmd+=(--no-probe-upstream --no-fetch-raw)
  fi
  print_cmd "${nemo_rl_drift_cmd[@]}"
  if ! "${nemo_rl_drift_cmd[@]}"; then
    echo "WARN: NeMo-RL Eagle3 drift/support report returned nonzero; inspect $NEMO_RL_DRIFT_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "NeMo-RL Eagle3 drift/support"
  echo "Skipped because RUN_NEMO_RL_DRIFT=$RUN_NEMO_RL_DRIFT"
fi

if is_true "$RUN_ROLLOUT_CAPTURE_VALIDATION"; then
  step "RL rollout capture validation"
  rollout_capture_cmd=(
    python3 "$EXP_DIR/validate_rollout_capture_config.py"
    --config "$NEMO_RL_CONFIG"
    --specdec-rl-dir "$SPECDEC_RL_DIR"
    --artifact-root "$ARTIFACT_ROOT"
    --chat-template "$CHAT_TEMPLATE"
    --markdown-out "$ROLLOUT_CAPTURE_MARKDOWN"
    --json-out "$ROLLOUT_CAPTURE_JSON"
    --env-out "$ROLLOUT_CAPTURE_ENV"
  )
  print_cmd "${rollout_capture_cmd[@]}"
  if ! "${rollout_capture_cmd[@]}"; then
    echo "WARN: RL rollout capture validation returned nonzero; inspect $ROLLOUT_CAPTURE_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "RL rollout capture validation"
  echo "Skipped because RUN_ROLLOUT_CAPTURE_VALIDATION=$RUN_ROLLOUT_CAPTURE_VALIDATION"
fi

if is_true "$RUN_ROLLOUT_SUBMIT_PREFLIGHT"; then
  step "RL rollout capture submit preflight"
  rollout_submit_preflight_cmd=(
    python3 "$EXP_DIR/preflight_rollout_capture_submit.py"
    --artifact-root "$ARTIFACT_ROOT"
    --repo-root "$SWE_REPO_ROOT"
    --config "$NEMO_RL_CONFIG"
    --chat-template "$CHAT_TEMPLATE"
    --rollout-log-dir "$ROLLOUT_LOG_DIR"
    --output-conversations "$ROLLOUT_CONVERSATIONS"
    --sbatch-account "$SBATCH_ACCOUNT"
    --sbatch-partition "$SBATCH_PARTITION"
    --markdown-out "$ROLLOUT_SUBMIT_PREFLIGHT_MARKDOWN"
    --json-out "$ROLLOUT_SUBMIT_PREFLIGHT_JSON"
  )
  print_cmd "${rollout_submit_preflight_cmd[@]}"
  if ! "${rollout_submit_preflight_cmd[@]}"; then
    echo "WARN: rollout submit preflight returned nonzero; inspect $ROLLOUT_SUBMIT_PREFLIGHT_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "RL rollout capture submit preflight"
  echo "Skipped because RUN_ROLLOUT_SUBMIT_PREFLIGHT=$RUN_ROLLOUT_SUBMIT_PREFLIGHT"
fi

if is_true "$RUN_ROLLOUT_CAPTURE_VALIDATION"; then
  step "RL rollout capture artifact analysis"
  rollout_analysis_cmd=(
    python3 "$EXP_DIR/analyze_rollout_capture.py"
    --artifact-root "$ARTIFACT_ROOT"
    --json-out "$ROLLOUT_CAPTURE_ANALYSIS_JSON"
    --markdown-out "$ROLLOUT_CAPTURE_ANALYSIS_MARKDOWN"
  )
  print_cmd "${rollout_analysis_cmd[@]}"
  if ! "${rollout_analysis_cmd[@]}"; then
    echo "WARN: rollout capture artifact analysis returned nonzero; inspect $ROLLOUT_CAPTURE_ANALYSIS_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
fi

if is_true "$RUN_ROLLOUT_CAPTURE_JOB_ANALYSIS"; then
  step "RL rollout capture job analysis"
  rollout_job_cmd=(
    python3 "$EXP_DIR/analyze_rollout_capture_job.py"
    --artifact-root "$ARTIFACT_ROOT"
    --repo-root "$SWE_REPO_ROOT"
    --json-out "$ROLLOUT_CAPTURE_JOB_JSON"
    --markdown-out "$ROLLOUT_CAPTURE_JOB_MARKDOWN"
  )
  print_cmd "${rollout_job_cmd[@]}"
  if ! "${rollout_job_cmd[@]}"; then
    echo "WARN: rollout capture job analysis returned nonzero; inspect $ROLLOUT_CAPTURE_JOB_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "RL rollout capture job analysis"
  echo "Skipped because RUN_ROLLOUT_CAPTURE_JOB_ANALYSIS=$RUN_ROLLOUT_CAPTURE_JOB_ANALYSIS"
fi

if is_true "$RUN_ROLLOUT_STATE_ADVANCE"; then
  step "RL rollout capture state advance"
  rollout_state_cmd=(
    python3 "$EXP_DIR/advance_rollout_capture_state.py"
    --artifact-root "$ARTIFACT_ROOT"
    --repo-root "$SWE_REPO_ROOT"
    --rollout-log-dir "$ROLLOUT_LOG_DIR"
    --output-data "$ROLLOUT_CONVERSATIONS"
    --target-context "$EAGLE3_TARGET_CONTEXT"
    --sbatch-account "$SBATCH_ACCOUNT"
    --sbatch-partition "$SBATCH_PARTITION"
    --markdown-out "$ROLLOUT_STATE_ADVANCE_MARKDOWN"
    --json-out "$ROLLOUT_STATE_ADVANCE_JSON"
  )
  print_cmd "${rollout_state_cmd[@]}"
  if ! "${rollout_state_cmd[@]}"; then
    echo "WARN: rollout state advance returned nonzero; inspect $ROLLOUT_STATE_ADVANCE_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "RL rollout capture state advance"
  echo "Skipped because RUN_ROLLOUT_STATE_ADVANCE=$RUN_ROLLOUT_STATE_ADVANCE"
fi

if is_true "$RUN_CORPUS_STRATEGY"; then
  step "Eagle3 corpus strategy"
  corpus_strategy_cmd=(
    python3 "$EXP_DIR/analyze_corpus_strategy.py"
    --artifact-root "$ARTIFACT_ROOT"
    --target-context "$EAGLE3_TARGET_CONTEXT"
    --input-data "${INPUT_DATA:-$ROLLOUT_CONVERSATIONS}"
    --rollout-capture-analysis-json "$ROLLOUT_CAPTURE_ANALYSIS_JSON"
    --markdown-out "$CORPUS_STRATEGY_MARKDOWN"
    --json-out "$CORPUS_STRATEGY_JSON"
  )
  print_cmd "${corpus_strategy_cmd[@]}"
  if ! "${corpus_strategy_cmd[@]}"; then
    echo "WARN: corpus strategy report returned nonzero; inspect $CORPUS_STRATEGY_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "Eagle3 corpus strategy"
  echo "Skipped because RUN_CORPUS_STRATEGY=$RUN_CORPUS_STRATEGY"
fi

if is_true "$RUN_TRAINING_SCALE_PLAN"; then
  step "Eagle3 training scale plan"
  scale_cmd=(
    python3 "$EXP_DIR/estimate_eagle3_training_scale.py"
    --artifact-root "$ARTIFACT_ROOT"
    --input-data "${INPUT_DATA:-$ROLLOUT_CONVERSATIONS}"
    --corpus-strategy-json "$CORPUS_STRATEGY_JSON"
    --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
    --target-context "$EAGLE3_TARGET_CONTEXT"
    --gpus "${TRAIN_GPUS_PER_NODE:-8}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --epochs "${NUM_TRAIN_EPOCHS:-1}"
    --markdown-out "$TRAINING_SCALE_MARKDOWN"
    --json-out "$TRAINING_SCALE_JSON"
  )
  print_cmd "${scale_cmd[@]}"
  "${scale_cmd[@]}"
else
  step "Eagle3 training scale plan"
  echo "Skipped because RUN_TRAINING_SCALE_PLAN=$RUN_TRAINING_SCALE_PLAN"
fi

if is_true "$RUN_PIPELINE_SUBMIT_PREFLIGHT"; then
  step "Eagle3 pipeline submit preflight"
  pipeline_submit_preflight_cmd=(
    python3 "$EXP_DIR/preflight_eagle3_pipeline_submit.py"
    --artifact-root "$ARTIFACT_ROOT"
    --input-data "${INPUT_DATA:-$ROLLOUT_CONVERSATIONS}"
    --hidden-states-dir "${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
    --output-dir "${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
    --trained-ckpt "${TRAINED_CKPT:-${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}}"
    --export-dir "${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
    --vllm-draft-dir "${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
    --verifier-config-dir "${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
    --chat-template "$CHAT_TEMPLATE"
    --modelopt-dir "$MODELOPT_DIR"
    --reference-arch "${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}"
    --arch-env-file "${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}"
    --container-preflight-json "$CONTAINER_PREFLIGHT_JSON"
    --corpus-strategy-json "$CORPUS_STRATEGY_JSON"
    --rollout-state-json "$ROLLOUT_STATE_ADVANCE_JSON"
    --sbatch-account "$SBATCH_ACCOUNT"
    --sbatch-partition "$SBATCH_PARTITION"
    --container "${CONTAINER:-}"
    --mounts "${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
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

if is_true "$RUN_HAYATE_INVENTORY"; then
  step "Hayate/Hiso inventory"
  hayate_cmd=(
    env
    MODEL_OPT_DIR="$HAYATE_MODEL_OPT_DIR"
    NEMO_RL_DIR="$HAYATE_NEMO_RL_DIR"
    DRAFT_MODELS_DIR="$HAYATE_DRAFT_MODELS_DIR"
    bash "$EXP_DIR/inventory_hayate_eagle3_artifacts.sh"
  )
  print_cmd "${hayate_cmd[@]}"
  if ! "${hayate_cmd[@]}" > "$HAYATE_INVENTORY" 2>&1; then
    echo "WARN: Hayate inventory failed; captured output at $HAYATE_INVENTORY" >&2
  fi
else
  step "Hayate/Hiso inventory"
  echo "Skipped because RUN_HAYATE_INVENTORY=$RUN_HAYATE_INVENTORY"
fi

if is_true "$RUN_HAYATE_WORKFLOW"; then
  step "Hayate/Hiso ModelOpt workflow"
  hayate_workflow_cmd=(
    python3 "$EXP_DIR/analyze_hayate_modelopt_workflow.py"
    --hayate-modelopt-dir "$HAYATE_MODEL_OPT_DIR"
    --markdown-out "$HAYATE_WORKFLOW_MARKDOWN"
    --json-out "$HAYATE_WORKFLOW_JSON"
  )
  print_cmd "${hayate_workflow_cmd[@]}"
  if ! "${hayate_workflow_cmd[@]}"; then
    echo "WARN: Hayate workflow report returned nonzero; inspect $HAYATE_WORKFLOW_MARKDOWN" >&2
  fi
else
  step "Hayate/Hiso ModelOpt workflow"
  echo "Skipped because RUN_HAYATE_WORKFLOW=$RUN_HAYATE_WORKFLOW"
fi

if is_true "$RUN_DRAFT_INVENTORY"; then
  step "Draft config inventory"
  draft_roots=(
    "$HAYATE_DRAFT_MODELS_DIR"
    "${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
    "${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
  )
  draft_cmd=(
    python3 "$EXP_DIR/inventory_eagle3_draft_configs.py"
    "${draft_roots[@]}"
    --markdown-out "$DRAFT_INVENTORY_MARKDOWN"
    --json-out "$DRAFT_INVENTORY_JSON"
  )
  print_cmd "${draft_cmd[@]}"
  "${draft_cmd[@]}"
else
  step "Draft config inventory"
  echo "Skipped because RUN_DRAFT_INVENTORY=$RUN_DRAFT_INVENTORY"
fi

if is_true "$RUN_SPECFORGE_REFERENCE"; then
  step "SpecForge/SGLang reference"
  specforge_cmd=(
    python3 "$EXP_DIR/analyze_specforge_reference.py"
    --markdown-out "$SPECFORGE_REFERENCE_MARKDOWN"
    --json-out "$SPECFORGE_REFERENCE_JSON"
  )
  print_cmd "${specforge_cmd[@]}"
  if ! "${specforge_cmd[@]}"; then
    echo "WARN: SpecForge reference report returned nonzero; inspect $SPECFORGE_REFERENCE_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "SpecForge/SGLang reference"
  echo "Skipped because RUN_SPECFORGE_REFERENCE=$RUN_SPECFORGE_REFERENCE"
fi

if is_true "$RUN_UPSTREAM_DRIFT"; then
  step "ModelOpt upstream drift"
  drift_cmd=(
    python3 "$EXP_DIR/check_modelopt_upstream_drift.py"
    --modelopt-dir "$MODELOPT_DIR"
    --hayate-modelopt-dir "$HAYATE_MODEL_OPT_DIR"
    --json-out "$UPSTREAM_DRIFT_JSON"
    --markdown-out "$UPSTREAM_DRIFT_MARKDOWN"
  )
  if ! is_true "$PROBE_UPSTREAM"; then
    drift_cmd+=(--no-probe-upstream)
  fi
  print_cmd "${drift_cmd[@]}"
  if ! "${drift_cmd[@]}"; then
    echo "WARN: ModelOpt upstream drift report returned nonzero; inspect $UPSTREAM_DRIFT_MARKDOWN" >&2
  fi
else
  step "ModelOpt upstream drift"
  echo "Skipped because RUN_UPSTREAM_DRIFT=$RUN_UPSTREAM_DRIFT"
fi

if is_true "$RUN_MODELOPT_LOSS_MASK_CHECK"; then
  step "ModelOpt loss-mask patch validation"
  loss_mask_cmd=(
    python3 "$EXP_DIR/validate_modelopt_loss_mask_patch.py"
    --modelopt-dir "$MODELOPT_DIR"
    --json-out "$MODELOPT_LOSS_MASK_JSON"
    --markdown-out "$MODELOPT_LOSS_MASK_MARKDOWN"
  )
  print_cmd "${loss_mask_cmd[@]}"
  if ! "${loss_mask_cmd[@]}"; then
    echo "WARN: ModelOpt loss-mask patch validation failed; inspect $MODELOPT_LOSS_MASK_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
else
  step "ModelOpt loss-mask patch validation"
  echo "Skipped because RUN_MODELOPT_LOSS_MASK_CHECK=$RUN_MODELOPT_LOSS_MASK_CHECK"
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
    --export-artifacts-json "${EXPORT_ARTIFACTS_JSON:-$REPORT_DIR/eagle3_export_artifacts.json}"
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

if is_true "$RUN_MODELOPT_PATCH"; then
  step "ModelOpt patch bundle"
  patch_cmd=(
    python3 "$EXP_DIR/export_modelopt_eagle3_patch_bundle.py"
    --modelopt-dir "$MODELOPT_DIR"
    --out-dir "$MODELOPT_PATCH_DIR"
  )
  if [[ -n "$COMPAT_MODELOPT_DIR" ]]; then
    patch_cmd+=(--compat-modelopt-dir "$COMPAT_MODELOPT_DIR")
  fi
  print_cmd "${patch_cmd[@]}"
  if ! "${patch_cmd[@]}"; then
    echo "WARN: ModelOpt patch bundle export returned nonzero; inspect $MODELOPT_PATCH_DIR" >&2
  fi
else
  step "ModelOpt patch bundle"
  echo "Skipped because RUN_MODELOPT_PATCH=$RUN_MODELOPT_PATCH"
fi

if is_true "$RUN_PROVENANCE"; then
  step "Provenance capture"
  provenance_cmd=(
    python3 "$EXP_DIR/collect_eagle3_provenance.py"
    --artifact-root "$ARTIFACT_ROOT"
    --modelopt-dir "$MODELOPT_DIR"
    --hayate-modelopt-dir "$HAYATE_MODEL_OPT_DIR"
    --hayate-nemo-rl-dir "$HAYATE_NEMO_RL_DIR"
    --hayate-draft-models-dir "$HAYATE_DRAFT_MODELS_DIR"
    --verifier-config-dir "${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
    --input-data "${INPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
    --hidden-states-dir "${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
    --output-dir "${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
    --export-dir "${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
    --vllm-draft-dir "${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
    --json-out "$PROVENANCE_JSON"
    --markdown-out "$PROVENANCE_MARKDOWN"
  )
  print_cmd "${provenance_cmd[@]}"
  "${provenance_cmd[@]}"
else
  step "Provenance capture"
  echo "Skipped because RUN_PROVENANCE=$RUN_PROVENANCE"
fi

if is_true "$RUN_BOOTSTRAP"; then
  step "Bootstrap dry-run/submission"
  bootstrap_cmd=(
    env
    ARTIFACT_ROOT="$ARTIFACT_ROOT"
    MODELOPT_DIR="$MODELOPT_DIR"
    SUBMIT="$SUBMIT"
    RUN_PILOT="$RUN_PILOT"
    PREP_DRY_RUN="$PREP_DRY_RUN"
    SBATCH_ACCOUNT="$SBATCH_ACCOUNT"
    SBATCH_PARTITION="$SBATCH_PARTITION"
    RUN_PROVENANCE="$BOOTSTRAP_RUN_PROVENANCE"
    RUN_TRAINING_SCALE_PLAN="$RUN_TRAINING_SCALE_PLAN"
    BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
    VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
    TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-}"
    MODE="${MODE:-${DATA_MODE:-discover}}"
    INPUT_PATHS="${INPUT_PATHS:-}"
    REFERENCE_ARCH="${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}"
    ARCH_ENV_FILE="${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}"
    CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
    INPUT_DATA="${INPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"
    HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
    OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
    TRAINING_CKPT_VALIDATION_JSON="$TRAINING_CKPT_VALIDATION_JSON"
    TRAINING_CKPT_VALIDATION_MARKDOWN="$TRAINING_CKPT_VALIDATION_MARKDOWN"
    EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
    VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
    CONTAINER_PREFLIGHT_JSON="$CONTAINER_PREFLIGHT_JSON"
    MODELOPT_LOSS_MASK_JSON="$MODELOPT_LOSS_MASK_JSON"
    NEMO_RL_SPECDEC_JSON="$NEMO_RL_SPECDEC_JSON"
    NEMO_RL_DRIFT_JSON="$NEMO_RL_DRIFT_JSON"
    ROLLOUT_CAPTURE_JSON="$ROLLOUT_CAPTURE_JSON"
    ROLLOUT_CAPTURE_ANALYSIS_JSON="$ROLLOUT_CAPTURE_ANALYSIS_JSON"
    ROLLOUT_CAPTURE_JOB_JSON="$ROLLOUT_CAPTURE_JOB_JSON"
    ROLLOUT_SUBMIT_PREFLIGHT_JSON="$ROLLOUT_SUBMIT_PREFLIGHT_JSON"
    SWE_REPO_ROOT="$SWE_REPO_ROOT"
    CORPUS_STRATEGY_JSON="$CORPUS_STRATEGY_JSON"
    TRAINING_SCALE_JSON="$TRAINING_SCALE_JSON"
    EAGLE3_TARGET_CONTEXT="$EAGLE3_TARGET_CONTEXT"
    bash "$EXP_DIR/bootstrap_eagle3_path.sh"
  )
  print_cmd "${bootstrap_cmd[@]}"
  "${bootstrap_cmd[@]}"
else
  step "Bootstrap dry-run/submission"
  echo "Skipped because RUN_BOOTSTRAP=$RUN_BOOTSTRAP"
fi

if is_true "$RUN_HANDOFF"; then
  step "Handoff bundle"
  handoff_cmd=(
    python3 "$EXP_DIR/create_eagle3_handoff_bundle.py"
    --out-dir "$HANDOFF_DIR"
    --artifact-root "$ARTIFACT_ROOT"
    --sbatch-account "$SBATCH_ACCOUNT"
    --provenance-json "$PROVENANCE_JSON"
    --input-discovery-json "$INPUT_DISCOVERY_JSON"
    --cluster-probe-json "$CLUSTER_PROBE_JSON"
    --upstream-drift-json "$UPSTREAM_DRIFT_JSON"
    --modelopt-loss-mask-json "$MODELOPT_LOSS_MASK_JSON"
    --modelopt-patch-manifest "$MODELOPT_PATCH_MANIFEST"
    --readiness-json "$READINESS_JSON"
    --container-preflight-json "$CONTAINER_PREFLIGHT_JSON"
    --nemo-rl-specdec-json "$NEMO_RL_SPECDEC_JSON"
    --nemo-rl-drift-json "$NEMO_RL_DRIFT_JSON"
    --rollout-capture-json "$ROLLOUT_CAPTURE_JSON"
    --rollout-capture-analysis-json "$ROLLOUT_CAPTURE_ANALYSIS_JSON"
    --rollout-capture-job-json "$ROLLOUT_CAPTURE_JOB_JSON"
    --rollout-submit-preflight-json "$ROLLOUT_SUBMIT_PREFLIGHT_JSON"
    --rollout-state-advance-json "$ROLLOUT_STATE_ADVANCE_JSON"
    --corpus-strategy-json "$CORPUS_STRATEGY_JSON"
    --training-scale-json "$TRAINING_SCALE_JSON"
    --next-action-plan-json "$NEXT_ACTION_PLAN_JSON"
    --next-action-plan-validation-json "$NEXT_ACTION_PLAN_VALIDATION_JSON"
    --next-action-transitions-json "$NEXT_ACTION_TRANSITIONS_JSON"
    --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
    --specforge-reference-json "$SPECFORGE_REFERENCE_JSON"
    --hayate-workflow-json "$HAYATE_WORKFLOW_JSON"
    --pipeline-analysis-json "$PIPELINE_ANALYSIS_JSON"
    --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON"
    --export-artifacts-json "${EXPORT_ARTIFACTS_JSON:-$REPORT_DIR/eagle3_export_artifacts.json}"
    --sweep-json "$SWEEP_JSON"
    --completion-json "$COMPLETION_JSON"
    --hayate-inventory "$HAYATE_INVENTORY"
    --draft-inventory-json "$DRAFT_INVENTORY_JSON"
  )
  print_cmd "${handoff_cmd[@]}"
  "${handoff_cmd[@]}"
else
  step "Handoff bundle"
  echo "Skipped because RUN_HANDOFF=$RUN_HANDOFF"
fi

step "Next"
cat <<EOF
Reports:
  discovery:  $INPUT_DISCOVERY_MARKDOWN
  provenance: $PROVENANCE_MARKDOWN
  scale:      $TRAINING_SCALE_MARKDOWN
  next plan:  $NEXT_ACTION_PLAN_MARKDOWN
  next val:   $NEXT_ACTION_PLAN_VALIDATION_MARKDOWN
  transitions:$NEXT_ACTION_TRANSITIONS_MARKDOWN
  readiness:  $READINESS_JSON
  handoff:    $HANDOFF_DIR/RUNBOOK.md

If this was a dry-run, inspect reports and rerun with:
  SUBMIT=true PREP_DRY_RUN=false RUN_PILOT=true ARTIFACT_ROOT=$ARTIFACT_ROOT SBATCH_ACCOUNT=<account> bash $0
EOF
