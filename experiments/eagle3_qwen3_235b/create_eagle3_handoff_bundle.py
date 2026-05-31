#!/usr/bin/env python3
"""Create a handoff bundle for the Qwen3-235B Eagle3 workstream.

The bundle is intentionally file-based: it does not submit jobs or require GPU
access. It gathers the current repo runbook, optional reports, a command sheet,
and a machine-readable manifest so another teammate can continue from the same
state on the cluster.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_REMOTE_HOST = "oci-hsg-cs-001-vscode-02"
DEFAULT_REMOTE_WORKDIR = "/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap"
DEFAULT_REMOTE_ARTIFACT_ROOT = "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3"
DEFAULT_SBATCH_ACCOUNT = "coreai_dlalgo_nemorl"
DUMMY_SBATCH_ACCOUNTS = {"", "dummy", "<account>"}

DEFAULT_COMMANDS = [
    (
        "0_restore_materialized_static_inputs",
        """copy_static_input() {
  local src="$1"
  local dst="$2"
  if [[ -f "$HANDOFF_DIR/$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$HANDOFF_DIR/$src" "$dst"
    echo "restored $dst"
  else
    echo "WARN: handoff static input not found: $HANDOFF_DIR/$src" >&2
  fi
}

copy_static_input materialized_verifier_config_config.json "$VERIFIER_CONFIG_DIR/config.json"
copy_static_input materialized_generation_config_generation_config.json "$VERIFIER_CONFIG_DIR/generation_config.json"
copy_static_input materialized_tokenizer_config_tokenizer_config.json "$VERIFIER_CONFIG_DIR/tokenizer_config.json"
copy_static_input materialized_chat_template_qwen3_generation_template.jinja2 "$CHAT_TEMPLATE"
copy_static_input materialized_chat_template_mask_validation_qwen3_generation_template.mask_validation.json "${CHAT_TEMPLATE%.jinja2}.mask_validation.json"
copy_static_input materialized_architecture_json_eagle3_architecture.json "$REFERENCE_ARCH"
copy_static_input materialized_architecture_env_eagle3_architecture.env "$ARCH_ENV_FILE"
copy_static_input materialized_architecture_dotlist_eagle3_architecture.dotlist "$ARTIFACT_ROOT/architecture/eagle3_architecture.dotlist"
""",
    ),
    (
        "0_collect_provenance",
        """python3 experiments/eagle3_qwen3_235b/collect_eagle3_provenance.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --verifier-config-dir "$VERIFIER_CONFIG_DIR" \\
  --input-data "$INPUT_DATA" \\
  --hidden-states-dir "$HIDDEN_STATES_DIR" \\
  --output-dir "$OUTPUT_DIR" \\
  --export-dir "$EXPORT_DIR" \\
  --vllm-draft-dir "$VLLM_DRAFT_DIR" \\
  --json-out "$ARTIFACT_ROOT/reports/eagle3_provenance.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_provenance.md"
""",
    ),
    (
        "1_discover_inputs",
        """python3 experiments/eagle3_qwen3_235b/discover_eagle3_run_inputs.py \\
  /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso \\
  /lustre/fsw/portfolios/coreai/users/sna \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --env-out "$ARTIFACT_ROOT/eagle3_inputs.env" \\
  --markdown-out "$ARTIFACT_ROOT/eagle3_input_discovery.md" \\
  --json-out "$ARTIFACT_ROOT/eagle3_input_discovery.json"
source "$ARTIFACT_ROOT/eagle3_inputs.env"
""",
    ),
    (
        "1b_cluster_pilot_entrypoint",
        """SUBMIT=false RUN_PILOT=true PREP_DRY_RUN=true \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \\
SBATCH_PARTITION="$SBATCH_PARTITION" \\
bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh
""",
    ),
    (
        "1b_remote_cluster_pilot_entrypoint",
        """PRINT_ONLY=true \\
REMOTE_HOST="${REMOTE_HOST:-oci-hsg-cs-001-vscode-02}" \\
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap}" \\
REMOTE_ARTIFACT_ROOT="$REMOTE_ARTIFACT_ROOT" \\
SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \\
SBATCH_PARTITION="$SBATCH_PARTITION" \\
bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh
""",
    ),
    (
        "1b_remote_host_probe",
        """python3 experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py \\
  --include-ssh-config-hosts \\
  --hosts oci-hsg-cs-001-vscode-02 oci-hsg-cs-001-vscode-01 oci-hsg-cs-001-vscode-03 oci-hsg-cs-001-login-01.nvidia.com oci-hsg \\
  --remote-workdir "${REMOTE_WORKDIR:-/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap}" \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --json-out "$ARTIFACT_ROOT/reports/eagle3_remote_host_probe.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_remote_host_probe.md"
""",
    ),
    (
        "1b_remote_access_diagnostics",
        """python3 experiments/eagle3_qwen3_235b/diagnose_eagle3_remote_access.py \\
  --remote-host-probe-json "$ARTIFACT_ROOT/reports/eagle3_remote_host_probe.json" \\
  --json-out "$ARTIFACT_ROOT/reports/eagle3_remote_access_diagnostics.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_remote_access_diagnostics.md"
""",
    ),
    (
        "1b_cluster_environment_probe",
        """python3 experiments/eagle3_qwen3_235b/probe_cluster_environment.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --verifier-config-dir "$VERIFIER_CONFIG_DIR" \\
  --input-data "$INPUT_DATA" \\
  --container "$CONTAINER" \\
  --mounts "$MOUNTS" \\
  --sbatch-account "$SBATCH_ACCOUNT" \\
  --sbatch-partition "$SBATCH_PARTITION" \\
  --json-out "$ARTIFACT_ROOT/reports/cluster_environment_probe.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/cluster_environment_probe.md"
""",
    ),
    (
        "1b_container_preflight",
        """SUBMIT=false \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \\
SBATCH_PARTITION="$SBATCH_PARTITION" \\
MODELOPT_DIR="$MODELOPT_DIR" \\
VERIFIER_CONFIG_DIR="$VERIFIER_CONFIG_DIR" \\
INPUT_DATA="$INPUT_DATA" \\
CHAT_TEMPLATE="$CHAT_TEMPLATE" \\
CONTAINER="$CONTAINER" \\
MOUNTS="$MOUNTS" \\
PREFLIGHT_JSON="$CONTAINER_PREFLIGHT_PIPELINE_JSON" \\
PREFLIGHT_MARKDOWN="$CONTAINER_PREFLIGHT_PIPELINE_MARKDOWN" \\
bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh

python3 experiments/eagle3_qwen3_235b/analyze_container_preflight.py \\
  --job-file latest_eagle3_container_preflight_job.txt \\
  --logs-dir logs \\
  --cluster-probe-json "$ARTIFACT_ROOT/reports/container_preflight_cluster_probe.json" \\
  --pipeline-preflight-json "$CONTAINER_PREFLIGHT_PIPELINE_JSON" \\
  --pipeline-preflight-markdown "$CONTAINER_PREFLIGHT_PIPELINE_MARKDOWN" \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --verifier-config-dir "$VERIFIER_CONFIG_DIR" \\
  --input-data "$INPUT_DATA" \\
  --chat-template "$CHAT_TEMPLATE" \\
  --container "$CONTAINER" \\
  --mounts "$MOUNTS" \\
  --sbatch-account "$SBATCH_ACCOUNT" \\
  --sbatch-partition "$SBATCH_PARTITION" \\
  --markdown-out "$ARTIFACT_ROOT/reports/container_preflight_analysis.md" \\
  --json-out "$CONTAINER_PREFLIGHT_JSON"
""",
    ),
    (
        "1c_validate_specdec_remote_patch_bundle",
        """python3 experiments/eagle3_qwen3_235b/validate_specdec_rl_remote_patch_bundle.py \\
  --patch-root "${SPECDEC_REMOTE_PATCH_ROOT:-$HANDOFF_DIR/specdec_rl_remote_patches}" \\
  --target-specdec-rl-dir "$SPECDEC_RL_DIR" \\
  --json-out "$SPECDEC_REMOTE_PATCH_BUNDLE_JSON" \\
  --markdown-out "$SPECDEC_REMOTE_PATCH_BUNDLE_MARKDOWN"

if is_true "${APPLY_SPECDEC_REMOTE_PATCHES:-false}"; then
  rsync -a --exclude "__pycache__" --exclude "*.pyc" \\
    "${SPECDEC_REMOTE_PATCH_ROOT:-$HANDOFF_DIR/specdec_rl_remote_patches}/" \\
    "$SPECDEC_RL_DIR/"
  python3 experiments/eagle3_qwen3_235b/validate_specdec_rl_remote_patch_bundle.py \\
    --patch-root "${SPECDEC_REMOTE_PATCH_ROOT:-$HANDOFF_DIR/specdec_rl_remote_patches}" \\
    --target-specdec-rl-dir "$SPECDEC_RL_DIR" \\
    --require-target-applied \\
    --json-out "$SPECDEC_REMOTE_PATCH_BUNDLE_JSON" \\
    --markdown-out "$SPECDEC_REMOTE_PATCH_BUNDLE_MARKDOWN"
else
  echo "SpecDec-RL overlay not copied. Set APPLY_SPECDEC_REMOTE_PATCHES=true after reviewing $SPECDEC_REMOTE_PATCH_BUNDLE_MARKDOWN."
fi
""",
    ),
    (
        "1c_rollout_capture_gate",
        """python3 experiments/eagle3_qwen3_235b/validate_rollout_capture_config.py \\
  --config "${NEMO_RL_CONFIG:-grpo_qwen3_235b_swe.yaml}" \\
  --specdec-rl-dir "${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --chat-template "$CHAT_TEMPLATE" \\
  --markdown-out "$ARTIFACT_ROOT/reports/rollout_capture_validation.md" \\
  --json-out "$ROLLOUT_CAPTURE_JSON" \\
  --env-out "$ARTIFACT_ROOT/reports/rollout_capture.env"

APPLY=false \\
SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
bash experiments/eagle3_qwen3_235b/apply_specdec_rl_rollout_role_logging_patch.sh
""",
    ),
    (
        "1d_rollout_capture_smoke_plan",
        """DRY_RUN=true \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}" \\
OUTPUT_CONVERSATIONS="${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
bash experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh
""",
    ),
    (
        "1d_pre_submit_rollout_capture",
        """python3 experiments/eagle3_qwen3_235b/preflight_rollout_capture_submit.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --repo-root "$SWE_REPO_ROOT" \\
  --config "$NEMO_RL_CONFIG" \\
  --chat-template "$CHAT_TEMPLATE" \\
  --rollout-log-dir "${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}" \\
  --output-conversations "${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
  --sbatch-account "$SBATCH_ACCOUNT" \\
  --sbatch-partition "$SBATCH_PARTITION" \\
  --markdown-out "$ARTIFACT_ROOT/reports/rollout_capture_submit_preflight.md" \\
  --json-out "$ROLLOUT_SUBMIT_PREFLIGHT_JSON"
""",
    ),
    (
        "1e_materialize_rollout_corpus",
        """ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}" \\
OUTPUT_DATA="${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
bash experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh
""",
    ),
    (
        "1f_analyze_rollout_capture_artifacts",
        """python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --rollout-log-dir "${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}" \\
  --output-data "${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/rollout_capture_analysis.md" \\
  --json-out "$ROLLOUT_CAPTURE_ANALYSIS_JSON"
""",
    ),
    (
        "1g_analyze_rollout_capture_job",
        """python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --repo-root "${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --rollout-log-dir "${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}" \\
  --output-data "${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/rollout_capture_job_analysis.md" \\
  --json-out "$ROLLOUT_CAPTURE_JOB_JSON"
""",
    ),
    (
        "1h_advance_rollout_capture_state",
        """python3 experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --repo-root "$SWE_REPO_ROOT" \\
  --rollout-log-dir "${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke}" \\
  --output-data "${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
  --target-context "${EAGLE3_TARGET_CONTEXT:-swe_rl}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/rollout_capture_state_advance.md" \\
  --json-out "$ROLLOUT_STATE_ADVANCE_JSON"
""",
    ),
    (
        "1i_corpus_strategy",
        """python3 experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --target-context "${EAGLE3_TARGET_CONTEXT:-swe_rl}" \\
  --input-data "$INPUT_DATA" \\
  --rollout-capture-analysis-json "$ROLLOUT_CAPTURE_ANALYSIS_JSON" \\
  --markdown-out "$ARTIFACT_ROOT/reports/corpus_strategy.md" \\
  --json-out "$CORPUS_STRATEGY_JSON"
""",
    ),
    (
        "1j_training_scale_plan",
        """if [[ -f "$SLURM_CAPACITY_ENV" ]]; then
  source "$SLURM_CAPACITY_ENV"
fi

python3 experiments/eagle3_qwen3_235b/estimate_eagle3_training_scale.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --input-data "${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
  --corpus-strategy-json "$CORPUS_STRATEGY_JSON" \\
  --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON" \\
  --target-context "${EAGLE3_TARGET_CONTEXT:-swe_rl}" \\
  --gpus "${TRAIN_GPUS_PER_NODE:-8}" \\
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \\
  --epochs "${NUM_TRAIN_EPOCHS:-1}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_training_scale.md" \\
  --json-out "$TRAINING_SCALE_JSON"
""",
    ),
    (
        "1j_next_action_plan",
        """python3 experiments/eagle3_qwen3_235b/plan_eagle3_next_actions.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --container-preflight-json "$CONTAINER_PREFLIGHT_JSON" \\
  --rollout-submit-preflight-json "$ROLLOUT_SUBMIT_PREFLIGHT_JSON" \\
  --rollout-state-json "$ROLLOUT_STATE_ADVANCE_JSON" \\
  --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON" \\
  --pipeline-analysis-json "$PIPELINE_ANALYSIS_JSON" \\
  --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON" \\
  --export-artifacts-json "$EXPORT_ARTIFACTS_JSON" \\
  --sweep-json "$SWEEP_JSON" \\
  --training-scale-json "$TRAINING_SCALE_JSON" \\
  --modelopt-loss-mask-json "$MODELOPT_LOSS_MASK_JSON" \\
  --nemo-rl-drift-json "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_drift.json" \\
  --readiness-json "$ARTIFACT_ROOT/reports/eagle3_readiness.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_next_actions.md" \\
  --json-out "$NEXT_ACTION_PLAN_JSON"
""",
    ),
    (
        "1j_training_path_manifest",
        """python3 experiments/eagle3_qwen3_235b/build_eagle3_training_path_manifest.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --remote-host-probe-json "$ARTIFACT_ROOT/reports/eagle3_remote_host_probe.json" \\
  --remote-access-diagnostics-json "$ARTIFACT_ROOT/reports/eagle3_remote_access_diagnostics.json" \\
  --upstream-drift-json "$ARTIFACT_ROOT/reports/modelopt_upstream_drift.json" \\
  --json-out "$TRAINING_PATH_MANIFEST_JSON" \\
  --markdown-out "$TRAINING_PATH_MANIFEST_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_training_path_manifest.py \\
  --json-out "$TRAINING_PATH_MANIFEST_VALIDATION_JSON" \\
  --markdown-out "$TRAINING_PATH_MANIFEST_VALIDATION_MARKDOWN"
""",
    ),
    (
        "1j_operator_resume_state",
        """EXECUTE_SAFE_ACTIONS=false \\
EXECUTE_SLURM_ACTIONS=false \\
SLURM_ACTION_IDS="submit_vllm_source_build submit_source_vllm_abi_probe submit_megatron_compat_probe submit_container_preflight" \\
RUN_AFTER_SLURM_ACTIONS=false \\
ALLOW_HEAVY_GPU_ACTIONS=false \\
RUN_FULL_REFRESH=true \\
SAFE_ACTION_IDS="probe_remote_hosts poll_megatron_compat_probe" \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
PROBE_JOB_ID="${PROBE_JOB_ID:-2867766}" \\
bash experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh
""",
    ),
    (
        "1j_operator_safe_actions",
        """# No Slurm submission. This scopes ready-submit preflight via --action-ids
# to only the allowlisted non-Slurm/non-heavy actions before executing them.
EXECUTE_SAFE_ACTIONS=true \\
SAFE_ACTION_IDS="probe_remote_hosts poll_megatron_compat_probe" \\
RUN_AFTER_SAFE_ACTIONS=false \\
EXECUTE_SLURM_ACTIONS=false \\
RUN_AFTER_SLURM_ACTIONS=false \\
ALLOW_HEAVY_GPU_ACTIONS=false \\
REQUIRE_SLURM=false \\
RUN_FULL_REFRESH=false \\
OPERATOR_READY_SUBMIT_PREFLIGHT_JSON="$OPERATOR_SAFE_ACTIONS_PREFLIGHT_JSON" \\
OPERATOR_READY_SUBMIT_PREFLIGHT_MARKDOWN="$OPERATOR_SAFE_ACTIONS_PREFLIGHT_MARKDOWN" \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
PROBE_JOB_ID="${PROBE_JOB_ID:-2867766}" \\
bash experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh

# After the safe actions complete, regenerate the full no-submit evidence matrix.
python3 experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --json-out "$OPERATOR_STATE_REFRESH_JSON" \\
  --markdown-out "$OPERATOR_STATE_REFRESH_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_state_refresh.py \\
  --json-out "$OPERATOR_STATE_REFRESH_VALIDATION_JSON" \\
  --markdown-out "$OPERATOR_STATE_REFRESH_VALIDATION_MARKDOWN"
""",
    ),
    (
        "1j_validate_preflight_robustness",
        """python3 experiments/eagle3_qwen3_235b/validate_eagle3_preflight_robustness.py \\
  --json-out "$PREFLIGHT_ROBUSTNESS_VALIDATION_JSON" \\
  --markdown-out "$PREFLIGHT_ROBUSTNESS_VALIDATION_MARKDOWN"
""",
    ),
    (
        "1j_validate_modelopt_recipe_overrides",
        """python3 experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py \\
  --wrapper experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh \\
  --training-mode offline \\
  --modelopt-dir "${MODELOPT_DIR:-Model-Optimizer}" \\
  --json-out "$MODELOPT_RECIPE_OVERRIDES_JSON" \\
  --markdown-out "$MODELOPT_RECIPE_OVERRIDES_MARKDOWN"
""",
    ),
    (
        "1j_megatron_probe_followup",
        """python3 experiments/eagle3_qwen3_235b/validate_megatron_probe_followup.py \\
  --json-out "$MEGATRON_PROBE_FOLLOWUP_VALIDATION_JSON" \\
  --markdown-out "$MEGATRON_PROBE_FOLLOWUP_VALIDATION_MARKDOWN"

# Poll the recorded Megatron compatibility probe and, if PASS, print the next
# balanced 24n4g rollout retry. This is no-submit unless SUBMIT_ROLLOUT=true
# and ALLOW_HEAVY_GPU=true are both explicitly set.
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
PROBE_JOB_ID="$PROBE_JOB_ID" \\
SUBMIT_ROLLOUT=false \\
bash experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh

# Optional local-to-remote resume path. Keep PRINT_ONLY=true until the SSH and
# rsync commands are inspected.
PRINT_ONLY=true \\
SYNC_EXPERIMENTS=true \\
SYNC_PROBE_JOB_FILE=true \\
REMOTE_HOST="${REMOTE_HOST:-oci-hsg-cs-001-vscode-02}" \\
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap}" \\
REMOTE_ARTIFACT_ROOT="$REMOTE_ARTIFACT_ROOT" \\
REMOTE_ENTRYPOINT=experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh \\
PROBE_JOB_ID="$PROBE_JOB_ID" \\
SUBMIT_ROLLOUT=false \\
bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh
""",
    ),
    (
        "1j_operator_sheet",
        """python3 experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --markdown-out "$OPERATOR_SHEET_MARKDOWN" \\
  --json-out "$OPERATOR_SHEET_JSON"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_sheet.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --json-out "$OPERATOR_SHEET_VALIDATION_JSON" \\
  --markdown-out "$OPERATOR_SHEET_VALIDATION_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_execution.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --json-out "$OPERATOR_EXECUTION_JSON" \\
  --markdown-out "$OPERATOR_EXECUTION_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_followups.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --json-out "$OPERATOR_FOLLOWUP_VALIDATION_JSON" \\
  --markdown-out "$OPERATOR_FOLLOWUP_VALIDATION_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/create_eagle3_operator_submit_packet.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --operator-sheet-validation-json "$OPERATOR_SHEET_VALIDATION_JSON" \\
  --operator-followup-validation-json "$OPERATOR_FOLLOWUP_VALIDATION_JSON" \\
  --operator-execution-json "$OPERATOR_EXECUTION_JSON" \\
  --goal-evidence-json "$GOAL_EVIDENCE_JSON" \\
  --json-out "$OPERATOR_SUBMIT_PACKET_JSON" \\
  --markdown-out "$OPERATOR_SUBMIT_PACKET_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_submit_packet.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --operator-submit-packet-json "$OPERATOR_SUBMIT_PACKET_JSON" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --operator-sheet-validation-json "$OPERATOR_SHEET_VALIDATION_JSON" \\
  --operator-followup-validation-json "$OPERATOR_FOLLOWUP_VALIDATION_JSON" \\
  --operator-execution-json "$OPERATOR_EXECUTION_JSON" \\
  --json-out "$OPERATOR_SUBMIT_PACKET_VALIDATION_JSON" \\
  --markdown-out "$OPERATOR_SUBMIT_PACKET_VALIDATION_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --operator-submit-packet-validation-json "$OPERATOR_SUBMIT_PACKET_VALIDATION_JSON" \\
  --rollout-submit-preflight-json "$ROLLOUT_SUBMIT_PREFLIGHT_JSON" \\
  --json-out "$OPERATOR_READY_SUBMIT_PREFLIGHT_JSON" \\
  --markdown-out "$OPERATOR_READY_SUBMIT_PREFLIGHT_MARKDOWN"

python3 experiments/eagle3_qwen3_235b/summarize_eagle3_operator_queue.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --operator-execution-json "$OPERATOR_EXECUTION_JSON" \\
  --operator-followup-validation-json "$OPERATOR_FOLLOWUP_VALIDATION_JSON" \\
  --operator-ready-submit-preflight-json "$OPERATOR_READY_SUBMIT_PREFLIGHT_JSON" \\
  --json-out "$OPERATOR_QUEUE_JSON" \\
  --markdown-out "$OPERATOR_QUEUE_MARKDOWN"
""",
    ),
    (
        "1j_operator_state_refresh",
        """python3 experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --json-out "$OPERATOR_STATE_REFRESH_JSON" \\
  --markdown-out "$OPERATOR_STATE_REFRESH_MARKDOWN"
""",
    ),
    (
        "1j_validate_operator_state_refresh",
        """python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_state_refresh.py \\
  --json-out "$OPERATOR_STATE_REFRESH_VALIDATION_JSON" \\
  --markdown-out "$OPERATOR_STATE_REFRESH_VALIDATION_MARKDOWN"
""",
    ),
    (
        "1j_next_action_operator",
        """python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --list

# Print the first ready action without executing it:
python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --plan-json "$NEXT_ACTION_PLAN_JSON"

# Explicit execution examples, only after review:
# python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py --plan-json "$NEXT_ACTION_PLAN_JSON" --action-id submit_container_preflight --execute --allow-slurm
# python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py --plan-json "$NEXT_ACTION_PLAN_JSON" --action-id submit_rollout_capture --execute --allow-slurm --allow-heavy-gpu
# python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py --plan-json "$NEXT_ACTION_PLAN_JSON" --action-id submit_trained_draft_spec_tokens_sweep --execute --allow-slurm --allow-heavy-gpu
""",
    ),
    (
        "1j_slurm_followup_guard",
        """# Inspect terminal state before running Slurm follow-up analyzers:
for action_id in submit_vllm_source_build submit_container_preflight; do
  python3 experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py \\
    --artifact-root "$ARTIFACT_ROOT" \\
    --plan-json "$NEXT_ACTION_PLAN_JSON" \\
    --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
    --action-id "$action_id" \\
    --json-out "$ARTIFACT_ROOT/reports/operator_followups/${action_id}.json" \\
    --markdown-out "$ARTIFACT_ROOT/reports/operator_followups/${action_id}.md"
done

# Only after a guard reports READY_FOR_FOLLOWUP:
# python3 experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py \\
#   --artifact-root "$ARTIFACT_ROOT" \\
#   --plan-json "$NEXT_ACTION_PLAN_JSON" \\
#   --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
#   --action-id submit_vllm_source_build \\
#   --execute-after
#
# After the source build passes and submit_source_vllm_abi_probe appears:
# python3 experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py \\
#   --artifact-root "$ARTIFACT_ROOT" \\
#   --plan-json "$NEXT_ACTION_PLAN_JSON" \\
#   --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
#   --action-id submit_source_vllm_abi_probe
""",
    ),
    (
        "1j_validate_next_action_plan",
        """python3 experiments/eagle3_qwen3_235b/validate_eagle3_next_action_plan.py \\
  --plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_next_actions_validation.md" \\
  --json-out "$NEXT_ACTION_PLAN_VALIDATION_JSON"
""",
    ),
    (
        "1j_validate_next_action_transitions",
        """python3 experiments/eagle3_qwen3_235b/validate_eagle3_next_action_transitions.py \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_next_action_transitions.md" \\
  --json-out "$ARTIFACT_ROOT/reports/eagle3_next_action_transitions.json"

python3 experiments/eagle3_qwen3_235b/validate_eagle3_operator_queue_transitions.py \\
  --markdown-out "$OPERATOR_QUEUE_TRANSITIONS_MARKDOWN" \\
  --json-out "$OPERATOR_QUEUE_TRANSITIONS_JSON"
""",
    ),
    (
        "1j_validate_completion_contract",
        """python3 experiments/eagle3_qwen3_235b/validate_eagle3_completion_contract.py \\
  --markdown-out "$COMPLETION_CONTRACT_MARKDOWN" \\
  --json-out "$COMPLETION_CONTRACT_JSON"
""",
    ),
    (
        "1j_probe_slurm_capacity",
        """python3 experiments/eagle3_qwen3_235b/probe_eagle3_slurm_capacity.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --sbatch-partition "$SBATCH_PARTITION" \\
  --json-out "$SLURM_CAPACITY_JSON" \\
  --markdown-out "$SLURM_CAPACITY_MARKDOWN" \\
  --env-out "$SLURM_CAPACITY_ENV"

if [[ -f "$SLURM_CAPACITY_ENV" ]]; then
  source "$SLURM_CAPACITY_ENV"
fi
""",
    ),
    (
        "1j_validate_resource_profile_application",
        """python3 experiments/eagle3_qwen3_235b/validate_eagle3_resource_profile_application.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --resource-profile-env "$SLURM_CAPACITY_ENV" \\
  --json-out "$RESOURCE_PROFILE_APPLICATION_JSON" \\
  --markdown-out "$RESOURCE_PROFILE_APPLICATION_MARKDOWN"
""",
    ),
    (
        "1k_pre_submit_eagle3_pipeline",
        """if [[ -f "$SLURM_CAPACITY_ENV" ]]; then
  source "$SLURM_CAPACITY_ENV"
fi

python3 experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --input-data "${ROLLOUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}" \\
  --hidden-states-dir "$HIDDEN_STATES_DIR" \\
  --output-dir "$OUTPUT_DIR" \\
  --trained-ckpt "${TRAINED_CKPT:-$OUTPUT_DIR}" \\
  --export-dir "$EXPORT_DIR" \\
  --vllm-draft-dir "$VLLM_DRAFT_DIR" \\
  --verifier-config-dir "$VERIFIER_CONFIG_DIR" \\
  --chat-template "$CHAT_TEMPLATE" \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --reference-arch "${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}" \\
  --arch-env-file "${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}" \\
  --container-preflight-json "$CONTAINER_PREFLIGHT_JSON" \\
  --corpus-strategy-json "$CORPUS_STRATEGY_JSON" \\
  --rollout-state-json "$ROLLOUT_STATE_ADVANCE_JSON" \\
  --sbatch-account "$SBATCH_ACCOUNT" \\
  --sbatch-partition "$SBATCH_PARTITION" \\
  --container "$CONTAINER" \\
  --mounts "$MOUNTS" \\
  --slurm-capacity-json "$SLURM_CAPACITY_JSON" \\
  --slurm-capacity-markdown "$SLURM_CAPACITY_MARKDOWN" \\
  --slurm-capacity-env "$SLURM_CAPACITY_ENV" \\
  --run-pilot true \\
  --target-context "${EAGLE3_TARGET_CONTEXT:-swe_rl}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_pipeline_submit_preflight.md" \\
  --json-out "$PIPELINE_SUBMIT_PREFLIGHT_JSON"
""",
    ),
    (
        "1l_specforge_reference",
        """python3 experiments/eagle3_qwen3_235b/analyze_specforge_reference.py \\
  --markdown-out "$ARTIFACT_ROOT/reports/specforge_reference.md" \\
  --json-out "$SPECFORGE_REFERENCE_JSON"
""",
    ),
    (
        "1l_hayate_specforge_reference",
        """python3 experiments/eagle3_qwen3_235b/analyze_hayate_specforge_reference.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --json-out "$HAYATE_SPECFORGE_REFERENCE_JSON" \\
  --markdown-out "$HAYATE_SPECFORGE_REFERENCE_MARKDOWN"
""",
    ),
    (
        "1m_modelopt_upstream_drift",
        """python3 experiments/eagle3_qwen3_235b/check_modelopt_upstream_drift.py \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --hayate-modelopt-dir "${HAYATE_MODEL_OPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer}" \\
  --json-out "$ARTIFACT_ROOT/reports/modelopt_upstream_drift.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/modelopt_upstream_drift.md"
""",
    ),
    (
        "1n_nemo_rl_eagle3_drift",
        """python3 experiments/eagle3_qwen3_235b/check_nemo_rl_eagle3_drift.py \\
  --nemo-rl-dir "${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --json-out "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_drift.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_drift.md"
""",
    ),
    (
        "1o_hayate_modelopt_workflow",
        """python3 experiments/eagle3_qwen3_235b/analyze_hayate_modelopt_workflow.py \\
  --hayate-modelopt-dir "${HAYATE_MODEL_OPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer}" \\
  --json-out "$ARTIFACT_ROOT/reports/hayate_modelopt_workflow.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/hayate_modelopt_workflow.md"
""",
    ),
    (
        "1o_draft_config_inventory",
        """python3 experiments/eagle3_qwen3_235b/inventory_eagle3_draft_configs.py \\
  ${DRAFT_INVENTORY_ROOTS:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec/models /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge/outputs} \\
  "$VLLM_DRAFT_DIR" \\
  "$EXPORT_DIR" \\
  --reference-arch "$REFERENCE_ARCH" \\
  --json-out "$DRAFT_INVENTORY_JSON" \\
  --markdown-out "$DRAFT_INVENTORY_MARKDOWN"
""",
    ),
    (
        "1p_modelopt_loss_mask_patch_check",
        """python3 experiments/eagle3_qwen3_235b/validate_modelopt_loss_mask_patch.py \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --json-out "$ARTIFACT_ROOT/reports/modelopt_loss_mask_patch.json" \\
  --markdown-out "$ARTIFACT_ROOT/reports/modelopt_loss_mask_patch.md"
""",
    ),
    (
        "1q_export_modelopt_patch_bundle",
        """patch_args=(
  python3 experiments/eagle3_qwen3_235b/export_modelopt_eagle3_patch_bundle.py
  --modelopt-dir "$MODELOPT_DIR"
  --out-dir "$ARTIFACT_ROOT/patches/modelopt_eagle3_qwen3"
)
if [[ -n "${COMPAT_MODELOPT_DIR:-}" ]]; then
  patch_args+=(--compat-modelopt-dir "$COMPAT_MODELOPT_DIR")
fi
"${patch_args[@]}"
""",
    ),
    (
        "2_bootstrap_dry_run",
        """if [[ -f "$SLURM_CAPACITY_ENV" ]]; then
  source "$SLURM_CAPACITY_ENV"
fi

SUBMIT=false RUN_PILOT=true \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \\
SBATCH_PARTITION="$SBATCH_PARTITION" \\
MODELOPT_DIR="$MODELOPT_DIR" \\
VERIFIER_CONFIG_DIR="$VERIFIER_CONFIG_DIR" \\
bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh
""",
    ),
    (
        "3_submit_pilot",
        """if [[ -f "$SLURM_CAPACITY_ENV" ]]; then
  source "$SLURM_CAPACITY_ENV"
fi

SUBMIT=true RUN_PILOT=true PREP_DRY_RUN=false \\
RUN_TRAINED_DRAFT_SMOKE=true RUN_TRAINED_DRAFT_SWEEP=true \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \\
SBATCH_PARTITION="$SBATCH_PARTITION" \\
MODELOPT_DIR="$MODELOPT_DIR" \\
VERIFIER_CONFIG_DIR="$VERIFIER_CONFIG_DIR" \\
bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh
""",
    ),
    (
        "4_analyze_pipeline",
        """python3 experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py \\
  --job-file latest_eagle3_pipeline_jobs.txt \\
  --logs-dir logs \\
  --base-model "$BASE_MODEL" \\
  --modelopt-dir "$MODELOPT_DIR" \\
  --verifier-config-dir "$VERIFIER_CONFIG_DIR" \\
  --reference-arch "$REFERENCE_ARCH" \\
  --arch-env-file "$ARCH_ENV_FILE" \\
  --chat-template "$CHAT_TEMPLATE" \\
  --container "$CONTAINER" \\
  --mounts "$MOUNTS" \\
  --input-data "$INPUT_DATA" \\
  --hidden-states-dir "$HIDDEN_STATES_DIR" \\
  --hidden-validation-json "$HIDDEN_STATES_DIR/validation_summary.json" \\
  --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON" \\
  --output-dir "$OUTPUT_DIR" \\
  --export-dir "$EXPORT_DIR" \\
  --vllm-draft-dir "$VLLM_DRAFT_DIR" \\
  --export-artifacts-json "$ARTIFACT_ROOT/reports/eagle3_export_artifacts.json" \\
  --sbatch-account "$SBATCH_ACCOUNT" \\
  --sbatch-partition "$SBATCH_PARTITION" \\
  --run-pilot true \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_pipeline_analysis.md" \\
  --json-out "$ARTIFACT_ROOT/reports/eagle3_pipeline_analysis.json"
""",
    ),
    (
        "5_sweep_trained_draft",
        """python3 experiments/eagle3_qwen3_235b/validate_nemo_rl_specdec_integration.py \\
  --config grpo_qwen3_235b_swe.yaml \\
  --draft-model "$VLLM_DRAFT_DIR" \\
  --integration-mode generation-only \\
  --num-speculative-tokens 3 \\
  --draft-tensor-parallel-size 1 \\
  --specdec-rl-dir "${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/nemo_rl_specdec_integration.md" \\
  --json-out "$ARTIFACT_ROOT/reports/nemo_rl_specdec_integration.json" \\
  --env-out "$ARTIFACT_ROOT/reports/nemo_rl_specdec_overrides.env"

SUBMIT=true \\
ARTIFACT_ROOT="$ARTIFACT_ROOT" \\
REPO_ROOT="$REPO_ROOT" \\
SWE_REPO_ROOT="$SWE_REPO_ROOT" \\
CONFIG_FILE="$CONFIG_FILE" \\
ENV_FILE="$ENV_FILE" \\
CHAT_TEMPLATE="$CHAT_TEMPLATE" \\
VLLM_DRAFT_DIR="$VLLM_DRAFT_DIR" \\
SPEC_TOKENS_LIST="2 3 4" \\
MAX_NUM_STEPS=2 \\
EAGLE3_DRAFT_TP="${EAGLE3_DRAFT_TP:-1}" \\
bash experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh

python3 experiments/eagle3_qwen3_235b/analyze_spec_tokens_sweep.py \\
  --job-file latest_trained_draft_spec_tokens_sweep_jobs.txt \\
  --repo-root "${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/trained_draft_spec_tokens_sweep.md" \\
  --json-out "$ARTIFACT_ROOT/reports/trained_draft_spec_tokens_sweep.json" \\
  --fail-on-missing-spec-metrics
""",
    ),
    (
        "5b_validate_online_draft_training",
        """python3 experiments/eagle3_qwen3_235b/validate_nemo_rl_specdec_integration.py \\
  --config grpo_qwen3_235b_swe.yaml \\
  --draft-model "$VLLM_DRAFT_DIR" \\
  --integration-mode online-draft-training \\
  --draft-loss-weight "${EAGLE3_DRAFT_LOSS_WEIGHT:-1.0}" \\
  --num-speculative-tokens 3 \\
  --draft-tensor-parallel-size 1 \\
  --specdec-rl-dir "${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}" \\
  --markdown-out "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_online_draft_integration.md" \\
  --json-out "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_online_draft_integration.json" \\
  --env-out "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_online_draft_overrides.env"
""",
    ),
    (
        "6_completion_audit",
        """python3 experiments/eagle3_qwen3_235b/audit_eagle3_completion.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --input-discovery-json "$ARTIFACT_ROOT/eagle3_input_discovery.json" \\
  --cluster-probe-json "$ARTIFACT_ROOT/reports/cluster_environment_probe.json" \\
  --readiness-json "$ARTIFACT_ROOT/reports/eagle3_readiness.json" \\
  --provenance-json "$ARTIFACT_ROOT/reports/eagle3_provenance.json" \\
  --remote-host-probe-json "$ARTIFACT_ROOT/reports/eagle3_remote_host_probe.json" \\
  --hayate-workflow-json "$HAYATE_WORKFLOW_JSON" \\
  --hayate-specforge-reference-json "$HAYATE_SPECFORGE_REFERENCE_JSON" \\
  --upstream-drift-json "$ARTIFACT_ROOT/reports/modelopt_upstream_drift.json" \\
  --modelopt-recipe-overrides-json "$MODELOPT_RECIPE_OVERRIDES_JSON" \\
  --modelopt-patch-manifest "$ARTIFACT_ROOT/patches/modelopt_eagle3_qwen3/manifest.json" \\
  --next-action-plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --next-action-plan-validation-json "$ARTIFACT_ROOT/reports/eagle3_next_actions_validation.json" \\
  --operator-queue-transitions-json "$OPERATOR_QUEUE_TRANSITIONS_JSON" \\
  --operator-followup-validation-json "$OPERATOR_FOLLOWUP_VALIDATION_JSON" \\
  --megatron-probe-followup-validation-json "$MEGATRON_PROBE_FOLLOWUP_VALIDATION_JSON" \\
  --preflight-robustness-validation-json "$PREFLIGHT_ROBUSTNESS_VALIDATION_JSON" \\
  --operator-submit-packet-validation-json "$OPERATOR_SUBMIT_PACKET_VALIDATION_JSON" \\
  --operator-ready-submit-preflight-json "$OPERATOR_READY_SUBMIT_PREFLIGHT_JSON" \\
  --operator-queue-json "$OPERATOR_QUEUE_JSON" \\
  --completion-contract-json "$COMPLETION_CONTRACT_JSON" \\
  --slurm-capacity-json "$SLURM_CAPACITY_JSON" \\
  --resource-profile-application-json "$RESOURCE_PROFILE_APPLICATION_JSON" \\
  --container-preflight-json "$ARTIFACT_ROOT/reports/container_preflight_analysis.json" \\
  --rollout-state-json "$ARTIFACT_ROOT/reports/rollout_capture_state_advance.json" \\
  --corpus-strategy-json "$ARTIFACT_ROOT/reports/corpus_strategy.json" \\
  --pipeline-submit-preflight-json "$ARTIFACT_ROOT/reports/eagle3_pipeline_submit_preflight.json" \\
  --pipeline-analysis-json "$ARTIFACT_ROOT/reports/eagle3_pipeline_analysis.json" \\
  --hidden-validation-json "$HIDDEN_STATES_DIR/validation_summary.json" \\
  --output-dir "$OUTPUT_DIR" \\
  --export-dir "$EXPORT_DIR" \\
  --vllm-draft-dir "$VLLM_DRAFT_DIR" \\
  --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON" \\
  --export-artifacts-json "$ARTIFACT_ROOT/reports/eagle3_export_artifacts.json" \\
  --export-config-compare-json "$EXPORT_DIR/config_compare.json" \\
  --vllm-config-compare-json "$VLLM_DRAFT_DIR/config_compare.json" \\
  --sweep-json "$ARTIFACT_ROOT/reports/trained_draft_spec_tokens_sweep.json" \\
  --draft-inventory-json "$ARTIFACT_ROOT/reports/eagle3_draft_config_inventory.json" \\
  --hayate-inventory "$ARTIFACT_ROOT/reports/hayate_inventory.txt" \\
  --markdown-out "$ARTIFACT_ROOT/reports/eagle3_completion_audit.md" \\
  --json-out "$ARTIFACT_ROOT/reports/eagle3_completion_audit.json"
""",
    ),
    (
        "6b_goal_evidence_matrix",
        """python3 experiments/eagle3_qwen3_235b/audit_eagle3_goal_evidence.py \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --reference-arch "$REFERENCE_ARCH" \\
  --remote-host-probe-json "$ARTIFACT_ROOT/reports/eagle3_remote_host_probe.json" \\
  --hayate-workflow-json "$HAYATE_WORKFLOW_JSON" \\
  --hayate-specforge-reference-json "$HAYATE_SPECFORGE_REFERENCE_JSON" \\
  --draft-inventory-json "$DRAFT_INVENTORY_JSON" \\
  --modelopt-loss-mask-json "$MODELOPT_LOSS_MASK_JSON" \\
  --modelopt-recipe-overrides-json "$MODELOPT_RECIPE_OVERRIDES_JSON" \\
  --upstream-drift-json "$ARTIFACT_ROOT/reports/modelopt_upstream_drift.json" \\
  --nemo-rl-drift-json "$ARTIFACT_ROOT/reports/nemo_rl_eagle3_drift.json" \\
  --corpus-strategy-json "$CORPUS_STRATEGY_JSON" \\
  --rollout-state-json "$ROLLOUT_STATE_ADVANCE_JSON" \\
  --container-preflight-json "$CONTAINER_PREFLIGHT_JSON" \\
  --pipeline-submit-preflight-json "$PIPELINE_SUBMIT_PREFLIGHT_JSON" \\
  --pipeline-analysis-json "$PIPELINE_ANALYSIS_JSON" \\
  --hidden-validation-json "$HIDDEN_STATES_DIR/validation_summary.json" \\
  --training-checkpoint-json "$TRAINING_CKPT_VALIDATION_JSON" \\
  --export-artifacts-json "$EXPORT_ARTIFACTS_JSON" \\
  --sweep-json "$SWEEP_JSON" \\
  --next-action-plan-json "$NEXT_ACTION_PLAN_JSON" \\
  --next-action-validation-json "$NEXT_ACTION_PLAN_VALIDATION_JSON" \\
  --operator-sheet-json "$OPERATOR_SHEET_JSON" \\
  --operator-execution-json "$OPERATOR_EXECUTION_JSON" \\
  --operator-followup-validation-json "$OPERATOR_FOLLOWUP_VALIDATION_JSON" \\
  --megatron-probe-followup-validation-json "$MEGATRON_PROBE_FOLLOWUP_VALIDATION_JSON" \\
  --preflight-robustness-validation-json "$PREFLIGHT_ROBUSTNESS_VALIDATION_JSON" \\
  --completion-audit-json "$ARTIFACT_ROOT/reports/eagle3_completion_audit.json" \\
  --markdown-out "$GOAL_EVIDENCE_MARKDOWN" \\
  --json-out "$GOAL_EVIDENCE_JSON"
""",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "outputs/qwen3_235b_eagle3")
    parser.add_argument("--remote-artifact-root", default=DEFAULT_REMOTE_ARTIFACT_ROOT)
    parser.add_argument(
        "--sbatch-account",
        default=None,
        help=(
            "Slurm account written to commands.sh. If omitted, infer it from "
            f"existing reports and fall back to {DEFAULT_SBATCH_ACCOUNT}."
        ),
    )
    parser.add_argument("--sbatch-partition", default="batch")
    parser.add_argument("--input-discovery-json", type=Path)
    parser.add_argument("--static-inputs-json", type=Path)
    parser.add_argument("--static-inputs-validation-json", type=Path)
    parser.add_argument("--remote-host-probe-json", type=Path)
    parser.add_argument("--remote-access-diagnostics-json", type=Path)
    parser.add_argument("--cluster-probe-json", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--upstream-drift-json", type=Path)
    parser.add_argument("--modelopt-loss-mask-json", type=Path)
    parser.add_argument("--modelopt-recipe-overrides-json", type=Path)
    parser.add_argument("--modelopt-patch-manifest", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--container-preflight-json", type=Path)
    parser.add_argument("--nemo-rl-specdec-json", type=Path)
    parser.add_argument("--nemo-rl-drift-json", type=Path)
    parser.add_argument("--specdec-remote-patch-bundle-json", type=Path)
    parser.add_argument("--rollout-capture-json", type=Path)
    parser.add_argument("--rollout-capture-analysis-json", type=Path)
    parser.add_argument("--rollout-capture-job-json", type=Path)
    parser.add_argument("--rollout-submit-preflight-json", type=Path)
    parser.add_argument("--rollout-state-advance-json", type=Path)
    parser.add_argument("--corpus-strategy-json", type=Path)
    parser.add_argument("--training-scale-json", type=Path)
    parser.add_argument("--training-path-manifest-json", type=Path)
    parser.add_argument("--training-path-manifest-markdown", type=Path)
    parser.add_argument("--training-path-manifest-validation-json", type=Path)
    parser.add_argument("--training-path-manifest-validation-markdown", type=Path)
    parser.add_argument("--next-action-plan-json", type=Path)
    parser.add_argument("--next-action-plan-validation-json", type=Path)
    parser.add_argument("--next-action-transitions-json", type=Path)
    parser.add_argument("--operator-queue-transitions-json", type=Path)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--operator-sheet-validation-json", type=Path)
    parser.add_argument("--operator-execution-json", type=Path)
    parser.add_argument("--operator-followup-validation-json", type=Path)
    parser.add_argument("--megatron-probe-followup-validation-json", type=Path)
    parser.add_argument("--preflight-robustness-validation-json", type=Path)
    parser.add_argument("--operator-submit-packet-json", type=Path)
    parser.add_argument("--operator-submit-packet-validation-json", type=Path)
    parser.add_argument("--operator-ready-submit-preflight-json", type=Path)
    parser.add_argument("--operator-safe-actions-preflight-json", type=Path)
    parser.add_argument("--operator-queue-json", type=Path)
    parser.add_argument("--operator-state-refresh-json", type=Path)
    parser.add_argument("--operator-state-refresh-validation-json", type=Path)
    parser.add_argument("--completion-contract-json", type=Path)
    parser.add_argument("--slurm-capacity-json", type=Path)
    parser.add_argument("--slurm-capacity-env", type=Path)
    parser.add_argument("--resource-profile-application-json", type=Path)
    parser.add_argument("--pipeline-submit-preflight-json", type=Path)
    parser.add_argument("--specforge-reference-json", type=Path)
    parser.add_argument("--hayate-specforge-reference-json", type=Path)
    parser.add_argument("--hayate-workflow-json", type=Path)
    parser.add_argument("--pipeline-analysis-json", type=Path)
    parser.add_argument("--training-checkpoint-json", type=Path)
    parser.add_argument("--export-artifacts-json", type=Path)
    parser.add_argument("--sweep-json", type=Path)
    parser.add_argument("--completion-json", type=Path)
    parser.add_argument("--goal-evidence-json", type=Path)
    parser.add_argument("--hayate-inventory", type=Path)
    parser.add_argument("--draft-inventory-json", type=Path)
    parser.add_argument("--include-html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clean-stale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove stale files from previous handoff generations that are no longer in the manifest.",
    )
    return parser.parse_args()


def bundle_dest_name(label: str, src: Path) -> str:
    if src.name in {"manifest.json", "commands.sh", "RUNBOOK.md"}:
        return f"{label}_{src.name}"
    if label.startswith("materialized_"):
        return f"{label}_{src.name}"
    return src.name


def clean_stale_outputs(out_dir: Path, expected_names: set[str], managed_labels: set[str]) -> list[str]:
    removed: list[str] = []
    manifest_path = out_dir / "manifest.json"
    candidates: set[Path] = set()
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
        for item in (previous.get("inputs") or {}).values():
            if not isinstance(item, dict) or not item.get("bundle_path"):
                continue
            path = Path(str(item["bundle_path"]))
            if path.parent == out_dir:
                candidates.add(path)

    for path in out_dir.iterdir() if out_dir.exists() else []:
        if not path.is_file():
            continue
        if path.name in expected_names:
            continue
        if any(path.name.startswith(f"{label}_") for label in managed_labels):
            candidates.add(path)

    for path in sorted(candidates):
        if path.name in expected_names or not path.exists() or not path.is_file():
            continue
        path.unlink()
        removed.append(path.name)
    return removed


def copy_if_exists(src: Path | None, dest_dir: Path, label: str, manifest: dict[str, Any]) -> str | None:
    if src is None:
        manifest["inputs"][label] = {"status": "not_provided"}
        return None
    if not src.exists():
        manifest["inputs"][label] = {"status": "missing", "path": str(src)}
        return None
    dest = dest_dir / bundle_dest_name(label, src)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    manifest["inputs"][label] = {"status": "copied", "source": str(src), "bundle_path": str(dest)}
    return dest.name


def copy_tree_if_exists(src: Path | None, dest_dir: Path, label: str, manifest: dict[str, Any]) -> str | None:
    if src is None:
        manifest["inputs"][label] = {"status": "not_provided"}
        return None
    if not src.exists() or not src.is_dir():
        manifest["inputs"][label] = {"status": "missing", "path": str(src)}
        return None
    dest = dest_dir / label
    if dest.exists():
        shutil.rmtree(dest)
    copied_files = 0
    copied_bytes = 0
    skipped_files: list[str] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
            skipped_files.append(str(rel))
            continue
        out_path = dest / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
        copied_files += 1
        copied_bytes += out_path.stat().st_size
    manifest["inputs"][label] = {
        "status": "copied",
        "source": str(src),
        "bundle_path": str(dest),
        "file_count": copied_files,
        "bytes": copied_bytes,
        "skipped_generated_files": len(skipped_files),
    }
    manifest["summaries"][label] = {
        "format": "directory",
        "file_count": copied_files,
        "bytes": copied_bytes,
        "skipped_generated_files": len(skipped_files),
    }
    return dest.name


def json_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return {"error": str(exc)}
        first_heading = next((line.strip() for line in text.splitlines() if line.lstrip().startswith("#")), None)
        return {
            "format": "text",
            "bytes": path.stat().st_size,
            "lines": len(text.splitlines()),
            "first_heading": first_heading,
        }
    summary: dict[str, Any] = {}
    for key in ("overall_status", "status", "job_id", "counts", "configs_scanned", "files_scanned", "generated_at"):
        if key in payload:
            summary[key] = payload[key]
    if "followup_state_counts" in payload:
        summary["followup_state_counts"] = payload["followup_state_counts"]
    if "container" in payload:
        summary["container"] = payload["container"]
    if "checks" in payload and isinstance(payload["checks"], dict):
        summary["checks"] = {
            key: value.get("status")
            for key, value in payload["checks"].items()
            if isinstance(value, dict) and "status" in value
        }
    if "checks" in payload and isinstance(payload["checks"], list):
        status_counts: dict[str, int] = {}
        failed_checks: list[dict[str, Any]] = []
        for item in payload["checks"]:
            if isinstance(item, dict):
                status = str(item.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status in {"fail", "warn"} and len(failed_checks) < 8:
                    failed_checks.append(
                        {
                            "area": item.get("area"),
                            "name": item.get("name"),
                            "status": status,
                            "detail": item.get("detail"),
                        }
                    )
        summary["check_status_counts"] = status_counts
        if failed_checks:
            summary["failed_checks"] = failed_checks
    if "repos" in payload:
        summary["repos"] = [
            {
                "label": item.get("label"),
                "exists": item.get("exists"),
                "branch": item.get("branch"),
                "head": str(item.get("head") or "")[:12],
            }
            for item in payload["repos"][:4]
        ]
    if "recommendation" in payload:
        summary["recommendation"] = payload["recommendation"]
    if "recommendations" in payload:
        summary["recommendations"] = payload.get("recommendations", [])[:4]
    if "target_status" in payload:
        summary["target_status"] = payload.get("target_status")
    if "file_count" in payload:
        summary["file_count"] = payload.get("file_count")
    if "ignored_file_count" in payload:
        summary["ignored_file_count"] = payload.get("ignored_file_count")
    if "visible_capacity" in payload:
        capacity = payload.get("visible_capacity") or {}
        summary["visible_capacity"] = {
            "max_gpu_per_node": capacity.get("max_gpu_per_node"),
            "unique_gres": capacity.get("unique_gres"),
        }
    if "scenarios" in payload:
        summary["scenarios"] = [
            {"name": item.get("name"), "status": item.get("status")}
            for item in (payload.get("scenarios") or [])[:6]
            if isinstance(item, dict)
        ]
    if "training_defaults" in payload:
        defaults = payload["training_defaults"] or {}
        summary["training_defaults"] = {
            "effective_global_batch": defaults.get("effective_global_batch"),
            "epochs": defaults.get("epochs"),
            "max_seq_len": defaults.get("max_seq_len"),
        }
    if "corpus" in payload:
        corpus = payload["corpus"] or {}
        estimated_tokens = corpus.get("estimated_tokens") or {}
        summary["corpus"] = {
            "status": corpus.get("status"),
            "total_rows": corpus.get("total_rows"),
            "avg_estimated_tokens": estimated_tokens.get("avg"),
            "p95_estimated_tokens": estimated_tokens.get("p95"),
        }
    if "train_data" in payload:
        train_data = payload["train_data"] or {}
        summary["train_data"] = {
            "file_count": train_data.get("file_count"),
            "rows_sampled": train_data.get("rows_sampled"),
            "extractable_conversations": train_data.get("extractable_conversations"),
            "invalid_json": train_data.get("invalid_json"),
        }
    if "output_data" in payload:
        output_data = payload["output_data"] or {}
        if isinstance(output_data, dict):
            summary["output_data"] = {
                "status": output_data.get("status"),
                "path": output_data.get("path"),
            }
        else:
            summary["output_data"] = {"path": str(output_data)}
    if "artifacts" in payload:
        artifacts = payload["artifacts"] or {}
        train_data = (artifacts.get("train_data") or {})
        output_data = (artifacts.get("output_data") or {})
        summary["artifacts"] = {
            "status": artifacts.get("overall_status"),
            "train_file_count": train_data.get("file_count"),
            "extractable_conversations": train_data.get("extractable_conversations"),
            "output_status": output_data.get("status"),
        }
    if "decision" in payload:
        decision = payload["decision"] or {}
        if isinstance(decision, dict):
            summary["decision"] = {
                "overall_status": decision.get("overall_status"),
                "primary_source": decision.get("primary_source"),
                "next_action": decision.get("next_action") or decision.get("next_step"),
            }
    if "next_actions" in payload:
        actions = payload.get("next_actions") or []
        if all(isinstance(item, str) for item in actions[:3]):
            summary["next_actions"] = actions[:3]
        else:
            summary["next_actions"] = [
                {
                    "id": item.get("id"),
                    "status": item.get("status"),
                    "submits_slurm": item.get("submits_slurm"),
                    "heavy_gpu": item.get("heavy_gpu"),
                }
                for item in actions[:3]
                if isinstance(item, dict)
            ]
    if "ready_actions" in payload:
        actions = payload.get("ready_actions") or []
        if all(isinstance(item, str) for item in actions[:4]):
            summary["ready_actions"] = actions[:4]
        else:
            summary["ready_actions"] = [
                {
                    "id": item.get("id"),
                    "status": item.get("status"),
                    "submits_slurm": item.get("submits_slurm"),
                    "heavy_gpu": item.get("heavy_gpu"),
                }
                for item in actions[:4]
                if isinstance(item, dict)
            ]
    if "open_gates" in payload:
        summary["open_gates"] = payload.get("open_gates", [])[:8]
    if "gate_closure_contracts" in payload and isinstance(payload.get("gate_closure_contracts"), list):
        summary["gate_closure_contracts"] = [
            {
                "id": item.get("id"),
                "closed": item.get("closed"),
                "missing": item.get("closure_evidence_missing") or [],
                "candidate_next_action_ids": item.get("candidate_next_action_ids") or [],
            }
            for item in payload.get("gate_closure_contracts", [])[:8]
            if isinstance(item, dict)
        ]
    if "operator_gate_action_matrix" in payload and isinstance(payload.get("operator_gate_action_matrix"), list):
        summary["operator_gate_action_matrix"] = [
            {
                "gate_id": item.get("gate_id"),
                "status": item.get("status"),
                "current_ready_action_ids": item.get("current_ready_action_ids") or [],
                "future_candidate_action_ids": item.get("future_candidate_action_ids") or [],
                "missing_evidence": item.get("missing_evidence") or [],
            }
            for item in payload.get("operator_gate_action_matrix", [])[:8]
            if isinstance(item, dict)
        ]
    if "artifact_flow" in payload and isinstance(payload.get("artifact_flow"), list):
        summary["artifact_flow_complete"] = payload.get("artifact_flow_complete")
        summary["artifact_flow"] = [
            {
                "id": item.get("id"),
                "proof_status": item.get("proof_status"),
                "producer_gate": item.get("producer_gate"),
                "consumer_gate": item.get("consumer_gate"),
                "required_reports": item.get("required_reports") or [],
                "required_invariants": item.get("required_invariants") or [],
                "closure_action_ids": item.get("closure_action_ids") or [],
                "current_closure_action_ids": item.get("current_closure_action_ids") or [],
                "future_closure_action_ids": item.get("future_closure_action_ids") or [],
                "report_statuses": item.get("report_statuses") or {},
                "path_visible": item.get("path_visible"),
                "path": item.get("path"),
            }
            for item in payload.get("artifact_flow", [])[:8]
            if isinstance(item, dict)
        ]
    if "host_discovery" in payload and isinstance(payload.get("host_discovery"), dict):
        discovery = payload["host_discovery"]
        hosts = payload.get("hosts") if isinstance(payload.get("hosts"), list) else []
        host_diagnostics = (
            payload.get("host_diagnostics")
            if isinstance(payload.get("host_diagnostics"), list)
            else []
        )
        first_host = next((item for item in hosts if isinstance(item, dict)), {})
        if not first_host:
            first_host = next((item for item in host_diagnostics if isinstance(item, dict)), {})
        resolution = (
            first_host.get("local_resolution")
            if isinstance(first_host.get("local_resolution"), dict)
            else {}
        )
        summary["host_discovery"] = {
            "include_ssh_config_hosts": discovery.get("include_ssh_config_hosts"),
            "ssh_config_host_count": len(discovery.get("ssh_config_hosts") or []),
            "first_host": first_host.get("host"),
            "first_query": resolution.get("query") or first_host.get("resolution_query"),
            "first_resolved": resolution.get("resolved")
            if "resolved" in resolution
            else first_host.get("resolved"),
        }
    if "diagnosis" in payload:
        counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        interpretation = (
            payload.get("gate_interpretation")
            if isinstance(payload.get("gate_interpretation"), dict)
            else {}
        )
        summary["diagnosis"] = {
            "overall_status": payload.get("overall_status"),
            "diagnosis": payload.get("diagnosis"),
            "hosts": counts.get("hosts"),
            "resolved_hosts": counts.get("resolved_hosts"),
            "reachable_hosts": counts.get("reachable_hosts"),
            "ssh_config_hostname_warnings": counts.get("ssh_config_hostname_warnings"),
            "remote_path_absence_proven": interpretation.get("remote_path_absence_proven"),
        }
        findings = payload.get("configuration_findings") if isinstance(payload.get("configuration_findings"), list) else []
        if findings:
            summary["configuration_findings"] = [
                {
                    "host": item.get("host"),
                    "configured_hostname": item.get("configured_hostname"),
                    "finding": item.get("finding"),
                }
                for item in findings[:4]
                if isinstance(item, dict)
            ]
    if "reference_evidence" in payload and isinstance(payload.get("reference_evidence"), dict):
        evidence = payload["reference_evidence"]
        remote = evidence.get("remote_probe") if isinstance(evidence.get("remote_probe"), dict) else {}
        hayate_modelopt = evidence.get("hayate_modelopt") if isinstance(evidence.get("hayate_modelopt"), dict) else {}
        hayate_specforge = evidence.get("hayate_specforge") if isinstance(evidence.get("hayate_specforge"), dict) else {}
        summary["reference_evidence"] = {
            "remote_reference_proven": evidence.get("remote_reference_proven"),
            "remote_probe_status": remote.get("status"),
            "reachable_hosts": len(remote.get("reachable_hosts") or []),
            "remote_configuration_findings": [
                item.get("finding")
                for item in (remote.get("configuration_findings") or [])[:4]
                if isinstance(item, dict)
            ],
            "hayate_modelopt_source": hayate_modelopt.get("source"),
            "hayate_modelopt_remote_path_visible": hayate_modelopt.get("remote_path_visible"),
            "hayate_specforge_source": hayate_specforge.get("source"),
            "hayate_specforge_remote_path_visible": hayate_specforge.get("remote_path_visible"),
        }
    if "reference_decisions" in payload and isinstance(payload.get("reference_decisions"), dict):
        decisions = payload["reference_decisions"]
        modelopt = decisions.get("modelopt_source") if isinstance(decisions.get("modelopt_source"), dict) else {}
        specforge = (
            decisions.get("specforge_qwen3_235b")
            if isinstance(decisions.get("specforge_qwen3_235b"), dict)
            else {}
        )
        rejected = specforge.get("rejected_fields") if isinstance(specforge.get("rejected_fields"), list) else []
        summary["reference_decisions"] = {
            "modelopt_source_of_truth": modelopt.get("source_of_truth"),
            "upstream_drift_status": modelopt.get("upstream_drift_status"),
            "specforge_matched_fields": (specforge.get("matched_fields") or [])[:8],
            "specforge_rejected_fields": [
                item.get("field") for item in rejected[:8] if isinstance(item, dict)
            ],
        }
        remote = decisions.get("remote_probe") if isinstance(decisions.get("remote_probe"), dict) else {}
        if remote.get("configuration_findings"):
            summary["reference_decisions"]["remote_configuration_findings"] = [
                item.get("finding")
                for item in (remote.get("configuration_findings") or [])[:4]
                if isinstance(item, dict)
            ]
    if "latest_by_action" in payload:
        latest = payload.get("latest_by_action") or {}
        summary["operator_execution"] = {
            "overall_status": payload.get("overall_status"),
            "record_count": len(payload.get("records") or []),
            "latest_actions": sorted(latest.keys())[:8],
        }
    if "blockers" in payload:
        blockers = payload.get("blockers") or []
        summary["blockers"] = [
            {"id": item.get("id"), "severity": item.get("severity")}
            for item in blockers[:4]
            if isinstance(item, dict)
        ]
    if "open_requirements" in payload:
        summary["goal_evidence"] = {
            "overall_status": payload.get("overall_status"),
            "draft_model_trained": payload.get("draft_model_trained"),
            "current_ready_actions": payload.get("current_ready_actions"),
            "open_requirements": payload.get("open_requirements", [])[:8],
        }
    if "next_action" in payload:
        next_action = payload["next_action"]
        summary["next_action"] = {
            "summary": next_action.get("summary"),
            "first_open_stage": next_action.get("first_open_stage"),
            "first_open_status": next_action.get("first_open_status"),
            "has_submit_command": bool(next_action.get("submit_command")),
        }
    for key in ("patch_sha256", "patch_nonempty", "patch_paths", "snapshot_paths"):
        if key in payload:
            summary[key] = payload[key]
    if "verifier_candidates" in payload:
        summary["top_verifier"] = payload["verifier_candidates"][:1]
    if "conversation_candidates" in payload:
        summary["top_conversation"] = payload["conversation_candidates"][:1]
    return summary


def is_concrete_sbatch_account(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in DUMMY_SBATCH_ACCOUNTS


def iter_sbatch_accounts(payload: Any) -> list[str]:
    accounts: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"sbatch_account", "SBATCH_ACCOUNT"} and is_concrete_sbatch_account(value):
                accounts.append(str(value))
            else:
                accounts.extend(iter_sbatch_accounts(value))
    elif isinstance(payload, list):
        for item in payload:
            accounts.extend(iter_sbatch_accounts(item))
    return accounts


def infer_sbatch_account_from_reports(paths: list[Path | None]) -> str | None:
    for path in paths:
        if path is None or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for account in iter_sbatch_accounts(payload):
            return account
    return None


def render_gate_closure_markdown(manifest: dict[str, Any], *, standalone: bool) -> str:
    def csv(values: Any) -> str:
        return ", ".join(str(item) for item in values or []) or "-"

    def statuses(values: Any) -> str:
        if not isinstance(values, dict) or not values:
            return "-"
        return ", ".join(f"{key}={value}" for key, value in values.items()) or "-"

    summaries = manifest.get("summaries") if isinstance(manifest.get("summaries"), dict) else {}
    training = summaries.get("training_path_manifest") if isinstance(summaries.get("training_path_manifest"), dict) else {}
    contracts = training.get("gate_closure_contracts") if isinstance(training.get("gate_closure_contracts"), list) else []
    if not contracts:
        return ""

    heading = "# Eagle3 Gate Closure Checklist" if standalone else "## Gate Closure Checklist"
    lines = [
        heading,
        "",
        f"Training path manifest: `{training.get('overall_status') or 'unknown'}`",
        f"Open gates: `{', '.join(training.get('open_gates') or []) or '-'}`",
        "",
        "| order | gate | closed | missing evidence | candidate actions |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for idx, item in enumerate(contracts, 1):
        missing = ", ".join(item.get("missing") or []) or "-"
        actions = ", ".join(item.get("candidate_next_action_ids") or []) or "-"
        lines.append(
            f"| {idx} | {item.get('id') or '-'} | {str(item.get('closed')).lower()} | "
            f"{missing.replace('|', '/')} | {actions.replace('|', '/')} |"
        )

    matrix = training.get("operator_gate_action_matrix") if isinstance(training.get("operator_gate_action_matrix"), list) else []
    if matrix:
        lines += [
            "",
            "### Current Ready Action Mapping",
            "",
            "| gate | status | current ready actions | future candidate actions |",
            "| --- | --- | --- | --- |",
        ]
        for item in matrix:
            ready = ", ".join(item.get("current_ready_action_ids") or []) or "-"
            future = ", ".join(item.get("future_candidate_action_ids") or []) or "-"
            lines.append(
                f"| {item.get('gate_id') or '-'} | {item.get('status') or '-'} | "
                f"{ready.replace('|', '/')} | {future.replace('|', '/')} |"
            )
    flow = training.get("artifact_flow") if isinstance(training.get("artifact_flow"), list) else []
    if flow:
        lines += [
            "",
            "### Artifact Flow",
            "",
            f"Complete: `{str(training.get('artifact_flow_complete')).lower()}`",
            "",
            "| artifact | proof | current actions | future actions | reports | invariants | visible | path |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in flow:
            current_actions = csv(item.get("current_closure_action_ids")).replace("|", "/")
            future_actions = csv(item.get("future_closure_action_ids")).replace("|", "/")
            reports = statuses(item.get("report_statuses")).replace("|", "/")
            invariants = csv(item.get("required_invariants")).replace("|", "/")
            lines.append(
                f"| {item.get('id') or '-'} | {item.get('proof_status') or '-'} | "
                f"{current_actions} | {future_actions} | {reports} | {invariants} | {str(item.get('path_visible')).lower()} | "
                f"`{item.get('path') or '-'}` |"
            )

    remote = summaries.get("remote_access_diagnostics") if isinstance(summaries.get("remote_access_diagnostics"), dict) else {}
    operator_queue = summaries.get("operator_queue") if isinstance(summaries.get("operator_queue"), dict) else {}
    ready_submit = (
        summaries.get("operator_ready_submit_preflight")
        if isinstance(summaries.get("operator_ready_submit_preflight"), dict)
        else {}
    )
    safe_submit = (
        summaries.get("operator_safe_actions_preflight")
        if isinstance(summaries.get("operator_safe_actions_preflight"), dict)
        else {}
    )
    if remote or operator_queue or ready_submit or safe_submit:
        lines += ["", "### Current Execution Guardrails", ""]
        if remote:
            diagnosis = remote.get("diagnosis") if isinstance(remote.get("diagnosis"), dict) else {}
            lines.append(
                f"- Remote access diagnostics: `{remote.get('overall_status') or diagnosis.get('overall_status') or 'unknown'}`; "
                f"reachable_hosts=`{diagnosis.get('reachable_hosts')}`; remote_path_absence_proven=`{diagnosis.get('remote_path_absence_proven')}`."
            )
            findings = remote.get("configuration_findings") if isinstance(remote.get("configuration_findings"), list) else []
            for item in findings:
                lines.append(
                    f"- SSH config finding: `{item.get('host')}` uses HostName `{item.get('configured_hostname')}` "
                    f"({item.get('finding')})."
                )
        if operator_queue:
            counts = operator_queue.get("counts") if isinstance(operator_queue.get("counts"), dict) else {}
            lines.append(
                f"- Operator queue: `{operator_queue.get('overall_status') or 'unknown'}`; "
                f"ready_actions=`{counts.get('ready_actions')}`; slurm_actions=`{counts.get('slurm_actions')}`; heavy_gpu_actions=`{counts.get('heavy_gpu_actions')}`."
            )
        if ready_submit:
            counts = ready_submit.get("check_status_counts") or ready_submit.get("counts") or {}
            lines.append(
                f"- Ready-submit preflight: `{ready_submit.get('overall_status') or 'unknown'}`; "
                f"fail=`{counts.get('fail', 0)}`; warn=`{counts.get('warn', 0)}`."
            )
            for item in ready_submit.get("failed_checks") or []:
                lines.append(
                    f"- Ready-submit blocker: `{item.get('area')}/{item.get('name')}` "
                    f"{str(item.get('status') or '').upper()} - {item.get('detail')}"
                )
        if safe_submit:
            counts = safe_submit.get("check_status_counts") or safe_submit.get("counts") or {}
            lines.append(
                f"- Safe-action preflight: `{safe_submit.get('overall_status') or 'unknown'}`; "
                f"ready_safe_actions=`{len(safe_submit.get('ready_actions') or [])}`; pass=`{counts.get('pass', 0)}`."
            )

    if standalone:
        lines += [
            "",
            "Use `commands.sh` sections `1j_operator_safe_actions` and `1j_operator_resume_state` to execute only the ready action ids allowed for the current host.",
            "Do not advance to rollout capture, hidden-state dump, or training until the relevant gate row is closed.",
        ]
    return "\n".join(lines) + "\n"


def render_commands(args: argparse.Namespace) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'export HANDOFF_DIR="${HANDOFF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
        f'export REMOTE_HOST="${{REMOTE_HOST:-{DEFAULT_REMOTE_HOST}}}"',
        f'export REMOTE_WORKDIR="${{REMOTE_WORKDIR:-{DEFAULT_REMOTE_WORKDIR}}}"',
        'export EAGLE3_REPO_ROOT="${EAGLE3_REPO_ROOT:-}"',
        'if [[ -z "$EAGLE3_REPO_ROOT" ]]; then',
        '  if [[ -d "$PWD/experiments/eagle3_qwen3_235b" ]]; then',
        '    export EAGLE3_REPO_ROOT="$PWD"',
        '  elif [[ -d "$(dirname "$HANDOFF_DIR")/experiments/eagle3_qwen3_235b" ]]; then',
        '    export EAGLE3_REPO_ROOT="$(dirname "$HANDOFF_DIR")"',
        '  elif [[ -d "$REMOTE_WORKDIR/experiments/eagle3_qwen3_235b" ]]; then',
        '    export EAGLE3_REPO_ROOT="$REMOTE_WORKDIR"',
        f'  elif [[ -d {shlex.quote(str(ROOT / "experiments" / "eagle3_qwen3_235b"))} ]]; then',
        f'    export EAGLE3_REPO_ROOT={shlex.quote(str(ROOT))}',
        '  else',
        '    export EAGLE3_REPO_ROOT="$PWD"',
        '  fi',
        'fi',
        f"export LOCAL_ARTIFACT_ROOT={shlex.quote(str(args.artifact_root))}",
        f'export REMOTE_ARTIFACT_ROOT="${{REMOTE_ARTIFACT_ROOT:-{args.remote_artifact_root}}}"',
        'export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REMOTE_ARTIFACT_ROOT}"',
        f"export SBATCH_ACCOUNT={shlex.quote(args.sbatch_account)}",
        f"export SBATCH_PARTITION={shlex.quote(args.sbatch_partition)}",
        'export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"',
        'export MODELOPT_DIR="${MODELOPT_DIR:-$EAGLE3_REPO_ROOT/Model-Optimizer}"',
        'export COMPAT_MODELOPT_DIR="${COMPAT_MODELOPT_DIR:-}"',
        'export VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"',
        'export REFERENCE_ARCH="${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}"',
        'export ARCH_ENV_FILE="${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}"',
        'export CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"',
        'export SWE_REPO_ROOT="${SWE_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"',
        'export SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-$SWE_REPO_ROOT}"',
        'export REPO_ROOT="${REPO_ROOT:-$SWE_REPO_ROOT}"',
        'export SPECDEC_REMOTE_PATCH_ROOT="${SPECDEC_REMOTE_PATCH_ROOT:-$HANDOFF_DIR/specdec_rl_remote_patches}"',
        'export SPECDEC_REMOTE_PATCH_BUNDLE_JSON="${SPECDEC_REMOTE_PATCH_BUNDLE_JSON:-$ARTIFACT_ROOT/reports/specdec_rl_remote_patch_bundle.json}"',
        'export SPECDEC_REMOTE_PATCH_BUNDLE_MARKDOWN="${SPECDEC_REMOTE_PATCH_BUNDLE_MARKDOWN:-$ARTIFACT_ROOT/reports/specdec_rl_remote_patch_bundle.md}"',
        'export NEMO_RL_CONFIG="${NEMO_RL_CONFIG:-grpo_qwen3_235b_swe.yaml}"',
        'export CONFIG_FILE="${CONFIG_FILE:-$EAGLE3_REPO_ROOT/grpo_qwen3_235b_swe.yaml}"',
        'export ENV_FILE="${ENV_FILE:-$EAGLE3_REPO_ROOT/env.sh}"',
        'export INPUT_DATA="${INPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"',
        'export HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"',
        'export OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"',
        'export EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"',
        'export VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"',
        'export EXPORT_ARTIFACTS_JSON="${EXPORT_ARTIFACTS_JSON:-$ARTIFACT_ROOT/reports/eagle3_export_artifacts.json}"',
        'export EXPORT_ARTIFACTS_MARKDOWN="${EXPORT_ARTIFACTS_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_export_artifacts.md}"',
        'export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"',
        'export MOUNTS="${MOUNTS:-/lustre:/lustre,$EAGLE3_REPO_ROOT:$EAGLE3_REPO_ROOT,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"',
        'export CONTAINER_PREFLIGHT_JSON="${CONTAINER_PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/container_preflight_analysis.json}"',
        'export CONTAINER_PREFLIGHT_PIPELINE_JSON="${CONTAINER_PREFLIGHT_PIPELINE_JSON:-$ARTIFACT_ROOT/reports/container_preflight_pipeline.json}"',
        'export CONTAINER_PREFLIGHT_PIPELINE_MARKDOWN="${CONTAINER_PREFLIGHT_PIPELINE_MARKDOWN:-$ARTIFACT_ROOT/reports/container_preflight_pipeline.md}"',
        'export ROLLOUT_CAPTURE_JSON="${ROLLOUT_CAPTURE_JSON:-$ARTIFACT_ROOT/reports/rollout_capture_validation.json}"',
        'export ROLLOUT_CAPTURE_ANALYSIS_JSON="${ROLLOUT_CAPTURE_ANALYSIS_JSON:-$ARTIFACT_ROOT/reports/rollout_capture_analysis.json}"',
        'export ROLLOUT_CAPTURE_JOB_JSON="${ROLLOUT_CAPTURE_JOB_JSON:-$ARTIFACT_ROOT/reports/rollout_capture_job_analysis.json}"',
        'export ROLLOUT_SUBMIT_PREFLIGHT_JSON="${ROLLOUT_SUBMIT_PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/rollout_capture_submit_preflight.json}"',
        'export ROLLOUT_STATE_ADVANCE_JSON="${ROLLOUT_STATE_ADVANCE_JSON:-$ARTIFACT_ROOT/reports/rollout_capture_state_advance.json}"',
        'export CORPUS_STRATEGY_JSON="${CORPUS_STRATEGY_JSON:-$ARTIFACT_ROOT/reports/corpus_strategy.json}"',
        'export TRAINING_SCALE_JSON="${TRAINING_SCALE_JSON:-$ARTIFACT_ROOT/reports/eagle3_training_scale.json}"',
        'export TRAINING_PATH_MANIFEST_JSON="${TRAINING_PATH_MANIFEST_JSON:-$ARTIFACT_ROOT/reports/eagle3_training_path_manifest.json}"',
        'export TRAINING_PATH_MANIFEST_MARKDOWN="${TRAINING_PATH_MANIFEST_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_training_path_manifest.md}"',
        'export TRAINING_PATH_MANIFEST_VALIDATION_JSON="${TRAINING_PATH_MANIFEST_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_training_path_manifest_validation.json}"',
        'export TRAINING_PATH_MANIFEST_VALIDATION_MARKDOWN="${TRAINING_PATH_MANIFEST_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_training_path_manifest_validation.md}"',
        'export TRAINING_CKPT_VALIDATION_JSON="${TRAINING_CKPT_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_training_checkpoint.json}"',
        'export TRAINING_CKPT_VALIDATION_MARKDOWN="${TRAINING_CKPT_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_training_checkpoint.md}"',
        'export NEXT_ACTION_PLAN_JSON="${NEXT_ACTION_PLAN_JSON:-$ARTIFACT_ROOT/reports/eagle3_next_actions.json}"',
        'export NEXT_ACTION_PLAN_VALIDATION_JSON="${NEXT_ACTION_PLAN_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_next_actions_validation.json}"',
        'export NEXT_ACTION_TRANSITIONS_JSON="${NEXT_ACTION_TRANSITIONS_JSON:-$ARTIFACT_ROOT/reports/eagle3_next_action_transitions.json}"',
        'export OPERATOR_QUEUE_TRANSITIONS_JSON="${OPERATOR_QUEUE_TRANSITIONS_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_queue_transitions.json}"',
        'export OPERATOR_QUEUE_TRANSITIONS_MARKDOWN="${OPERATOR_QUEUE_TRANSITIONS_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_queue_transitions.md}"',
        'export OPERATOR_SHEET_JSON="${OPERATOR_SHEET_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_sheet.json}"',
        'export OPERATOR_SHEET_MARKDOWN="${OPERATOR_SHEET_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_sheet.md}"',
        'export OPERATOR_SHEET_VALIDATION_JSON="${OPERATOR_SHEET_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_sheet_validation.json}"',
        'export OPERATOR_SHEET_VALIDATION_MARKDOWN="${OPERATOR_SHEET_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_sheet_validation.md}"',
        'export OPERATOR_EXECUTION_JSON="${OPERATOR_EXECUTION_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_execution.json}"',
        'export OPERATOR_EXECUTION_MARKDOWN="${OPERATOR_EXECUTION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_execution.md}"',
        'export OPERATOR_FOLLOWUP_VALIDATION_JSON="${OPERATOR_FOLLOWUP_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_followups_validation.json}"',
        'export OPERATOR_FOLLOWUP_VALIDATION_MARKDOWN="${OPERATOR_FOLLOWUP_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_followups_validation.md}"',
        'export MEGATRON_PROBE_FOLLOWUP_VALIDATION_JSON="${MEGATRON_PROBE_FOLLOWUP_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/megatron_probe_followup_validation.json}"',
        'export MEGATRON_PROBE_FOLLOWUP_VALIDATION_MARKDOWN="${MEGATRON_PROBE_FOLLOWUP_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/megatron_probe_followup_validation.md}"',
        'export PREFLIGHT_ROBUSTNESS_VALIDATION_JSON="${PREFLIGHT_ROBUSTNESS_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_preflight_robustness_validation.json}"',
        'export PREFLIGHT_ROBUSTNESS_VALIDATION_MARKDOWN="${PREFLIGHT_ROBUSTNESS_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_preflight_robustness_validation.md}"',
        'export MODELOPT_RECIPE_OVERRIDES_JSON="${MODELOPT_RECIPE_OVERRIDES_JSON:-$ARTIFACT_ROOT/reports/modelopt_recipe_overrides_current.json}"',
        'export MODELOPT_RECIPE_OVERRIDES_MARKDOWN="${MODELOPT_RECIPE_OVERRIDES_MARKDOWN:-$ARTIFACT_ROOT/reports/modelopt_recipe_overrides_current.md}"',
        'export PROBE_JOB_ID="${PROBE_JOB_ID:-2867766}"',
        'export OPERATOR_SUBMIT_PACKET_JSON="${OPERATOR_SUBMIT_PACKET_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_submit_packet.json}"',
        'export OPERATOR_SUBMIT_PACKET_MARKDOWN="${OPERATOR_SUBMIT_PACKET_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_submit_packet.md}"',
        'export OPERATOR_SUBMIT_PACKET_VALIDATION_JSON="${OPERATOR_SUBMIT_PACKET_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_submit_packet_validation.json}"',
        'export OPERATOR_SUBMIT_PACKET_VALIDATION_MARKDOWN="${OPERATOR_SUBMIT_PACKET_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_submit_packet_validation.md}"',
        'export OPERATOR_READY_SUBMIT_PREFLIGHT_JSON="${OPERATOR_READY_SUBMIT_PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_ready_submit_preflight.json}"',
        'export OPERATOR_READY_SUBMIT_PREFLIGHT_MARKDOWN="${OPERATOR_READY_SUBMIT_PREFLIGHT_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_ready_submit_preflight.md}"',
        'export OPERATOR_SAFE_ACTIONS_PREFLIGHT_JSON="${OPERATOR_SAFE_ACTIONS_PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_safe_actions_preflight.json}"',
        'export OPERATOR_SAFE_ACTIONS_PREFLIGHT_MARKDOWN="${OPERATOR_SAFE_ACTIONS_PREFLIGHT_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_safe_actions_preflight.md}"',
        'export OPERATOR_QUEUE_JSON="${OPERATOR_QUEUE_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_queue.json}"',
        'export OPERATOR_QUEUE_MARKDOWN="${OPERATOR_QUEUE_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_queue.md}"',
        'export OPERATOR_STATE_REFRESH_JSON="${OPERATOR_STATE_REFRESH_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_state_refresh.json}"',
        'export OPERATOR_STATE_REFRESH_MARKDOWN="${OPERATOR_STATE_REFRESH_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_state_refresh.md}"',
        'export OPERATOR_STATE_REFRESH_VALIDATION_JSON="${OPERATOR_STATE_REFRESH_VALIDATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_operator_state_refresh_validation.json}"',
        'export OPERATOR_STATE_REFRESH_VALIDATION_MARKDOWN="${OPERATOR_STATE_REFRESH_VALIDATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_operator_state_refresh_validation.md}"',
        'export COMPLETION_CONTRACT_JSON="${COMPLETION_CONTRACT_JSON:-$ARTIFACT_ROOT/reports/eagle3_completion_contract.json}"',
        'export COMPLETION_CONTRACT_MARKDOWN="${COMPLETION_CONTRACT_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_completion_contract.md}"',
        'export SLURM_CAPACITY_JSON="${SLURM_CAPACITY_JSON:-$ARTIFACT_ROOT/reports/eagle3_slurm_capacity.json}"',
        'export SLURM_CAPACITY_MARKDOWN="${SLURM_CAPACITY_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_slurm_capacity.md}"',
        'export SLURM_CAPACITY_ENV="${SLURM_CAPACITY_ENV:-$ARTIFACT_ROOT/reports/eagle3_resource_profile.env}"',
        'export RESOURCE_PROFILE_APPLICATION_JSON="${RESOURCE_PROFILE_APPLICATION_JSON:-$ARTIFACT_ROOT/reports/eagle3_resource_profile_application.json}"',
        'export RESOURCE_PROFILE_APPLICATION_MARKDOWN="${RESOURCE_PROFILE_APPLICATION_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_resource_profile_application.md}"',
        'export GOAL_EVIDENCE_JSON="${GOAL_EVIDENCE_JSON:-$ARTIFACT_ROOT/reports/eagle3_goal_evidence.json}"',
        'export GOAL_EVIDENCE_MARKDOWN="${GOAL_EVIDENCE_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_goal_evidence.md}"',
        'export MODELOPT_LOSS_MASK_JSON="${MODELOPT_LOSS_MASK_JSON:-$ARTIFACT_ROOT/reports/modelopt_loss_mask_patch.json}"',
        'export PIPELINE_SUBMIT_PREFLIGHT_JSON="${PIPELINE_SUBMIT_PREFLIGHT_JSON:-$ARTIFACT_ROOT/reports/eagle3_pipeline_submit_preflight.json}"',
        'export PIPELINE_ANALYSIS_JSON="${PIPELINE_ANALYSIS_JSON:-$ARTIFACT_ROOT/reports/eagle3_pipeline_analysis.json}"',
        'export SWEEP_JSON="${SWEEP_JSON:-$ARTIFACT_ROOT/reports/trained_draft_spec_tokens_sweep.json}"',
        'export SPECFORGE_REFERENCE_JSON="${SPECFORGE_REFERENCE_JSON:-$ARTIFACT_ROOT/reports/specforge_reference.json}"',
        'export HAYATE_SPECFORGE_REFERENCE_JSON="${HAYATE_SPECFORGE_REFERENCE_JSON:-$ARTIFACT_ROOT/reports/hayate_specforge_reference.json}"',
        'export HAYATE_SPECFORGE_REFERENCE_MARKDOWN="${HAYATE_SPECFORGE_REFERENCE_MARKDOWN:-$ARTIFACT_ROOT/reports/hayate_specforge_reference.md}"',
        'export HAYATE_WORKFLOW_JSON="${HAYATE_WORKFLOW_JSON:-$ARTIFACT_ROOT/reports/hayate_modelopt_workflow.json}"',
        'export DRAFT_INVENTORY_JSON="${DRAFT_INVENTORY_JSON:-$ARTIFACT_ROOT/reports/eagle3_draft_config_inventory.json}"',
        'export DRAFT_INVENTORY_MARKDOWN="${DRAFT_INVENTORY_MARKDOWN:-$ARTIFACT_ROOT/reports/eagle3_draft_config_inventory.md}"',
        'export EAGLE3_TARGET_CONTEXT="${EAGLE3_TARGET_CONTEXT:-swe_rl}"',
        "",
        "EAGLE3_AVAILABLE_SECTIONS=(",
    ]
    for title, _body in DEFAULT_COMMANDS:
        lines.append(f"  {shlex.quote(title)}")
    lines += [
        ")",
        'EAGLE3_SUBMIT_SECTIONS=(3_submit_pilot 5_sweep_trained_draft)',
        "",
        "is_true() {",
        "  case \"${1:-}\" in",
        "    true|True|TRUE|1|yes|Yes|YES) return 0 ;;",
        "    *) return 1 ;;",
        "  esac",
        "}",
        "",
        "print_sections() {",
        "  cat <<'EOF'",
        "Usage:",
        "  bash commands.sh <section> [section ...]",
        "  EAGLE3_COMMAND_SECTIONS=\"section section\" bash commands.sh",
        "  EAGLE3_RUN_ALL_SECTIONS=true bash commands.sh",
        "  EAGLE3_PRINT_SELECTED_SECTIONS=true EAGLE3_RUN_ALL_SECTIONS=true bash commands.sh",
        "",
        "Default behavior is print-only. No command section runs unless selected explicitly.",
        "Submit sections require EAGLE3_ALLOW_SUBMIT_SECTIONS=true even when selected explicitly.",
        "RUN_ALL mode excludes submit sections; select submit sections by name after reviewing gate evidence.",
        "Available sections:",
        "EOF",
        "  printf '  %s\\n' \"${EAGLE3_AVAILABLE_SECTIONS[@]}\"",
        "}",
        "",
        "is_submit_section() {",
        "  local section=\"$1\"",
        "  local item",
        "  for item in \"${EAGLE3_SUBMIT_SECTIONS[@]}\"; do",
        "    [[ \"$section\" == \"$item\" ]] && return 0",
        "  done",
        "  return 1",
        "}",
        "",
        "guard_submit_section() {",
        "  local section=\"$1\"",
        "  if is_submit_section \"$section\" && ! is_true \"${EAGLE3_ALLOW_SUBMIT_SECTIONS:-false}\"; then",
        "    echo \"refusing submit section $section without EAGLE3_ALLOW_SUBMIT_SECTIONS=true\" >&2",
        "    return 3",
        "  fi",
        "  return 0",
        "}",
        "",
        "ensure_repo_root() {",
        "  if [[ ! -d \"$EAGLE3_REPO_ROOT/experiments/eagle3_qwen3_235b\" ]]; then",
        "    echo \"EAGLE3_REPO_ROOT must point at the Nemo-RL_Qwen3_Roadmap checkout; current value: $EAGLE3_REPO_ROOT\" >&2",
        "    return 4",
        "  fi",
        "  cd \"$EAGLE3_REPO_ROOT\"",
        "}",
        "",
        "run_section() {",
        "  local section=\"$1\"",
        "  guard_submit_section \"$section\"",
        "  case \"$section\" in",
    ]
    for title, body in DEFAULT_COMMANDS:
        lines += [
            f"    {title})",
            f"      echo '# === {title} ==='",
        ]
        if title != "0_restore_materialized_static_inputs":
            lines.append("      ensure_repo_root")
        body_lines = body.rstrip().splitlines()
        lines.extend(f"      {line}" if line else "" for line in body_lines)
        lines += [
            "      ;;",
        ]
    lines += [
        "    *)",
        "      echo \"unknown handoff command section: $section\" >&2",
        "      print_sections >&2",
        "      return 2",
        "      ;;",
        "  esac",
        "}",
        "",
        "sections_to_run=()",
        "if [[ \"$#\" -gt 0 ]]; then",
        "  sections_to_run=(\"$@\")",
        "elif [[ -n \"${EAGLE3_COMMAND_SECTIONS:-}\" ]]; then",
        "  read -r -a sections_to_run <<< \"$EAGLE3_COMMAND_SECTIONS\"",
        "elif is_true \"${EAGLE3_RUN_ALL_SECTIONS:-false}\"; then",
        "  for section in \"${EAGLE3_AVAILABLE_SECTIONS[@]}\"; do",
        "    if is_submit_section \"$section\"; then",
        "      echo \"skipping submit section $section in RUN_ALL mode; select it by name after gate review\" >&2",
        "      continue",
        "    fi",
        "    sections_to_run+=(\"$section\")",
        "  done",
        "else",
        "  print_sections",
        "  exit 0",
        "fi",
        "",
        "if is_true \"${EAGLE3_PRINT_SELECTED_SECTIONS:-false}\"; then",
        "  printf '%s\\n' \"${sections_to_run[@]}\"",
        "  exit 0",
        "fi",
        "",
        "for section in \"${sections_to_run[@]}\"; do",
        "  run_section \"$section\"",
        "done",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_runbook(manifest: dict[str, Any], copied: dict[str, str | None]) -> str:
    command_account = manifest.get("command_sbatch_account") or DEFAULT_SBATCH_ACCOUNT
    lines = [
        "# Qwen3-235B Eagle3 Handoff Bundle",
        "",
        f"Generated: `{manifest['generated_at']}`",
        f"Repo root: `{manifest['repo_root']}`",
        f"Evidence artifact root: `{manifest['artifact_root']}`",
        f"Command artifact root default: `{manifest['command_artifact_root_default']}`",
        f"Command Slurm account default: `{command_account}`",
        "",
        "## Current Status",
        "",
        "| report | status | summary |",
        "| --- | --- | --- |",
    ]
    for label, item in manifest["inputs"].items():
        summary = manifest["summaries"].get(label)
        compact = json.dumps(summary, sort_keys=True) if summary else "-"
        lines.append(f"| {label} | {item['status']} | `{compact}` |")

    gate_closure = render_gate_closure_markdown(manifest, standalone=False)
    if gate_closure:
        lines += ["", gate_closure.rstrip()]

    lines += [
        "",
        "## Files In This Bundle",
        "",
    ]
    for label, name in copied.items():
        if name:
            lines.append(f"- {label}: `{name}`")
    lines += [
        "- command sheet: `commands.sh`",
        "- manifest: `manifest.json`",
        "",
        f"Stale files removed during generation: `{len(manifest.get('stale_files_removed') or [])}`",
        "",
        "## Next Action",
        "",
        "`commands.sh` is section-selectable: running it without arguments prints the available sections and executes nothing. `EAGLE3_RUN_ALL_SECTIONS=true` excludes submit sections. Submit sections (`3_submit_pilot`, `5_sweep_trained_draft`) must be selected by name and also require `EAGLE3_ALLOW_SUBMIT_SECTIONS=true`. Before running provenance or preflight commands from a copied handoff, run `bash commands.sh 0_restore_materialized_static_inputs` to rebuild `$ARTIFACT_ROOT/verifier_config`, `$ARTIFACT_ROOT/templates`, and `$ARTIFACT_ROOT/architecture` from this bundle.",
        "",
        f"1. Confirm `SBATCH_ACCOUNT` in `commands.sh` (currently `{command_account}`) and override it only if the cluster Slurm association differs. It defaults `ARTIFACT_ROOT` to the remote Lustre artifact root; override `ARTIFACT_ROOT` only for local dry-runs.",
        "2. Run `probe_eagle3_remote_host.py` and keep the report; completion requires a reachable host with visible ModelOpt, Hayate/SpecForge, artifact-root, `git`, and `python3` evidence.",
        "3. Run `diagnose_eagle3_remote_access.py` after each remote probe; if it reports `blocked_local_dns`, fix VPN/DNS or run the handoff from a cluster login context before treating remote Hayate paths as absent.",
        "4. On the cluster host, run discovery and source the generated env file.",
        "5. Run the container preflight dry-run, then analyze it with `analyze_container_preflight.py`.",
        "6. Run the rollout capture gate and `preflight_rollout_capture_submit.py`; then dry-run or submit the short capture plan and use `advance_rollout_capture_state.py` to choose submit, poll, materialize, or pipeline dry-run.",
        "7. Run `plan_eagle3_next_actions.py` to consolidate container, rollout, pipeline, and training-scale gates into one ordered action list.",
        "8. Generate `eagle3_operator_sheet.md` with `create_eagle3_operator_sheet.py`; it lists ready actions in execution order, the safe print-only wrapper, explicit execute flags, and follow-up analyzer commands.",
        "9. Run `validate_eagle3_operator_sheet.py` and require PASS before copying any execute command from the sheet.",
        "10. Run `validate_eagle3_next_action_transitions.py` to prove the planner promotes the expected sequence before any Slurm execution.",
        "11. Run `validate_eagle3_preflight_robustness.py` to prove lightweight-host preflights emit structured evidence without leaking token env values.",
        "12. Run `validate_eagle3_operator_state_refresh.py` to prove the broader refresh preserves ModelOpt loss-mask, recipe, and goal-evidence reports.",
        "13. Run `validate_modelopt_recipe_overrides.py` so the current offline training wrapper has machine-readable ModelOpt recipe evidence.",
        "14. Run `validate_megatron_probe_followup.py`; then use `followup_megatron_probe_to_rollout.sh` to poll the Megatron probe and print the rollout retry command only after PASS.",
        "15. Use the `1j_operator_safe_actions` section to run only `probe_remote_hosts` and `poll_megatron_compat_probe`; it scopes ready-submit preflight with `--action-ids` and does not submit Slurm.",
        "16. On the cluster host only, after ready-submit preflight PASS for runtime/container actions, set `EXECUTE_SLURM_ACTIONS=true` and a narrow `SLURM_ACTION_IDS` value on `resume_eagle3_operator_state.sh` to submit only the current non-heavy runtime/container gates.",
        "17. Run `validate_eagle3_completion_contract.py` to prove the final audit accepts good trained/export/sweep evidence and rejects stale sweep evidence.",
        "18. Run `probe_eagle3_slurm_capacity.py`; if the partition exposes 4-GPU nodes, do not submit an 8-GPU-per-node hidden-state/train pipeline without changing the resource env or partition.",
        "19. Use `python3 experiments/eagle3_qwen3_235b/run_eagle3_next_action.py --artifact-root \"$ARTIFACT_ROOT\" --list` to inspect the operator actions; it is print-only unless `--execute` and the relevant allow flags are passed.",
        "20. For Slurm actions, use `run_eagle3_slurm_followups.py` to prove the submitted job ids are terminal before running `after_commands`.",
        "21. Submit the preflight-only job and require container preflight PASS before hidden-state dump.",
        "22. Run `estimate_eagle3_training_scale.py` to choose pilot, calibration, or production-candidate scale.",
        "23. Run `preflight_eagle3_pipeline_submit.py` and require `submit_ready=true` before the expensive pipeline.",
        "24. Run `SUBMIT=false RUN_PILOT=true ... bootstrap_eagle3_path.sh` and inspect readiness.",
        "25. Submit the pilot only after readiness has no unexpected failures.",
        "26. Use `analyze_eagle3_pipeline.py` after Slurm logs appear; follow `next_action.resume_command` if a stage fails.",
        "27. Require `eagle3_training_checkpoint.json` PASS before trusting an export; it proves the ModelOpt checkpoint contains HF weights, trainer state, and `modelopt_state.pth`.",
        "28. Generate `eagle3_goal_evidence.md` to keep a requirement-by-requirement proof matrix for the user-facing goal.",
        "29. After export, run the trained-draft token sweep before longer RL runs; its job file must record `VLLM_DRAFT_DIR`, `ARTIFACT_ROOT`, `CONFIG_FILE`, `ENV_FILE`, `CHAT_TEMPLATE`, and the SpecDec-RL repo path for the completion audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report_dir = args.artifact_root / "reports"
    report_defaults = {
        "provenance_json": report_dir / "eagle3_provenance.json",
        "remote_host_probe_json": report_dir / "eagle3_remote_host_probe.json",
        "remote_access_diagnostics_json": report_dir / "eagle3_remote_access_diagnostics.json",
        "input_discovery_json": args.artifact_root / "eagle3_input_discovery.json",
        "static_inputs_json": report_dir / "qwen3_static_inputs.json",
        "static_inputs_validation_json": report_dir / "qwen3_static_inputs_materialization_validation.json",
        "cluster_probe_json": report_dir / "cluster_environment_probe.json",
        "upstream_drift_json": report_dir / "modelopt_upstream_drift.json",
        "modelopt_loss_mask_json": report_dir / "modelopt_loss_mask_patch.json",
        "modelopt_recipe_overrides_json": report_dir / "modelopt_recipe_overrides_current.json",
        "modelopt_patch_manifest": args.artifact_root / "patches" / "modelopt_eagle3_qwen3" / "manifest.json",
        "readiness_json": report_dir / "eagle3_readiness.json",
        "container_preflight_json": report_dir / "container_preflight_analysis.json",
        "nemo_rl_specdec_json": report_dir / "nemo_rl_specdec_integration.json",
        "nemo_rl_drift_json": report_dir / "nemo_rl_eagle3_drift.json",
        "specdec_remote_patch_bundle_json": report_dir / "specdec_rl_remote_patch_bundle.json",
        "rollout_capture_json": report_dir / "rollout_capture_validation.json",
        "rollout_capture_analysis_json": report_dir / "rollout_capture_analysis.json",
        "rollout_capture_job_json": report_dir / "rollout_capture_job_analysis.json",
        "rollout_submit_preflight_json": report_dir / "rollout_capture_submit_preflight.json",
        "rollout_state_advance_json": report_dir / "rollout_capture_state_advance.json",
        "corpus_strategy_json": report_dir / "corpus_strategy.json",
        "training_scale_json": report_dir / "eagle3_training_scale.json",
        "training_path_manifest_json": report_dir / "eagle3_training_path_manifest.json",
        "training_path_manifest_markdown": report_dir / "eagle3_training_path_manifest.md",
        "training_path_manifest_validation_json": report_dir / "eagle3_training_path_manifest_validation.json",
        "training_path_manifest_validation_markdown": report_dir / "eagle3_training_path_manifest_validation.md",
        "next_action_plan_json": report_dir / "eagle3_next_actions.json",
        "next_action_plan_validation_json": report_dir / "eagle3_next_actions_validation.json",
        "next_action_transitions_json": report_dir / "eagle3_next_action_transitions.json",
        "operator_queue_transitions_json": report_dir / "eagle3_operator_queue_transitions.json",
        "operator_sheet_json": report_dir / "eagle3_operator_sheet.json",
        "operator_sheet_validation_json": report_dir / "eagle3_operator_sheet_validation.json",
        "operator_execution_json": report_dir / "eagle3_operator_execution.json",
        "operator_followup_validation_json": report_dir / "eagle3_operator_followups_validation.json",
        "megatron_probe_followup_validation_json": report_dir / "megatron_probe_followup_validation.json",
        "preflight_robustness_validation_json": report_dir / "eagle3_preflight_robustness_validation.json",
        "operator_submit_packet_json": report_dir / "eagle3_operator_submit_packet.json",
        "operator_submit_packet_validation_json": report_dir / "eagle3_operator_submit_packet_validation.json",
        "operator_ready_submit_preflight_json": report_dir / "eagle3_operator_ready_submit_preflight.json",
        "operator_safe_actions_preflight_json": report_dir / "eagle3_operator_safe_actions_preflight.json",
        "operator_queue_json": report_dir / "eagle3_operator_queue.json",
        "operator_state_refresh_json": report_dir / "eagle3_operator_state_refresh.json",
        "operator_state_refresh_validation_json": report_dir / "eagle3_operator_state_refresh_validation.json",
        "completion_contract_json": report_dir / "eagle3_completion_contract.json",
        "slurm_capacity_json": report_dir / "eagle3_slurm_capacity.json",
        "slurm_capacity_env": report_dir / "eagle3_resource_profile.env",
        "resource_profile_application_json": report_dir / "eagle3_resource_profile_application.json",
        "pipeline_submit_preflight_json": report_dir / "eagle3_pipeline_submit_preflight.json",
        "specforge_reference_json": report_dir / "specforge_reference.json",
        "hayate_specforge_reference_json": report_dir / "hayate_specforge_reference.json",
        "hayate_workflow_json": report_dir / "hayate_modelopt_workflow.json",
        "pipeline_analysis_json": report_dir / "eagle3_pipeline_analysis.json",
        "training_checkpoint_json": report_dir / "eagle3_training_checkpoint.json",
        "export_artifacts_json": report_dir / "eagle3_export_artifacts.json",
        "sweep_json": report_dir / "trained_draft_spec_tokens_sweep.json",
        "completion_json": report_dir / "eagle3_completion_audit.json",
        "goal_evidence_json": report_dir / "eagle3_goal_evidence.json",
        "hayate_inventory": report_dir / "hayate_inventory.txt",
        "draft_inventory_json": report_dir / "eagle3_draft_config_inventory.json",
    }
    for attr, default in report_defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, default)
    if not is_concrete_sbatch_account(args.sbatch_account):
        args.sbatch_account = (
            infer_sbatch_account_from_reports(
                [
                    args.cluster_probe_json,
                    args.operator_sheet_json,
                    args.operator_submit_packet_json,
                    args.operator_ready_submit_preflight_json,
                    args.rollout_submit_preflight_json,
                    args.pipeline_submit_preflight_json,
                    args.container_preflight_json,
                    args.operator_state_refresh_json,
                ]
            )
            or DEFAULT_SBATCH_ACCOUNT
        )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "repo_root": str(ROOT),
        "artifact_root": str(args.artifact_root),
        "command_artifact_root_default": str(args.remote_artifact_root),
        "command_sbatch_account": args.sbatch_account,
        "command_sbatch_partition": args.sbatch_partition,
        "inputs": {},
        "summaries": {},
        "stale_files_removed": [],
    }

    copied: dict[str, str | None] = {}
    report_specs = {
        "provenance": args.provenance_json,
        "remote_host_probe": args.remote_host_probe_json,
        "remote_access_diagnostics": args.remote_access_diagnostics_json,
        "input_discovery": args.input_discovery_json,
        "static_inputs": args.static_inputs_json,
        "static_inputs_materialization_validation": args.static_inputs_validation_json,
        "cluster_probe": args.cluster_probe_json,
        "upstream_drift": args.upstream_drift_json,
        "modelopt_loss_mask": args.modelopt_loss_mask_json,
        "modelopt_recipe_overrides": args.modelopt_recipe_overrides_json,
        "modelopt_patch": args.modelopt_patch_manifest,
        "readiness": args.readiness_json,
        "container_preflight": args.container_preflight_json,
        "nemo_rl_specdec": args.nemo_rl_specdec_json,
        "nemo_rl_drift": args.nemo_rl_drift_json,
        "specdec_remote_patch_bundle": args.specdec_remote_patch_bundle_json,
        "rollout_capture": args.rollout_capture_json,
        "rollout_capture_analysis": args.rollout_capture_analysis_json,
        "rollout_capture_job": args.rollout_capture_job_json,
        "rollout_submit_preflight": args.rollout_submit_preflight_json,
        "rollout_state_advance": args.rollout_state_advance_json,
        "corpus_strategy": args.corpus_strategy_json,
        "training_scale": args.training_scale_json,
        "training_path_manifest": args.training_path_manifest_json,
        "training_path_manifest_markdown": args.training_path_manifest_markdown,
        "training_path_manifest_validation": args.training_path_manifest_validation_json,
        "training_path_manifest_validation_markdown": args.training_path_manifest_validation_markdown,
        "next_action_plan": args.next_action_plan_json,
        "next_action_plan_validation": args.next_action_plan_validation_json,
        "next_action_transitions": args.next_action_transitions_json,
        "operator_queue_transitions": args.operator_queue_transitions_json,
        "operator_sheet": args.operator_sheet_json,
        "operator_sheet_validation": args.operator_sheet_validation_json,
        "operator_execution": args.operator_execution_json,
        "operator_followup_validation": args.operator_followup_validation_json,
        "megatron_probe_followup_validation": args.megatron_probe_followup_validation_json,
        "preflight_robustness_validation": args.preflight_robustness_validation_json,
        "operator_submit_packet": args.operator_submit_packet_json,
        "operator_submit_packet_validation": args.operator_submit_packet_validation_json,
        "operator_ready_submit_preflight": args.operator_ready_submit_preflight_json,
        "operator_safe_actions_preflight": args.operator_safe_actions_preflight_json,
        "operator_queue": args.operator_queue_json,
        "operator_state_refresh": args.operator_state_refresh_json,
        "operator_state_refresh_validation": args.operator_state_refresh_validation_json,
        "completion_contract": args.completion_contract_json,
        "slurm_capacity": args.slurm_capacity_json,
        "slurm_capacity_env": args.slurm_capacity_env,
        "resource_profile_application": args.resource_profile_application_json,
        "pipeline_submit_preflight": args.pipeline_submit_preflight_json,
        "specforge_reference": args.specforge_reference_json,
        "hayate_specforge_reference": args.hayate_specforge_reference_json,
        "hayate_workflow": args.hayate_workflow_json,
        "pipeline_analysis": args.pipeline_analysis_json,
        "training_checkpoint": args.training_checkpoint_json,
        "export_artifacts": args.export_artifacts_json,
        "spec_tokens_sweep": args.sweep_json,
        "completion_audit": args.completion_json,
        "goal_evidence": args.goal_evidence_json,
        "hayate_inventory": args.hayate_inventory,
        "draft_inventory": args.draft_inventory_json,
    }
    static_specs = {
        "repo_readme": EXP / "README.md",
        "draft_model_playbook": EXP / "EAGLE3_DRAFT_MODEL_PLAYBOOK.md",
        "remote_cluster_status": EXP / "REMOTE_CLUSTER_STATUS.md",
        "remote_execution_inputs": EXP / "REMOTE_EXECUTION_INPUTS.md",
        "specdec_remote_patches_doc": EXP / "SPECDEC_RL_REMOTE_PATCHES.md",
        "specdec_role_logging_patch": EXP / "specdec_rl_rollout_role_logging.patch",
        "specdec_rl_remote_patches": EXP / "remote_patches/SpecDec-RL",
        "megatron_compat_probe_job": ROOT / "latest_megatron_compat_probe_job.txt",
        "architecture_reference": EXP / "qwen3_235b_thinking_eagle3_architecture.json",
        "materialized_verifier_config": args.artifact_root / "verifier_config/config.json",
        "materialized_generation_config": args.artifact_root / "verifier_config/generation_config.json",
        "materialized_tokenizer_config": args.artifact_root / "verifier_config/tokenizer_config.json",
        "materialized_chat_template": args.artifact_root / "templates/qwen3_generation_template.jinja2",
        "materialized_chat_template_mask_validation": args.artifact_root
        / "templates/qwen3_generation_template.mask_validation.json",
        "materialized_architecture_json": args.artifact_root / "architecture/eagle3_architecture.json",
        "materialized_architecture_env": args.artifact_root / "architecture/eagle3_architecture.env",
        "materialized_architecture_dotlist": args.artifact_root / "architecture/eagle3_architecture.dotlist",
        "nemo_rl_specdec_overlay": EXP / "nemo_rl_specdec_overlay.yaml",
        "nemo_rl_eagle3_online_draft_overlay": EXP / "nemo_rl_eagle3_online_draft_overlay.yaml",
    }
    if args.include_html:
        static_specs["dashboard"] = EXP / "specdec_progress.html"

    if args.clean_stale:
        expected_names = {"commands.sh", "GATE_CLOSURE.md", "RUNBOOK.md", "manifest.json"}
        managed_labels = set(report_specs) | set(static_specs)
        for label, src in {**report_specs, **static_specs}.items():
            if src is not None and src.exists():
                expected_names.add(bundle_dest_name(label, src))
        manifest["stale_files_removed"] = clean_stale_outputs(out, expected_names, managed_labels)

    for label, src in report_specs.items():
        copied[label] = copy_if_exists(src, out, label, manifest)
        manifest["summaries"][label] = json_summary(src)

    copied["readme"] = copy_if_exists(static_specs["repo_readme"], out, "repo_readme", manifest)
    copied["playbook"] = copy_if_exists(static_specs["draft_model_playbook"], out, "draft_model_playbook", manifest)
    copied["remote_cluster_status"] = copy_if_exists(static_specs["remote_cluster_status"], out, "remote_cluster_status", manifest)
    copied["remote_execution_inputs"] = copy_if_exists(static_specs["remote_execution_inputs"], out, "remote_execution_inputs", manifest)
    copied["specdec_remote_patches_doc"] = copy_if_exists(
        static_specs["specdec_remote_patches_doc"],
        out,
        "specdec_remote_patches_doc",
        manifest,
    )
    copied["specdec_role_logging_patch"] = copy_if_exists(
        static_specs["specdec_role_logging_patch"],
        out,
        "specdec_role_logging_patch",
        manifest,
    )
    copied["specdec_rl_remote_patches"] = copy_tree_if_exists(
        static_specs["specdec_rl_remote_patches"],
        out,
        "specdec_rl_remote_patches",
        manifest,
    )
    copied["megatron_compat_probe_job"] = copy_if_exists(static_specs["megatron_compat_probe_job"], out, "megatron_compat_probe_job", manifest)
    copied["architecture_reference"] = copy_if_exists(static_specs["architecture_reference"], out, "architecture_reference", manifest)
    for label in [
        "materialized_verifier_config",
        "materialized_generation_config",
        "materialized_tokenizer_config",
        "materialized_chat_template",
        "materialized_chat_template_mask_validation",
        "materialized_architecture_json",
        "materialized_architecture_env",
        "materialized_architecture_dotlist",
    ]:
        copied[label] = copy_if_exists(static_specs[label], out, label, manifest)
    copied["fixed_draft_overlay"] = copy_if_exists(static_specs["nemo_rl_specdec_overlay"], out, "nemo_rl_specdec_overlay", manifest)
    copied["online_draft_overlay"] = copy_if_exists(
        static_specs["nemo_rl_eagle3_online_draft_overlay"],
        out,
        "nemo_rl_eagle3_online_draft_overlay",
        manifest,
    )
    if args.include_html:
        copied["dashboard"] = copy_if_exists(static_specs["dashboard"], out, "dashboard", manifest)

    commands = render_commands(args)
    (out / "commands.sh").write_text(commands)
    (out / "commands.sh").chmod(0o755)

    gate_closure = render_gate_closure_markdown(manifest, standalone=True)
    if gate_closure:
        gate_closure_path = out / "GATE_CLOSURE.md"
        gate_closure_path.write_text(gate_closure)
        copied["gate_closure"] = "GATE_CLOSURE.md"
        manifest["inputs"]["gate_closure"] = {
            "source": "generated_from_training_path_manifest_summary",
            "bundle_path": str(gate_closure_path),
            "status": "generated",
        }
        manifest["summaries"]["gate_closure"] = json_summary(gate_closure_path)

    runbook = render_runbook(manifest, copied)
    (out / "RUNBOOK.md").write_text(runbook)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Wrote handoff bundle: {out}")
    print(f"Open: {out / 'RUNBOOK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
