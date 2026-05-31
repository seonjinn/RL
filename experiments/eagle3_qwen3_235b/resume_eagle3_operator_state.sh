#!/usr/bin/env bash
set -euo pipefail

# Rebuild the no-submit operator state for the Qwen3-235B Eagle3 workstream.
# This script is safe to run on the remote workspace after SSH recovers. By
# default it only refreshes reports; explicit flags are required to execute
# non-Slurm, non-heavy actions.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  bash experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh

Safe defaults:
  EXECUTE_SAFE_ACTIONS=false
  EXECUTE_SLURM_ACTIONS=false
  RUN_AFTER_SAFE_ACTIONS=false
  REQUIRE_SLURM=true
  RUN_FULL_REFRESH=false

Useful env:
  EXECUTE_SAFE_ACTIONS=true       Execute only allowed non-Slurm/non-heavy actions.
  SAFE_ACTION_IDS="probe_remote_hosts poll_megatron_compat_probe"
                                  Allowlist for safe actions when executing.
  RUN_AFTER_SAFE_ACTIONS=true     Run after_commands for those non-Slurm actions.
  EXECUTE_SLURM_ACTIONS=true      Execute explicitly allowlisted Slurm submit actions.
  SLURM_ACTION_IDS="submit_vllm_source_build submit_source_vllm_abi_probe submit_megatron_compat_probe submit_container_preflight"
                                  Allowlist for Slurm actions when executing.
  RUN_AFTER_SLURM_ACTIONS=false   Keep false unless the Slurm job is already terminal.
  ALLOW_HEAVY_GPU_ACTIONS=false   Keep false for container/runtime gates.
  REQUIRE_SLURM=false             Let ready-submit preflight warn instead of fail on missing Slurm.
  FAIL_ON_WARN=true               Treat validator warnings as failures where supported.
  RUN_FULL_REFRESH=true           After optional safe actions, run the broader no-submit evidence refresh.
  FULL_REFRESH_SKIP_REMOTE_HOST_PROBE=true
                                  Skip the remote-host probe inside the broader refresh.
  FULL_REFRESH_FAIL_ON_ERROR=true Make broader refresh return nonzero on hard failures.
  PROBE_JOB_ID=2867766            Used by the Megatron follow-up action planner.

Remote wrapper example:
  PRINT_ONLY=false \
  REMOTE_HOST=oci-hsg-cs-001-vscode-02 \
  REMOTE_WORKDIR=/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap \
  REMOTE_ARTIFACT_ROOT=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3 \
  REMOTE_ENTRYPOINT=experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh \
  bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh
EOF
  exit 0
fi

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  echo "+ $*"
  "$@"
}

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
PLAN_JSON="${NEXT_ACTION_PLAN_JSON:-$REPORT_DIR/eagle3_next_actions.json}"
PLAN_MARKDOWN="${NEXT_ACTION_PLAN_MARKDOWN:-$REPORT_DIR/eagle3_next_actions.md}"
PLAN_VALIDATION_JSON="${NEXT_ACTION_VALIDATION_JSON:-$REPORT_DIR/eagle3_next_actions_validation.json}"
PLAN_VALIDATION_MARKDOWN="${NEXT_ACTION_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_next_actions_validation.md}"
SHEET_JSON="${OPERATOR_SHEET_JSON:-$REPORT_DIR/eagle3_operator_sheet.json}"
SHEET_MARKDOWN="${OPERATOR_SHEET_MARKDOWN:-$REPORT_DIR/eagle3_operator_sheet.md}"
SHEET_VALIDATION_JSON="${OPERATOR_SHEET_VALIDATION_JSON:-$REPORT_DIR/eagle3_operator_sheet_validation.json}"
SHEET_VALIDATION_MARKDOWN="${OPERATOR_SHEET_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_operator_sheet_validation.md}"
EXECUTION_JSON="${OPERATOR_EXECUTION_JSON:-$REPORT_DIR/eagle3_operator_execution.json}"
EXECUTION_MARKDOWN="${OPERATOR_EXECUTION_MARKDOWN:-$REPORT_DIR/eagle3_operator_execution.md}"
FOLLOWUP_VALIDATION_JSON="${OPERATOR_FOLLOWUP_VALIDATION_JSON:-$REPORT_DIR/eagle3_operator_followups_validation.json}"
FOLLOWUP_VALIDATION_MARKDOWN="${OPERATOR_FOLLOWUP_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_operator_followups_validation.md}"
PACKET_JSON="${OPERATOR_SUBMIT_PACKET_JSON:-$REPORT_DIR/eagle3_operator_submit_packet.json}"
PACKET_MARKDOWN="${OPERATOR_SUBMIT_PACKET_MARKDOWN:-$REPORT_DIR/eagle3_operator_submit_packet.md}"
PACKET_VALIDATION_JSON="${OPERATOR_SUBMIT_PACKET_VALIDATION_JSON:-$REPORT_DIR/eagle3_operator_submit_packet_validation.json}"
PACKET_VALIDATION_MARKDOWN="${OPERATOR_SUBMIT_PACKET_VALIDATION_MARKDOWN:-$REPORT_DIR/eagle3_operator_submit_packet_validation.md}"
READY_PREFLIGHT_JSON="${OPERATOR_READY_SUBMIT_PREFLIGHT_JSON:-$REPORT_DIR/eagle3_operator_ready_submit_preflight.json}"
READY_PREFLIGHT_MARKDOWN="${OPERATOR_READY_SUBMIT_PREFLIGHT_MARKDOWN:-$REPORT_DIR/eagle3_operator_ready_submit_preflight.md}"
QUEUE_JSON="${OPERATOR_QUEUE_JSON:-$REPORT_DIR/eagle3_operator_queue.json}"
QUEUE_MARKDOWN="${OPERATOR_QUEUE_MARKDOWN:-$REPORT_DIR/eagle3_operator_queue.md}"

EXECUTE_SAFE_ACTIONS="${EXECUTE_SAFE_ACTIONS:-false}"
SAFE_ACTION_IDS="${SAFE_ACTION_IDS:-probe_remote_hosts poll_megatron_compat_probe}"
RUN_AFTER_SAFE_ACTIONS="${RUN_AFTER_SAFE_ACTIONS:-false}"
EXECUTE_SLURM_ACTIONS="${EXECUTE_SLURM_ACTIONS:-false}"
SLURM_ACTION_IDS="${SLURM_ACTION_IDS:-submit_vllm_source_build submit_source_vllm_abi_probe submit_megatron_compat_probe submit_container_preflight}"
RUN_AFTER_SLURM_ACTIONS="${RUN_AFTER_SLURM_ACTIONS:-false}"
ALLOW_HEAVY_GPU_ACTIONS="${ALLOW_HEAVY_GPU_ACTIONS:-false}"
REQUIRE_SLURM="${REQUIRE_SLURM:-true}"
FAIL_ON_WARN="${FAIL_ON_WARN:-false}"
RUN_FULL_REFRESH="${RUN_FULL_REFRESH:-false}"
FULL_REFRESH_SKIP_REMOTE_HOST_PROBE="${FULL_REFRESH_SKIP_REMOTE_HOST_PROBE:-false}"
FULL_REFRESH_FAIL_ON_ERROR="${FULL_REFRESH_FAIL_ON_ERROR:-false}"

mkdir -p "$REPORT_DIR" "$REPORT_DIR/operator_execution" "$REPORT_DIR/operator_followups"

run_cmd python3 "$SCRIPT_DIR/plan_eagle3_next_actions.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --json-out "$PLAN_JSON" \
  --markdown-out "$PLAN_MARKDOWN"

run_cmd python3 "$SCRIPT_DIR/validate_eagle3_next_action_plan.py" \
  --plan-json "$PLAN_JSON" \
  --json-out "$PLAN_VALIDATION_JSON" \
  --markdown-out "$PLAN_VALIDATION_MARKDOWN"

run_cmd python3 "$SCRIPT_DIR/create_eagle3_operator_sheet.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --plan-json "$PLAN_JSON" \
  --json-out "$SHEET_JSON" \
  --markdown-out "$SHEET_MARKDOWN"

sheet_validation_cmd=(
  python3 "$SCRIPT_DIR/validate_eagle3_operator_sheet.py"
  --artifact-root "$ARTIFACT_ROOT" \
  --plan-json "$PLAN_JSON" \
  --operator-sheet-json "$SHEET_JSON" \
  --json-out "$SHEET_VALIDATION_JSON" \
  --markdown-out "$SHEET_VALIDATION_MARKDOWN"
)
if is_true "$FAIL_ON_WARN"; then
  sheet_validation_cmd+=(--fail-on-warn)
fi
run_cmd "${sheet_validation_cmd[@]}"

run_cmd python3 "$SCRIPT_DIR/validate_eagle3_operator_execution.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --plan-json "$PLAN_JSON" \
  --operator-sheet-json "$SHEET_JSON" \
  --json-out "$EXECUTION_JSON" \
  --markdown-out "$EXECUTION_MARKDOWN"

run_cmd python3 "$SCRIPT_DIR/validate_eagle3_operator_followups.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --plan-json "$PLAN_JSON" \
  --operator-sheet-json "$SHEET_JSON" \
  --json-out "$FOLLOWUP_VALIDATION_JSON" \
  --markdown-out "$FOLLOWUP_VALIDATION_MARKDOWN"

run_cmd python3 "$SCRIPT_DIR/create_eagle3_operator_submit_packet.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --operator-sheet-json "$SHEET_JSON" \
  --json-out "$PACKET_JSON" \
  --markdown-out "$PACKET_MARKDOWN"

packet_validation_cmd=(
  python3 "$SCRIPT_DIR/validate_eagle3_operator_submit_packet.py"
  --artifact-root "$ARTIFACT_ROOT" \
  --operator-submit-packet-json "$PACKET_JSON" \
  --operator-sheet-json "$SHEET_JSON" \
  --operator-sheet-validation-json "$SHEET_VALIDATION_JSON" \
  --operator-followup-validation-json "$FOLLOWUP_VALIDATION_JSON" \
  --operator-execution-json "$EXECUTION_JSON" \
  --json-out "$PACKET_VALIDATION_JSON" \
  --markdown-out "$PACKET_VALIDATION_MARKDOWN"
)
if is_true "$FAIL_ON_WARN"; then
  packet_validation_cmd+=(--fail-on-warn)
fi
run_cmd "${packet_validation_cmd[@]}"

ready_preflight_cmd=(
  python3 "$SCRIPT_DIR/preflight_eagle3_operator_ready_submit.py"
  --artifact-root "$ARTIFACT_ROOT" \
  --operator-sheet-json "$SHEET_JSON" \
  --operator-submit-packet-validation-json "$PACKET_VALIDATION_JSON" \
  --json-out "$READY_PREFLIGHT_JSON" \
  --markdown-out "$READY_PREFLIGHT_MARKDOWN"
)
ready_preflight_action_ids=()
if is_true "$EXECUTE_SAFE_ACTIONS" || is_true "$EXECUTE_SLURM_ACTIONS"; then
  ready_preflight_filter="$(
    python3 -c '
import json
import sys
from pathlib import Path

sheet = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
execute_safe = sys.argv[2].lower() in {"true", "1", "yes"}
safe_ids = set(sys.argv[3].split())
execute_slurm = sys.argv[4].lower() in {"true", "1", "yes"}
slurm_ids = set(sys.argv[5].split())
allow_heavy = sys.argv[6].lower() in {"true", "1", "yes"}
selected: list[str] = []
for item in sheet.get("ready_actions") or []:
    if not isinstance(item, dict):
        continue
    action_id = str(item.get("id") or "")
    if not action_id:
        continue
    if execute_safe and action_id in safe_ids and not item.get("submits_slurm") and not item.get("heavy_gpu"):
        selected.append(action_id)
        continue
    if execute_slurm and action_id in slurm_ids and item.get("submits_slurm"):
        if not item.get("heavy_gpu") or allow_heavy:
            selected.append(action_id)
for action_id in selected:
    print(action_id)
' "$SHEET_JSON" "$EXECUTE_SAFE_ACTIONS" "$SAFE_ACTION_IDS" "$EXECUTE_SLURM_ACTIONS" "$SLURM_ACTION_IDS" "$ALLOW_HEAVY_GPU_ACTIONS"
  )"
  while IFS= read -r action_id; do
    [[ -n "$action_id" ]] || continue
    ready_preflight_action_ids+=("$action_id")
  done <<< "$ready_preflight_filter"
fi
if [[ "${#ready_preflight_action_ids[@]}" -gt 0 ]]; then
  ready_preflight_cmd+=(--action-ids "${ready_preflight_action_ids[@]}")
fi
if ! is_true "$REQUIRE_SLURM"; then
  ready_preflight_cmd+=(--no-require-slurm)
fi
if is_true "$FAIL_ON_WARN"; then
  ready_preflight_cmd+=(--fail-on-warn)
fi
run_cmd "${ready_preflight_cmd[@]}"

run_cmd python3 "$SCRIPT_DIR/summarize_eagle3_operator_queue.py" \
  --artifact-root "$ARTIFACT_ROOT" \
  --plan-json "$PLAN_JSON" \
  --operator-sheet-json "$SHEET_JSON" \
  --operator-execution-json "$EXECUTION_JSON" \
  --operator-followup-validation-json "$FOLLOWUP_VALIDATION_JSON" \
  --operator-ready-submit-preflight-json "$READY_PREFLIGHT_JSON" \
  --json-out "$QUEUE_JSON" \
  --markdown-out "$QUEUE_MARKDOWN"

if is_true "$EXECUTE_SAFE_ACTIONS"; then
  safe_action_filter="$(
    python3 -c '
import json
import sys
from pathlib import Path

sheet = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed = set(sys.argv[2].split())
for item in sheet.get("ready_actions") or []:
    if not isinstance(item, dict):
        continue
    action_id = str(item.get("id") or "")
    if action_id not in allowed:
        continue
    if item.get("submits_slurm") or item.get("heavy_gpu"):
        continue
    print(action_id)
' "$SHEET_JSON" "$SAFE_ACTION_IDS"
  )"
  while IFS= read -r action_id; do
    [[ -n "$action_id" ]] || continue
    record="$REPORT_DIR/operator_execution/auto_${action_id}.json"
    action_cmd=(
      python3 "$SCRIPT_DIR/run_eagle3_next_action.py"
      --artifact-root "$ARTIFACT_ROOT"
      --plan-json "$PLAN_JSON"
      --action-id "$action_id"
      --execute
      --json-out "$record"
    )
    if is_true "$RUN_AFTER_SAFE_ACTIONS"; then
      action_cmd+=(--run-after)
    fi
    run_cmd "${action_cmd[@]}"
  done <<< "$safe_action_filter"
fi

if is_true "$EXECUTE_SLURM_ACTIONS"; then
  slurm_action_filter="$(
    python3 -c '
import json
import sys
from pathlib import Path

sheet = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed = set(sys.argv[2].split())
allow_heavy = sys.argv[3].lower() in {"true", "1", "yes"}
for item in sheet.get("ready_actions") or []:
    if not isinstance(item, dict):
        continue
    action_id = str(item.get("id") or "")
    if action_id not in allowed:
        continue
    if not item.get("submits_slurm"):
        continue
    if item.get("heavy_gpu") and not allow_heavy:
        continue
    print(action_id)
' "$SHEET_JSON" "$SLURM_ACTION_IDS" "$ALLOW_HEAVY_GPU_ACTIONS"
  )"
  while IFS= read -r action_id; do
    [[ -n "$action_id" ]] || continue
    record="$REPORT_DIR/operator_execution/auto_${action_id}.json"
    action_cmd=(
      python3 "$SCRIPT_DIR/run_eagle3_next_action.py"
      --artifact-root "$ARTIFACT_ROOT"
      --plan-json "$PLAN_JSON"
      --action-id "$action_id"
      --execute
      --allow-slurm
      --json-out "$record"
    )
    if is_true "$ALLOW_HEAVY_GPU_ACTIONS"; then
      action_cmd+=(--allow-heavy-gpu)
    fi
    if is_true "$RUN_AFTER_SLURM_ACTIONS"; then
      action_cmd+=(--run-after --allow-run-after-for-slurm)
    fi
    run_cmd "${action_cmd[@]}"
  done <<< "$slurm_action_filter"
fi

if is_true "$RUN_FULL_REFRESH"; then
  full_refresh_cmd=(
    python3 "$SCRIPT_DIR/refresh_eagle3_operator_state.py"
    --artifact-root "$ARTIFACT_ROOT"
    --json-out "$REPORT_DIR/eagle3_operator_state_refresh.json"
    --markdown-out "$REPORT_DIR/eagle3_operator_state_refresh.md"
  )
  if is_true "$FULL_REFRESH_SKIP_REMOTE_HOST_PROBE"; then
    full_refresh_cmd+=(--skip-remote-host-probe)
  fi
  if is_true "$FULL_REFRESH_FAIL_ON_ERROR"; then
    full_refresh_cmd+=(--fail-on-error)
  fi
  run_cmd "${full_refresh_cmd[@]}"
fi

echo
echo "Operator queue: $QUEUE_MARKDOWN"
echo "Ready-submit preflight: $READY_PREFLIGHT_MARKDOWN"
