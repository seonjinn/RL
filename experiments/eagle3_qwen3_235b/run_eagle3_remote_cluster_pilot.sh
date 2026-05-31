#!/usr/bin/env bash
set -euo pipefail

# SSH wrapper for running the Qwen3-235B Eagle3 cluster pilot entrypoint on a
# remote host. Safe defaults: no Slurm submission and no local file sync.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  REMOTE_HOST=oci-hsg-cs-001-vscode-02 \\
  REMOTE_WORKDIR=/lustre/.../Nemo-RL_Qwen3_Roadmap \\
  SBATCH_ACCOUNT=<account> \\
  bash experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh

Safe defaults:
  SUBMIT=false
  RUN_PILOT=true
  PREP_DRY_RUN=true
  SYNC_EXPERIMENTS=false

Useful env:
  PRINT_ONLY=true                 Print ssh/rsync commands without executing.
  SYNC_EXPERIMENTS=true           Rsync only experiments/eagle3_qwen3_235b.
  SYNC_PROBE_JOB_FILE=true        Rsync latest_megatron_compat_probe_job.txt.
  REMOTE_ENTRYPOINT=path.sh       Override the remote script to execute.
  REMOTE_ARTIFACT_ROOT=/lustre/... Override remote ARTIFACT_ROOT.
  SSH_PROXY_JUMP=host             Add ssh -J for environments that need a jump host.
  SSH_EXTRA_OPTS="-o Key=Value"   Append simple whitespace-split ssh options.
  DISCOVERY_ROOTS="path path ..." Override cluster discovery roots.
  RUN_STATIC_INPUT_PREP=auto|true|false
                                  Preserve static Qwen3 verifier input prep mode.
  STATIC_INPUT_SOURCE_DIR=/path   Optional local HF snapshot/config source on remote.
  VERIFIER_CONFIG_DIR=/path       Preserve explicit verifier config.
  INPUT_DATA=/path                Preserve explicit conversation JSONL.
  CONTAINER=/path.sqsh            Pass container image to Slurm srun.
  MOUNTS=/lustre:/lustre,...      Pass Pyxis container mounts.
  EXECUTE_SLURM_ACTIONS=true      Remote resume only: execute allowlisted Slurm actions.
  SLURM_ACTION_IDS="submit_vllm_source_build submit_source_vllm_abi_probe submit_container_preflight"
                                  Remote resume Slurm action allowlist.
EOF
  exit 0
fi

quote() {
  printf "%q" "$1"
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

REMOTE_HOST="${REMOTE_HOST:-oci-hsg-cs-001-vscode-02}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:?set REMOTE_WORKDIR to the repo path on the remote host}"
REMOTE_ARTIFACT_ROOT="${REMOTE_ARTIFACT_ROOT:-$REMOTE_WORKDIR/outputs/qwen3_235b_eagle3}"
PRINT_ONLY="${PRINT_ONLY:-false}"
SYNC_EXPERIMENTS="${SYNC_EXPERIMENTS:-false}"
SYNC_PROBE_JOB_FILE="${SYNC_PROBE_JOB_FILE:-false}"
REMOTE_ENTRYPOINT="${REMOTE_ENTRYPOINT:-experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh}"

SSH_OPTS=(
  -S none
  -o ControlMaster=no
  -o BatchMode=yes
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT:-10}"
)
if [[ -n "${SSH_PROXY_JUMP:-}" ]]; then
  SSH_OPTS+=(-J "$SSH_PROXY_JUMP")
fi
if [[ -n "${SSH_EXTRA_OPTS:-}" ]]; then
  # Intentionally simple: use repeated key/value tokens such as "-o Key=Value".
  read -r -a ssh_extra_opts <<< "$SSH_EXTRA_OPTS"
  SSH_OPTS+=("${ssh_extra_opts[@]}")
fi

RSYNC_BIN="${RSYNC_BIN:-rsync}"

remote_env_keys=(
  ARTIFACT_ROOT
  SBATCH_ACCOUNT
  SBATCH_PARTITION
  SUBMIT
  RUN_PILOT
  PREP_DRY_RUN
  REVISION
  RUN_STATIC_INPUT_PREP
  STATIC_INPUT_SOURCE_DIR
  STATIC_INPUT_FORCE
  STATIC_INPUT_SKIP_TEMPLATE_VALIDATION
  STATIC_INPUT_MODEL_OR_TOKENIZER
  RUN_INPUT_DISCOVERY
  SOURCE_DISCOVERY_ENV
  RUN_CLUSTER_PROBE
  RUN_HAYATE_INVENTORY
  RUN_DRAFT_INVENTORY
  RUN_UPSTREAM_DRIFT
  PROBE_UPSTREAM
  RUN_MODELOPT_PATCH
  RUN_PROVENANCE
  RUN_BOOTSTRAP
  RUN_HANDOFF
  DISCOVERY_ROOTS
  MODELOPT_DIR
  COMPAT_MODELOPT_DIR
  HAYATE_MODEL_OPT_DIR
  HAYATE_NEMO_RL_DIR
  HAYATE_DRAFT_MODELS_DIR
  VERIFIER_CONFIG_DIR
  TOKENIZER_CONFIG
  MODE
  DATA_MODE
  INPUT_PATHS
  INPUT_DATA
  CHAT_TEMPLATE
  HIDDEN_STATES_DIR
  OUTPUT_DIR
  EXPORT_DIR
  VLLM_DRAFT_DIR
  CONTAINER
  MOUNTS
  PROBE_JOB_ID
  JOB_FILE
  REPORT_JOB_FILE
  JSON_OUT
  SUBMIT_ROLLOUT
  ALLOW_HEAVY_GPU
  FAIL_ON_NOT_READY
  EXECUTE_SAFE_ACTIONS
  SAFE_ACTION_IDS
  RUN_AFTER_SAFE_ACTIONS
  EXECUTE_SLURM_ACTIONS
  SLURM_ACTION_IDS
  RUN_AFTER_SLURM_ACTIONS
  ALLOW_HEAVY_GPU_ACTIONS
  REQUIRE_SLURM
  FAIL_ON_WARN
  RUN_FULL_REFRESH
  FULL_REFRESH_SKIP_REMOTE_HOST_PROBE
  FULL_REFRESH_FAIL_ON_ERROR
  START_WATCHER
  REQUIRE_SOURCE_BUILD_PASS
  NUM_GPU
  NUM_NODES
  NUM_GEN_NODES
  TP
  ETP
  EP
  CP
  PP
  VLLM_TP
  PP_FIRST_STAGE
  PP_LAST_STAGE
  SBATCH_EXCLUDE
  WANDB_NAME
  ROLLOUT_LOG_DIR
  OUTPUT_CONVERSATIONS
  ROLLOUT_REPORT_PREFIX_TAG
)

export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REMOTE_ARTIFACT_ROOT}"
export SUBMIT="${SUBMIT:-false}"
export RUN_PILOT="${RUN_PILOT:-true}"
export PREP_DRY_RUN="${PREP_DRY_RUN:-true}"

if [[ "${SBATCH_ACCOUNT:-}" == "" ]]; then
  echo "WARN: SBATCH_ACCOUNT is not set; remote cluster probe will fail until it is provided." >&2
fi

remote_env=()
for key in "${remote_env_keys[@]}"; do
  if [[ -n "${!key+x}" ]]; then
    remote_env+=("$key=${!key}")
  fi
done

if is_true "$SYNC_EXPERIMENTS"; then
  rsync_cmd=(
    "$RSYNC_BIN"
    -a
    --delete
    "$EXP_DIR/"
    "$REMOTE_HOST:$REMOTE_WORKDIR/experiments/eagle3_qwen3_235b/"
  )
  echo "# sync experiments"
  print_cmd "${rsync_cmd[@]}"
  if ! is_true "$PRINT_ONLY"; then
    "${rsync_cmd[@]}"
  fi
fi

if is_true "$SYNC_PROBE_JOB_FILE"; then
  probe_job_file="$ROOT_DIR/latest_megatron_compat_probe_job.txt"
  rsync_probe_cmd=(
    "$RSYNC_BIN"
    -a
    "$probe_job_file"
    "$REMOTE_HOST:$REMOTE_WORKDIR/latest_megatron_compat_probe_job.txt"
  )
  echo "# sync Megatron compatibility probe job file"
  print_cmd "${rsync_probe_cmd[@]}"
  if ! is_true "$PRINT_ONLY"; then
    "${rsync_probe_cmd[@]}"
  fi
fi

env_prefix=()
for item in "${remote_env[@]}"; do
  env_prefix+=("$(quote "$item")")
done

remote_script="cd $(quote "$REMOTE_WORKDIR") && env ${env_prefix[*]} bash $(quote "$REMOTE_ENTRYPOINT")"
ssh_cmd=(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$remote_script")

echo "# remote cluster pilot"
print_cmd "${ssh_cmd[@]}"
if ! is_true "$PRINT_ONLY"; then
  "${ssh_cmd[@]}"
fi
