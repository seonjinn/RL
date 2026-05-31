#!/usr/bin/env bash
set -euo pipefail

# Reusable probe for the Megatron/NeMo-RL compatibility shims used by the
# Qwen3-235B rollout capture path.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
SPECDEC_RL_DIR="${SPECDEC_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-00:10:00}"
PREFLIGHT_GPUS_PER_NODE="${PREFLIGHT_GPUS_PER_NODE:-4}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$SPECDEC_RL_DIR:$SPECDEC_RL_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR="$SCRIPT_DIR/megatron_bridge_qwen3moe"
MEGATRON_BRIDGE_PLUGIN_DIR="${MEGATRON_BRIDGE_PLUGIN_DIR:-}"
if [[ -z "$MEGATRON_BRIDGE_PLUGIN_DIR" && -d "$DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR" ]]; then
  MEGATRON_BRIDGE_PLUGIN_DIR="$DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR"
fi
MEGATRON_BRIDGE_QWEN3MOE_PLUGIN="${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN:-1}"
MEGATRON_BRIDGE_SRC="${MEGATRON_BRIDGE_SRC:-}"
MEGATRON_LM_SRC="${MEGATRON_LM_SRC:-}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/megatron_compat_probe.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/megatron_compat_probe.md}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_megatron_compat_probe_job.txt}"
REPORT_JOB_FILE="${REPORT_JOB_FILE:-$REPORT_DIR/megatron_compat_probe_job.env}"

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

print_cmd() {
  printf "%q " "$@"
  printf "\n"
}

mkdir -p "$ROOT_DIR/logs"
if ! mkdir -p "$REPORT_DIR" 2>/dev/null; then
  if is_true "$SUBMIT"; then
    echo "Could not create REPORT_DIR=$REPORT_DIR." >&2
    exit 1
  fi
  echo "# dry-run note: report dir is not locally writable: $REPORT_DIR" >&2
fi

export REPO_ROOT="$ROOT_DIR"
export ARTIFACT_ROOT SPECDEC_RL_DIR REPORT_DIR CONTAINER MOUNTS PYTHON_BIN
export MEGATRON_BRIDGE_PLUGIN_DIR MEGATRON_BRIDGE_QWEN3MOE_PLUGIN
export MEGATRON_BRIDGE_SRC MEGATRON_LM_SRC
export JSON_OUT MARKDOWN_OUT

cmd=(
  sbatch --parsable
  --account="$SBATCH_ACCOUNT"
  --partition="$SBATCH_PARTITION"
  --time="$SBATCH_TIME"
  --gres="gpu:$PREFLIGHT_GPUS_PER_NODE"
  --export=ALL
  "$SCRIPT_DIR/slurm_megatron_compat_probe.sbatch"
)

cat <<EOF
# Megatron compatibility probe
ARTIFACT_ROOT=$ARTIFACT_ROOT
SPECDEC_RL_DIR=$SPECDEC_RL_DIR
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
PREFLIGHT_GPUS_PER_NODE=$PREFLIGHT_GPUS_PER_NODE
CONTAINER=$CONTAINER
PYTHON_BIN=$PYTHON_BIN
MEGATRON_BRIDGE_PLUGIN_DIR=${MEGATRON_BRIDGE_PLUGIN_DIR:-<disabled>}
MEGATRON_BRIDGE_QWEN3MOE_PLUGIN=$MEGATRON_BRIDGE_QWEN3MOE_PLUGIN
MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-<container-default>}
MEGATRON_LM_SRC=${MEGATRON_LM_SRC:-<container-default>}
JSON_OUT=$JSON_OUT
MARKDOWN_OUT=$MARKDOWN_OUT
SUBMIT=$SUBMIT
EOF

if is_true "$SUBMIT"; then
  if [[ "$SBATCH_ACCOUNT" == "dummy" || -z "$SBATCH_ACCOUNT" ]]; then
    echo "SUBMIT=true requires a real SBATCH_ACCOUNT." >&2
    exit 1
  fi
  job_id="$("${cmd[@]}")"
  {
    echo "megatron_compat_probe_job=$job_id"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } | tee "$JOB_FILE" "$REPORT_JOB_FILE"
  echo "Submitted Megatron compatibility probe job: $job_id"
else
  echo "# dry-run sbatch"
  print_cmd "${cmd[@]}"
  {
    echo "megatron_compat_probe_job=MEGATRON_COMPAT_PROBE_JOB_ID"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } > "$JOB_FILE"
  if [[ -d "$REPORT_DIR" && -w "$REPORT_DIR" ]]; then
    cp "$JOB_FILE" "$REPORT_JOB_FILE"
  fi
  echo "# dry run only. Set SUBMIT=true to submit."
fi
