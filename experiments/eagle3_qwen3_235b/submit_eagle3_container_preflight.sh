#!/usr/bin/env bash
set -euo pipefail

# Dry-run-first Slurm wrapper for proving the selected container can run the
# Qwen3-235B Eagle3 ModelOpt preflight before hidden-state dump/training jobs.
#
# Default mode prints the sbatch command only:
#
#   SBATCH_ACCOUNT=coreai_dlalgo_nemorl \
#   CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh \
#   bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh
#
# To actually submit this preflight-only job, set SUBMIT=true.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  SBATCH_ACCOUNT=coreai_dlalgo_nemorl \\
  CONTAINER=/path/to/container.sqsh \\
  bash experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh

Important env:
  SUBMIT=false|true                         Submit only when true.
  RUN_CLUSTER_PROBE=true|false              Probe paths/Slurm/container before sbatch.
  ARTIFACT_ROOT=/path/qwen3_235b_eagle3     Root for reports and default paths.
  SBATCH_ACCOUNT=<account>                  Slurm account. Required for SUBMIT=true.
  SBATCH_PARTITION=batch                    Slurm partition.
  RESOURCE_PROFILE_ENV=...                  Optional resource profile env.
  PREFLIGHT_GPUS_PER_NODE=...               GPU request for GPU-only partitions.
  CONTAINER=/path/container.sqsh            Pyxis/enroot image to verify.
  MOUNTS=/lustre:/lustre,...                Container mounts.
  VERIFIER_CONFIG_DIR=/path/verifier        Defaults under ARTIFACT_ROOT.
  TOKENIZER_CONFIG=/path/tokenizer_config   Optional; used by template prep outside this wrapper.
  INPUT_DATA=/path/conversations.jsonl      Defaults under ARTIFACT_ROOT.
  CHAT_TEMPLATE=/path/template.jinja2       Defaults under ARTIFACT_ROOT.
  PREFLIGHT_REQUIRE_MODELOPT_IMPORT=true    Require ModelOpt recipe import.
  PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK=true Require Transformers assistant-mask validation.
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

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/outputs/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
RESOURCE_PROFILE_ENV="${RESOURCE_PROFILE_ENV:-$REPORT_DIR/eagle3_resource_profile.env}"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-00:30:00}"
if [[ -z "${PREFLIGHT_GPUS_PER_NODE:-}" && -f "$RESOURCE_PROFILE_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$RESOURCE_PROFILE_ENV"
fi
PREFLIGHT_GPUS_PER_NODE="${PREFLIGHT_GPUS_PER_NODE:-${DUMP_GPUS_PER_NODE:-1}}"
SUBMIT="${SUBMIT:-false}"
RUN_CLUSTER_PROBE="${RUN_CLUSTER_PROBE:-true}"

VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:-$ARTIFACT_ROOT/verifier_config}"
INPUT_DATA="${INPUT_DATA:-$ARTIFACT_ROOT/data/pilot_existing_chat_content_64.jsonl}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ARTIFACT_ROOT/templates/qwen3_generation_template.jinja2}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ARTIFACT_ROOT/hidden_states}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/modelopt_ckpt}"
TRAINED_CKPT="${TRAINED_CKPT:-$OUTPUT_DIR}"
EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_hf}"
VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:-$ARTIFACT_ROOT/vllm_draft}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$ARTIFACT_ROOT/architecture/eagle3_architecture.json}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-$ARTIFACT_ROOT/architecture/eagle3_architecture.env}"

CONTAINER="${CONTAINER:-}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREFLIGHT_REQUIRE_MODELOPT_IMPORT="${PREFLIGHT_REQUIRE_MODELOPT_IMPORT:-true}"
PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK="${PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK:-true}"
PREFLIGHT_SKIP_EXISTING_PATH_CHECKS="${PREFLIGHT_SKIP_EXISTING_PATH_CHECKS:-false}"

CLUSTER_PROBE_JSON="${CLUSTER_PROBE_JSON:-$REPORT_DIR/container_preflight_cluster_probe.json}"
CLUSTER_PROBE_MARKDOWN="${CLUSTER_PROBE_MARKDOWN:-$REPORT_DIR/container_preflight_cluster_probe.md}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-$REPORT_DIR/container_preflight_pipeline.json}"
PREFLIGHT_MARKDOWN="${PREFLIGHT_MARKDOWN:-$REPORT_DIR/container_preflight_pipeline.md}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_eagle3_container_preflight_job.txt}"

mkdir -p "$REPORT_DIR" "$ROOT_DIR/logs"

cat <<EOF
# container preflight env
ARTIFACT_ROOT=$ARTIFACT_ROOT
MODELOPT_DIR=$MODELOPT_DIR
BASE_MODEL=$BASE_MODEL
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
RESOURCE_PROFILE_ENV=$RESOURCE_PROFILE_ENV
PREFLIGHT_GPUS_PER_NODE=$PREFLIGHT_GPUS_PER_NODE
SUBMIT=$SUBMIT
RUN_CLUSTER_PROBE=$RUN_CLUSTER_PROBE
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VERIFIER_CONFIG_DIR=$VERIFIER_CONFIG_DIR
INPUT_DATA=$INPUT_DATA
CHAT_TEMPLATE=$CHAT_TEMPLATE
REFERENCE_ARCH=$REFERENCE_ARCH
PREFLIGHT_JSON=$PREFLIGHT_JSON
PREFLIGHT_MARKDOWN=$PREFLIGHT_MARKDOWN
PREFLIGHT_REQUIRE_MODELOPT_IMPORT=$PREFLIGHT_REQUIRE_MODELOPT_IMPORT
PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK=$PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK
PREFLIGHT_SKIP_EXISTING_PATH_CHECKS=$PREFLIGHT_SKIP_EXISTING_PATH_CHECKS
EOF

if is_true "$SUBMIT"; then
  if [[ "$SBATCH_ACCOUNT" == "dummy" || -z "$SBATCH_ACCOUNT" ]]; then
    echo "SUBMIT=true requires a real SBATCH_ACCOUNT." >&2
    exit 1
  fi
  if [[ -z "$CONTAINER" ]]; then
    echo "SUBMIT=true requires CONTAINER so the preflight proves the target image." >&2
    exit 1
  fi
  if [[ ! -e "$CONTAINER" ]]; then
    echo "CONTAINER is not visible from this host: $CONTAINER" >&2
    exit 1
  fi
fi

if is_true "$RUN_CLUSTER_PROBE"; then
  probe_cmd=(
    python3 "$SCRIPT_DIR/probe_cluster_environment.py"
    --artifact-root "$ARTIFACT_ROOT"
    --modelopt-dir "$MODELOPT_DIR"
    --sbatch-account "$SBATCH_ACCOUNT"
    --sbatch-partition "$SBATCH_PARTITION"
    --verifier-config-dir "$VERIFIER_CONFIG_DIR"
    --input-data "$INPUT_DATA"
    --json-out "$CLUSTER_PROBE_JSON"
    --markdown-out "$CLUSTER_PROBE_MARKDOWN"
  )
  [[ -n "$CONTAINER" ]] && probe_cmd+=(--container "$CONTAINER")
  [[ -n "$MOUNTS" ]] && probe_cmd+=(--mounts "$MOUNTS")
  if is_true "$SUBMIT"; then
    probe_cmd+=(--strict)
  fi
  echo "# cluster probe"
  print_cmd "${probe_cmd[@]}"
  if ! "${probe_cmd[@]}"; then
    echo "Cluster probe returned nonzero; inspect $CLUSTER_PROBE_MARKDOWN" >&2
    if is_true "$SUBMIT"; then
      exit 1
    fi
  fi
fi

export REPO_ROOT="$ROOT_DIR"
export MODELOPT_DIR BASE_MODEL VERIFIER_CONFIG_DIR INPUT_DATA CHAT_TEMPLATE
export HIDDEN_STATES_DIR OUTPUT_DIR TRAINED_CKPT EXPORT_DIR VLLM_DRAFT_DIR
export REFERENCE_ARCH ARCH_ENV_FILE CONTAINER MOUNTS PYTHON_BIN
export PREFLIGHT_GPUS_PER_NODE
export PREFLIGHT_JSON PREFLIGHT_MARKDOWN
export PREFLIGHT_REQUIRE_MODELOPT_IMPORT PREFLIGHT_REQUIRE_CHAT_TEMPLATE_MASK
export PREFLIGHT_SKIP_EXISTING_PATH_CHECKS

preflight_cmd=(
  sbatch --parsable
  --account="$SBATCH_ACCOUNT"
  --partition="$SBATCH_PARTITION"
  --gres="gpu:$PREFLIGHT_GPUS_PER_NODE"
  --time="$SBATCH_TIME"
  --export=ALL
  "$SCRIPT_DIR/slurm_preflight.sbatch"
)

if is_true "$SUBMIT"; then
  job_id="$("${preflight_cmd[@]}")"
  {
    echo "preflight_job=$job_id"
    echo "preflight_gpus_per_node=$PREFLIGHT_GPUS_PER_NODE"
    echo "container=$CONTAINER"
    echo "report=$CLUSTER_PROBE_MARKDOWN"
    echo "preflight_json=$PREFLIGHT_JSON"
    echo "preflight_markdown=$PREFLIGHT_MARKDOWN"
  } | tee "$JOB_FILE"
  echo "Submitted container preflight job: $job_id"
else
  echo "# preflight sbatch"
  print_cmd "${preflight_cmd[@]}"
  {
    echo "preflight_job=PREFLIGHT_JOB_ID"
    echo "preflight_gpus_per_node=$PREFLIGHT_GPUS_PER_NODE"
    echo "container=$CONTAINER"
    echo "report=$CLUSTER_PROBE_MARKDOWN"
    echo "preflight_json=$PREFLIGHT_JSON"
    echo "preflight_markdown=$PREFLIGHT_MARKDOWN"
  } > "$JOB_FILE"
  echo "# dry run only. Set SUBMIT=true to submit the container preflight job."
fi
