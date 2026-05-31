#!/usr/bin/env bash
set -euo pipefail

# Probe the source-built vLLM site beyond native ABI: import AsyncLLM,
# Qwen3-MoE model code, and create a minimal engine config for Qwen3-235B.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-00:30:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
VLLM_SITE="${VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
HF_HOME_PATH="${HF_HOME_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_runtime_probe.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_runtime_probe.md}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_vllm_runtime_probe_job.txt}"

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

mkdir -p "$REPORT_DIR" "$ROOT_DIR/logs"

export REPO_ROOT="$ROOT_DIR"
export ARTIFACT_ROOT REPORT_DIR CONTAINER MOUNTS PYTHON_BIN GPUS_PER_NODE
export VLLM_SITE MODEL_NAME HF_HOME_PATH JSON_OUT MARKDOWN_OUT

cmd=(
  sbatch --parsable
  --account="$SBATCH_ACCOUNT"
  --partition="$SBATCH_PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$SBATCH_TIME"
  --export=ALL
  "$SCRIPT_DIR/slurm_vllm_runtime_probe.sbatch"
)

cat <<EOF
# vLLM runtime probe
ARTIFACT_ROOT=$ARTIFACT_ROOT
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
CONTAINER=$CONTAINER
PYTHON_BIN=$PYTHON_BIN
VLLM_SITE=$VLLM_SITE
MODEL_NAME=$MODEL_NAME
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
    echo "vllm_runtime_probe_job=$job_id"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } | tee "$JOB_FILE"
  echo "Submitted vLLM runtime probe job: $job_id"
else
  echo "# dry-run sbatch"
  print_cmd "${cmd[@]}"
  {
    echo "vllm_runtime_probe_job=VLLM_RUNTIME_PROBE_JOB_ID"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } > "$JOB_FILE"
  echo "# dry run only. Set SUBMIT=true to submit."
fi
