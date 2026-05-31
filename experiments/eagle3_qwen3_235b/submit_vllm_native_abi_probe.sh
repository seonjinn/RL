#!/usr/bin/env bash
set -euo pipefail

# Submit a short container job that verifies vLLM native extension ABI
# compatibility against the runtime torch in /opt/venv. A plain `import vllm`
# is not enough; Qwen3 rollout workers import vllm._C through vllm.config.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-00:20:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_vllm_native_abi_probe_job.txt}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_native_abi_probe.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_native_abi_probe.md}"
VLLM_SITE_CANDIDATES="${VLLM_SITE_CANDIDATES:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_nodeps_py312 $ARTIFACT_ROOT/python_site/vllm_0_11_2_nodeps_py312 $ARTIFACT_ROOT/python_site/vllm_0_13_0_nodeps_py312}"

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
export VLLM_SITE_CANDIDATES JSON_OUT MARKDOWN_OUT

cmd=(
  sbatch --parsable
  --account="$SBATCH_ACCOUNT"
  --partition="$SBATCH_PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$SBATCH_TIME"
  --export=ALL
  "$SCRIPT_DIR/slurm_vllm_native_abi_probe.sbatch"
)

cat <<EOF
# vLLM native ABI probe
ARTIFACT_ROOT=$ARTIFACT_ROOT
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
CONTAINER=$CONTAINER
PYTHON_BIN=$PYTHON_BIN
JSON_OUT=$JSON_OUT
MARKDOWN_OUT=$MARKDOWN_OUT
VLLM_SITE_CANDIDATES=$VLLM_SITE_CANDIDATES
SUBMIT=$SUBMIT
EOF

if is_true "$SUBMIT"; then
  if [[ "$SBATCH_ACCOUNT" == "dummy" || -z "$SBATCH_ACCOUNT" ]]; then
    echo "SUBMIT=true requires a real SBATCH_ACCOUNT." >&2
    exit 1
  fi
  job_id="$("${cmd[@]}")"
  {
    echo "vllm_native_abi_probe_job=$job_id"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } | tee "$JOB_FILE"
  echo "Submitted vLLM native ABI probe job: $job_id"
else
  echo "# dry-run sbatch"
  print_cmd "${cmd[@]}"
  {
    echo "vllm_native_abi_probe_job=VLLM_ABI_PROBE_JOB_ID"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } > "$JOB_FILE"
  echo "# dry run only. Set SUBMIT=true to submit."
fi
