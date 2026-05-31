#!/usr/bin/env bash
set -euo pipefail

# Build vLLM from source inside the target NeMo container so vllm._C links
# against the container's torch build. Use this after wheel ABI probes fail.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
VLLM_SOURCE_SPEC="${VLLM_SOURCE_SPEC:-https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/vllm-0.10.2.tar.gz}"
OUTPUT_SITE="${OUTPUT_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_native_source_build.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_native_source_build.md}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_vllm_native_source_build_job.txt}"

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
export VLLM_SOURCE_SPEC OUTPUT_SITE JSON_OUT MARKDOWN_OUT

cmd=(
  sbatch --parsable
  --account="$SBATCH_ACCOUNT"
  --partition="$SBATCH_PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$SBATCH_TIME"
  --export=ALL
  "$SCRIPT_DIR/slurm_build_vllm_native_site.sbatch"
)

cat <<EOF
# vLLM native source build
ARTIFACT_ROOT=$ARTIFACT_ROOT
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
CONTAINER=$CONTAINER
PYTHON_BIN=$PYTHON_BIN
VLLM_SOURCE_SPEC=$VLLM_SOURCE_SPEC
OUTPUT_SITE=$OUTPUT_SITE
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
    echo "vllm_native_source_build_job=$job_id"
    echo "output_site=$OUTPUT_SITE"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } | tee "$JOB_FILE"
  echo "Submitted vLLM native source build job: $job_id"
else
  echo "# dry-run sbatch"
  print_cmd "${cmd[@]}"
  {
    echo "vllm_native_source_build_job=VLLM_SOURCE_BUILD_JOB_ID"
    echo "output_site=$OUTPUT_SITE"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
  } > "$JOB_FILE"
  echo "# dry run only. Set SUBMIT=true to submit."
fi
