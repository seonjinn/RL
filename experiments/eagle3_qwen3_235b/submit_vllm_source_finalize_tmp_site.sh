#!/usr/bin/env bash
set -euo pipefail

# Reuse a source-build tmp site that already contains a built vLLM wheel but
# failed a late import probe because a pure Python runtime dependency was
# missing. This avoids rebuilding vLLM from scratch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-dummy}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-01:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$ROOT_DIR:$ROOT_DIR,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
SOURCE_TMP_SITE="${SOURCE_TMP_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312.tmp.2855535}"
OUTPUT_SITE="${OUTPUT_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
VLLM_SOURCE_SPEC="${VLLM_SOURCE_SPEC:-https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/vllm-0.10.2.tar.gz}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_native_source_build.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_native_source_build.md}"
FINALIZE_DEPS="${FINALIZE_DEPS:-pybase64}"
IN_PLACE="${IN_PLACE:-false}"
SUBMIT="${SUBMIT:-false}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_vllm_source_finalize_job.txt}"

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
export SOURCE_TMP_SITE OUTPUT_SITE VLLM_SOURCE_SPEC JSON_OUT MARKDOWN_OUT FINALIZE_DEPS IN_PLACE

cmd=(
  sbatch --parsable
  --account="$SBATCH_ACCOUNT"
  --partition="$SBATCH_PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$SBATCH_TIME"
  --export=ALL
  "$SCRIPT_DIR/slurm_finalize_vllm_source_site.sbatch"
)

cat <<EOF
# vLLM source tmp-site finalize
ARTIFACT_ROOT=$ARTIFACT_ROOT
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_PARTITION=$SBATCH_PARTITION
CONTAINER=$CONTAINER
PYTHON_BIN=$PYTHON_BIN
SOURCE_TMP_SITE=$SOURCE_TMP_SITE
OUTPUT_SITE=$OUTPUT_SITE
JSON_OUT=$JSON_OUT
MARKDOWN_OUT=$MARKDOWN_OUT
FINALIZE_DEPS=$FINALIZE_DEPS
IN_PLACE=$IN_PLACE
SUBMIT=$SUBMIT
EOF

if is_true "$SUBMIT"; then
  if [[ "$SBATCH_ACCOUNT" == "dummy" || -z "$SBATCH_ACCOUNT" ]]; then
    echo "SUBMIT=true requires a real SBATCH_ACCOUNT." >&2
    exit 1
  fi
  job_id="$("${cmd[@]}")"
  {
    echo "vllm_source_finalize_job=$job_id"
    echo "source_tmp_site=$SOURCE_TMP_SITE"
    echo "output_site=$OUTPUT_SITE"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
    echo "finalize_deps=$FINALIZE_DEPS"
    echo "in_place=$IN_PLACE"
  } | tee "$JOB_FILE"
  echo "Submitted vLLM source finalize job: $job_id"
else
  echo "# dry-run sbatch"
  print_cmd "${cmd[@]}"
  {
    echo "vllm_source_finalize_job=VLLM_SOURCE_FINALIZE_JOB_ID"
    echo "source_tmp_site=$SOURCE_TMP_SITE"
    echo "output_site=$OUTPUT_SITE"
    echo "json=$JSON_OUT"
    echo "markdown=$MARKDOWN_OUT"
    echo "finalize_deps=$FINALIZE_DEPS"
    echo "in_place=$IN_PLACE"
  } > "$JOB_FILE"
  echo "# dry run only. Set SUBMIT=true to submit."
fi
