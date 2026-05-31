#!/usr/bin/env bash
set -euo pipefail

# Submit a Ray-backed direct vLLM math generation smoke for Eagle3 corpus
# creation. This bypasses NeMo-RL policy checkpoint import/save.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_direct_vllm_math_rollout.sh}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap}}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
DATA_DIR="${DATA_DIR:-$ARTIFACT_ROOT/data}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${NUM_NODES:-2}"
JOB_NAME="${JOB_NAME:-q235b-math-direct-vllm-smoke}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
DRY_RUN="${DRY_RUN:-true}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:---mem=0}"
SBATCH_EXTRA_ARGV=()
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  SBATCH_EXTRA_ARGV=($SBATCH_EXTRA_ARGS)
fi

SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
PROMPT_DATA="${PROMPT_DATA:-$DATA_DIR/openmath_direct_vllm_prompts_smoke.jsonl}"
OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$DATA_DIR/qwen3_235b_math_direct_vllm_conversations_smoke.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$REPORT_DIR/direct_vllm_math_rollout_summary.json}"
SERVER_LOG="${SERVER_LOG:-$REPORT_DIR/direct_vllm_math_rollout_server_${JOB_NAME}.log}"
GENERATION_LOG="${GENERATION_LOG:-$REPORT_DIR/direct_vllm_math_rollout_generation_${JOB_NAME}.log}"

mkdir -p "$REPORT_DIR" "$DATA_DIR"

MOUNTS="${MOUNTS:-/lustre:/lustre,$REMOTE_REPO_ROOT:$REMOTE_REPO_ROOT,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
SBATCH_PYTHONPATH="$SOURCE_VLLM_SITE:$REMOTE_REPO_ROOT:${PYTHONPATH:-}"
HF_HOME_PATH="${HF_HOME_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-64}"
SETUP_COMMAND="${SETUP_COMMAND:-$(cat <<EOF
python - <<'PY'
from pathlib import Path
import site

paths = [
    "$SOURCE_VLLM_SITE",
    "$REMOTE_REPO_ROOT",
]
line = "import sys; [sys.path.insert(0, p) for p in reversed(%r) if p not in sys.path]\\n" % paths
for site_dir in site.getsitepackages():
    site_path = Path(site_dir)
    if not site_path.exists():
        continue
    p = site_path / "qwen3_eagle3_source_paths.pth"
    p.write_text(line, encoding="utf-8")
    print(f"wrote {p}")
PY
python - <<'PY'
import sys
import huggingface_hub
import transformers
import vllm
print("setup huggingface_hub", huggingface_hub.__version__, huggingface_hub.__file__)
print("setup transformers", transformers.__version__, transformers.__file__)
print("setup vllm", vllm.__version__, vllm.__file__)
print("setup sys.path head", sys.path[:5])
PY
EOF
)}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd "$REMOTE_REPO_ROOT"
ARTIFACT_ROOT="$ARTIFACT_ROOT" \
REPORT_DIR="$REPORT_DIR" \
DATA_DIR="$DATA_DIR" \
SOURCE_VLLM_SITE="$SOURCE_VLLM_SITE" \
MODEL_PATH="$MODEL_PATH" \
PROMPT_DATA="$PROMPT_DATA" \
OUTPUT_CONVERSATIONS="$OUTPUT_CONVERSATIONS" \
SUMMARY_JSON="$SUMMARY_JSON" \
SERVER_LOG="$SERVER_LOG" \
GENERATION_LOG="$GENERATION_LOG" \
PROMPT_LIMIT="${PROMPT_LIMIT:-4}" \
PROMPT_OFFSET="${PROMPT_OFFSET:-0}" \
SKIP_PROMPT_MATERIALIZE="${SKIP_PROMPT_MATERIALIZE:-false}" \
LIMIT="${LIMIT:-}" \
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}" \
APPEND="${APPEND:-false}" \
ID_KEY="${ID_KEY:-}" \
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-modelopt}" \
NUM_RESPONSES="${NUM_RESPONSES:-1}" \
GENERATION_CONCURRENCY="${GENERATION_CONCURRENCY:-1}" \
GENERATION_SKIP_FAILED="${GENERATION_SKIP_FAILED:-false}" \
TEMPERATURE="${TEMPERATURE:-1.0}" \
TOP_P="${TOP_P:-1.0}" \
MAX_TOKENS="${MAX_TOKENS:-2048}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
VLLM_TP="${VLLM_TP:-8}" \
VLLM_PP="${VLLM_PP:-1}" \
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-ray}" \
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.82}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}" \
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}" \
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-true}" \
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}" \
VLLM_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG:-}" \
VLLM_SPECULATIVE_CONFIG_FILE="${VLLM_SPECULATIVE_CONFIG_FILE:-}" \
MODEL_LABEL="${MODEL_LABEL:-}" \
CHUNK_INDEX="${CHUNK_INDEX:-}" \
CHUNK_SIZE="${CHUNK_SIZE:-}" \
WAVE_LIMIT="${WAVE_LIMIT:-}" \
BASE_OFFSET="${BASE_OFFSET:-}" \
CHUNK_DIR="${CHUNK_DIR:-}" \
REPAIR_CHUNKS="${REPAIR_CHUNKS:-}" \
REPAIR_TAG="${REPAIR_TAG:-}" \
MISSING_PROMPTS="${MISSING_PROMPTS:-}" \
GENERATED_REPAIR_OUTPUT="${GENERATED_REPAIR_OUTPUT:-}" \
  bash "$RUN_SCRIPT"
EOF
)

echo "# direct vLLM math rollout smoke"
echo "REMOTE_REPO_ROOT=$REMOTE_REPO_ROOT"
echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
echo "SOURCE_VLLM_SITE=$SOURCE_VLLM_SITE"
echo "RUN_SCRIPT=$RUN_SCRIPT"
echo "OUTPUT_CONVERSATIONS=$OUTPUT_CONVERSATIONS"
echo "SUMMARY_JSON=$SUMMARY_JSON"
echo "NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE CPUS_PER_WORKER=$CPUS_PER_WORKER VLLM_TP=${VLLM_TP:-8} VLLM_DISTRIBUTED_EXECUTOR_BACKEND=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
echo "SBATCH_EXTRA_ARGS=$SBATCH_EXTRA_ARGS"
echo "DRY_RUN=$DRY_RUN"

sbatch_cmd=(
  sbatch
  --nodes="$NUM_NODES"
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --gres="gpu:$GPUS_PER_NODE"
  --time="$TIME_LIMIT"
  --job-name="$JOB_NAME"
)
if ((${#SBATCH_EXTRA_ARGV[@]})); then
  sbatch_cmd+=("${SBATCH_EXTRA_ARGV[@]}")
fi
sbatch_cmd+=("$REMOTE_REPO_ROOT/scripts/share/ray.sub")

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  printf '%q ' env \
    COMMAND="$COMMAND" \
    CONTAINER="$CONTAINER" \
    MOUNTS="$MOUNTS" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    CPUS_PER_WORKER="$CPUS_PER_WORKER" \
    SETUP_COMMAND="$SETUP_COMMAND" \
    PYTHONPATH="$SBATCH_PYTHONPATH" \
    HF_HOME="$HF_HOME_PATH" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_PATH/datasets}" \
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_PATH/hub}" \
    VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ARTIFACT_ROOT/vllm_cache}" \
    VLLM_DISABLE_USAGE_STATS=1 \
    VLLM_USE_V1="${VLLM_USE_V1:-0}" \
    VLLM_USE_RAY_COMPILED_DAG="${VLLM_USE_RAY_COMPILED_DAG:-0}" \
    VLLM_USE_RAY_SPMD_WORKER="${VLLM_USE_RAY_SPMD_WORKER:-0}" \
    VLLM_USE_RAY_WRAPPED_PP_COMM="${VLLM_USE_RAY_WRAPPED_PP_COMM:-0}" \
    RAY_DEDUP_LOGS=0 \
    "${sbatch_cmd[@]}"
  printf '\n'
  exit 0
fi

submit_out=$(
  COMMAND="$COMMAND" \
  CONTAINER="$CONTAINER" \
  MOUNTS="$MOUNTS" \
  GPUS_PER_NODE="$GPUS_PER_NODE" \
  CPUS_PER_WORKER="$CPUS_PER_WORKER" \
  SETUP_COMMAND="$SETUP_COMMAND" \
  PYTHONPATH="$SBATCH_PYTHONPATH" \
  HF_HOME="$HF_HOME_PATH" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_PATH/datasets}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_PATH/hub}" \
  VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ARTIFACT_ROOT/vllm_cache}" \
  VLLM_DISABLE_USAGE_STATS=1 \
  VLLM_USE_V1="${VLLM_USE_V1:-0}" \
  VLLM_USE_RAY_COMPILED_DAG="${VLLM_USE_RAY_COMPILED_DAG:-0}" \
  VLLM_USE_RAY_SPMD_WORKER="${VLLM_USE_RAY_SPMD_WORKER:-0}" \
  VLLM_USE_RAY_WRAPPED_PP_COMM="${VLLM_USE_RAY_WRAPPED_PP_COMM:-0}" \
  RAY_DEDUP_LOGS=0 \
  "${sbatch_cmd[@]}"
)
echo "$submit_out"
job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out" | tail -1)"
if [[ -n "$job_id" ]]; then
  echo "$job_id" > "$REPORT_DIR/latest_direct_vllm_math_rollout_job.txt"
  echo "job_id=$job_id"
fi
