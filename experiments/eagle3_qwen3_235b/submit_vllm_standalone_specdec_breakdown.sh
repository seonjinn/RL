#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

REMOTE_HOST="${REMOTE_HOST:-oci-hsg-cs-001-vscode-02}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/guyueh/rl_projects/vllm/vllm-runs/vllm-hsg-nightly-nsys.sqsh}"
JOB_CACHE_DIR="${JOB_CACHE_DIR:-${REMOTE_REPO}/.container_cache}"
LOG_ROOT="${LOG_ROOT:-${REMOTE_REPO}/vllm-runs}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_30ba3b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers48_mlen8193/0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-1}"
TP="${TP:-1}"
PP="${PP:-1}"
GPUS="${GPUS:-4}"
ISL="${ISL:-1000}"
OSL="${OSL:-1000}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
TAG="${TAG:-qwen30ba3b_standalone_specdec_breakdown_k${NUM_SPECULATIVE_TOKENS}_$(date +%Y%m%d_%H%M%S)}"
LOGS_DIR="${LOGS_DIR:-${LOG_ROOT}/${TAG}}"
JOB_FILE="${JOB_FILE:-${ROOT_DIR}/latest_vllm_standalone_specdec_breakdown_jobs.txt}"
EAGER_ARG=""
case "${ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES|y|Y|on|ON) EAGER_ARG="--enforce-eager" ;;
esac

SPECULATIVE_CONFIG="$(python3 - <<PY
import json
print(json.dumps({
  "method": "eagle3",
  "model": "${DRAFT_MODEL}",
  "num_speculative_tokens": int("${NUM_SPECULATIVE_TOKENS}"),
  "draft_tensor_parallel_size": 1,
}))
PY
)"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/specdec_breakdown_instrumentation' '$LOGS_DIR'"
scp -q "$SCRIPT_DIR/standalone_vllm_specdec_breakdown.py" "$REMOTE_HOST:$REMOTE_REPO/standalone_vllm_specdec_breakdown.py"
scp -q "$SCRIPT_DIR/specdec_breakdown_instrumentation/sitecustomize.py" "$REMOTE_HOST:$REMOTE_REPO/specdec_breakdown_instrumentation/sitecustomize.py"

remote_cmd=$(cat <<EOF
cd '$REMOTE_REPO'
mkdir -p vllm-runs '$LOGS_DIR' '$JOB_CACHE_DIR'
cat > '${LOGS_DIR}/speculative_config.json' <<'SPECJSON'
${SPECULATIVE_CONFIG}
SPECJSON
sbatch --parsable \\
  --partition='$PARTITION' \\
  --account='$ACCOUNT' \\
  --job-name='vllm-specdec-breakdown-q30-k${NUM_SPECULATIVE_TOKENS}' \\
  --nodes=1 \\
  --gres=gpu:${GPUS} \\
  --ntasks-per-node=1 \\
  --cpus-per-task=64 \\
  --mem=0 \\
  --time='$TIME_LIMIT' \\
  --output='${LOGS_DIR}/slurm-%j.out' \\
  <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
export HF_DATASETS_CACHE=\$HF_HOME/cache
export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
export VLLM_USE_V1=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPECDEC_BREAKDOWN_INSTRUMENTATION=1
export SPECDEC_BREAKDOWN_WRAP_TARGET=1

srun --nodes=1 --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre,${JOB_CACHE_DIR}:/root/.cache' \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail; \\
    cd '${REMOTE_REPO}'; \\
    python3 -m pip install -q --no-cache-dir --target '${LOGS_DIR}/pydeps' 'huggingface-hub>=0.34.0,<1.0'; \\
    export PYTHONPATH='${LOGS_DIR}/pydeps:${REMOTE_REPO}/specdec_breakdown_instrumentation':\\\${PYTHONPATH:-}; \\
    python3 standalone_vllm_specdec_breakdown.py \\
    --model '${MODEL}' \\
    --speculative-config '@${LOGS_DIR}/speculative_config.json' \\
    ${EAGER_ARG} \\
    --tp ${TP} --pp ${PP} \\
    --distributed-executor-backend none \\
    --attention-backend TRITON_ATTN \\
    --gpu-memory-utilization 0.82 \\
    --isl ${ISL} --osl ${OSL} \\
    --batch-sizes ${BATCH_SIZES} \\
    --profile-dir '${LOGS_DIR}/profile' \\
    --output '${LOGS_DIR}/breakdown.json' \\
    --tag '${TAG}'"
SBATCH
EOF
)

job_id="$(ssh "$REMOTE_HOST" "$remote_cmd" | tail -n 1)"
{
  echo "# vLLM standalone SpecDec timing breakdown"
  echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "job_id=${job_id}"
  echo "job_name=vllm-specdec-breakdown-q30-k${NUM_SPECULATIVE_TOKENS}"
  echo "model=${MODEL}"
  echo "draft_model=${DRAFT_MODEL}"
  echo "num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo "scope=vLLM standalone LLM.generate torch-profiler breakdown, not NeMo-RL E2E"
  echo "figure4_buckets=Drafting,Verification,Rejection Sampling,Other vLLM overheads"
  echo "logs_dir=${LOGS_DIR}"
  echo "output_json=${LOGS_DIR}/breakdown.json"
  echo "profile_dir=${LOGS_DIR}/profile"
  echo "speculative_config_json=${LOGS_DIR}/speculative_config.json"
  echo "status=${job_id} submitted"
} | tee "$JOB_FILE"
