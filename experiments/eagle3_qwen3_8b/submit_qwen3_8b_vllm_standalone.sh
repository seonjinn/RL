#!/usr/bin/env bash
set -euo pipefail

# Submit Qwen3-8B standalone vLLM static-batch sweeps.
# Run from the vllm-benchmark checkout on oci-hsg:
#   /lustre/fs1/.../users/sna/vllm-benchmark

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/sna/containers/vllm-hsg-ultra-rl-v0.20.2-nemo-speed-pr24.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre,${ROOT_DIR}:/workspace}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
TP="${TP:-1}"
PP="${PP:-1}"
ISL="${ISL:-1000}"
OSL="${OSL:-512}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-64000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
DISABLE_VLLM_PROFILER="${DISABLE_VLLM_PROFILER:-true}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"
SUBMIT_BASELINE="${SUBMIT_BASELINE:-true}"
SPEC_K_LIST="${SPEC_K_LIST:-1 3}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

submit_one() {
  local mode="$1"
  local k="${2:-0}"
  local job_suffix
  local spec_arg=""
  local spec_file=""
  if [[ "${mode}" == "baseline" ]]; then
    job_suffix="baseline"
  else
    job_suffix="spec_k${k}"
  fi

  local job_name="qwen3-8b-vllm-${job_suffix}-bs1-32"
  local run_dir="${ROOT_DIR}/vllm-runs/qwen3_8b_vllm_${job_suffix}_bs1-32_cuda_graph_${TS}"
  mkdir -p "${run_dir}/profile" "${run_dir}/pydeps"

  if [[ "${mode}" == "spec" ]]; then
    spec_file="${run_dir}/speculative_config.json"
    python3 - "${spec_file}" "${DRAFT_MODEL}" "${k}" <<'PY'
import json
import sys

path, model, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "method": "eagle3",
            "model": model,
            "num_speculative_tokens": k,
            "draft_tensor_parallel_size": 1,
        },
        f,
        indent=2,
    )
PY
    spec_arg="--speculative-config @${spec_file}"
  fi

  local profiler_flag=""
  if [[ "${DISABLE_VLLM_PROFILER}" == "true" || "${DISABLE_VLLM_PROFILER}" == "1" ]]; then
    profiler_flag="--disable-vllm-profiler"
  fi
  local allreduce_flag=""
  if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" || "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    allreduce_flag="--disable-custom-all-reduce"
  fi

  local sbatch_file="${run_dir}/submit.sbatch"
  cat > "${sbatch_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --gres=gpu:4
#SBATCH --job-name=${job_name}
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

srun --container-image="${CONTAINER_IMAGE}" \\
  --container-mounts="${MOUNTS}" \\
  --mpi=pmix \\
  bash -lc '
set -euo pipefail
cd /workspace
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
export CUDA_MODULE_LOADING=LAZY
export FLASHINFER_WORKSPACE_BASE=/tmp
python3 -m pip install --quiet --target "${run_dir}/pydeps" \
  "typing_extensions>=4.15.0" "platformdirs" "lxml" >/dev/null || true
export PYTHONPATH="${run_dir}/pydeps:${PYTHONPATH:-}"

python3 standalone_vllm_specdec_breakdown.py \\
  --model "${MODEL}" \\
  --tp "${TP}" \\
  --pp "${PP}" \\
  --distributed-executor-backend none \\
  --attention-backend "${ATTENTION_BACKEND}" \\
  --dtype auto \\
  --kv-cache-dtype auto \\
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \\
  --max-model-len "${MAX_MODEL_LEN}" \\
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \\
  --isl "${ISL}" \\
  --osl "${OSL}" \\
  --batch-sizes ${BATCH_SIZES} \\
  --warmup-repeats "${WARMUP_REPEATS}" \\
  --profile-dir "${run_dir}/profile" \\
  --output "${run_dir}/breakdown.json" \\
  --tag "$(basename "${run_dir}")" \\
  ${allreduce_flag} \\
  ${profiler_flag} \\
  ${spec_arg}
'
EOF

  local job_id
  job_id="$(sbatch --parsable "${sbatch_file}")"
  echo "${job_id} ${mode} ${job_suffix} ${run_dir}"
}

status_file="${ROOT_DIR}/latest_qwen3_8b_vllm_standalone_jobs.txt"
{
  echo "# submitted $(date)"
  echo "# model=${MODEL}"
  echo "# drafter=${DRAFT_MODEL}"
  echo "# isl=${ISL} osl=${OSL} batch_sizes=${BATCH_SIZES}"
  if [[ "${SUBMIT_BASELINE}" == "true" || "${SUBMIT_BASELINE}" == "1" ]]; then
    submit_one baseline
  fi
  for k in ${SPEC_K_LIST}; do
    submit_one spec "${k}"
  done
} | tee "${status_file}"
