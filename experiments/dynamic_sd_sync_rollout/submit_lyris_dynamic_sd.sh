#!/usr/bin/env bash
# Submit DynamicSD sync-rollout benchmark jobs to Lyris (GB200, vLLM 0.24 venv).
#
# MODE=profile  submits one job per K (0=baseline) sweeping BATCH_SIZES.
# MODE=rollout  submits one job per variant: baseline, eagle3 fixed-K, dynamic
#               (dynamic needs DYNAMIC_SPEC_JSON produced by derive_dynamic_k_table.py).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
PYTHON_BIN="${PYTHON_BIN:-/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/vllm024/bin/python}"
HF_HOME_REMOTE="${HF_HOME_REMOTE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
LOG_ROOT="${LOG_ROOT:-${REMOTE_REPO}/dynamic_sd_runs}"
RUN_TAG_DATE="${RUN_TAG_DATE:-$(date +%Y%m%d)}"
JOB_FILE="${JOB_FILE:-${ROOT_DIR}/experiments/dynamic_sd_sync_rollout/latest_lyris_jobs.txt}"

MODE="${MODE:-profile}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_30ba3b_thinking}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3}"
TP="${TP:-1}"
BENCH="${BENCH:-math}"
PROMPT_JSONL="${PROMPT_JSONL:-${REMOTE_REPO}/data/math_500.jsonl}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
ISL_CAP="${ISL_CAP:-4096}"
# Sampling matches NeMo-RL grpo_math_1B.yaml (temperature/top_p 1.0, seed 42).
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-42}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Pin backends so baseline/fixed/dynamic jobs can never diverge via vLLM auto
# selection. FLASHINFER is the smoke-validated default on GB200 + vllm024 venv.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
MOE_BACKEND="${MOE_BACKEND:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

# profile mode
K_VALUES="${K_VALUES:-0 1 2 3 5}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128}"
OSL="${OSL:-1024}"
REPEATS="${REPEATS:-2}"

# rollout mode: one vLLM DP-worker shard of the NeMo-RL GB200 SyncRL recipes
# (grpo-qwen3-*-4n4g / 16n4g: G=32 gens per prompt; N set per model so
# N*G = per-engine sequence count).
ROLLOUT_VARIANTS="${ROLLOUT_VARIANTS:-baseline fixed dynamic}"
FIXED_K="${FIXED_K:-3}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-4}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-32}"
NUM_STEPS="${NUM_STEPS:-4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
DYNAMIC_SPEC_JSON="${DYNAMIC_SPEC_JSON:-}"

# vLLM 0.24: how the drafter samples ("greedy" argmax vs "probabilistic"
# stochastic sampling with cached draft logits for exact rejection sampling).
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}"
TAG_SUFFIX=""
if [[ "${DRAFT_SAMPLE_METHOD}" != "greedy" ]]; then
  TAG_SUFFIX="_${DRAFT_SAMPLE_METHOD}"
fi

# EAGLE3 heads are single-layer: draft TP=1 regardless of target TP (matches
# prior specdec scripts in this repo).
spec_json_fixed() {
  local k="$1"
  printf '{"method": "eagle3", "model": "%s", "num_speculative_tokens": %d, "draft_tensor_parallel_size": 1, "draft_sample_method": "%s"}' \
    "${DRAFT_MODEL}" "${k}" "${DRAFT_SAMPLE_METHOD}"
}

spec_json_with_sample_method() {
  local json_path="$1"
  python3 - "$json_path" "${DRAFT_SAMPLE_METHOD}" <<'PY'
import json, sys
spec = json.loads(open(sys.argv[1]).read())
spec["draft_sample_method"] = sys.argv[2]
print(json.dumps(spec))
PY
}

sync_harness() {
  ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO}' '${LOG_ROOT}'"
  scp -q \
    "${ROOT_DIR}/experiments/dynamic_sd_sync_rollout/sync_rollout_dynamic_sd.py" \
    "${REMOTE_HOST}:${REMOTE_REPO}/sync_rollout_dynamic_sd.py"
}

submit_job() {
  local tag="$1"
  local spec_json="$2"   # empty = baseline
  local mode_args="$3"

  local logs_dir="${LOG_ROOT}/${tag}"
  local spec_arg=""
  local cudagraph_arg=""
  local backend_args=""
  if [[ -n "${CUDAGRAPH_SIZES}" ]]; then
    cudagraph_arg="--cudagraph-capture-sizes ${CUDAGRAPH_SIZES}"
  fi
  if [[ -n "${ATTENTION_BACKEND}" ]]; then
    backend_args="--attention-backend ${ATTENTION_BACKEND}"
  fi
  if [[ -n "${MOE_BACKEND}" ]]; then
    backend_args="${backend_args} --moe-backend ${MOE_BACKEND}"
  fi
  ssh "${REMOTE_HOST}" "mkdir -p '${logs_dir}'"
  if [[ -n "${spec_json}" ]]; then
    printf "%s\n" "${spec_json}" | ssh "${REMOTE_HOST}" "cat > '${logs_dir}/speculative_config.json'"
    spec_arg="--speculative-config @${logs_dir}/speculative_config.json"
  fi

  ssh "${REMOTE_HOST}" "cat > '${logs_dir}/run.sbatch'" <<SBATCH
#!/usr/bin/env bash
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${ACCOUNT}-dynsd.${tag}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${logs_dir}/slurm-%j.out

set -euo pipefail
export HF_HOME=${HF_HOME_REMOTE}
export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --nodes=1 --ntasks=1 bash -lc "set -euo pipefail; \\
  cd '${REMOTE_REPO}'; \\
  ${PYTHON_BIN} sync_rollout_dynamic_sd.py \\
    --model '${MODEL}' \\
    ${spec_arg} \\
    --tp ${TP} \\
    --disable-custom-all-reduce \\
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --prompt-jsonl '${PROMPT_JSONL}' \\
    --prompt-offset ${PROMPT_OFFSET} \\
    --isl-cap ${ISL_CAP} \\
    --temperature ${TEMPERATURE} --top-p ${TOP_P} --top-k ${TOP_K} \\
    --seed ${SEED} \\
    ${cudagraph_arg} \\
    ${backend_args} \\
    ${mode_args} \\
    --output '${logs_dir}/results.json' \\
    --tag '${tag}'"
SBATCH

  local job_id
  job_id="$(ssh "${REMOTE_HOST}" "sbatch --parsable ${SBATCH_EXTRA_ARGS} '${logs_dir}/run.sbatch'" | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "ERROR: failed to submit ${tag}" >&2
    exit 1
  fi
  printf "%s,%s,%s,%s,%s\n" "${job_id}" "${tag}" "${MODEL}" "${BENCH}" "${logs_dir}"
}

sync_harness

{
  echo "# DynamicSD ${MODE} jobs (${RUN_TAG_DATE})"
  echo "remote_host=${REMOTE_HOST}"
  echo "model=${MODEL}"
  echo "draft=${DRAFT_MODEL}"
  echo "bench=${BENCH} prompts=${PROMPT_JSONL}"
  echo "temperature=${TEMPERATURE} tp=${TP} max_num_seqs=${MAX_NUM_SEQS}"
  echo "csv_header=job_id,tag,model,bench,logs_dir"

  if [[ "${MODE}" == "profile" ]]; then
    for k in ${K_VALUES}; do
      tag="${MODEL_LABEL}_${BENCH}_profile_k${k}${TAG_SUFFIX}_${RUN_TAG_DATE}"
      spec_json=""
      if [[ "${k}" != "0" ]]; then
        spec_json="$(spec_json_fixed "${k}")"
      fi
      mode_args="--mode profile --batch-sizes ${BATCH_SIZES} --osl ${OSL} --repeats ${REPEATS}"
      submit_job "${tag}" "${spec_json}" "${mode_args}"
    done
  elif [[ "${MODE}" == "rollout" ]]; then
    mode_args="--mode rollout --num-prompts-per-step ${NUM_PROMPTS_PER_STEP} --num-generations-per-prompt ${NUM_GENERATIONS_PER_PROMPT} --num-steps ${NUM_STEPS} --max-tokens ${MAX_TOKENS} --per-request-seed"
    for variant in ${ROLLOUT_VARIANTS}; do
      case "${variant}" in
        baseline)
          submit_job "${MODEL_LABEL}_${BENCH}_rollout_baseline_${RUN_TAG_DATE}" "" "${mode_args}"
          ;;
        fixed)
          submit_job "${MODEL_LABEL}_${BENCH}_rollout_fixed_k${FIXED_K}${TAG_SUFFIX}_${RUN_TAG_DATE}" \
            "$(spec_json_fixed "${FIXED_K}")" "${mode_args}"
          ;;
        suffix)
          # exploratory: vLLM 0.24 native suffix decoding (needs arctic-inference in the venv)
          submit_job "${MODEL_LABEL}_${BENCH}_rollout_suffix_${RUN_TAG_DATE}" \
            '{"method": "suffix"}' "${mode_args}"
          ;;
        dynamic)
          if [[ -z "${DYNAMIC_SPEC_JSON}" || ! -f "${DYNAMIC_SPEC_JSON}" ]]; then
            echo "ERROR: dynamic variant needs DYNAMIC_SPEC_JSON=<local path>" >&2
            exit 2
          fi
          submit_job "${MODEL_LABEL}_${BENCH}_rollout_dynamic${TAG_SUFFIX}_${RUN_TAG_DATE}" \
            "$(spec_json_with_sample_method "${DYNAMIC_SPEC_JSON}")" "${mode_args}"
          ;;
        *)
          echo "ERROR: unknown variant '${variant}'" >&2
          exit 2
          ;;
      esac
    done
  else
    echo "ERROR: unknown MODE '${MODE}'" >&2
    exit 2
  fi
} | tee "${JOB_FILE}"
