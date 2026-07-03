#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
MODELS="${MODELS:-qwen30ba3b qwen32 qwen235b}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_nemorl_perfcfg_sync}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/nemorl-perfcfg-sync}"
PROMPT_JSONL="${PROMPT_JSONL:-${LUSTRE_ROOT}/vllm024-dynamicsd/datasets/openmathinstruct2_469216e3f46f_prompts_1024_offset0.jsonl}"
SMOKE="${SMOKE:-true}"
VARIANTS="${VARIANTS:-baseline static dynamic}"
ROLLOUT_BATCHES="${ROLLOUT_BATCHES:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

for model_key in ${MODELS}; do
  case "${model_key}" in
    qwen30ba3b)
      model="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
      draft_model="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
      global_prompts=64
      recipe_nodes=4
      target_tp=1
      max_sequence_length=4096
      gpu_memory_utilization=0.6
      moe_backend=triton
      prompt_offset=0
      ;;
    qwen32)
      model="${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
      draft_model="${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47"
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      global_prompts=64
      recipe_nodes=4
      target_tp=2
      max_sequence_length=4096
      gpu_memory_utilization=0.6
      moe_backend=""
      prompt_offset=100
      ;;
    qwen235b)
      model="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65"
      draft_model="${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba"
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml"
      global_prompts=16
      recipe_nodes=32
      target_tp=8
      max_sequence_length=8192
      gpu_memory_utilization=0.6
      moe_backend=triton
      prompt_offset=200
      ;;
    *)
      echo "Unsupported model: ${model_key}" >&2
      exit 2
      ;;
  esac

  generations_per_prompt=32
  total_gpus=$((recipe_nodes * 4))
  generation_replicas=$((total_gpus / target_tp))
  if (( global_prompts % generation_replicas != 0 )); then
    echo "Recipe prompts do not divide generation replicas for ${model_key}" >&2
    exit 3
  fi
  per_engine_prompts=$((global_prompts / generation_replicas))
  per_engine_requests=$((per_engine_prompts * generations_per_prompt))
  benchmark_nodes=$(((target_tp + 3) / 4))
  distributed_backend=""
  if (( benchmark_nodes > 1 )); then
    distributed_backend=ray
  fi

  if [[ "${SMOKE}" == "true" ]]; then
    max_new_tokens=256
    effective_rollout_batches="${ROLLOUT_BATCHES:-1}"
  else
    max_new_tokens="${max_sequence_length}"
    effective_rollout_batches="${ROLLOUT_BATCHES:-3}"
  fi

  echo "model=${model_key}"
  echo "recipe=${recipe}"
  echo "global_prompts=${global_prompts}"
  echo "generations_per_prompt=${generations_per_prompt}"
  echo "generation_replicas=${generation_replicas}"
  echo "per_engine_prompts=${per_engine_prompts}"
  echo "per_engine_requests=${per_engine_requests}"
  echo "target_tp=${target_tp}"
  echo "max_sequence_length=${max_sequence_length}"

  env \
    CLUSTER="${CLUSTER:-auto}" \
    ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
    PARTITION="${PARTITION:-gb200}" \
    LUSTRE_ROOT="${LUSTRE_ROOT}" \
    HF_HOME="${HF_HOME}" \
    MODEL="${model}" \
    DRAFT_MODEL="${draft_model}" \
    RESULT_ROOT="${RESULT_ROOT}/${RUN_ID}" \
    RUN_ID="${model_key}" \
    JOB_LABEL="perfcfg-${model_key}" \
    VARIANTS="${VARIANTS}" \
    STATIC_K=5 \
    DYNAMIC_SCHEDULE="1:16:5,17:32:4,33:64:3,65:128:1,129:512:0" \
    TP="${target_tp}" \
    PP=1 \
    NODES="${benchmark_nodes}" \
    SEGMENT="${benchmark_nodes}" \
    DISTRIBUTED_EXECUTOR_BACKEND="${distributed_backend}" \
    TEMPERATURE=1.0 \
    TOP_P=1.0 \
    GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
    CUDAGRAPH_MODE=PIECEWISE \
    ENGINE_MAX_NUM_SEQS="${per_engine_requests}" \
    MOE_BACKEND="${moe_backend}" \
    PROMPT_JSONL="${PROMPT_JSONL}" \
    PROMPT_OFFSET="${prompt_offset}" \
    SOURCE_RECIPE="${recipe}" \
    GLOBAL_NUM_PROMPTS="${global_prompts}" \
    GLOBAL_GENERATION_REPLICAS="${generation_replicas}" \
    SMOKE="${SMOKE}" \
    NUM_PROMPTS="${per_engine_prompts}" \
    SAMPLES_PER_PROMPT="${generations_per_prompt}" \
    ROLLOUT_BATCHES="${effective_rollout_batches}" \
    MAX_PROMPT_TOKENS="${max_sequence_length}" \
    MAX_NEW_TOKENS="${max_new_tokens}" \
    MAX_MODEL_LEN="${max_sequence_length}" \
    MAX_NUM_BATCHED_TOKENS=recipe \
    TIME_LIMIT="${TIME_LIMIT:-05:00:00}" \
    DRY_RUN="${DRY_RUN}" \
    TEST_ONLY="${TEST_ONLY}" \
    REQUIRE_GIT_PULL=false \
    "${SCRIPT_DIR}/submit_sync_rollout.sh"
done
