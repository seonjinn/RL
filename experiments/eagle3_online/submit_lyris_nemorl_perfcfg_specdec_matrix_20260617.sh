#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/project/coreai_dlalgo_llm/users/sna/RL-latest-main-canary-20260618}"
REMOTE_SSH_OPTS="${REMOTE_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
REMOTE_SCP_OPTS="${REMOTE_SCP_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
OUT="${OUT:-${ROOT_DIR}/latest_lyris_nemorl_perfcfg_specdec_reference_20260618_jobs.csv}"
SUBMIT="${SUBMIT:-false}"

CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:---comment=metrics --network=sharp}"
BASE_SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS}"
LAUNCHER_GRES_FLAG="${LAUNCHER_GRES_FLAG:-}"
INITIAL_DEPENDENCY="${INITIAL_DEPENDENCY:-}"
SERIALIZE_SUBMISSIONS="${SERIALIZE_SUBMISSIONS:-false}"
RUN_ID="${RUN_ID:-20260618_lyris_nemorl_perfcfg_specdec_reference_recipe_osl}"
RUN_WORK_ROOT="${RUN_WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${RUN_WORK_ROOT}/logs}"
RUN_CACHE_ROOT="${RUN_CACHE_ROOT:-${RUN_WORK_ROOT}/cache}"
NODE_LOCAL_CACHE_ROOT="${NODE_LOCAL_CACHE_ROOT:-/tmp/${USER:-sna}/nemorl_${RUN_ID}}"
# Leave empty so ray.sub does not mount over /root/.cache/uv. The nightly
# image's base venv contains symlinks into that cache.
UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_CACHE_ROOT}/pip}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${NODE_LOCAL_CACHE_ROOT}/torch_extensions}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${NODE_LOCAL_CACHE_ROOT}/xdg}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${NODE_LOCAL_CACHE_ROOT}/vllm}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${NODE_LOCAL_CACHE_ROOT}/flashinfer}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${NODE_LOCAL_CACHE_ROOT}/triton}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${NODE_LOCAL_CACHE_ROOT}/torchinductor}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${NODE_LOCAL_CACHE_ROOT}/cuda}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${NODE_LOCAL_CACHE_ROOT}/pycache}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
MEGATRON_DATASET_HELPERS_BUILD_DIR="${MEGATRON_DATASET_HELPERS_BUILD_DIR:-${NODE_LOCAL_CACHE_ROOT}/megatron_dataset_helpers}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-perfcfg-specdec-lyris}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_NETRC_HOME="${WANDB_NETRC_HOME:-}"

MODELS="${MODELS:-qwen30ba3b qwen32 qwen235b}"
MODES="${MODES:-sync async1off}"
METHODS="${METHODS:-baseline pard pard2 eagle3 suffix}"

MAX_STEPS="${MAX_STEPS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-recipe}"
case "${MAX_NEW_TOKENS}" in
  recipe|default|config|"")
    USE_RECIPE_MAX_NEW_TOKENS=true
    MIN_TOKENS="${MIN_TOKENS:-}"
    DEFAULT_IGNORE_EOS=false
    DEFAULT_DISABLE_STOPS=false
    ;;
  *)
    USE_RECIPE_MAX_NEW_TOKENS=false
    MIN_TOKENS="${MIN_TOKENS:-${MAX_NEW_TOKENS}}"
    DEFAULT_IGNORE_EOS=true
    DEFAULT_DISABLE_STOPS=true
    ;;
esac
NRL_VLLM_GENERATION_IGNORE_EOS="${NRL_VLLM_GENERATION_IGNORE_EOS:-${DEFAULT_IGNORE_EOS}}"
NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS="${NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS:-${DEFAULT_DISABLE_STOPS}}"
NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS="${NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS:-${DEFAULT_DISABLE_STOPS}}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_P="${GENERATION_TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:--1}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-triton}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_USE_TQDM="${VLLM_USE_TQDM:-}"
VLLM_ENABLE_METRICS_LOGGER="${VLLM_ENABLE_METRICS_LOGGER:-true}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-}"
ENV_MATH_NUM_WORKERS="${ENV_MATH_NUM_WORKERS:-}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-}"
LOGGER_MONITOR_GPUS="${LOGGER_MONITOR_GPUS:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
QWEN235B_HYBRIDEP_NUM_RANKS_PER_NVLINK_DOMAIN="${QWEN235B_HYBRIDEP_NUM_RANKS_PER_NVLINK_DOMAIN:-}"
QWEN235B_HYBRIDEP_USE_MNNVL="${QWEN235B_HYBRIDEP_USE_MNNVL:-}"
QWEN235B_MOE_TOKEN_DISPATCHER_TYPE="${QWEN235B_MOE_TOKEN_DISPATCHER_TYPE:-}"
QWEN235B_MOE_FLEX_DISPATCHER_BACKEND="${QWEN235B_MOE_FLEX_DISPATCHER_BACKEND:-}"
QWEN235B_SEQUENCE_PACKING_FUSE_LOSS="${QWEN235B_SEQUENCE_PACKING_FUSE_LOSS:-}"
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-}"
QWEN235B_CUDA_DEVICE_MAX_CONNECTIONS="${QWEN235B_CUDA_DEVICE_MAX_CONNECTIONS:-1}"

WALLTIME_SMALL="${WALLTIME_SMALL:-05:00:00}"
WALLTIME_235B="${WALLTIME_235B:-05:00:00}"

QWEN30_LOCAL_SNAPSHOT="${QWEN30_LOCAL_SNAPSHOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
QWEN32_LOCAL_SNAPSHOT="${QWEN32_LOCAL_SNAPSHOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
QWEN235B_LOCAL_SNAPSHOT="${QWEN235B_LOCAL_SNAPSHOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65}"

PARD_DRAFT_MODEL="${PARD_DRAFT_MODEL:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd}"
QWEN30_PARD2_DRAFT_MODEL="${QWEN30_PARD2_DRAFT_MODEL:-}"
QWEN32_PARD2_DRAFT_MODEL="${QWEN32_PARD2_DRAFT_MODEL:-}"
QWEN235B_PARD2_DRAFT_MODEL="${QWEN235B_PARD2_DRAFT_MODEL:-amd/PARD2-Qwen3-8B}"
QWEN30_EAGLE3_DRAFT_MODEL="${QWEN30_EAGLE3_DRAFT_MODEL:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
QWEN32_EAGLE3_DRAFT_MODEL="${QWEN32_EAGLE3_DRAFT_MODEL:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
QWEN235B_EAGLE3_DRAFT_MODEL="${QWEN235B_EAGLE3_DRAFT_MODEL:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba}"
ARCTIC_SITE="${ARCTIC_SITE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/build_deps/arctic-inference-0.1.1-py313-native}"

PARD_SPEC_TOKENS="${PARD_SPEC_TOKENS:-}"
PARD2_SPEC_TOKENS="${PARD2_SPEC_TOKENS:-}"
PARD_DRAFT_TP="${PARD_DRAFT_TP:-}"
PARD2_DRAFT_TP="${PARD2_DRAFT_TP:-}"
QWEN30_PARD_SPEC_TOKENS="${QWEN30_PARD_SPEC_TOKENS:-5}"
QWEN32_PARD_SPEC_TOKENS="${QWEN32_PARD_SPEC_TOKENS:-5}"
QWEN235B_PARD_SPEC_TOKENS="${QWEN235B_PARD_SPEC_TOKENS:-11}"
QWEN30_PARD2_SPEC_TOKENS="${QWEN30_PARD2_SPEC_TOKENS:-5}"
QWEN32_PARD2_SPEC_TOKENS="${QWEN32_PARD2_SPEC_TOKENS:-5}"
QWEN235B_PARD2_SPEC_TOKENS="${QWEN235B_PARD2_SPEC_TOKENS:-11}"
EAGLE3_SPEC_TOKENS="${EAGLE3_SPEC_TOKENS:-3}"
SUFFIX_SPEC_TOKENS="${SUFFIX_SPEC_TOKENS:-32}"
SUFFIX_DECODING_MAX_TREE_DEPTH="${SUFFIX_DECODING_MAX_TREE_DEPTH:-24}"
SUFFIX_DECODING_MAX_CACHED_REQUESTS="${SUFFIX_DECODING_MAX_CACHED_REQUESTS:-10000}"
SUFFIX_DECODING_MAX_SPEC_FACTOR="${SUFFIX_DECODING_MAX_SPEC_FACTOR:-1.0}"
SUFFIX_DECODING_MIN_TOKEN_PROB="${SUFFIX_DECODING_MIN_TOKEN_PROB:-0.1}"

UV_PYTHON="${UV_PYTHON:-3.13.13}"
RAY_VERSION="${RAY_VERSION:-2.55.1}"
RAY_PYTHON_VERSION="${RAY_PYTHON_VERSION:-3.13.13}"
RAY_PYTHON_SPEC="${RAY_PYTHON_SPEC:-3.13.13}"
RAY_USE_EXISTING_ENV="${RAY_USE_EXISTING_ENV:-true}"
USE_SYSTEM_ENV="${USE_SYSTEM_ENV:-true}"
# With USE_SYSTEM_ENV=true the driver uses the container's active
# /opt/nemo_rl_venv. Keep global/vLLM actors off the system executable so vLLM
# gets its dependency env; MCore stays on its mcore actor env unless explicitly
# overridden because policy workers need transformer-engine/modelopt.
NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-0}"
NEMO_RL_MCORE_PY_EXECUTABLES_SYSTEM="${NEMO_RL_MCORE_PY_EXECUTABLES_SYSTEM:-0}"
SYSTEM_PYDEPS_SITE="${SYSTEM_PYDEPS_SITE:-${RUN_CACHE_ROOT}/system_pydeps}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-12.0}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-120}"
CUDAARCHS="${CUDAARCHS:-120}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${RUN_WORK_ROOT}/driver_venvs/reference_py313_mcore}"
AUTO_INIT_SUBMODULES="${AUTO_INIT_SUBMODULES:-true}"
PARD2_OFFICIAL_VLLM_PATCH_DIR="${PARD2_OFFICIAL_VLLM_PATCH_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/patches/pard2_official_target_feat_20260612}"
PARD2_OFFICIAL_VLLM_SITE="${PARD2_OFFICIAL_VLLM_SITE:-${RUN_CACHE_ROOT}/vllm_pard2_official_target_feat}"
PARD2_VLLM_PATCH_PYTHON="${PARD2_VLLM_PATCH_PYTHON:-/opt/nemo_rl_venv/bin/python}"
PARD2_VLLM_SOURCE_SITE="${PARD2_VLLM_SOURCE_SITE:-}"
NRL_ACTOR_UV_LOCK_MODE="${NRL_ACTOR_UV_LOCK_MODE:---locked}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
NRL_FORCE_REBUILD_ACTOR_VENVS="${NRL_FORCE_REBUILD_ACTOR_VENVS:-false}"
NRL_SERIALIZE_ACTOR_VENV_CREATION="${NRL_SERIALIZE_ACTOR_VENV_CREATION:-false}"

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

effective_max_new_tokens_label() {
  local model="${1}"
  if [[ "${USE_RECIPE_MAX_NEW_TOKENS}" == "true" ]]; then
    case "${model}" in
      qwen235b) echo "8192" ;;
      qwen30ba3b|qwen32) echo "4096" ;;
      *) echo "recipe" ;;
    esac
  else
    echo "${MAX_NEW_TOKENS}"
  fi
}

pard2_drafter_for_model() {
  case "$1" in
    qwen30ba3b)
      if [[ -z "${QWEN30_PARD2_DRAFT_MODEL}" ]]; then
        echo "ERROR: qwen30ba3b PARD-2 requires QWEN30_PARD2_DRAFT_MODEL with pard2_target_dim=8192." >&2
        return 2
      fi
      echo "${QWEN30_PARD2_DRAFT_MODEL}"
      ;;
    qwen32)
      if [[ -z "${QWEN32_PARD2_DRAFT_MODEL}" ]]; then
        echo "ERROR: qwen32 PARD-2 requires QWEN32_PARD2_DRAFT_MODEL with matching pard2_target_dim." >&2
        return 2
      fi
      echo "${QWEN32_PARD2_DRAFT_MODEL}"
      ;;
    qwen235b)
      echo "${QWEN235B_PARD2_DRAFT_MODEL}"
      ;;
    *)
      echo "ERROR: unsupported model for PARD-2: $1" >&2
      return 2
      ;;
  esac
}

spec_tokens_for_model() {
  local method="$1"
  local model="$2"
  if [[ "${method}" == "pard" && -n "${PARD_SPEC_TOKENS}" ]]; then
    echo "${PARD_SPEC_TOKENS}"
    return
  fi
  if [[ "${method}" == "pard2" && -n "${PARD2_SPEC_TOKENS}" ]]; then
    echo "${PARD2_SPEC_TOKENS}"
    return
  fi
  case "${method}:${model}" in
    pard:qwen30ba3b) echo "${QWEN30_PARD_SPEC_TOKENS}" ;;
    pard:qwen32) echo "${QWEN32_PARD_SPEC_TOKENS}" ;;
    pard:qwen235b) echo "${QWEN235B_PARD_SPEC_TOKENS}" ;;
    pard2:qwen30ba3b) echo "${QWEN30_PARD2_SPEC_TOKENS}" ;;
    pard2:qwen32) echo "${QWEN32_PARD2_SPEC_TOKENS}" ;;
    pard2:qwen235b) echo "${QWEN235B_PARD2_SPEC_TOKENS}" ;;
    *)
      echo "ERROR: unsupported spec token lookup '${method}:${model}'" >&2
      return 2
      ;;
  esac
}

run_remote_cmd() {
  local cmd="$1"
  if [[ "${REMOTE_HOST}" == "local" ]]; then
    bash -lc "${cmd}"
  else
    printf "%s\n" "${cmd}" | ssh ${REMOTE_SSH_OPTS} "${REMOTE_HOST}" bash -s
  fi
}

stage_helper() {
  if [[ "${REMOTE_HOST}" == "local" ]]; then
    mkdir -p "${REMOTE_REPO}/experiments/eagle3_online"
    if [[ "${ROOT_DIR}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh" != "${REMOTE_REPO}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh" ]]; then
      cp -f "${ROOT_DIR}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh" \
        "${REMOTE_REPO}/experiments/eagle3_online/"
    fi
    if [[ "${ROOT_DIR}/experiments/eagle3_online/prepare_pard2_official_vllm_site.sh" != "${REMOTE_REPO}/experiments/eagle3_online/prepare_pard2_official_vllm_site.sh" ]]; then
      cp -f "${ROOT_DIR}/experiments/eagle3_online/prepare_pard2_official_vllm_site.sh" \
        "${REMOTE_REPO}/experiments/eagle3_online/"
    fi
  else
    ssh ${REMOTE_SSH_OPTS} "${REMOTE_HOST}" \
      "mkdir -p '${REMOTE_REPO}/experiments/eagle3_online'"
    scp -q ${REMOTE_SCP_OPTS} "${ROOT_DIR}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh" \
      "${REMOTE_HOST}:${REMOTE_REPO}/experiments/eagle3_online/"
    scp -q ${REMOTE_SCP_OPTS} "${ROOT_DIR}/experiments/eagle3_online/prepare_pard2_official_vllm_site.sh" \
      "${REMOTE_HOST}:${REMOTE_REPO}/experiments/eagle3_online/"
  fi
}

remote_preflight() {
  local method_optional_checks=""
  if [[ " ${METHODS} " == *" pard "* ]]; then
    method_optional_checks="${method_optional_checks} test -e '${PARD_DRAFT_MODEL}';"
  fi
  if [[ " ${METHODS} " == *" suffix "* || " ${METHODS} " == *" suffixdecoding "* || " ${METHODS} " == *" pard "* ]]; then
    method_optional_checks="${method_optional_checks} test -d '${ARCTIC_SITE}/arctic_inference/suffix_decoding';"
  fi
  if [[ " ${METHODS} " == *" pard2 "* || " ${METHODS} " == *" pard-2 "* ]]; then
    method_optional_checks="${method_optional_checks} test -s '${PARD2_OFFICIAL_VLLM_PATCH_DIR}/vllm_pard2_official_target_feat.patch';"
    method_optional_checks="${method_optional_checks} test -s '${PARD2_OFFICIAL_VLLM_PATCH_DIR}/apply_pard2_alias_idempotent.py';"
    method_optional_checks="${method_optional_checks} test -s '${PARD2_OFFICIAL_VLLM_PATCH_DIR}/check_pard2_official_patch.py';"
  fi

  run_remote_cmd "set -euo pipefail; \
    test -d '${REMOTE_REPO}'; \
    test -s '${REMOTE_REPO}/examples/run_grpo.py'; \
    test -s '${REMOTE_REPO}/ray.sub'; \
	    test -s '${REMOTE_REPO}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh'; \
	    bash -n '${REMOTE_REPO}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh'; \
	    if [[ '${AUTO_INIT_SUBMODULES}' == 'true' ]]; then \
	      git -C '${REMOTE_REPO}' submodule update --init --recursive \
	        3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
	        3rdparty/Automodel-workspace/Automodel \
	        3rdparty/Gym-workspace/Gym >/dev/null; \
	    fi; \
	    test -d '${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron'; \
	    PYTHONPATH='${REMOTE_REPO}:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM' \
	      python3 -c 'import megatron'; \
	    test -s '${REMOTE_REPO}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml'; \
    test -s '${REMOTE_REPO}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml'; \
    test -s '${REMOTE_REPO}/examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml'; \
    test -s '${REMOTE_REPO}/examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml'; \
    test -s '${REMOTE_REPO}/examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml'; \
    test -s '${REMOTE_REPO}/examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml'; \
    test -s '${CONTAINER}'; \
    mkdir -p '${HF_HOME}' '${HF_DATASETS_CACHE}' '${RUN_CACHE_ROOT}' '${PIP_CACHE_DIR}' '${TORCH_EXTENSIONS_DIR}' '${XDG_CACHE_HOME}' '${VLLM_CACHE_ROOT}' '${FLASHINFER_CACHE_DIR}' '${TRITON_CACHE_DIR}' '${TORCHINDUCTOR_CACHE_DIR}' '${CUDA_CACHE_PATH}' '${PYTHONPYCACHEPREFIX}' '${MEGATRON_DATASET_HELPERS_BUILD_DIR}'; \
    mkdir -p '${RUN_WORK_ROOT}/.preflight_write_probe'; \
    rmdir '${RUN_WORK_ROOT}/.preflight_write_probe'; \
    test -e '${QWEN30_LOCAL_SNAPSHOT}'; \
    test -e '${QWEN32_LOCAL_SNAPSHOT}'; \
    test -e '${QWEN235B_LOCAL_SNAPSHOT}'; \
    test -e '${PARD_DRAFT_MODEL}'; \
    test -e '${QWEN30_EAGLE3_DRAFT_MODEL}'; \
    test -e '${QWEN32_EAGLE3_DRAFT_MODEL}'; \
    test -e '${QWEN235B_EAGLE3_DRAFT_MODEL}'; \
    ${method_optional_checks} \
    echo remote_head=\$(git -C '${REMOTE_REPO}' rev-parse --short HEAD 2>/dev/null || true)"
}

model_base() {
  local model="$1"
  case "${model}" in
    qwen30ba3b)
      model_label=qwen30ba3b
      target_model="${QWEN30_LOCAL_SNAPSHOT}"
      tokenizer="${QWEN30_LOCAL_SNAPSHOT}"
      eagle3_draft_model="${QWEN30_EAGLE3_DRAFT_MODEL}"
      ;;
    qwen32)
      model_label=qwen32
      target_model="${QWEN32_LOCAL_SNAPSHOT}"
      tokenizer="${QWEN32_LOCAL_SNAPSHOT}"
      eagle3_draft_model="${QWEN32_EAGLE3_DRAFT_MODEL}"
      ;;
    qwen235b)
      model_label=qwen235b
      target_model="${QWEN235B_LOCAL_SNAPSHOT}"
      tokenizer="${QWEN235B_LOCAL_SNAPSHOT}"
      eagle3_draft_model="${QWEN235B_EAGLE3_DRAFT_MODEL}"
      ;;
    *)
      echo "ERROR: unknown model '${model}'" >&2
      exit 2
      ;;
  esac
}

mode_shape() {
  local model="$1"
  local mode="$2"
  case "${model}:${mode}" in
    qwen30ba3b:sync)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
      num_nodes=4
      segment=4
      num_prompts=64
      num_generations=32
      train_global_batch_size=512
      draft_tp=1
      ;;
    qwen30ba3b:async1off)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml
      num_nodes=4
      segment=4
      num_prompts=64
      num_generations=32
      train_global_batch_size=512
      draft_tp=1
      ;;
    qwen32:sync)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml
      num_nodes=4
      segment=4
      num_prompts=64
      num_generations=32
      train_global_batch_size=512
      draft_tp=2
      ;;
    qwen32:async1off)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml
      num_nodes=8
      segment=8
      num_prompts=64
      num_generations=32
      train_global_batch_size=512
      draft_tp=1
      ;;
    qwen235b:sync)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml
      num_nodes=32
      # Lyris segment size must be <= one 18-node NVL block and divide NumNodes.
      # For 32 nodes, 16 is the largest valid balanced segment.
      segment=16
      num_prompts=16
      num_generations=32
      train_global_batch_size=512
      draft_tp=8
      ;;
    qwen235b:async1off)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml
      num_nodes=32
      # Lyris segment size must be <= one 18-node NVL block and divide NumNodes.
      # For 32 nodes, 16 is the largest valid balanced segment.
      segment=16
      num_prompts=16
      num_generations=32
      train_global_batch_size=512
      draft_tp=8
      ;;
    *)
      echo "ERROR: unknown model/mode '${model}:${mode}'" >&2
      exit 2
      ;;
  esac
  gpus_per_node=4
}

method_contract() {
  local method="$1"

  method_label="${method}"
  enable_vllm_specdec=true
  draft_format="${method}"
  specdec_method="${method}"
  draft_model=""
  spec_tokens=0
  include_draft_tp=true
  parallel_drafting=false
  omit_generation_logprobs=false
  source_vllm_site=""

  case "${method}" in
    baseline|base|none|nospec)
      method_label=baseline
      enable_vllm_specdec=false
      draft_format=auto
      specdec_method=eagle3
      draft_model="${target_model}"
      include_draft_tp=false
      ;;
    pard)
      draft_format=pard
      specdec_method=draft_model
      draft_model="${PARD_DRAFT_MODEL}"
      spec_tokens="$(spec_tokens_for_model pard "${model}")"
      # draft_model currently expects target and draft TP to match in the vLLM
      # proposer. Keep the recipe/model TP unless a specific smoke test overrides it.
      if [[ -n "${PARD_DRAFT_TP}" ]]; then
        draft_tp="${PARD_DRAFT_TP}"
      fi
      parallel_drafting=true
      omit_generation_logprobs=true
      ;;
    pard2|pard-2)
      method_label=pard2
      draft_format=pard2
      specdec_method=pard2
      draft_model="$(pard2_drafter_for_model "${model}")"
      spec_tokens="$(spec_tokens_for_model pard2 "${model}")"
      if [[ -n "${PARD2_DRAFT_TP}" ]]; then
        draft_tp="${PARD2_DRAFT_TP}"
      fi
      parallel_drafting=true
      omit_generation_logprobs=true
      source_vllm_site="${PARD2_OFFICIAL_VLLM_SITE}"
      ;;
    eagle3|eagle-3)
      method_label=eagle3
      draft_format=eagle3
      specdec_method=eagle3
      draft_model="${eagle3_draft_model}"
      spec_tokens="${EAGLE3_SPEC_TOKENS}"
      ;;
    suffix|suffixdecoding)
      method_label=suffix
      draft_format=suffix
      specdec_method=suffix
      draft_model="${target_model}"
      spec_tokens="${SUFFIX_SPEC_TOKENS}"
      include_draft_tp=false
      source_vllm_site="${ARCTIC_SITE}"
      ;;
    *)
      echo "ERROR: unknown method '${method}'" >&2
      exit 2
      ;;
  esac
}

submit_one() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local dry_run="$4"

  model_base "${model}"
  mode_shape "${model}" "${mode}"
  method_contract "${method}"

  local walltime="${WALLTIME_SMALL}"
  if [[ "${model}" == "qwen235b" ]]; then
    walltime="${WALLTIME_235B}"
  fi

  local max_new_tokens_label
  max_new_tokens_label="$(effective_max_new_tokens_label "${model}")"
  local method_tag="${method_label}"
  if [[ "${enable_vllm_specdec}" == "true" && "${spec_tokens}" != "0" ]]; then
    method_tag="${method_label}k${spec_tokens}"
  fi
  local job_tag="${model}-perfcfg-${mode}-${method_tag}-step${MAX_STEPS}-osl${max_new_tokens_label}-temp${GENERATION_TEMPERATURE}${JOB_TAG_SUFFIX:-}"
  local cache_suffix="${RUN_ID}_${model}_${mode}_${method_label}"
  local job_node_local_cache_root="${NODE_LOCAL_CACHE_ROOT}/${cache_suffix}"
  local base_log_dir="${LOG_ROOT}/${model}_${mode}_${method_label}"
  local checkpoint_dir="${RUN_WORK_ROOT}/checkpoints/${model}_${mode}_${method_label}"
  local job_torch_extensions_dir="${job_node_local_cache_root}/torch_extensions"
  local job_xdg_cache_home="${job_node_local_cache_root}/xdg"
  local job_vllm_cache_root="${job_node_local_cache_root}/vllm"
  local job_flashinfer_cache_dir="${job_node_local_cache_root}/flashinfer"
  local job_triton_cache_dir="${job_node_local_cache_root}/triton"
  local job_torchinductor_cache_dir="${job_node_local_cache_root}/torchinductor"
  local job_cuda_cache_path="${job_node_local_cache_root}/cuda"
  local job_pythonpycacheprefix="${job_node_local_cache_root}/pycache"
  local job_megatron_dataset_helpers_build_dir="${job_node_local_cache_root}/megatron_dataset_helpers"
  local job_pip_cache_dir="${PIP_CACHE_DIR}/${cache_suffix}"
  local job_system_pydeps_site="${SYSTEM_PYDEPS_SITE}"
  local common_overrides
  local cuda_device_max_connections="${CUDA_DEVICE_MAX_CONNECTIONS}"
  if [[ "${model}" == "qwen235b" && -z "${cuda_device_max_connections}" ]]; then
    cuda_device_max_connections="${QWEN235B_CUDA_DEVICE_MAX_CONNECTIONS}"
  fi
  local vllm_ray_extra_env_vars=""
  if [[ -n "${cuda_device_max_connections}" ]]; then
    vllm_ray_extra_env_vars="CUDA_DEVICE_MAX_CONNECTIONS"
  fi
  common_overrides="logger.log_dir=${base_log_dir}/nemo_logs \
policy.generation.temperature=${GENERATION_TEMPERATURE} \
policy.generation.top_p=${GENERATION_TOP_P} \
policy.generation.top_k=${GENERATION_TOP_K}"
  if [[ "${USE_RECIPE_MAX_NEW_TOKENS}" != "true" ]]; then
    common_overrides="${common_overrides} policy.generation.max_new_tokens=${MAX_NEW_TOKENS}"
  fi
  [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]] && common_overrides="${common_overrides} policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
  [[ -n "${VLLM_USE_TQDM}" ]] && common_overrides="${common_overrides} policy.generation.vllm_cfg.use_tqdm=${VLLM_USE_TQDM}"
  [[ -n "${VLLM_ENABLE_METRICS_LOGGER}" ]] && common_overrides="${common_overrides} policy.generation.vllm_cfg.enable_vllm_metrics_logger=${VLLM_ENABLE_METRICS_LOGGER}"
  [[ -n "${VLLM_ENABLE_PREFIX_CACHING}" ]] && common_overrides="${common_overrides} ++policy.generation.vllm_cfg.enable_prefix_caching=${VLLM_ENABLE_PREFIX_CACHING}"
  [[ -n "${VLLM_ENABLE_CHUNKED_PREFILL}" ]] && common_overrides="${common_overrides} ++policy.generation.vllm_kwargs.enable_chunked_prefill=${VLLM_ENABLE_CHUNKED_PREFILL}"
  [[ -n "${VLLM_MOE_BACKEND}" ]] && common_overrides="${common_overrides} ++policy.generation.vllm_kwargs.kernel_config.moe_backend=${VLLM_MOE_BACKEND}"
  [[ -n "${VLLM_MAX_NUM_SEQS}" ]] && common_overrides="${common_overrides} ++policy.generation.vllm_kwargs.max_num_seqs=${VLLM_MAX_NUM_SEQS}"
  [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]] && common_overrides="${common_overrides} ++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
  [[ -n "${GENERATION_BATCH_SIZE}" ]] && common_overrides="${common_overrides} policy.generation_batch_size=${GENERATION_BATCH_SIZE}"
  [[ -n "${ENV_MATH_NUM_WORKERS}" ]] && common_overrides="${common_overrides} env.math.num_workers=${ENV_MATH_NUM_WORKERS}"
  [[ -n "${DATA_NUM_WORKERS}" ]] && common_overrides="${common_overrides} data.num_workers=${DATA_NUM_WORKERS}"
  [[ -n "${LOGGER_MONITOR_GPUS}" ]] && common_overrides="${common_overrides} logger.monitor_gpus=${LOGGER_MONITOR_GPUS}"
  [[ -n "${EXTRA_OVERRIDES}" ]] && common_overrides="${common_overrides} ${EXTRA_OVERRIDES}"
  if [[ "${model}" == "qwen235b" ]]; then
    [[ -n "${QWEN235B_HYBRIDEP_NUM_RANKS_PER_NVLINK_DOMAIN}" ]] && common_overrides="${common_overrides} ++policy.megatron_cfg.hybridep_num_ranks_per_nvlink_domain=${QWEN235B_HYBRIDEP_NUM_RANKS_PER_NVLINK_DOMAIN}"
    [[ -n "${QWEN235B_HYBRIDEP_USE_MNNVL}" ]] && common_overrides="${common_overrides} ++policy.megatron_cfg.hybridep_use_mnnvl=${QWEN235B_HYBRIDEP_USE_MNNVL}"
    [[ -n "${QWEN235B_MOE_TOKEN_DISPATCHER_TYPE}" ]] && common_overrides="${common_overrides} policy.megatron_cfg.moe_token_dispatcher_type=${QWEN235B_MOE_TOKEN_DISPATCHER_TYPE}"
    if [[ -n "${QWEN235B_MOE_FLEX_DISPATCHER_BACKEND}" ]]; then
      common_overrides="${common_overrides} policy.megatron_cfg.moe_flex_dispatcher_backend=${QWEN235B_MOE_FLEX_DISPATCHER_BACKEND}"
    fi
    [[ -n "${QWEN235B_SEQUENCE_PACKING_FUSE_LOSS}" ]] && common_overrides="${common_overrides} policy.sequence_packing.fuse_loss=${QWEN235B_SEQUENCE_PACKING_FUSE_LOSS}"
    [[ -n "${cuda_device_max_connections}" ]] && common_overrides="${common_overrides} ++policy.megatron_cfg.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\\\"${cuda_device_max_connections}\\\""
  fi

  local wandb_name="${model_label}_PerfCfg_${mode}_${method_label}_${RUN_ID}"
  local remote_cmd
  remote_cmd="$(cat <<EOF
set -euo pipefail
cd '${REMOTE_REPO}'
mkdir -p '${base_log_dir}'
CONTAINER='${CONTAINER}' \\
HF_HOME='${HF_HOME}' \\
HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \\
WANDB_PROJECT='${WANDB_PROJECT}' \\
WANDB_ENABLED='${WANDB_ENABLED}' \\
WANDB_API_KEY='${WANDB_API_KEY:-}' \\
WANDB_NETRC_HOME='${WANDB_NETRC_HOME}' \\
BASE_LOG_DIR='${base_log_dir}' \\
ACCOUNT='${ACCOUNT}' \\
PARTITION='${PARTITION}' \\
NEMO_RL_DIR='${REMOTE_REPO}' \\
PYTHONPATH='${job_system_pydeps_site}' \\
SYSTEM_PYDEPS_SITE='${job_system_pydeps_site}' \\
MODEL_LABEL='${model_label}-perfcfg-${mode}-${method_label}' \\
CONFIG_FILE='${config}' \\
TARGET_MODEL_ID='${target_model}' \\
TOKENIZER_NAME='${tokenizer}' \\
DRAFT_MODEL='${draft_model}' \\
NUM_NODES='${num_nodes}' \\
GPUS_PER_NODE='${gpus_per_node}' \\
SEGMENT='${segment}' \\
GRES_FLAG='${LAUNCHER_GRES_FLAG}' \\
CPUS_PER_WORKER=144 \\
SBATCH_RESOURCE_ARGS='--ntasks-per-node=1 --cpus-per-task=144 --mem=0' \\
SBATCH_EXTRA_ARGS='${SBATCH_EXTRA_ARGS}' \\
MOUNTS='/lustre:/lustre,/project:/project' \\
NUM_PROMPTS='${num_prompts}' \\
NUM_GENERATIONS='${num_generations}' \\
TRAIN_GLOBAL_BATCH_SIZE='${train_global_batch_size}' \\
MAX_STEPS='${MAX_STEPS}' \\
WALLTIME='${walltime}' \\
UV_PYTHON='${UV_PYTHON}' \\
RAY_VERSION='${RAY_VERSION}' \\
RAY_PYTHON_VERSION='${RAY_PYTHON_VERSION}' \\
RAY_PYTHON_SPEC='${RAY_PYTHON_SPEC}' \\
RAY_USE_EXISTING_ENV='${RAY_USE_EXISTING_ENV}' \\
USE_SYSTEM_ENV='${USE_SYSTEM_ENV}' \\
NEMO_RL_VENV_DIR='${NEMO_RL_VENV_DIR}' \\
NEMO_RL_PY_EXECUTABLES_SYSTEM='${NEMO_RL_PY_EXECUTABLES_SYSTEM}' \\
NEMO_RL_MCORE_PY_EXECUTABLES_SYSTEM='${NEMO_RL_MCORE_PY_EXECUTABLES_SYSTEM}' \\
RUN_CACHE_ROOT='${RUN_CACHE_ROOT}' \\
NODE_LOCAL_CACHE_ROOT='${job_node_local_cache_root}' \\
UV_CACHE_DIR_OVERRIDE='${UV_CACHE_DIR_OVERRIDE}' \\
PIP_CACHE_DIR='${job_pip_cache_dir}' \\
TORCH_EXTENSIONS_DIR='${job_torch_extensions_dir}' \\
XDG_CACHE_HOME='${job_xdg_cache_home}' \\
VLLM_CACHE_ROOT='${job_vllm_cache_root}' \\
FLASHINFER_CACHE_DIR='${job_flashinfer_cache_dir}' \\
TRITON_CACHE_DIR='${job_triton_cache_dir}' \\
TORCHINDUCTOR_CACHE_DIR='${job_torchinductor_cache_dir}' \\
CUDA_CACHE_PATH='${job_cuda_cache_path}' \\
TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' \\
CUDA_ARCH_LIST='${CUDA_ARCH_LIST}' \\
CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}' \\
CUDAARCHS='${CUDAARCHS}' \\
PYTHONPYCACHEPREFIX='${job_pythonpycacheprefix}' \\
PYTHONDONTWRITEBYTECODE='${PYTHONDONTWRITEBYTECODE}' \\
MEGATRON_DATASET_HELPERS_BUILD_DIR='${job_megatron_dataset_helpers_build_dir}' \\
RAY_CGRAPH_GET_TIMEOUT=7200 \\
NRL_FORCE_REBUILD_VENVS='${NRL_FORCE_REBUILD_VENVS}' \\
NRL_FORCE_REBUILD_ACTOR_VENVS='${NRL_FORCE_REBUILD_ACTOR_VENVS}' \\
NRL_ACTOR_VENV_CACHE_SUFFIX='${cache_suffix}' \\
NRL_ACTOR_UV_LOCK_MODE='${NRL_ACTOR_UV_LOCK_MODE}' \\
NRL_SERIALIZE_ACTOR_VENV_CREATION='${NRL_SERIALIZE_ACTOR_VENV_CREATION}' \\
DRIVER_UV_PROJECT_ENVIRONMENT='${DRIVER_UV_PROJECT_ENVIRONMENT}' \\
NRL_MEGATRON_CHECKPOINT_DIR='${checkpoint_dir}' \\
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=3600 \\
NCCL_COLLNET_ENABLE='${NCCL_COLLNET_ENABLE:-1}' \\
NRL_MEGATRON_TOKENIZER_MODEL='${tokenizer}' \\
CUDA_DEVICE_MAX_CONNECTIONS='${cuda_device_max_connections}' \\
FORCE_RECONVERT_FROM_HF=false \\
MAX_JOBS=4 \\
CMAKE_BUILD_PARALLEL_LEVEL=4 \\
NVTE_BUILD_MAX_JOBS=4 \\
NINJAFLAGS=-j4 \\
MAKEFLAGS=-j4 \\
DRIVER_SRUN_CPUS_PER_TASK=8 \\
DRIVER_SRUN_MEM=128G \\
SOURCE_VLLM_SITE='${source_vllm_site}' \\
PARD2_OFFICIAL_VLLM_PATCH_DIR='${PARD2_OFFICIAL_VLLM_PATCH_DIR}' \\
PARD2_OFFICIAL_VLLM_SITE='${PARD2_OFFICIAL_VLLM_SITE}' \\
PARD2_VLLM_PATCH_PYTHON='${PARD2_VLLM_PATCH_PYTHON}' \\
PARD2_VLLM_SOURCE_SITE='${PARD2_VLLM_SOURCE_SITE}' \\
VLLM_PRECISION=bfloat16 \\
VLLM_KV_CACHE_DTYPE='${VLLM_KV_CACHE_DTYPE}' \\
VLLM_ATTENTION_BACKEND=TRITON_ATTN \\
VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='${vllm_ray_extra_env_vars}' \\
VLLM_ENFORCE_EAGER='${VLLM_ENFORCE_EAGER}' \\
PRESERVE_RECIPE_ASYNC=true \\
PRESERVE_RECIPE_SEQUENCE_PACKING=true \\
NRL_ALLOW_SPECDEC_LOGPROB_REPAIR_WITH_SAMPLER_MISMATCH=true \\
NRL_VLLM_DISABLE_LOG_STATS=false \\
NRL_VLLM_OMIT_GENERATION_LOGPROBS='${omit_generation_logprobs}' \\
NRL_VLLM_GENERATION_MIN_TOKENS='${MIN_TOKENS}' \\
NRL_VLLM_GENERATION_IGNORE_EOS='${NRL_VLLM_GENERATION_IGNORE_EOS}' \\
NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS='${NRL_VLLM_GENERATION_DISABLE_STOP_STRINGS}' \\
NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS='${NRL_VLLM_GENERATION_DISABLE_STOP_TOKEN_IDS}' \\
DRAFT_FORMAT='${draft_format}' \\
POLICY_DRAFT_ENABLED=false \\
ENABLE_VLLM_SPECDEC='${enable_vllm_specdec}' \\
SPECDEC_METHOD='${specdec_method}' \\
SPECDEC_PARALLEL_DRAFTING='${parallel_drafting}' \\
INCLUDE_DRAFT_TP='${include_draft_tp}' \\
DRAFT_TP='${draft_tp}' \\
GATE_MODE=always \\
NUM_SPECULATIVE_TOKENS='${spec_tokens}' \\
SUFFIX_DECODING_MAX_TREE_DEPTH='${SUFFIX_DECODING_MAX_TREE_DEPTH}' \\
SUFFIX_DECODING_MAX_CACHED_REQUESTS='${SUFFIX_DECODING_MAX_CACHED_REQUESTS}' \\
SUFFIX_DECODING_MAX_SPEC_FACTOR='${SUFFIX_DECODING_MAX_SPEC_FACTOR}' \\
SUFFIX_DECODING_MIN_TOKEN_PROB='${SUFFIX_DECODING_MIN_TOKEN_PROB}' \\
ONLINE_EXTRA_OVERRIDES='${common_overrides}' \\
JOB_TAG='${job_tag}' \\
WANDB_NAME='${wandb_name}' \\
DRY_RUN='${dry_run}' \\
bash '${REMOTE_REPO}/experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh'
EOF
)"

  local output job_id
  output="$(run_remote_cmd "${remote_cmd}" 2>&1)"
  if [[ "${dry_run}" == "true" ]]; then
    printf "%s\n" "${output}" | sed "s/^/[${model}_${mode}_${method_label} dry-run] /" >&2
    printf "DRY_RUN"
    return
  fi

  job_id="$(printf "%s\n" "${output}" | sed -n 's/^Submitted batch job //p' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    printf "%s\n" "${output}" >&2
    exit 1
  fi
  printf "%s" "${job_id}"
}

dry_run=true
if is_true "${SUBMIT}"; then
  dry_run=false
fi

stage_helper
remote_preflight >&2

{
  echo "job_id,model,mode,method,max_steps,max_new_tokens,min_tokens,temperature,top_p,top_k,enforce_eager,vllm_enable_prefix_caching,vllm_moe_backend,vllm_max_num_seqs,vllm_max_num_batched_tokens,target_model,draft_model,num_speculative_tokens,num_nodes,gpus_per_node,segment,gres_flag,use_system_env,ray_use_existing_env,system_pydeps_site,config,run_id,remote_repo,container,log_dir,dependency,wandb_enabled,wandb_project,wandb_name,wandb_url"
  prev_job_id=""
  for model in ${MODELS}; do
    for mode in ${MODES}; do
      for method in ${METHODS}; do
        dependency="${INITIAL_DEPENDENCY}"
        if is_true "${SERIALIZE_SUBMISSIONS}" && [[ -n "${prev_job_id}" ]]; then
          if [[ -n "${dependency}" ]]; then
            dependency="${dependency},afterany:${prev_job_id}"
          else
            dependency="afterany:${prev_job_id}"
          fi
        fi
        SBATCH_EXTRA_ARGS="${BASE_SBATCH_EXTRA_ARGS}"
        if [[ -n "${dependency}" ]]; then
          SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS} --dependency=${dependency}"
        fi
        model_base "${model}"
        mode_shape "${model}" "${mode}"
        method_contract "${method}"
        job_id="$(submit_one "${model}" "${mode}" "${method}" "${dry_run}")"
        if is_true "${SERIALIZE_SUBMISSIONS}" && [[ "${job_id}" != "DRY_RUN" ]]; then
          prev_job_id="${job_id}"
        fi
        max_new_tokens_label="$(effective_max_new_tokens_label "${model}")"
        min_tokens_label="${MIN_TOKENS:-default}"
        csv_system_pydeps_site="${SYSTEM_PYDEPS_SITE}"
        wandb_name="${model_label}_PerfCfg_${mode}_${method_label}_${RUN_ID}"
        echo "${job_id},${model},${mode},${method_label},${MAX_STEPS},${max_new_tokens_label},${min_tokens_label},${GENERATION_TEMPERATURE},${GENERATION_TOP_P},${GENERATION_TOP_K},${VLLM_ENFORCE_EAGER},${VLLM_ENABLE_PREFIX_CACHING},${VLLM_MOE_BACKEND},${VLLM_MAX_NUM_SEQS},${VLLM_MAX_NUM_BATCHED_TOKENS},${target_model},${draft_model},${spec_tokens},${num_nodes},${gpus_per_node},${segment},${LAUNCHER_GRES_FLAG:-none},${USE_SYSTEM_ENV},${RAY_USE_EXISTING_ENV},${csv_system_pydeps_site},${config},${RUN_ID},${REMOTE_REPO},${CONTAINER},${LOG_ROOT}/${model}_${mode}_${method_label},${dependency},${WANDB_ENABLED},${WANDB_PROJECT},${wandb_name},"
      done
    done
  done
} | tee "${OUT}"
