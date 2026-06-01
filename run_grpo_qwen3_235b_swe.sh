#!/bin/bash
# ============================================================================
# NeMo-RL Async GRPO SWE RL Training: Qwen3-235B-A22B
#
# Model:      Qwen3-235B-A22B (MoE 235B total / 22B active)
# Train data: R2E-Gym (r2e-gym subset, 4518 samples)
# Eval data:  SWE-bench Verified
# Mode:       Async GRPO with non-colocated generation
# Env:        swe_agents (OpenHands agent + singularity sandbox)
#
# Parallelism (from examples grpo-qwen3-235b-32n8g-async-1off):
#   TP=4, ETP=1, EP=16, CP=1, PP=8, vLLM_TP=8
#   num_layers_in_first/last_pipeline_stage=11 (94 layers total)
#   16 actor nodes + 8 generation nodes (same as 30B script)
#
# Key diffs from Qwen3-30B-A3B-Thinking script:
#   - 235B model: 94 layers, 128 experts, hidden_size=4096
#   - seq_len=16384 (16k), max_tool_calls=20
#   - TP=4, ETP=1, EP=16, CP=1, PP=8 (vs TP=2, EP=8, CP=4, PP=2)
#   - vLLM_TP=8 (vs 2)
#   - 16 actor nodes + 8 gen nodes (same as 30B script)
#
# Usage:
#   bash run_grpo_qwen3_235b_swe.sh
#
# Override:
#   NUM_GPU=4 NUM_NODES=32 NUM_GEN_NODES=16 bash run_grpo_qwen3_235b_swe.sh
#   MODEL_PATH=/path/to/checkpoint bash run_grpo_qwen3_235b_swe.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SPECDEC_RL_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
LEGACY_SWE_REPO_ROOT="/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe"
USE_LEGACY_SWE_REPO_ROOT="${USE_LEGACY_SWE_REPO_ROOT:-false}"
REQUIRE_SPECDEC_RL_PATCHES="${REQUIRE_SPECDEC_RL_PATCHES:-true}"
if [ -d "${DEFAULT_SPECDEC_RL_DIR}" ]; then
  DEFAULT_REPO_ROOT="${DEFAULT_SPECDEC_RL_DIR}"
elif [[ "${USE_LEGACY_SWE_REPO_ROOT}" == "true" || "${USE_LEGACY_SWE_REPO_ROOT}" == "True" ]] && [ -d "${LEGACY_SWE_REPO_ROOT}" ]; then
  DEFAULT_REPO_ROOT="${LEGACY_SWE_REPO_ROOT}"
else
  DEFAULT_REPO_ROOT="${SCRIPT_DIR}"
fi

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ "${REQUIRE_SPECDEC_RL_PATCHES}" == "true" || "${REQUIRE_SPECDEC_RL_PATCHES}" == "True" ]]; then
  if [[ "${REPO_ROOT}" == "${LEGACY_SWE_REPO_ROOT}" ]]; then
    echo "ERROR: REPO_ROOT points at the legacy SWE repo, which may not contain the current SpecDec-RL patches." >&2
    echo "Set REPO_ROOT=${DEFAULT_SPECDEC_RL_DIR}, or set REQUIRE_SPECDEC_RL_PATCHES=false for a non-SpecDec SWE run." >&2
    exit 2
  fi
fi
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/grpo_qwen3_235b_swe.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/results}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env.sh}"
if [ ! -f "${ENV_FILE}" ] && [ -f "${REPO_ROOT}/env.sh" ]; then
  ENV_FILE="${REPO_ROOT}/env.sh"
fi
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
DEFAULT_CHAT_TEMPLATE="${ARTIFACT_ROOT}/templates/qwen3_generation_template.jinja2"
if [ -z "${CHAT_TEMPLATE:-}" ] && [ -f "${DEFAULT_CHAT_TEMPLATE}" ]; then
  CHAT_TEMPLATE="${DEFAULT_CHAT_TEMPLATE}"
fi

# ================ Scaling 实验核心参数 ================
PPS="${PPS:-32}"
GPP="${GPP:-8}"
GBS="${GBS:-256}"
LR="${LR:-1e-06}"
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-200}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"

# ================ Sync/Async 模式选择 ================
ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1

# ================ 根据 Sync/Async 自动配置 ================
NUM_ACTOR_NODES="${NUM_NODES:-16}"
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=False
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=2

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COLOCATED_ENABLED=False
  VLLM_GPU_UTIL=0.8
  NUM_GENERATION_NODES="${NUM_GEN_NODES:-8}"
  OVERLAP_GRAD_REDUCE=False
  ADVANTAGE_CLIP_LOW=-100
  ADVANTAGE_CLIP_HIGH=100
  TIS_THRESHOLD=5
else
  COLOCATED_ENABLED=True
  VLLM_GPU_UTIL=0.5
  OVERLAP_GRAD_REDUCE=True
fi

# ================ 固定参数 ================
SEQLEN=16384
NUM_GPU="${NUM_GPU:-${GPUS_PER_NODE:-8}}"
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((NUM_GPU * 16))}"

TP="${TP:-4}"
ETP="${ETP:-1}"
EP="${EP:-16}"
CP="${CP:-1}"
if [[ -z "${PP+x}" ]]; then
  if [[ "$NUM_GPU" == "4" ]]; then
    PP=4
  else
    PP=8
  fi
fi

VLLM_TP="${VLLM_TP:-8}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
VLLM_COMPILATION_LEVEL="${VLLM_COMPILATION_LEVEL:-}"
VLLM_USE_INDUCTOR="${VLLM_USE_INDUCTOR:-}"

SEQUENCE_PACKING=True
TOKEN_LEVEL_LOSS=True
SEQ_LEVEL_IS=False
NORMALIZE_REWARDS=True
OVERLONG_FILTERING=True

USE_ON_POLICY_KL_APPROXIMATION=True
IMPORTANCE_SAMPLING_CORRECTION=True
KL=0
CLIP_MIN=0.2
CLIP_MAX=0.28
TEMPERATURE=1.0

SAVE_PERIOD="${SAVE_PERIOD:-5}"
VAL_PERIOD="${VAL_PERIOD:-1000}"
KEEP_TOP_K="${KEEP_TOP_K:-2}"

MOE_FREEZE_ROUTER=True
MOE_PERMUTE_FUSION=True
MOE_ENABLE_DEEPEP=False
MOE_TOKEN_DISPATCHER_TYPE="alltoall"
MOE_AUX_LOSS_COEFF=0
MOE_ROUTER_LOAD_BALANCING_TYPE="none"
MOE_ROUTER_BIAS_UPDATE_RATE="1e-3"

if [[ -z "${PP_FIRST_STAGE+x}" ]]; then
  if [[ "$PP" == "4" ]]; then
    PP_FIRST_STAGE=23
  else
    PP_FIRST_STAGE=11
  fi
fi
if [[ -z "${PP_LAST_STAGE+x}" ]]; then
  if [[ "$PP" == "4" ]]; then
    PP_LAST_STAGE=23
  else
    PP_LAST_STAGE=11
  fi
fi

# ================ 数据/模型路径 ================
DEFAULT_TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DEFAULT_TRAIN_DATA_PATH}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$TRAIN_DATA_PATH}"
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-235B-A22B-Thinking-2507"}

validate_jsonl_path() {
  local label="$1"
  local path="$2"
  if [ -z "$path" ]; then
    echo "ERROR: ${label} is empty" >&2
    exit 1
  fi
  if [ ! -f "$path" ]; then
    echo "ERROR: ${label} is not visible: ${path}" >&2
    exit 1
  fi
  if [ ! -s "$path" ]; then
    echo "ERROR: ${label} is empty: ${path}" >&2
    exit 1
  fi
}

validate_jsonl_path "TRAIN_DATA_PATH" "${TRAIN_DATA_PATH}"
validate_jsonl_path "VAL_DATA_PATH" "${VAL_DATA_PATH}"

# ================ 实验命名 ================
WANDB_PROJ="sna-nemo-rl"
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
SMOKE_SUFFIX="${MAX_NUM_STEPS:+-smoke${MAX_NUM_STEPS:-1}step}"
DEFAULT_EXP_SUFFIX="qwen3-235b-a22b-thinking-swe-${SYNC_MODE}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}${SMOKE_SUFFIX}"
EXP_SUFFIX="${EXP_SUFFIX_OVERRIDE:-$DEFAULT_EXP_SUFFIX}"
WANDB_NAME="${WANDB_NAME:-$EXP_SUFFIX}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-$WANDB_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_SUBDIR}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${REPO_ROOT}}"

mkdir -p "${CHECKPOINT_DIR}"

# ================ 环境变量 ================
if [ ! -d "${REPO_ROOT}" ]; then
  echo "ERROR: REPO_ROOT is not visible: ${REPO_ROOT}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: CONFIG_FILE is not visible: ${CONFIG_FILE}" >&2
  exit 1
fi
if [ ! -f "${REPO_ROOT}/ray.sub" ]; then
  echo "ERROR: ray.sub is not visible under REPO_ROOT: ${REPO_ROOT}/ray.sub" >&2
  exit 1
fi
if [ ! -f "${REPO_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py" ]; then
  echo "ERROR: NeMo-Gym GRPO entrypoint is not visible under REPO_ROOT" >&2
  exit 1
fi
if [ -f "${ENV_FILE}" ]; then
  source "${ENV_FILE}"
else
  echo "WARN: ENV_FILE is not visible; relying on caller-provided tokens/env: ${ENV_FILE}" >&2
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
export UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/sna/uv_cache
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ================ Node-local cache 配置 ================
PERSISTENT_CACHE="/lustre/fsw/portfolios/coreai/users/sna/.cache/qwen3_235b_swe"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY=120

mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-4:0:0}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"
DEFAULT_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
export CONTAINER="${CONTAINER:-$DEFAULT_CONTAINER}"
RUN_UV_SYNC="${RUN_UV_SYNC:-false}"
DRIVER_LAUNCHER="${DRIVER_LAUNCHER:-/opt/venv/bin/python}"
NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-0}"
NEMO_RL_VLLM_EXECUTABLE_SYSTEM="${NEMO_RL_VLLM_EXECUTABLE_SYSTEM:-1}"
NEMO_RL_MCORE_EXECUTABLE_SYSTEM="${NEMO_RL_MCORE_EXECUTABLE_SYSTEM:-1}"
NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM="${NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM:-1}"
INSTALL_VLLM_IN_SYSTEM="${INSTALL_VLLM_IN_SYSTEM:-true}"
SHARED_VLLM_SITE="${SHARED_VLLM_SITE:-${ARTIFACT_ROOT}/python_site/vllm_0_10_2_nodeps_py312}"
SYSTEM_VLLM_BOOTSTRAP="${SYSTEM_VLLM_BOOTSTRAP:-${SCRIPT_DIR}/experiments/eagle3_qwen3_235b/bootstrap_system_vllm_site.sh}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==0.10.2}"
DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR="${SCRIPT_DIR}/experiments/eagle3_qwen3_235b/megatron_bridge_qwen3moe"
MEGATRON_BRIDGE_PLUGIN_DIR="${MEGATRON_BRIDGE_PLUGIN_DIR:-}"
if [ -z "${MEGATRON_BRIDGE_PLUGIN_DIR}" ] && [ -d "${DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR}" ]; then
  MEGATRON_BRIDGE_PLUGIN_DIR="${DEFAULT_MEGATRON_BRIDGE_PLUGIN_DIR}"
fi
MEGATRON_BRIDGE_QWEN3MOE_PLUGIN="${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN:-1}"
MEGATRON_BRIDGE_SRC="${MEGATRON_BRIDGE_SRC:-}"
MEGATRON_LM_SRC="${MEGATRON_LM_SRC:-}"
if [ -n "${MEGATRON_BRIDGE_PLUGIN_DIR}" ] && [ ! -d "${MEGATRON_BRIDGE_PLUGIN_DIR}" ]; then
  echo "ERROR: MEGATRON_BRIDGE_PLUGIN_DIR is set but not visible: ${MEGATRON_BRIDGE_PLUGIN_DIR}" >&2
  exit 1
fi
if [ -n "${MEGATRON_BRIDGE_SRC}" ] && [ ! -d "${MEGATRON_BRIDGE_SRC}" ]; then
  echo "ERROR: MEGATRON_BRIDGE_SRC is set but not visible: ${MEGATRON_BRIDGE_SRC}" >&2
  exit 1
fi
if [ -n "${MEGATRON_LM_SRC}" ] && [ ! -d "${MEGATRON_LM_SRC}" ]; then
  echo "ERROR: MEGATRON_LM_SRC is set but not visible: ${MEGATRON_LM_SRC}" >&2
  exit 1
fi
MEGATRON_EXTRA_PYTHONPATH=""
if [ -n "${MEGATRON_BRIDGE_PLUGIN_DIR}" ]; then
  MEGATRON_EXTRA_PYTHONPATH="${MEGATRON_EXTRA_PYTHONPATH}${MEGATRON_BRIDGE_PLUGIN_DIR}:"
fi
if [ -n "${MEGATRON_BRIDGE_SRC}" ]; then
  MEGATRON_EXTRA_PYTHONPATH="${MEGATRON_EXTRA_PYTHONPATH}${MEGATRON_BRIDGE_SRC}:"
fi
if [ -n "${MEGATRON_LM_SRC}" ]; then
  MEGATRON_EXTRA_PYTHONPATH="${MEGATRON_EXTRA_PYTHONPATH}${MEGATRON_LM_SRC}:"
fi
if [ ! -f "${CONTAINER}" ] && [[ "${DRY_RUN:-false}" != "true" && "${DRY_RUN:-false}" != "True" ]]; then
  echo "ERROR: CONTAINER is not visible: ${CONTAINER}" >&2
  exit 1
fi
if [ ! -f "${SYSTEM_VLLM_BOOTSTRAP}" ]; then
  echo "ERROR: SYSTEM_VLLM_BOOTSTRAP is not visible: ${SYSTEM_VLLM_BOOTSTRAP}" >&2
  exit 1
fi

echo "=========================================="
echo "Experiment: ${EXP_SUFFIX}"
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG_FILE}"
echo "Env file: ${ENV_FILE}"
echo "Chat template: ${CHAT_TEMPLATE:-<config-default>}"
echo "Container: ${CONTAINER}"
echo "Mode: ${SYNC_MODE}, Colocated: ${COLOCATED_ENABLED}"
echo "Nodes: ${NUM_ACTOR_NODES}, GPUs/node: ${NUM_GPU}"
echo "Ray worker CPUs: ${CPUS_PER_WORKER}"
echo "Parallelism: TP=${TP}, ETP=${ETP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}"
echo "PP stages: first=${PP_FIRST_STAGE}, last=${PP_LAST_STAGE}"
echo "Training: PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, LR=${LR}"
echo "SeqLen: ${SEQLEN}"
echo "Agent: max_turns=${AGENT_MAX_TURNS}, timeout=${AGENT_TIMEOUT}s"
echo "Model: ${MODEL_PATH}"
echo "Train data: ${TRAIN_DATA_PATH}"
echo "Val data: ${VAL_DATA_PATH}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Slurm dependency: ${SBATCH_DEPENDENCY:-singleton}"
echo "Slurm exclude: ${SBATCH_EXCLUDE:-<none>}"
echo "Driver launcher: ${DRIVER_LAUNCHER}"
echo "Run uv sync: ${RUN_UV_SYNC}"
echo "Use system actor executables: ${NEMO_RL_PY_EXECUTABLES_SYSTEM}"
echo "Use system vLLM executable: ${NEMO_RL_VLLM_EXECUTABLE_SYSTEM}"
echo "Use system MCore executable: ${NEMO_RL_MCORE_EXECUTABLE_SYSTEM}"
echo "Use system NemoGym executable: ${NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM}"
echo "Install vLLM in system env: ${INSTALL_VLLM_IN_SYSTEM}"
echo "Shared vLLM site: ${SHARED_VLLM_SITE}"
echo "vLLM pip spec: ${VLLM_PIP_SPEC}"
echo "Megatron bridge plugin: ${MEGATRON_BRIDGE_PLUGIN_DIR:-<disabled>}"
echo "Qwen3MoE bridge plugin enabled: ${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN}"
echo "Megatron-Bridge src: ${MEGATRON_BRIDGE_SRC:-<container-default>}"
echo "Megatron-LM src: ${MEGATRON_LM_SRC:-<container-default>}"
echo "Dry run: ${DRY_RUN:-false}"
echo "=========================================="

cd "${SNAPSHOT_DIR}"

# ================ SETUP_COMMAND ================
VLLM_WHEEL="${VLLM_WHEEL_LOCATION:-}"

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[SETUP] Installing apptainer for SWE sandbox..."
apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
RET=1
RETRIES=3
for attempt in \$(seq 1 \$RETRIES); do
  if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
    echo "[SETUP] singularity/apptainer already available"
    RET=0
    break
  fi
  cd /tmp && \
  wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
  apt install -y ./apptainer_1.3.1_amd64.deb && \
  ln -sf /usr/bin/apptainer /usr/bin/singularity
  if command -v apptainer >/dev/null 2>&1; then
    echo "[SETUP] apptainer installed successfully"
    RET=0
    break
  fi
  echo "[SETUP] apptainer install attempt \$attempt failed, retrying..."
  sleep 10
done
if [ \$RET -ne 0 ]; then
  echo "[SETUP] WARNING: apptainer installation failed after \$RETRIES attempts"
fi

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

find "${LUSTRE_INDUCTOR_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find "${LUSTRE_TRITON_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true

_seed_cache() {
  local lustre="\$1" local_dir="\$2" name="\$3"
  if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from Lustre" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

_seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
_seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"

_found_warm=""
if [ -n "${LUSTRE_VLLM_CACHE}" ]; then
  _base="\$(basename "${LUSTRE_VLLM_CACHE}")"
  _parent="\$(dirname "${LUSTRE_VLLM_CACHE}")"
  _found_warm="\$(
    ls -1dt "\${_parent}/\${_base}_"* 2>/dev/null \
      | while IFS= read -r d; do
          [ -d "\$d" ] && [ "\$(ls -A "\$d" 2>/dev/null)" ] && echo "\$d" && break
        done
  )"
fi
if [ -n "\$_found_warm" ]; then
  rm -rf "${NRL_VLLM_CACHE_SEED_DIR}"
  _seed_cache "\$_found_warm" "${NRL_VLLM_CACHE_SEED_DIR}" "vLLM (from \$(basename "\$_found_warm"))"
else
  echo "[CACHE SEED] vLLM: no warm cache on Lustre yet"
  rm -rf "${NRL_VLLM_CACHE_SEED_DIR}"
fi
echo "[CACHE SEED] Done."

if [[ "${RUN_UV_SYNC}" == "true" || "${RUN_UV_SYNC}" == "True" ]]; then
  if [[ -n "${VLLM_WHEEL}" ]]; then
    VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_WHEEL} \
      UV_HTTP_TIMEOUT=3600 \
      uv sync --frozen
  else
    UV_HTTP_TIMEOUT=3600 uv sync --frozen
  fi
else
  echo "[SETUP] Skipping uv sync; using container preinstalled /opt/venv to avoid torch/native-library skew."
fi
if [[ "${INSTALL_VLLM_IN_SYSTEM}" == "true" || "${INSTALL_VLLM_IN_SYSTEM}" == "True" ]]; then
  echo "[SETUP] Skipping /opt/venv vLLM mutation; shared PYTHONPATH bootstrap runs in the driver command."
fi
SETUPEOF
export SETUP_COMMAND

# ================ 训练命令 ================
export COMMAND="export ARTIFACT_ROOT=${ARTIFACT_ROOT} INSTALL_VLLM_IN_SYSTEM=${INSTALL_VLLM_IN_SYSTEM} VLLM_WHEEL_LOCATION=${VLLM_WHEEL} VLLM_PIP_SPEC=${VLLM_PIP_SPEC} SHARED_VLLM_SITE=${SHARED_VLLM_SITE}; \
  . ${SYSTEM_VLLM_BOOTSTRAP} && \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  MEGATRON_BRIDGE_QWEN3MOE_PLUGIN=${MEGATRON_BRIDGE_QWEN3MOE_PLUGIN} \
  PYTHONPATH=${MEGATRON_EXTRA_PYTHONPATH}${REPO_ROOT}:\${PYTHONPATH:-} \
  NEMO_RL_PY_EXECUTABLES_SYSTEM=${NEMO_RL_PY_EXECUTABLES_SYSTEM} \
  NEMO_RL_VLLM_EXECUTABLE_SYSTEM=${NEMO_RL_VLLM_EXECUTABLE_SYSTEM} \
  NEMO_RL_MCORE_EXECUTABLE_SYSTEM=${NEMO_RL_MCORE_EXECUTABLE_SYSTEM} \
  NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM=${NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM} \
  WANDB_API_KEY=\${WANDB_API_KEY:-} \
  HUGGINGFACE_TOKEN=\${HUGGINGFACE_TOKEN:-} \
  GITHUB_TOKEN=\${GITHUB_TOKEN:-} \
  GITLAB_TOKEN=\${GITLAB_TOKEN:-} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
  DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=true \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  ${DRIVER_LAUNCHER} ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
  grpo.normalize_rewards=${NORMALIZE_REWARDS} \
  grpo.overlong_filtering=${OVERLONG_FILTERING} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
  grpo.async_grpo.enabled=${ASYNC_GRPO_ENABLED} \
  grpo.async_grpo.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
  policy.generation.colocated.enabled=${COLOCATED_ENABLED} \
  policy.model_name=${MODEL_PATH} \
  policy.max_total_sequence_length=${SEQLEN} \
  policy.dynamic_batching.enabled=False \
  policy.train_global_batch_size=${GBS} \
  policy.make_sequence_length_divisible_by=8 \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.num_layers_in_first_pipeline_stage=${PP_FIRST_STAGE} \
  policy.megatron_cfg.num_layers_in_last_pipeline_stage=${PP_LAST_STAGE} \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.bias_activation_fusion=False \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} \
  policy.megatron_cfg.moe_permute_fusion=${MOE_PERMUTE_FUSION} \
  policy.megatron_cfg.moe_enable_deepep=${MOE_ENABLE_DEEPEP} \
  policy.megatron_cfg.moe_token_dispatcher_type=${MOE_TOKEN_DISPATCHER_TYPE} \
  policy.megatron_cfg.moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
  policy.megatron_cfg.moe_router_load_balancing_type=${MOE_ROUTER_LOAD_BALANCING_TYPE} \
  policy.megatron_cfg.moe_router_bias_update_rate=${MOE_ROUTER_BIAS_UPDATE_RATE} \
  policy.megatron_cfg.freeze_moe_router=${MOE_FREEZE_ROUTER} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${LR} \
  policy.megatron_cfg.optimizer.weight_decay=0 \
  policy.megatron_cfg.activation_checkpointing=True \
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
  loss_fn.reference_policy_kl_penalty=${KL} \
  loss_fn.ratio_clip_min=${CLIP_MIN} \
  loss_fn.ratio_clip_max=${CLIP_MAX} \
  loss_fn.use_on_policy_kl_approximation=${USE_ON_POLICY_KL_APPROXIMATION} \
  loss_fn.use_importance_sampling_correction=${IMPORTANCE_SAMPLING_CORRECTION} \
  loss_fn.sequence_level_importance_ratios=${SEQ_LEVEL_IS} \
  loss_fn.token_level_loss=${TOKEN_LEVEL_LOSS} \
  loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  ++checkpointing.metric_name=train:total_reward/mean \
  ++checkpointing.checkpoint_must_save_by=00:03:35:00 \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  grpo.max_num_steps=${MAX_NUM_STEPS:-1}"

if [ -n "${VLLM_ENFORCE_EAGER}" ]; then
  export COMMAND="${COMMAND} policy.generation.vllm_cfg.enforce_eager=${VLLM_ENFORCE_EAGER}"
fi
if [ -n "${VLLM_COMPILATION_LEVEL}" ]; then
  export COMMAND="${COMMAND} +policy.generation.vllm_kwargs.compilation_config.level=${VLLM_COMPILATION_LEVEL}"
fi
if [ -n "${VLLM_USE_INDUCTOR}" ]; then
  export COMMAND="${COMMAND} +policy.generation.vllm_kwargs.compilation_config.use_inductor=${VLLM_USE_INDUCTOR}"
fi

if [ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]; then
  export COMMAND="${COMMAND} ${EXTRA_HYDRA_OVERRIDES}"
fi

if [ -n "${CHAT_TEMPLATE:-}" ]; then
  if [ ! -f "${CHAT_TEMPLATE}" ]; then
    echo "ERROR: CHAT_TEMPLATE is set but not visible: ${CHAT_TEMPLATE}" >&2
    exit 1
  fi
  export COMMAND="${COMMAND} policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=${CHAT_TEMPLATE}"
fi

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  export COMMAND="${COMMAND} \
  policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
  grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT}"
fi

# ================ 容器和挂载配置 ================
GYM_CODE="${GYM_CODE:-${REPO_ROOT}/3rdparty/Gym-workspace/Gym}"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ================ 提交任务 ================
SBATCH_CMD=(
  sbatch
  --nodes="${NUM_ACTOR_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --job-name="${WANDB_NAME}"
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --gres=gpu:${NUM_GPU}
  --exclusive
  --mem="${SBATCH_MEM:-0}"
  --dependency="${SBATCH_DEPENDENCY:-singleton}"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}'
)
if [[ -n "$SBATCH_EXCLUDE" ]]; then
  SBATCH_CMD+=(--exclude="$SBATCH_EXCLUDE")
fi
SBATCH_CMD+=(ray.sub)

redact_command_for_log() {
  sed -E \
    -e 's/(WANDB_API_KEY=)[^[:space:]]+/\1<redacted>/g' \
    -e 's/(HUGGINGFACE_TOKEN=)[^[:space:]]+/\1<redacted>/g' \
    -e 's/(GITHUB_TOKEN=)[^[:space:]]+/\1<redacted>/g' \
    -e 's/(GITLAB_TOKEN=)[^[:space:]]+/\1<redacted>/g'
}

if [[ "${DRY_RUN:-false}" == "true" || "${DRY_RUN:-false}" == "True" ]]; then
  echo "[DRY-RUN] COMMAND:"
  printf '%s\n' "$COMMAND" | redact_command_for_log
  echo "[DRY-RUN] sbatch:"
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

"${SBATCH_CMD[@]}" | tee /dev/stderr | grep -o '[0-9]\+' > latest_235b_swe_job_id.txt

JOB_ID="$(cat latest_235b_swe_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Logs: ${CHECKPOINT_DIR}/"
echo "=========================================="

# ================ 后台监控进程 ================
(
  echo "[$(date)] Waiting for job $JOB_ID to start running..."
  MAX_WAIT_ITERATIONS=100000000
  for i in $(seq 1 $MAX_WAIT_ITERATIONS); do
    if squeue -j $JOB_ID -h -o "%T" 2>/dev/null | grep -q "RUNNING"; then
      echo "[$(date)] Job $JOB_ID is now RUNNING."
      break
    fi
    if ! squeue -j $JOB_ID &>/dev/null; then
      echo "[$(date)] Job $JOB_ID is no longer in queue."
      exit 0
    fi
    sleep 60
  done
  
  LOG_DIR="${SNAPSHOT_DIR}/${JOB_ID}-logs"
  RAY_DRIVER_LOG="${LOG_DIR}/ray-driver.log"
  
  for minute in $(seq 1 30); do
    if ! squeue -j $JOB_ID &>/dev/null; then
      echo "[$(date)] Job $JOB_ID is no longer running."
      exit 0
    fi
    
    if [ -f "$RAY_DRIVER_LOG" ]; then
      if grep -q "AssertionError: Attempting to report device id" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found vLLM initialization error. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "Failed to build.*mamba-ssm" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found mamba-ssm build failure. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "RuntimeError: Engine core initialization failed" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found engine core initialization failure. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "CUDA error: an illegal memory access was encountered" "$RAY_DRIVER_LOG" 2>/dev/null || \
         grep -q "RayTaskError(AcceleratorError)" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Found CUDA illegal memory access / AcceleratorError. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
      if grep -q "ERROR:nemo_rl.utils.venvs:Failed to create venv" "$RAY_DRIVER_LOG" 2>/dev/null; then
        echo "[$(date)] Venv creation failure. Killing job $JOB_ID."
        scancel $JOB_ID
        exit 0
      fi
    fi
    
    if [ $minute -lt 30 ]; then
      sleep 60
    fi
  done
  
  echo "[$(date)] Monitoring complete. Job $JOB_ID appears stable."
) >> "${SNAPSHOT_DIR}/monitor_235b_swe_${JOB_ID}.log" 2>&1 &

MONITOR_PID=$!
echo "Started monitoring process (PID: $MONITOR_PID)"

cd - > /dev/null
