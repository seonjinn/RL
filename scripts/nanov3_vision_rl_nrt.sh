#!/usr/bin/env bash
#
# Parity launcher: runs the DFW nanov3 vision RL workload on the current
# NRT nemo-rl-super codebase, container, and repo vLLM checkout.
#
# Both launchers point at examples/omni/nanov3_vision_rl.yaml with the
# same 4-node scale, same logical model / MMPR-Tiny data, and DFW's vLLM
# engine settings.
#
# Wandb is enabled and forced to the same project as the recipes
# baseline (nemo-rl-omni). JOB_NAME_BASE defaults to a full4h name so
# validation runs are easy to tell apart in the dashboard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SOURCE_NEMORL="$(cd "${SOURCE_NEMORL}" && pwd)"
NEMORL="${SOURCE_NEMORL}"

if [[ -f "${SOURCE_NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SOURCE_NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/omni/nanov3_vision_rl.yaml}"
NUM_NODES="${NUM_NODES:-4}"
SNAPSHOT_CODE="${SNAPSHOT_CODE:-1}"
JOB_NAME_BASE="${JOB_NAME_BASE:-image-grpo-vllm20-nrt-full4h}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE:-}}"
GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-${MAX_STEPS:-}}"
MODEL_NAME="${IMAGE_GRPO_MODEL_NAME:-${MODEL_NAME:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/checkpoints/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-0303/step_400}}"
CACHE_DIR="${IMAGE_GRPO_CACHE_DIR:-${CACHE_DIR:-${SOURCE_NEMORL}/.cache/mmpr_tiny}}"
WANDB_PROJECT="${WANDB_PROJECT:-sna-nemotron-omni-dynamiccp}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
RESULTS_ROOT="${RESULTS_ROOT:-${SOURCE_NEMORL}/../jobs}"
RESULTS_DIR="${RESULTS_ROOT}/${JOB_NAME}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${LOGS_DIR}}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-llmservice_fm_vision}"
# Full validation run default. Override SBATCH_TIME from the environment for
# shorter probes.
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
if [[ -z "${SBATCH_PARTITION:-}" ]]; then
  if [[ -n "${PARTITION:-}" ]]; then
    SBATCH_PARTITION="${PARTITION}"
  elif [[ "$(hostname)" == *"draco-oci"* ]]; then
    SBATCH_PARTITION="batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4"
  elif [[ "$(hostname)" == *"cw-dfw"* ]]; then
    SBATCH_PARTITION="batch,backfill,batch_short"
  elif [[ "$(hostname)" == *"cs-oci-ord"* ]]; then
    SBATCH_PARTITION="backfill_block1,grizzly,polar,polar3,polar4"
  elif [[ "$(hostname)" == *"oci-nrt"* ]]; then
    SBATCH_PARTITION="batch_block1"
  else
    SBATCH_PARTITION="batch,batch_large,batch_large_long,batch_long"
  fi
fi
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# ray.sub only sees exported vars; without this it falls back to its own
# default and trips the "GPUS_PER_NODE doesn't match cluster GRES" check.
export NUM_NODES

# Container + mounts. Default to the validated super-v3-omni-vllm20 image on
# OCI-NRT. Overridable via .env for other clusters.
CONTAINER_ROOT="${CONTAINER_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/aroshanghias/containers}"
export CONTAINER="${CONTAINER:-${CONTAINER_ROOT}/super-omni-20260527-d58a158.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre,/home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Trust the baked /opt/ray_venvs/<actor>/ in the container so
# create_local_venv() short-circuits and we don't re-resolve nemo-rl
# extras through the private flashinfer-cubin index at runtime.
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
# flashinfer-jit-cache=0.6.5+cu129 vs flashinfer=0.6.9 ships in the
# image; the strict version assert is harmless for this workload.
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

if [[ -n "${NEMO_RL_ISOLATED_CACHE_ROOT:-}" ]]; then
  CACHE_ROOT="${NEMO_RL_ISOLATED_CACHE_ROOT}"
  HF_HOME="${CACHE_ROOT}/huggingface"
  HF_MODULES_CACHE="${HF_HOME}/modules"
  NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_ROOT}/nemo_rl"
fi
export CACHE_ROOT="${CACHE_ROOT:-${SOURCE_NEMORL}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
TMP_RUN_ID="${RUN_ID//[^A-Za-z0-9]/}"
TMP_RUN_ID="${TMP_RUN_ID:0:18}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${TMP_RUN_ID:-run}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export NEMO_RL_TRAIN_STEP_MEM_DIAG="${NEMO_RL_TRAIN_STEP_MEM_DIAG:-1}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1000}"
export NRL_DEBUG="${NRL_DEBUG:-0}"
export NEMO_RL_VLLM_PRECOMPUTED_IMG_SIZES="${NEMO_RL_VLLM_PRECOMPUTED_IMG_SIZES:-0}"
export NEMO_RL_VLLM_DERIVE_MAX_NUM_PATCHES="${NEMO_RL_VLLM_DERIVE_MAX_NUM_PATCHES:-0}"
export NEMO_RL_VLLM_CAP_MAX_TOKENS_TO_CONTEXT="${NEMO_RL_VLLM_CAP_MAX_TOKENS_TO_CONTEXT:-0}"
export MCORE_DISABLE_TORCH_COMPILE_JIT="${MCORE_DISABLE_TORCH_COMPILE_JIT:-false}"
export USE_REPO_VLLM="${USE_REPO_VLLM:-0}"
if [[ "${USE_REPO_VLLM}" == "1" ]]; then
  SOURCE_VLLM_DIR="${SOURCE_VLLM_DIR:-${SOURCE_NEMORL}/3rdparty/vllm}"
  SOURCE_VLLM_DIR="$(cd "${SOURCE_VLLM_DIR}" && pwd)"
else
  SOURCE_VLLM_DIR="container"
fi
# Provide auth credentials for the private flashinfer-cubin gitlab pypi
# index if NRL_VENVS_TRUST_EXISTING is ever flipped off. Sourced from
# the user's glab CLI config (no token literal in the script).
if [[ -z "${GITLAB_FLASHINFER_TOKEN:-}" ]] && [[ -f "${HOME}/.config/glab-cli/config.yml" ]]; then
  GITLAB_FLASHINFER_TOKEN=$(grep -A 1 "gitlab-master.nvidia.com:" "${HOME}/.config/glab-cli/config.yml" | grep -oE 'glpat-[A-Za-z0-9_-]+' | head -1 || true)
fi
if [[ -n "${GITLAB_FLASHINFER_TOKEN:-}" ]]; then
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME="${UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME:-oauth2}"
  export UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD="${GITLAB_FLASHINFER_TOKEN}"
fi

if [[ ! -f "${SOURCE_NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${SOURCE_NEMORL}" >&2
  exit 1
fi

if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_ABS_PATH="${CONFIG_PATH}"
else
  CONFIG_ABS_PATH="${SOURCE_NEMORL}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_ABS_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

SNAPSHOT_CODE_LOWER="${SNAPSHOT_CODE,,}"
if [[ "${SNAPSHOT_CODE_LOWER}" == "1" || "${SNAPSHOT_CODE_LOWER}" == "true" || "${SNAPSHOT_CODE_LOWER}" == "yes" ]]; then
  SNAPSHOT_NEMORL="${SNAPSHOT_NEMORL:-${RESULTS_DIR}/code}"
  mkdir -p "${SNAPSHOT_NEMORL}"
  SNAPSHOT_NEMORL="$(cd "${SNAPSHOT_NEMORL}" && pwd)"
  if [[ "${SNAPSHOT_NEMORL}" == "${SOURCE_NEMORL}" || "${SNAPSHOT_NEMORL}/" == "${SOURCE_NEMORL}/"* ]]; then
    echo "[ERROR] SNAPSHOT_NEMORL must be outside SOURCE_NEMORL to avoid recursive rsync: ${SNAPSHOT_NEMORL}" >&2
    exit 1
  fi

  echo "Snapshotting code from ${SOURCE_NEMORL} to ${SNAPSHOT_NEMORL}"
  RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='.env'
    --exclude='.venv/'
    --exclude='.cache/'
    --exclude='.cache*/'
    --exclude='.tmp/'
    --exclude='.pytest_cache/'
    --exclude='.mypy_cache/'
    --exclude='.ruff_cache/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.out'
    --exclude='slurm-*.out'
    --exclude='wandb/'
    --exclude='logs/'
    --exclude='*-logs/'
    --exclude='results/'
    --exclude='jobs/'
    --exclude='checkpoints/'
    --exclude='build/'
    --exclude='*.o'
    --exclude='*.a'
    --exclude='*.egg-info/'
    --exclude='scripts/omnirl_scripts/tmp_docs/'
  )
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${SOURCE_NEMORL}/" "${SNAPSHOT_NEMORL}/"
  {
    echo "source_nemorl=${SOURCE_NEMORL}"
    echo "use_repo_vllm=${USE_REPO_VLLM}"
    echo "source_vllm_dir=${SOURCE_VLLM_DIR}"
    echo "snapshot_nemorl=${SNAPSHOT_NEMORL}"
    echo "job_name=${JOB_NAME}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${SNAPSHOT_NEMORL}/.nemo_rl_snapshot_info"
  NEMORL="${SNAPSHOT_NEMORL}"
else
  NEMORL="${SOURCE_NEMORL}"
fi
export NEMORL

VLLM_PYTHONPATH_PREFIX=""
if [[ "${USE_REPO_VLLM}" == "1" ]]; then
  if [[ "${SOURCE_VLLM_DIR}" == "${SOURCE_NEMORL}/3rdparty/vllm" ]]; then
    RUNTIME_VLLM_DIR="${NEMORL}/3rdparty/vllm"
  else
    RUNTIME_VLLM_DIR="${SOURCE_VLLM_DIR}"
  fi
  if [[ ! -f "${RUNTIME_VLLM_DIR}/vllm/__init__.py" ]]; then
    echo "[ERROR] repo vLLM checkout missing at ${RUNTIME_VLLM_DIR}." >&2
    echo "[ERROR] Set USE_REPO_VLLM=0 to use the vLLM packaged in the container." >&2
    exit 1
  fi
  if [[ -f "${RUNTIME_VLLM_DIR}/nemo-rl.env" ]]; then
    # shellcheck disable=SC1091
    source "${RUNTIME_VLLM_DIR}/nemo-rl.env"
  fi
  VLLM_PYTHONPATH_PREFIX="${RUNTIME_VLLM_DIR}:"
fi
# The precompiled wheel location is build-time metadata from build-custom-vllm.sh.
# Do not leak it into vLLM20 runtime, where it is reported as an unknown env var.
unset VLLM_PRECOMPILED_WHEEL_LOCATION

if [[ "${NEMO_RL_GPU_KEEPALIVE_SECONDS:-0}" != "0" ]]; then
  read -r -d '' GPU_KEEPALIVE_SETUP <<'SETUPEOF' || true
if [[ "${NEMO_RL_GPU_KEEPALIVE_SECONDS:-0}" != "0" ]]; then
  KEEPALIVE_PYTHON="${NEMO_RL_GPU_KEEPALIVE_PYTHON:-/opt/nemo_rl_venv/bin/python}"
  if [[ ! -x "${KEEPALIVE_PYTHON}" ]]; then
    KEEPALIVE_PYTHON="python3"
  fi
  "${KEEPALIVE_PYTHON}" - "${NEMO_RL_GPU_KEEPALIVE_SECONDS}" <<'PY' &
import os
import subprocess
import sys

seconds = float(sys.argv[1])
if seconds <= 0:
    raise SystemExit(0)

try:
    import torch
    num_gpus = torch.cuda.device_count()
except Exception as exc:
    print(f"[GPU_KEEPALIVE] disabled before startup: {exc}", flush=True)
    raise SystemExit(0)

driver_log = os.path.join(os.environ.get("LOG_DIR", ""), "ray-driver.log")

worker = r"""
import os
import sys
import time

seconds = float(sys.argv[1])
gpu = sys.argv[2]
driver_log = sys.argv[3]
matrix_size = int(os.environ.get("NEMO_RL_GPU_KEEPALIVE_MATRIX_SIZE", "4096"))
host_sleep = float(os.environ.get("NEMO_RL_GPU_KEEPALIVE_HOST_SLEEP", "0.0"))
sleep_cycles = int(os.environ.get("NEMO_RL_GPU_KEEPALIVE_CYCLES", "20000000"))
sync_every = max(1, int(os.environ.get("NEMO_RL_GPU_KEEPALIVE_SYNC_EVERY", "8")))

def should_stop():
    if not driver_log:
        return False
    try:
        size = os.path.getsize(driver_log)
        with open(driver_log, "rb") as f:
            f.seek(max(0, size - 131072))
            tail = f.read()
        return b"========================= Step 1/" in tail
    except FileNotFoundError:
        return False
    except Exception:
        return False

try:
    import torch
    torch.cuda.set_device(0)
    a = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
    b = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
    end = time.time() + seconds
    iters = 0
    use_cuda_sleep = hasattr(torch.cuda, "_sleep")
    while time.time() < end:
        if should_stop():
            print(f"[GPU_KEEPALIVE] gpu={gpu} stopping at Step 1", flush=True)
            break
        if use_cuda_sleep:
            torch.cuda._sleep(sleep_cycles)
        else:
            c = a @ b
        if iters % sync_every == 0:
            torch.cuda.synchronize()
        iters += 1
        if host_sleep > 0:
            time.sleep(host_sleep)
    torch.cuda.synchronize()
    print(f"[GPU_KEEPALIVE] gpu={gpu} finished after {seconds:.0f}s", flush=True)
except Exception as exc:
    print(f"[GPU_KEEPALIVE] gpu={gpu} disabled: {exc}", flush=True)
"""

for gpu in range(num_gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.Popen([sys.executable, "-c", worker, str(seconds), str(gpu), driver_log], env=env)

print(f"[GPU_KEEPALIVE] launched on {num_gpus} GPU(s) for {seconds:.0f}s", flush=True)
PY
fi
SETUPEOF
  if [[ -n "${SETUP_COMMAND:-}" ]]; then
    SETUP_COMMAND="${GPU_KEEPALIVE_SETUP}"$'\n'"${SETUP_COMMAND}"
  else
    SETUP_COMMAND="${GPU_KEEPALIVE_SETUP}"
  fi
fi
export SETUP_COMMAND

EXTRA_OVERRIDES=""
if [[ -n "${CONTEXT_PARALLEL_SIZE}" ]]; then
  EXTRA_OVERRIDES+=" policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}"
fi
if [[ -n "${POLICY_TP:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.megatron_cfg.tensor_model_parallel_size=${POLICY_TP}"
fi
if [[ -n "${POLICY_EP:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.megatron_cfg.expert_model_parallel_size=${POLICY_EP}"
fi
if [[ -n "${VLLM_TP:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP}"
fi
if [[ -n "${HYBRID_CP_ENABLED:-${DYNAMIC_CP_ENABLED:-}}" ]]; then
  EXTRA_OVERRIDES+=" policy.hybrid_cp.enabled=${HYBRID_CP_ENABLED:-${DYNAMIC_CP_ENABLED}}"
fi
if [[ -n "${HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.hybrid_cp.max_seqlen_per_dp_cp_rank=${HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK}"
fi
if [[ -n "${HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.hybrid_cp.microbatch_budget_multiplier=${HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER}"
fi
if [[ -n "${HYBRID_CP_FORCE_FULL_CP:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.hybrid_cp.force_full_cp=${HYBRID_CP_FORCE_FULL_CP}"
fi
if [[ -n "${POLICY_MAX_TOTAL_SEQUENCE_LENGTH:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.max_total_sequence_length=${POLICY_MAX_TOTAL_SEQUENCE_LENGTH}"
fi
if [[ -n "${GRPO_MAX_NUM_STEPS:-}" ]]; then
  EXTRA_OVERRIDES+=" grpo.max_num_steps=${GRPO_MAX_NUM_STEPS}"
fi
if [[ -n "${GRPO_SEED:-}" ]]; then
  EXTRA_OVERRIDES+=" grpo.seed=${GRPO_SEED}"
fi
if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.generation.vllm_cfg.max_model_len=${VLLM_MAX_MODEL_LEN}"
fi
if [[ -n "${GENERATION_MAX_NEW_TOKENS:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.generation.max_new_tokens=${GENERATION_MAX_NEW_TOKENS}"
fi
if [[ -n "${GENERATION_MIN_NEW_TOKENS:-}" ]]; then
  EXTRA_OVERRIDES+=" ++policy.generation.min_new_tokens=${GENERATION_MIN_NEW_TOKENS}"
fi
if [[ -n "${GRPO_NUM_PROMPTS_PER_STEP:-}" ]]; then
  EXTRA_OVERRIDES+=" grpo.num_prompts_per_step=${GRPO_NUM_PROMPTS_PER_STEP}"
fi
if [[ -n "${GRPO_NUM_GENERATIONS_PER_PROMPT:-}" ]]; then
  EXTRA_OVERRIDES+=" grpo.num_generations_per_prompt=${GRPO_NUM_GENERATIONS_PER_PROMPT}"
fi
if [[ "${ENABLE_FLASHINFER_AUTOTUNE:-true}" != "true" ]]; then
  EXTRA_OVERRIDES+=" ++policy.generation.vllm_kwargs.enable_flashinfer_autotune=false"
fi
if [[ -n "${POLICY_TRAIN_GLOBAL_BATCH_SIZE:-}" ]]; then
  EXTRA_OVERRIDES+=" policy.train_global_batch_size=${POLICY_TRAIN_GLOBAL_BATCH_SIZE}"
fi
if [[ -n "${GRPO_VAL_PERIOD:-}" ]]; then
  EXTRA_OVERRIDES+=" grpo.val_period=${GRPO_VAL_PERIOD}"
fi
# Match DFW's vLLM runtime settings while keeping current NRT code/infra.
EXTRA_OVERRIDES+=" policy.generation.vllm_cfg.enforce_eager=${VLLM_ENFORCE_EAGER:-false}"
EXTRA_OVERRIDES+=" ++policy.generation.vllm_cfg.enable_prefix_caching=${VLLM_ENABLE_PREFIX_CACHING:-true}"
EXTRA_OVERRIDES+=" policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
if [[ -n "${VLLM_LOAD_FORMAT:-}" ]]; then
  EXTRA_OVERRIDES+=" ++policy.generation.vllm_cfg.load_format=${VLLM_LOAD_FORMAT}"
fi
# This repo's grpo.py requires grpo.val_at_end (recipes' grpo.py doesn't
# read this key). The recipes-derived omni YAML doesn't define it, so inject
# the super-side default (false) here so the run reaches Step 1.
EXTRA_OVERRIDES+=" ++grpo.val_at_end=${GRPO_VAL_AT_END:-false}"
# In this branch, true means Gym/logging owns response logging and GRPO skips
# the expensive per-step train_data_step*.jsonl dump.
EXTRA_OVERRIDES+=" ++env.should_log_nemo_gym_responses=${NEMO_GYM_LOG_RESPONSES:-true}"
# Resume the same wandb run instead of starting a new one when WANDB_RUN_ID
# is set. Use Hydra's `++` so the keys are added or overridden safely.
# Pair WANDB_RESUME=allow with a pre-chosen id to chain a fresh run + N
# continuations under one wandb run (first to start creates, rest attach).
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  EXTRA_OVERRIDES+=" ++logger.wandb.id=${WANDB_RUN_ID} ++logger.wandb.resume=${WANDB_RESUME:-must}"
fi
if [[ -n "${EXTRA_OVERRIDES_APPEND:-}" ]]; then
  EXTRA_OVERRIDES+=" ${EXTRA_OVERRIDES_APPEND}"
fi

PYTHONPATH_ROOTS="${VLLM_PYTHONPATH_PREFIX}${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM"

# Match recipes' Hydra override surface 1:1 and explicitly enable wandb
# against the same project so the two runs land side-by-side.
export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' '${CACHE_DIR}' && \
if [[ ! -e '${NEMORL}/3rdparty/vllm' && -d /opt/nemo-rl/3rdparty/vllm ]]; then mkdir -p '${NEMORL}/3rdparty' && ln -s /opt/nemo-rl/3rdparty/vllm '${NEMORL}/3rdparty/vllm'; fi && \
export PYTHONPATH=${PYTHONPATH_ROOTS}\${PYTHONPATH:+:\$PYTHONPATH} && \
export MCORE_DISABLE_TORCH_COMPILE_JIT='${MCORE_DISABLE_TORCH_COMPILE_JIT}' && \
uv run --no-sync examples/run_vlm_grpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name='${MODEL_NAME}' \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}' \
data.train.cache_dir='${CACHE_DIR}'\
${EXTRA_OVERRIDES}"

cd "${NEMORL}"

SBATCH_ARGS=(
    --nodes="${NUM_NODES}"
    --account="${SBATCH_ACCOUNT}"
    --job-name="${JOB_NAME}"
    --partition="${SBATCH_PARTITION}"
    --time="${SBATCH_TIME}"
    --gres="gpu:${GPUS_PER_NODE}"
    --output="${LOGS_DIR}/%x_%j.log"
)

if [[ -n "${SBATCH_SWITCHES:-}" ]]; then
    SBATCH_ARGS+=(--switches="${SBATCH_SWITCHES}")
fi
if [[ -n "${SBATCH_NODELIST:-}" ]]; then
    SBATCH_ARGS+=(--nodelist="${SBATCH_NODELIST}")
fi
if [[ -n "${SBATCH_EXCLUDE:-}" ]]; then
    SBATCH_ARGS+=(--exclude="${SBATCH_EXCLUDE}")
fi
if [[ -n "${SBATCH_CONSTRAINT:-}" ]]; then
    SBATCH_ARGS+=(--constraint="${SBATCH_CONSTRAINT}")
fi
if [[ -n "${SBATCH_COMMENT:-}" ]]; then
    SBATCH_ARGS+=(--comment="${SBATCH_COMMENT}")
fi
if [[ -n "${SBATCH_MEM:-}" ]]; then
    SBATCH_ARGS+=(--mem="${SBATCH_MEM}")
fi

sbatch "${SBATCH_ARGS[@]}" ray.sub
