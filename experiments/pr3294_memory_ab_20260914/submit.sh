#!/usr/bin/env bash
set -euo pipefail

ACTION=${ACTION:-render}
PREQUANT=${PREQUANT:-false}
MEMORY_PROBE=${MEMORY_PROBE:-0}
case "${PREQUANT}" in true|false) ;; *) exit 2 ;; esac
case "${MEMORY_PROBE}" in 0|1) ;; *) exit 2 ;; esac
CLUSTER=${CLUSTER:-lyris}
MODEL=${MODEL:-qwen30}
MODE=${MODE:-sync}
ARM=${ARM:-bf16-mxfp8}
RUN_GROUP=${RUN_GROUP:-20260914-prequant-${PREQUANT}-memory-${MEMORY_PROBE}}
MAX_STEPS=${MAX_STEPS:-20}
WALLTIME=${WALLTIME:-04:00:00}
PARTITION=${PARTITION:-}
AFTEROK_JOB_ID=${AFTEROK_JOB_ID:-}
EXPERIMENT=experiments/pr3294_memory_ab_20260914
[[ "${MODEL}:${MODE}:${ARM}" == qwen30:sync:bf16-mxfp8 ]] || exit 2
case "${ACTION}" in render|test-only|submit) ;; *) exit 2 ;; esac
case "${MODEL}" in qwen30|qwen235|super|qwen35|lightning) ;; *) echo "No audited recipe" >&2; exit 2 ;; esac
case "${MODE}" in sync|async) ;; *) exit 2 ;; esac
case "${ARM}" in bf16-bf16|bf16-mxfp8) ;; *) exit 2 ;; esac
case "${CLUSTER}" in
  oci)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=(--gres=gpu:4)
    ;;
  ptyche)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/tmp/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=()
    ;;
  lyris)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=()
    ;;
  *) echo "CLUSTER must be oci, ptyche, or lyris" >&2; exit 2 ;;
esac

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT after checking FairShare}"
: "${WANDB_HOME:=/home/${USER}}"
: "${NRL_DISABLE_NUMA_MEMBIND:=1}"
: "${NRL_FORCE_REBUILD_VENVS:=false}"
: "${NEMO_RL_PY_EXECUTABLES_SYSTEM:=0}"
: "${ACTOR_VENV_ROOT:=/opt/ray_venvs}"


PERF_DIR=examples/configs/recipes/llm/performance
case "${MODEL}" in
 qwen30)
  STEM=grpo-qwen3-30ba3b-4n4g
  NUM_NODES=4
  SEGMENT_SIZE=2
  if [[ "${MODE}" == sync ]]; then SEGMENT_SIZE=4; fi
  MODEL_CACHE=models--Qwen--Qwen3-30B-A3B
  ;;
 qwen235)
  if [[ "${MODE}" == sync ]]; then STEM=grpo-qwen3-235b-16n4g; NUM_NODES=16
  else STEM=grpo-qwen3-235b-32n4g; NUM_NODES=32; fi
  SEGMENT_SIZE=16
  MODEL_CACHE=models--Qwen--Qwen3-235B-A22B
  ;;
 super)
  STEM=grpo-nemotron3-super-120BA12B-32n4g
  NUM_NODES=32
  SEGMENT_SIZE=8
  MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16
  ;;
 qwen35|lightning)
  STEM=custom
  NUM_NODES=8
  SEGMENT_SIZE=4
  if [[ "${MODE}" == sync ]]; then SEGMENT_SIZE=8; fi
  if [[ "${MODEL}" == qwen35 ]]; then
   MODEL_CACHE=models--Qwen--Qwen3.5-35B-A3B-Base
  else
   MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
  fi
  ;;
esac
if [[ "${MODE}" == async ]]; then STEM=${STEM}-async-1off; fi
CONFIG=${EXPERIMENT}/qwen30-sync.yaml
if [[ "${MODEL}" == qwen35 ]]; then CONFIG=${EXPERIMENT}/qwen35-performance-${MODE}.yaml; fi
if [[ "${MODEL}" == lightning ]]; then CONFIG=${EXPERIMENT}/lightning-${MODE}.yaml; fi
RUN_NAME="strict-${MODEL}-${MODE}-${ARM}-${RUN_GROUP}"
JOB_NAME="${SLURM_ACCOUNT}.${RUN_NAME}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
LOCAL_JOB_ROOT="${LOCAL_ROOT}/${RUN_NAME}"
RUN_REPO="${LOCAL_JOB_ROOT}/source"
DATASETS_CACHE="${HF_HOME_SOURCE}/datasets"
DATASET_STAGE_COMMAND=""
USE_SHARED_MODEL=${USE_SHARED_MODEL:-1}
MOE_BACKEND=flashinfer_trtllm
COMMON_OVERRIDES=(
 "grpo.max_num_steps=${MAX_STEPS}"
 "policy.generation.vllm_cfg.refit_prequantize=${PREQUANT}"
 "++policy.generation.vllm_cfg.env_vars.NRL_REFIT_MEMORY_PROBE=${MEMORY_PROBE}"
 "++policy.generation.vllm_kwargs.moe_backend=${MOE_BACKEND}"
 "logger.log_dir=${RUN_ROOT}/logs"
 "logger.wandb_enabled=true"
 "logger.wandb.project=nemo-rl-pr3294-memory-ab"
 "logger.wandb.name=${RUN_NAME}"
)
PRECISION_OVERRIDES=()
if [[ "${MODEL}:${MODE}" == qwen30:sync ]]; then
 COMMON_OVERRIDES+=("loss_fn.use_importance_sampling_correction=true")
fi
if [[ "${ARM}" == bf16-mxfp8 ]]; then
 if [[ "${MODEL}" == super || "${MODEL}" == lightning ]]; then
  IGNORE_PATTERNS='["*layers.*.mixer.qkv_proj","*layers.*.mixer.o_proj","*layers.*.mixer.in_proj","*layers.*.mixer.out_proj","*layers.*.mixer.up_proj","*layers.*.mixer.down_proj","*layers.*.mixer.gate","*layers.*.mixer.shared_experts.*","*layers.*.mixer.fc1_latent_proj","*layers.*.mixer.fc2_latent_proj","*mtp.*","lm_head"]'
 elif [[ "${MODEL}" == qwen35 ]]; then
  IGNORE_PATTERNS='["*layers.*.self_attn.*","*layers.*.linear_attn.*","*layers.*.mlp.gate","*layers.*.mlp.shared_expert.*","*layers.*.mlp.shared_expert_gate","*visual.*","*mtp.*","lm_head"]'
 else
  IGNORE_PATTERNS='["*layers.*.self_attn.*","*layers.*.mlp.gate","*layers.*.mlp.shared_experts.*","*mtp.*","lm_head"]'
 fi
 PRECISION_OVERRIDES=(
  "policy.generation.vllm_cfg.precision=fp8"
  "++policy.generation.vllm_cfg.is_mx=true"
  "++policy.generation.vllm_cfg.quantization_ignore_patterns=${IGNORE_PATTERNS}"
 )
 if [[ "${MODEL}" == qwen35 || "${MODEL}" == lightning ]]; then
  PRECISION_OVERRIDES+=(
   "++policy.generation.vllm_cfg.num_first_layers_in_bf16=2"
   "++policy.generation.vllm_cfg.num_last_layers_in_bf16=6"
  )
 fi
fi
printf 'recipe=%s\narm=%s\nnodes=%s\n' "${CONFIG}" "${ARM}" "${NUM_NODES}"
printf 'overrides:'
printf ' %q' "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}"
printf '\n'
if [[ "${ACTION}" == render ]]; then
  exit 0
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" \
  "${HF_HOME_SOURCE}/hub/${MODEL_CACHE}" "${WANDB_HOME}/.netrc"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

for override in "${PRECISION_OVERRIDES[@]}"; do
  if [[ "${override}" == ++policy.megatron_cfg.te_precision_config_file=* ]]; then
    te_config_path=${override#*=}
    te_config_relative=${te_config_path#"${RUN_REPO}/"}
    if [[ ! -f "${REPO}/${te_config_relative}" ]]; then
      echo "Missing TE precision recipe: ${REPO}/${te_config_relative}" >&2
      exit 2
    fi
    git -C "${REPO}" ls-files --error-unmatch "${te_config_relative}" >/dev/null
  fi
done

MODEL_STAGE_COMMAND="rsync -a --ignore-existing ${HF_HOME_SOURCE}/hub/${MODEL_CACHE}/ ${LOCAL_JOB_ROOT}/hf/hub/${MODEL_CACHE}/;"
if [[ "${USE_SHARED_MODEL}" == 1 ]]; then
  MODEL_REF_FILE="${HF_HOME_SOURCE}/hub/${MODEL_CACHE}/refs/main"
  if [[ ! -f "${MODEL_REF_FILE}" ]]; then
    echo "Missing model ref: ${MODEL_REF_FILE}" >&2
    exit 2
  fi
  MODEL_SNAPSHOT="${HF_HOME_SOURCE}/hub/${MODEL_CACHE}/snapshots/$(<"${MODEL_REF_FILE}")"
  if [[ ! -d "${MODEL_SNAPSHOT}" ]]; then
    echo "Missing model snapshot: ${MODEL_SNAPSHOT}" >&2
    exit 2
  fi
  COMMON_OVERRIDES+=("policy.model_name=${MODEL_SNAPSHOT}")
  MODEL_STAGE_COMMAND=""
fi

if [[ "${ACTION}" == submit ]]; then
  git -C "${REPO}" pull --ff-only origin sna/pr3294-memory-ab-20260914
  git -C "${REPO}" submodule update --init --recursive --checkout
  if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=none)" ]]; then
    echo "Repository and pinned submodules must be clean before submission" >&2
    exit 2
  fi
fi

SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "${SOURCE_SHA}" != "${EXPECTED_SOURCE_SHA}" ]]; then
  echo "Source changed after preflight: expected ${EXPECTED_SOURCE_SHA}, got ${SOURCE_SHA}" >&2
  exit 2
fi
SOURCE_STATE=$(git -C "${REPO}" submodule status --recursive)
SOURCE_ID=$(printf '%s\n%s\n' "${SOURCE_SHA}" "${SOURCE_STATE}" | sha256sum | cut -c1-16)
SOURCE_ARCHIVE_ROOT=${SOURCE_ARCHIVE_ROOT:-/home/${USER}/.cache/nemo-rl-source-archives}
SOURCE_ARCHIVE="${SOURCE_ARCHIVE_ROOT}/nemo-rl-${SOURCE_ID}.tar"

if [[ "${ACTION}" == submit && ! -f "${SOURCE_ARCHIVE}" ]]; then
  mkdir -p "${SOURCE_ARCHIVE_ROOT}"
  SOURCE_MANIFEST=$(mktemp "${TMPDIR:-/tmp}/nemo-rl-source-manifest.XXXXXX")
  SOURCE_ARCHIVE_TMP=$(mktemp "${TMPDIR:-/tmp}/nemo-rl-source.XXXXXX.tar")
  SOURCE_ARCHIVE_STAGE=$(mktemp "${SOURCE_ARCHIVE_ROOT}/.nemo-rl-${SOURCE_ID}.XXXXXX")
  trap 'rm -f "${SOURCE_MANIFEST:-}" "${SOURCE_ARCHIVE_TMP:-}" "${SOURCE_ARCHIVE_STAGE:-}"' EXIT
  git -C "${REPO}" ls-files -z --recurse-submodules --cached --full-name > "${SOURCE_MANIFEST}"
  tar --null -cf "${SOURCE_ARCHIVE_TMP}" -C "${REPO}" -T "${SOURCE_MANIFEST}"
  if [[ ! -f "${SOURCE_ARCHIVE}" ]]; then
    # Cross-filesystem mv can expose a truncated final file on quota failure.
    cp "${SOURCE_ARCHIVE_TMP}" "${SOURCE_ARCHIVE_STAGE}"
    cmp -s "${SOURCE_ARCHIVE_TMP}" "${SOURCE_ARCHIVE_STAGE}"
    mv "${SOURCE_ARCHIVE_STAGE}" "${SOURCE_ARCHIVE}"
  fi
  rm -f "${SOURCE_MANIFEST}" "${SOURCE_ARCHIVE_TMP}" "${SOURCE_ARCHIVE_STAGE}"
  trap - EXIT
fi

if [[ "${ACTION}" == submit && ! -f "${SOURCE_ARCHIVE}" ]]; then
  echo "Failed to create source archive: ${SOURCE_ARCHIVE}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"

COMMAND=$(printf '%q ' /opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config "${CONFIG}" "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}")
COMMAND="set -euo pipefail; cd ${RUN_REPO}; \
export HOME=/root; \
export HF_HOME=${LOCAL_JOB_ROOT}/hf; \
export HF_DATASETS_CACHE=${DATASETS_CACHE}; \
export HUGGINGFACE_HUB_CACHE=${LOCAL_JOB_ROOT}/hf/hub; \
export NRL_MEGATRON_CHECKPOINT_DIR=${HF_HOME_SOURCE}/nemo_rl; \
export NEMO_RL_VENV_DIR=${ACTOR_VENV_ROOT}; \
export VLLM_CACHE_ROOT=${LOCAL_JOB_ROOT}/vllm; \
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_JOB_ROOT}/inductor; \
export TRITON_CACHE_DIR=${LOCAL_JOB_ROOT}/triton; \
export UV_CACHE_DIR=${LOCAL_JOB_ROOT}/uv; \
export RAY_TMPDIR=${LOCAL_JOB_ROOT}/ray; \
export PYTHONPATH=${RUN_REPO}:${RUN_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${RUN_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM; \
export NRL_REFIT_MEMORY_PROBE=${MEMORY_PROBE}; \
export RAY_DEDUP_LOGS=0; \
export NRL_TRACE_REFIT_PHASES=${NRL_TRACE_REFIT_PHASES:-0}; \
export NRL_DISABLE_NUMA_MEMBIND=${NRL_DISABLE_NUMA_MEMBIND}; \
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS}; \
export NEMO_RL_PY_EXECUTABLES_SYSTEM=${NEMO_RL_PY_EXECUTABLES_SYSTEM}; \
${COMMAND}"

SETUP_COMMAND="set -euo pipefail; \
rm -rf ${LOCAL_JOB_ROOT}; \
mkdir -p ${RUN_REPO} ${LOCAL_JOB_ROOT}/hf/hub ${LOCAL_JOB_ROOT}/hf/datasets ${LOCAL_JOB_ROOT}/vllm ${LOCAL_JOB_ROOT}/inductor ${LOCAL_JOB_ROOT}/triton ${LOCAL_JOB_ROOT}/uv ${LOCAL_JOB_ROOT}/ray; \
tar -xf ${SOURCE_ARCHIVE} -C ${RUN_REPO}; \
${MODEL_STAGE_COMMAND} \
${DATASET_STAGE_COMMAND}"

if [[ "${MEMORY_PROBE}" == 1 ]]; then
  SETUP_COMMAND+=" PYTHONPATH=${RUN_REPO} /opt/nemo_rl_venv/bin/python ${RUN_REPO}/${EXPERIMENT}/probe_smoke.py;"
fi

export CONTAINER
# Without this bind, the container's /raid/scratch lives in its tmpfs root.
export MOUNTS="/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,${WANDB_HOME}/.netrc:/root/.netrc"
if [[ -n "${NRL_CUMEM_EXTENSION_FILE:-}" ]]; then
  : "${NRL_CUMEM_EXTENSION_SHA256:?}"
  : "${NRL_CUMEM_PYTHON_FILE:?}"
  : "${NRL_CUMEM_PYTHON_SHA256:?}"
  [[ "${NRL_FORCE_REBUILD_VENVS}" == false && "${NEMO_RL_PY_EXECUTABLES_SYSTEM}" == 0 ]] || exit 2
  for pair in extension python; do
    if [[ "$pair" == extension ]]; then
      file=$NRL_CUMEM_EXTENSION_FILE
      expected=$NRL_CUMEM_EXTENSION_SHA256
      destination=cumem_allocator.abi3.so
    else
      file=$NRL_CUMEM_PYTHON_FILE
      expected=$NRL_CUMEM_PYTHON_SHA256
      destination=device_allocator/cumem.py
    fi
    actual=$(sha256sum "$file")
    [[ "${actual%% *}" == "$expected" ]] || exit 2
    printf 'experimental_cumem_%s=%s\n' "$pair" "$actual"
    MOUNTS+=",${file}:${ACTOR_VENV_ROOT}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/lib/python3.13/site-packages/vllm/${destination}:ro"
  done
  COMMAND="export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=${LOCAL_JOB_ROOT}/pycache; ${COMMAND}"
fi
if [[ -n "${VLLM_PADDING_SOURCE:-}" ]]; then
  : "${VLLM_PADDING_SHA:?Pin the tested vLLM commit}"
  if [[ "${NRL_FORCE_REBUILD_VENVS}" != false || "${NEMO_RL_PY_EXECUTABLES_SYSTEM}" != 0 ]]; then
    echo "The tested vLLM overlay requires the prebuilt actor environment" >&2
    exit 2
  fi
  VLLM_RESOLVED_SHA=$(git -C "${VLLM_PADDING_SOURCE}" rev-parse "${VLLM_PADDING_SHA}^{commit}")
  if [[ "${VLLM_RESOLVED_SHA}" != "${VLLM_PADDING_SHA}" ]]; then
    echo "VLLM_PADDING_SHA must be a full commit SHA" >&2
    exit 2
  fi
  VLLM_FILES=(
    vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
    vllm/model_executor/layers/fused_moe/oracle/unquantized.py
  )
  VLLM_SNAPSHOT_ROOT=/home/${USER}/.cache/nemo-rl-vllm-overlays
  VLLM_SNAPSHOT=${VLLM_SNAPSHOT_ROOT}/${VLLM_RESOLVED_SHA}
  mkdir -p "${VLLM_SNAPSHOT_ROOT}"
  if [[ ! -d "${VLLM_SNAPSHOT}" ]]; then
    VLLM_STAGE=$(mktemp -d "${VLLM_SNAPSHOT_ROOT}/.stage.XXXXXX")
    git -C "${VLLM_PADDING_SOURCE}" archive "${VLLM_RESOLVED_SHA}" "${VLLM_FILES[@]}" | tar -xf - -C "${VLLM_STAGE}"
    mv -T "${VLLM_STAGE}" "${VLLM_SNAPSHOT}"
  fi
  VLLM_PACKAGE=${ACTOR_VENV_ROOT}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/lib/python3.13/site-packages
  for file in "${VLLM_FILES[@]}"; do
    git -C "${VLLM_PADDING_SOURCE}" show "${VLLM_RESOLVED_SHA}:${file}" | cmp -s - "${VLLM_SNAPSHOT}/${file}"
    MOUNTS+=",${VLLM_SNAPSHOT}/${file}:${VLLM_PACKAGE}/${file}:ro"
  done
  printf 'vllm_overlay_commit=%s\n' "${VLLM_RESOLVED_SHA}"
  sha256sum "${VLLM_FILES[@]/#/${VLLM_SNAPSHOT}/}"
  COMMAND="export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=${LOCAL_JOB_ROOT}/pycache; ${COMMAND}"
fi
if [[ "${CLUSTER}" == oci ]]; then
  MOUNTS="${MOUNTS},/raid/scratch:/raid/scratch"
fi
export CONTAINER_REMAP_ROOT=1
export COMMAND
export SETUP_COMMAND
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
export BASE_LOG_DIR="${RUN_ROOT}"
export RAY_TMPDIR_ROOT="${LOCAL_JOB_ROOT}/ray"

SBATCH_MODE=()
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_MODE=(--test-only)
fi

SBATCH_PARTITION=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_PARTITION=(--partition="${PARTITION}")
fi

SBATCH_DEPENDENCY=(--dependency=)
if [[ -n "${AFTEROK_JOB_ID}" ]]; then
  SBATCH_DEPENDENCY=(--dependency="afterok:${AFTEROK_JOB_ID}")
fi

if [[ "${CLUSTER}" == oci ]]; then
  export PATH="/cm/local/apps/slurm/current/bin:/usr/local/bin:${PATH}"
fi

SBATCH_EXPORT=()
if [[ "${CLUSTER}" == oci ]]; then
  SBATCH_EXPORT=(--export="ALL,PATH=${PATH}")
fi

exec sbatch "${SBATCH_MODE[@]}" \
  "${SBATCH_EXPORT[@]}" \
  --nodes="${NUM_NODES}" \
  "${GPU_REQUEST[@]}" \
  --exclusive \
  --account="${SLURM_ACCOUNT}" \
  "${SBATCH_PARTITION[@]}" \
  --time="${WALLTIME}" \
  --segment="${SEGMENT_SIZE}" \
  "${SBATCH_DEPENDENCY[@]}" \
  --job-name="${JOB_NAME}" \
  --output="${RUN_ROOT}/slurm-%j.out" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"precision matrix startup"}}' \
  "${REPO}/ray.sub"
