#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MODEL=${MODEL:-qwen30}
ARM=${ARM:-bf16}
MAX_STEPS=${MAX_STEPS:-1}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-sync-ipc-bf16-mxfp8-ab-20260816}
EXPECTED_HEAD=${EXPECTED_HEAD:-}

case "${MODEL}" in
  qwen30)
    TOTAL_NODES=4
    TRAIN_GLOBAL_BATCH_SIZE=2048
    case "${ARM}" in
      bf16)
        CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
        ROLLOUT_PRECISION=bfloat16
        IS_MX=false
        ;;
      mxfp8)
        CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
        ROLLOUT_PRECISION=fp8
        IS_MX=true
        ;;
      *)
        echo "ARM must be bf16 or mxfp8" >&2
        exit 2
        ;;
    esac
    ;;
  nano)
    TOTAL_NODES=8
    TRAIN_GLOBAL_BATCH_SIZE=16
    case "${ARM}" in
      bf16)
        CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-bf16-rollout-nccl.yaml
        ROLLOUT_PRECISION=bfloat16
        IS_MX=false
        ;;
      mxfp8)
        CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-mxfp8-rollout-nccl.yaml
        ROLLOUT_PRECISION=fp8
        IS_MX=true
        ;;
      *)
        echo "ARM must be bf16 or mxfp8" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "MODEL must be qwen30 or nano" >&2
    exit 2
    ;;
esac

RUN_ARGS=(
  --config "${CONFIG}"
  "cluster.num_nodes=${TOTAL_NODES}"
  "cluster.gpus_per_node=4"
  "cluster.segment_size=4"
  "policy.precision=bfloat16"
  "grpo.async_grpo.enabled=false"
  "data_plane.enabled=false"
  "policy.generation.colocated.enabled=true"
  "policy.generation.colocated.resources.num_nodes=${TOTAL_NODES}"
  "policy.generation.colocated.resources.gpus_per_node=4"
  "policy.generation.refit_transport=null"
  "policy.generation.real_quant_export_cpu_offload=false"
  "policy.generation.vllm_cfg.async_engine=false"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "policy.generation.vllm_cfg.precision=${ROLLOUT_PRECISION}"
  "policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm"
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.seed=42"
  "grpo.val_at_start=false"
  "++grpo.val_at_end=false"
  "checkpointing.enabled=false"
)

if [[ "${IS_MX}" == true ]]; then
  RUN_ARGS+=("policy.generation.vllm_cfg.is_mx=true")
fi

printf -v RUN_COMMAND '%q ' /opt/nemo_rl_venv/bin/python examples/run_grpo.py "${RUN_ARGS[@]}"

if [[ "${ACTION}" == render ]]; then
  printf '%s\n' "${RUN_COMMAND}"
  exit 0
fi

BASE=${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
REPO=${REPO:-${BASE}/RL-sync-ipc-bf16-mxfp8-ab-20260816}
CONTAINER=${CONTAINER:-${BASE}/containers/nemo2606/nemo_rl_nightly_nemo2606_20260812_2574124.sqsh}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
RESULT_ROOT=${RESULT_ROOT:-${BASE}/experiments/sync-ipc-bf16-mxfp8-ab}
EXPERIMENT_ROOT=${RESULT_ROOT}/${MODEL}/${ARM}/${RUN_SUFFIX}
CACHE_ROOT=${BASE}/.cache/sync-ipc-bf16-mxfp8-ab/${MODEL}/${ARM}/${RUN_SUFFIX}
WORKER_VENV_ROOT=/tmp/nemo_rl_worker_venvs/sync-ipc/${MODEL}/${ARM}/${RUN_SUFFIX}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-36x2-a01r}
WALLTIME=${WALLTIME:-05:00:00}
WANDB_ENABLED=${WANDB_ENABLED:-false}
WANDB_PROJECT=${WANDB_PROJECT:-sna-sync-ipc-bf16-mxfp8-ab}
WANDB_NAME=${WANDB_NAME:-${MODEL}-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *)
    echo "ACTION must be render, test-only, or submit" >&2
    exit 2
    ;;
esac

git -C "${REPO}" fetch origin "${BRANCH}"
git -C "${REPO}" pull --ff-only origin "${BRANCH}"
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "origin/${BRANCH}")
test "${LOCAL_HEAD}" = "${REMOTE_HEAD}"
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
test -f "${REPO}/${CONFIG}"
test -f "${REPO}/ray.sub"
test -f "${CONTAINER}"
test -d "${HF_HOME}"

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}"

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo_sha=${LOCAL_HEAD}
branch=${BRANCH}
model=${MODEL}
arm=${ARM}
training_precision=bf16
rollout_precision=${ROLLOUT_PRECISION}
quantization_scope=$([[ "${IS_MX}" == true ]] && echo routed_expert_fc1_fc2_only || echo none)
refit_transport=ipc
colocated=true
sync_grpo=true
cuda_graphs=enabled
moe_backend=flashinfer_trtllm
nodes=${TOTAL_NODES}
gpus_per_node=4
max_steps=${MAX_STEPS}
seed=42
container=${CONTAINER}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}
export UV_PYTHON=/opt/nemo_rl_venv/bin/python
export UV_NO_MANAGED_PYTHON=1
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_LOCK_TIMEOUT=7200
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
export PYTHONPATH=${REPO}
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
${RUN_COMMAND} \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=${WANDB_ENABLED} \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
/opt/nemo_rl_venv/bin/python tests/json_dump_tb_logs.py ${EXPERIMENT_ROOT}/logs \
  --output_path ${EXPERIMENT_ROOT}/metrics.json || true
EOF
)

export BASE_LOG_DIR=${EXPERIMENT_ROOT}
export COMMAND
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE=4
export MOUNTS=/lustre:/lustre
export UV_CACHE_DIR_OVERRIDE=${CACHE_ROOT}/uv-cache

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --gres=gpu:4
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment=4
  --job-name="sna-sync-ipc-${MODEL}-${ARM}-${MAX_STEPS}s"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
)

printf 'action=%s\nmodel=%s\narm=%s\nsha=%s\nresult=%s\n' \
  "${ACTION}" "${MODEL}" "${ARM}" "${LOCAL_HEAD}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
