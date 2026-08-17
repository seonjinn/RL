#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-test-only}
MODEL=${MODEL:-qwen30}
ARM=${ARM:-mxfp8}
BRANCH=${BRANCH:-sna/exp-current-bf16-mxfp8-superset-20260816}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${BASE}/RL-exp-current-bf16-mxfp8-superset-20260816}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/mkar/containers/nemo-rl-nightly-ngc-20260815_212622.sqsh}
RUNTIME_SITE_PACKAGES=${RUNTIME_SITE_PACKAGES:-}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${BASE}/experiments/current-mxfp8-moe-rollout-ab/${MODEL}/${ARM}/${RUN_SUFFIX}}
CACHE_ROOT=${CACHE_ROOT:-${BASE}/.cache/current-mxfp8-moe-rollout-ab/${MODEL}/${ARM}/${RUN_SUFFIX}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-current-mxfp8-moe-rollout-ab}
WANDB_NAME=${WANDB_NAME:-${MODEL}-${ARM}-20step-${RUN_SUFFIX}}
WANDB_ENABLED=${WANDB_ENABLED:-true}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
PARTITION=${PARTITION:-batch}
TOTAL_NODES=${TOTAL_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GEN_NODES=${GEN_NODES:-4}
SEGMENT_SIZE=${SEGMENT_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-20}
WALLTIME=${WALLTIME:-04:00:00}

case "${MODEL}:${ARM}" in
  qwen30:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
    MODEL_LABEL=Qwen3-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=2048
    ROLLOUT_PRECISION=bf16
    QUANTIZATION_SCOPE=none
    ;;
  qwen30:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
    MODEL_LABEL=Qwen3-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=2048
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=routed_expert_fc1_fc2_only
    ;;
  nano:bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-bf16-rollout-nccl.yaml
    MODEL_LABEL=Nemotron3-Nano-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=16
    ROLLOUT_PRECISION=bf16
    QUANTIZATION_SCOPE=none
    ;;
  nano:mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-mxfp8-rollout-nccl.yaml
    MODEL_LABEL=Nemotron3-Nano-30B-A3B
    TRAIN_GLOBAL_BATCH_SIZE=16
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=routed_expert_fc1_fc2_only
    ;;
  *)
    echo "MODEL must be qwen30 or nano and ARM must be bf16 or mxfp8" >&2
    exit 2
    ;;
esac

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be submit or test-only" >&2; exit 2 ;;
esac

git -C "${REPO}" fetch origin "${BRANCH}"
git -C "${REPO}" pull --ff-only origin "${BRANCH}"
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "origin/${BRANCH}")
if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
  echo "Local and remote heads differ: local=${LOCAL_HEAD}, remote=${REMOTE_HEAD}" >&2
  exit 2
fi
if [[ -n "${EXPECTED_HEAD}" && "${LOCAL_HEAD}" != "${EXPECTED_HEAD}" ]]; then
  echo "Expected HEAD ${EXPECTED_HEAD}; got ${LOCAL_HEAD}" >&2
  exit 2
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned submodules must be initialized" >&2
  exit 2
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" "${HF_HOME}"; do
  test -e "${path}"
done
if [[ -n "${RUNTIME_SITE_PACKAGES}" ]]; then
  test -f "${RUNTIME_SITE_PACKAGES}/urllib3/exceptions.py"
  test -f "${RUNTIME_SITE_PACKAGES}/nvidia/cudnn/lib/libcudnn.so.9"
fi

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ "${WANDB_ENABLED}" == true && -f "${HOME}/.netrc" ]]; then
  umask 077
  awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
        if (found && $i == "password") { print $(i + 1); exit }
      }
    }
  ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}"
elif [[ "${WANDB_ENABLED}" == true && -n "${WANDB_API_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}"
fi
if [[ "${WANDB_ENABLED}" == true ]]; then
  test -s "${WANDB_KEY_FILE}"
  WANDB_KEY_SETUP="export WANDB_API_KEY=\"\$(cat ${WANDB_KEY_FILE})\""
elif [[ "${WANDB_ENABLED}" == false ]]; then
  WANDB_KEY_SETUP=:
else
  echo "WANDB_ENABLED must be true or false" >&2
  exit 2
fi

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo=${REPO}
repo_sha=${LOCAL_HEAD}
branch=${BRANCH}
config=${CONFIG}
container=${CONTAINER}
runtime_site_packages=${RUNTIME_SITE_PACKAGES:-container_default}
python_version=3.13.11
python_floor_override=experiment_only_metadata_override_from_3.13.14
gym_source_commit=5a6fc589c0196f73a5931781b06da61f668a80d7
cluster=oci-hsg-cs-001
hardware=GB200
model=${MODEL_LABEL}
arm=${ARM}
total_nodes=${TOTAL_NODES}
gpus_per_node=${GPUS_PER_NODE}
trainer_nodes=$((TOTAL_NODES - GEN_NODES))
generation_nodes=${GEN_NODES}
training_precision=bf16
rollout_precision=${ROLLOUT_PRECISION}
quantization_scope=${QUANTIZATION_SCOPE}
moe_backend=flashinfer_trtllm
refit_transport=nccl_reshard
cuda_graphs=enabled
max_steps=${MAX_STEPS}
seed=42
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
wandb_enabled=${WANDB_ENABLED}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/cache
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${CACHE_ROOT}/worker-venvs
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
${WANDB_KEY_SETUP}
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
uv run --frozen examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.segment_size=${SEGMENT_SIZE} \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
  policy.generation.refit_transport=nccl_reshard \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.generation.vllm_cfg.async_engine=false \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.use_tqdm=false \
  policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.seed=42 \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=${WANDB_ENABLED} \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
uv run --frozen tests/json_dump_tb_logs.py ${EXPERIMENT_ROOT}/logs \
  --output_path ${EXPERIMENT_ROOT}/metrics.json || true
EOF
)

export BASE_LOG_DIR=${EXPERIMENT_ROOT}
export COMMAND
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE
export MOUNTS=/lustre:/lustre
if [[ -n "${RUNTIME_SITE_PACKAGES}" ]]; then
  MOUNTS+=",${RUNTIME_SITE_PACKAGES}:/opt/nemo_rl_venv/lib/python3.13/site-packages"
fi
export UV_CACHE_DIR_OVERRIDE=${CACHE_ROOT}/uv-cache

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --gres="gpu:${GPUS_PER_NODE}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment="${SEGMENT_SIZE}"
  --job-name="sna-${MODEL}-${ARM}-${MAX_STEPS}s"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
)

printf 'action=%s\nmodel=%s\narm=%s\nsha=%s\nresult=%s\n' \
  "${ACTION}" "${MODEL}" "${ARM}" "${LOCAL_HEAD}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
