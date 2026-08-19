#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MODEL=${MODEL:-qwen30}
ARM=${ARM:-moe_only}
MAX_STEPS=${MAX_STEPS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-pr3652-pr3477-qkvo-metrics-20260818}
EXPECTED_HEAD=${EXPECTED_HEAD:-}

case "${MODEL}:${ARM}" in
  qwen30:moe_only) CONFIG=experiments/pr3477_qkvo_metrics/qwen30_moe_only.yaml ;;
  qwen30:qkvo) CONFIG=experiments/pr3477_qkvo_metrics/qwen30_qkvo.yaml ;;
  nano:moe_only) CONFIG=experiments/pr3477_qkvo_metrics/nano_moe_only.yaml ;;
  nano:qkvo) CONFIG=experiments/pr3477_qkvo_metrics/nano_qkvo.yaml ;;
  *) echo "Unsupported MODEL=${MODEL}, ARM=${ARM}" >&2; exit 2 ;;
esac

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${BASE}/worktrees/RL-pr3477-qkvo-metrics-20260818}
CONTAINER=${CONTAINER:-${BASE}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
WANDB_HOME=${WANDB_HOME:-${BASE}/wandb_netrc_home}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-05:00:00}
RESULT_ROOT=${RESULT_ROOT:-${BASE}/experiments/pr3477-qkvo-metrics}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${RESULT_ROOT}/${MODEL}/${ARM}/${RUN_SUFFIX}}
CACHE_ROOT=${CACHE_ROOT:-${BASE}/.cache/pr3477-qkvo-metrics/${MODEL}/${ARM}/${RUN_SUFFIX}}
WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-/tmp/nemo_rl_worker_venvs/pr3477-qkvo/${MODEL}/${ARM}/${RUN_SUFFIX}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3477-qkvo-metrics}
WANDB_NAME=${WANDB_NAME:-${MODEL}-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}

if [[ "${ACTION}" == render ]]; then
  printf 'model=%s\narm=%s\nconfig=%s\nsteps=%s\n' \
    "${MODEL}" "${ARM}" "${CONFIG}" "${MAX_STEPS}"
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

git -C "${REPO}" fetch origin "${BRANCH}"
git -C "${REPO}" pull --ff-only origin "${BRANCH}"
git -C "${REPO}" submodule update --init --recursive
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "origin/${BRANCH}")
test "${LOCAL_HEAD}" = "${REMOTE_HEAD}"
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)"
if git -C "${REPO}" submodule status --recursive | grep -q '^[+-U]'; then
  echo "All submodules must be initialized at pinned revisions" >&2
  exit 2
fi
for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" \
  "${HF_HOME}" "${WANDB_HOME}/.netrc"; do
  test -e "${path}"
done

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}"
cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo_sha=${LOCAL_HEAD}
branch=${BRANCH}
config=${CONFIG}
container=${CONTAINER}
hardware=GB200
model=${MODEL}
arm=${ARM}
nodes=8
gpus_per_node=4
generation_nodes=4
training_precision=bf16
rollout_precision=mxfp8
quantization_scope=${ARM}
refit_transport=nccl_reshard
cuda_graphs=enabled
moe_backend=flashinfer_trtllm
aggregate_steps=3-20
max_steps=${MAX_STEPS}
seed=42
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HOME=/root
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_PYTHON=/opt/nemo_rl_venv/bin/python
export UV_NO_MANAGED_PYTHON=1
export UV_LOCK_TIMEOUT=7200
unset UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR WANDB_API_KEY
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
/opt/nemo_rl_venv/bin/python experiments/pr3477_qkvo_metrics/audit_scope.py \
  --config ${CONFIG} \
  --model ${MODEL} \
  --arm ${ARM} \
  --output ${EXPERIMENT_ROOT}/scope-audit.json
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=4 \
  cluster.segment_size=4 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=4 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  policy.generation.refit_transport=nccl_reshard \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.generation.vllm_cfg.async_engine=false \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.use_tqdm=false \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  ++grpo.skip_reference_policy_logprobs_calculation=false \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.seed=42 \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre,${WANDB_HOME}/.netrc:/root/.netrc
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE=4
export BASE_LOG_DIR=${EXPERIMENT_ROOT}

SBATCH_ARGS=(
  --nodes=8
  --gpus-per-node=4
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="${ACCOUNT}-p3477-qkvo.${MODEL}-${ARM}"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"QKVO MXFP8 scope validation"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nresult=%s\n' \
  "${REPO}" "${LOCAL_HEAD}" "${CONFIG}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
