#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
ARM=${ARM:-mxfp8}
MAX_STEPS=${MAX_STEPS:-20}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-pr3652-qwen30-async-ab-20260822}
GIT_REMOTE=${GIT_REMOTE:-origin}
EXPECTED_HEAD=${EXPECTED_HEAD:-}

case "${ARM}" in
  bf16)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml
    ROLLOUT_PRECISION=bfloat16
    QUANTIZATION_SCOPE=none
    MOE_BACKEND=triton
    ;;
  mxfp8)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml
    ROLLOUT_PRECISION=mxfp8
    QUANTIZATION_SCOPE=routed_experts_only
    MOE_BACKEND=flashinfer_trtllm
    ;;
  *) echo "ARM must be bf16 or mxfp8" >&2; exit 2 ;;
esac

if [[ "${ACTION}" == render ]]; then
  cat <<EOF
model=Qwen3-30B-A3B
mode=async_disaggregated_nccl_reshard
arm=${ARM}
rollout_precision=${ROLLOUT_PRECISION}
quantization_scope=${QUANTIZATION_SCOPE}
moe_backend=${MOE_BACKEND}
refit_transport=nccl_reshard
refit_prequantize=false
max_steps=${MAX_STEPS}
cuda_graphs=enabled
reference_logprobs=enabled
EOF
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-/home/sna/RL-pr3652-qwen30-async-ab-20260822}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
HF_HOME=${HF_HOME:-${ROOT}/hf_home}
WANDB_HOME=${WANDB_HOME:-${ROOT}/wandb_netrc_home}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/experiments/pr3652-qwen30-async-ab}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/${ARM}}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3652-qwen30-async-ab}
WANDB_NAME=${WANDB_NAME:-qwen30-async-${ARM}-${MAX_STEPS}step-${RUN_GROUP}}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/sna}

git -C "${REPO}" fetch "${GIT_REMOTE}" "${BRANCH}"
git -C "${REPO}" pull --ff-only "${GIT_REMOTE}" "${BRANCH}"
git -C "${REPO}" submodule update --init --recursive
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "${GIT_REMOTE}/${BRANCH}")
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

mkdir -p "${RUN_ROOT}/logs"
cat >"${RUN_ROOT}/metadata.env" <<EOF
source_commit=${LOCAL_HEAD}
branch=${BRANCH}
container=${CONTAINER}
hardware=GB200
model=Qwen3-30B-A3B
mode=async_disaggregated_nccl_reshard
arm=${ARM}
nodes=4
gpus_per_node=4
generation_nodes=2
training_precision=bf16
rollout_precision=${ROLLOUT_PRECISION}
quantization_scope=${QUANTIZATION_SCOPE}
cuda_graphs=enabled
moe_backend=${MOE_BACKEND}
refit_transport=nccl_reshard
refit_prequantize=false
logprob_work=previous_policy_and_reference_policy
max_steps=${MAX_STEPS}
aggregate_steps=3-20
seed=42
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

MXFP8_OVERRIDES=
if [[ "${ARM}" == mxfp8 ]]; then
  MXFP8_OVERRIDES=$(cat <<EOF
  policy.generation.vllm_cfg.precision=fp8 \\
  ++policy.generation.vllm_cfg.is_mx=true \\
  policy.generation.vllm_cfg.refit_prequantize=false \\
EOF
)
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HOME=/root
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=\${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=${LOCAL_SCRATCH}/nemo-rl-worker-cache/pr3652-qwen30-async-${LOCAL_HEAD}
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache
export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python
export UV_LOCK_TIMEOUT=7200
unset UV_PROJECT_ENVIRONMENT WANDB_API_KEY
mkdir -p "\${NEMO_RL_VENV_DIR}" "\${UV_CACHE_DIR}" "\${UV_PYTHON_INSTALL_DIR}"
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \\
  --config ${CONFIG} \\
  cluster.num_nodes=4 \\
  cluster.gpus_per_node=4 \\
  cluster.segment_size=2 \\
  policy.precision=bfloat16 \\
  policy.train_global_batch_size=2048 \\
  grpo.async_grpo.enabled=true \\
  grpo.async_grpo.max_trajectory_age_steps=1 \\
  grpo.async_grpo.in_flight_weight_updates=true \\
  policy.generation.colocated.enabled=false \\
  policy.generation.colocated.resources.num_nodes=2 \\
  policy.generation.colocated.resources.gpus_per_node=4 \\
  policy.generation.refit_transport=nccl_reshard \\
  policy.generation.real_quant_export_cpu_offload=false \\
  policy.generation.vllm_cfg.async_engine=true \\
  policy.generation.vllm_cfg.enforce_eager=false \\
  policy.generation.vllm_cfg.use_tqdm=false \\
  ++policy.generation.vllm_kwargs.moe_backend=${MOE_BACKEND} \\
  ++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \\
  policy.megatron_cfg.expert_tensor_parallel_size=1 \\
${MXFP8_OVERRIDES}  loss_fn.force_on_policy_ratio=false \\
  loss_fn.use_importance_sampling_correction=true \\
  ++grpo.skip_reference_policy_logprobs_calculation=false \\
  grpo.max_num_steps=${MAX_STEPS} \\
  grpo.seed=42 \\
  grpo.val_at_start=false \\
  ++grpo.val_at_end=false \\
  checkpointing.enabled=false \\
  logger.log_dir=${RUN_ROOT}/logs \\
  logger.wandb_enabled=true \\
  logger.tensorboard_enabled=true \\
  logger.monitor_gpus=true \\
  ++logger.wandb.entity=nvidia \\
  logger.wandb.project=${WANDB_PROJECT} \\
  logger.wandb.name=${WANDB_NAME}
/opt/nemo_rl_venv/bin/python tests/json_dump_tb_logs.py ${RUN_ROOT}/logs \\
  --output_path ${RUN_ROOT}/metrics.json || true
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,${WANDB_HOME}/.netrc:/root/.netrc
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE=4
export BASE_LOG_DIR=${RUN_ROOT}
export RAY_LOG_SYNC_FREQUENCY=30
export SETUP_COMMAND="mkdir -p ${LOCAL_SCRATCH}/nemo-rl-worker-cache ${LOCAL_SCRATCH}/uv-cache ${LOCAL_SCRATCH}/uv-python"

SBATCH_ARGS=(
  --nodes=4
  --gres=gpu:4
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment=2
  --job-name="${ACCOUNT}-pr3652.q30-${ARM}"
  --output="${RUN_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"Matched Qwen3-30B Async BF16 versus MXFP8 NCCL Reshard"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nresult=%s\n' \
  "${REPO}" "${LOCAL_HEAD}" "${CONFIG}" "${RUN_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
