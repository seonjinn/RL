#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-q30-dapomath47k-dspark-20260911
readonly RECIPE="${SOURCE_ROOT}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml"
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
readonly VLLM_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
readonly TARGET_MODEL=/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B
readonly DRAFTER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/specdec_ptv23/ptv3_swa/sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/exported-checkpoint-44000
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-performance40k-vllm0251-dspark-20260911
readonly ACCOUNT="${Q30_PERF40K_ACCOUNT:-coreai_dlalgo_nemorl}"
readonly MAX_STEPS="${Q30_PERF40K_MAX_STEPS:-1}"
readonly MAX_NUM_SEQS=128

usage() {
  echo "usage: $0 --render|--test-only|--submit baseline|dspark_k3|dspark_k5" >&2
  exit 2
}

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q30_PERF40K_MAX_STEPS must be a positive integer: ${MAX_STEPS}" >&2
  exit 2
fi

mode="${1:-}"
arm="${2:-}"
case "${mode}" in --render|--test-only|--submit) ;; *) usage ;; esac

k=0
arm_label=Baseline
case "${arm}" in
  baseline) ;;
  dspark_k3|dspark_k5)
    k="${arm##*_k}"
    arm_label="DSparkK${k}"
    ;;
  *) usage ;;
esac

case "${k}" in
  0) capture_sizes='[1,2,4,8,16,32,64,128]' ;;
  3) capture_sizes='[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512]' ;;
  5) capture_sizes='[1,2,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,384,640,768]' ;;
  *) usage ;;
esac
readonly capture_sizes

walltime=04:00:00
if ((MAX_STEPS > 1)); then
  walltime=12:00:00
fi
readonly walltime

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="Qwen3-30BA3B-Performance40K-vLLM0251-${arm_label}-${MAX_STEPS}step-FAP-${timestamp}"
artifact_dir="${DURABLE_ROOT}/${run_id}"

setup_dspark=''
if ((k > 0)); then
  setup_dspark="; ${VLLM_PYTHON} ${SOURCE_ROOT}/experiments/qwen3_30ba3b_bf16_flashinfer_specdec_latest_main_20260909/prepare_vllm_dspark_fap_overlay.py --overlay-root \"\${Q30_VLLM_OVERLAY}\""
fi

# Keep the native performance-40K workload and parallelism in the recipe.
# Only physical 32-GPU topology, local artifacts, matched runtime, and the
# SpecDec arm are supplied here.
overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "policy.model_name=${TARGET_MODEL}"
  "policy.tokenizer.name=${TARGET_MODEL}"
  'policy.precision=bfloat16'
  'policy.draft.enabled=false'
  '++policy.offload_optimizer_for_refit=false'
  'policy.generation.refit_transport=null'
  'policy.generation.vllm_cfg.refit_with_reload_api=false'
  'policy.generation.vllm_cfg.enforce_eager=false'
  'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm'
  "++policy.generation.vllm_kwargs.max_num_seqs=${MAX_NUM_SEQS}"
  '++policy.generation.vllm_kwargs.max_num_batched_tokens=40960'
  '++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE'
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes}"
  'cluster.gpus_per_node=4'
  'cluster.num_nodes=8'
  '++cluster.segment_size=4'
  'logger.wandb_enabled=true'
  'logger.wandb.project=sna-specdec'
  '++logger.wandb.group=q30-performance40k-vllm0251-dspark-fixed'
  "logger.wandb.name=${run_id}"
  "logger.log_dir=${artifact_dir}/logs"
)
if [[ "${arm}" == baseline ]]; then
  overrides+=('++policy.generation.vllm_kwargs.speculative_config=null')
else
  overrides+=(
    '++policy.generation.vllm_kwargs.speculative_config.method=dspark'
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFTER}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${k}"
    '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1'
    '++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN'
    '++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false'
  )
fi
printf -v override_args ' %q' "${overrides[@]}"

render() {
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.${run_id}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=${walltime}
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export PATH=/cm/local/apps/slurm/25.11/bin:\${PATH}
test -n "\${WANDB_API_KEY:-}"
test -z "\$(git -C ${SOURCE_ROOT} status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
$(if ((k > 0)); then printf 'test -f "%s/model.safetensors"\n' "${DRAFTER}"; fi)
mkdir -p "${artifact_dir}"
git -C "${SOURCE_ROOT}" rev-parse HEAD | tee "${artifact_dir}/source_sha.txt"
git -C "${SOURCE_ROOT}" submodule status --recursive >"${artifact_dir}/submodules.txt"
export CONTAINER="${CONTAINER}"
export MOUNTS=/lustre:/lustre,/home:/home,/raid:/raid
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${artifact_dir}"
export Q30_NODE_ROOT="/raid/scratch/sna/q30-performance40k-dspark-\${SLURM_JOB_ID}"
export Q30_MCORE_SOURCE="${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q30_MCORE_OVERLAY="\${Q30_NODE_ROOT}/mcore-overlay"
export Q30_VLLM_OVERLAY="\${Q30_NODE_ROOT}/vllm-overlay"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q30_MCORE_OVERLAY}"; cp -a "\${Q30_MCORE_SOURCE}/megatron" "\${Q30_MCORE_OVERLAY}/"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"; test -x ${VLLM_PYTHON}; ${VLLM_PYTHON} -c "import vllm"${setup_dspark}'
export COMMAND="cd ${SOURCE_ROOT} && /opt/nemo_rl_venv/bin/python examples/run_grpo.py --config ${RECIPE}${override_args}"
exec bash "${SOURCE_ROOT}/ray.sub"
EOF
}

load_wandb_api_key() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return
  fi
  if [[ -r "${HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(python3 - <<'PY'
from netrc import netrc

credentials = netrc().authenticators("api.wandb.ai")
print(credentials[2] if credentials else "")
PY
)"
    export WANDB_API_KEY
  fi
  test -n "${WANDB_API_KEY:-}"
}

if [[ "${mode}" == --render ]]; then
  render
  exit 0
fi

test -e "${SOURCE_ROOT}/.git"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)"
test -r "${CONTAINER}"
test -f "${RECIPE}"
test -d "${TARGET_MODEL}"
[[ "${arm}" == baseline ]] || test -f "${DRAFTER}/model.safetensors"
load_wandb_api_key
mkdir -p "${artifact_dir}"
sbatch_path="${artifact_dir}/job.sbatch"
render >"${sbatch_path}"
chmod 700 "${sbatch_path}"
sbatch --test-only "${sbatch_path}" 2>&1 | tee "${artifact_dir}/test-only.txt"
if [[ "${mode}" == --test-only ]]; then
  exit 0
fi
sbatch "${sbatch_path}" | tee "${artifact_dir}/submission.txt"
