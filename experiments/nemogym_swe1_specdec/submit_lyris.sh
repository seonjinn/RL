#!/usr/bin/env bash
# NemoGym SWE1 rollout benchmark (PR #3243 eval mode) with SpecDec variants on Ptyche.
# Stack: RL-wt-nemogym-dynsd-lyris worktree (main@0715 + vllm 0.25.1 + eagle3 fullcg + PR 3243).
set -euo pipefail

MODE="${MODE:-dry-run}"
VARIANT="${VARIANT:-baseline}"   # baseline | eagle3_k3 | eagle3_k3_dynsd
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
TIME_LIMIT="${TIME_LIMIT:-02:30:00}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
NUM_GENS="${NUM_GENS:-4}"
WT="/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-wt-nemogym-dynsd-lyris"
CONFIG="${CONFIG:-${WT}/examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml}"
DATA="${DATA:-${WT}/data/swe1/val-split.jsonl}"
CONTAINER="/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh"
MOUNTS="/lustre:/lustre,/dev/fuse:/dev/fuse"
HF_HOME="/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
TARGET_SNAPSHOT_GLOB="${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots"
DRAFT_SNAPSHOT="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"
DYNAMIC_SD_SCHEDULE="[[1,8,3],[9,32,2],[33,512,1]]"
METRICS="${METRICS:-false}"
CAP512="${CAP512:-false}"
TAG="${TAG:-${VARIANT}}"
RUN_TAG="${TAG}-$(date +%m%d-%H%M%S)"
RUN_DIR="${WT}/experiments/nemogym_swe1_specdec/runs/${RUN_TAG}"

case "${VARIANT}" in
  baseline)        K=0; SCHED="" ;;
  eagle3_k3)       K=3; SCHED="" ;;
  eagle3_k3_dynsd) K=3; SCHED="${DYNAMIC_SD_SCHEDULE}" ;;
  *) echo "bad VARIANT ${VARIANT}" >&2; exit 2 ;;
esac

overrides=(
  "data.train.data_path=${DATA}"
  "data.validation.data_path=${DATA}"
  "cluster.num_nodes=1"
  "cluster.gpus_per_node=4"
  "grpo.num_prompts_per_step=${NUM_PROMPTS}"
  "grpo.num_generations_per_prompt=${NUM_GENS}"
  "logger.wandb_enabled=false"
  "logger.tensorboard_enabled=false"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE"
)
if [[ "${K}" -gt 0 ]]; then
  overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_SNAPSHOT}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${K}"
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
  )
fi
if [[ -n "${SCHED}" ]]; then
  overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=${SCHED}"
  )
fi
if [[ "${METRICS}" == "true" ]]; then
  overrides+=(
    "++policy.generation.vllm_kwargs.cudagraph_metrics=true"
    "policy.generation.vllm_cfg.enable_vllm_metrics_logger=true"
  )
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  overrides+=(${EXTRA_OVERRIDES})
fi
if [[ "${CAP512}" == "true" ]]; then
  overrides+=(
    "++policy.generation.vllm_kwargs.compilation_config.max_cudagraph_capture_size=512"
  )
fi

command_env=(
  "VLLM_USE_V2_MODEL_RUNNER=1"
  "HF_HOME=${HF_HOME}"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "PYTHONPATH=${WT}"
  "NEMO_RL_VENV_DIR=/tmp/nemorl-ngym-${RUN_TAG}"
  "NRL_FORCE_REBUILD_VENVS=true"
  "TRITON_CACHE_DIR=/tmp/nemorl-triton-${RUN_TAG}"
  "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-inductor-${RUN_TAG}"
  "UV_LOCK_TIMEOUT=3600"
  "PYTHONFAULTHANDLER=1"
  "RAY_DEDUP_LOGS=0"
  "NRL_SWE_UTIL_SYNTH=/lustre/fsw/coreai_dlalgo_llm/users/sna/swe_util_synth"
  "APPTAINER_CACHEDIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/apptainer_cache"
  "APPTAINER_TMPDIR=/tmp/apptainer-${RUN_TAG}"
)
if [[ -n "${SCHED}" ]]; then
  command_env+=(
    "NRL_VENV_POST_SYNC_SCRIPT=${WT}/experiments/vllm_0251_eagle3_perfcfg/apply_vllm0251_dynamic_sd_cg_fix.py"
    "NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
  )
fi

command_parts=(
  env
  "${command_env[@]}"
  /opt/nemo_rl_venv/bin/python
  "${WT}/examples/nemo_gym/run_grpo_rollout_benchmark.py"
  --config
  "${CONFIG}"
  "${overrides[@]}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks-per-node=1
  --exclusive
  --time="${TIME_LIMIT}"
  --job-name="${ACCOUNT}-nemorl.ngym-swe1-${VARIANT}"
  --output="${RUN_DIR}/slurm-%j.out"
)

case "${MODE}" in
  dry-run)
    printf '[DRY-RUN] %s\n' "${command}"
    ;;
  submit)
    if ! ls "${TARGET_SNAPSHOT_GLOB}"/*/model.safetensors.index.json >/dev/null 2>&1; then
      echo "Target Thinking-2507 snapshot not complete yet" >&2
      exit 3
    fi
    if [[ "${K}" -gt 0 && ! -d "${DRAFT_SNAPSHOT}" ]]; then
      echo "drafter missing" >&2
      exit 3
    fi
    mkdir -p "${RUN_DIR}"
    {
      printf 'run_tag=%s\nvariant=%s\nrepo_head=%s\ncontainer=%s\nschedule=%s\ncommand=%s\n' \
        "${RUN_TAG}" "${VARIANT}" "$(git -C "${WT}" rev-parse HEAD)" "${CONTAINER}" "${SCHED:-disabled}" "${command}"
    } > "${RUN_DIR}/provenance.txt"
    export COMMAND="${command}" CONTAINER MOUNTS GPUS_PER_NODE=4 BASE_LOG_DIR="${RUN_DIR}"
    job_id="$(cd "${RUN_DIR}" && sbatch --parsable "${sbatch_args[@]}" "${WT}/ray.sub")"
    printf 'job_id=%s\nrun_dir=%s\n' "${job_id}" "${RUN_DIR}"
    ;;
  *) echo "MODE must be dry-run or submit" >&2; exit 2 ;;
esac
