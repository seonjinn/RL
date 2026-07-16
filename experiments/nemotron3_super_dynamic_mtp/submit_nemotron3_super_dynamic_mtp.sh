#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-dry-run}"
CLUSTER="${CLUSTER:-auto}"
VARIANT="${VARIANT:-dynamic_native_mtp_k5}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_STEPS="${MAX_STEPS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-nemotron3-super-mtp}"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${LUSTRE_ROOT}/.secrets/wandb_api_key}"
REPO_DIR="${REPO_DIR:-${LUSTRE_ROOT}/RL-nemotron3-super-dynamic-mtp-v024}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${LUSTRE_ROOT}/nemorl_reference_runs}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-${HF_HOME}/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e}"

if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER=lyris ;;
    *ptyche*) CLUSTER=ptyche ;;
    *)
      printf 'Set CLUSTER=lyris or CLUSTER=ptyche\n' >&2
      exit 2
      ;;
  esac
fi

case "${CLUSTER}" in
  lyris)
    PARTITION="${PARTITION:-gb200}"
    CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/containers/nemo_rl_nightly_20260715.sqsh}"
    ;;
  ptyche)
    PARTITION="${PARTITION:-batch}"
    CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/nemo-rl-cg/containers/nemo_rl_nightly_20260715.sqsh}"
    ;;
  *)
    printf 'Unsupported CLUSTER=%s\n' "${CLUSTER}" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in
  pr_baseline)
    RECIPE=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml
    ;;
  mtp_off)
    RECIPE=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-mtp-off.yaml
    ;;
  native_mtp_k5)
    RECIPE=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-native-mtp-k5.yaml
    ;;
  dynamic_native_mtp_k5)
    RECIPE=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-dynamic-native-mtp-k5.yaml
    ;;
  *)
    printf 'VARIANT must be pr_baseline, mtp_off, native_mtp_k5, or dynamic_native_mtp_k5; got %s\n' "${VARIANT}" >&2
    exit 2
    ;;
esac

NUM_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=8
RUN_TAG="${RUN_TAG:-super-v024-${VARIANT}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_TAG}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_DIR}}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"

overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "checkpointing.enabled=false"
  "checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "cluster.segment_size=${SEGMENT_SIZE}"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.tensorboard_enabled=true"
  "logger.log_dir=${RUN_DIR}/nemo_logs"
)

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  overrides+=(
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${RUN_TAG}"
  )
fi

command_parts=(
  env
  "NCCL_NVLS_ENABLE=0"
  "VLLM_USE_V2_MODEL_RUNNER=0"
  "VLLM_ATTENTION_BACKEND=TRITON_ATTN"
  "VLLM_ENGINE_READY_TIMEOUT_S=3600"
  "TORCH_CUDA_ARCH_LIST=10.0"
  "NRL_DISABLE_VLLM_PORT_OVERRIDE=1"
  "NRL_FORCE_REBUILD_VENVS=true"
  "NEMO_RL_VENV_DIR=/tmp/nrl-super-v024-${RUN_TAG}"
  "TMPDIR=/tmp"
  "HF_HOME=${HF_HOME}"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "PYTHONPATH=${REPO_DIR}"
  "WANDB_RUN_GROUP=nemotron3-super-v024-mtp"
  "WANDB_RESUME=never"
  "PYTHONFAULTHANDLER=1"
  "RAY_DEDUP_LOGS=0"
)

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  command_parts+=(
    "WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE}"
    bash
    -c
    'set +x; export WANDB_API_KEY="$(< "${WANDB_API_KEY_FILE}")"; exec "$@"'
    nemo-rl-with-wandb-key
  )
fi

command_parts+=(
  /opt/nemo_rl_venv/bin/python
  examples/run_grpo.py
  --config
  "${RECIPE}"
  "${overrides[@]}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  "--account=${ACCOUNT}"
  "--partition=${PARTITION}"
  "--nodes=${NUM_NODES}"
  --ntasks-per-node=1
  --exclusive
  "--time=${TIME_LIMIT}"
  "--segment=${SEGMENT_SIZE}"
  --dependency=
  "--chdir=${REPO_DIR}"
  "--job-name=${ACCOUNT}-super-mtp.${VARIANT}"
  "--output=${RUN_DIR}/slurm-%j.out"
  --comment=metrics
)

has_safetensors_checkpoint() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/model.safetensors" || \
     -f "${checkpoint_dir}/model.safetensors.index.json" ]]
}

case "${MODE}" in
  dry-run)
    printf '[DRY-RUN] command %s\n' "${command}"
    printf '[DRY-RUN] environment CONTAINER=%s MOUNTS=%s BASE_LOG_DIR=%s\n' \
      "${CONTAINER}" "${MOUNTS}" "${BASE_LOG_DIR}"
    printf '[DRY-RUN] sbatch'
    printf ' %s' "${sbatch_args[@]}"
    printf ' %s\n' "${REPO_DIR}/ray.sub"
    ;;
  test-only|submit)
    if [[ ! -f "${CONTAINER}" ]]; then
      printf 'Container does not exist: %s\n' "${CONTAINER}" >&2
      exit 2
    fi
    if ! has_safetensors_checkpoint "${MODEL_SNAPSHOT}"; then
      printf 'Model snapshot is incomplete: %s\n' "${MODEL_SNAPSHOT}" >&2
      exit 2
    fi
    if [[ "${WANDB_ENABLED}" == "true" && ! -r "${WANDB_API_KEY_FILE}" ]]; then
      printf 'W&B API key file is unavailable: %s\n' "${WANDB_API_KEY_FILE}" >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty || \
       ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
      printf 'Submission requires a clean tracked checkout\n' >&2
      exit 2
    fi
    for tracked_file in "${RECIPE}" experiments/nemotron3_super_dynamic_mtp/submit_nemotron3_super_dynamic_mtp.sh; do
      if ! git -C "${REPO_DIR}" ls-files --error-unmatch "${tracked_file}" >/dev/null 2>&1; then
        printf 'Submission dependency must be committed: %s\n' "${tracked_file}" >&2
        exit 2
      fi
    done
    if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
      printf 'HEAD must be pushed before submission\n' >&2
      exit 2
    fi

    mkdir -p "${RUN_DIR}"
    {
      printf 'run_tag=%s\ncluster=%s\nvariant=%s\n' "${RUN_TAG}" "${CLUSTER}" "${VARIANT}"
      printf 'repo_head=%s\n' "$(git -C "${REPO_DIR}" rev-parse HEAD)"
      printf 'container=%s\nrecipe=%s\n' "${CONTAINER}" "${RECIPE}"
      printf 'model_snapshot=%s\n' "${MODEL_SNAPSHOT}"
      printf 'max_steps=%s\nnum_nodes=%s\ngpus_per_node=%s\nsegment=%s\n' \
        "${MAX_STEPS}" "${NUM_NODES}" "${GPUS_PER_NODE}" "${SEGMENT_SIZE}"
      printf 'vllm_version=0.24.0\ncuda_graph_enabled=true\nmodel_runner=v1\n'
      printf 'wandb_project=%s\ncommand=%s\n' "${WANDB_PROJECT}" "${command}"
    } >"${RUN_DIR}/provenance.txt"

    export COMMAND="${command}" CONTAINER MOUNTS BASE_LOG_DIR
    if [[ "${MODE}" == "test-only" ]]; then
      sbatch --test-only "${sbatch_args[@]}" "${REPO_DIR}/ray.sub"
    else
      job_id="$(sbatch --parsable "${sbatch_args[@]}" "${REPO_DIR}/ray.sub")"
      printf 'job_id=%s\nrun_dir=%s\n' "${job_id}" "${RUN_DIR}"
    fi
    ;;
  *)
    printf 'MODE must be dry-run, test-only, or submit; got %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
