#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

MODE="${1:-test-only}"
MODEL_SELECTION="${2:-all}"
VARIANT_SELECTION="${3:-all}"
MAX_STEPS="${MAX_STEPS:-20}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]}"
ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
PARTITION="${PARTITION:-batch_long}"
USE_GRES="${USE_GRES:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-dynamicsd-aws-dfw}"

if [[ -z "${REPO_DIR:-}" ]]; then
  logical_pwd="$(pwd -L)"
  repo_prefix="$(git rev-parse --show-prefix)"
  if [[ -n "${repo_prefix}" ]]; then
    REPO_DIR="${logical_pwd%/${repo_prefix%/}}"
  else
    REPO_DIR="${logical_pwd}"
  fi
fi
AWS_ROOT="${AWS_ROOT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna}"
CONTAINER="${CONTAINER:-${AWS_ROOT}/containers/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-${AWS_ROOT}/hf_home}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${AWS_ROOT}/.secrets/wandb_api_key}"
WANDB_NETRC_HOME="${WANDB_NETRC_HOME:-}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
RUN_TAG="${RUN_TAG:-vllm024-dynamicsd-step20-20260707}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_024_upgrade/runs/${RUN_TAG}}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-20001}"
WALLTIME="${WALLTIME:-04:00:00}"
TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
CONTAINER_SHA256="${CONTAINER_SHA256:-}"

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

if [[ "${MODE}" == "submit" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -r "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
  elif [[ -n "${WANDB_NETRC_HOME}" && -r "${WANDB_NETRC_HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(awk '$1 == "password" {print $2; exit}' "${WANDB_NETRC_HOME}/.netrc")"
  else
    echo "ERROR: set WANDB_API_KEY or create ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  export WANDB_API_KEY
fi

if [[ "${MODE}" == "submit" ]]; then
  if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty \
    || ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
    echo "ERROR: submit requires a clean tracked checkout" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" ls-files --error-unmatch \
    experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh >/dev/null 2>&1; then
    echo "ERROR: launcher must be committed before submit" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
    echo "ERROR: HEAD is not present on a known remote branch" >&2
    exit 2
  fi
fi

case "${MODEL_SELECTION}" in
  all) models=(qwen30ba3b qwen32b qwen235b) ;;
  qwen30ba3b|qwen32b|qwen235b) models=("${MODEL_SELECTION}") ;;
  *)
    echo "ERROR: model must be all, qwen30ba3b, qwen32b, or qwen235b" >&2
    exit 2
    ;;
esac

case "${VARIANT_SELECTION}" in
  all) variants=(baseline eagle3_k5 eagle3_k7 eagle3_k9 dynamic) ;;
  aggressive) variants=(eagle3_k7 eagle3_k9) ;;
  baseline|eagle3_k5|eagle3_k7|eagle3_k9|dynamic) variants=("${VARIANT_SELECTION}") ;;
  *)
    echo "ERROR: variant must be all, aggressive, baseline, eagle3_k5, eagle3_k7, eagle3_k9, or dynamic" >&2
    exit 2
    ;;
esac

submit_one() {
  local model="$1"
  local variant="$2"
  local recipe
  local draft_model
  local draft_k="${STATIC_K}"
  local nodes
  local model_port_offset

  case "${model}" in
    qwen30ba3b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
      draft_model="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
      nodes=4
      model_port_offset=0
      ;;
    qwen32b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      draft_model="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
      nodes=4
      model_port_offset=1000
      ;;
    qwen235b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml"
      draft_model="${QWEN235_DRAFT_MODEL:-${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba}"
      nodes=16
      model_port_offset=2000
      ;;
  esac

  if [[ "${MODE}" != "dry-run" && "${variant}" != "baseline" && ! -d "${draft_model}" ]]; then
    echo "ERROR: draft model directory not found: ${draft_model}" >&2
    exit 2
  fi

  local run_dir="${EXPERIMENT_ROOT}/${model}/${variant}"
  local wandb_run_id="${RUN_TAG}-${ATTEMPT_ID}-${model}-${variant}"
  local wandb_name="${wandb_run_id}"
  local triton_cache_dir="/tmp/nemorl-vllm024-triton-${RUN_TAG}-${model}-${variant}"
  local inductor_cache_dir="/tmp/nemorl-vllm024-inductor-${RUN_TAG}-${model}-${variant}"
  local variant_port_offset=0
  case "${variant}" in
    eagle3_k5)
      draft_k=5
      variant_port_offset=200
      ;;
    dynamic)
      variant_port_offset=400
      ;;
    eagle3_k7)
      draft_k=7
      variant_port_offset=600
      ;;
    eagle3_k9)
      draft_k=9
      variant_port_offset=800
      ;;
  esac
  local vllm_port=$((VLLM_PORT_BASE + model_port_offset + variant_port_offset))

  local overrides=(
    "grpo.max_num_steps=${MAX_STEPS}"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
    "policy.generation.vllm_cfg.enforce_eager=false"
    "policy.generation.temperature=1.0"
    "policy.generation.top_p=1.0"
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE"
    "cluster.segment_size=${nodes}"
    "logger.wandb_enabled=true"
    "logger.tensorboard_enabled=false"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${wandb_name}"
    "++logger.wandb.entity=${WANDB_ENTITY}"
    "logger.log_dir=${run_dir}/nemo_logs"
  )
  if [[ "${variant}" != "baseline" ]]; then
    local specdec_overrides=(
      "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
      "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
      "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
      "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
    )
    if [[ "${variant}" == "dynamic" ]]; then
      specdec_overrides+=(
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=${DYNAMIC_SCHEDULE}"
      )
    fi
    overrides+=("${specdec_overrides[@]}")
  fi

  local command_env=(
    "WANDB_RUN_ID=${wandb_run_id}"
    "WANDB_RUN_GROUP=${RUN_TAG}"
    "WANDB_RESUME=never"
    "VLLM_PORT=${vllm_port}"
    "PYTHONPATH=${REPO_DIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  local command_parts=(
    env
    "${command_env[@]}"
    /opt/nemo_rl_venv/bin/python
    examples/run_grpo.py
    --config
    "${recipe}"
    "${overrides[@]}"
  )
  local command
  printf -v command '%q ' "${command_parts[@]}"
  command="${command% }"

  local environment=(
    "CONTAINER=${CONTAINER}"
    "MOUNTS=/lustre:/lustre"
    "CONTAINER_WORKDIR=${REPO_DIR}"
    "COMMAND=${command}"
    "BASE_LOG_DIR=${run_dir}"
    "GPUS_PER_NODE=4"
    "HF_HOME=${HF_HOME}"
    "NEMO_RL_VENV_DIR=${run_dir}/venvs"
    "NRL_FORCE_REBUILD_VENVS=true"
    "PYTHONPATH=${REPO_DIR}"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=60"
    "TMPDIR=${TMPDIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes="${nodes}"
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment="${nodes}"
    --job-name="${ACCOUNT}-nemorl.dynamicsd-${model}-${variant}"
    --output="${run_dir}/slurm-%j.out"
    --comment=metrics
  )
  if [[ "${USE_GRES}" == "true" ]]; then
    sbatch_args+=(--gres=gpu:4)
  fi

  case "${MODE}" in
    dry-run)
      printf '[DRY-RUN] env'
      printf ' %q' "${environment[@]}"
      printf ' sbatch'
      printf ' %q' "${sbatch_args[@]}"
      printf ' %q\n' "${REPO_DIR}/ray.sub"
      printf '[DRY-RUN] command %s\n' "${command}"
      printf '[DRY-RUN] wandb https://wandb.ai/%s/%s/runs/%s\n' \
        "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${wandb_run_id}"
      ;;
    test-only)
      mkdir -p "${run_dir}"
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" "${REPO_DIR}/ray.sub"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local job_id
      job_id="$(env "${environment[@]}" sbatch --parsable "${sbatch_args[@]}" "${REPO_DIR}/ray.sub")"
      local manifest="${EXPERIMENT_ROOT}/submissions.tsv"
      if [[ ! -f "${manifest}" ]]; then
        printf 'timestamp\tmodel\tvariant\tjob_id\tnodes\tsegment\tcommit\twandb_run_id\twandb_url\trecipe\tdraft_model\tcontainer\tcontainer_sha256\tmax_steps\tstatic_k\tdynamic_schedule\tcommand\n' > "${manifest}"
      fi
      local resolved_container
      resolved_container="$(readlink -f "${CONTAINER}")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${model}" "${variant}" "${job_id}" \
        "${nodes}" "${nodes}" "$(git -C "${REPO_DIR}" rev-parse HEAD)" \
        "${wandb_run_id}" "https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${wandb_run_id}" \
        "${recipe}" "${draft_model}" "${resolved_container}" "${CONTAINER_SHA256}" \
        "${MAX_STEPS}" "${draft_k}" "${DYNAMIC_SCHEDULE}" "${command}" >> "${manifest}"
      ;;
    *)
      echo "ERROR: mode must be dry-run, test-only, or submit" >&2
      exit 2
      ;;
  esac
}

if [[ "${MODE}" != "dry-run" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}"
fi
for model in "${models[@]}"; do
  for variant in "${variants[@]}"; do
    submit_one "${model}" "${variant}"
  done
done
