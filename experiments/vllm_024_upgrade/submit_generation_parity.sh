#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

MODE="${1:-test-only}"
VARIANT_SELECTION="${2:-all}"
DECODE_SELECTION="${3:-all}"

if [[ -z "${REPO_DIR:-}" ]]; then
  logical_pwd="$(pwd -L)"
  repo_prefix="$(git rev-parse --show-prefix)"
  if [[ -n "${repo_prefix}" ]]; then
    REPO_DIR="${logical_pwd%/${repo_prefix%/}}"
  else
    REPO_DIR="${logical_pwd}"
  fi
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
LYRIS_ROOT="${LYRIS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
CONTAINER="${CONTAINER:-${LYRIS_ROOT}/containers/nemo_rl_nightly_20260707.sqsh}"
HF_HOME="${HF_HOME:-${LYRIS_ROOT}/hf_home}"
TARGET_MODEL="${TARGET_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
DRAFT_MODEL="${DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_DIR}/experiments/vllm_024_upgrade/data/parity_prompts.jsonl}"
RUN_TAG="${RUN_TAG:-vllm024-generation-parity-20260709}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${LYRIS_ROOT}/experiments/vllm024-generation-parity/${RUN_TAG}}"
WALLTIME="${WALLTIME:-01:00:00}"
TARGET_TP="${TARGET_TP:-2}"
DRAFT_TP="${DRAFT_TP:-1}"
RUNNER_GPUS_PER_NODE="${RUNNER_GPUS_PER_NODE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
PROMPT_LIMIT="${PROMPT_LIMIT:-4}"
SAMPLED_REPEATS="${SAMPLED_REPEATS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"

case "${VARIANT_SELECTION}" in
  all) variants=(baseline eagle3_k5) ;;
  baseline|eagle3_k5) variants=("${VARIANT_SELECTION}") ;;
  *)
    echo "ERROR: variant must be all, baseline, or eagle3_k5" >&2
    exit 2
    ;;
esac

case "${DECODE_SELECTION}" in
  all) decode_modes=(greedy sampled) ;;
  greedy|sampled) decode_modes=("${DECODE_SELECTION}") ;;
  *)
    echo "ERROR: decode mode must be all, greedy, or sampled" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" != "dry-run" ]]; then
  for path in "${CONTAINER}" "${TARGET_MODEL}" "${PROMPT_DATA}"; do
    if [[ ! -e "${path}" ]]; then
      echo "ERROR: required path not found: ${path}" >&2
      exit 2
    fi
  done
fi

if [[ "${MODE}" == "submit" ]]; then
  if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty \
    || ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
    echo "ERROR: submit requires a clean tracked checkout" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" ls-files --error-unmatch \
    experiments/vllm_024_upgrade/submit_generation_parity.sh >/dev/null 2>&1; then
    echo "ERROR: launcher must be committed before submit" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
    echo "ERROR: HEAD is not present on a known remote branch" >&2
    exit 2
  fi
fi

submit_one() {
  local variant="$1"
  local decode_mode="$2"
  local run_name="${RUN_TAG}-${variant}-${decode_mode}"
  local run_dir="${EXPERIMENT_ROOT}/${variant}/${decode_mode}"
  local output_jsonl="${run_dir}/samples.jsonl"
  local metadata_json="${run_dir}/metadata.json"
  local ray_log_dir="/tmp/nrp-${variant}-${decode_mode}"
  local samples_per_prompt=1
  if [[ "${decode_mode}" == "sampled" ]]; then
    samples_per_prompt="${SAMPLED_REPEATS}"
  fi

  local command_parts=(
    env
    "PYTHONPATH=${REPO_DIR}"
    "HF_HOME=${HF_HOME}"
    "TRITON_CACHE_DIR=/tmp/nemorl-parity-triton-${variant}-${decode_mode}"
    "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-parity-inductor-${variant}-${decode_mode}"
    "TORCH_CUDA_ARCH_LIST=10.0a"
    /opt/nemo_rl_venv/bin/python
    experiments/vllm_024_upgrade/run_generation_parity.py
    --model "${TARGET_MODEL}"
    --tokenizer "${TARGET_MODEL}"
    --method eagle3
    --num-speculative-tokens 5
    --target-tp "${TARGET_TP}"
    --draft-tp "${DRAFT_TP}"
    --num-nodes 1
    --gpus-per-node "${RUNNER_GPUS_PER_NODE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --prompt-data "${PROMPT_DATA}"
    --prompt-limit "${PROMPT_LIMIT}"
    --samples-per-prompt "${samples_per_prompt}"
    --batch-size "${BATCH_SIZE}"
    --mode "${decode_mode}"
    --temperature "$([[ "${decode_mode}" == "greedy" ]] && printf '0.0' || printf '1.0')"
    --top-p 1.0
    --output-jsonl "${output_jsonl}"
    --metadata-json "${metadata_json}"
    --ray-log-dir "${ray_log_dir}"
  )
  if [[ "${variant}" == "eagle3_k5" ]]; then
    if [[ "${MODE}" != "dry-run" && ! -d "${DRAFT_MODEL}" ]]; then
      echo "ERROR: draft model directory not found: ${DRAFT_MODEL}" >&2
      exit 2
    fi
    command_parts+=(--draft-model "${DRAFT_MODEL}")
  fi

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
    "PYTHONPATH=${REPO_DIR}"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=30"
    "TMPDIR=${TMPDIR}"
  )
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes=1
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment=1
    --job-name="${ACCOUNT}-nemorl.parity-${variant}-${decode_mode}"
    --output="${run_dir}/slurm-%j.out"
    --comment=metrics
  )

  case "${MODE}" in
    dry-run)
      printf '[DRY-RUN] %s env' "${run_name}"
      printf ' %q' "${environment[@]}"
      printf ' sbatch'
      printf ' %q' "${sbatch_args[@]}"
      printf ' %q\n' "${REPO_DIR}/ray.sub"
      printf '[DRY-RUN] command %s\n' "${command}"
      ;;
    test-only)
      mkdir -p "${run_dir}"
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" \
        "${REPO_DIR}/ray.sub"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local job_id
      job_id="$(env "${environment[@]}" sbatch --parsable "${sbatch_args[@]}" \
        "${REPO_DIR}/ray.sub")"
      local manifest="${EXPERIMENT_ROOT}/submissions.tsv"
      if [[ ! -f "${manifest}" ]]; then
        printf 'timestamp\tvariant\tmode\tjob_id\tcommit\tcontainer\ttarget_model\tdraft_model\tprompt_data\toutput_jsonl\tmetadata_json\tcommand\n' >"${manifest}"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${variant}" "${decode_mode}" "${job_id}" \
        "$(git -C "${REPO_DIR}" rev-parse HEAD)" "$(readlink -f "${CONTAINER}")" \
        "${TARGET_MODEL}" "$([[ "${variant}" == "baseline" ]] && printf '' || printf '%s' "${DRAFT_MODEL}")" \
        "${PROMPT_DATA}" "${output_jsonl}" "${metadata_json}" "${command}" >>"${manifest}"
      printf '%s\t%s\t%s\n' "${job_id}" "${variant}" "${decode_mode}"
      ;;
    *)
      echo "ERROR: mode must be dry-run, test-only, or submit" >&2
      exit 2
      ;;
  esac
}

for variant in "${variants[@]}"; do
  for decode_mode in "${decode_modes[@]}"; do
    submit_one "${variant}" "${decode_mode}"
  done
done
