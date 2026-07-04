#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
NEMO_RL_SHA="1271b1530181a7378e40de40b4b46ad223e6596c"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

QWEN30_MODEL="${QWEN30_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
QWEN32_MODEL="${QWEN32_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
QWEN235B_MODEL="${QWEN235B_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65}"
QWEN32_EAGLE3_MODEL="${QWEN32_EAGLE3_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
QWEN235B_EAGLE3_MODEL="${QWEN235B_EAGLE3_MODEL:-${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba}"
SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/build_deps/arctic-inference-0.1.1-py313-native}"

MODELS="${MODELS:-qwen30ba3b qwen32 qwen235b}"
MODES="${MODES:-sync async1off}"
METHODS="${METHODS:-baseline suffix eagle3}"
MAX_STEPS="${MAX_STEPS:-20}"
DRY_RUN="${DRY_RUN:-false}"

RUN_ID="${RUN_ID:-20260704_lyris_nemorl_math_continuation_tritonattn}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
WANDB_HOME="${WANDB_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/wandb_netrc_home}"
WANDB_PROJECT="${WANDB_PROJECT:-sna-nemorl-specdec-lyris}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-05:00:00}"
OUT="${OUT:-${ROOT_DIR}/docs/latest_lyris_nemorl_math_continuation_20260704_jobs.csv}"

normalize_list() {
  local raw="$1"
  printf '%s\n' "${raw//,/ }"
}

canonical_mode() {
  case "$1" in
    sync) printf 'sync' ;;
    async1off|async-1off) printf 'async1off' ;;
    *)
      echo "ERROR: unsupported mode: $1" >&2
      return 2
      ;;
  esac
}

validate_model() {
  case "$1" in
    qwen30ba3b|qwen32|qwen235b) ;;
    *)
      echo "ERROR: unsupported model: $1" >&2
      return 2
      ;;
  esac
}

validate_method() {
  case "$1" in
    baseline|suffix|eagle3) ;;
    *)
      echo "ERROR: unsupported method: $1" >&2
      return 2
      ;;
  esac
}

supports_method() {
  case "$1:$2" in
    qwen30ba3b:suffix) return 0 ;;
    qwen32:suffix|qwen32:eagle3) return 0 ;;
    qwen235b:baseline|qwen235b:suffix|qwen235b:eagle3) return 0 ;;
    *) return 1 ;;
  esac
}

model_contract() {
  local model="$1"
  model_label="${model}"
  eagle3_model=""
  case "${model}" in
    qwen30ba3b)
      target_model="${QWEN30_MODEL}"
      ;;
    qwen32)
      target_model="${QWEN32_MODEL}"
      eagle3_model="${QWEN32_EAGLE3_MODEL}"
      ;;
    qwen235b)
      target_model="${QWEN235B_MODEL}"
      eagle3_model="${QWEN235B_EAGLE3_MODEL}"
      ;;
  esac
}

mode_contract() {
  local model="$1"
  local mode="$2"
  case "${model}:${mode}" in
    qwen30ba3b:sync)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
      nodes=4
      segment=4
      ;;
    qwen30ba3b:async1off)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml"
      nodes=4
      segment=4
      ;;
    qwen32:sync)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      nodes=4
      segment=4
      ;;
    qwen32:async1off)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml"
      nodes=8
      segment=8
      ;;
    qwen235b:sync)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g.yaml"
      nodes=32
      segment=16
      ;;
    qwen235b:async1off)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml"
      nodes=32
      segment=16
      ;;
  esac
}

method_contract() {
  local method="$1"
  spec_k=0
  draft_model=""
  source_site=""
  case "${method}" in
    baseline) ;;
    suffix)
      spec_k=32
      source_site="${SOURCE_VLLM_SITE}"
      ;;
    eagle3)
      spec_k=3
      draft_model="${eagle3_model}"
      ;;
  esac
}

network_contract() {
  case "$1" in
    qwen30ba3b|qwen32) printf 'sharp' ;;
    qwen235b) printf '' ;;
  esac
}

render_command() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local log_root="${RUN_ROOT}/logs/${model}_${mode}_${method}"
  local checkpoint_root="${RUN_ROOT}/megatron_checkpoints/${model}_${mode}_${method}"
  local training_checkpoint_root="${RUN_ROOT}/training_checkpoints/${model}_${mode}_${method}"
  local node_cache="/tmp/${USER:-sna}/${RUN_ID}_${model}_${mode}_${method}"
  local pythonpath="${REMOTE_REPO}:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

  model_contract "${model}"
  mode_contract "${model}" "${mode}"
  method_contract "${method}"
  if [[ -n "${source_site}" ]]; then
    pythonpath="${source_site}:${pythonpath}"
  fi

  cat <<EOF
set -euo pipefail
cd '${REMOTE_REPO}'
export HF_HOME='${HF_HOME}'
export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
export HOME='${WANDB_HOME}'
export NRL_IGNORE_VERSION_MISMATCH=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export NEMO_RL_VENV_DIR='${REMOTE_REPO}/venvs'
export NRL_MEGATRON_CHECKPOINT_DIR='${checkpoint_root}'
export NRL_MEGATRON_TOKENIZER_MODEL='${target_model}'
export NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800
export RAY_CGRAPH_GET_TIMEOUT=7200
export RAY_CGRAPH_get_timeout=7200
export NODE_LOCAL_CACHE_ROOT='${node_cache}'
export PIP_CACHE_DIR='${RUN_ROOT}/cache/pip/${model}_${mode}_${method}'
export XDG_CACHE_HOME='${node_cache}/xdg'
export VLLM_CACHE_ROOT='${node_cache}/vllm'
export FLASHINFER_WORKSPACE_BASE='${node_cache}/flashinfer_workspace'
export FLASHINFER_CACHE_DIR='${node_cache}/flashinfer_workspace/.cache/flashinfer'
export TORCHINDUCTOR_CACHE_DIR='${node_cache}/torchinductor'
export TRITON_CACHE_DIR='${node_cache}/triton'
export CUDA_CACHE_PATH='${node_cache}/cuda'
export TORCH_EXTENSIONS_DIR='${node_cache}/torch_extensions'
export PYTHONPYCACHEPREFIX='${node_cache}/pycache'
export PYTHONDONTWRITEBYTECODE=1
export MEGATRON_DATASET_HELPERS_BUILD_DIR='${node_cache}/megatron_dataset_helpers'
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='HF_HOME,HF_DATASETS_CACHE,FLASHINFER_WORKSPACE_BASE,FLASHINFER_CACHE_DIR,TORCHINDUCTOR_CACHE_DIR,TRITON_CACHE_DIR,CUDA_CACHE_PATH,XDG_CACHE_HOME,TORCH_EXTENSIONS_DIR,PYTHONPYCACHEPREFIX'
EOF
  if [[ -n "${source_site}" ]]; then
    printf "export SOURCE_VLLM_SITE='%s'\n" "${source_site}"
  fi
  cat <<EOF
export PYTHONPATH='${pythonpath}'
python '${REMOTE_REPO}/examples/run_grpo.py' \\
  --config '${REMOTE_REPO}/${config}' \\
  policy.model_name='${target_model}' \\
  policy.tokenizer.name='${target_model}' \\
  policy.generation.vllm_cfg.enforce_eager=false \\
  ++policy.generation.vllm_kwargs.attention_backend=TRITON_ATTN \\
  grpo.max_num_steps=${MAX_STEPS} \\
  checkpointing.checkpoint_dir='${training_checkpoint_root}' \\
EOF
  if [[ "${model}" == "qwen235b" ]]; then
    printf '%s\n' "  ++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false \\"
  fi
  case "${method}" in
    suffix)
      cat <<EOF
  policy.draft.enabled=false \\
  ++policy.generation.vllm_kwargs.speculative_config.method=suffix \\
  ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=32 \\
  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_tree_depth=24 \\
  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_cached_requests=10000 \\
  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_spec_factor=1.0 \\
  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_min_token_prob=0.1 \\
EOF
      ;;
    eagle3)
      cat <<EOF
  policy.draft.enabled=false \\
  ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 \\
  ++policy.generation.vllm_kwargs.speculative_config.model='${draft_model}' \\
  ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 \\
  ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 \\
EOF
      ;;
  esac
  cat <<EOF
  logger.log_dir='${log_root}/nemo_logs' \\
  logger.wandb_enabled=true \\
  logger.wandb.project='${WANDB_PROJECT}' \\
  logger.wandb.name='${model}_math_${mode}_${method}_${RUN_ID}'
EOF
}

render_preflight() {
  cat <<EOF
set -euo pipefail
require_file() { [[ -s "\$1" ]] || { echo "ERROR: missing required file: \$1" >&2; exit 2; }; }
require_dir() { [[ -d "\$1" ]] || { echo "ERROR: missing required directory: \$1" >&2; exit 2; }; }
require_dir '${REMOTE_REPO}'
require_file '${REMOTE_REPO}/examples/run_grpo.py'
require_file '${REMOTE_REPO}/ray.sub'
require_file '${CONTAINER}'
repo_head="\$(git -C '${REMOTE_REPO}' rev-parse HEAD)"
if [[ "\${repo_head}" != '${NEMO_RL_SHA}' ]]; then
  echo "ERROR: NeMo-RL HEAD \${repo_head} does not match pinned SHA ${NEMO_RL_SHA}" >&2
  exit 2
fi
container_sha256="\$(sha256sum '${CONTAINER}' | awk '{print \$1}')"
printf 'container_sha256=%s\\n' "\${container_sha256}" >&2
EOF

  local record model mode method
  for record in "${records[@]}"; do
    IFS='|' read -r model mode method <<< "${record}"
    model_contract "${model}"
    mode_contract "${model}" "${mode}"
    method_contract "${method}"
    printf "require_file '%s'\n" "${REMOTE_REPO}/${config}"
    printf "require_file '%s/config.json'\n" "${target_model}"
    if [[ "${method}" == "eagle3" ]]; then
      printf "require_file '%s/config.json'\n" "${draft_model}"
    fi
    if [[ "${method}" == "suffix" ]]; then
      printf "require_dir '%s/arctic_inference/suffix_decoding'\n" "${source_site}"
    fi
  done
}

render_job() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local command network log_root job_name
  model_contract "${model}"
  mode_contract "${model}" "${mode}"
  method_contract "${method}"
  command="$(render_command "${model}" "${mode}" "${method}")"
  network="$(network_contract "${model}")"
  log_root="${RUN_ROOT}/logs/${model}_${mode}_${method}"
  job_name="${ACCOUNT}-specdec.${model}-${mode}-${method}"

  printf '[DRY-RUN] model=%s mode=%s method=%s k=%s\n' \
    "${model}" "${mode}" "${method}" "${spec_k}"
  cat <<EOF
COMMAND=\$(cat <<'NEMO_RL_COMMAND'
${command}
NEMO_RL_COMMAND
)
CONTAINER='${CONTAINER}' \\
MOUNTS='/lustre:/lustre,/project:/project' \\
BASE_LOG_DIR='${log_root}' \\
GPUS_PER_NODE=4 \\
HF_HOME='${HF_HOME}' \\
HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \\
EOF
  if [[ -n "${source_site}" ]]; then
    printf "SOURCE_VLLM_SITE='%s' \\\\\n" "${source_site}"
  fi
  printf '%s\n' "COMMAND=\"\${COMMAND}\" \\"
  printf 'sbatch --nodes=%s --account=%s --job-name=%s --partition=%s --time=%s --segment=%s' \
    "${nodes}" "${ACCOUNT}" "${job_name}" "${PARTITION}" "${WALLTIME}" "${segment}"
  if [[ -n "${network}" ]]; then
    printf ' --network=%s' "${network}"
  fi
  printf ' --output=%s/slurm-%%j.out ray.sub\n' "${log_root}"
}

read -r -a selected_models <<< "$(normalize_list "${MODELS}")"
read -r -a raw_modes <<< "$(normalize_list "${MODES}")"
read -r -a selected_methods <<< "$(normalize_list "${METHODS}")"

selected_modes=()
for model in "${selected_models[@]}"; do
  validate_model "${model}"
done
for raw_mode in "${raw_modes[@]}"; do
  selected_modes+=("$(canonical_mode "${raw_mode}")")
done
for method in "${selected_methods[@]}"; do
  validate_method "${method}"
done

records=()
for model in "${selected_models[@]}"; do
  for mode in "${selected_modes[@]}"; do
    for method in "${selected_methods[@]}"; do
      if supports_method "${model}" "${method}"; then
        records+=("${model}|${mode}|${method}")
      fi
    done
  done
done

if [[ "${#records[@]}" -eq 0 ]]; then
  echo "ERROR: selection contains no supported model/mode/method combinations" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo '[DRY-RUN] preflight'
  render_preflight
  for record in "${records[@]}"; do
    IFS='|' read -r model mode method <<< "${record}"
    render_job "${model}" "${mode}" "${method}"
  done
  exit 0
fi

remote_payload="$(render_preflight)"
remote_payload+=$'\n'
remote_payload+=$(cat <<'REMOTE'
submit_job() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local k="$4"
  local nodes="$5"
  local segment="$6"
  local config="$7"
  local log_root="$8"
  local wandb_name="$9"
  local network="${10}"
  local source_site="${11}"
  local command="${12}"
  local output job_id
  local sbatch_args=(
    --nodes="${nodes}"
    --account="${ACCOUNT}"
    --job-name="${ACCOUNT}-specdec.${model}-${mode}-${method}"
    --partition="${PARTITION}"
    --time="${WALLTIME}"
    --segment="${segment}"
    --output="${log_root}/slurm-%j.out"
  )
  if [[ -n "${network}" ]]; then
    sbatch_args+=(--network="${network}")
  fi

  mkdir -p "${log_root}" "${RUN_ROOT}/cache"
  output=$(
    CONTAINER="${CONTAINER}" \
    MOUNTS="/lustre:/lustre,/project:/project" \
    BASE_LOG_DIR="${log_root}" \
    GPUS_PER_NODE=4 \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    SOURCE_VLLM_SITE="${source_site}" \
    COMMAND="${command}" \
      sbatch "${sbatch_args[@]}" ray.sub
  )
  job_id="$(printf '%s\n' "${output}" | sed -n 's/^Submitted batch job //p' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    printf '%s\n' "${output}" >&2
    exit 1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${job_id}" "${model}" "${mode}" "${method}" "${k}" "${repo_head}" \
    "${config}" "${nodes}" "${segment}" "${CONTAINER}" "${container_sha256}" \
    "${HF_HOME}" "${HF_DATASETS_CACHE}" "${wandb_name}"
}

printf 'job_id,model,mode,method,k,repo_sha,config,nodes,segment,container,container_sha256,hf_home,hf_datasets_cache,wandb_name\n'
REMOTE
)

for record in "${records[@]}"; do
  IFS='|' read -r model mode method <<< "${record}"
  model_contract "${model}"
  mode_contract "${model}" "${mode}"
  method_contract "${method}"
  command="$(render_command "${model}" "${mode}" "${method}")"
  network="$(network_contract "${model}")"
  log_root="${RUN_ROOT}/logs/${model}_${mode}_${method}"
  wandb_name="${model}_math_${mode}_${method}_${RUN_ID}"
  printf -v submit_call 'submit_job %q %q %q %q %q %q %q %q %q %q %q %q' \
    "${model}" "${mode}" "${method}" "${spec_k}" "${nodes}" "${segment}" \
    "${config}" "${log_root}" "${wandb_name}" "${network}" "${source_site}" "${command}"
  remote_payload+=$'\n'
  remote_payload+="${submit_call}"
done

mkdir -p "$(dirname "${OUT}")"
printf '%s\n' "${remote_payload}" | \
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE_HOST}" \
    env \
      ACCOUNT="${ACCOUNT}" \
      PARTITION="${PARTITION}" \
      WALLTIME="${WALLTIME}" \
      RUN_ROOT="${RUN_ROOT}" \
      CONTAINER="${CONTAINER}" \
      HF_HOME="${HF_HOME}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      bash -s | tee "${OUT}"
