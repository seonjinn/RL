#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-login-lyris}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-specdec-cudagraph-780f483a-20260701}"
EXPECTED_REPO_HEAD="1271b1530181a7378e40de40b4b46ad223e6596c"
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
RUN_KIND="${RUN_KIND:-final}"
MAX_STEPS="${MAX_STEPS:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"

RUN_ID="${RUN_ID:-20260704_lyris_nemorl_v020_best_math_triton}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/${RUN_ID}}"
WANDB_HOME="${WANDB_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/wandb_netrc_home}"
WANDB_PROJECT="${WANDB_PROJECT:-sna-nemorl-specdec-lyris}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
WALLTIME="${WALLTIME:-05:00:00}"
OUT="${OUT:-${ROOT_DIR}/docs/latest_lyris_nemorl_v020_best_math_20260704_jobs.csv}"

normalize_list() {
  local raw="$1"
  printf '%s\n' "${raw//,/ }"
}

validate_bool() {
  local name="$1"
  local value="$2"
  case "${value}" in
    true|false) ;;
    *)
      printf 'ERROR: %s must be true or false, got %s\n' "${name}" "${value}" >&2
      return 2
      ;;
  esac
}

validate_fixed_contracts() {
  local required_steps

  if [[ "${REMOTE_HOST}" != "login-lyris" ]]; then
    echo "ERROR: REMOTE_HOST must be login-lyris" >&2
    return 2
  fi
  if [[ "${PARTITION}" != "gb200" ]]; then
    echo "ERROR: PARTITION must be gb200" >&2
    return 2
  fi
  case "${RUN_KIND}" in
    final) required_steps=20 ;;
    smoke) required_steps=2 ;;
    *)
      printf 'ERROR: unsupported RUN_KIND: %s\n' "${RUN_KIND}" >&2
      return 2
      ;;
  esac
  if [[ -z "${MAX_STEPS}" ]]; then
    MAX_STEPS="${required_steps}"
  fi
  if [[ "${MAX_STEPS}" != "${required_steps}" ]]; then
    printf 'ERROR: RUN_KIND=%s requires MAX_STEPS=%s\n' \
      "${RUN_KIND}" "${required_steps}" >&2
    return 2
  fi
  validate_bool DRY_RUN "${DRY_RUN}"
  validate_bool TEST_ONLY "${TEST_ONLY}"
}

canonical_mode() {
  case "$1" in
    sync) printf 'sync' ;;
    async1off|async-1off) printf 'async1off' ;;
    *)
      printf 'ERROR: unsupported mode: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

validate_model() {
  case "$1" in
    qwen30ba3b|qwen32|qwen235b) ;;
    *)
      printf 'ERROR: unsupported model: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

validate_method() {
  case "$1" in
    baseline|suffix|eagle3) ;;
    *)
      printf 'ERROR: unsupported method: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

supports_method() {
  case "$1:$2" in
    qwen30ba3b:baseline|qwen30ba3b:suffix) return 0 ;;
    qwen32:baseline|qwen32:suffix|qwen32:eagle3) return 0 ;;
    qwen235b:baseline|qwen235b:suffix|qwen235b:eagle3) return 0 ;;
    *) return 1 ;;
  esac
}

model_contract() {
  local model="$1"
  eagle3_model=""
  fuse_allreduce_rms=""
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
      fuse_allreduce_rms=false
      ;;
  esac
}

mode_contract() {
  local model="$1"
  local mode="$2"
  cluster_segment_size=""
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
      cluster_segment_size=4
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
  local model="$1"
  network=""
  case "${model}" in
    qwen30ba3b|qwen32) network=sharp ;;
    qwen235b) ;;
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
  local pythonpath

  model_contract "${model}"
  mode_contract "${model}" "${mode}"
  method_contract "${method}"
  pythonpath="${REMOTE_REPO}:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REMOTE_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
  if [[ -n "${source_site}" ]]; then
    pythonpath="${source_site}:${pythonpath}"
  fi

  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "${REMOTE_REPO}"
  printf 'export HF_HOME=%q\n' "${HF_HOME}"
  printf 'export HF_DATASETS_CACHE=%q\n' "${HF_DATASETS_CACHE}"
  printf 'export HOME=%q\n' "${WANDB_HOME}"
  printf 'export NRL_IGNORE_VERSION_MISMATCH=1\n'
  printf 'export NEMO_RL_PY_EXECUTABLES_SYSTEM=0\n'
  printf 'export NEMO_RL_VENV_DIR=%q\n' '/opt/ray_venvs'
  printf 'export NRL_MEGATRON_CHECKPOINT_DIR=%q\n' "${checkpoint_root}"
  printf 'export NRL_MEGATRON_TOKENIZER_MODEL=%q\n' "${target_model}"
  printf 'export NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800\n'
  printf 'export RAY_CGRAPH_GET_TIMEOUT=7200\n'
  printf 'export RAY_CGRAPH_get_timeout=7200\n'
  printf 'export NODE_LOCAL_CACHE_ROOT=%q\n' "${node_cache}"
  printf 'export PIP_CACHE_DIR=%q\n' "${RUN_ROOT}/cache/pip/${model}_${mode}_${method}"
  printf 'export XDG_CACHE_HOME=%q\n' "${node_cache}/xdg"
  printf 'export VLLM_CACHE_ROOT=%q\n' "${node_cache}/vllm"
  printf 'export FLASHINFER_WORKSPACE_BASE=%q\n' "${node_cache}/flashinfer_workspace"
  printf 'export FLASHINFER_CACHE_DIR=%q\n' "${node_cache}/flashinfer_workspace/.cache/flashinfer"
  printf 'export TORCHINDUCTOR_CACHE_DIR=%q\n' "${node_cache}/torchinductor"
  printf 'export TRITON_CACHE_DIR=%q\n' "${node_cache}/triton"
  printf 'export CUDA_CACHE_PATH=%q\n' "${node_cache}/cuda"
  printf 'export TORCH_EXTENSIONS_DIR=%q\n' "${node_cache}/torch_extensions"
  printf 'export PYTHONPYCACHEPREFIX=%q\n' "${node_cache}/pycache"
  printf 'export PYTHONDONTWRITEBYTECODE=1\n'
  printf 'export MEGATRON_DATASET_HELPERS_BUILD_DIR=%q\n' "${node_cache}/megatron_dataset_helpers"
  printf 'export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=%q\n' 'HF_HOME,HF_DATASETS_CACHE,FLASHINFER_WORKSPACE_BASE,FLASHINFER_CACHE_DIR,TORCHINDUCTOR_CACHE_DIR,TRITON_CACHE_DIR,CUDA_CACHE_PATH,XDG_CACHE_HOME,TORCH_EXTENSIONS_DIR,PYTHONPYCACHEPREFIX'
  if [[ -n "${source_site}" ]]; then
    printf 'export SOURCE_VLLM_SITE=%q\n' "${source_site}"
  fi
  printf 'export PYTHONPATH=%q\n' "${pythonpath}"
  printf 'python %q \\\n' "${REMOTE_REPO}/examples/run_grpo.py"
  printf '  --config %q \\\n' "${REMOTE_REPO}/${config}"
  printf '  %q \\\n' "policy.model_name=${target_model}"
  printf '  %q \\\n' "policy.tokenizer.name=${target_model}"
  printf '  policy.generation.temperature=1.0 \\\n'
  printf '  policy.generation.top_p=1.0 \\\n'
  printf '  policy.generation.vllm_cfg.enforce_eager=false \\\n'
  printf '  ++policy.generation.vllm_kwargs.attention_backend=TRITON_ATTN \\\n'
  printf '  ++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton \\\n'
  printf '  grpo.max_num_steps=%s \\\n' "${MAX_STEPS}"
  printf '  %q \\\n' "checkpointing.checkpoint_dir=${training_checkpoint_root}"
  if [[ -n "${cluster_segment_size}" ]]; then
    printf '  cluster.segment_size=%s \\\n' "${cluster_segment_size}"
  fi
  if [[ -n "${fuse_allreduce_rms}" ]]; then
    printf '  ++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=%s \\\n' "${fuse_allreduce_rms}"
  fi
  case "${method}" in
    baseline) ;;
    suffix)
      printf '  policy.draft.enabled=false \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.method=suffix \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=32 \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_tree_depth=24 \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_cached_requests=10000 \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_max_spec_factor=1.0 \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.suffix_decoding_min_token_prob=0.1 \\\n'
      ;;
    eagle3)
      printf '  policy.draft.enabled=false \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 \\\n'
      printf '  %q \\\n' "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
      printf '  ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 \\\n'
      printf '  ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 \\\n'
      ;;
  esac
  printf '  %q \\\n' "logger.log_dir=${log_root}/nemo_logs"
  printf '  logger.wandb_enabled=true \\\n'
  printf '  %q \\\n' "logger.wandb.project=${WANDB_PROJECT}"
  printf '  %q\n' "logger.wandb.name=${model}_math_${mode}_${method}_${RUN_ID}"
}

render_preflight() {
  printf 'set -euo pipefail\n'
  cat <<'REMOTE_PREFLIGHT'
require_file() { [[ -s "$1" ]] || { printf 'ERROR: missing required file: %s\n' "$1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { printf 'ERROR: missing required directory: %s\n' "$1" >&2; exit 2; }; }
REMOTE_PREFLIGHT
  printf 'require_dir %q\n' "${REMOTE_REPO}"
  printf 'require_file %q\n' "${REMOTE_REPO}/examples/run_grpo.py"
  printf 'require_file %q\n' "${REMOTE_REPO}/ray.sub"
  printf 'require_file %q\n' "${CONTAINER}"
  printf 'repo_head="$(git -C %q rev-parse HEAD)"\n' "${REMOTE_REPO}"
  printf 'if [[ "${repo_head}" != %q ]]; then\n' "${EXPECTED_REPO_HEAD}"
  printf "  printf 'ERROR: NeMo-RL HEAD %%s does not match pinned SHA %%s\\n' \"\${repo_head}\" %q >&2\n" "${EXPECTED_REPO_HEAD}"
  printf '  exit 2\nfi\n'
  printf 'repo_status="$(git -C %q status --porcelain --untracked-files=normal)"\n' "${REMOTE_REPO}"
  cat <<'REMOTE_PREFLIGHT'
if [[ -n "${repo_status}" ]]; then
  echo 'ERROR: remote NeMo-RL worktree is not clean' >&2
  printf '%s\n' "${repo_status}" >&2
  exit 2
fi
REMOTE_PREFLIGHT
  printf 'container_sha256="$(sha256sum %q | awk '\''{print $1}'\'')"\n' "${CONTAINER}"
  printf "printf 'container_sha256=%%s\\n' \"\${container_sha256}\" >&2\n"

  local record model mode method
  for record in "${records[@]}"; do
    IFS='|' read -r model mode method <<< "${record}"
    model_contract "${model}"
    mode_contract "${model}" "${mode}"
    method_contract "${method}"
    printf 'require_file %q\n' "${REMOTE_REPO}/${config}"
    printf 'require_file %q\n' "${target_model}/config.json"
    if [[ "${method}" == "eagle3" ]]; then
      printf 'require_file %q\n' "${draft_model}/config.json"
    fi
    if [[ "${method}" == "suffix" ]]; then
      printf 'require_dir %q\n' "${source_site}/arctic_inference/suffix_decoding"
    fi
  done
}

build_sbatch_args() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local log_root="${RUN_ROOT}/logs/${model}_${mode}_${method}"

  mode_contract "${model}" "${mode}"
  network_contract "${model}"
  sbatch_args=(
    "--nodes=${nodes}"
    "--account=${ACCOUNT}"
    "--job-name=${ACCOUNT}-specdec.${model}-${mode}-${method}"
    "--partition=${PARTITION}"
    "--time=${WALLTIME}"
    "--segment=${segment}"
    "--output=${log_root}/slurm-%j.out"
  )
  if [[ -n "${network}" ]]; then
    sbatch_args+=("--network=${network}")
  fi
}

render_sbatch() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local test_only="$4"
  local arg

  build_sbatch_args "${model}" "${mode}" "${method}"
  printf 'sbatch'
  if [[ "${test_only}" == "true" ]]; then
    printf ' --test-only'
  fi
  for arg in "${sbatch_args[@]}"; do
    printf ' %q' "${arg}"
  done
  printf ' %q\n' "${REMOTE_REPO}/ray.sub"
}

render_job() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local command log_root

  model_contract "${model}"
  mode_contract "${model}" "${mode}"
  method_contract "${method}"
  command="$(render_command "${model}" "${mode}" "${method}")"
  log_root="${RUN_ROOT}/logs/${model}_${mode}_${method}"

  printf '[DRY-RUN] model=%s mode=%s method=%s k=%s\n' \
    "${model}" "${mode}" "${method}" "${spec_k}"
  printf 'COMMAND=$(cat <<'\''NEMO_RL_COMMAND'\''\n%s\nNEMO_RL_COMMAND\n)\n' "${command}"
  printf 'CONTAINER=%q \\\n' "${CONTAINER}"
  printf 'MOUNTS=%q \\\n' '/lustre:/lustre,/project:/project'
  printf 'BASE_LOG_DIR=%q \\\n' "${log_root}"
  printf 'GPUS_PER_NODE=4 \\\n'
  printf 'HF_HOME=%q \\\n' "${HF_HOME}"
  printf 'HF_DATASETS_CACHE=%q \\\n' "${HF_DATASETS_CACHE}"
  if [[ -n "${source_site}" ]]; then
    printf 'SOURCE_VLLM_SITE=%q \\\n' "${source_site}"
  fi
  printf 'COMMAND="${COMMAND}" \\\n'
  printf '[DRY-RUN] '
  render_sbatch "${model}" "${mode}" "${method}" true
  printf '[DRY-RUN] '
  render_sbatch "${model}" "${mode}" "${method}" false
}

local_preflight() {
  local status upstream ahead

  status="$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=normal)"
  if [[ -n "${status}" ]]; then
    echo "ERROR: local worktree is not clean" >&2
    printf '%s\n' "${status}" >&2
    return 2
  fi
  if ! upstream="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"; then
    echo "ERROR: local branch has no upstream" >&2
    return 2
  fi
  ahead="$(git -C "${ROOT_DIR}" rev-list --count "${upstream}..HEAD")"
  if [[ "${ahead}" != "0" ]]; then
    printf 'ERROR: local HEAD is ahead of upstream by %s commit(s)\n' "${ahead}" >&2
    return 2
  fi
}

validate_fixed_contracts

read -r -a selected_models <<< "$(normalize_list "${MODELS}")"
read -r -a raw_modes <<< "$(normalize_list "${MODES}")"
read -r -a selected_methods <<< "$(normalize_list "${METHODS}")"

selected_modes=()
for model in "${selected_models[@]}"; do
  validate_model "${model}"
  if [[ "${RUN_KIND}" == "smoke" && "${model}" != "qwen235b" ]]; then
    echo "ERROR: RUN_KIND=smoke permits only qwen235b" >&2
    exit 2
  fi
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
  echo '[DRY-RUN] remote preflight'
  render_preflight
  for record in "${records[@]}"; do
    IFS='|' read -r model mode method <<< "${record}"
    render_job "${model}" "${mode}" "${method}"
  done
  exit 0
fi

local_preflight

remote_payload="$(render_preflight)"
remote_payload+=$'\n'
remote_payload+=$(cat <<'REMOTE'
csv_field() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "${value}"
}

csv_row() {
  local separator=""
  local value
  for value in "$@"; do
    printf '%s' "${separator}"
    csv_field "${value}"
    separator=,
  done
  printf '\n'
}

submit_job() {
  local model="$1"
  local mode="$2"
  local method="$3"
  local k="$4"
  local nodes="$5"
  local gpus_per_node="$6"
  local segment="$7"
  local cluster_segment_size="$8"
  local network="$9"
  local config="${10}"
  local log_root="${11}"
  local wandb_name="${12}"
  local source_site="${13}"
  local target_model="${14}"
  local draft_model="${15}"
  local fuse_allreduce_rms="${16}"
  local command="${17}"
  local test_only_output test_only_job_id actual_output actual_job_id
  local sbatch_args=(
    "--nodes=${nodes}"
    "--account=${ACCOUNT}"
    "--job-name=${ACCOUNT}-specdec.${model}-${mode}-${method}"
    "--partition=${PARTITION}"
    "--time=${WALLTIME}"
    "--segment=${segment}"
    "--output=${log_root}/slurm-%j.out"
  )
  if [[ -n "${network}" ]]; then
    sbatch_args+=("--network=${network}")
  fi

  mkdir -p "${log_root}" "${RUN_ROOT}/cache"
  if ! test_only_output=$(
    CONTAINER="${CONTAINER}" \
    MOUNTS="/lustre:/lustre,/project:/project" \
    BASE_LOG_DIR="${log_root}" \
    GPUS_PER_NODE="${gpus_per_node}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    SOURCE_VLLM_SITE="${source_site}" \
    COMMAND="${command}" \
      sbatch --test-only "${sbatch_args[@]}" "${REMOTE_REPO}/ray.sub" 2>&1
  ); then
    printf '%s\n' "${test_only_output}" >&2
    return 1
  fi
  printf '%s\n' "${test_only_output}" >&2
  test_only_job_id="$(sed -nE 's/.*Job ([0-9]+).*/\1/p' <<< "${test_only_output}" | head -1)"
  if [[ -z "${test_only_job_id}" ]]; then
    echo "ERROR: sbatch --test-only succeeded without a parseable job ID" >&2
    return 1
  fi

  actual_job_id=""
  if [[ "${TEST_ONLY}" != "true" ]]; then
    actual_output=$(
      CONTAINER="${CONTAINER}" \
      MOUNTS="/lustre:/lustre,/project:/project" \
      BASE_LOG_DIR="${log_root}" \
      GPUS_PER_NODE="${gpus_per_node}" \
      HF_HOME="${HF_HOME}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      SOURCE_VLLM_SITE="${source_site}" \
      COMMAND="${command}" \
        sbatch "${sbatch_args[@]}" "${REMOTE_REPO}/ray.sub"
    )
    actual_job_id="$(sed -n 's/^Submitted batch job //p' <<< "${actual_output}" | tail -1)"
    if [[ -z "${actual_job_id}" ]]; then
      printf '%s\n' "${actual_output}" >&2
      return 1
    fi
  fi

  csv_row \
    "${actual_job_id}" "${test_only_job_id}" "${model}" "${mode}" "${method}" \
    "${k}" "${repo_head}" "${config}" "${RUN_KIND}" "${MAX_STEPS}" 1.0 1.0 \
    false TRITON_ATTN triton "${fuse_allreduce_rms}" "${nodes}" \
    "${gpus_per_node}" "${segment}" "${cluster_segment_size}" "${network}" \
    "${CONTAINER}" "${container_sha256}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
    "${source_site}" "${target_model}" "${draft_model}" "${RUN_ID}" \
    "${log_root}" true "${WANDB_PROJECT}" "${wandb_name}" "${command}"
}

printf '%s\n' 'actual_job_id,test_only_job_id,model,mode,method,k,repo_sha,config,run_kind,max_steps,temperature,top_p,enforce_eager,attention_backend,moe_backend,fuse_allreduce_rms,nodes,gpus_per_node,segment,cluster_segment_size,network,container,container_sha256,hf_home,hf_datasets_cache,source_vllm_site,target_model,draft_model,run_id,log_dir,wandb_enabled,wandb_project,wandb_name,rendered_command'
REMOTE
)

for record in "${records[@]}"; do
  IFS='|' read -r model mode method <<< "${record}"
  model_contract "${model}"
  mode_contract "${model}" "${mode}"
  method_contract "${method}"
  network_contract "${model}"
  command="$(render_command "${model}" "${mode}" "${method}")"
  log_root="${RUN_ROOT}/logs/${model}_${mode}_${method}"
  wandb_name="${model}_math_${mode}_${method}_${RUN_ID}"
  printf -v submit_call 'submit_job %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q' \
    "${model}" "${mode}" "${method}" "${spec_k}" "${nodes}" 4 "${segment}" \
    "${cluster_segment_size}" "${network}" "${config}" "${log_root}" \
    "${wandb_name}" "${source_site}" "${target_model}" "${draft_model}" \
    "${fuse_allreduce_rms}" "${command}"
  remote_payload+=$'\n'
  remote_payload+="${submit_call}"
done

remote_assignments=(
  "REMOTE_REPO=${REMOTE_REPO}"
  "CONTAINER=${CONTAINER}"
  "HF_HOME=${HF_HOME}"
  "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
  "RUN_ROOT=${RUN_ROOT}"
  "RUN_ID=${RUN_ID}"
  "RUN_KIND=${RUN_KIND}"
  "WANDB_PROJECT=${WANDB_PROJECT}"
  "ACCOUNT=${ACCOUNT}"
  "PARTITION=${PARTITION}"
  "WALLTIME=${WALLTIME}"
  "MAX_STEPS=${MAX_STEPS}"
  "TEST_ONLY=${TEST_ONLY}"
)
printf -v remote_command 'env'
for remote_assignment in "${remote_assignments[@]}"; do
  printf -v quoted_assignment '%q' "${remote_assignment}"
  remote_command+=" ${quoted_assignment}"
done
remote_command+=' bash -s'

mkdir -p "$(dirname "${OUT}")"
printf '%s\n' "${remote_payload}" | \
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE_HOST}" "${remote_command}" | \
  tee "${OUT}"
