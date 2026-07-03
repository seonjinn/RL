#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-auto}"
if [[ "${CLUSTER}" == "auto" ]]; then
  case "$(hostname)" in
    *lyris*) CLUSTER="lyris" ;;
    *ptyche*) CLUSTER="ptyche" ;;
    *)
      echo "Set CLUSTER=lyris or CLUSTER=ptyche" >&2
      exit 2
      ;;
  esac
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
case "${CLUSTER}" in
  lyris) PARTITION="${PARTITION:-gb200}" ;;
  ptyche) PARTITION="${PARTITION:-batch}" ;;
  *)
    echo "Unsupported cluster: ${CLUSTER}" >&2
    exit 2
    ;;
esac

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
ASSET_ROOT="${ASSET_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/assets}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${LUSTRE_ROOT}/containers/vllm-openai-v0.24.0-aarch64-ubuntu2404.sqsh}"
ANGELSLIM_COMMIT="6a97dab2f17c0a3c031065329f092c4f61108a6f"
ANGELSLIM_SOURCE="${ASSET_ROOT}/src/angelslim-${ANGELSLIM_COMMIT}"
ANGELSLIM_RUNTIME_SITE="${ANGELSLIM_RUNTIME_SITE:-${ASSET_ROOT}/python/angelslim_runtime}"
MODEL="${MODEL:-${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
DFLASH_MODEL="${DFLASH_MODEL:-${HF_HOME}/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}"
DFLARE_MODEL="${DFLARE_MODEL:-${HF_HOME}/hub/models--AngelSlim--Qwen3-8b-dflare/snapshots/55e2c8d86d76ce1e79fa3b8642c7f80091285a82}"
METHODS="${METHODS:-dflash dflare}"
DOMAINS="${DOMAINS:-Math SWE}"
TEMPERATURES="${TEMPERATURES:-0.0 1.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_angelslim_native}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/angelslim-native}"
SMOKE="${SMOKE:-true}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

if [[ "${SMOKE}" == "true" ]]; then
  MAX_SAMPLES="${MAX_SAMPLES:-4}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
else
  MAX_SAMPLES="${MAX_SAMPLES:-128}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
  TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
fi

normalize_temperature() {
  printf '%s' "$1" | tr '.' 'p'
}

render_sbatch() {
  local method="$1"
  local draft_model="$2"
  local dataset="$3"
  local domain_label="$4"
  local temperature="$5"
  local run_dir="$6"
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --job-name=coreai_dlalgo_llm-angelslim.q8-${domain_label}-${method}-t$(normalize_temperature "${temperature}")
#SBATCH --output=${run_dir}/slurm-%j.out

set -euo pipefail

test -s '${CONTAINER_IMAGE}'
test -d '${ANGELSLIM_SOURCE}'
test -d '${ANGELSLIM_RUNTIME_SITE}'
test -d '${MODEL}'
test -d '${draft_model}'

export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
export NODE_LOCAL_CACHE_ROOT="/tmp/sna/angelslim_\${SLURM_JOB_ID}_${method}_${domain_label}"
export XDG_CACHE_HOME="\${NODE_LOCAL_CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="\${NODE_LOCAL_CACHE_ROOT}/triton"
mkdir -p "\${XDG_CACHE_HOME}" "\${TORCHINDUCTOR_CACHE_DIR}" "\${TRITON_CACHE_DIR}"

echo 'backend=angelslim_transformers_native'
echo 'draft_arch=${method}'
echo 'dataset=${dataset}'
echo 'temperature=${temperature}'
echo 'top_p=1.0'
echo 'block_size=16'

srun --ntasks=1 \\
  --container-image='${CONTAINER_IMAGE}' \\
  --container-mounts='/lustre:/lustre' \\
  --no-container-mount-home \\
  --container-remap-root \\
  --mpi=pmix \\
  bash -lc "set -euo pipefail
export PYTHONPATH='${ANGELSLIM_RUNTIME_SITE}:${ANGELSLIM_SOURCE}'
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export HF_DATASETS_CACHE='${HF_HOME}/datasets'
torchrun --standalone --nproc-per-node=4 '${ANGELSLIM_SOURCE}/tools/dflash_benchmark.py' \\
  --model-name-or-path '${MODEL}' \\
  --draft-name-or-path '${draft_model}' \\
  --draft-arch '${method}' \\
  --block-size '16' \\
  --dataset '${dataset}' \\
  --max-samples '${MAX_SAMPLES}' \\
  --max-new-tokens '${MAX_NEW_TOKENS}' \\
  --temperature '${temperature}' \\
  --output-json '${run_dir}/result.json'" \\
  2>&1 | tee '${run_dir}/benchmark.log'
EOF
}

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

MANIFEST="${RESULT_ROOT}/${RUN_ID}/jobs.tsv"
if [[ "${DRY_RUN}" != "true" ]]; then
  mkdir -p "$(dirname "${MANIFEST}")"
  printf 'job_id\tmethod\tdomain\ttemperature\trun_dir\n' >"${MANIFEST}"
fi

for domain in ${DOMAINS}; do
  case "${domain}" in
    Math)
      dataset=math500
      domain_label=math
      ;;
    SWE)
      dataset=swe-bench
      domain_label=swe
      ;;
    *)
      echo "Unsupported domain: ${domain}" >&2
      exit 2
      ;;
  esac

  for method in ${METHODS}; do
    case "${method}" in
      dflash) draft_model="${DFLASH_MODEL}" ;;
      dflare) draft_model="${DFLARE_MODEL}" ;;
      *)
        echo "Unsupported method: ${method}" >&2
        exit 2
        ;;
    esac
    for temperature in ${TEMPERATURES}; do
      run_dir="${RESULT_ROOT}/${RUN_ID}/${domain_label}/${method}_t$(normalize_temperature "${temperature}")"
      if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY-RUN] native_method=${method} domain=${domain} temperature=${temperature}"
        render_sbatch "${method}" "${draft_model}" "${dataset}" "${domain_label}" "${temperature}" "${run_dir}"
        continue
      fi

      mkdir -p "${run_dir}"
      sbatch_file="${run_dir}/submit.sbatch"
      render_sbatch "${method}" "${draft_model}" "${dataset}" "${domain_label}" "${temperature}" "${run_dir}" >"${sbatch_file}"
      sbatch_args=()
      if [[ -n "${DEPENDENCY}" ]]; then
        sbatch_args+=("--dependency=${DEPENDENCY}")
      fi
      if [[ "${TEST_ONLY}" == "true" ]]; then
        sbatch --test-only "${sbatch_args[@]}" "${sbatch_file}"
        printf 'test-only\t%s\t%s\t%s\t%s\n' "${method}" "${domain}" "${temperature}" "${run_dir}" >>"${MANIFEST}"
        continue
      fi
      job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_file}")"
      printf '%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${method}" "${domain}" "${temperature}" "${run_dir}" | tee -a "${MANIFEST}"
    done
  done
done

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "manifest=${MANIFEST}"
fi
