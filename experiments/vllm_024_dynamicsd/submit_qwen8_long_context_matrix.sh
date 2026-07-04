#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
HF_HOME="${HF_HOME:-${LUSTRE_ROOT}/hf_home}"
VIEW_ROOT="${VIEW_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/long-context-models/yarn4}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/long-context}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen8_yarn4}"
PROFILES="${PROFILES:-64k 128k}"
BATCH_SIZES_64K="${BATCH_SIZES_64K:-1}"
BATCH_SIZES_128K="${BATCH_SIZES_128K:-1}"
DOMAINS="${DOMAINS:-Math SWE}"
TEMPERATURES="${TEMPERATURES:-0.0 1.0}"
METHODS="${METHODS:-baseline suffix pard pard2 dflash}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

TARGET_SOURCE="${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PARD_SOURCE="${HF_HOME}/hub/models--amd--PARD-Qwen3-0.6B/snapshots/f9f650fbab180c26498817718f0db5cae8f25136"
PARD2_SOURCE="${HF_HOME}/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135"
DFLASH_SOURCE="${HF_HOME}/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4"
DFLARE_SOURCE="${HF_HOME}/hub/models--AngelSlim--Qwen3-8b-dflare/snapshots/55e2c8d86d76ce1e79fa3b8642c7f80091285a82"

TARGET_VIEW="${VIEW_ROOT}/qwen3-8b"
PARD_VIEW="${VIEW_ROOT}/pard-qwen3-0.6b"
PARD2_VIEW="${VIEW_ROOT}/pard2-qwen3-8b"
DFLASH_VIEW="${VIEW_ROOT}/qwen3-8b-dflash-b16"

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  python3 "${SCRIPT_DIR}/materialize_long_context_model_views.py" \
    --view-root "${VIEW_ROOT}" \
    --max-position-embeddings 131072 \
    --rope-factor 4.0 \
    --model-view "qwen3-8b=${TARGET_SOURCE}" \
    --model-view "pard-qwen3-0.6b=${PARD_SOURCE}" \
    --model-view "pard2-qwen3-8b=${PARD2_SOURCE}" \
    --model-view "qwen3-8b-dflash-b16=${DFLASH_SOURCE}" \
    --model-view "qwen3-8b-dflare=${DFLARE_SOURCE}"
fi

for profile in ${PROFILES}; do
  case "${profile}" in
    64k)
      isl=4096
      osl=65536
      max_model_len=69632
      batch_sizes="${BATCH_SIZES_64K}"
      ;;
    128k)
      isl=4096
      osl=126976
      max_model_len=131072
      batch_sizes="${BATCH_SIZES_128K}"
      ;;
    *)
      echo "Unsupported context profile: ${profile}" >&2
      exit 2
      ;;
  esac

  echo "context_profile=${profile} isl=${isl} osl=${osl} total=$((isl + osl))"
  for batch_size in ${batch_sizes}; do
    env \
      CLUSTER="${CLUSTER:-auto}" \
      ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
      PARTITION="${PARTITION:-gb200}" \
      LUSTRE_ROOT="${LUSTRE_ROOT}" \
      HF_HOME="${HF_HOME}" \
      MODEL="${TARGET_VIEW}" \
      PARD_MODEL="${PARD_VIEW}" \
      PARD2_MODEL="${PARD2_VIEW}" \
      DFLASH_MODEL="${DFLASH_VIEW}" \
      RESULT_ROOT="${RESULT_ROOT}" \
      RUN_ID="${RUN_ID}_${profile}_bs${batch_size}" \
      JOB_LABEL_PREFIX="q8-lc${profile}" \
      METHODS="${METHODS}" \
      DOMAINS="${DOMAINS}" \
      TEMPERATURES="${TEMPERATURES}" \
      ISL="${isl}" \
      OSL="${osl}" \
      MAX_MODEL_LEN="${max_model_len}" \
      BATCH_SIZES="${batch_size}" \
      WARMUP_REPEATS=0 \
      MEASURE_REPEATS=1 \
      TIME_LIMIT="${TIME_LIMIT}" \
      SMOKE=false \
      DRY_RUN="${DRY_RUN}" \
      TEST_ONLY="${TEST_ONLY}" \
      REQUIRE_GIT_PULL=false \
      "${SCRIPT_DIR}/submit_qwen8_extended_methods_matrix.sh"
  done
done
