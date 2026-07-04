#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}"
VIEW_ROOT="${VIEW_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/long-context-models/yarn4}"
RESULT_ROOT="${RESULT_ROOT:-${LUSTRE_ROOT}/vllm024-dynamicsd/angelslim-long-context}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen8_dflare_spec}"
PROFILES="${PROFILES:-64k 128k}"
DOMAINS="${DOMAINS:-Math SWE}"
TEMPERATURES="${TEMPERATURES:-0.0 1.0}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
DEPENDENCY="${DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
TEST_ONLY="${TEST_ONLY:-false}"
REQUIRE_GIT_PULL="${REQUIRE_GIT_PULL:-true}"

TARGET_VIEW="${VIEW_ROOT}/qwen3-8b"
DFLARE_VIEW="${VIEW_ROOT}/qwen3-8b-dflare"

if [[ "${DRY_RUN}" != "true" && "${REQUIRE_GIT_PULL}" == "true" ]]; then
  git -C "${SCRIPT_DIR}" pull --ff-only
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  test -d "${TARGET_VIEW}"
  test -d "${DFLARE_VIEW}"
fi

for profile in ${PROFILES}; do
  case "${profile}" in
    64k)
      isl=4096
      osl=65536
      ;;
    128k)
      isl=4096
      osl=126976
      ;;
    *)
      echo "Unsupported context profile: ${profile}" >&2
      exit 2
      ;;
  esac

  echo "context_profile=${profile} isl=${isl} osl=${osl} total=$((isl + osl))"
  env \
    CLUSTER="${CLUSTER:-auto}" \
    ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}" \
    PARTITION="${PARTITION:-gb200}" \
    LUSTRE_ROOT="${LUSTRE_ROOT}" \
    MODEL="${TARGET_VIEW}" \
    DFLARE_MODEL="${DFLARE_VIEW}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    RUN_ID="${RUN_ID}_${profile}" \
    METHODS=dflare \
    DOMAINS="${DOMAINS}" \
    TEMPERATURES="${TEMPERATURES}" \
    RUN_MODE=spec \
    ISL="${isl}" \
    MAX_NEW_TOKENS="${osl}" \
    MAX_SAMPLES=4 \
    IGNORE_EOS=true \
    TIME_LIMIT="${TIME_LIMIT}" \
    SMOKE=false \
    DEPENDENCY="${DEPENDENCY}" \
    DRY_RUN="${DRY_RUN}" \
    TEST_ONLY="${TEST_ONLY}" \
    REQUIRE_GIT_PULL=false \
    "${SCRIPT_DIR}/submit_angelslim_matrix.sh"
done
