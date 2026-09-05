#!/bin/bash

set -euo pipefail

case $(basename "$0") in
enroot)
  : "${FAKE_ENROOT_LOG:?}"

  case "${1:-}" in
  version)
    printf '%s\n' "${FAKE_ENROOT_VERSION:-3.5.0}"
    ;;
  import)
    echo 'command=import' >>"${FAKE_ENROOT_LOG}"
    echo "enroot_cache=${ENROOT_CACHE_PATH:-missing}" >>"${FAKE_ENROOT_LOG}"
    echo "enroot_data=${ENROOT_DATA_PATH:-missing}" >>"${FAKE_ENROOT_LOG}"
    echo "enroot_runtime=${ENROOT_RUNTIME_PATH:-missing}" >>"${FAKE_ENROOT_LOG}"
    echo "tmpdir=${TMPDIR:-missing}" >>"${FAKE_ENROOT_LOG}"
    echo "tmp=${TMP:-missing}" >>"${FAKE_ENROOT_LOG}"
    echo "temp=${TEMP:-missing}" >>"${FAKE_ENROOT_LOG}"
    if [[ "${2:-}" != '-o' || -z "${3:-}" || -z "${4:-}" ]]; then
      echo 'unexpected fake enroot arguments' >&2
      exit 89
    fi
    if [[ "${4}" == *'@sha256:'* || "${4}" != docker://*#*:* ||
      "${4}" =~ [[:space:]] ]]; then
      echo 'fake Enroot 3.5 rejected unsupported Docker reference syntax' >&2
      exit 90
    fi
    echo "output=${3}" >>"${FAKE_ENROOT_LOG}"
    echo "uri=${4}" >>"${FAKE_ENROOT_LOG}"
    printf 'fake-sqsh\n' >"${3}"
    if [[ "${FAKE_ENROOT_SYMLINK_CANDIDATE:-0}" == 1 ]]; then
      /bin/mv "${3}" "${3}.target"
      /bin/ln -s "${3}.target" "${3}"
    fi
    if [[ "${FAKE_SCRATCH_REPLACE_AFTER_IMPORT:-0}" == 1 ]]; then
      scratch_directory=$(dirname "${3}")
      scratch_owner=
      if [[ "${FAKE_SCRATCH_REPLAY_OWNER:-0}" == 1 ]]; then
        scratch_owner=$(<"${scratch_directory}/.stage-owner")
      fi
      /bin/rm -rf "${scratch_directory}"
      /bin/mkdir -p "${scratch_directory}"
      if [[ -n "${scratch_owner}" ]]; then
        printf '%s\n' "${scratch_owner}" >"${scratch_directory}/.stage-owner"
      fi
      printf 'fake-sqsh\n' >"${3}"
      printf '%s\n' "${FAKE_SCRATCH_REPLACEMENT_MARKER:-unknown-replacement}" \
        >"${scratch_directory}/unknown-sentinel"
      if [[ "${FAKE_SCRATCH_REPLACE_SUCCESS:-0}" != 1 ]]; then
        exit 94
      fi
    fi
    if [[ "${FAKE_ENROOT_SWAP_CONTAINER:-0}" == 1 ]]; then
      : "${FAKE_CONTAINER_SWAP_TARGET:?}"
      /bin/rmdir "${FAKE_CONTAINER_DIR_TO_SWAP}"
      /bin/ln -s "${FAKE_CONTAINER_SWAP_TARGET}" \
        "${FAKE_CONTAINER_DIR_TO_SWAP}"
    fi
    if [[ "${FAKE_ENROOT_FAIL:-0}" == 1 ]]; then
      echo 'requested fake enroot import failure' >&2
      exit 91
    fi
    ;;
  *)
    echo "unexpected fake enroot command: ${1:-missing}" >&2
    exit 88
    ;;
  esac
  exit 0
  ;;
ln)
  if [[ "${2:-}" == *.metadata.txt &&
    "${FAKE_LN_SIGNAL_AFTER_RECEIPT_LINK:-0}" == 1 ]]; then
    /bin/ln "$@"
    /bin/kill -TERM "${PPID}"
    exit 0
  fi
  if [[ "${2:-}" == *.metadata.txt &&
    "${FAKE_LN_REPLACE_PARTIAL_AFTER_RECEIPT_LINK:-0}" == 1 ]]; then
    /bin/ln "$@"
    /bin/rm -f "${1}"
    printf 'unknown-partial-replacement\n' >"${1}"
    exit 0
  fi
  if [[ "${FAKE_LN_FAIL_RECEIPT:-0}" == 1 && "${2:-}" == *.metadata.txt ]]; then
    if [[ "${FAKE_LN_REPLACE_IMAGE_ON_RECEIPT_FAILURE:-0}" == 1 ]]; then
      image_path=${2%.metadata.txt}
      /bin/rm -f "${image_path}"
      printf 'unknown-replacement\n' >"${image_path}"
    fi
    echo 'requested fake receipt publish failure' >&2
    exit 92
  fi
  exec /bin/ln "$@"
  ;;
sha256sum)
  : "${REAL_SHA256SUM:?}"
  if [[ -n "${FAKE_SHA_LOG:-}" ]]; then
    printf '%s\n' "${1:-missing}" >>"${FAKE_SHA_LOG}"
  fi
  if [[ "${FAKE_SHA_REPLACE_SCRATCH_CANDIDATE:-0}" == 1 &&
    "${1:-}" == */candidate.sqsh.partial ]]; then
    printf 'hash\n' >"${FAKE_CANDIDATE_REPLACEMENT_EVENT_LOG:?}"
    /bin/mv "${1}" "${1}.original"
    printf 'fake-sqsh\n' >"${1}"
  fi
  if [[ "${FAKE_SHA_SWAP_CONTAINER_ON_CANDIDATE:-0}" == 1 &&
    "${1:-}" == ./*.sqsh ]]; then
    : "${FAKE_CONTAINER_DIR_TO_SWAP:?}"
    : "${FAKE_CONTAINER_SWAP_TARGET:?}"
    /bin/mv "${FAKE_CONTAINER_DIR_TO_SWAP}" \
      "${FAKE_CONTAINER_DIR_TO_SWAP}.pinned"
    /bin/ln -s "${FAKE_CONTAINER_SWAP_TARGET}" \
      "${FAKE_CONTAINER_DIR_TO_SWAP}"
  fi
  if [[ "${FAKE_SHA256SUM_INVALID:-0}" == 1 ]]; then
    echo 'not-a-valid-sha256  fake-input'
    exit 0
  fi
  exec "${REAL_SHA256SUM}" "$@"
  ;;
mktemp)
  : "${REAL_MKTEMP:?}"
  mktemp_template=${!#}
  if [[ "${FAKE_MKTEMP_SWAP_CONTAINER:-0}" == 1 &&
    "${mktemp_template}" == *'.sqsh.partial.job-'* ]]; then
    : "${FAKE_CONTAINER_DIR_TO_SWAP:?}"
    : "${FAKE_CONTAINER_SWAP_TARGET:?}"
    /bin/mv "${FAKE_CONTAINER_DIR_TO_SWAP}" \
      "${FAKE_CONTAINER_DIR_TO_SWAP}.pinned"
    /bin/ln -s "${FAKE_CONTAINER_SWAP_TARGET}" \
      "${FAKE_CONTAINER_DIR_TO_SWAP}"
  fi
  if [[ "${FAKE_MKTEMP_FAIL:-0}" == 1 ]]; then
    if [[ "${mktemp_template}" != *'.cleanup-'* ]]; then
      echo 'requested fake exclusive partial allocation failure' >&2
      exit 93
    fi
  fi
  exec "${REAL_MKTEMP}" "$@"
  ;;
rm)
  if [[ "${FAKE_SWAP_SCRATCH_DURING_CLEANUP:-0}" == 1 &&
    "${1:-}" == '-rf' && "${2:-}" == "${FAKE_EXPECTED_SCRATCH_DIRECTORY:-}" ]]; then
    scratch_owner=$(<"${2}/.stage-owner")
    /bin/mv "${2}" "${2}.owned-before-swap"
    /bin/mkdir "${2}"
    printf '%s\n' "${scratch_owner}" >"${2}/.stage-owner"
    printf 'unknown-cleanup-replacement\n' >"${2}/unknown-sentinel"
    printf 'rm-swap\n' >"${FAKE_SCRATCH_SWAP_EVENT_LOG}"
  fi
  exec /bin/rm "$@"
  ;;
mv)
  if [[ "${FAKE_SWAP_SCRATCH_DURING_CLEANUP:-0}" == 1 &&
    "${1:-}" == "${FAKE_EXPECTED_SCRATCH_DIRECTORY:-}" ]]; then
    scratch_owner=$(<"${1}/.stage-owner")
    /bin/mv "${1}" "${1}.owned-before-swap"
    /bin/mkdir "${1}"
    printf '%s\n' "${scratch_owner}" >"${1}/.stage-owner"
    printf 'unknown-cleanup-replacement\n' >"${1}/unknown-sentinel"
    printf 'mv-swap\n' >"${FAKE_SCRATCH_SWAP_EVENT_LOG}"
  fi
  exec /bin/mv "$@"
  ;;
cp)
  if [[ "${FAKE_CP_REQUIRE_SEALED_SOURCE:-0}" == 1 &&
    "${1:-}" == */candidate.sqsh.partial ]]; then
    if candidate_mode=$(stat -Lc '%a' "${1}" 2>/dev/null); then
      :
    else
      candidate_mode=$(stat -f '%Lp' "${1}")
    fi
    if ((8#${candidate_mode} & 8#222)); then
      echo 'scratch candidate was not sealed before publication' >&2
      exit 95
    fi
  fi
  if [[ "${FAKE_CP_REPLACE_SCRATCH_CANDIDATE:-0}" == 1 &&
    "${1:-}" == */candidate.sqsh.partial ]]; then
    printf 'copy\n' >"${FAKE_CANDIDATE_REPLACEMENT_EVENT_LOG:?}"
    /bin/mv "${1}" "${1}.original"
    printf 'fake-sqsh\n' >"${1}"
  fi
  if [[ "${FAKE_CP_SWAP_CONTAINER:-0}" == 1 ]]; then
    : "${FAKE_CONTAINER_DIR_TO_SWAP:?}"
    : "${FAKE_CONTAINER_SWAP_TARGET:?}"
    /bin/mv "${FAKE_CONTAINER_DIR_TO_SWAP}" \
      "${FAKE_CONTAINER_DIR_TO_SWAP}.pinned"
    /bin/ln -s "${FAKE_CONTAINER_SWAP_TARGET}" \
      "${FAKE_CONTAINER_DIR_TO_SWAP}"
  fi
  exec /bin/cp "$@"
  ;;
esac

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIRECTORY
readonly STAGE_SCRIPT=${SCRIPT_DIRECTORY}/stage_enroot_image.sbatch
readonly EXPECTED_LOG_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/container-validation/nightly-stage/logs
readonly VALID_SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl:33940349909
readonly VALID_SOURCE_COMMIT=2682b7e49c8877bcf02681fa2861c752f3e447f4
readonly VALID_LOCK_SHA256=95f63521d28a2a4104ff372c5985fe63826ab27d6901b78bada1ab1a89a81bf7
readonly VALID_PYPROJECT_SHA256=827f5f82c37dcf99454e47982bb0c0b8aa82c48cb869223e2a475db0c90cf0f9
readonly EXPECTED_IMAGE_SHA256=5812a2555ab50ed62abb3148cdfc97a911a8a3664277c20b7b60b65031377f0b
REAL_SHA256SUM=$(command -v sha256sum)
readonly REAL_SHA256SUM
REAL_MKTEMP=$(command -v mktemp)
readonly REAL_MKTEMP

TEST_ROOT=$(cd "$(mktemp -d)" && pwd -P)
readonly TEST_ROOT
readonly TEST_SCRATCH_USER_ROOT=${TEST_ROOT}/raid/scratch/sna
readonly TEST_SCRATCH_BASE=${TEST_SCRATCH_USER_ROOT}/nemo-rl-image-stage
mkdir -p "${TEST_ROOT}/raid/scratch" "${TEST_ROOT}/lustre"
readonly FAKE_BINARY_DIRECTORY=${TEST_ROOT}/bin
readonly TEST_TOOLING_WORKTREE=${TEST_ROOT}/tooling-worktree
readonly TOOLING_SCRIPT_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/stage_enroot_image.sbatch
readonly INSTRUMENTED_STAGE_SCRIPT=${TEST_TOOLING_WORKTREE}/${TOOLING_SCRIPT_RELATIVE_PATH}
TEST_TOOLING_COMMIT=
RESOLVED_TEST_TOOLING_WORKTREE=
RUN_STAGE_SCRIPT=
RUN_DIRECTORY=
RUN_STATUS=0

cleanup() {
  rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_file_contains() {
  local expected=$1
  local file=$2

  grep -Fx -- "${expected}" "${file}" >/dev/null ||
    fail "${file} does not contain: ${expected}"
}

assert_file_matches() {
  local expected_regex=$1
  local file=$2

  grep -E -- "${expected_regex}" "${file}" >/dev/null ||
    fail "${file} does not match: ${expected_regex}"
}

assert_no_publish_artifacts() {
  local directory=$1

  if find "${directory}" -mindepth 1 \
    \( -name '*.sqsh' -o -name '*.metadata.txt' -o -name '*.partial.*' \) \
    -print -quit | grep -q .; then
    fail "unexpected publish artifact remains in ${directory}"
  fi
}

assert_no_partial_artifacts() {
  local directory=$1

  if find "${directory}" -mindepth 1 -name '*.partial.*' -print -quit |
    grep -q .; then
    fail "partial publish artifact remains in ${directory}"
  fi
}

replace_receipt_value() {
  local receipt=$1
  local key=$2
  local value=$3
  local rewritten=${receipt}.rewritten

  awk -v key="${key}" -v value="${value}" '
    index($0, key "=") == 1 {
      print key "=" value
      next
    }
    { print }
  ' "${receipt}" >"${rewritten}"
  mv "${rewritten}" "${receipt}"
  chmod a-w "${receipt}"
}

assert_rejected_before_import() {
  local case_name=$1
  local omitted_variable=$2
  shift 2

  run_stage "${case_name}" "${omitted_variable}" "$@"
  if ((RUN_STATUS == 0)); then
    fail "${case_name} unexpectedly succeeded"
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail "${case_name} invoked enroot import before rejecting input"
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

run_stage() {
  local case_name=$1
  local omitted_variable=$2
  shift 2
  local -a environment

  RUN_DIRECTORY=${TEST_ROOT}/lustre/runs/${case_name}
  rm -rf "${RUN_DIRECTORY}"
  mkdir -p "${RUN_DIRECTORY}/containers"
  : >"${RUN_DIRECTORY}/enroot.log"

  environment=(
    "PATH=${FAKE_BINARY_DIRECTORY}:/usr/bin:/bin:/usr/sbin:/sbin"
    "FAKE_ENROOT_LOG=${RUN_DIRECTORY}/enroot.log"
    "FAKE_SHA_LOG=${RUN_DIRECTORY}/sha256sum.log"
    "REAL_SHA256SUM=${REAL_SHA256SUM}"
    "REAL_MKTEMP=${REAL_MKTEMP}"
    'FAKE_CP_REQUIRE_SEALED_SOURCE=1'
    'OUTPUT_PREFIX=nemo_rl_nightly'
    "CONTAINER_DIR=${RUN_DIRECTORY}/containers"
    'SLURM_JOB_ID=42001'
    'SLURM_JOB_NAME=test-candidate-stage'
    'SLURMD_NODENAME=test-node-0'
    "SLURM_SUBMIT_DIR=${TEST_TOOLING_WORKTREE}"
    "TOOLING_WORKTREE=${TEST_TOOLING_WORKTREE}"
    "TOOLING_COMMIT=${TEST_TOOLING_COMMIT}"
    'HF_TOKEN=SECRET_MARKER_DO_NOT_PUBLISH'
    'NGC_API_KEY=SECRET_MARKER_DO_NOT_PUBLISH'
  )
  if [[ "${omitted_variable}" != SOURCE_IMAGE ]]; then
    environment+=("SOURCE_IMAGE=${VALID_SOURCE_IMAGE}")
  fi
  if [[ "${omitted_variable}" != SOURCE_COMMIT ]]; then
    environment+=("SOURCE_COMMIT=${VALID_SOURCE_COMMIT}")
  fi
  if [[ "${omitted_variable}" != SOURCE_LOCK_SHA256 ]]; then
    environment+=("SOURCE_LOCK_SHA256=${VALID_LOCK_SHA256}")
  fi
  if [[ "${omitted_variable}" != SOURCE_PYPROJECT_SHA256 ]]; then
    environment+=("SOURCE_PYPROJECT_SHA256=${VALID_PYPROJECT_SHA256}")
  fi

  set +e
  env -i "${environment[@]}" "$@" \
    /bin/bash "${RUN_STAGE_SCRIPT}" \
    >"${RUN_DIRECTORY}/stdout" 2>"${RUN_DIRECTORY}/stderr"
  RUN_STATUS=$?
  set -e
}

test_required_and_exact_pins() {
  assert_rejected_before_import missing-source-image SOURCE_IMAGE
  assert_rejected_before_import missing-source-commit SOURCE_COMMIT
  assert_rejected_before_import missing-lock-sha SOURCE_LOCK_SHA256
  assert_rejected_before_import missing-pyproject-sha SOURCE_PYPROJECT_SHA256

  assert_rejected_before_import invalid-source-no-registry '' \
    SOURCE_IMAGE=nemo-rl:nightly
  assert_rejected_before_import invalid-source-url '' \
    SOURCE_IMAGE=https://nvcr.io/nvidian/nemo-rl:33940349909
  assert_rejected_before_import invalid-source-implicit-tag '' \
    SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl
  assert_rejected_before_import invalid-source-commit '' \
    SOURCE_COMMIT=ABCDEF0123456789abcdef0123456789abcdef01
  assert_rejected_before_import invalid-lock-sha '' \
    SOURCE_LOCK_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  assert_rejected_before_import invalid-pyproject-sha '' \
    SOURCE_PYPROJECT_SHA256=BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
  assert_rejected_before_import invalid-publish-filesystem '' \
    "CONTAINER_DIR=${TEST_ROOT}/shared/containers"

  assert_rejected_before_import unsupported-enroot-digest '' \
    SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  assert_rejected_before_import unsupported-enroot-version '' \
    FAKE_ENROOT_VERSION=4.0.0
}

test_canonical_lustre_containment() {
  local outside_directory=${TEST_ROOT}/outside-publish
  local symlink_path=${TEST_ROOT}/lustre/linked-outside
  local writable_path=${TEST_ROOT}/lustre/peer-writable

  mkdir -p "${outside_directory}" "${TEST_ROOT}/lustre"
  assert_rejected_before_import traversal-publish-path '' \
    "CONTAINER_DIR=${TEST_ROOT}/lustre/../outside-publish"

  ln -s "${outside_directory}" "${symlink_path}"
  assert_rejected_before_import symlink-publish-path '' \
    "CONTAINER_DIR=${symlink_path}"
  mkdir "${writable_path}"
  chmod 0770 "${writable_path}"
  assert_rejected_before_import peer-writable-publish-path '' \
    "CONTAINER_DIR=${writable_path}"
  if find "${outside_directory}" -mindepth 1 -print -quit | grep -q .; then
    fail 'publish-path escape created artifacts outside the canonical Lustre root'
  fi
}

test_hostile_private_scratch_parents_are_rejected() {
  local outside_directory=${TEST_ROOT}/outside-scratch

  /bin/rm -rf "${TEST_SCRATCH_USER_ROOT}"
  mkdir "${outside_directory}"
  ln -s "${outside_directory}" "${TEST_SCRATCH_USER_ROOT}"
  assert_rejected_before_import symlink-scratch-user-root '' SLURM_JOB_ID=42034
  if find "${outside_directory}" -mindepth 1 -print -quit | grep -q .; then
    fail 'symlink scratch user root redirected staging outside private scratch'
  fi

  /bin/rm -f "${TEST_SCRATCH_USER_ROOT}"
  mkdir -m 700 "${TEST_SCRATCH_USER_ROOT}"
  ln -s "${outside_directory}" "${TEST_SCRATCH_BASE}"
  assert_rejected_before_import symlink-scratch-base '' SLURM_JOB_ID=42035
  if find "${outside_directory}" -mindepth 1 -print -quit | grep -q .; then
    fail 'symlink scratch base redirected staging outside private scratch'
  fi

  /bin/rm -f "${TEST_SCRATCH_BASE}"
  mkdir -m 700 "${TEST_SCRATCH_BASE}"
  chmod 0770 "${TEST_SCRATCH_BASE}"
  assert_rejected_before_import peer-writable-scratch-base '' SLURM_JOB_ID=42036
  chmod 0700 "${TEST_SCRATCH_BASE}"

  chmod 0702 "${TEST_SCRATCH_BASE}"
  assert_rejected_before_import other-writable-scratch-base '' SLURM_JOB_ID=42041
  chmod 0700 "${TEST_SCRATCH_BASE}"

  chmod 0770 "${TEST_SCRATCH_USER_ROOT}"
  assert_rejected_before_import peer-writable-scratch-user-root '' \
    SLURM_JOB_ID=42037
  chmod 0700 "${TEST_SCRATCH_USER_ROOT}"
}

test_preexisting_scratch_is_never_claimed_or_deleted() {
  local scratch_directory=${TEST_SCRATCH_BASE}/job-42021
  local sentinel=${scratch_directory}/unknown-sentinel

  mkdir -p "${scratch_directory}"
  printf 'do-not-delete\n' >"${sentinel}"
  run_stage preexisting-scratch '' SLURM_JOB_ID=42021
  if ((RUN_STATUS == 0)); then
    fail 'preexisting scratch directory was accepted'
  fi
  assert_file_contains 'do-not-delete' "${sentinel}"
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'preexisting scratch directory was rejected only after import'
  fi
}

test_successful_import_cannot_hide_scratch_replacement() {
  local sentinel

  run_stage replaced-scratch-success '' \
    FAKE_SCRATCH_REPLACE_AFTER_IMPORT=1 \
    FAKE_SCRATCH_REPLAY_OWNER=1 \
    FAKE_SCRATCH_REPLACE_SUCCESS=1 \
    FAKE_SCRATCH_REPLACEMENT_MARKER=unknown-replacement-success \
    SLURM_JOB_ID=42038
  if ((RUN_STATUS == 0)); then
    fail 'successful import hid a same-name scratch-directory replacement'
  fi
  sentinel=$(find "${TEST_SCRATCH_BASE}" \
    -name unknown-sentinel -exec grep -Fl 'unknown-replacement-success' {} \; |
    head -1)
  if [[ -z "${sentinel}" ]]; then
    sed -n '1,200p' "${RUN_DIRECTORY}/stderr" >&2
    fail 'replaced scratch bytes were deleted'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_candidate_replacement_while_hashing_is_rejected() {
  local event_log=${TEST_ROOT}/candidate-hash-replacement.event

  run_stage replaced-candidate-during-hash '' \
    FAKE_SHA_REPLACE_SCRATCH_CANDIDATE=1 \
    "FAKE_CANDIDATE_REPLACEMENT_EVENT_LOG=${event_log}" \
    SLURM_JOB_ID=42039
  if ((RUN_STATUS == 0)); then
    fail 'same-name scratch candidate replacement while hashing was accepted'
  fi
  assert_file_contains hash "${event_log}"
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_candidate_replacement_during_publish_is_rejected() {
  local event_log=${TEST_ROOT}/candidate-copy-replacement.event

  run_stage replaced-candidate-during-publish '' \
    FAKE_CP_REPLACE_SCRATCH_CANDIDATE=1 \
    "FAKE_CANDIDATE_REPLACEMENT_EVENT_LOG=${event_log}" \
    SLURM_JOB_ID=42040
  if ((RUN_STATUS == 0)); then
    fail 'same-name scratch candidate replacement during publish was accepted'
  fi
  assert_file_contains copy "${event_log}"
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_symlink_candidate_is_rejected() {
  run_stage symlink-candidate '' \
    FAKE_ENROOT_SYMLINK_CANDIDATE=1 \
    SLURM_JOB_ID=42042
  if ((RUN_STATUS == 0)); then
    fail 'symlink candidate returned by successful import was accepted'
  fi
  assert_file_contains 'command=import' "${RUN_DIRECTORY}/enroot.log"
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_replaced_scratch_is_preserved_on_cleanup() {
  local sentinel

  run_stage replaced-scratch '' \
    FAKE_SCRATCH_REPLACE_AFTER_IMPORT=1 \
    FAKE_SCRATCH_REPLAY_OWNER=1 \
    SLURM_JOB_ID=42022
  if ((RUN_STATUS == 0)); then
    fail 'scratch replacement failure unexpectedly succeeded'
  fi
  sentinel=$(find "${TEST_SCRATCH_BASE}" \
    -name unknown-sentinel -exec grep -Fl 'unknown-replacement' {} \; |
    head -1)
  if [[ -z "${sentinel}" ]]; then
    sed -n '1,200p' "${RUN_DIRECTORY}/stderr" >&2
    fail 'replaced scratch bytes were deleted'
  fi
  assert_file_contains 'unknown-replacement' "${sentinel}"
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_container_directory_swap_after_import_is_rejected() {
  local outside_directory=${TEST_ROOT}/post-import-outside

  mkdir "${outside_directory}"
  run_stage swapped-container-directory '' \
    FAKE_ENROOT_SWAP_CONTAINER=1 \
    "FAKE_CONTAINER_DIR_TO_SWAP=${TEST_ROOT}/lustre/runs/swapped-container-directory/containers" \
    "FAKE_CONTAINER_SWAP_TARGET=${outside_directory}" \
    SLURM_JOB_ID=42026
  if ((RUN_STATUS == 0)); then
    fail 'container directory replacement after import was accepted'
  fi
  if find "${outside_directory}" -mindepth 1 -print -quit | grep -q .; then
    fail 'container directory replacement redirected publication outside Lustre'
  fi
}

test_container_swap_between_check_and_write_cannot_escape() {
  local outside_directory=${TEST_ROOT}/between-check-outside

  mkdir "${outside_directory}"
  run_stage between-check-container-swap '' \
    FAKE_CP_SWAP_CONTAINER=1 \
    "FAKE_CONTAINER_DIR_TO_SWAP=${TEST_ROOT}/lustre/runs/between-check-container-swap/containers" \
    "FAKE_CONTAINER_SWAP_TARGET=${outside_directory}" \
    SLURM_JOB_ID=42031
  if ((RUN_STATUS == 0)); then
    fail 'container swap between validation and write was accepted'
  fi
  if find "${outside_directory}" -mindepth 1 -print -quit | grep -q .; then
    fail 'check/write container race redirected bytes outside Lustre'
  fi
}

test_scratch_swap_during_cleanup_preserves_unknown_bytes() {
  local scratch_directory=${TEST_SCRATCH_BASE}/job-42030
  local event_log=${TEST_ROOT}/scratch-cleanup-swap.event
  local preserved_sentinel

  run_stage scratch-cleanup-swap '' \
    FAKE_SWAP_SCRATCH_DURING_CLEANUP=1 \
    "FAKE_EXPECTED_SCRATCH_DIRECTORY=${scratch_directory}" \
    "FAKE_SCRATCH_SWAP_EVENT_LOG=${event_log}" \
    SLURM_JOB_ID=42030
  if ((RUN_STATUS == 0)); then
    fail 'scratch cleanup replacement was not reported as a failure'
  fi
  [[ -f "${event_log}" ]] || fail 'scratch cleanup race was not injected'
  preserved_sentinel=$(find "${TEST_SCRATCH_BASE}" \
    -name unknown-sentinel -exec grep -Fl 'unknown-cleanup-replacement' {} \; |
    head -1)
  [[ -n "${preserved_sentinel}" ]] ||
    fail 'scratch cleanup deleted replayed unknown bytes'
  assert_file_contains 'unknown-cleanup-replacement' "${preserved_sentinel}"
}

test_success_uses_job_scratch_and_publishes_receipt() {
  local expected_scratch=${TEST_SCRATCH_BASE}/job-42001
  local final_image
  local receipt
  local tooling_commit

  run_stage success ''
  if ((RUN_STATUS != 0)); then
    sed -n '1,200p' "${RUN_DIRECTORY}/stderr" >&2
    fail "valid stage failed with status ${RUN_STATUS}"
  fi

  assert_file_contains "enroot_cache=${expected_scratch}/enroot-cache" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains "enroot_data=${expected_scratch}/enroot-data" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains "enroot_runtime=${expected_scratch}/enroot-runtime" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains "tmpdir=${expected_scratch}/tmp" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains "tmp=${expected_scratch}/tmp" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains "temp=${expected_scratch}/tmp" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains "output=${expected_scratch}/candidate.sqsh.partial" \
    "${RUN_DIRECTORY}/enroot.log"
  assert_file_contains 'uri=docker://nvcr.io#nvidian/nemo-rl:33940349909' \
    "${RUN_DIRECTORY}/enroot.log"
  if [[ $(grep -Fxc -- "${expected_scratch}/candidate.sqsh.partial" \
    "${RUN_DIRECTORY}/sha256sum.log") != 1 ]]; then
    fail 'imported scratch candidate was not hashed exactly once'
  fi
  if [[ $(grep -Ec '^\./\.nemo_rl_nightly_sha256-.*\.sqsh\.partial\.job-42001\.' \
    "${RUN_DIRECTORY}/sha256sum.log") != 1 ]]; then
    fail 'shared-storage candidate partial was not hashed exactly once'
  fi

  if [[ -e "${expected_scratch}" ]]; then
    fail "job scratch was not cleaned after success"
  fi
  if find "${RUN_DIRECTORY}/containers" -maxdepth 1 \
    \( -name '.enroot*' -o -name 'tmp' \) -print -quit | grep -q .; then
    fail 'high-churn enroot state was created in CONTAINER_DIR'
  fi

  final_image=${RUN_DIRECTORY}/containers/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  receipt=${final_image}.metadata.txt
  if [[ ! -f "${final_image}" || -L "${final_image}" ]]; then
    sed -n '1,200p' "${RUN_DIRECTORY}/stdout" >&2
    sed -n '1,200p' "${RUN_DIRECTORY}/stderr" >&2
    find "${RUN_DIRECTORY}/containers" -maxdepth 1 -print >&2
    fail "content-addressed candidate image is missing or is a symlink"
  fi
  [[ -f "${receipt}" && ! -L "${receipt}" ]] ||
    fail "candidate receipt is missing or is a symlink"
  [[ ! -e "${RUN_DIRECTORY}/containers/nemo_rl_nightly.sqsh" ]] ||
    fail 'stage created a stable current/validated link before smoke validation'
  if find "${RUN_DIRECTORY}/containers" -maxdepth 1 -type l -print -quit |
    grep -q .; then
    fail 'stage created an unexpected symlink'
  fi
  assert_no_partial_artifacts "${RUN_DIRECTORY}/containers"
  printf 'fake-sqsh\n' | cmp -s - "${final_image}" ||
    fail 'published image bytes differ from imported bytes'

  tooling_commit=${TEST_TOOLING_COMMIT}
  assert_file_contains 'receipt_version=1' "${receipt}"
  assert_file_contains 'artifact_kind=nemo_rl_enroot_candidate' "${receipt}"
  assert_file_contains 'validation_state=unvalidated_candidate' "${receipt}"
  assert_file_contains "requested_source_ref=${VALID_SOURCE_IMAGE}" "${receipt}"
  assert_file_contains 'requested_source_registry=nvcr.io' "${receipt}"
  assert_file_contains 'requested_source_repository=nvidian/nemo-rl' "${receipt}"
  assert_file_contains 'requested_source_reference=33940349909' "${receipt}"
  assert_file_contains 'requested_source_reference_kind=tag' "${receipt}"
  assert_file_contains "expected_source_commit=${VALID_SOURCE_COMMIT}" "${receipt}"
  assert_file_contains "expected_source_lock_sha256=${VALID_LOCK_SHA256}" "${receipt}"
  assert_file_contains "expected_source_pyproject_sha256=${VALID_PYPROJECT_SHA256}" \
    "${receipt}"
  assert_file_contains 'imported_enroot_uri=docker://nvcr.io#nvidian/nemo-rl:33940349909' \
    "${receipt}"
  assert_file_contains "image_sha256=${EXPECTED_IMAGE_SHA256}" "${receipt}"
  assert_file_contains "image_filename=$(basename "${final_image}")" "${receipt}"
  assert_file_matches '^tooling_blob_sha256=[0-9a-f]{64}$' "${receipt}"
  assert_file_contains "tooling_commit=${tooling_commit}" "${receipt}"
  assert_file_contains "tooling_worktree=${RESOLVED_TEST_TOOLING_WORKTREE}" "${receipt}"
  assert_file_contains 'slurm_job_id=42001' "${receipt}"
  assert_file_contains 'slurm_job_name=test-candidate-stage' "${receipt}"
  assert_file_contains 'slurm_node=test-node-0' "${receipt}"
  assert_file_matches '^stage_host=.+$' "${receipt}"
  assert_file_matches \
    '^stage_started_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' \
    "${receipt}"
  assert_file_matches \
    '^stage_finished_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' \
    "${receipt}"
  assert_file_contains 'enroot_version=3.5.0' "${receipt}"
  if grep -Eq '^source_(commit|lock_sha256|pyproject_sha256)=' "${receipt}"; then
    fail 'unvalidated receipt claimed verified embedded source provenance'
  fi
  if grep -Fq 'SECRET_MARKER_DO_NOT_PUBLISH' "${receipt}"; then
    fail 'receipt leaked an injected credential'
  fi
}

test_verified_candidate_is_reused_without_overwrite() {
  local first_run=${TEST_ROOT}/lustre/runs/success
  local final_image=${first_run}/containers/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  local receipt=${final_image}.metadata.txt
  local snapshot_directory=${TEST_ROOT}/reuse-snapshots

  mkdir -p "${snapshot_directory}"
  cp "${final_image}" "${snapshot_directory}/candidate.sqsh"
  cp "${receipt}" "${snapshot_directory}/receipt.txt"

  RUN_DIRECTORY=${first_run}
  : >"${RUN_DIRECTORY}/enroot.log"
  set +e
  env -i \
    "PATH=${FAKE_BINARY_DIRECTORY}:/usr/bin:/bin:/usr/sbin:/sbin" \
    "FAKE_ENROOT_LOG=${RUN_DIRECTORY}/enroot.log" \
    "REAL_SHA256SUM=${REAL_SHA256SUM}" \
    "REAL_MKTEMP=${REAL_MKTEMP}" \
    "SOURCE_IMAGE=${VALID_SOURCE_IMAGE}" \
    "SOURCE_COMMIT=${VALID_SOURCE_COMMIT}" \
    "SOURCE_LOCK_SHA256=${VALID_LOCK_SHA256}" \
    "SOURCE_PYPROJECT_SHA256=${VALID_PYPROJECT_SHA256}" \
    OUTPUT_PREFIX=nemo_rl_nightly \
    "CONTAINER_DIR=${RUN_DIRECTORY}/containers" \
    SLURM_JOB_ID=42002 \
    SLURM_JOB_NAME=test-candidate-reuse \
    SLURMD_NODENAME=test-node-1 \
    "SLURM_SUBMIT_DIR=${TEST_TOOLING_WORKTREE}" \
    "TOOLING_WORKTREE=${TEST_TOOLING_WORKTREE}" \
    "TOOLING_COMMIT=${TEST_TOOLING_COMMIT}" \
    /bin/bash "${RUN_STAGE_SCRIPT}" \
    >"${RUN_DIRECTORY}/reuse.stdout" 2>"${RUN_DIRECTORY}/reuse.stderr"
  RUN_STATUS=$?
  set -e

  if ((RUN_STATUS != 0)); then
    sed -n '1,200p' "${RUN_DIRECTORY}/reuse.stderr" >&2
    fail "exact candidate reuse failed with status ${RUN_STATUS}"
  fi
  cmp -s "${snapshot_directory}/candidate.sqsh" "${final_image}" ||
    fail 'verified candidate bytes were overwritten'
  cmp -s "${snapshot_directory}/receipt.txt" "${receipt}" ||
    fail 'verified candidate receipt was overwritten'
  assert_file_contains 'candidate_status=reused_verified' \
    "${RUN_DIRECTORY}/reuse.stdout"
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42002" ]] ||
    fail 'job scratch was not cleaned after verified reuse'
  assert_no_partial_artifacts "${RUN_DIRECTORY}/containers"
}

test_immutable_image_collision_fails_closed() {
  local first_run=${TEST_ROOT}/lustre/runs/success
  local final_image=${first_run}/containers/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  local receipt=${final_image}.metadata.txt
  local receipt_snapshot=${TEST_ROOT}/collision-receipt.txt

  cp "${receipt}" "${receipt_snapshot}"
  chmod u+w "${final_image}"
  printf 'corrupt-existing-candidate\n' >"${final_image}"

  RUN_DIRECTORY=${first_run}
  : >"${RUN_DIRECTORY}/enroot.log"
  set +e
  env -i \
    "PATH=${FAKE_BINARY_DIRECTORY}:/usr/bin:/bin:/usr/sbin:/sbin" \
    "FAKE_ENROOT_LOG=${RUN_DIRECTORY}/enroot.log" \
    "REAL_SHA256SUM=${REAL_SHA256SUM}" \
    "REAL_MKTEMP=${REAL_MKTEMP}" \
    "SOURCE_IMAGE=${VALID_SOURCE_IMAGE}" \
    "SOURCE_COMMIT=${VALID_SOURCE_COMMIT}" \
    "SOURCE_LOCK_SHA256=${VALID_LOCK_SHA256}" \
    "SOURCE_PYPROJECT_SHA256=${VALID_PYPROJECT_SHA256}" \
    OUTPUT_PREFIX=nemo_rl_nightly \
    "CONTAINER_DIR=${RUN_DIRECTORY}/containers" \
    SLURM_JOB_ID=42003 \
    SLURM_JOB_NAME=test-candidate-collision \
    SLURMD_NODENAME=test-node-2 \
    "SLURM_SUBMIT_DIR=${TEST_TOOLING_WORKTREE}" \
    "TOOLING_WORKTREE=${TEST_TOOLING_WORKTREE}" \
    "TOOLING_COMMIT=${TEST_TOOLING_COMMIT}" \
    /bin/bash "${RUN_STAGE_SCRIPT}" \
    >"${RUN_DIRECTORY}/collision.stdout" 2>"${RUN_DIRECTORY}/collision.stderr"
  RUN_STATUS=$?
  set -e

  if ((RUN_STATUS == 0)); then
    fail 'corrupt immutable candidate was accepted'
  fi
  assert_file_contains 'corrupt-existing-candidate' "${final_image}"
  cmp -s "${receipt_snapshot}" "${receipt}" ||
    fail 'collision handling overwrote the existing receipt'
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42003" ]] ||
    fail 'job scratch was not cleaned after immutable collision'
  assert_no_partial_artifacts "${RUN_DIRECTORY}/containers"
}

test_immutable_receipt_collision_fails_closed() {
  local final_image
  local receipt
  local image_snapshot

  run_stage receipt-collision '' SLURM_JOB_ID=42004
  if ((RUN_STATUS != 0)); then
    fail "receipt collision setup failed with status ${RUN_STATUS}"
  fi
  final_image=${RUN_DIRECTORY}/containers/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  receipt=${final_image}.metadata.txt
  image_snapshot=${TEST_ROOT}/receipt-collision-image.sqsh
  cp "${final_image}" "${image_snapshot}"
  chmod u+w "${receipt}"
  printf 'receipt_version=corrupt\n' >"${receipt}"

  run_stage receipt-collision-rerun '' \
    SLURM_JOB_ID=42005 \
    "CONTAINER_DIR=$(dirname "${final_image}")"
  if ((RUN_STATUS == 0)); then
    fail 'corrupt immutable receipt was accepted'
  fi
  cmp -s "${image_snapshot}" "${final_image}" ||
    fail 'receipt collision handling overwrote image bytes'
  assert_file_contains 'receipt_version=corrupt' "${receipt}"
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42005" ]] ||
    fail 'job scratch was not cleaned after receipt collision'
  assert_no_partial_artifacts "$(dirname "${final_image}")"
}

test_import_failure_cleans_partial_state() {
  run_stage import-failure '' FAKE_ENROOT_FAIL=1 SLURM_JOB_ID=42006
  if ((RUN_STATUS != 91)); then
    sed -n '1,200p' "${RUN_DIRECTORY}/stderr" >&2
    fail "import failure returned ${RUN_STATUS}, expected 91"
  fi
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42006" ]] ||
    fail 'job scratch was not cleaned after import failure'
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_reuse_rejects_different_tooling_identity() {
  local setup_directory
  local final_image
  local receipt
  local current_tooling_blob
  local current_tooling_commit

  run_stage stale-tooling-blob-setup '' SLURM_JOB_ID=42008
  if ((RUN_STATUS != 0)); then
    fail "stale tooling blob setup failed with status ${RUN_STATUS}"
  fi
  setup_directory=${RUN_DIRECTORY}/containers
  final_image=${setup_directory}/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  receipt=${final_image}.metadata.txt
  current_tooling_blob=$(awk -F= '$1 == "tooling_blob_sha256" { print $2 }' "${receipt}")
  replace_receipt_value "${receipt}" tooling_blob_sha256 \
    cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  run_stage stale-tooling-blob-rerun '' \
    SLURM_JOB_ID=42009 \
    "CONTAINER_DIR=${setup_directory}"
  if ((RUN_STATUS == 0)); then
    fail 'reuse accepted a candidate created by a different tooling blob'
  fi
  assert_file_contains \
    'tooling_blob_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' \
    "${receipt}"
  replace_receipt_value "${receipt}" tooling_blob_sha256 "${current_tooling_blob}"

  current_tooling_commit=$(awk -F= '$1 == "tooling_commit" { print $2 }' "${receipt}")
  replace_receipt_value "${receipt}" tooling_commit \
    dddddddddddddddddddddddddddddddddddddddd
  run_stage stale-tooling-commit-rerun '' \
    SLURM_JOB_ID=42010 \
    "CONTAINER_DIR=${setup_directory}"
  if ((RUN_STATUS == 0)); then
    fail 'reuse accepted a candidate created from a different tooling commit'
  fi
  assert_file_contains \
    'tooling_commit=dddddddddddddddddddddddddddddddddddddddd' "${receipt}"
  replace_receipt_value "${receipt}" tooling_commit "${current_tooling_commit}"
  assert_no_partial_artifacts "${setup_directory}"
}

test_hostile_metadata_values_cannot_publish_invalid_receipt() {
  run_stage hostile-job-name '' \
    SLURM_JOB_ID=42013 \
    $'SLURM_JOB_NAME=hostile\nreceipt_version=99'
  if ((RUN_STATUS == 0)); then
    fail 'newline-injected Slurm job name published an invalid receipt'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42013" ]] ||
    fail 'job scratch was not cleaned after hostile metadata rejection'

  run_stage hostile-enroot-version '' \
    SLURM_JOB_ID=42014 \
    $'FAKE_ENROOT_VERSION=3.5.0\nreceipt_version=99'
  if ((RUN_STATUS == 0)); then
    fail 'newline-injected enroot version was accepted'
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'hostile enroot version was rejected only after import'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_tooling_trust_failures_precede_import() {
  local dirty_marker=${TEST_TOOLING_WORKTREE}/dirty-marker
  local mismatched_script=${TEST_ROOT}/mismatched-stage-script.sbatch
  local tooling_snapshot=${TEST_ROOT}/committed-stage-script.snapshot
  local assume_unchanged_status

  touch "${dirty_marker}"
  run_stage dirty-tooling-worktree '' SLURM_JOB_ID=42017
  rm -f "${dirty_marker}"
  if ((RUN_STATUS == 0)); then
    fail 'dirty tooling worktree was accepted'
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'dirty tooling worktree was rejected only after import'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"

  run_stage mismatched-tooling-commit '' \
    SLURM_JOB_ID=42018 \
    TOOLING_COMMIT=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
  if ((RUN_STATUS == 0)); then
    fail 'mismatched tooling commit was accepted'
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'mismatched tooling commit was rejected only after import'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"

  cp "${INSTRUMENTED_STAGE_SCRIPT}" "${mismatched_script}"
  printf '\n# simulated Slurm spool mutation\n' >>"${mismatched_script}"
  chmod +x "${mismatched_script}"
  RUN_STAGE_SCRIPT=${mismatched_script}
  run_stage mismatched-spooled-blob '' SLURM_JOB_ID=42019
  RUN_STAGE_SCRIPT=${INSTRUMENTED_STAGE_SCRIPT}
  if ((RUN_STATUS == 0)); then
    fail 'spooled script bytes different from committed tooling were accepted'
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'spooled tooling blob mismatch was rejected only after import'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"

  cp -p "${INSTRUMENTED_STAGE_SCRIPT}" "${tooling_snapshot}"
  git -C "${TEST_TOOLING_WORKTREE}" update-index --assume-unchanged \
    "${TOOLING_SCRIPT_RELATIVE_PATH}"
  printf '\n# hidden worktree mutation\n' >>"${INSTRUMENTED_STAGE_SCRIPT}"
  run_stage assume-unchanged-tooling-blob '' SLURM_JOB_ID=42020
  assume_unchanged_status=${RUN_STATUS}
  cp -p "${tooling_snapshot}" "${INSTRUMENTED_STAGE_SCRIPT}"
  git -C "${TEST_TOOLING_WORKTREE}" update-index --no-assume-unchanged \
    "${TOOLING_SCRIPT_RELATIVE_PATH}"
  if ((assume_unchanged_status == 0)); then
    fail 'worktree bytes hidden from git status were accepted as committed tooling'
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'hidden worktree blob mismatch was rejected only after import'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_receipt_publish_failure_rolls_back_new_candidate() {
  run_stage receipt-publish-failure '' \
    FAKE_LN_FAIL_RECEIPT=1 \
    SLURM_JOB_ID=42007
  if ((RUN_STATUS == 0)); then
    fail 'receipt publish failure unexpectedly succeeded'
  fi
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42007" ]] ||
    fail 'job scratch was not cleaned after receipt publish failure'
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
}

test_receipt_publish_failure_never_deletes_replacement_bytes() {
  local final_image

  run_stage receipt-publish-replaced-image '' \
    FAKE_LN_FAIL_RECEIPT=1 \
    FAKE_LN_REPLACE_IMAGE_ON_RECEIPT_FAILURE=1 \
    SLURM_JOB_ID=42023
  if ((RUN_STATUS == 0)); then
    fail 'receipt publish replacement failure unexpectedly succeeded'
  fi
  final_image=${RUN_DIRECTORY}/containers/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  assert_file_contains 'unknown-replacement' "${final_image}"
  [[ ! -e "${final_image}.metadata.txt" ]] ||
    fail 'receipt was published after replacement failure'
  assert_no_partial_artifacts "${RUN_DIRECTORY}/containers"
}

test_orphan_image_is_recovered_without_overwrite() {
  local setup_directory
  local final_image
  local image_snapshot=${TEST_ROOT}/orphan-image.snapshot

  run_stage orphan-image-setup '' SLURM_JOB_ID=42024
  if ((RUN_STATUS != 0)); then
    fail "orphan image setup failed with status ${RUN_STATUS}"
  fi
  setup_directory=${RUN_DIRECTORY}/containers
  final_image=${setup_directory}/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  cp "${final_image}" "${image_snapshot}"
  chmod u+w "${final_image}.metadata.txt"
  rm -f "${final_image}.metadata.txt"

  run_stage orphan-image-recovery '' \
    SLURM_JOB_ID=42025 \
    "CONTAINER_DIR=${setup_directory}"
  if ((RUN_STATUS != 0)); then
    sed -n '1,200p' "${RUN_DIRECTORY}/stderr" >&2
    fail "exact orphan image was not recovered: status ${RUN_STATUS}"
  fi
  cmp -s "${image_snapshot}" "${final_image}" ||
    fail 'orphan-image recovery overwrote existing image bytes'
  [[ -f "${final_image}.metadata.txt" ]] ||
    fail 'orphan-image recovery did not publish a receipt'
  assert_file_contains 'candidate_status=recovered_orphan_image' \
    "${RUN_DIRECTORY}/stdout"
  assert_no_partial_artifacts "${setup_directory}"
}

test_signal_after_receipt_link_preserves_complete_candidate() {
  local setup_directory
  local final_image

  run_stage signal-after-receipt-link '' \
    FAKE_LN_SIGNAL_AFTER_RECEIPT_LINK=1 \
    SLURM_JOB_ID=42027
  if ((RUN_STATUS != 143)); then
    fail "signal after receipt link returned ${RUN_STATUS}, expected 143"
  fi
  setup_directory=${RUN_DIRECTORY}/containers
  final_image=${setup_directory}/nemo_rl_nightly_sha256-${EXPECTED_IMAGE_SHA256}.sqsh
  [[ -f "${final_image}" && -f "${final_image}.metadata.txt" ]] ||
    fail 'signal after receipt link left an incomplete final candidate'
  assert_no_partial_artifacts "${setup_directory}"

  run_stage signal-after-receipt-rerun '' \
    SLURM_JOB_ID=42028 \
    "CONTAINER_DIR=${setup_directory}"
  if ((RUN_STATUS != 0)); then
    fail 'candidate left by receipt-link signal was not reusable'
  fi
  assert_file_contains 'candidate_status=reused_verified' \
    "${RUN_DIRECTORY}/stdout"
}

test_success_cleanup_preserves_replaced_partial() {
  local replaced_partial

  run_stage success-replaced-partial '' \
    FAKE_LN_REPLACE_PARTIAL_AFTER_RECEIPT_LINK=1 \
    SLURM_JOB_ID=42029
  if ((RUN_STATUS == 0)); then
    fail 'success cleanup silently deleted a replaced partial'
  fi
  replaced_partial=$(find "${RUN_DIRECTORY}/containers" \
    -name '*.metadata.txt.partial.*' -print -quit)
  [[ -n "${replaced_partial}" ]] ||
    fail 'replaced partial was deleted during success cleanup'
  assert_file_contains 'unknown-partial-replacement' "${replaced_partial}"
}

test_reuse_rejects_container_swap_during_validation() {
  local setup_directory
  local outside_directory=${TEST_ROOT}/reuse-validation-outside

  run_stage reuse-swap-setup '' SLURM_JOB_ID=42032
  if ((RUN_STATUS != 0)); then
    fail "reuse swap setup failed with status ${RUN_STATUS}"
  fi
  setup_directory=${RUN_DIRECTORY}/containers
  mkdir "${outside_directory}"
  run_stage reuse-swap-rerun '' \
    FAKE_SHA_SWAP_CONTAINER_ON_CANDIDATE=1 \
    "FAKE_CONTAINER_DIR_TO_SWAP=${setup_directory}" \
    "FAKE_CONTAINER_SWAP_TARGET=${outside_directory}" \
    "CONTAINER_DIR=${setup_directory}" \
    SLURM_JOB_ID=42033
  if ((RUN_STATUS == 0)); then
    fail 'reuse reported success after its absolute container path changed'
  fi
  if find "${outside_directory}" -mindepth 1 -print -quit | grep -q .; then
    fail 'reuse validation wrote through a replaced container path'
  fi
}

test_invalid_hash_tool_output_fails_before_import() {
  run_stage invalid-sha256-output '' \
    FAKE_SHA256SUM_INVALID=1 \
    SLURM_JOB_ID=42015
  if ((RUN_STATUS == 0)); then
    fail 'invalid sha256sum output unexpectedly produced a candidate'
  fi
  if grep -Fq 'command=import' "${RUN_DIRECTORY}/enroot.log"; then
    fail 'invalid tooling hash was not rejected before enroot import'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42015" ]] ||
    fail 'job scratch was not cleaned after invalid hash output'
}

test_exclusive_partial_allocation_failure_leaves_no_candidate() {
  run_stage partial-allocation-failure '' \
    FAKE_MKTEMP_FAIL=1 \
    SLURM_JOB_ID=42016
  if ((RUN_STATUS == 0)); then
    fail 'candidate publish did not use exclusive partial allocation'
  fi
  assert_no_publish_artifacts "${RUN_DIRECTORY}/containers"
  [[ ! -e "${TEST_SCRATCH_BASE}/job-42016" ]] ||
    fail 'job scratch was not cleaned after partial allocation failure'
}

[[ -x "${STAGE_SCRIPT}" ]] || fail "missing executable stage script: ${STAGE_SCRIPT}"
mkdir -p "${FAKE_BINARY_DIRECTORY}"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/enroot"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/ln"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/mktemp"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/rm"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/mv"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/cp"
ln -s "${SCRIPT_DIRECTORY}/test_stage_enroot_image_static.sh" \
  "${FAKE_BINARY_DIRECTORY}/sha256sum"
mkdir -p "$(dirname "${INSTRUMENTED_STAGE_SCRIPT}")"
sed \
  -e "s#readonly SCRATCH_ROOT=/raid/scratch#readonly SCRATCH_ROOT=${TEST_ROOT}/raid/scratch#g" \
  -e "s#/raid/scratch/sna/nemo-rl-image-stage#${TEST_SCRATCH_BASE}#g" \
  -e "s#/raid/scratch/nemo-rl-image-stage#${TEST_ROOT}/raid/scratch/nemo-rl-image-stage#g" \
  -e "s#/lustre#${TEST_ROOT}/lustre#g" \
  "${STAGE_SCRIPT}" >"${INSTRUMENTED_STAGE_SCRIPT}"
chmod +x "${INSTRUMENTED_STAGE_SCRIPT}"
git -C "${TEST_TOOLING_WORKTREE}" init --quiet
git -C "${TEST_TOOLING_WORKTREE}" config user.name 'Candidate Stage Test'
git -C "${TEST_TOOLING_WORKTREE}" config user.email candidate-stage-test@example.invalid
git -C "${TEST_TOOLING_WORKTREE}" add "${TOOLING_SCRIPT_RELATIVE_PATH}"
git -C "${TEST_TOOLING_WORKTREE}" commit --quiet -m 'test tooling snapshot'
TEST_TOOLING_COMMIT=$(git -C "${TEST_TOOLING_WORKTREE}" rev-parse HEAD)
readonly TEST_TOOLING_COMMIT
RESOLVED_TEST_TOOLING_WORKTREE=$(cd "${TEST_TOOLING_WORKTREE}" && pwd -P)
readonly RESOLVED_TEST_TOOLING_WORKTREE
RUN_STAGE_SCRIPT=${INSTRUMENTED_STAGE_SCRIPT}

test_required_and_exact_pins
test_canonical_lustre_containment
test_hostile_private_scratch_parents_are_rejected
test_preexisting_scratch_is_never_claimed_or_deleted
test_tooling_trust_failures_precede_import
test_replaced_scratch_is_preserved_on_cleanup
test_successful_import_cannot_hide_scratch_replacement
test_candidate_replacement_while_hashing_is_rejected
test_candidate_replacement_during_publish_is_rejected
test_symlink_candidate_is_rejected
test_container_directory_swap_after_import_is_rejected
test_container_swap_between_check_and_write_cannot_escape
test_scratch_swap_during_cleanup_preserves_unknown_bytes
test_success_uses_job_scratch_and_publishes_receipt
test_verified_candidate_is_reused_without_overwrite
test_immutable_image_collision_fails_closed
test_immutable_receipt_collision_fails_closed
test_import_failure_cleans_partial_state
test_receipt_publish_failure_rolls_back_new_candidate
test_receipt_publish_failure_never_deletes_replacement_bytes
test_orphan_image_is_recovered_without_overwrite
test_signal_after_receipt_link_preserves_complete_candidate
test_success_cleanup_preserves_replaced_partial
test_reuse_rejects_container_swap_during_validation
test_reuse_rejects_different_tooling_identity
test_hostile_metadata_values_cannot_publish_invalid_receipt
test_invalid_hash_tool_output_fails_before_import
test_exclusive_partial_allocation_failure_leaves_no_candidate

grep -Fx -- "#SBATCH --output=${EXPECTED_LOG_ROOT}/slurm-%j.out" \
  "${STAGE_SCRIPT}" >/dev/null || fail 'SBATCH output is not in the durable log root'
grep -Fx -- "#SBATCH --error=${EXPECTED_LOG_ROOT}/slurm-%j.err" \
  "${STAGE_SCRIPT}" >/dev/null || fail 'SBATCH error is not in the durable log root'

echo 'stage_enroot_image executable tests passed'
