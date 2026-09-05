#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

case $(basename "$0") in
sbatch)
  fake_root=$(cd "$(dirname "$0")/.." && pwd -P)
  /usr/bin/env | LC_ALL=C sort >"${fake_root}/capture.env"
  printf '%s\n' "$@" >"${fake_root}/capture.args"
  /bin/cat >"${fake_root}/capture.stdin"
  if [[ "${FAKE_SBATCH_FAIL:-0}" == 1 ]]; then
    exit 91
  fi
  echo 'Submitted batch job 12345'
  exit 0
  ;;
esac

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIRECTORY
readonly WRAPPER_SOURCE=${SCRIPT_DIRECTORY}/submit_oci_hsg_stage_enroot_image.sh
readonly BATCH_SOURCE=${SCRIPT_DIRECTORY}/stage_enroot_image.sbatch
readonly WRAPPER_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/submit_oci_hsg_stage_enroot_image.sh
readonly BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/stage_enroot_image.sbatch
readonly VALID_SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl:33940349909
readonly VALID_SOURCE_COMMIT=2682b7e49c8877bcf02681fa2861c752f3e447f4
readonly VALID_LOCK_SHA256=95f63521d28a2a4104ff372c5985fe63826ab27d6901b78bada1ab1a89a81bf7
readonly VALID_PYPROJECT_SHA256=827f5f82c37dcf99454e47982bb0c0b8aa82c48cb869223e2a475db0c90cf0f9

TEST_ROOT=$(cd "$(mktemp -d)" && pwd -P)
readonly TEST_ROOT
readonly TOOLING_ROOT=${TEST_ROOT}/tooling,INJECTED_VAR=1
readonly INSTRUMENTED_WRAPPER=${TOOLING_ROOT}/${WRAPPER_RELATIVE_PATH}
readonly INSTRUMENTED_BATCH=${TOOLING_ROOT}/${BATCH_RELATIVE_PATH}
readonly FAKE_ROOT=${TEST_ROOT}/fake
readonly FAKE_SBATCH=${FAKE_ROOT}/bin/sbatch
readonly FAKE_SLURM_CONF=${TEST_ROOT}/slurm.conf
readonly FAKE_PUBLISH_ROOT=${TEST_ROOT}/lustre
readonly FAKE_VALIDATION_ROOT=${FAKE_PUBLISH_ROOT}/semantic-precision-refit/container-validation
readonly FAKE_STAGE_ROOT=${FAKE_VALIDATION_ROOT}/nightly-stage
readonly FAKE_CONTAINER_DIR=${FAKE_STAGE_ROOT}/candidates
readonly FAKE_LOG_DIR=${FAKE_STAGE_ROOT}/logs
readonly FAKE_HOME=${TEST_ROOT}/home/sna
readonly CAPTURE_ENV=${FAKE_ROOT}/capture.env
readonly CAPTURE_ARGS=${FAKE_ROOT}/capture.args
readonly CAPTURE_STDIN=${FAKE_ROOT}/capture.stdin
readonly HOSTILE_BASH_ENV=${TEST_ROOT}/hostile-bash-env.sh
RUN_STATUS=0

cleanup() {
  rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_line() {
  local expected=$1
  local file=$2

  grep -Fx -- "${expected}" "${file}" >/dev/null ||
    fail "${file} does not contain exact line: ${expected}"
}

assert_absent() {
  local forbidden=$1
  local file=$2

  if grep -Fq -- "${forbidden}" "${file}"; then
    fail "${file} contains forbidden text: ${forbidden}"
  fi
}

assert_no_line() {
  local forbidden=$1
  local file=$2

  if grep -Fx -- "${forbidden}" "${file}" >/dev/null; then
    fail "${file} contains forbidden exact line: ${forbidden}"
  fi
}

clear_capture() {
  rm -f "${CAPTURE_ENV}" "${CAPTURE_ARGS}" "${CAPTURE_STDIN}"
}

run_wrapper() {
  local action=$1
  shift
  local -a wrapper_command=(/bin/bash "${INSTRUMENTED_WRAPPER}")

  if [[ -n "${action}" ]]; then
    wrapper_command+=("${action}")
  fi

  clear_capture
  set +e
  env \
    SOURCE_IMAGE="${VALID_SOURCE_IMAGE}" \
    SOURCE_COMMIT="${VALID_SOURCE_COMMIT}" \
    SOURCE_LOCK_SHA256="${VALID_LOCK_SHA256}" \
    SOURCE_PYPROJECT_SHA256="${VALID_PYPROJECT_SHA256}" \
    OUTPUT_PREFIX=nemo_rl_nightly \
    SBATCH_ACCOUNT=hostile-account \
    SBATCH_PARTITION=hostile-partition \
    SBATCH_GRES=gpu:999 \
    SBATCH_EXPORT=ALL \
    SLURM_CONF=/hostile/slurm.conf \
    BASH_ENV="${HOSTILE_BASH_ENV}" \
    ENV="${HOSTILE_BASH_ENV}" \
    ENROOT_CACHE_PATH=/hostile/enroot-cache \
    ENROOT_DATA_PATH=/hostile/enroot-data \
    ENROOT_RUNTIME_PATH=/hostile/enroot-runtime \
    ENROOT_CONFIG_PATH=/hostile/enroot-config \
    HF_TOKEN=HOSTILE_SECRET_MARKER \
    NGC_API_KEY=HOSTILE_SECRET_MARKER \
    "$@" "${wrapper_command[@]}" \
    >"${TEST_ROOT}/wrapper.stdout" 2>"${TEST_ROOT}/wrapper.stderr"
  RUN_STATUS=$?
  set -e
}

assert_clean_boundary() {
  local tooling_sha
  local expected_exports

  tooling_sha=$(git -C "${TOOLING_ROOT}" rev-parse HEAD)
  expected_exports='--export=HOME,PATH,SOURCE_IMAGE,SOURCE_COMMIT,SOURCE_LOCK_SHA256,SOURCE_PYPROJECT_SHA256,OUTPUT_PREFIX,CONTAINER_DIR,TOOLING_WORKTREE,TOOLING_COMMIT'

  assert_line '--nodes=1' "${CAPTURE_ARGS}"
  assert_line '--ntasks-per-node=1' "${CAPTURE_ARGS}"
  assert_line '--gres=gpu:1' "${CAPTURE_ARGS}"
  assert_line '--mem=0' "${CAPTURE_ARGS}"
  assert_line '--time=02:00:00' "${CAPTURE_ARGS}"
  assert_line '--job-name=stage_nemo_rl_candidate' "${CAPTURE_ARGS}"
  assert_line "--output=${FAKE_LOG_DIR}/slurm-%j.out" "${CAPTURE_ARGS}"
  assert_line "--error=${FAKE_LOG_DIR}/slurm-%j.err" "${CAPTURE_ARGS}"
  assert_line "--chdir=${TOOLING_ROOT}" "${CAPTURE_ARGS}"
  assert_line "${expected_exports}" "${CAPTURE_ARGS}"
  assert_absent '--export=ALL' "${CAPTURE_ARGS}"

  assert_line 'SLURM_CONF='"${FAKE_SLURM_CONF}" "${CAPTURE_ENV}"
  assert_line 'PATH=/usr/bin:/bin' "${CAPTURE_ENV}"
  assert_line "HOME=${FAKE_HOME}" "${CAPTURE_ENV}"
  assert_line "SOURCE_IMAGE=${VALID_SOURCE_IMAGE}" "${CAPTURE_ENV}"
  assert_line "SOURCE_COMMIT=${VALID_SOURCE_COMMIT}" "${CAPTURE_ENV}"
  assert_line "SOURCE_LOCK_SHA256=${VALID_LOCK_SHA256}" "${CAPTURE_ENV}"
  assert_line "SOURCE_PYPROJECT_SHA256=${VALID_PYPROJECT_SHA256}" \
    "${CAPTURE_ENV}"
  assert_line 'OUTPUT_PREFIX=nemo_rl_nightly' "${CAPTURE_ENV}"
  assert_line "CONTAINER_DIR=${FAKE_CONTAINER_DIR}" "${CAPTURE_ENV}"
  assert_line "TOOLING_WORKTREE=${TOOLING_ROOT}" "${CAPTURE_ENV}"
  assert_line "TOOLING_COMMIT=${tooling_sha}" "${CAPTURE_ENV}"
  assert_no_line 'INJECTED_VAR=1' "${CAPTURE_ENV}"
  assert_absent 'SBATCH_' "${CAPTURE_ENV}"
  assert_absent 'BASH_ENV=' "${CAPTURE_ENV}"
  assert_absent 'ENV=' "${CAPTURE_ENV}"
  assert_absent 'ENROOT_' "${CAPTURE_ENV}"
  assert_absent 'HF_TOKEN=' "${CAPTURE_ENV}"
  assert_absent 'NGC_API_KEY=' "${CAPTURE_ENV}"
  assert_absent 'HOSTILE_SECRET_MARKER' "${CAPTURE_ENV}"
  assert_absent 'hostile-account' "${CAPTURE_ARGS}"
  assert_absent 'hostile-partition' "${CAPTURE_ARGS}"
  assert_absent 'gpu:999' "${CAPTURE_ARGS}"

  git -C "${TOOLING_ROOT}" show "${tooling_sha}:${BATCH_RELATIVE_PATH}" |
    cmp -s - "${CAPTURE_STDIN}" ||
    fail 'sbatch stdin did not exactly match the committed batch blob'
}

test_test_only_is_default_and_sanitized() {
  local candidate_mode

  run_wrapper ''
  if ((RUN_STATUS != 0)); then
    sed -n '1,200p' "${TEST_ROOT}/wrapper.stderr" >&2
    fail "default test-only wrapper failed with status ${RUN_STATUS}"
  fi
  assert_line '--test-only' "${CAPTURE_ARGS}"
  assert_clean_boundary
  if candidate_mode=$(stat -Lc '%a' "${FAKE_CONTAINER_DIR}" 2>/dev/null); then
    :
  else
    candidate_mode=$(stat -f '%Lp' "${FAKE_CONTAINER_DIR}")
  fi
  if ((8#${candidate_mode} & 8#022)); then
    fail 'candidate publish directory is writable by peers'
  fi
}

test_submit_uses_same_boundary_without_test_only() {
  run_wrapper submit
  if ((RUN_STATUS != 0)); then
    fail "fake submit wrapper failed with status ${RUN_STATUS}"
  fi
  if grep -Fx -- '--test-only' "${CAPTURE_ARGS}" >/dev/null; then
    fail 'submit action retained --test-only'
  fi
  assert_clean_boundary
}

test_invalid_input_fails_before_sbatch() {
  run_wrapper test-only SOURCE_COMMIT=not-a-commit
  if ((RUN_STATUS == 0)); then
    fail 'invalid source commit was accepted'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'invalid input reached sbatch'

  run_wrapper invalid-action
  if ((RUN_STATUS == 0)); then
    fail 'invalid wrapper action was accepted'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'invalid action reached sbatch'

  run_wrapper test-only \
    SOURCE_IMAGE=nvcr.io/nvidian/nemo-rl@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  if ((RUN_STATUS == 0)); then
    fail 'Enroot-3.5-incompatible digest source reached submit boundary'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'digest source reached sbatch'
}

test_symlinked_stage_directory_fails_before_sbatch() {
  local outside=${TEST_ROOT}/outside-stage

  rmdir "${FAKE_CONTAINER_DIR}" "${FAKE_LOG_DIR}" "${FAKE_STAGE_ROOT}"
  mkdir "${outside}"
  ln -s "${outside}" "${FAKE_STAGE_ROOT}"
  run_wrapper test-only
  rm "${FAKE_STAGE_ROOT}"
  if ((RUN_STATUS == 0)); then
    fail 'symlinked stage directory was accepted'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'symlinked stage directory reached sbatch'
  if find "${outside}" -mindepth 1 -print -quit | grep -q .; then
    fail 'symlinked stage directory caused writes outside the validation root'
  fi
  mkdir "${FAKE_STAGE_ROOT}"
}

test_dirty_or_unpushed_tooling_fails_before_sbatch() {
  local dirty_file=${TOOLING_ROOT}/dirty-file
  local head

  printf 'dirty\n' >"${dirty_file}"
  run_wrapper test-only
  rm -f "${dirty_file}"
  if ((RUN_STATUS == 0)); then
    fail 'dirty tooling checkout was accepted'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'dirty tooling checkout reached sbatch'

  head=$(git -C "${TOOLING_ROOT}" rev-parse HEAD)
  git -C "${TOOLING_ROOT}" update-ref refs/remotes/origin/main "${head}^"
  run_wrapper test-only
  git -C "${TOOLING_ROOT}" update-ref refs/remotes/origin/main "${head}"
  if ((RUN_STATUS == 0)); then
    fail 'tooling HEAD different from upstream was accepted'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'unpushed tooling checkout reached sbatch'
}

test_hidden_executed_wrapper_mutation_fails_before_sbatch() {
  local wrapper_snapshot=${TEST_ROOT}/wrapper.snapshot
  local hidden_status

  cp -p "${INSTRUMENTED_WRAPPER}" "${wrapper_snapshot}"
  git -C "${TOOLING_ROOT}" update-index --assume-unchanged \
    "${WRAPPER_RELATIVE_PATH}"
  printf '\n# hidden mutation\n' >>"${INSTRUMENTED_WRAPPER}"
  run_wrapper test-only
  hidden_status=${RUN_STATUS}
  cp -p "${wrapper_snapshot}" "${INSTRUMENTED_WRAPPER}"
  git -C "${TOOLING_ROOT}" update-index --no-assume-unchanged \
    "${WRAPPER_RELATIVE_PATH}"
  if ((hidden_status == 0)); then
    fail 'executed wrapper bytes different from the commit were accepted'
  fi
  [[ ! -e "${CAPTURE_ARGS}" ]] || fail 'mutated executed wrapper reached sbatch'
}

[[ -x "${WRAPPER_SOURCE}" ]] || fail "missing executable submit wrapper: ${WRAPPER_SOURCE}"
[[ -x "${BATCH_SOURCE}" ]] || fail "missing executable batch script: ${BATCH_SOURCE}"
mkdir -p "$(dirname "${INSTRUMENTED_WRAPPER}")" "${FAKE_ROOT}/bin" \
  "${FAKE_VALIDATION_ROOT}" "${FAKE_HOME}"
touch "${FAKE_SLURM_CONF}"
ln -s "${SCRIPT_DIRECTORY}/test_submit_oci_hsg_stage_enroot_image_static.sh" \
  "${FAKE_SBATCH}"
{
  echo 'export SBATCH_ACCOUNT=hostile-from-bash-env'
  echo 'export ENROOT_CACHE_PATH=/hostile/from-bash-env'
  echo 'export HF_TOKEN=HOSTILE_SECRET_MARKER'
} >"${HOSTILE_BASH_ENV}"

sed \
  -e "s#/cm/local/apps/slurm/current/bin/sbatch#${FAKE_SBATCH}#g" \
  -e "s#/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf#${FAKE_SLURM_CONF}#g" \
  -e "s#/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/container-validation#${FAKE_VALIDATION_ROOT}#g" \
  -e "s#readonly PUBLISH_ROOT=/lustre#readonly PUBLISH_ROOT=${FAKE_PUBLISH_ROOT}#g" \
  -e "s#readonly HOME_DIRECTORY=/home/sna#readonly HOME_DIRECTORY=${FAKE_HOME}#g" \
  "${WRAPPER_SOURCE}" >"${INSTRUMENTED_WRAPPER}"
cp "${BATCH_SOURCE}" "${INSTRUMENTED_BATCH}"
chmod +x "${INSTRUMENTED_WRAPPER}" "${INSTRUMENTED_BATCH}"

git -C "${TOOLING_ROOT}" init --quiet
git -C "${TOOLING_ROOT}" config user.name 'Candidate Stage Submit Test'
git -C "${TOOLING_ROOT}" config user.email candidate-stage-submit@example.invalid
git -C "${TOOLING_ROOT}" add "${WRAPPER_RELATIVE_PATH}" "${BATCH_RELATIVE_PATH}"
git -C "${TOOLING_ROOT}" commit --quiet -m 'base fixture'
git -C "${TOOLING_ROOT}" commit --quiet --allow-empty -m 'submitted fixture'
git -C "${TOOLING_ROOT}" update-ref refs/remotes/origin/main HEAD
git -C "${TOOLING_ROOT}" config remote.origin.url \
  "${TEST_ROOT}/unused-origin.git"
git -C "${TOOLING_ROOT}" config remote.origin.fetch \
  '+refs/heads/*:refs/remotes/origin/*'
git -C "${TOOLING_ROOT}" config branch.main.remote origin
git -C "${TOOLING_ROOT}" config branch.main.merge refs/heads/main

test_test_only_is_default_and_sanitized
test_submit_uses_same_boundary_without_test_only
test_invalid_input_fails_before_sbatch
test_symlinked_stage_directory_fails_before_sbatch
test_dirty_or_unpushed_tooling_fails_before_sbatch
test_hidden_executed_wrapper_mutation_fails_before_sbatch

echo 'stage_enroot_image submit boundary tests passed'
