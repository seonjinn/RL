#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "$script_dir" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
readonly SCRIPTS_DIRECTORY=$SCRIPT_ROOT/experiments/pr3652_validation_container/scripts
readonly DOWNLOAD_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_download_validated_nightly.sbatch
readonly SMOKE_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_smoke_validated_nightly.sbatch
readonly SMOKE_BODY=$SCRIPTS_DIRECTORY/oci_hsg_smoke_validated_nightly.sh
readonly DEFAULT_PTYCHE_UPLOAD_BATCH=$SCRIPTS_DIRECTORY/ptyche_upload_validated_nightly.sbatch
readonly dollar='$'

ptyche_upload_batch=$DEFAULT_PTYCHE_UPLOAD_BATCH
ptyche_runtime_only=false
if (( $# > 0 )); then
  if (( $# != 2 )) || [[ $1 != --ptyche-upload-runtime-only ]]; then
    echo "Usage: $0 [--ptyche-upload-runtime-only BATCH_SCRIPT]" >&2
    exit 2
  fi
  ptyche_runtime_only=true
  ptyche_upload_batch=$2
fi
readonly PTYCHE_UPLOAD_BATCH=$ptyche_upload_batch
readonly PTYCHE_RUNTIME_ONLY=$ptyche_runtime_only

fail_if_present() {
  local pattern=$1
  local path=$2

  if grep -Fq -- "$pattern" "$path"; then
    echo "Unexpected pattern in $path: $pattern" >&2
    exit 1
  fi
}

require_pattern() {
  local pattern=$1
  local path=$2

  if ! grep -Fq -- "$pattern" "$path"; then
    echo "Missing required pattern in $path: $pattern" >&2
    exit 1
  fi
}

line_number() {
  local pattern=$1
  local occurrence=$2
  local path=$3

  grep -n -- "$pattern" "$path" | sed -n "$occurrence"p | cut -d: -f1
}

assert_runtime_diagnostic() {
  local stderr_path=$1
  local expected_step=$2
  local expected_exit_status=$3
  local expected_job_id=$4
  local expected_node=$5
  local diagnostic_pattern

  diagnostic_pattern="^ptyche upload failed: step=${expected_step} exit=${expected_exit_status} line=[1-9][0-9]* job=${expected_job_id} node=${expected_node}$"
  if ! grep -Eq -- "$diagnostic_pattern" "$stderr_path"; then
    echo "Missing expected Ptyche failure diagnostic in $stderr_path" >&2
    sed -n '1,120p' "$stderr_path" >&2
    exit 1
  fi
}

assert_cleanup_diagnostic() {
  local stderr_path=$1
  local expected_step=$2
  local expected_exit_status=$3
  local expected_job_id=$4
  local expected_node=$5
  local diagnostic_pattern

  diagnostic_pattern="^ptyche upload cleanup failed: step=${expected_step} exit=${expected_exit_status} line=[1-9][0-9]* job=${expected_job_id} node=${expected_node}$"
  if ! grep -Eq -- "$diagnostic_pattern" "$stderr_path"; then
    echo "Missing expected Ptyche cleanup diagnostic in $stderr_path" >&2
    sed -n '1,120p' "$stderr_path" >&2
    exit 1
  fi
}

assert_no_sensitive_diagnostic_material() {
  local stderr_path=$1
  local forbidden_literal

  for forbidden_literal in \
    'AKIA_RUNTIME_SECRET_MUST_NOT_LEAK' \
    'runtime-secret-key-must-not-leak' \
    'runtime-session-token-must-not-leak' \
    'runtime-rclone-password-must-not-leak' \
    'https://storage.invalid/object?X-Amz-Credential=must-not-leak' \
    'pbss-team-nemo-ci-s3:'; do
    fail_if_present "$forbidden_literal" "$stderr_path"
  done

  if grep -Eiq -- '(https?://|X-Amz-|AWS4-HMAC-SHA256|(^|[?&[:space:]])(credential|signature|token)=)' "$stderr_path"; then
    echo "Sensitive or signed-URL-like material appeared in $stderr_path" >&2
    sed -n '1,120p' "$stderr_path" >&2
    exit 1
  fi
}

run_expected_ptyche_failure() {
  local batch_script=$1
  local stub_path=$2
  local expected_step=$3
  local expected_exit_status=$4
  local job_id=$5
  local node=$6
  local case_directory=$7
  local exit_status
  local stderr_path=$case_directory/stderr
  local stdout_path=$case_directory/stdout

  exit_status=0
  if env -i \
    PATH="$stub_path" \
    SLURM_JOB_ID="$job_id" \
    SLURMD_NODENAME="$node" \
    AWS_ACCESS_KEY_ID=AKIA_RUNTIME_SECRET_MUST_NOT_LEAK \
    AWS_SECRET_ACCESS_KEY=runtime-secret-key-must-not-leak \
    AWS_SESSION_TOKEN=runtime-session-token-must-not-leak \
    RCLONE_CONFIG_PASS=runtime-rclone-password-must-not-leak \
    PTYCHE_TEST_SIGNED_URL='https://storage.invalid/object?X-Amz-Credential=must-not-leak' \
    PTYCHE_UPLOAD_TEST_MISSING_SOURCE="$case_directory/guaranteed-missing-source.sqsh" \
    "$batch_script" >"$stdout_path" 2>"$stderr_path"; then
    echo "Expected $batch_script to fail at $expected_step" >&2
    exit 1
  else
    exit_status=$?
  fi

  if [[ $exit_status != "$expected_exit_status" ]]; then
    echo "Expected exit $expected_exit_status from $batch_script, received $exit_status" >&2
    sed -n '1,120p' "$stderr_path" >&2
    exit 1
  fi
  if [[ -s $stdout_path ]]; then
    echo "Unexpected stdout from $batch_script" >&2
    sed -n '1,120p' "$stdout_path" >&2
    exit 1
  fi
  assert_runtime_diagnostic \
    "$stderr_path" \
    "$expected_step" \
    "$expected_exit_status" \
    "$job_id" \
    "$node"
  assert_no_sensitive_diagnostic_material "$stderr_path"
}

test_ptyche_upload_failure_diagnostics() {
  local batch_script=$1
  local runtime_directory=$TEST_DIRECTORY/ptyche-runtime
  local missing_rclone_directory=$runtime_directory/missing-rclone
  local missing_source_directory=$runtime_directory/missing-source
  local missing_rclone_script=$missing_rclone_directory/ptyche-upload.sbatch
  local missing_source_script=$missing_source_directory/ptyche-upload.sbatch
  local missing_source_assignment="readonly SOURCE=\${PTYCHE_UPLOAD_TEST_MISSING_SOURCE:?}"
  local original_source_assignment_count
  local instrumented_total_source_assignment_count
  local instrumented_source_assignment_count
  local command_name

  mkdir -p \
    "$missing_rclone_directory/empty-bin" \
    "$missing_source_directory/bin"

  cp -- "$batch_script" "$missing_rclone_script"
  if ! cmp -s -- "$batch_script" "$missing_rclone_script"; then
    echo 'Missing-rclone executable fixture differs from the batch script' >&2
    exit 1
  fi
  chmod 755 "$missing_rclone_script"

  original_source_assignment_count=$(grep -Ec '^readonly SOURCE=' "$batch_script")
  if [[ $original_source_assignment_count != 1 ]]; then
    echo "Expected exactly one SOURCE assignment in $batch_script" >&2
    exit 1
  fi
  awk -v replacement="$missing_source_assignment" '
    /^readonly SOURCE=/ {
      print replacement
      replaced += 1
      next
    }
    { print }
    END {
      if (replaced != 1) {
        exit 64
      }
    }
  ' "$batch_script" >"$missing_source_script"
  instrumented_total_source_assignment_count=$(grep -Ec '^readonly SOURCE=' "$missing_source_script")
  instrumented_source_assignment_count=$(grep -Fxc -- "$missing_source_assignment" "$missing_source_script")
  if [[ $instrumented_total_source_assignment_count != 1 || $instrumented_source_assignment_count != 1 ]]; then
    echo 'Instrumented batch script does not contain exactly one expected SOURCE assignment' >&2
    exit 1
  fi
  if ! cmp -s \
    <(grep -Ev '^readonly SOURCE=' "$batch_script") \
    <(grep -Ev '^readonly SOURCE=' "$missing_source_script"); then
    echo 'Instrumented batch script changed content outside the SOURCE assignment' >&2
    exit 1
  fi
  chmod 755 "$missing_source_script"

  for command_name in rclone sha256sum stat awk srun; do
    cat >"$missing_source_directory/bin/$command_name" <<'EOF'
#!/bin/bash
printf 'forbidden dependency stub executed: %s\n' "${0##*/}" >&2
exit 97
EOF
    chmod 755 "$missing_source_directory/bin/$command_name"
  done

  run_expected_ptyche_failure \
    "$missing_rclone_script" \
    "$missing_rclone_directory/empty-bin" \
    preflight-required-commands \
    127 \
    424241 \
    ptyche-runtime-missing-rclone \
    "$missing_rclone_directory"
  require_pattern \
    'ptyche upload preflight failed: required command is unavailable: rclone' \
    "$missing_rclone_directory/stderr"

  run_expected_ptyche_failure \
    "$missing_source_script" \
    "$missing_source_directory/bin" \
    preflight-source-files \
    1 \
    424242 \
    ptyche-runtime-missing-source \
    "$missing_source_directory"
  require_pattern \
    "ptyche upload preflight failed: required regular file is missing: $missing_source_directory/guaranteed-missing-source.sqsh" \
    "$missing_source_directory/stderr"
  fail_if_present 'forbidden dependency stub executed:' "$missing_source_directory/stderr"
}

create_ptyche_cleanup_probe_script() {
  local batch_script=$1
  local probe_script=$2
  local source_assignment="readonly SOURCE=\${PTYCHE_UPLOAD_TEST_SOURCE:?}"
  local source_hash_assignment="readonly SOURCE_HASH_FILE=\${PTYCHE_UPLOAD_TEST_SOURCE_HASH_FILE:?}"
  local scratch_assignment="readonly SCRATCH_DIRECTORY=\${PTYCHE_UPLOAD_TEST_SCRATCH_DIRECTORY:?}"
  local assignment_pattern='^(readonly SOURCE=|readonly SOURCE_HASH_FILE=|readonly SCRATCH_DIRECTORY=)'
  local expected_assignment

  for expected_assignment in \
    'readonly SOURCE=' \
    'readonly SOURCE_HASH_FILE=' \
    'readonly SCRATCH_DIRECTORY='; do
    if [[ $(grep -Fc -- "$expected_assignment" "$batch_script") != 1 ]]; then
      echo "Expected exactly one $expected_assignment assignment in $batch_script" >&2
      exit 1
    fi
  done

  awk \
    -v source_assignment="$source_assignment" \
    -v source_hash_assignment="$source_hash_assignment" \
    -v scratch_assignment="$scratch_assignment" '
      /^readonly SOURCE=/ {
        print source_assignment
        source_replacements += 1
        next
      }
      /^readonly SOURCE_HASH_FILE=/ {
        print source_hash_assignment
        source_hash_replacements += 1
        next
      }
      /^readonly SCRATCH_DIRECTORY=/ {
        print scratch_assignment
        scratch_replacements += 1
        next
      }
      { print }
      END {
        if (source_replacements != 1 || source_hash_replacements != 1 || scratch_replacements != 1) {
          exit 64
        }
      }
    ' "$batch_script" >"$probe_script"

  for expected_assignment in \
    "$source_assignment" \
    "$source_hash_assignment" \
    "$scratch_assignment"; do
    if [[ $(grep -Fxc -- "$expected_assignment" "$probe_script") != 1 ]]; then
      echo "Cleanup probe does not contain exactly one $expected_assignment assignment" >&2
      exit 1
    fi
  done
  if [[ $(grep -Ec "$assignment_pattern" "$probe_script") != 3 ]]; then
    echo 'Cleanup probe contains an unexpected instrumented assignment' >&2
    exit 1
  fi
  if ! cmp -s \
    <(grep -Ev "$assignment_pattern" "$batch_script") \
    <(grep -Ev "$assignment_pattern" "$probe_script"); then
    echo 'Cleanup probe changed content outside its three path assignments' >&2
    exit 1
  fi
  chmod 755 "$probe_script"
}

create_ptyche_cleanup_probe_stubs() {
  local stub_directory=$1
  local expected_sha256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e

  mkdir -p "$stub_directory"

  cat >"$stub_directory/rclone" <<'EOF'
#!/bin/bash
set -euo pipefail
: "${PTYCHE_TEST_RCLONE_LOG:?}"
printf '%s\n' "$*" >>"$PTYCHE_TEST_RCLONE_LOG"
case ${1:-} in
  purge | lsjson)
    exit 0
    ;;
  *)
    exit 96
    ;;
esac
EOF
  cat >"$stub_directory/sha256sum" <<EOF
#!/bin/bash
set -euo pipefail
if [[ \${PTYCHE_TEST_TRIGGER_ORIGINAL_FAILURE:-false} = true ]]; then
  exit 42
fi
printf '%s  %s\\n' '$expected_sha256' "\${1:-fixture}"
EOF
  cat >"$stub_directory/stat" <<'EOF'
#!/bin/bash
set -euo pipefail
printf '17\n'
EOF
  cat >"$stub_directory/awk" <<'EOF'
#!/bin/bash
set -euo pipefail
first_field=
if (( $# > 1 )); then
  input_path=${!#}
  IFS=' ' read -r first_field _ <"$input_path" || :
else
  IFS=' ' read -r first_field _ || :
fi
printf '%s\n' "$first_field"
EOF
  cat >"$stub_directory/srun" <<'EOF'
#!/bin/bash
set -euo pipefail
: "${PTYCHE_TEST_SRUN_LOG:?}"
printf '%s\n' "$*" >>"$PTYCHE_TEST_SRUN_LOG"
EOF
  cat >"$stub_directory/mkdir" <<'EOF'
#!/bin/bash
set -euo pipefail
exit 0
EOF
  cat >"$stub_directory/rm" <<'EOF'
#!/bin/bash
set -euo pipefail
: "${PTYCHE_TEST_RM_LOG:?}"
printf '%s\n' "$@" >>"$PTYCHE_TEST_RM_LOG"
exit 73
EOF
  chmod 755 "$stub_directory"/*
}

run_ptyche_cleanup_probe() {
  local probe_script=$1
  local case_directory=$2
  local trigger_original_failure=$3
  local expected_exit_status=$4
  local job_id=$5
  local node=$6
  local source_path=$case_directory/source.sqsh
  local source_hash_path=$case_directory/source.sqsh.sha256
  local scratch_path=$case_directory/job-owned-scratch
  local stub_directory=$case_directory/bin
  local stderr_path=$case_directory/stderr
  local stdout_path=$case_directory/stdout
  local rm_log=$case_directory/rm.log
  local expected_rm_log=$case_directory/expected-rm.log
  local rclone_log=$case_directory/rclone.log
  local srun_log=$case_directory/srun.log
  local exit_status

  printf 'source fixture\n' >"$source_path"
  printf '%s  %s\n' \
    c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e \
    "$source_path" >"$source_hash_path"
  create_ptyche_cleanup_probe_stubs "$stub_directory"

  exit_status=0
  if env -i \
    PATH="$stub_directory" \
    SLURM_JOB_ID="$job_id" \
    SLURMD_NODENAME="$node" \
    AWS_ACCESS_KEY_ID=AKIA_RUNTIME_SECRET_MUST_NOT_LEAK \
    AWS_SECRET_ACCESS_KEY=runtime-secret-key-must-not-leak \
    AWS_SESSION_TOKEN=runtime-session-token-must-not-leak \
    RCLONE_CONFIG_PASS=runtime-rclone-password-must-not-leak \
    PTYCHE_TEST_SIGNED_URL='https://storage.invalid/object?X-Amz-Credential=must-not-leak' \
    PTYCHE_UPLOAD_TEST_SOURCE="$source_path" \
    PTYCHE_UPLOAD_TEST_SOURCE_HASH_FILE="$source_hash_path" \
    PTYCHE_UPLOAD_TEST_SCRATCH_DIRECTORY="$scratch_path" \
    PTYCHE_TEST_TRIGGER_ORIGINAL_FAILURE="$trigger_original_failure" \
    PTYCHE_TEST_RCLONE_LOG="$rclone_log" \
    PTYCHE_TEST_SRUN_LOG="$srun_log" \
    PTYCHE_TEST_RM_LOG="$rm_log" \
    "$probe_script" >"$stdout_path" 2>"$stderr_path"; then
    echo 'Expected Ptyche cleanup probe to fail' >&2
    exit 1
  else
    exit_status=$?
  fi

  if [[ $exit_status != "$expected_exit_status" ]]; then
    echo "Expected cleanup probe exit $expected_exit_status, received $exit_status" >&2
    sed -n '1,120p' "$stderr_path" >&2
    exit 1
  fi
  if [[ -s $stdout_path ]]; then
    echo 'Unexpected stdout from Ptyche cleanup probe' >&2
    sed -n '1,120p' "$stdout_path" >&2
    exit 1
  fi
  assert_cleanup_diagnostic \
    "$stderr_path" \
    cleanup-job-scratch \
    73 \
    "$job_id" \
    "$node"
  assert_no_sensitive_diagnostic_material "$stderr_path"

  printf '%s\n' -rf -- "$scratch_path" >"$expected_rm_log"
  if ! cmp -s "$expected_rm_log" "$rm_log"; then
    echo 'Cleanup attempted a deletion outside the exact job-owned scratch path' >&2
    exit 1
  fi
  require_pattern 'purge ' "$rclone_log"
  fail_if_present 'copy ' "$rclone_log"
}

test_ptyche_cleanup_failure_diagnostics() {
  local batch_script=$1
  local cleanup_directory=$TEST_DIRECTORY/ptyche-runtime/cleanup-failure
  local probe_script=$cleanup_directory/ptyche-upload.sbatch
  local original_failure_directory=$cleanup_directory/original-failure
  local success_cleanup_failure_directory=$cleanup_directory/success-cleanup-failure

  mkdir -p \
    "$cleanup_directory" \
    "$original_failure_directory" \
    "$success_cleanup_failure_directory"
  create_ptyche_cleanup_probe_script "$batch_script" "$probe_script"

  run_ptyche_cleanup_probe \
    "$probe_script" \
    "$original_failure_directory" \
    true \
    42 \
    424243 \
    ptyche-runtime-original-failure
  assert_runtime_diagnostic \
    "$original_failure_directory/stderr" \
    hash-source-before-upload \
    42 \
    424243 \
    ptyche-runtime-original-failure

  run_ptyche_cleanup_probe \
    "$probe_script" \
    "$success_cleanup_failure_directory" \
    false \
    73 \
    424244 \
    ptyche-runtime-success-cleanup-failure
  fail_if_present 'ptyche upload failed:' "$success_cleanup_failure_directory/stderr"
}

test_directory=$(mktemp -d)
readonly TEST_DIRECTORY=$test_directory
readonly STUB_DIRECTORY=$TEST_DIRECTORY/bin
REAL_GIT=$(command -v git)
readonly REAL_GIT
export REAL_GIT
mkdir -p "$STUB_DIRECTORY"
trap 'rm -rf -- "$TEST_DIRECTORY"' EXIT

test_ptyche_upload_failure_diagnostics "$PTYCHE_UPLOAD_BATCH"
test_ptyche_cleanup_failure_diagnostics "$PTYCHE_UPLOAD_BATCH"

if [[ $PTYCHE_RUNTIME_ONLY = true ]]; then
  printf 'Ptyche upload runtime failure checks passed\n'
  exit 0
fi

for batch_script in "$DOWNLOAD_BATCH" "$SMOKE_BATCH"; do
  fail_if_present 'BASH_SOURCE' "$batch_script"
  fail_if_present "test -z \"\$(git " "$batch_script"
  grep -Fq 'SCRIPT_ROOT:?Set SCRIPT_ROOT' "$batch_script"
  grep -Fq 'EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA' "$batch_script"
  grep -Fq 'validate_tooling_root' "$batch_script"
  require_pattern "git -C \"$dollar{SCRIPT_ROOT}\" hash-object" "$batch_script"
done

grep -Fq 'VALIDATOR_SNAPSHOT' "$DOWNLOAD_BATCH"
grep -Fq 'snapshot_validator' "$DOWNLOAD_BATCH"
grep -Fq 'VALIDATOR_SNAPSHOT' "$SMOKE_BATCH"
grep -Fq 'SMOKE_BODY_SNAPSHOT' "$SMOKE_BATCH"
grep -Fq 'snapshot_tooling_file' "$SMOKE_BATCH"
grep -Fq '#SBATCH --output=/lustre/' "$SMOKE_BATCH"
grep -Fq '#SBATCH --error=/lustre/' "$SMOKE_BATCH"
grep -Fq 'readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh' "$SMOKE_BATCH"
fail_if_present "CONTAINER=$dollar{CONTAINER:-" "$SMOKE_BATCH"
grep -Fq 'readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python' "$SMOKE_BODY"
fail_if_present "MAIN_PYTHON=$dollar{MAIN_PYTHON:-" "$SMOKE_BODY"
fail_if_present "test -z \"\$(git " "$SMOKE_BODY"
grep -Fq "PYTHONPYCACHEPREFIX=$dollar{SCRATCH_DIRECTORY}/pycache" "$SMOKE_BATCH"
grep -Fq "XDG_CACHE_HOME=$dollar{SCRATCH_DIRECTORY}/xdg-cache" "$SMOKE_BATCH"
grep -Fq "UV_CACHE_DIR=$dollar{SCRATCH_DIRECTORY}/uv-cache" "$SMOKE_BATCH"
grep -Fq "TORCHINDUCTOR_CACHE_DIR=$dollar{SCRATCH_DIRECTORY}/torchinductor-cache" "$SMOKE_BATCH"
grep -Fq "TRITON_CACHE_DIR=$dollar{SCRATCH_DIRECTORY}/triton-cache" "$SMOKE_BATCH"

pytest_line=$(line_number 'tests/unit/precision_policy' 1 "$SMOKE_BODY")
head_before_line=$(line_number 'rev-parse HEAD' 1 "$SMOKE_BODY")
head_after_line=$(line_number 'rev-parse HEAD' 2 "$SMOKE_BODY")
clean_before_line=$(line_number "semantic_worktree_status=\$(git" 1 "$SMOKE_BODY")
clean_after_line=$(line_number "semantic_worktree_status=\$(git" 2 "$SMOKE_BODY")
test -n "$pytest_line"
test -n "$head_before_line"
test -n "$head_after_line"
test -n "$clean_before_line"
test -n "$clean_after_line"
test "$head_before_line" -lt "$pytest_line"
test "$clean_before_line" -lt "$pytest_line"
test "$pytest_line" -lt "$head_after_line"
test "$pytest_line" -lt "$clean_after_line"

download_root_validation_line=$(line_number '^validate_tooling_root$' 1 "$DOWNLOAD_BATCH")
download_snapshot_line=$(line_number '^snapshot_validator$' 1 "$DOWNLOAD_BATCH")
download_rclone_line=$(line_number 'rclone copyto' 1 "$DOWNLOAD_BATCH")
download_validator_line=$(line_number "VALIDATOR_SNAPSHOT}\" \"$dollar{DESTINATION}" 1 "$DOWNLOAD_BATCH")
download_success_scratch_cleanup_line=$(line_number "rm -rf -- \"$dollar{JOB_SCRATCH_DIRECTORY}\"" 2 "$DOWNLOAD_BATCH")
download_success_trap_disable_line=$(line_number '^trap - EXIT$' 1 "$DOWNLOAD_BATCH")
test "$download_root_validation_line" -lt "$download_snapshot_line"
test "$download_snapshot_line" -lt "$download_rclone_line"
test "$download_validator_line" -lt "$download_success_scratch_cleanup_line"
test "$download_success_scratch_cleanup_line" -lt "$download_success_trap_disable_line"
smoke_root_validation_line=$(line_number '^validate_tooling_root$' 1 "$SMOKE_BATCH")
smoke_snapshot_line=$(line_number "snapshot_tooling_file \"$dollar{VALIDATOR_RELATIVE_PATH}\"" 1 "$SMOKE_BATCH")
smoke_validator_line=$(line_number "VALIDATOR_SNAPSHOT}\" \"$dollar{CONTAINER}" 1 "$SMOKE_BATCH")
smoke_srun_line=$(line_number '/cm/local/apps/slurm/current/bin/srun' 1 "$SMOKE_BATCH")
test "$smoke_root_validation_line" -lt "$smoke_snapshot_line"
test "$smoke_snapshot_line" -lt "$smoke_validator_line"
test "$smoke_validator_line" -lt "$smoke_srun_line"

# The Ptyche job can fail before rclone is launched.  Every prerequisite and
# step boundary must therefore produce a non-secret diagnostic rather than
# relying on silent command/test status under `set -e`.
require_pattern 'set -Eeuo pipefail' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'trap report_error ERR' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command rclone' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command sha256sum' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command stat' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command awk' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command srun' "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_regular_file \"$dollar{SOURCE}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_regular_file \"$dollar{SOURCE_HASH_FILE}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_absent_path \"$dollar{SCRATCH_DIRECTORY}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_equal \"$dollar{source_sha256}\" \"$dollar{EXPECTED_SHA256}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_equal \"$dollar{sidecar_sha256}\" \"$dollar{source_sha256}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_equal \"$dollar{source_sha256_after_upload}\" \"$dollar{EXPECTED_SHA256}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern 'ptyche upload failed:' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'Final directory promotion returned nonzero; verifying immutable final bytes' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=upload-job-temporary-directory' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=download-and-hash-job-temporary-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=promote-job-temporary-directory' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=download-and-hash-final-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=inspect-final-remote-object' "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'command -v rclone >/dev/null' "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'command -v sha256sum >/dev/null' "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'command -v stat >/dev/null' "$PTYCHE_UPLOAD_BATCH"
fail_if_present "test -f \"$dollar{SOURCE}\"" "$PTYCHE_UPLOAD_BATCH"
fail_if_present "test -f \"$dollar{SOURCE_HASH_FILE}\"" "$PTYCHE_UPLOAD_BATCH"
fail_if_present "test ! -e \"$dollar{SCRATCH_DIRECTORY}\"" "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'test ' "$PTYCHE_UPLOAD_BATCH"
fail_if_present '|| true' "$PTYCHE_UPLOAD_BATCH"
fail_if_present '|| :' "$PTYCHE_UPLOAD_BATCH"

cat >"$STUB_DIRECTORY/git" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ " $* " == *" status --porcelain "* ]]; then
  exit 0
fi
exec "$REAL_GIT" "$@"
EOF
cat >"$STUB_DIRECTORY/mkdir" <<'EOF'
#!/bin/bash
exit 0
EOF
cat >"$STUB_DIRECTORY/sbatch" <<'EOF'
#!/bin/bash
set -euo pipefail

: "$SBATCH_CAPTURE"
: "$SBATCH_EXPECTED_BATCH_RELATIVE_PATH"
seen_chdir=0
seen_export=0
seen_test_only=0
script_root=
expected_sha=

for argument in "$@"; do
  case "$argument" in
    --test-only)
      if (( seen_test_only )); then
        echo 'Duplicate --test-only' >&2
        exit 2
      fi
      seen_test_only=1
      ;;
    --chdir=*)
      if (( seen_chdir )); then
        echo 'Duplicate --chdir' >&2
        exit 2
      fi
      script_root=${argument#--chdir=}
      seen_chdir=1
      ;;
    --export=ALL,SCRIPT_ROOT=*,EXPECTED_TOOLING_SHA=*)
      if (( seen_export )); then
        echo 'Duplicate --export' >&2
        exit 2
      fi
      export_payload=${argument#--export=ALL,SCRIPT_ROOT=}
      export_root=${export_payload%%,EXPECTED_TOOLING_SHA=*}
      expected_sha=${export_payload#*,EXPECTED_TOOLING_SHA=}
      if [[ "$export_root" != "$script_root" ]]; then
        echo 'Mismatched --chdir and SCRIPT_ROOT export' >&2
        exit 2
      fi
      seen_export=1
      ;;
    *)
      echo "Unexpected sbatch argument: $argument" >&2
      exit 2
      ;;
  esac
done

if (( ! seen_chdir || ! seen_export )); then
  echo 'Missing required --chdir or --export' >&2
  exit 2
fi

printf '%s\n' "$@" >"$SBATCH_CAPTURE.args"
cat >"$SBATCH_CAPTURE.script"
"$REAL_GIT" -C "$script_root" show "${expected_sha}:${SBATCH_EXPECTED_BATCH_RELATIVE_PATH}" >"$SBATCH_CAPTURE.expected"
if ! cmp -s "$SBATCH_CAPTURE.script" "$SBATCH_CAPTURE.expected"; then
  echo 'sbatch stdin differs from the expected immutable batch blob' >&2
  exit 1
fi
EOF
chmod 755 "$STUB_DIRECTORY/git" "$STUB_DIRECTORY/mkdir" "$STUB_DIRECTORY/sbatch"

assert_wrapper_call() {
  local capture_prefix=$1
  local expect_test_only=$2
  local expected_sha

  expected_sha=$(git -C "$SCRIPT_ROOT" rev-parse HEAD)
  grep -Fx -- "--chdir=$SCRIPT_ROOT" "$capture_prefix.args" >/dev/null
  grep -Fx -- "--export=ALL,SCRIPT_ROOT=$SCRIPT_ROOT,EXPECTED_TOOLING_SHA=$expected_sha" "$capture_prefix.args" >/dev/null
  test -s "$capture_prefix.script"
  cmp -s "$capture_prefix.script" "$capture_prefix.expected"
  if [[ "$expect_test_only" = true ]]; then
    grep -Fx -- '--test-only' "$capture_prefix.args" >/dev/null
  else
    fail_if_present '--test-only' "$capture_prefix.args"
  fi
}

for wrapper in "$SCRIPTS_DIRECTORY"/submit_oci_hsg_*_validated_nightly.sh; do
  grep -Fq "readonly ACTION=$dollar{1:-test-only}" "$wrapper"
  fail_if_present 'ACTION:-' "$wrapper"

  batch_relative_path=$(grep -F 'readonly BATCH_RELATIVE_PATH=' "$wrapper" | cut -d= -f2-)
  test -n "$batch_relative_path"

  no_argument_capture=$TEST_DIRECTORY/$(basename "$wrapper").no-argument
  SBATCH_CAPTURE=$no_argument_capture SBATCH_EXPECTED_BATCH_RELATIVE_PATH=$batch_relative_path ACTION=submit PATH="$STUB_DIRECTORY:$PATH" "$wrapper"
  assert_wrapper_call "$no_argument_capture" true

  inherited_action_capture=$TEST_DIRECTORY/$(basename "$wrapper").inherited-action
  SBATCH_CAPTURE=$inherited_action_capture SBATCH_EXPECTED_BATCH_RELATIVE_PATH=$batch_relative_path ACTION=submit PATH="$STUB_DIRECTORY:$PATH" "$wrapper"
  assert_wrapper_call "$inherited_action_capture" true

  explicit_test_only_capture=$TEST_DIRECTORY/$(basename "$wrapper").test-only
  SBATCH_CAPTURE=$explicit_test_only_capture SBATCH_EXPECTED_BATCH_RELATIVE_PATH=$batch_relative_path ACTION=submit PATH="$STUB_DIRECTORY:$PATH" "$wrapper" test-only
  assert_wrapper_call "$explicit_test_only_capture" true

  submit_capture=$TEST_DIRECTORY/$(basename "$wrapper").submit
  SBATCH_CAPTURE=$submit_capture SBATCH_EXPECTED_BATCH_RELATIVE_PATH=$batch_relative_path ACTION=test-only PATH="$STUB_DIRECTORY:$PATH" "$wrapper" submit
  assert_wrapper_call "$submit_capture" false
done

printf 'validation tooling static checks passed\n'
