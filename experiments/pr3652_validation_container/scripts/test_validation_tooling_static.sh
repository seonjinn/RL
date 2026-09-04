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
readonly PTYCHE_RCLONE_PROVISIONER=$SCRIPTS_DIRECTORY/provision_ptyche_rclone.sh
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

sha256_file() {
  local path=$1

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo 'Neither sha256sum nor shasum is available for the test harness' >&2
    return 127
  fi
}

create_sha256sum_compatibility_stub() {
  local stub_path=$1
  local real_sha256_tool

  if real_sha256_tool=$(command -v sha256sum); then
    cat >"$stub_path" <<EOF
#!/bin/bash
exec "$real_sha256_tool" "\$@"
EOF
  elif real_sha256_tool=$(command -v shasum); then
    cat >"$stub_path" <<EOF
#!/bin/bash
exec "$real_sha256_tool" -a 256 "\$@"
EOF
  else
    echo 'Neither sha256sum nor shasum is available for the test harness' >&2
    return 127
  fi
  chmod 755 "$stub_path"
}

create_rclone_fixture() {
  local fixture_path=$1
  local version_exit_status=$2

  cat >"$fixture_path" <<EOF
#!/bin/bash
set -euo pipefail
if [[ \${1:-} != version || \$# != 1 ]]; then
  printf 'forbidden rclone fixture invocation: %s\n' "\$*" >&2
  exit 98
fi
if [[ -n \${PTYCHE_RCLONE_FIXTURE_LOG:-} ]]; then
  printf '%s\n' "\$*" >>"\$PTYCHE_RCLONE_FIXTURE_LOG"
fi
exit $version_exit_status
EOF
  chmod 755 "$fixture_path"
}

create_ptyche_upload_probe_script() {
  local batch_script=$1
  local probe_script=$2
  local rclone_source_assignment="readonly RCLONE_SOURCE=${dollar}{PTYCHE_UPLOAD_TEST_RCLONE_SOURCE:?}"
  local rclone_sha_assignment="readonly EXPECTED_RCLONE_SHA256=${dollar}{PTYCHE_UPLOAD_TEST_EXPECTED_RCLONE_SHA256:?}"
  local source_assignment="readonly SOURCE=${dollar}{PTYCHE_UPLOAD_TEST_SOURCE:?}"
  local source_hash_assignment="readonly SOURCE_HASH_FILE=${dollar}{PTYCHE_UPLOAD_TEST_SOURCE_HASH_FILE:?}"
  local scratch_assignment="readonly SCRATCH_DIRECTORY=${dollar}{PTYCHE_UPLOAD_TEST_SCRATCH_DIRECTORY:?}"
  local assignment_pattern='^(readonly RCLONE_SOURCE=|readonly EXPECTED_RCLONE_SHA256=|readonly SOURCE=|readonly SOURCE_HASH_FILE=|readonly SCRATCH_DIRECTORY=)'
  local expected_assignment

  for expected_assignment in \
    'readonly RCLONE_SOURCE=' \
    'readonly EXPECTED_RCLONE_SHA256=' \
    'readonly SOURCE=' \
    'readonly SOURCE_HASH_FILE=' \
    'readonly SCRATCH_DIRECTORY='; do
    if [[ $(grep -Fc -- "$expected_assignment" "$batch_script") != 1 ]]; then
      echo "Expected exactly one $expected_assignment assignment in $batch_script" >&2
      exit 1
    fi
  done

  awk \
    -v rclone_source_assignment="$rclone_source_assignment" \
    -v rclone_sha_assignment="$rclone_sha_assignment" \
    -v source_assignment="$source_assignment" \
    -v source_hash_assignment="$source_hash_assignment" \
    -v scratch_assignment="$scratch_assignment" '
      /^readonly RCLONE_SOURCE=/ {
        print rclone_source_assignment
        rclone_source_replacements += 1
        next
      }
      /^readonly EXPECTED_RCLONE_SHA256=/ {
        print rclone_sha_assignment
        rclone_sha_replacements += 1
        next
      }
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
        if (rclone_source_replacements != 1 || rclone_sha_replacements != 1 || source_replacements != 1 || source_hash_replacements != 1 || scratch_replacements != 1) {
          exit 64
        }
      }
    ' "$batch_script" >"$probe_script"

  for expected_assignment in \
    "$rclone_source_assignment" \
    "$rclone_sha_assignment" \
    "$source_assignment" \
    "$source_hash_assignment" \
    "$scratch_assignment"; do
    if [[ $(grep -Fxc -- "$expected_assignment" "$probe_script") != 1 ]]; then
      echo "Upload probe does not contain exactly one $expected_assignment assignment" >&2
      exit 1
    fi
  done
  if [[ $(grep -Ec "$assignment_pattern" "$probe_script") != 5 ]]; then
    echo 'Upload probe contains an unexpected instrumented assignment' >&2
    exit 1
  fi
  if ! cmp -s \
    <(grep -Ev "$assignment_pattern" "$batch_script") \
    <(grep -Ev "$assignment_pattern" "$probe_script"); then
    echo 'Upload probe changed content outside its five path/integrity assignments' >&2
    exit 1
  fi
  chmod 755 "$probe_script"
}

run_expected_ptyche_failure() {
  local batch_script=$1
  local stub_path=$2
  local rclone_path=$3
  local expected_rclone_sha256=$4
  local expected_step=$5
  local expected_exit_status=$6
  local job_id=$7
  local node=$8
  local case_directory=$9
  local exit_status
  local stderr_path=$case_directory/stderr
  local stdout_path=$case_directory/stdout
  local source_path=$case_directory/guaranteed-missing-source.sqsh
  local source_hash_path=$case_directory/source.sqsh.sha256

  if [[ ${expected_step} != preflight-source-files ]]; then
    printf 'source fixture\n' >"${source_path}"
  fi
  printf 'fixture hash\n' >"${source_hash_path}"

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
    PTYCHE_UPLOAD_TEST_RCLONE_SOURCE="$rclone_path" \
    PTYCHE_UPLOAD_TEST_EXPECTED_RCLONE_SHA256="$expected_rclone_sha256" \
    PTYCHE_UPLOAD_TEST_SOURCE="$source_path" \
    PTYCHE_UPLOAD_TEST_SOURCE_HASH_FILE="$source_hash_path" \
    PTYCHE_UPLOAD_TEST_SCRATCH_DIRECTORY="$case_directory/job-scratch" \
    PTYCHE_RCLONE_FIXTURE_LOG="$case_directory/rclone.log" \
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
  local tampered_rclone_directory=$runtime_directory/tampered-rclone
  local incompatible_rclone_directory=$runtime_directory/incompatible-rclone
  local missing_source_directory=$runtime_directory/missing-source
  local probe_script=$runtime_directory/ptyche-upload.sbatch
  local stub_directory=$runtime_directory/bin
  local expected_sha256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a
  local incompatible_sha256
  local compatible_sha256

  mkdir -p \
    "$missing_rclone_directory" \
    "$tampered_rclone_directory" \
    "$incompatible_rclone_directory" \
    "$missing_source_directory" \
    "$stub_directory"
  create_ptyche_upload_probe_script "$batch_script" "$probe_script"
  create_sha256sum_compatibility_stub "$stub_directory/sha256sum"
  cat >"$stub_directory/srun" <<'EOF'
#!/bin/bash
printf 'forbidden dependency stub executed: %s\n' "${0##*/}" >&2
exit 97
EOF
  cat >"$stub_directory/uname" <<'EOF'
#!/bin/bash
if [[ ${1:-} = -m && $# = 1 ]]; then
  printf 'aarch64\n'
  exit 0
fi
exit 96
EOF
  chmod 755 "$stub_directory/srun" "$stub_directory/uname"

  create_rclone_fixture "$tampered_rclone_directory/rclone" 0
  create_rclone_fixture "$incompatible_rclone_directory/rclone" 126
  create_rclone_fixture "$missing_source_directory/rclone" 0
  incompatible_sha256=$(sha256_file "$incompatible_rclone_directory/rclone")
  compatible_sha256=$(sha256_file "$missing_source_directory/rclone")

  run_expected_ptyche_failure \
    "$probe_script" \
    "$stub_directory:/usr/bin:/bin" \
    "$missing_rclone_directory/rclone" \
    "$expected_sha256" \
    preflight-source-rclone-binary \
    1 \
    424241 \
    ptyche-runtime-missing-rclone \
    "$missing_rclone_directory"
  require_pattern \
    "ptyche upload preflight failed: required regular executable is missing: $missing_rclone_directory/rclone" \
    "$missing_rclone_directory/stderr"

  run_expected_ptyche_failure \
    "$probe_script" \
    "$stub_directory:/usr/bin:/bin" \
    "$tampered_rclone_directory/rclone" \
    "$expected_sha256" \
    preflight-source-rclone-integrity \
    1 \
    424242 \
    ptyche-runtime-tampered-rclone \
    "$tampered_rclone_directory"
  require_pattern \
    'ptyche upload integrity check failed: source rclone SHA256 did not match the expected value' \
    "$tampered_rclone_directory/stderr"
  if [[ -e $tampered_rclone_directory/rclone.log ]]; then
    echo 'Tampered rclone binary was executed before its hash was rejected' >&2
    exit 1
  fi

  run_expected_ptyche_failure \
    "$probe_script" \
    "$stub_directory:/usr/bin:/bin" \
    "$incompatible_rclone_directory/rclone" \
    "$incompatible_sha256" \
    preflight-runtime-rclone-compatibility \
    126 \
    424243 \
    ptyche-runtime-incompatible-rclone \
    "$incompatible_rclone_directory"
  require_pattern \
    'ptyche upload preflight failed: rclone binary is incompatible with this node' \
    "$incompatible_rclone_directory/stderr"
  require_pattern 'version' "$incompatible_rclone_directory/rclone.log"

  run_expected_ptyche_failure \
    "$probe_script" \
    "$stub_directory:/usr/bin:/bin" \
    "$missing_source_directory/rclone" \
    "$compatible_sha256" \
    preflight-source-files \
    1 \
    424244 \
    ptyche-runtime-missing-source \
    "$missing_source_directory"
  require_pattern \
    "ptyche upload preflight failed: required regular file is missing: $missing_source_directory/guaranteed-missing-source.sqsh" \
    "$missing_source_directory/stderr"
  fail_if_present 'forbidden dependency stub executed:' "$missing_source_directory/stderr"
  if [[ -e $missing_source_directory/rclone.log ]]; then
    echo 'Source rclone was staged or executed before missing source files were rejected' >&2
    exit 1
  fi
}

create_ptyche_rclone_provisioner_probe() {
  local provisioner=$1
  local probe_script=$2
  local archive_url_assignment="readonly ARCHIVE_URL=${dollar}{PTYCHE_RCLONE_TEST_ARCHIVE_URL:?}"
  local expected_archive_sha_assignment="readonly EXPECTED_ARCHIVE_SHA256=${dollar}{PTYCHE_RCLONE_TEST_EXPECTED_ARCHIVE_SHA256:?}"
  local expected_sha_assignment="readonly EXPECTED_SHA256=${dollar}{PTYCHE_RCLONE_TEST_EXPECTED_SHA256:?}"
  local destination_assignment="readonly DESTINATION=${dollar}{PTYCHE_RCLONE_TEST_DESTINATION:?}"
  local assignment_pattern='^(readonly ARCHIVE_URL=|readonly EXPECTED_ARCHIVE_SHA256=|readonly EXPECTED_SHA256=|readonly DESTINATION=)'
  local expected_assignment

  if [[ ! -f $provisioner ]]; then
    echo "Missing Ptyche rclone provisioner: $provisioner" >&2
    exit 1
  fi
  for expected_assignment in \
    'readonly ARCHIVE_URL=' \
    'readonly EXPECTED_ARCHIVE_SHA256=' \
    'readonly EXPECTED_SHA256=' \
    'readonly DESTINATION='; do
    if [[ $(grep -Fc -- "$expected_assignment" "$provisioner") != 1 ]]; then
      echo "Expected exactly one $expected_assignment assignment in $provisioner" >&2
      exit 1
    fi
  done

  awk \
    -v archive_url_assignment="$archive_url_assignment" \
    -v expected_archive_sha_assignment="$expected_archive_sha_assignment" \
    -v expected_sha_assignment="$expected_sha_assignment" \
    -v destination_assignment="$destination_assignment" '
      /^readonly ARCHIVE_URL=/ {
        print archive_url_assignment
        archive_url_replacements += 1
        next
      }
      /^readonly EXPECTED_ARCHIVE_SHA256=/ {
        print expected_archive_sha_assignment
        expected_archive_sha_replacements += 1
        next
      }
      /^readonly EXPECTED_SHA256=/ {
        print expected_sha_assignment
        expected_sha_replacements += 1
        next
      }
      /^readonly DESTINATION=/ {
        print destination_assignment
        destination_replacements += 1
        next
      }
      { print }
      END {
        if (archive_url_replacements != 1 || expected_archive_sha_replacements != 1 || expected_sha_replacements != 1 || destination_replacements != 1) {
          exit 64
        }
      }
    ' "$provisioner" >"$probe_script"

  for expected_assignment in \
    "$archive_url_assignment" \
    "$expected_archive_sha_assignment" \
    "$expected_sha_assignment" \
    "$destination_assignment"; do
    if [[ $(grep -Fxc -- "$expected_assignment" "$probe_script") != 1 ]]; then
      echo "Provisioner probe does not contain exactly one $expected_assignment assignment" >&2
      exit 1
    fi
  done
  if [[ $(grep -Ec "$assignment_pattern" "$probe_script") != 4 ]]; then
    echo 'Provisioner probe contains an unexpected instrumented assignment' >&2
    exit 1
  fi
  if ! cmp -s \
    <(grep -Ev "$assignment_pattern" "$provisioner") \
    <(grep -Ev "$assignment_pattern" "$probe_script"); then
    echo 'Provisioner probe changed content outside its four path/integrity assignments' >&2
    exit 1
  fi
  chmod 755 "$probe_script"
}

run_ptyche_rclone_provisioner() {
  local probe_script=$1
  local action=$2
  local source_path=$3
  local destination_path=$4
  local expected_sha256=$5
  local case_directory=$6
  local archive_source=$case_directory/release.zip
  local expected_archive_sha256
  local stdout_path=$case_directory/stdout
  local stderr_path=$case_directory/stderr

  if [[ ! -f ${archive_source} ]]; then
    printf 'pinned archive fixture\n' >"${archive_source}"
  fi
  expected_archive_sha256=${PTYCHE_TEST_EXPECTED_ARCHIVE_SHA256_OVERRIDE:-$(sha256_file "${archive_source}")}

  if [[ $action == default ]]; then
    env -i \
      PATH="$PTYCHE_PROVISIONER_TOOL_PATH" \
      AWS_ACCESS_KEY_ID=AKIA_RUNTIME_SECRET_MUST_NOT_LEAK \
      AWS_SECRET_ACCESS_KEY=runtime-secret-key-must-not-leak \
      AWS_SESSION_TOKEN=runtime-session-token-must-not-leak \
      RCLONE_CONFIG_PASS=runtime-rclone-password-must-not-leak \
      PTYCHE_RCLONE_TEST_ARCHIVE_URL=https://downloads.invalid/rclone-arm64.zip \
      PTYCHE_RCLONE_TEST_ARCHIVE_SOURCE="$archive_source" \
      PTYCHE_RCLONE_TEST_EXPECTED_ARCHIVE_SHA256="$expected_archive_sha256" \
      PTYCHE_RCLONE_TEST_SOURCE="$source_path" \
      PTYCHE_RCLONE_TEST_FILE_MODE="${PTYCHE_RCLONE_TEST_FILE_MODE:-arm64}" \
      PTYCHE_RCLONE_TEST_EXPECTED_SHA256="$expected_sha256" \
      PTYCHE_RCLONE_TEST_DESTINATION="$destination_path" \
      PTYCHE_RCLONE_FIXTURE_LOG="$case_directory/rclone.log" \
      PTYCHE_TEST_LN_RACE_KIND="${PTYCHE_TEST_LN_RACE_KIND:-}" \
      PTYCHE_TEST_LN_RACE_BACKING="${PTYCHE_TEST_LN_RACE_BACKING:-}" \
      PTYCHE_TEST_LN_LOG="${PTYCHE_TEST_LN_LOG:-}" \
      "$probe_script" >"$stdout_path" 2>"$stderr_path"
  else
    env -i \
      PATH="$PTYCHE_PROVISIONER_TOOL_PATH" \
      AWS_ACCESS_KEY_ID=AKIA_RUNTIME_SECRET_MUST_NOT_LEAK \
      AWS_SECRET_ACCESS_KEY=runtime-secret-key-must-not-leak \
      AWS_SESSION_TOKEN=runtime-session-token-must-not-leak \
      RCLONE_CONFIG_PASS=runtime-rclone-password-must-not-leak \
      PTYCHE_RCLONE_TEST_ARCHIVE_URL=https://downloads.invalid/rclone-arm64.zip \
      PTYCHE_RCLONE_TEST_ARCHIVE_SOURCE="$archive_source" \
      PTYCHE_RCLONE_TEST_EXPECTED_ARCHIVE_SHA256="$expected_archive_sha256" \
      PTYCHE_RCLONE_TEST_SOURCE="$source_path" \
      PTYCHE_RCLONE_TEST_FILE_MODE="${PTYCHE_RCLONE_TEST_FILE_MODE:-arm64}" \
      PTYCHE_RCLONE_TEST_EXPECTED_SHA256="$expected_sha256" \
      PTYCHE_RCLONE_TEST_DESTINATION="$destination_path" \
      PTYCHE_RCLONE_FIXTURE_LOG="$case_directory/rclone.log" \
      PTYCHE_TEST_LN_RACE_KIND="${PTYCHE_TEST_LN_RACE_KIND:-}" \
      PTYCHE_TEST_LN_RACE_BACKING="${PTYCHE_TEST_LN_RACE_BACKING:-}" \
      PTYCHE_TEST_LN_LOG="${PTYCHE_TEST_LN_LOG:-}" \
      "$probe_script" "$action" >"$stdout_path" 2>"$stderr_path"
  fi
}

expect_ptyche_rclone_provisioner_failure() {
  local probe_script=$1
  local action=$2
  local source_path=$3
  local destination_path=$4
  local expected_sha256=$5
  local expected_exit_status=$6
  local expected_diagnostic=$7
  local case_directory=$8
  local exit_status

  exit_status=0
  if run_ptyche_rclone_provisioner \
    "$probe_script" \
    "$action" \
    "$source_path" \
    "$destination_path" \
    "$expected_sha256" \
    "$case_directory"; then
    echo "Expected Ptyche rclone provisioner to fail: $case_directory" >&2
    exit 1
  else
    exit_status=$?
  fi
  if [[ $exit_status != "$expected_exit_status" ]]; then
    echo "Expected provisioner exit $expected_exit_status, received $exit_status" >&2
    sed -n '1,120p' "$case_directory/stderr" >&2
    exit 1
  fi
  if [[ -s $case_directory/stdout ]]; then
    echo 'Unexpected stdout from failed Ptyche rclone provisioner' >&2
    sed -n '1,120p' "$case_directory/stdout" >&2
    exit 1
  fi
  require_pattern "$expected_diagnostic" "$case_directory/stderr"
  assert_no_sensitive_diagnostic_material "$case_directory/stderr"
}

expect_exact_target_publish_race_failure() {
  local probe_script=$1
  local source_path=$2
  local destination_path=$3
  local expected_sha256=$4
  local case_directory=$5
  local artifact_directory=$6
  local exit_status
  local artifact_count

  exit_status=0
  if run_ptyche_rclone_provisioner \
    "$probe_script" \
    stage \
    "$source_path" \
    "$destination_path" \
    "$expected_sha256" \
    "$case_directory"; then
    echo 'Expected exact-target publish race to fail closed' >&2
    exit 1
  else
    exit_status=$?
  fi
  if [[ $exit_status != 1 ]]; then
    echo "Expected publish race exit 1, received $exit_status" >&2
    sed -n '1,120p' "$case_directory/stderr" >&2
    exit 1
  fi
  artifact_count=$(find "$artifact_directory" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')
  if [[ $artifact_count != 0 ]]; then
    echo 'Publish race created a link inside a directory instead of failing at the exact target' >&2
    find "$artifact_directory" -mindepth 1 -maxdepth 1 -print >&2
    exit 1
  fi
  require_pattern \
    'Ptyche rclone stage failed: could not publish the binary without overwriting a path' \
    "$case_directory/stderr"
  assert_no_sensitive_diagnostic_material "$case_directory/stderr"
}

test_ptyche_rclone_provisioner() {
  local provisioner=$1
  local runtime_directory=$TEST_DIRECTORY/ptyche-rclone-provisioner
  local probe_script=$runtime_directory/provision-ptyche-rclone.sh
  local tool_directory=$runtime_directory/bin
  local absent_directory=$runtime_directory/absent
  local tampered_directory=$runtime_directory/tampered
  local archive_tampered_directory=$runtime_directory/archive-tampered
  local incompatible_directory=$runtime_directory/incompatible
  local no_clobber_directory=$runtime_directory/no-clobber
  local directory_race_directory=$runtime_directory/directory-race
  local symlink_race_directory=$runtime_directory/symlink-race
  local stage_directory=$runtime_directory/stage
  local expected_sha256
  local incompatible_sha256
  local original_destination_sha256
  local staged_sha256
  local staged_entries
  local real_ln
  local real_mkdir

  mkdir -p \
    "$runtime_directory" \
    "$tool_directory" \
    "$absent_directory" \
    "$tampered_directory" \
    "$archive_tampered_directory" \
    "$incompatible_directory" \
    "$no_clobber_directory" \
    "$directory_race_directory/publish" \
    "$symlink_race_directory/publish" \
    "$stage_directory/source"
  create_sha256sum_compatibility_stub "$tool_directory/sha256sum"
  real_ln=$(command -v ln)
  real_mkdir=$(command -v mkdir)
  cat >"$tool_directory/ln" <<EOF
#!/bin/bash
set -euo pipefail
readonly REAL_LN=$real_ln
readonly REAL_MKDIR=$real_mkdir
no_target_directory=false
if [[ \${1:-} == -T ]]; then
  no_target_directory=true
  shift
fi
if [[ \${1:-} != -- || \$# != 3 ]]; then
  printf 'unexpected ln invocation: %s\n' "\$*" >&2
  exit 98
fi
shift
source_path=\$1
destination_path=\$2
if [[ -n \${PTYCHE_TEST_LN_LOG:-} ]]; then
  printf '%s\n' "\$no_target_directory \$source_path \$destination_path" >"\$PTYCHE_TEST_LN_LOG"
fi
case \${PTYCHE_TEST_LN_RACE_KIND:-} in
  '')
    ;;
  directory)
    "\$REAL_MKDIR" -- "\$destination_path"
    ;;
  symlink-directory)
    "\$REAL_MKDIR" -- "\${PTYCHE_TEST_LN_RACE_BACKING:?}"
    "\$REAL_LN" -s -- "\$PTYCHE_TEST_LN_RACE_BACKING" "\$destination_path"
    ;;
  *)
    printf 'unexpected ln race kind: %s\n' "\$PTYCHE_TEST_LN_RACE_KIND" >&2
    exit 98
    ;;
esac
if [[ \$no_target_directory == true && ( -e \$destination_path || -L \$destination_path ) ]]; then
  exit 1
fi
exec "\$REAL_LN" -- "\$source_path" "\$destination_path"
EOF
  cat >"$tool_directory/curl" <<'EOF'
#!/bin/bash
set -euo pipefail
output_path=
requested_url=
while (( $# > 0 )); do
  case $1 in
    --output)
      output_path=$2
      shift 2
      ;;
    http://* | https://*)
      requested_url=$1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
test "${requested_url}" = "${PTYCHE_RCLONE_TEST_ARCHIVE_URL:?}"
/bin/cp -- "${PTYCHE_RCLONE_TEST_ARCHIVE_SOURCE:?}" "${output_path:?}"
EOF
  cat >"$tool_directory/unzip" <<'EOF'
#!/bin/bash
set -euo pipefail
destination=
while (( $# > 0 )); do
  case $1 in
    -d)
      destination=$2
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
target_directory=${destination:?}/rclone-v1.75.0-linux-arm64
/bin/mkdir -p -- "${target_directory}"
/bin/cp -- "${PTYCHE_RCLONE_TEST_SOURCE:?}" "${target_directory}/rclone"
/bin/chmod 500 "${target_directory}/rclone"
EOF
  cat >"$tool_directory/file" <<'EOF'
#!/bin/bash
set -euo pipefail
if [[ ${PTYCHE_RCLONE_TEST_FILE_MODE:-arm64} == arm64 ]]; then
  printf 'ELF 64-bit LSB executable, ARM aarch64, statically linked\n'
else
  printf 'ELF 64-bit LSB executable, x86-64, dynamically linked\n'
fi
EOF
  chmod 755 "$tool_directory/ln" "$tool_directory/curl" "$tool_directory/unzip" "$tool_directory/file"
  PTYCHE_PROVISIONER_TOOL_PATH="$tool_directory:/usr/bin:/bin"
  readonly PTYCHE_PROVISIONER_TOOL_PATH
  create_ptyche_rclone_provisioner_probe "$provisioner" "$probe_script"

  create_rclone_fixture "$tampered_directory/expected-rclone" 0
  expected_sha256=$(sha256_file "$tampered_directory/expected-rclone")
  create_rclone_fixture "$tampered_directory/rclone" 0
  printf '# tampered\n' >>"$tampered_directory/rclone"
  expect_ptyche_rclone_provisioner_failure \
    "$probe_script" \
    default \
    "$absent_directory/missing-source" \
    "$absent_directory/content-address/rclone" \
    "$expected_sha256" \
    1 \
    "Ptyche rclone check failed: required regular executable is missing: $absent_directory/content-address/rclone" \
    "$absent_directory"
  if [[ -e $absent_directory/content-address ]]; then
    echo 'Default provisioner check created the absent destination directory' >&2
    exit 1
  fi

  expect_ptyche_rclone_provisioner_failure \
    "$probe_script" \
    default \
    "$tampered_directory/expected-rclone" \
    "$tampered_directory/rclone" \
    "$expected_sha256" \
    1 \
    'Ptyche rclone check failed: SHA256 did not match the expected content address' \
    "$tampered_directory"
  if [[ -e $tampered_directory/rclone.log ]]; then
    echo 'Tampered provisioned rclone binary was executed before hash rejection' >&2
    exit 1
  fi

  create_rclone_fixture "$archive_tampered_directory/source-rclone" 0
  expected_sha256=$(sha256_file "$archive_tampered_directory/source-rclone")
  PTYCHE_TEST_EXPECTED_ARCHIVE_SHA256_OVERRIDE=0000000000000000000000000000000000000000000000000000000000000000
  expect_ptyche_rclone_provisioner_failure \
    "$probe_script" \
    stage \
    "$archive_tampered_directory/source-rclone" \
    "$archive_tampered_directory/content-address/rclone" \
    "$expected_sha256" \
    1 \
    'Ptyche rclone stage failed: archive SHA256 did not match the pinned release' \
    "$archive_tampered_directory"
  unset PTYCHE_TEST_EXPECTED_ARCHIVE_SHA256_OVERRIDE
  if [[ -e $archive_tampered_directory/content-address/rclone ]]; then
    echo 'Archive-integrity failure published an rclone binary' >&2
    exit 1
  fi

  create_rclone_fixture "$incompatible_directory/rclone" 126
  incompatible_sha256=$(sha256_file "$incompatible_directory/rclone")
  PTYCHE_RCLONE_TEST_FILE_MODE=incompatible
  expect_ptyche_rclone_provisioner_failure \
    "$probe_script" \
    default \
    "$incompatible_directory/rclone" \
    "$incompatible_directory/rclone" \
    "$incompatible_sha256" \
    1 \
    'Ptyche rclone check failed: binary is not Linux ARM64' \
    "$incompatible_directory"
  unset PTYCHE_RCLONE_TEST_FILE_MODE
  if [[ -e $incompatible_directory/rclone.log ]]; then
    echo 'Architecture-mismatched provisioned rclone binary was executed' >&2
    exit 1
  fi

  create_rclone_fixture "$no_clobber_directory/source-rclone" 0
  printf 'must survive unchanged\n' >"$no_clobber_directory/rclone"
  chmod 755 "$no_clobber_directory/rclone"
  original_destination_sha256=$(sha256_file "$no_clobber_directory/rclone")
  expected_sha256=$(sha256_file "$no_clobber_directory/source-rclone")
  expect_ptyche_rclone_provisioner_failure \
    "$probe_script" \
    stage \
    "$no_clobber_directory/source-rclone" \
    "$no_clobber_directory/rclone" \
    "$expected_sha256" \
    1 \
    "Ptyche rclone stage failed: destination already exists; refusing to overwrite: $no_clobber_directory/rclone" \
    "$no_clobber_directory"
  if [[ $(sha256_file "$no_clobber_directory/rclone") != "$original_destination_sha256" ]]; then
    echo 'No-clobber stage modified an existing destination' >&2
    exit 1
  fi

  create_rclone_fixture "$symlink_race_directory/source-rclone" 0
  expected_sha256=$(sha256_file "$symlink_race_directory/source-rclone")
  PTYCHE_TEST_LN_RACE_KIND=symlink-directory
  PTYCHE_TEST_LN_RACE_BACKING=$symlink_race_directory/race-backing
  PTYCHE_TEST_LN_LOG=$symlink_race_directory/ln.log
  expect_exact_target_publish_race_failure \
    "$probe_script" \
    "$symlink_race_directory/source-rclone" \
    "$symlink_race_directory/publish/rclone" \
    "$expected_sha256" \
    "$symlink_race_directory" \
    "$symlink_race_directory/race-backing"
  if [[ ! -L $symlink_race_directory/publish/rclone ]]; then
    echo 'Symlink publish race did not retain the externally-created target symlink' >&2
    exit 1
  fi

  create_rclone_fixture "$directory_race_directory/source-rclone" 0
  expected_sha256=$(sha256_file "$directory_race_directory/source-rclone")
  PTYCHE_TEST_LN_RACE_KIND=directory
  PTYCHE_TEST_LN_RACE_BACKING=
  PTYCHE_TEST_LN_LOG=$directory_race_directory/ln.log
  expect_exact_target_publish_race_failure \
    "$probe_script" \
    "$directory_race_directory/source-rclone" \
    "$directory_race_directory/publish/rclone" \
    "$expected_sha256" \
    "$directory_race_directory" \
    "$directory_race_directory/publish/rclone"
  if [[ ! -d $directory_race_directory/publish/rclone || -L $directory_race_directory/publish/rclone ]]; then
    echo 'Directory publish race did not retain the externally-created target directory' >&2
    exit 1
  fi
  unset PTYCHE_TEST_LN_RACE_KIND PTYCHE_TEST_LN_RACE_BACKING PTYCHE_TEST_LN_LOG

  create_rclone_fixture "$stage_directory/source/rclone" 0
  printf 'runtime-secret-key-must-not-leak\n' >"$stage_directory/source/rclone.conf"
  expected_sha256=$(sha256_file "$stage_directory/source/rclone")
  run_ptyche_rclone_provisioner \
    "$probe_script" \
    stage \
    "$stage_directory/source/rclone" \
    "$stage_directory/content-address/rclone" \
    "$expected_sha256" \
    "$stage_directory"
  require_pattern \
    "Ptyche rclone staged and verified: $stage_directory/content-address/rclone" \
    "$stage_directory/stdout"
  if [[ -s $stage_directory/stderr ]]; then
    echo 'Unexpected stderr from successful Ptyche rclone stage' >&2
    sed -n '1,120p' "$stage_directory/stderr" >&2
    exit 1
  fi
  staged_sha256=$(sha256_file "$stage_directory/content-address/rclone")
  if [[ $staged_sha256 != "$expected_sha256" || ! -x $stage_directory/content-address/rclone ]]; then
    echo 'Staged Ptyche rclone does not match the executable source bytes' >&2
    exit 1
  fi
  staged_entries=$(find "$stage_directory/content-address" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')
  if [[ $staged_entries != 1 ]]; then
    echo 'Ptyche rclone stage left files other than the immutable binary' >&2
    find "$stage_directory/content-address" -mindepth 1 -maxdepth 1 -print >&2
    exit 1
  fi
  fail_if_present 'runtime-secret-key-must-not-leak' "$stage_directory/stdout"
  fail_if_present 'runtime-secret-key-must-not-leak' "$stage_directory/stderr"
  if [[ -e $stage_directory/rclone.log ]]; then
    echo 'Provisioner executed a cross-architecture rclone binary on the login node' >&2
    exit 1
  fi
}

create_ptyche_cleanup_probe_script() {
  local batch_script=$1
  local probe_script=$2
  local rclone_source_assignment="readonly RCLONE_SOURCE=\${PTYCHE_UPLOAD_TEST_RCLONE_SOURCE:?}"
  local rclone_sha_assignment="readonly EXPECTED_RCLONE_SHA256=\${PTYCHE_UPLOAD_TEST_EXPECTED_RCLONE_SHA256:?}"
  local source_assignment="readonly SOURCE=\${PTYCHE_UPLOAD_TEST_SOURCE:?}"
  local source_hash_assignment="readonly SOURCE_HASH_FILE=\${PTYCHE_UPLOAD_TEST_SOURCE_HASH_FILE:?}"
  local scratch_assignment="readonly SCRATCH_DIRECTORY=\${PTYCHE_UPLOAD_TEST_SCRATCH_DIRECTORY:?}"
  local assignment_pattern='^(readonly RCLONE_SOURCE=|readonly EXPECTED_RCLONE_SHA256=|readonly SOURCE=|readonly SOURCE_HASH_FILE=|readonly SCRATCH_DIRECTORY=)'
  local expected_assignment

  for expected_assignment in \
    'readonly RCLONE_SOURCE=' \
    'readonly EXPECTED_RCLONE_SHA256=' \
    'readonly SOURCE=' \
    'readonly SOURCE_HASH_FILE=' \
    'readonly SCRATCH_DIRECTORY='; do
    if [[ $(grep -Fc -- "$expected_assignment" "$batch_script") != 1 ]]; then
      echo "Expected exactly one $expected_assignment assignment in $batch_script" >&2
      exit 1
    fi
  done

  awk \
    -v rclone_source_assignment="$rclone_source_assignment" \
    -v rclone_sha_assignment="$rclone_sha_assignment" \
    -v source_assignment="$source_assignment" \
    -v source_hash_assignment="$source_hash_assignment" \
    -v scratch_assignment="$scratch_assignment" '
      /^readonly RCLONE_SOURCE=/ {
        print rclone_source_assignment
        rclone_source_replacements += 1
        next
      }
      /^readonly EXPECTED_RCLONE_SHA256=/ {
        print rclone_sha_assignment
        rclone_sha_replacements += 1
        next
      }
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
        if (rclone_source_replacements != 1 || rclone_sha_replacements != 1 || source_replacements != 1 || source_hash_replacements != 1 || scratch_replacements != 1) {
          exit 64
        }
      }
    ' "$batch_script" >"$probe_script"

  for expected_assignment in \
    "$rclone_source_assignment" \
    "$rclone_sha_assignment" \
    "$source_assignment" \
    "$source_hash_assignment" \
    "$scratch_assignment"; do
    if [[ $(grep -Fxc -- "$expected_assignment" "$probe_script") != 1 ]]; then
      echo "Cleanup probe does not contain exactly one $expected_assignment assignment" >&2
      exit 1
    fi
  done
  if [[ $(grep -Ec "$assignment_pattern" "$probe_script") != 5 ]]; then
    echo 'Cleanup probe contains an unexpected instrumented assignment' >&2
    exit 1
  fi
  if ! cmp -s \
    <(grep -Ev "$assignment_pattern" "$batch_script") \
    <(grep -Ev "$assignment_pattern" "$probe_script"); then
    echo 'Cleanup probe changed content outside its five path assignments' >&2
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
  version | purge | lsjson)
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
if [[ \${PTYCHE_TEST_TRIGGER_ORIGINAL_FAILURE:-false} = true && \${1:-} = \${PTYCHE_UPLOAD_TEST_SOURCE:?} ]]; then
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
/bin/mkdir "$@"
EOF
  cat >"$stub_directory/cp" <<'EOF'
#!/bin/bash
set -euo pipefail
/bin/cp "$@"
EOF
  cat >"$stub_directory/chmod" <<'EOF'
#!/bin/bash
set -euo pipefail
/bin/chmod "$@"
EOF
  cat >"$stub_directory/uname" <<'EOF'
#!/bin/bash
set -euo pipefail
if [[ ${1:-} = -m && $# = 1 ]]; then
  printf 'aarch64\n'
  exit 0
fi
exit 96
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
  local expect_remote_cleanup=$7
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
    PTYCHE_UPLOAD_TEST_RCLONE_SOURCE="$stub_directory/rclone" \
    PTYCHE_UPLOAD_TEST_EXPECTED_RCLONE_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e \
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
  if [[ ${expect_remote_cleanup} == true ]]; then
    require_pattern 'purge ' "$rclone_log"
  else
    fail_if_present 'purge ' "$rclone_log"
  fi
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
    ptyche-runtime-original-failure \
    false
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
    ptyche-runtime-success-cleanup-failure \
    true
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
test_ptyche_rclone_provisioner "$PTYCHE_RCLONE_PROVISIONER"

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
require_pattern 'readonly RCLONE_SOURCE=/home/sna/.local/libexec/nemo-rl/rclone/sha256-a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a/rclone' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'readonly EXPECTED_RCLONE_SHA256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'readonly EXPECTED_COMPUTE_ARCHITECTURE=aarch64' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command sha256sum' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command stat' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command awk' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'require_command srun' "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_equal \"$dollar{compute_architecture}\" \"$dollar{EXPECTED_COMPUTE_ARCHITECTURE}\" 'compute architecture'" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_regular_executable \"$dollar{RCLONE_SOURCE}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_equal \"$dollar{source_rclone_sha256}\" \"$dollar{EXPECTED_RCLONE_SHA256}\" 'source rclone SHA256'" "$PTYCHE_UPLOAD_BATCH"
require_pattern "cp -- \"$dollar{RCLONE_SOURCE}\" \"$dollar{RCLONE}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_equal \"$dollar{runtime_rclone_sha256}\" \"$dollar{EXPECTED_RCLONE_SHA256}\" 'runtime rclone SHA256'" "$PTYCHE_UPLOAD_BATCH"
require_pattern "require_compatible_rclone \"$dollar{RCLONE}\"" "$PTYCHE_UPLOAD_BATCH"
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
if [[ $(grep -Fc -- "\"$dollar{RCLONE}\" copy" "$PTYCHE_UPLOAD_BATCH") != 4 ]]; then
  echo 'Every Ptyche remote copy must invoke the pinned absolute rclone path' >&2
  exit 1
fi
require_pattern "\"$dollar{RCLONE}\" purge" "$PTYCHE_UPLOAD_BATCH"
require_pattern "\"$dollar{RCLONE}\" lsjson" "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'require_command rclone' "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'command -v rclone >/dev/null' "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'command -v sha256sum >/dev/null' "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'command -v stat >/dev/null' "$PTYCHE_UPLOAD_BATCH"
fail_if_present "test -f \"$dollar{SOURCE}\"" "$PTYCHE_UPLOAD_BATCH"
fail_if_present "test -f \"$dollar{SOURCE_HASH_FILE}\"" "$PTYCHE_UPLOAD_BATCH"
fail_if_present "test ! -e \"$dollar{SCRATCH_DIRECTORY}\"" "$PTYCHE_UPLOAD_BATCH"
fail_if_present 'test ' "$PTYCHE_UPLOAD_BATCH"
fail_if_present '|| true' "$PTYCHE_UPLOAD_BATCH"
fail_if_present '|| :' "$PTYCHE_UPLOAD_BATCH"

rclone_binary_line=$(line_number '^CURRENT_STEP=preflight-source-rclone-binary$' 1 "$PTYCHE_UPLOAD_BATCH")
rclone_integrity_line=$(line_number '^CURRENT_STEP=preflight-source-rclone-integrity$' 1 "$PTYCHE_UPLOAD_BATCH")
rclone_stage_line=$(line_number '^CURRENT_STEP=stage-runtime-rclone$' 1 "$PTYCHE_UPLOAD_BATCH")
rclone_compatibility_line=$(line_number '^CURRENT_STEP=preflight-runtime-rclone-compatibility$' 1 "$PTYCHE_UPLOAD_BATCH")
rclone_first_remote_line=$(line_number '^CURRENT_STEP=upload-job-temporary-directory$' 1 "$PTYCHE_UPLOAD_BATCH")
test "$rclone_binary_line" -lt "$rclone_integrity_line"
test "$rclone_integrity_line" -lt "$rclone_stage_line"
test "$rclone_stage_line" -lt "$rclone_compatibility_line"
test "$rclone_compatibility_line" -lt "$rclone_first_remote_line"

require_pattern '# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern 'readonly ARCHIVE_URL=https://downloads.rclone.org/v1.75.0/rclone-v1.75.0-linux-arm64.zip' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern 'readonly EXPECTED_ARCHIVE_SHA256=d0ad88ba4c8e285b7c9efa591e0ab643280a91741e13c27f3a9c0957ccfa5203' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern 'readonly EXPECTED_SHA256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern 'readonly DESTINATION=/home/sna/.local/libexec/nemo-rl/rclone/sha256-a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a/rclone' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern '  --connect-timeout 20' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern '  --max-time 300' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern '  --retry 2' "$PTYCHE_RCLONE_PROVISIONER"
require_pattern "readonly ACTION=$dollar{1:-check}" "$PTYCHE_RCLONE_PROVISIONER"
require_pattern "if [[ ${dollar}{ACTION} == check ]]" "$PTYCHE_RCLONE_PROVISIONER"
require_pattern "if [[ -e ${dollar}{DESTINATION} || -L ${dollar}{DESTINATION} ]]" "$PTYCHE_RCLONE_PROVISIONER"
require_pattern "temporary_path=\$(mktemp \"${dollar}{DESTINATION_DIRECTORY}/.rclone.stage.XXXXXXXX\")" "$PTYCHE_RCLONE_PROVISIONER"
require_pattern "ln -T -- \"${dollar}{temporary_path}\" \"${dollar}{DESTINATION}\"" "$PTYCHE_RCLONE_PROVISIONER"
require_pattern "rm -f -- \"${dollar}{temporary_path}\"" "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present 'rm -rf' "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present 'cp -f' "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present 'ln -f' "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present '.config' "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present 'rclone.conf' "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present 'RCLONE_CONFIG' "$PTYCHE_RCLONE_PROVISIONER"
fail_if_present 'AWS_' "$PTYCHE_RCLONE_PROVISIONER"

batch_rclone_path=$(sed -n 's/^readonly RCLONE_SOURCE=//p' "$PTYCHE_UPLOAD_BATCH")
provisioned_rclone_path=$(sed -n 's/^readonly DESTINATION=//p' "$PTYCHE_RCLONE_PROVISIONER")
batch_rclone_sha256=$(sed -n 's/^readonly EXPECTED_RCLONE_SHA256=//p' "$PTYCHE_UPLOAD_BATCH")
provisioned_rclone_sha256=$(sed -n 's/^readonly EXPECTED_SHA256=//p' "$PTYCHE_RCLONE_PROVISIONER")
if [[ $batch_rclone_path != "$provisioned_rclone_path" ]]; then
  echo 'Ptyche upload and provisioner rclone paths differ' >&2
  exit 1
fi
if [[ $batch_rclone_sha256 != "$provisioned_rclone_sha256" ]]; then
  echo 'Ptyche upload and provisioner rclone SHA256 values differ' >&2
  exit 1
fi

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
