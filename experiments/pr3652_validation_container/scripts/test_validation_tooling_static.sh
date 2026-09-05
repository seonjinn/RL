#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "$script_dir" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
readonly SCRIPTS_DIRECTORY=$SCRIPT_ROOT/experiments/pr3652_validation_container/scripts
readonly DOWNLOAD_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_download_validated_nightly.sbatch
readonly SMOKE_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_smoke_validated_nightly.sbatch
readonly SMOKE_BODY=$SCRIPTS_DIRECTORY/oci_hsg_smoke_validated_nightly.sh
readonly SMOKE_SUBMIT=$SCRIPTS_DIRECTORY/submit_oci_hsg_smoke_validated_nightly.sh
readonly CAPTURE_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_capture_precision_source_evidence.sbatch
readonly CAPTURE_BODY=$SCRIPTS_DIRECTORY/oci_hsg_capture_precision_source_evidence.sh
readonly CAPTURE_SUBMIT=$SCRIPTS_DIRECTORY/submit_oci_hsg_capture_precision_source_evidence.sh
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

assert_hostile_scheduler_environment_absent() {
  local environment_path=$1
  local variable_name

  for variable_name in \
    SBATCH_ACCOUNT \
    SBATCH_PARTITION \
    SBATCH_GPUS \
    SBATCH_GRES \
    SBATCH_EXCLUSIVE \
    SBATCH_TIME \
    SLURM_CLUSTERS \
    SLURM_HINT; do
    if grep -Eq "^${variable_name}=" "$environment_path"; then
      echo "Hostile scheduler variable reached sbatch: ${variable_name}" >&2
      exit 1
    fi
  done
}

assert_sanitized_scheduler_environment() {
  local environment_path=$1
  local expected_slurm_conf=$2

  assert_hostile_scheduler_environment_absent "$environment_path"
  grep -Fx -- 'PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin' "$environment_path" >/dev/null
  grep -Fx -- "SLURM_CONF=${expected_slurm_conf}" "$environment_path" >/dev/null
  if grep -Eq '^HOME=' "$environment_path"; then
    echo 'HOME reached sanitized sbatch environment' >&2
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
  lsf)
    printf '17;nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh\n'
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
  local expected_rclone_log=$case_directory/expected-rclone.log
  local srun_log=$case_directory/srun.log
  local expected_srun_log=$case_directory/expected-srun.log
  local basename=nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
  local final_directory=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers
  local destination=${final_directory}/${basename}
  local temporary_directory=${final_directory}/.temporary-${basename}.${job_id}
  local temporary_object=${temporary_directory}/${basename}
  local runtime_rclone=${scratch_path}/rclone/rclone
  local temporary_download=${scratch_path}/temporary/${basename}
  local final_download=${scratch_path}/final/${basename}
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
    {
      printf '%s\n' \
        "-J upload-validated-nightly-temporary ${runtime_rclone} copyto --verbose --checksum --checkers=8 --transfers=4 --multi-thread-streams=32 --s3-upload-concurrency=32 --s3-chunk-size=128M --buffer-size=128M ${source_path} ${temporary_object}" \
        "-J verify-validated-nightly-temporary ${runtime_rclone} copyto --verbose --checksum ${temporary_object} ${temporary_download}" \
        "-J publish-validated-nightly ${runtime_rclone} copyto --verbose --immutable --checksum ${temporary_object} ${destination}" \
        "-J verify-validated-nightly-final ${runtime_rclone} copyto --verbose --checksum ${destination} ${final_download}"
    } >"${expected_srun_log}"
    if ! cmp -s "${expected_srun_log}" "${srun_log}"; then
      echo 'Ptyche transfer commands did not preserve the exact ordered object paths' >&2
      diff -u "${expected_srun_log}" "${srun_log}" >&2 || :
      exit 1
    fi
    {
      printf '%s\n' \
        version \
        "lsf --files-only --format sp ${temporary_directory}" \
        "lsf --files-only --format sp --include /${basename} ${final_directory}" \
        "lsjson --stat ${destination}" \
        "purge ${temporary_directory}"
    } >"${expected_rclone_log}"
  else
    if [[ -e ${srun_log} ]]; then
      echo 'Pre-upload failure unexpectedly launched an srun transfer' >&2
      exit 1
    fi
    printf 'version\n' >"${expected_rclone_log}"
  fi
  if ! cmp -s "${expected_rclone_log}" "${rclone_log}"; then
    echo 'Ptyche direct rclone checks or cleanup did not preserve exact ordering and paths' >&2
    diff -u "${expected_rclone_log}" "${rclone_log}" >&2 || :
    exit 1
  fi
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

create_oci_raw_metadata_fixture() {
  local root=$1
  local entries=$root/.SHA256SUMS.entries
  local relative_path
  local payload_sha256

  for relative_path in \
    checkpoints/kimi_k2/config.json \
    checkpoints/kimi_k2/model.safetensors.index.json \
    checkpoints/kimi_k2/safetensors_header_manifest.json \
    checkpoints/kimi_k3/config.json \
    checkpoints/kimi_k3/model.safetensors.index.json \
    checkpoints/kimi_k3/safetensors_header_manifest.json \
    checkpoints/kimi_k25/config.json \
    checkpoints/kimi_k25/model.safetensors.index.json \
    checkpoints/kimi_k25/safetensors_header_manifest.json \
    checkpoints/qwen3_bf16/config.json \
    checkpoints/qwen3_bf16/model.safetensors.index.json \
    checkpoints/qwen3_bf16/safetensors_header_manifest.json \
    checkpoints/nemotron_lightning_nvfp4/config.json \
    checkpoints/nemotron_lightning_nvfp4/model.safetensors.index.json \
    checkpoints/nemotron_lightning_nvfp4/safetensors_header_manifest.json \
    checkpoints/qwen_a95b_fp8/config.json \
    checkpoints/qwen_a95b_fp8/model.safetensors.index.json \
    checkpoints/qwen_a95b_fp8/safetensors_header_byte_lengths.json \
    checkpoints/qwen_a95b_fp8/safetensors_header_manifest.json; do
    mkdir -p "$root/${relative_path%/*}"
    printf 'raw metadata fixture: %s\n' "$relative_path" >"$root/$relative_path"
    payload_sha256=$(sha256_file "$root/$relative_path")
    printf '%s  %s\n' "$payload_sha256" "$relative_path" >>"$entries"
  done
  LC_ALL=C sort "$entries" >"$root/SHA256SUMS"
  rm -f -- "$entries"
}

create_oci_capture_body_probe() {
  local body_script=$1
  local probe_script=$2
  local runtime_directory=$3
  local python_stub=$4
  local raw_manifest_sha256=$5
  local escaped_python_stub=${python_stub//\#/\\#}
  local escaped_runtime_directory=${runtime_directory//\#/\\#}

  sed \
    -e "s#^readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python\$#readonly MAIN_PYTHON=${escaped_python_stub}#" \
    -e "s#^readonly EXPECTED_RAW_MANIFEST_SHA256=.*\$#readonly EXPECTED_RAW_MANIFEST_SHA256=${raw_manifest_sha256}#" \
    -e "s#/raid/scratch/nemo-rl-semantic-precision-evidence#${escaped_runtime_directory}/scratch#g" \
    -e "s#/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured#${escaped_runtime_directory}/published#g" \
    "$body_script" >"$probe_script"
  chmod 755 "$probe_script"
}

test_oci_capture_check_failure_does_not_publish() {
  local runtime_directory=$TEST_DIRECTORY/oci-capture-body
  local tool_directory=$runtime_directory/bin
  local semantic_worktree=$runtime_directory/semantic
  local compressed_root=$runtime_directory/compressed-tensors
  local modelopt_root=$runtime_directory/model-optimizer
  local transformer_engine_root=$runtime_directory/transformer-engine
  local raw_root=$runtime_directory/raw
  local scratch_directory=$runtime_directory/scratch/oci-capture-424242
  local captured_base=$runtime_directory/published
  local probe_script=$runtime_directory/capture-body.sh
  local python_stub=$tool_directory/python
  local marker=$semantic_worktree/tests/fixtures/precision_policy/do-not-write
  local exit_status=0
  local raw_manifest_sha256
  local real_sha256_kind
  local real_sha256_tool

  if real_sha256_tool=$(command -v sha256sum); then
    real_sha256_kind=sha256sum
  else
    real_sha256_tool=$(command -v shasum)
    real_sha256_kind=shasum
  fi

  mkdir -p \
    "$tool_directory" \
    "$semantic_worktree/tools" \
    "$semantic_worktree/tests/fixtures/precision_policy" \
    "$compressed_root" \
    "$modelopt_root" \
    "$transformer_engine_root" \
    "$raw_root" \
    "$scratch_directory" \
    "$captured_base/runs"
  printf 'capture tool fixture\n' >"$semantic_worktree/tools/capture_precision_policy_source_evidence.py"
  printf 'must remain unchanged\n' >"$marker"
  create_oci_raw_metadata_fixture "$raw_root"
  raw_manifest_sha256=$(sha256_file "$raw_root/SHA256SUMS")

  cat >"$tool_directory/git" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ ${1:-} != -C || $# < 4 ]]; then
  exit 96
fi
path=$2
shift 2
case "$*" in
  'rev-parse --is-inside-work-tree')
    printf 'true\n'
    ;;
  'rev-parse --show-toplevel')
    printf '%s\n' "$path"
    ;;
  'status --porcelain')
    ;;
  'rev-parse HEAD')
    case $path in
      "$OCI_CAPTURE_TEST_SEMANTIC_WORKTREE") printf '%s\n' "$OCI_CAPTURE_TEST_SEMANTIC_SHA" ;;
      "$OCI_CAPTURE_TEST_COMPRESSED_ROOT") printf '%s\n' f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0 ;;
      "$OCI_CAPTURE_TEST_MODELOPT_ROOT") printf '%s\n' c897fbeaaff66d53d61033f107885b7c5432f235 ;;
      "$OCI_CAPTURE_TEST_TE_ROOT") printf '%s\n' 42b840051647eef89761a16dfdff87e82bb253ab ;;
      */3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM) printf '%s\n' 7c9c3a027c503ae9ae1e8ad7b14397abb8269378 ;;
      */3rdparty/Megatron-Bridge-workspace/Megatron-Bridge) printf '%s\n' b11414c71b15e54d333eb49346ed199f20fa9021 ;;
      */3rdparty/Automodel-workspace/Automodel) printf '%s\n' 1814c6c93a66b9d59d254960ef6a99a64249b671 ;;
      *) exit 95 ;;
    esac
    ;;
  'config --get-all remote.origin.url')
    case $path in
      "$OCI_CAPTURE_TEST_COMPRESSED_ROOT") printf '%s\n' https://github.com/vllm-project/compressed-tensors.git ;;
      "$OCI_CAPTURE_TEST_MODELOPT_ROOT") printf '%s\n' https://github.com/NVIDIA/Model-Optimizer.git ;;
      "$OCI_CAPTURE_TEST_TE_ROOT") printf '%s\n' https://github.com/NVIDIA/TransformerEngine.git ;;
      *) exit 95 ;;
    esac
    ;;
  *)
    printf 'unexpected capture git invocation: %s\n' "$*" >&2
    exit 94
    ;;
esac
EOF
  cat >"$tool_directory/sha256sum" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ ${1:-} == --check ]]; then
  manifest=
  for argument in "$@"; do
    manifest=$argument
  done
  while read -r expected_hash relative_path; do
    if [[ $OCI_CAPTURE_TEST_REAL_SHA256_KIND == shasum ]]; then
      actual_hash=$("$OCI_CAPTURE_TEST_REAL_SHA256" -a 256 "$relative_path" | awk '{print $1}')
    else
      actual_hash=$("$OCI_CAPTURE_TEST_REAL_SHA256" "$relative_path" | awk '{print $1}')
    fi
    test "$actual_hash" = "$expected_hash"
  done <"$manifest"
  exit 0
fi
if [[ $OCI_CAPTURE_TEST_REAL_SHA256_KIND == shasum ]]; then
  exec "$OCI_CAPTURE_TEST_REAL_SHA256" -a 256 "$@"
fi
exec "$OCI_CAPTURE_TEST_REAL_SHA256" "$@"
EOF
  cat >"$python_stub" <<'EOF'
#!/bin/bash
set -euo pipefail

count=0
if [[ -f $OCI_CAPTURE_TEST_INVOCATION_COUNT ]]; then
  count=$(<"$OCI_CAPTURE_TEST_INVOCATION_COUNT")
fi
count=$((count + 1))
printf '%s\n' "$count" >"$OCI_CAPTURE_TEST_INVOCATION_COUNT"
printf '%s\n' "$@" >"$OCI_CAPTURE_TEST_INVOCATIONS.$count"
output_directory=
check=false
while (( $# > 0 )); do
  case $1 in
    --output-directory)
      output_directory=$2
      shift 2
      ;;
    --check)
      check=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done
test "$output_directory" = "$OCI_CAPTURE_TEST_OUTPUT_DIRECTORY"
case $output_directory in
  "$OCI_CAPTURE_TEST_SEMANTIC_WORKTREE"/*)
    echo 'capture attempted to write into the semantic worktree' >&2
    exit 93
    ;;
esac
if [[ $check == false ]]; then
  printf '{}\n' >"$output_directory/source_format_evidence.json"
  printf '{}\n' >"$output_directory/producer_implementations.json"
  exit 0
fi
exit "${OCI_CAPTURE_TEST_CHECK_EXIT:-42}"
EOF
  cat >"$tool_directory/mv" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ ${1:-} == -Tn && ${2:-} == -- && $# == 4 ]]; then
  source_path=$3
  destination_path=$4
  printf '%s\t%s\n' "$source_path" "$destination_path" >>"$OCI_CAPTURE_TEST_MV_LOG"
  if [[ -e $destination_path || -L $destination_path ]]; then
    exit 0
  fi
  exec "$OCI_CAPTURE_TEST_REAL_MV" "$source_path" "$destination_path"
fi
exec "$OCI_CAPTURE_TEST_REAL_MV" "$@"
EOF
  chmod 755 "$tool_directory/git" "$tool_directory/sha256sum" "$tool_directory/mv" "$python_stub"
  create_oci_capture_body_probe "$CAPTURE_BODY" "$probe_script" "$runtime_directory" "$python_stub" "$raw_manifest_sha256"

  if env -i \
    PATH="$tool_directory:/usr/bin:/bin" \
    SLURM_JOB_ID=424242 \
    OCI_CAPTURE_TEST_SEMANTIC_WORKTREE="$semantic_worktree" \
    OCI_CAPTURE_TEST_SEMANTIC_SHA=1111111111111111111111111111111111111111 \
    OCI_CAPTURE_TEST_COMPRESSED_ROOT="$compressed_root" \
    OCI_CAPTURE_TEST_MODELOPT_ROOT="$modelopt_root" \
    OCI_CAPTURE_TEST_TE_ROOT="$transformer_engine_root" \
    OCI_CAPTURE_TEST_REAL_SHA256="$real_sha256_tool" \
    OCI_CAPTURE_TEST_REAL_SHA256_KIND="$real_sha256_kind" \
    OCI_CAPTURE_TEST_REAL_MV="$(command -v mv)" \
    OCI_CAPTURE_TEST_MV_LOG="$runtime_directory/mv.log" \
    OCI_CAPTURE_TEST_INVOCATION_COUNT="$runtime_directory/invocation-count" \
    OCI_CAPTURE_TEST_INVOCATIONS="$runtime_directory/invocations" \
    OCI_CAPTURE_TEST_OUTPUT_DIRECTORY="$scratch_directory/captured" \
    TMPDIR="$scratch_directory/tmp" \
    PYTHONPYCACHEPREFIX="$scratch_directory/pycache" \
    XDG_CACHE_HOME="$scratch_directory/xdg-cache" \
    UV_CACHE_DIR="$scratch_directory/uv-cache" \
    TORCHINDUCTOR_CACHE_DIR="$scratch_directory/torchinductor-cache" \
    TRITON_CACHE_DIR="$scratch_directory/triton-cache" \
    "$probe_script" \
    "$semantic_worktree" \
    1111111111111111111111111111111111111111 \
    "$compressed_root" \
    "$modelopt_root" \
    "$transformer_engine_root" \
    "$raw_root" \
    "$scratch_directory" \
    "$captured_base" \
    2222222222222222222222222222222222222222 \
    >"$runtime_directory/stdout" 2>"$runtime_directory/stderr"; then
    echo 'OCI capture body accepted a failing --check capture' >&2
    exit 1
  else
    exit_status=$?
  fi
  if [[ $exit_status != 42 ]]; then
    echo "Expected OCI capture check failure exit 42, received $exit_status" >&2
    sed -n '1,120p' "$runtime_directory/stderr" >&2
    exit 1
  fi
  test "$(<"$runtime_directory/invocation-count")" = 2
  printf '%s\n' \
    "$semantic_worktree/tools/capture_precision_policy_source_evidence.py" \
    --repository-root \
    "$semantic_worktree" \
    --compressed-tensors-source-root \
    "$compressed_root" \
    --modelopt-lightning-source-root \
    "$modelopt_root" \
    --staged-metadata-root \
    "$raw_root" \
    --transformer-engine-source-root \
    "$transformer_engine_root" \
    --output-directory \
    "$scratch_directory/captured" \
    --inspect-runtime >"$runtime_directory/expected-invocation.1"
  printf '%s\n' \
    "$semantic_worktree/tools/capture_precision_policy_source_evidence.py" \
    --repository-root \
    "$semantic_worktree" \
    --compressed-tensors-source-root \
    "$compressed_root" \
    --modelopt-lightning-source-root \
    "$modelopt_root" \
    --staged-metadata-root \
    "$raw_root" \
    --transformer-engine-source-root \
    "$transformer_engine_root" \
    --output-directory \
    "$scratch_directory/captured" \
    --check \
    --inspect-runtime >"$runtime_directory/expected-invocation.2"
  cmp -s "$runtime_directory/invocations.1" "$runtime_directory/expected-invocation.1"
  cmp -s "$runtime_directory/invocations.2" "$runtime_directory/expected-invocation.2"
  fail_if_present 'tests/fixtures/precision_policy' "$runtime_directory/invocations.1"
  fail_if_present 'tests/fixtures/precision_policy' "$runtime_directory/invocations.2"
  test "$(<"$marker")" = 'must remain unchanged'
  test -f "$scratch_directory/captured/source_format_evidence.json"
  test -f "$scratch_directory/captured/producer_implementations.json"
  test "$(find "$captured_base" -mindepth 1 -maxdepth 1 ! -name runs -print | wc -l | tr -d ' ')" = 0
  test "$(find "$captured_base/runs" -mindepth 1 -print | wc -l | tr -d ' ')" = 0

  run_successful_capture_probe() {
    local job_id=$1
    local job_scratch=$2
    local expected_status=${3:-0}
    local actual_status=0

    mkdir -p "$job_scratch"
    if env -i \
      PATH="$tool_directory:/usr/bin:/bin" \
      SLURM_JOB_ID="$job_id" \
      OCI_CAPTURE_TEST_SEMANTIC_WORKTREE="$semantic_worktree" \
      OCI_CAPTURE_TEST_SEMANTIC_SHA=1111111111111111111111111111111111111111 \
      OCI_CAPTURE_TEST_COMPRESSED_ROOT="$compressed_root" \
      OCI_CAPTURE_TEST_MODELOPT_ROOT="$modelopt_root" \
      OCI_CAPTURE_TEST_TE_ROOT="$transformer_engine_root" \
      OCI_CAPTURE_TEST_REAL_SHA256="$real_sha256_tool" \
      OCI_CAPTURE_TEST_REAL_SHA256_KIND="$real_sha256_kind" \
      OCI_CAPTURE_TEST_REAL_MV="$(command -v mv)" \
      OCI_CAPTURE_TEST_MV_LOG="$runtime_directory/mv.log" \
      OCI_CAPTURE_TEST_INVOCATION_COUNT="$runtime_directory/invocation-count" \
      OCI_CAPTURE_TEST_INVOCATIONS="$runtime_directory/invocations" \
      OCI_CAPTURE_TEST_OUTPUT_DIRECTORY="$job_scratch/captured" \
      OCI_CAPTURE_TEST_CHECK_EXIT=0 \
      TMPDIR="$job_scratch/tmp" \
      PYTHONPYCACHEPREFIX="$job_scratch/pycache" \
      XDG_CACHE_HOME="$job_scratch/xdg-cache" \
      UV_CACHE_DIR="$job_scratch/uv-cache" \
      TORCHINDUCTOR_CACHE_DIR="$job_scratch/torchinductor-cache" \
      TRITON_CACHE_DIR="$job_scratch/triton-cache" \
      "$probe_script" \
      "$semantic_worktree" \
      1111111111111111111111111111111111111111 \
      "$compressed_root" \
      "$modelopt_root" \
      "$transformer_engine_root" \
      "$raw_root" \
      "$job_scratch" \
      "$captured_base" \
      2222222222222222222222222222222222222222 \
      >"$runtime_directory/stdout.$job_id" 2>"$runtime_directory/stderr.$job_id"; then
      actual_status=0
    else
      actual_status=$?
    fi
    if [[ $actual_status != "$expected_status" ]]; then
      echo "Expected OCI capture probe $job_id exit $expected_status, received $actual_status" >&2
      sed -n '1,160p' "$runtime_directory/stderr.$job_id" >&2
      exit 1
    fi
  }

  local first_success_scratch=$runtime_directory/scratch/oci-capture-424243
  local identical_collision_scratch=$runtime_directory/scratch/oci-capture-424244
  local receipt_collision_scratch=$runtime_directory/scratch/oci-capture-424245
  local artifact_collision_scratch=$runtime_directory/scratch/oci-capture-424246
  local published_directory
  local published_count

  run_successful_capture_probe 424243 "$first_success_scratch"
  published_count=$(find "$captured_base" -mindepth 1 -maxdepth 1 -type d -name 'sha256-*' -print | wc -l | tr -d ' ')
  test "$published_count" = 1
  published_directory=$(find "$captured_base" -mindepth 1 -maxdepth 1 -type d -name 'sha256-*' -print)
  test -f "$published_directory/MANIFEST.sha256"
  test -f "$captured_base/runs/424243.json"
  test -z "$(find "$captured_base" -mindepth 1 -maxdepth 2 -name '*.stage' -print -quit)"

  run_successful_capture_probe 424244 "$identical_collision_scratch"
  published_count=$(find "$captured_base" -mindepth 1 -maxdepth 1 -type d -name 'sha256-*' -print | wc -l | tr -d ' ')
  test "$published_count" = 1
  test -f "$captured_base/runs/424244.json"
  test -z "$(find "$captured_base" -mindepth 1 -maxdepth 2 -name '*.stage' -print -quit)"

  printf 'conflicting receipt\n' >"$captured_base/runs/424245.json"
  run_successful_capture_probe 424245 "$receipt_collision_scratch" 1
  test "$(<"$captured_base/runs/424245.json")" = 'conflicting receipt'
  test -z "$(find "$captured_base" -mindepth 1 -maxdepth 2 -name '*.stage' -print -quit)"

  chmod u+w "$published_directory" "$published_directory/source_format_evidence.json"
  printf 'tampered\n' >>"$published_directory/source_format_evidence.json"
  chmod 444 "$published_directory/source_format_evidence.json"
  chmod 555 "$published_directory"
  run_successful_capture_probe 424246 "$artifact_collision_scratch" 1
  test ! -e "$captured_base/runs/424246.json"
  test -z "$(find "$captured_base" -mindepth 1 -maxdepth 2 -name '*.stage' -print -quit)"

  chmod u+w "$published_directory"
  chmod u+w "$published_directory"/*
}

create_oci_capture_submit_probe() {
  local submit_script=$1
  local probe_script=$2
  local semantic_worktree=$3
  local compressed_root=$4
  local modelopt_root=$5
  local transformer_engine_root=$6
  local staged_metadata_root=$7
  local raw_manifest_sha256=$8
  local captured_base=$9
  local scripts_directory=${10}

  awk \
    -v script_dir="script_dir=${scripts_directory}" \
    -v semantic_worktree="readonly SEMANTIC_WORKTREE=${semantic_worktree}" \
    -v compressed_root="readonly COMPRESSED_TENSORS_SOURCE_ROOT=${compressed_root}" \
    -v modelopt_root="readonly MODELOPT_LIGHTNING_SOURCE_ROOT=${modelopt_root}" \
    -v transformer_engine_root="readonly TRANSFORMER_ENGINE_SOURCE_ROOT=${transformer_engine_root}" \
    -v staged_metadata_root="readonly STAGED_METADATA_ROOT=${staged_metadata_root}" \
    -v raw_manifest_sha256="readonly EXPECTED_RAW_MANIFEST_SHA256=${raw_manifest_sha256}" \
    -v captured_base="readonly CAPTURED_BASE=${captured_base}" \
    -v sbatch_command='readonly SBATCH_COMMAND=${TEST_SBATCH_COMMAND:?}' \
    -v oci_slurm_conf='readonly OCI_SLURM_CONF=${TEST_OCI_SLURM_CONF:?}' '
      /^script_dir=/ { print script_dir; next }
      /^readonly SEMANTIC_WORKTREE=/ { print semantic_worktree; next }
      /^readonly COMPRESSED_TENSORS_SOURCE_ROOT=/ { print compressed_root; next }
      /^readonly MODELOPT_LIGHTNING_SOURCE_ROOT=/ { print modelopt_root; next }
      /^readonly TRANSFORMER_ENGINE_SOURCE_ROOT=/ { print transformer_engine_root; next }
      /^readonly STAGED_METADATA_ROOT=/ { print staged_metadata_root; next }
      /^readonly EXPECTED_RAW_MANIFEST_SHA256=/ { print raw_manifest_sha256; next }
      /^readonly CAPTURED_BASE=/ { print captured_base; next }
      /^readonly SBATCH_COMMAND=/ { print sbatch_command; next }
      /^readonly OCI_SLURM_CONF=/ { print oci_slurm_conf; next }
      { print }
    ' "$submit_script" >"$probe_script"
  chmod 755 "$probe_script"
}

create_oci_smoke_submit_probe() {
  local submit_script=$1
  local probe_script=$2
  local scripts_directory=$3

  awk \
    -v script_dir="script_dir=${scripts_directory}" \
    -v sbatch_command='readonly SBATCH_COMMAND=${TEST_SBATCH_COMMAND:?}' \
    -v oci_slurm_conf='readonly OCI_SLURM_CONF=${TEST_OCI_SLURM_CONF:?}' '
      /^script_dir=/ { print script_dir; next }
      /^readonly SBATCH_COMMAND=/ { print sbatch_command; next }
      /^readonly OCI_SLURM_CONF=/ { print oci_slurm_conf; next }
      { print }
    ' "$submit_script" >"$probe_script"
  chmod 755 "$probe_script"
}

run_oci_capture_submit_probe_action() {
  local probe_script=$1
  local capture_prefix=$2
  local action=$3
  shift 3

  touch "$capture_prefix"
  env \
    SBATCH_CAPTURE="$capture_prefix" \
    SBATCH_CAPTURE_WORKING_BATCH="$CAPTURE_BATCH" \
    SBATCH_EXPECTED_BATCH_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sbatch \
    TEST_SBATCH_COMMAND="$STUB_DIRECTORY/sbatch" \
    TEST_OCI_SLURM_CONF="$capture_prefix" \
    PATH="$STUB_DIRECTORY:$PATH" \
    "$@" \
    "$probe_script" "$action"
}

run_oci_capture_submit_probe() {
  local probe_script=$1
  local capture_prefix=$2
  shift 2

  run_oci_capture_submit_probe_action "$probe_script" "$capture_prefix" test-only "$@"
}

test_oci_capture_submit_gates() {
  local runtime_directory=$TEST_DIRECTORY/oci-capture-submit
  local compressed_root=$runtime_directory/compressed-tensors
  local modelopt_root=$runtime_directory/model-optimizer
  local transformer_engine_root=$runtime_directory/transformer-engine
  local staged_metadata_root=$runtime_directory/raw
  local captured_base=$runtime_directory/captured
  local probe_script=$runtime_directory/submit-oci-capture.sh
  local raw_manifest_sha256
  local success_capture=$runtime_directory/success
  local submit_capture=$runtime_directory/submit
  local failure_capture
  local expected_exports

  mkdir -p \
    "$compressed_root" \
    "$modelopt_root" \
    "$transformer_engine_root" \
    "$staged_metadata_root" \
    "$captured_base/logs" \
    "$captured_base/runs"
  create_oci_raw_metadata_fixture "$staged_metadata_root"
  raw_manifest_sha256=$(sha256_file "$staged_metadata_root/SHA256SUMS")
  create_oci_capture_submit_probe \
    "$CAPTURE_SUBMIT" \
    "$probe_script" \
    "$TEST_SEMANTIC_WORKTREE" \
    "$compressed_root" \
    "$modelopt_root" \
    "$transformer_engine_root" \
    "$staged_metadata_root" \
    "$raw_manifest_sha256" \
    "$captured_base" \
    "$SCRIPTS_DIRECTORY"

  export SBATCH_EXPECTED_COMPRESSED_ROOT=$compressed_root
  export SBATCH_EXPECTED_MODELOPT_ROOT=$modelopt_root
  export SBATCH_EXPECTED_TE_ROOT=$transformer_engine_root
  export SBATCH_EXPECTED_STAGED_METADATA_ROOT=$staged_metadata_root
  export SBATCH_COMPRESSED_HEAD_SHA=f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0
  export SBATCH_MODELOPT_HEAD_SHA=c897fbeaaff66d53d61033f107885b7c5432f235
  export SBATCH_TE_HEAD_SHA=42b840051647eef89761a16dfdff87e82bb253ab
  export SBATCH_COMPRESSED_ORIGIN=https://github.com/vllm-project/compressed-tensors.git
  export SBATCH_MODELOPT_ORIGIN=https://github.com/NVIDIA/Model-Optimizer.git
  export SBATCH_TE_ORIGIN=https://github.com/NVIDIA/TransformerEngine.git

  run_oci_capture_submit_probe \
    "$probe_script" \
    "$success_capture" \
    SBATCH_ACCOUNT=hostile-account \
    SBATCH_PARTITION=hostile-partition \
    SBATCH_GPUS=8 \
    SBATCH_GRES=gpu:8 \
    SBATCH_EXCLUSIVE=1 \
    SBATCH_TIME=7-00:00:00 \
    SLURM_CLUSTERS=hostile-cluster \
    SLURM_HINT=nomultithread
  expected_exports="--export=SCRIPT_ROOT=${SCRIPT_ROOT},EXPECTED_TOOLING_SHA=${TEST_TOOLING_HEAD_SHA},SEMANTIC_WORKTREE=${TEST_SEMANTIC_WORKTREE},EXPECTED_REPO_SHA=${TEST_SEMANTIC_HEAD_SHA},COMPRESSED_TENSORS_SOURCE_ROOT=${compressed_root},MODELOPT_LIGHTNING_SOURCE_ROOT=${modelopt_root},TRANSFORMER_ENGINE_SOURCE_ROOT=${transformer_engine_root},STAGED_METADATA_ROOT=${staged_metadata_root}"
  grep -Fx -- 'capture-explicit' "$success_capture.export-mode" >/dev/null
  grep -Fx -- '--test-only' "$success_capture.args" >/dev/null
  grep -Fx -- "--chdir=${SCRIPT_ROOT}" "$success_capture.args" >/dev/null
  grep -Fx -- "$expected_exports" "$success_capture.args" >/dev/null
  cmp -s "$success_capture.script" "$CAPTURE_BATCH"
  assert_sanitized_scheduler_environment "$success_capture.env" "$success_capture"

  run_oci_capture_submit_probe_action \
    "$probe_script" \
    "$submit_capture" \
    submit \
    SBATCH_ACCOUNT=hostile-account \
    SBATCH_PARTITION=hostile-partition \
    SBATCH_GPUS=8 \
    SBATCH_GRES=gpu:8 \
    SBATCH_EXCLUSIVE=1 \
    SBATCH_TIME=7-00:00:00 \
    SLURM_CLUSTERS=hostile-cluster \
    SLURM_HINT=nomultithread
  grep -Fx -- 'capture-explicit' "$submit_capture.export-mode" >/dev/null
  fail_if_present '--test-only' "$submit_capture.args"
  grep -Fx -- "--chdir=${SCRIPT_ROOT}" "$submit_capture.args" >/dev/null
  grep -Fx -- "$expected_exports" "$submit_capture.args" >/dev/null
  cmp -s "$submit_capture.script" "$CAPTURE_BATCH"
  assert_sanitized_scheduler_environment "$submit_capture.env" "$submit_capture"

  failure_capture=$runtime_directory/dirty-tooling
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" SBATCH_TOOLING_STATUS=' M capture-tooling' \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted a dirty tooling worktree' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"

  failure_capture=$runtime_directory/missing-semantic-upstream
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" SBATCH_SEMANTIC_HAS_UPSTREAM=false \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted a semantic worktree without an upstream' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"

  failure_capture=$runtime_directory/divergent-semantic-upstream
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" SBATCH_SEMANTIC_UPSTREAM_SHA=3333333333333333333333333333333333333333 \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted a semantic worktree divergent from upstream' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"

  failure_capture=$runtime_directory/dirty-source
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" SBATCH_SOURCE_STATUS=' M pinned-source' \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted a dirty pinned source tree' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"

  failure_capture=$runtime_directory/divergent-source
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" SBATCH_COMPRESSED_HEAD_SHA=4444444444444444444444444444444444444444 \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted a pinned source at the wrong revision' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"

  failure_capture=$runtime_directory/swapped-source-origin
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" SBATCH_COMPRESSED_ORIGIN=https://github.com/NVIDIA/TransformerEngine.git \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted a pinned source with a swapped origin' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"

  mv -- "$staged_metadata_root/SHA256SUMS" "$staged_metadata_root/SHA256SUMS.missing"
  failure_capture=$runtime_directory/missing-raw-manifest
  if run_oci_capture_submit_probe "$probe_script" "$failure_capture" \
    >"$failure_capture.stdout" 2>"$failure_capture.stderr"; then
    echo 'OCI capture submit accepted raw metadata without its pinned manifest' >&2
    exit 1
  fi
  test ! -e "$failure_capture.args"
  mv -- "$staged_metadata_root/SHA256SUMS.missing" "$staged_metadata_root/SHA256SUMS"

}

test_directory=$(mktemp -d)
readonly TEST_DIRECTORY=$test_directory
readonly STUB_DIRECTORY=$TEST_DIRECTORY/bin
readonly TEST_SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28
readonly TEST_SEMANTIC_HEAD_SHA=1111111111111111111111111111111111111111
TEST_TOOLING_HEAD_SHA=$(git -C "$SCRIPT_ROOT" rev-parse HEAD)
readonly TEST_TOOLING_HEAD_SHA
REAL_GIT=$(command -v git)
readonly REAL_GIT
export REAL_GIT
export SBATCH_EXPECTED_TOOLING_ROOT=$SCRIPT_ROOT
export SBATCH_TOOLING_HAS_UPSTREAM=true
export SBATCH_TOOLING_UPSTREAM_SHA=$TEST_TOOLING_HEAD_SHA
export SBATCH_EXPECTED_SEMANTIC_WORKTREE=$TEST_SEMANTIC_WORKTREE
export SBATCH_SEMANTIC_HAS_UPSTREAM=true
export SBATCH_SEMANTIC_HEAD_SHA=$TEST_SEMANTIC_HEAD_SHA
export SBATCH_SEMANTIC_UPSTREAM_SHA=$TEST_SEMANTIC_HEAD_SHA
mkdir -p "$STUB_DIRECTORY"
trap 'rm -rf -- "$TEST_DIRECTORY"' EXIT

test_ptyche_upload_failure_diagnostics "$PTYCHE_UPLOAD_BATCH"
test_ptyche_cleanup_failure_diagnostics "$PTYCHE_UPLOAD_BATCH"
test_ptyche_rclone_provisioner "$PTYCHE_RCLONE_PROVISIONER"
test_oci_capture_check_failure_does_not_publish

if [[ $PTYCHE_RUNTIME_ONLY = true ]]; then
  printf 'Ptyche upload runtime failure checks passed\n'
  exit 0
fi

for batch_script in "$DOWNLOAD_BATCH" "$SMOKE_BATCH" "$CAPTURE_BATCH"; do
  fail_if_present 'BASH_SOURCE' "$batch_script"
  fail_if_present "test -z \"\$(git " "$batch_script"
  grep -Fq 'SCRIPT_ROOT:?Set SCRIPT_ROOT' "$batch_script"
  grep -Fq 'EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA' "$batch_script"
  grep -Fq 'validate_tooling_root' "$batch_script"
  require_pattern "git -C \"$dollar{SCRIPT_ROOT}\" hash-object" "$batch_script"
done

for capture_script in "$CAPTURE_BATCH" "$CAPTURE_BODY" "$CAPTURE_SUBMIT"; do
  test -x "$capture_script"
  require_pattern '# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.' "$capture_script"
done

require_pattern '#SBATCH --account=nemotron_n3_post' "$CAPTURE_BATCH"
require_pattern '#SBATCH --partition=batch' "$CAPTURE_BATCH"
require_pattern '#SBATCH --nodes=1' "$CAPTURE_BATCH"
require_pattern '#SBATCH --ntasks-per-node=1' "$CAPTURE_BATCH"
require_pattern '#SBATCH --gpus-per-node=1' "$CAPTURE_BATCH"
require_pattern '#SBATCH --mem=32G' "$CAPTURE_BATCH"
require_pattern '#SBATCH --time=00:30:00' "$CAPTURE_BATCH"
fail_if_present '#SBATCH --gpus-per-node=4' "$CAPTURE_BATCH"
fail_if_present '#SBATCH --exclusive' "$CAPTURE_BATCH"
fail_if_present '#SBATCH --mem=0' "$CAPTURE_BATCH"
require_pattern 'readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh' "$CAPTURE_BATCH"
require_pattern 'readonly EXPECTED_CONTAINER_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e' "$CAPTURE_BATCH"
require_pattern 'readonly COMPRESSED_TENSORS_SOURCE_ROOT=/home/sna/nemorl-source-evidence/checkouts/compressed-tensors/sha256-f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0' "$CAPTURE_SUBMIT"
require_pattern 'readonly MODELOPT_LIGHTNING_SOURCE_ROOT=/home/sna/nemorl-source-evidence/checkouts/model-optimizer/sha256-c897fbeaaff66d53d61033f107885b7c5432f235' "$CAPTURE_SUBMIT"
require_pattern 'readonly TRANSFORMER_ENGINE_SOURCE_ROOT=/home/sna/nemorl-source-evidence/checkouts/transformer-engine/sha256-42b840051647eef89761a16dfdff87e82bb253ab' "$CAPTURE_SUBMIT"
require_pattern 'readonly STAGED_METADATA_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw/sha256-d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b' "$CAPTURE_SUBMIT"
require_pattern 'readonly CAPTURED_BASE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured' "$CAPTURE_SUBMIT"
for capture_script in "$CAPTURE_BATCH" "$CAPTURE_BODY" "$CAPTURE_SUBMIT"; do
  require_pattern 'f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0' "$capture_script"
  require_pattern 'c897fbeaaff66d53d61033f107885b7c5432f235' "$capture_script"
  require_pattern '42b840051647eef89761a16dfdff87e82bb253ab' "$capture_script"
  require_pattern 'https://github.com/vllm-project/compressed-tensors.git' "$capture_script"
  require_pattern 'https://github.com/NVIDIA/Model-Optimizer.git' "$capture_script"
  require_pattern 'https://github.com/NVIDIA/TransformerEngine.git' "$capture_script"
  require_pattern 'status --porcelain' "$capture_script"
done
for semantic_source_sha in \
  b11414c71b15e54d333eb49346ed199f20fa9021 \
  1814c6c93a66b9d59d254960ef6a99a64249b671 \
  7c9c3a027c503ae9ae1e8ad7b14397abb8269378; do
  require_pattern "$semantic_source_sha" "$CAPTURE_SUBMIT"
  require_pattern "$semantic_source_sha" "$CAPTURE_BATCH"
  require_pattern "$semantic_source_sha" "$CAPTURE_BODY"
done
require_pattern 'test "${file_count}" = 19' "$CAPTURE_SUBMIT"
require_pattern 'test "${manifest_line_count}" = 19' "$CAPTURE_SUBMIT"
for capture_script in "$CAPTURE_SUBMIT" "$CAPTURE_BATCH" "$CAPTURE_BODY"; do
  require_pattern 'sha256sum --check --strict SHA256SUMS' "$capture_script"
  fail_if_present 'STAGED_METADATA_ROOT}/MANIFEST.sha256' "$capture_script"
done
require_pattern 'readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python' "$CAPTURE_BODY"
require_pattern 'test "${PYTHONPYCACHEPREFIX:-}" = "${SCRATCH_DIRECTORY}/pycache"' "$CAPTURE_BODY"
require_pattern 'readonly VALIDATOR_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/validate_transferred_nightly.sh' "$CAPTURE_BATCH"
require_pattern 'snapshot_tooling_file "${VALIDATOR_RELATIVE_PATH}" "${VALIDATOR_SNAPSHOT}"' "$CAPTURE_BATCH"
require_pattern '"${VALIDATOR_SNAPSHOT}" "${CONTAINER}" "${CONTAINER}.metadata.txt" "${CONTAINER}.complete"' "$CAPTURE_BATCH"
require_pattern 'validator_snapshot_sha256=$(sha256sum "${VALIDATOR_SNAPSHOT}"' "$CAPTURE_BATCH"
require_pattern 'test "${validator_expected_image_sha256}" = "${EXPECTED_CONTAINER_SHA256}"' "$CAPTURE_BATCH"
require_pattern "printf 'validator_snapshot_path=%s\\n'" "$CAPTURE_BATCH"
require_pattern "printf 'validator_snapshot_sha256=%s\\n'" "$CAPTURE_BATCH"
require_pattern "printf 'expected_image_sha256=%s\\n'" "$CAPTURE_BATCH"
fail_if_present 'sha256sum "${CONTAINER}"' "$CAPTURE_BATCH"
require_pattern 'readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e' "$SCRIPTS_DIRECTORY/validate_transferred_nightly.sh"
if [[ $(grep -Fc -- 'sha256sum "${image_path}"' "$SCRIPTS_DIRECTORY/validate_transferred_nightly.sh") != 1 ]]; then
  echo 'Validated-image helper must perform exactly one full image SHA-256 scan' >&2
  exit 1
fi
require_pattern 'export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' "$CAPTURE_BATCH"
require_pattern 'export SLURM_EXPORT_ENV=ALL' "$CAPTURE_BATCH"
require_pattern '--gpus=1' "$CAPTURE_BATCH"
if [[ $(grep -Fc -- '--inspect-runtime' "$CAPTURE_BODY") != 2 ]]; then
  echo 'OCI capture body must inspect the pinned runtime in both capture passes' >&2
  exit 1
fi
if [[ $(grep -Fc -- '  --check \' "$CAPTURE_BODY") != 1 ]]; then
  echo 'OCI capture body must run exactly one read-only check pass' >&2
  exit 1
fi
require_pattern 'source_format_evidence.json' "$CAPTURE_BODY"
require_pattern 'producer_implementations.json' "$CAPTURE_BODY"
require_pattern 'MANIFEST.sha256' "$CAPTURE_BODY"
require_pattern 'SHA256SUMS' "$CAPTURE_BODY"
require_pattern 'mv -Tn -- "${PUBLISH_STAGE_DIRECTORY}" "${PUBLISH_DIRECTORY}"' "$CAPTURE_BODY"
require_pattern 'chmod u+w "${PUBLISH_STAGE_DIRECTORY}"' "$CAPTURE_BODY"
fail_if_present 'tests/fixtures/precision_policy' "$CAPTURE_BODY"
fail_if_present 'torch.cuda' "$CAPTURE_BODY"
fail_if_present 'nvidia-smi' "$CAPTURE_BODY"
require_pattern '--export="${EXPORTS}"' "$CAPTURE_SUBMIT"
fail_if_present '--export="ALL,' "$CAPTURE_SUBMIT"
require_pattern 'readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch' "$CAPTURE_SUBMIT"
require_pattern 'readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf' "$CAPTURE_SUBMIT"
require_pattern '| /usr/bin/env -i' "$CAPTURE_SUBMIT"
require_pattern 'PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin' "$CAPTURE_SUBMIT"
require_pattern 'SLURM_CONF="${OCI_SLURM_CONF}"' "$CAPTURE_SUBMIT"
fail_if_present '| sbatch' "$CAPTURE_SUBMIT"
fail_if_present 'export HOME=' "$CAPTURE_SUBMIT"
if [[ $(grep -Fc -- '| /usr/bin/env -i' "$CAPTURE_SUBMIT") != 2 ]]; then
  echo 'OCI capture submit must sanitize both test-only and submit scheduler invocations' >&2
  exit 1
fi
if [[ $(grep -Fc -- 'EXPECTED_REPO_SHA=$(git -C "${SEMANTIC_WORKTREE}" rev-parse HEAD)' "$CAPTURE_SUBMIT") != 1 ]]; then
  echo 'OCI capture submit must snapshot semantic HEAD exactly once' >&2
  exit 1
fi
require_pattern 'local probe_script=$runtime_directory/submit-oci-capture.sh' "$0"

capture_snapshot_line=$(line_number 'snapshot_tooling_file "${VALIDATOR_RELATIVE_PATH}"' 1 "$CAPTURE_BATCH")
capture_validator_line=$(line_number '"${VALIDATOR_SNAPSHOT}" "${CONTAINER}"' 1 "$CAPTURE_BATCH")
capture_export_line=$(line_number '^export SLURM_EXPORT_ENV=ALL$' 1 "$CAPTURE_BATCH")
capture_srun_line=$(line_number '/cm/local/apps/slurm/current/bin/srun' 1 "$CAPTURE_BATCH")
capture_first_pass_line=$(line_number 'PYTHONPATH="${SEMANTIC_WORKTREE}" "${MAIN_PYTHON}" "${CAPTURE_TOOL}"' 1 "$CAPTURE_BODY")
capture_check_pass_line=$(line_number 'PYTHONPATH="${SEMANTIC_WORKTREE}" "${MAIN_PYTHON}" "${CAPTURE_TOOL}"' 2 "$CAPTURE_BODY")
capture_manifest_line=$(line_number 'sha256sum producer_implementations.json source_format_evidence.json' 1 "$CAPTURE_BODY")
capture_publish_line=$(line_number 'mv -Tn -- "${PUBLISH_STAGE_DIRECTORY}" "${PUBLISH_DIRECTORY}"' 1 "$CAPTURE_BODY")
test "$capture_snapshot_line" -lt "$capture_export_line"
test "$capture_export_line" -lt "$capture_srun_line"
test "$capture_validator_line" -lt "$capture_srun_line"
test "$capture_first_pass_line" -lt "$capture_check_pass_line"
test "$capture_check_pass_line" -lt "$capture_manifest_line"
test "$capture_manifest_line" -lt "$capture_publish_line"

grep -Fq 'VALIDATOR_SNAPSHOT' "$DOWNLOAD_BATCH"
grep -Fq 'snapshot_validator' "$DOWNLOAD_BATCH"
grep -Fq 'VALIDATOR_SNAPSHOT' "$SMOKE_BATCH"
grep -Fq 'SMOKE_BODY_SNAPSHOT' "$SMOKE_BATCH"
grep -Fq 'snapshot_tooling_file' "$SMOKE_BATCH"
grep -Fq '#SBATCH --output=/lustre/' "$SMOKE_BATCH"
grep -Fq '#SBATCH --error=/lustre/' "$SMOKE_BATCH"
require_pattern '#SBATCH --partition=batch' "$SMOKE_BATCH"
fail_if_present '#SBATCH --partition=gpu' "$SMOKE_BATCH"
grep -Fq 'readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh' "$SMOKE_BATCH"
fail_if_present "CONTAINER=$dollar{CONTAINER:-" "$SMOKE_BATCH"
require_pattern ": \"${dollar}{SEMANTIC_WORKTREE:?Set SEMANTIC_WORKTREE to the absolute clean semantic policy worktree root}\"" "$SMOKE_BATCH"
require_pattern ": \"${dollar}{EXPECTED_REPO_SHA:?Set EXPECTED_REPO_SHA to the semantic worktree HEAD}\"" "$SMOKE_BATCH"
fail_if_present 'readonly SEMANTIC_WORKTREE=' "$SMOKE_BATCH"
fail_if_present "EXPECTED_REPO_SHA=$dollar{EXPECTED_REPO_SHA:-" "$SMOKE_BATCH"
fail_if_present '6d69f234aed4e0dfb2219308dd3160f55edf5480' "$SMOKE_BATCH"
require_pattern 'readonly SEMANTIC_WORKTREE=/home/sna/nemorl-semantic-precision-test-597c93b28' "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SCRIPT_ROOT}\" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SCRIPT_ROOT}\" rev-parse '@{upstream}'" "$SMOKE_SUBMIT"
require_pattern "test \"${dollar}{EXPECTED_TOOLING_SHA}\" = \"${dollar}{TOOLING_UPSTREAM_SHA}\"" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SCRIPT_ROOT}\" status --porcelain" "$SMOKE_SUBMIT"
require_pattern "test -z \"$dollar{worktree_status}\"" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SEMANTIC_WORKTREE}\" rev-parse --is-inside-work-tree" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SEMANTIC_WORKTREE}\" rev-parse --show-toplevel" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SEMANTIC_WORKTREE}\" status --porcelain" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SEMANTIC_WORKTREE}\" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SEMANTIC_WORKTREE}\" rev-parse HEAD" "$SMOKE_SUBMIT"
require_pattern "git -C \"$dollar{SEMANTIC_WORKTREE}\" rev-parse '@{upstream}'" "$SMOKE_SUBMIT"
require_pattern "test \"${dollar}{EXPECTED_REPO_SHA}\" = \"${dollar}{SEMANTIC_UPSTREAM_SHA}\"" "$SMOKE_SUBMIT"
require_pattern "--export=\"SCRIPT_ROOT=$dollar{SCRIPT_ROOT},EXPECTED_TOOLING_SHA=$dollar{EXPECTED_TOOLING_SHA},SEMANTIC_WORKTREE=$dollar{SEMANTIC_WORKTREE},EXPECTED_REPO_SHA=$dollar{EXPECTED_REPO_SHA}\"" "$SMOKE_SUBMIT"
fail_if_present '--export="ALL,' "$SMOKE_SUBMIT"
require_pattern 'readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch' "$SMOKE_SUBMIT"
require_pattern 'readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf' "$SMOKE_SUBMIT"
require_pattern '| /usr/bin/env -i' "$SMOKE_SUBMIT"
require_pattern 'PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin' "$SMOKE_SUBMIT"
require_pattern 'SLURM_CONF="${OCI_SLURM_CONF}"' "$SMOKE_SUBMIT"
fail_if_present '| sbatch' "$SMOKE_SUBMIT"
fail_if_present 'export HOME=' "$SMOKE_SUBMIT"
if [[ $(grep -Fc -- '| /usr/bin/env -i' "$SMOKE_SUBMIT") != 2 ]]; then
  echo 'OCI smoke submit must sanitize both test-only and submit scheduler invocations' >&2
  exit 1
fi
fail_if_present '6d69f234aed4e0dfb2219308dd3160f55edf5480' "$SMOKE_SUBMIT"

smoke_tooling_sha=$(git -C "$SCRIPT_ROOT" rev-parse HEAD)
missing_semantic_worktree_stderr=$TEST_DIRECTORY/missing-semantic-worktree.stderr
if env -i \
  PATH="$PATH" \
  SCRIPT_ROOT="$SCRIPT_ROOT" \
  EXPECTED_TOOLING_SHA="$smoke_tooling_sha" \
  /bin/bash "$SMOKE_BATCH" >"$TEST_DIRECTORY/missing-semantic-worktree.stdout" 2>"$missing_semantic_worktree_stderr"; then
  echo 'OCI-Hsg smoke batch accepted a missing SEMANTIC_WORKTREE' >&2
  exit 1
fi
require_pattern 'SEMANTIC_WORKTREE' "$missing_semantic_worktree_stderr"

missing_repo_sha_stderr=$TEST_DIRECTORY/missing-expected-repo-sha.stderr
if env -i \
  PATH="$PATH" \
  SCRIPT_ROOT="$SCRIPT_ROOT" \
  EXPECTED_TOOLING_SHA="$smoke_tooling_sha" \
  SEMANTIC_WORKTREE=/unreachable-semantic-worktree-fixture \
  /bin/bash "$SMOKE_BATCH" >"$TEST_DIRECTORY/missing-expected-repo-sha.stdout" 2>"$missing_repo_sha_stderr"; then
  echo 'OCI-Hsg smoke batch accepted a missing EXPECTED_REPO_SHA' >&2
  exit 1
fi
require_pattern 'EXPECTED_REPO_SHA' "$missing_repo_sha_stderr"

grep -Fq 'readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python' "$SMOKE_BODY"
fail_if_present "MAIN_PYTHON=$dollar{MAIN_PYTHON:-" "$SMOKE_BODY"
fail_if_present "test -z \"\$(git " "$SMOKE_BODY"
grep -Fq "PYTHONPYCACHEPREFIX=$dollar{SCRATCH_DIRECTORY}/pycache" "$SMOKE_BATCH"
grep -Fq "XDG_CACHE_HOME=$dollar{SCRATCH_DIRECTORY}/xdg-cache" "$SMOKE_BATCH"
grep -Fq "UV_CACHE_DIR=$dollar{SCRATCH_DIRECTORY}/uv-cache" "$SMOKE_BATCH"
grep -Fq "TORCHINDUCTOR_CACHE_DIR=$dollar{SCRATCH_DIRECTORY}/torchinductor-cache" "$SMOKE_BATCH"
grep -Fq "TRITON_CACHE_DIR=$dollar{SCRATCH_DIRECTORY}/triton-cache" "$SMOKE_BATCH"
require_pattern 'export SLURM_EXPORT_ENV=ALL' "$SMOKE_BATCH"

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
smoke_step_export_line=$(line_number '^export SLURM_EXPORT_ENV=ALL$' 1 "$SMOKE_BATCH")
smoke_srun_line=$(line_number '/cm/local/apps/slurm/current/bin/srun' 1 "$SMOKE_BATCH")
test "$smoke_root_validation_line" -lt "$smoke_snapshot_line"
test "$smoke_snapshot_line" -lt "$smoke_validator_line"
test "$smoke_validator_line" -lt "$smoke_srun_line"
test "$smoke_step_export_line" -lt "$smoke_srun_line"

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
require_pattern 'Final object promotion returned nonzero; verifying immutable final bytes' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=upload-job-temporary-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=inspect-job-temporary-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=download-and-hash-job-temporary-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=promote-job-temporary-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=inspect-final-remote-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=download-and-hash-final-object' "$PTYCHE_UPLOAD_BATCH"
require_pattern 'CURRENT_STEP=describe-final-remote-object' "$PTYCHE_UPLOAD_BATCH"
if [[ $(grep -Fc -- "\"$dollar{RCLONE}\" copyto" "$PTYCHE_UPLOAD_BATCH") != 4 ]]; then
  echo 'Every Ptyche transfer must use exact-object copyto with the pinned rclone path' >&2
  exit 1
fi
fail_if_present "\"$dollar{RCLONE}\" copy \\" "$PTYCHE_UPLOAD_BATCH"
require_pattern "  \"$dollar{SOURCE}\" \\" "$PTYCHE_UPLOAD_BATCH"
require_pattern "  \"$dollar{TEMP_OBJECT}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "  \"$dollar{TEMP_OBJECT}\" \\" "$PTYCHE_UPLOAD_BATCH"
require_pattern "  \"$dollar{TEMP_DOWNLOAD}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "  \"$dollar{DESTINATION}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "  \"$dollar{FINAL_DOWNLOAD}\"" "$PTYCHE_UPLOAD_BATCH"
require_pattern "\"$dollar{RCLONE}\" purge" "$PTYCHE_UPLOAD_BATCH"
require_pattern "\"$dollar{RCLONE}\" lsjson" "$PTYCHE_UPLOAD_BATCH"
if [[ $(grep -Fc -- "\"$dollar{RCLONE}\" lsf" "$PTYCHE_UPLOAD_BATCH") != 2 ]]; then
  echo 'Temporary and final remote objects must both have exact lsf path/size checks' >&2
  exit 1
fi
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
rclone_first_remote_line=$(line_number '^CURRENT_STEP=upload-job-temporary-object$' 1 "$PTYCHE_UPLOAD_BATCH")
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

if [[ -n ${SBATCH_EXPECTED_TOOLING_ROOT:-} && ${1:-} == -C && ${2:-} == "$SBATCH_EXPECTED_TOOLING_ROOT" ]]; then
  shift 2
  case "$*" in
    'rev-parse --abbrev-ref --symbolic-full-name @{upstream}')
      if [[ ${SBATCH_TOOLING_HAS_UPSTREAM:-false} != true ]]; then
        exit 128
      fi
      printf 'fork/staging-validation\n'
      exit 0
      ;;
    'rev-parse @{upstream}')
      printf '%s\n' "$SBATCH_TOOLING_UPSTREAM_SHA"
      exit 0
      ;;
    'status --porcelain')
      printf '%s' "${SBATCH_TOOLING_STATUS:-}"
      exit 0
      ;;
    show\ *:experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sbatch)
      test -n "${SBATCH_CAPTURE_WORKING_BATCH:-}"
      cat "$SBATCH_CAPTURE_WORKING_BATCH"
      exit 0
      ;;
    *)
      set -- -C "$SBATCH_EXPECTED_TOOLING_ROOT" "$@"
      ;;
  esac
fi

if [[ -n ${SBATCH_EXPECTED_COMPRESSED_ROOT:-} && ${1:-} == -C ]]; then
  source_root=$2
  source_expected_sha=
  source_expected_origin=
  case $source_root in
    "$SBATCH_EXPECTED_COMPRESSED_ROOT")
      source_expected_sha=${SBATCH_COMPRESSED_HEAD_SHA}
      source_expected_origin=${SBATCH_COMPRESSED_ORIGIN}
      ;;
    "$SBATCH_EXPECTED_MODELOPT_ROOT")
      source_expected_sha=${SBATCH_MODELOPT_HEAD_SHA}
      source_expected_origin=${SBATCH_MODELOPT_ORIGIN}
      ;;
    "$SBATCH_EXPECTED_TE_ROOT")
      source_expected_sha=${SBATCH_TE_HEAD_SHA}
      source_expected_origin=${SBATCH_TE_ORIGIN}
      ;;
  esac
  if [[ -n $source_expected_sha ]]; then
    shift 2
    case "$*" in
      'rev-parse --is-inside-work-tree') printf 'true\n' ;;
      'rev-parse --show-toplevel') printf '%s\n' "$source_root" ;;
      'rev-parse HEAD') printf '%s\n' "$source_expected_sha" ;;
      'status --porcelain') printf '%s' "${SBATCH_SOURCE_STATUS:-}" ;;
      'config --get-all remote.origin.url') printf '%s\n' "$source_expected_origin" ;;
      *)
        printf 'unexpected source-root git invocation: %s\n' "$*" >&2
        exit 96
        ;;
    esac
    exit 0
  fi
fi

if [[ -n ${SBATCH_EXPECTED_SEMANTIC_WORKTREE:-} && ${1:-} == -C ]]; then
  case ${2:-} in
    "$SBATCH_EXPECTED_SEMANTIC_WORKTREE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM")
      test "${3:-} ${4:-}" = 'rev-parse HEAD'
      printf '%s\n' 7c9c3a027c503ae9ae1e8ad7b14397abb8269378
      exit 0
      ;;
    "$SBATCH_EXPECTED_SEMANTIC_WORKTREE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge")
      test "${3:-} ${4:-}" = 'rev-parse HEAD'
      printf '%s\n' b11414c71b15e54d333eb49346ed199f20fa9021
      exit 0
      ;;
    "$SBATCH_EXPECTED_SEMANTIC_WORKTREE/3rdparty/Automodel-workspace/Automodel")
      test "${3:-} ${4:-}" = 'rev-parse HEAD'
      printf '%s\n' 1814c6c93a66b9d59d254960ef6a99a64249b671
      exit 0
      ;;
  esac
fi

if [[ -n ${SBATCH_EXPECTED_SEMANTIC_WORKTREE:-} && ${1:-} == -C && ${2:-} == "$SBATCH_EXPECTED_SEMANTIC_WORKTREE" ]]; then
  shift 2
  case "$*" in
    'rev-parse --is-inside-work-tree')
      printf 'true\n'
      ;;
    'rev-parse --show-toplevel')
      printf '%s\n' "$SBATCH_EXPECTED_SEMANTIC_WORKTREE"
      ;;
    'status --porcelain')
      printf '%s' "${SBATCH_SEMANTIC_STATUS:-}"
      ;;
    'rev-parse --abbrev-ref --symbolic-full-name @{upstream}')
      if [[ ${SBATCH_SEMANTIC_HAS_UPSTREAM:-false} != true ]]; then
        exit 128
      fi
      printf 'fork/semantic-validation\n'
      ;;
    'rev-parse HEAD')
      printf '%s\n' "$SBATCH_SEMANTIC_HEAD_SHA"
      ;;
    'rev-parse @{upstream}')
      printf '%s\n' "$SBATCH_SEMANTIC_UPSTREAM_SHA"
      ;;
    *)
      printf 'unexpected semantic-worktree git invocation: %s\n' "$*" >&2
      exit 96
      ;;
  esac
  exit 0
fi

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

capture_prefix=${SBATCH_CAPTURE:-${SLURM_CONF:?}}
seen_chdir=0
seen_export=0
seen_test_only=0
export_mode=
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
      export_mode=all
      seen_export=1
      ;;
    --export=SCRIPT_ROOT=*,EXPECTED_TOOLING_SHA=*,SEMANTIC_WORKTREE=*,EXPECTED_REPO_SHA=*,COMPRESSED_TENSORS_SOURCE_ROOT=*,MODELOPT_LIGHTNING_SOURCE_ROOT=*,TRANSFORMER_ENGINE_SOURCE_ROOT=*,STAGED_METADATA_ROOT=*)
      if (( seen_export )); then
        echo 'Duplicate --export' >&2
        exit 2
      fi
      export_payload=${argument#--export=SCRIPT_ROOT=}
      export_root=${export_payload%%,EXPECTED_TOOLING_SHA=*}
      export_payload=${export_payload#*,EXPECTED_TOOLING_SHA=}
      expected_sha=${export_payload%%,SEMANTIC_WORKTREE=*}
      export_payload=${export_payload#*,SEMANTIC_WORKTREE=}
      semantic_worktree=${export_payload%%,EXPECTED_REPO_SHA=*}
      export_payload=${export_payload#*,EXPECTED_REPO_SHA=}
      expected_repo_sha=${export_payload%%,COMPRESSED_TENSORS_SOURCE_ROOT=*}
      export_payload=${export_payload#*,COMPRESSED_TENSORS_SOURCE_ROOT=}
      compressed_root=${export_payload%%,MODELOPT_LIGHTNING_SOURCE_ROOT=*}
      export_payload=${export_payload#*,MODELOPT_LIGHTNING_SOURCE_ROOT=}
      modelopt_root=${export_payload%%,TRANSFORMER_ENGINE_SOURCE_ROOT=*}
      export_payload=${export_payload#*,TRANSFORMER_ENGINE_SOURCE_ROOT=}
      transformer_engine_root=${export_payload%%,STAGED_METADATA_ROOT=*}
      staged_metadata_root=${export_payload#*,STAGED_METADATA_ROOT=}
      if [[ "$export_root" != "$script_root" ]]; then
        echo 'Mismatched --chdir and SCRIPT_ROOT export' >&2
        exit 2
      fi
      if [[ -n ${SBATCH_EXPECTED_SEMANTIC_WORKTREE:-} && ( "$semantic_worktree" != "$SBATCH_EXPECTED_SEMANTIC_WORKTREE" || "$expected_repo_sha" != "$SBATCH_SEMANTIC_HEAD_SHA" ) ]]; then
        echo 'Mismatched semantic capture export' >&2
        exit 2
      fi
      if [[ -n ${SBATCH_EXPECTED_COMPRESSED_ROOT:-} && ( "$compressed_root" != "$SBATCH_EXPECTED_COMPRESSED_ROOT" || "$modelopt_root" != "$SBATCH_EXPECTED_MODELOPT_ROOT" || "$transformer_engine_root" != "$SBATCH_EXPECTED_TE_ROOT" ) ]]; then
        echo 'Mismatched source-root capture export' >&2
        exit 2
      fi
      if [[ -n ${SBATCH_EXPECTED_STAGED_METADATA_ROOT:-} && "$staged_metadata_root" != "$SBATCH_EXPECTED_STAGED_METADATA_ROOT" ]]; then
        echo 'Mismatched staged-metadata capture export' >&2
        exit 2
      fi
      export_mode=capture-explicit
      seen_export=1
      ;;
    --export=SCRIPT_ROOT=*,EXPECTED_TOOLING_SHA=*,SEMANTIC_WORKTREE=*,EXPECTED_REPO_SHA=*)
      if (( seen_export )); then
        echo 'Duplicate --export' >&2
        exit 2
      fi
      export_payload=${argument#--export=SCRIPT_ROOT=}
      export_root=${export_payload%%,EXPECTED_TOOLING_SHA=*}
      export_payload=${export_payload#*,EXPECTED_TOOLING_SHA=}
      expected_sha=${export_payload%%,SEMANTIC_WORKTREE=*}
      export_payload=${export_payload#*,SEMANTIC_WORKTREE=}
      semantic_worktree=${export_payload%%,EXPECTED_REPO_SHA=*}
      expected_repo_sha=${export_payload#*,EXPECTED_REPO_SHA=}
      if [[ "$export_root" != "$script_root" ]]; then
        echo 'Mismatched --chdir and SCRIPT_ROOT export' >&2
        exit 2
      fi
      if [[ -n ${SBATCH_EXPECTED_SEMANTIC_WORKTREE:-} && "$semantic_worktree" != "$SBATCH_EXPECTED_SEMANTIC_WORKTREE" ]]; then
        echo 'Mismatched SEMANTIC_WORKTREE export' >&2
        exit 2
      fi
      if [[ -n ${SBATCH_SEMANTIC_HEAD_SHA:-} && "$expected_repo_sha" != "$SBATCH_SEMANTIC_HEAD_SHA" ]]; then
        echo 'Mismatched EXPECTED_REPO_SHA export' >&2
        exit 2
      fi
      export_mode=semantic-explicit
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

printf '%s\n' "$@" >"$capture_prefix.args"
printf '%s\n' "$export_mode" >"$capture_prefix.export-mode"
env | LC_ALL=C sort >"$capture_prefix.env"
cat >"$capture_prefix.script"
if [[ -n ${SBATCH_EXPECTED_BATCH_RELATIVE_PATH:-} ]]; then
  if [[ $SBATCH_EXPECTED_BATCH_RELATIVE_PATH == experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sbatch ]]; then
    cp "$SBATCH_CAPTURE_WORKING_BATCH" "$capture_prefix.expected"
  else
    "$REAL_GIT" -C "$script_root" show "${expected_sha}:${SBATCH_EXPECTED_BATCH_RELATIVE_PATH}" >"$capture_prefix.expected"
  fi
  if ! cmp -s "$capture_prefix.script" "$capture_prefix.expected"; then
    echo 'sbatch stdin differs from the expected immutable batch blob' >&2
    exit 1
  fi
fi
EOF
chmod 755 "$STUB_DIRECTORY/git" "$STUB_DIRECTORY/mkdir" "$STUB_DIRECTORY/sbatch"

test_oci_capture_submit_gates

readonly SMOKE_SUBMIT_PROBE=$TEST_DIRECTORY/submit-oci-smoke.sh
create_oci_smoke_submit_probe "$SMOKE_SUBMIT" "$SMOKE_SUBMIT_PROBE" "$SCRIPTS_DIRECTORY"

run_validated_nightly_submit_probe() {
  local wrapper=$1
  local capture_prefix=$2
  local batch_relative_path=$3
  local action=$4
  local wrapper_command=$wrapper
  local -a command_arguments
  local -a invocation_environment
  shift 4

  invocation_environment=(
    "SBATCH_CAPTURE=$capture_prefix"
    "SBATCH_EXPECTED_BATCH_RELATIVE_PATH=$batch_relative_path"
    "PATH=$STUB_DIRECTORY:$PATH"
  )
  if [[ $wrapper == "$SMOKE_SUBMIT" ]]; then
    wrapper_command=$SMOKE_SUBMIT_PROBE
    touch "$capture_prefix"
    invocation_environment+=(
      "TEST_SBATCH_COMMAND=$STUB_DIRECTORY/sbatch"
      "TEST_OCI_SLURM_CONF=$capture_prefix"
    )
  fi
  command_arguments=("$wrapper_command")
  if [[ $action != default ]]; then
    command_arguments+=("$action")
  fi

  env \
    "${invocation_environment[@]}" \
    "$@" \
    "${command_arguments[@]}"
}

assert_wrapper_call() {
  local capture_prefix=$1
  local expect_test_only=$2
  local expect_semantic_export=$3
  local expected_batch_path=$4
  local expected_sha

  expected_sha=$(git -C "$SCRIPT_ROOT" rev-parse HEAD)
  grep -Fx -- "--chdir=$SCRIPT_ROOT" "$capture_prefix.args" >/dev/null
  if [[ $expect_semantic_export == true ]]; then
    grep -Fx -- "--export=SCRIPT_ROOT=$SCRIPT_ROOT,EXPECTED_TOOLING_SHA=$expected_sha,SEMANTIC_WORKTREE=$TEST_SEMANTIC_WORKTREE,EXPECTED_REPO_SHA=$TEST_SEMANTIC_HEAD_SHA" "$capture_prefix.args" >/dev/null
    grep -Fx -- 'semantic-explicit' "$capture_prefix.export-mode" >/dev/null
  else
    grep -Fx -- "--export=ALL,SCRIPT_ROOT=$SCRIPT_ROOT,EXPECTED_TOOLING_SHA=$expected_sha" "$capture_prefix.args" >/dev/null
    grep -Fx -- 'all' "$capture_prefix.export-mode" >/dev/null
  fi
  test -s "$capture_prefix.script"
  if [[ -e $capture_prefix.expected ]]; then
    cmp -s "$capture_prefix.script" "$capture_prefix.expected"
  else
    cmp -s "$capture_prefix.script" "$expected_batch_path"
  fi
  if [[ "$expect_test_only" = true ]]; then
    grep -Fx -- '--test-only' "$capture_prefix.args" >/dev/null
  else
    fail_if_present '--test-only' "$capture_prefix.args"
  fi
}

for wrapper in "$SCRIPTS_DIRECTORY"/submit_oci_hsg_*_validated_nightly.sh; do
  expect_semantic_export=false
  if [[ $wrapper == "$SMOKE_SUBMIT" ]]; then
    expect_semantic_export=true
  fi
  grep -Fq "readonly ACTION=$dollar{1:-test-only}" "$wrapper"
  fail_if_present 'ACTION:-' "$wrapper"

  batch_relative_path=$(grep -F 'readonly BATCH_RELATIVE_PATH=' "$wrapper" | cut -d= -f2-)
  test -n "$batch_relative_path"

  no_argument_capture=$TEST_DIRECTORY/$(basename "$wrapper").no-argument
  run_validated_nightly_submit_probe \
    "$wrapper" \
    "$no_argument_capture" \
    "$batch_relative_path" \
    default \
    SBATCH_ACCOUNT=hostile-account \
    SBATCH_PARTITION=hostile-partition \
    SBATCH_GPUS=8 \
    SBATCH_GRES=gpu:8 \
    SBATCH_EXCLUSIVE=1 \
    SBATCH_TIME=7-00:00:00 \
    SLURM_CLUSTERS=hostile-cluster \
    SLURM_HINT=nomultithread \
    ACTION=submit
  assert_wrapper_call "$no_argument_capture" true "$expect_semantic_export" "$SCRIPT_ROOT/$batch_relative_path"
  if [[ $wrapper == "$SMOKE_SUBMIT" ]]; then
    assert_sanitized_scheduler_environment "$no_argument_capture.env" "$no_argument_capture"
  fi

  inherited_action_capture=$TEST_DIRECTORY/$(basename "$wrapper").inherited-action
  run_validated_nightly_submit_probe "$wrapper" "$inherited_action_capture" "$batch_relative_path" default ACTION=submit
  assert_wrapper_call "$inherited_action_capture" true "$expect_semantic_export" "$SCRIPT_ROOT/$batch_relative_path"

  explicit_test_only_capture=$TEST_DIRECTORY/$(basename "$wrapper").test-only
  run_validated_nightly_submit_probe "$wrapper" "$explicit_test_only_capture" "$batch_relative_path" test-only ACTION=submit
  assert_wrapper_call "$explicit_test_only_capture" true "$expect_semantic_export" "$SCRIPT_ROOT/$batch_relative_path"

  submit_capture=$TEST_DIRECTORY/$(basename "$wrapper").submit
  run_validated_nightly_submit_probe \
    "$wrapper" \
    "$submit_capture" \
    "$batch_relative_path" \
    submit \
    SBATCH_ACCOUNT=hostile-account \
    SBATCH_PARTITION=hostile-partition \
    SBATCH_GPUS=8 \
    SBATCH_GRES=gpu:8 \
    SBATCH_EXCLUSIVE=1 \
    SBATCH_TIME=7-00:00:00 \
    SLURM_CLUSTERS=hostile-cluster \
    SLURM_HINT=nomultithread \
    ACTION=test-only
  assert_wrapper_call "$submit_capture" false "$expect_semantic_export" "$SCRIPT_ROOT/$batch_relative_path"
  if [[ $wrapper == "$SMOKE_SUBMIT" ]]; then
    assert_sanitized_scheduler_environment "$submit_capture.env" "$submit_capture"
  fi
done

stale_tooling_upstream_capture=$TEST_DIRECTORY/submit_oci_hsg_smoke_validated_nightly.sh.stale-tooling-upstream
if run_validated_nightly_submit_probe \
  "$SMOKE_SUBMIT" \
  "$stale_tooling_upstream_capture" \
  experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch \
  test-only \
  SBATCH_TOOLING_UPSTREAM_SHA=2222222222222222222222222222222222222222 \
  >"$stale_tooling_upstream_capture.stdout" 2>"$stale_tooling_upstream_capture.stderr"; then
  echo 'OCI-Hsg smoke submit accepted a staging worktree ahead of or behind its upstream' >&2
  exit 1
fi
test ! -e "$stale_tooling_upstream_capture.args"

missing_tooling_upstream_capture=$TEST_DIRECTORY/submit_oci_hsg_smoke_validated_nightly.sh.missing-tooling-upstream
if run_validated_nightly_submit_probe \
  "$SMOKE_SUBMIT" \
  "$missing_tooling_upstream_capture" \
  experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch \
  test-only \
  SBATCH_TOOLING_HAS_UPSTREAM=false \
  >"$missing_tooling_upstream_capture.stdout" 2>"$missing_tooling_upstream_capture.stderr"; then
  echo 'OCI-Hsg smoke submit accepted a staging worktree without an upstream' >&2
  exit 1
fi
test ! -e "$missing_tooling_upstream_capture.args"

dirty_tooling_capture=$TEST_DIRECTORY/submit_oci_hsg_smoke_validated_nightly.sh.dirty-tooling
if run_validated_nightly_submit_probe \
  "$SMOKE_SUBMIT" \
  "$dirty_tooling_capture" \
  experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch \
  test-only \
  SBATCH_TOOLING_STATUS=' M validation-tooling' \
  >"$dirty_tooling_capture.stdout" 2>"$dirty_tooling_capture.stderr"; then
  echo 'OCI-Hsg smoke submit accepted a dirty staging worktree' >&2
  exit 1
fi
test ! -e "$dirty_tooling_capture.args"

stale_upstream_capture=$TEST_DIRECTORY/submit_oci_hsg_smoke_validated_nightly.sh.stale-upstream
if run_validated_nightly_submit_probe \
  "$SMOKE_SUBMIT" \
  "$stale_upstream_capture" \
  experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch \
  test-only \
  SBATCH_SEMANTIC_UPSTREAM_SHA=2222222222222222222222222222222222222222 \
  >"$stale_upstream_capture.stdout" 2>"$stale_upstream_capture.stderr"; then
  echo 'OCI-Hsg smoke submit accepted a semantic worktree ahead of or behind its upstream' >&2
  exit 1
fi
test ! -e "$stale_upstream_capture.args"

missing_upstream_capture=$TEST_DIRECTORY/submit_oci_hsg_smoke_validated_nightly.sh.missing-upstream
if run_validated_nightly_submit_probe \
  "$SMOKE_SUBMIT" \
  "$missing_upstream_capture" \
  experiments/pr3652_validation_container/scripts/oci_hsg_smoke_validated_nightly.sbatch \
  test-only \
  SBATCH_SEMANTIC_HAS_UPSTREAM=false \
  >"$missing_upstream_capture.stdout" 2>"$missing_upstream_capture.stderr"; then
  echo 'OCI-Hsg smoke submit accepted a semantic worktree without an upstream' >&2
  exit 1
fi
test ! -e "$missing_upstream_capture.args"

printf 'validation tooling static checks passed\n'
