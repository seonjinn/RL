#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_ROOT=$(git -C "$script_dir" rev-parse --show-toplevel)
readonly SCRIPT_ROOT
readonly SCRIPTS_DIRECTORY=$SCRIPT_ROOT/experiments/pr3652_validation_container/scripts
readonly DOWNLOAD_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_download_validated_nightly.sbatch
readonly SMOKE_BATCH=$SCRIPTS_DIRECTORY/oci_hsg_smoke_validated_nightly.sbatch
readonly SMOKE_BODY=$SCRIPTS_DIRECTORY/oci_hsg_smoke_validated_nightly.sh
readonly PTYCHE_UPLOAD_BATCH=$SCRIPTS_DIRECTORY/ptyche_upload_validated_nightly.sbatch
readonly dollar='$'

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

test_directory=$(mktemp -d)
readonly TEST_DIRECTORY=$test_directory
readonly STUB_DIRECTORY=$TEST_DIRECTORY/bin
REAL_GIT=$(command -v git)
readonly REAL_GIT
export REAL_GIT
mkdir -p "$STUB_DIRECTORY"
trap 'rm -rf -- "$TEST_DIRECTORY"' EXIT

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
