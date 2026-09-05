#!/bin/bash

set -euo pipefail

die() {
  echo "$*" >&2
  exit 1
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIR=$script_dir
readonly BATCH=${SCRIPT_DIR}/oci_hsg_capture_precision_source_evidence.sbatch
real_git=$(command -v git) || die 'git is required for the capture batch probe'
readonly REAL_GIT=$real_git

sha256_file() {
  local path=$1

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
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
  else
    real_sha256_tool=$(command -v shasum)
    cat >"$stub_path" <<EOF
#!/bin/bash
exec "$real_sha256_tool" -a 256 "\$@"
EOF
  fi
  chmod 755 "$stub_path"
}

create_raw_metadata_fixture() {
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

create_batch_probe() {
  local probe_script=$1
  local container_path=$2
  local captured_base=$3
  local raw_manifest_sha256=$4
  local path_replacement='export PATH=${OCI_CAPTURE_TEST_TOOL_PATH:?}'
  local container_replacement="readonly CONTAINER=${container_path}"
  local captured_base_replacement="readonly CAPTURED_BASE=${captured_base}"
  local manifest_replacement="readonly EXPECTED_RAW_MANIFEST_SHA256=${raw_manifest_sha256}"
  local scratch_replacement='readonly SCRATCH_DIRECTORY=${OCI_CAPTURE_TEST_SCRATCH_BASE:?}/oci-capture-${SLURM_JOB_ID}'
  local srun_replacement='"${OCI_CAPTURE_TEST_SRUN:?}"'

  awk \
    -v path_replacement="$path_replacement" \
    -v container_replacement="$container_replacement" \
    -v captured_base_replacement="$captured_base_replacement" \
    -v manifest_replacement="$manifest_replacement" \
    -v scratch_replacement="$scratch_replacement" \
    -v srun_replacement="$srun_replacement" '
      /^export PATH=\/usr\/local\/sbin:/ {
        print path_replacement
        path_replacements += 1
        next
      }
      /^readonly CONTAINER=/ {
        print container_replacement
        container_replacements += 1
        next
      }
      /^readonly EXPECTED_RAW_MANIFEST_SHA256=/ {
        print manifest_replacement
        manifest_replacements += 1
        next
      }
      /^readonly CAPTURED_BASE=/ {
        print captured_base_replacement
        captured_base_replacements += 1
        next
      }
      /^readonly SCRATCH_DIRECTORY=\/raid\/scratch\/nemo-rl-semantic-precision-evidence\/oci-capture-/ {
        print scratch_replacement
        scratch_replacements += 1
        next
      }
      /^\/cm\/local\/apps\/slurm\/current\/bin\/srun \\$/ {
        print srun_replacement " " sprintf("%c", 92)
        srun_replacements += 1
        next
      }
      { print }
      END {
        if (path_replacements != 1 || container_replacements != 1 ||
            manifest_replacements != 1 || captured_base_replacements != 1 ||
            scratch_replacements != 1 || srun_replacements != 1) {
          exit 64
        }
      }
    ' "$BATCH" >"$probe_script"
  chmod 755 "$probe_script"
}

test_directory=$(mktemp -d)
readonly TEST_DIRECTORY=$test_directory
trap 'rm -rf -- "$TEST_DIRECTORY"' EXIT

readonly TOOL_DIRECTORY=$TEST_DIRECTORY/bin
readonly TOOLING_ROOT=$TEST_DIRECTORY/tooling
readonly SEMANTIC_WORKTREE=$TEST_DIRECTORY/semantic
readonly COMPRESSED_ROOT=$TEST_DIRECTORY/compressed-tensors
readonly MODELOPT_ROOT=$TEST_DIRECTORY/model-optimizer
readonly TRANSFORMER_ENGINE_ROOT=$TEST_DIRECTORY/transformer-engine
readonly RAW_ROOT=$TEST_DIRECTORY/raw
readonly CAPTURED_BASE=$TEST_DIRECTORY/captured
readonly SCRATCH_BASE=$TEST_DIRECTORY/scratch
readonly PROBE_SCRIPT=$TEST_DIRECTORY/capture-batch.sbatch
readonly CONTAINER_PATH=$TEST_DIRECTORY/nightly.sqsh
readonly VALIDATOR_FIXTURE=$TOOLING_ROOT/experiments/pr3652_validation_container/scripts/validate_transferred_nightly.sh
readonly CAPTURE_BODY_FIXTURE=$TOOLING_ROOT/experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sh
readonly FIRST_CHILD_ENVIRONMENT=$TEST_DIRECTORY/first-child.env
readonly VALIDATOR_ENVIRONMENT=$TEST_DIRECTORY/validator.env
readonly VALIDATOR_ARGUMENTS=$TEST_DIRECTORY/validator.args
readonly SRUN_ENVIRONMENT=$TEST_DIRECTORY/srun.env
readonly SRUN_ARGUMENTS=$TEST_DIRECTORY/srun.args
readonly TOOLING_SHA=1111111111111111111111111111111111111111
readonly SEMANTIC_SHA=2222222222222222222222222222222222222222

mkdir -p \
  "$TOOL_DIRECTORY" \
  "${VALIDATOR_FIXTURE%/*}" \
  "$SEMANTIC_WORKTREE" \
  "$COMPRESSED_ROOT" \
  "$MODELOPT_ROOT" \
  "$TRANSFORMER_ENGINE_ROOT" \
  "$RAW_ROOT" \
  "$CAPTURED_BASE/logs" \
  "$CAPTURED_BASE/runs" \
  "$SCRATCH_BASE"
: >"$CONTAINER_PATH"
create_raw_metadata_fixture "$RAW_ROOT"
raw_manifest_sha256=$(sha256_file "$RAW_ROOT/SHA256SUMS")
readonly RAW_MANIFEST_SHA256=$raw_manifest_sha256

cat >"$VALIDATOR_FIXTURE" <<'EOF'
#!/bin/bash
set -euo pipefail
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
/usr/bin/env | /usr/bin/sort >"${OCI_CAPTURE_TEST_VALIDATOR_ENVIRONMENT:?}"
printf '%s\n' "$@" >"${OCI_CAPTURE_TEST_VALIDATOR_ARGUMENTS:?}"
test "$#" = 3
EOF
cat >"$CAPTURE_BODY_FIXTURE" <<'EOF'
#!/bin/bash
set -euo pipefail
exit 0
EOF
cat >"$TOOL_DIRECTORY/git" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ ! -e ${OCI_CAPTURE_TEST_FIRST_CHILD_ENVIRONMENT:?} ]]; then
  /usr/bin/env | /usr/bin/sort >"${OCI_CAPTURE_TEST_FIRST_CHILD_ENVIRONMENT}"
fi
if [[ ${1:-} != -C || $# < 4 ]]; then
  exit 96
fi
path=$2
shift 2
case "$*" in
  'rev-parse --is-inside-work-tree') printf 'true\n' ;;
  'rev-parse --show-toplevel') printf '%s\n' "$path" ;;
  'status --porcelain') ;;
  'rev-parse HEAD')
    case $path in
      "${OCI_CAPTURE_TEST_TOOLING_ROOT}") printf '%s\n' "${OCI_CAPTURE_TEST_TOOLING_SHA}" ;;
      "${OCI_CAPTURE_TEST_SEMANTIC_WORKTREE}") printf '%s\n' "${OCI_CAPTURE_TEST_SEMANTIC_SHA}" ;;
      "${OCI_CAPTURE_TEST_COMPRESSED_ROOT}") printf '%s\n' f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0 ;;
      "${OCI_CAPTURE_TEST_MODELOPT_ROOT}") printf '%s\n' c897fbeaaff66d53d61033f107885b7c5432f235 ;;
      "${OCI_CAPTURE_TEST_TE_ROOT}") printf '%s\n' 42b840051647eef89761a16dfdff87e82bb253ab ;;
      */3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM) printf '%s\n' 7c9c3a027c503ae9ae1e8ad7b14397abb8269378 ;;
      */3rdparty/Megatron-Bridge-workspace/Megatron-Bridge) printf '%s\n' b11414c71b15e54d333eb49346ed199f20fa9021 ;;
      */3rdparty/Automodel-workspace/Automodel) printf '%s\n' 1814c6c93a66b9d59d254960ef6a99a64249b671 ;;
      *) exit 95 ;;
    esac
    ;;
  'config --get-all remote.origin.url')
    case $path in
      "${OCI_CAPTURE_TEST_COMPRESSED_ROOT}") printf '%s\n' https://github.com/vllm-project/compressed-tensors.git ;;
      "${OCI_CAPTURE_TEST_MODELOPT_ROOT}") printf '%s\n' https://github.com/NVIDIA/Model-Optimizer.git ;;
      "${OCI_CAPTURE_TEST_TE_ROOT}") printf '%s\n' https://github.com/NVIDIA/TransformerEngine.git ;;
      *) exit 95 ;;
    esac
    ;;
  rev-parse\ *:experiments/pr3652_validation_container/scripts/validate_transferred_nightly.sh)
    "${OCI_CAPTURE_TEST_REAL_GIT}" hash-object "${OCI_CAPTURE_TEST_VALIDATOR_FIXTURE}"
    ;;
  rev-parse\ *:experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sh)
    "${OCI_CAPTURE_TEST_REAL_GIT}" hash-object "${OCI_CAPTURE_TEST_CAPTURE_BODY_FIXTURE}"
    ;;
  show\ *:experiments/pr3652_validation_container/scripts/validate_transferred_nightly.sh)
    /bin/cat -- "${OCI_CAPTURE_TEST_VALIDATOR_FIXTURE}"
    ;;
  show\ *:experiments/pr3652_validation_container/scripts/oci_hsg_capture_precision_source_evidence.sh)
    /bin/cat -- "${OCI_CAPTURE_TEST_CAPTURE_BODY_FIXTURE}"
    ;;
  hash-object\ *) "${OCI_CAPTURE_TEST_REAL_GIT}" hash-object "$2" ;;
  *) exit 94 ;;
esac
EOF
cat >"$TOOL_DIRECTORY/srun" <<'EOF'
#!/bin/bash
set -euo pipefail
/usr/bin/env | /usr/bin/sort >"${OCI_CAPTURE_TEST_SRUN_ENVIRONMENT:?}"
printf '%s\n' "$@" >"${OCI_CAPTURE_TEST_SRUN_ARGUMENTS:?}"
EOF
create_sha256sum_compatibility_stub "$TOOL_DIRECTORY/sha256sum"
chmod 755 "$VALIDATOR_FIXTURE" "$CAPTURE_BODY_FIXTURE" "$TOOL_DIRECTORY/git" "$TOOL_DIRECTORY/srun"
create_batch_probe "$PROBE_SCRIPT" "$CONTAINER_PATH" "$CAPTURED_BASE" "$RAW_MANIFEST_SHA256"

env -i \
  PATH="$TOOL_DIRECTORY:/usr/bin:/bin" \
  SCRIPT_ROOT="$TOOLING_ROOT" \
  EXPECTED_TOOLING_SHA="$TOOLING_SHA" \
  SEMANTIC_WORKTREE="$SEMANTIC_WORKTREE" \
  EXPECTED_REPO_SHA="$SEMANTIC_SHA" \
  COMPRESSED_TENSORS_SOURCE_ROOT="$COMPRESSED_ROOT" \
  MODELOPT_LIGHTNING_SOURCE_ROOT="$MODELOPT_ROOT" \
  TRANSFORMER_ENGINE_SOURCE_ROOT="$TRANSFORMER_ENGINE_ROOT" \
  STAGED_METADATA_ROOT="$RAW_ROOT" \
  SLURM_JOB_ID=424247 \
  OCI_CAPTURE_TEST_TOOL_PATH="$TOOL_DIRECTORY:/usr/bin:/bin" \
  OCI_CAPTURE_TEST_SCRATCH_BASE="$SCRATCH_BASE" \
  OCI_CAPTURE_TEST_SRUN="$TOOL_DIRECTORY/srun" \
  OCI_CAPTURE_TEST_REAL_GIT="$REAL_GIT" \
  OCI_CAPTURE_TEST_TOOLING_ROOT="$TOOLING_ROOT" \
  OCI_CAPTURE_TEST_TOOLING_SHA="$TOOLING_SHA" \
  OCI_CAPTURE_TEST_SEMANTIC_WORKTREE="$SEMANTIC_WORKTREE" \
  OCI_CAPTURE_TEST_SEMANTIC_SHA="$SEMANTIC_SHA" \
  OCI_CAPTURE_TEST_COMPRESSED_ROOT="$COMPRESSED_ROOT" \
  OCI_CAPTURE_TEST_MODELOPT_ROOT="$MODELOPT_ROOT" \
  OCI_CAPTURE_TEST_TE_ROOT="$TRANSFORMER_ENGINE_ROOT" \
  OCI_CAPTURE_TEST_VALIDATOR_FIXTURE="$VALIDATOR_FIXTURE" \
  OCI_CAPTURE_TEST_CAPTURE_BODY_FIXTURE="$CAPTURE_BODY_FIXTURE" \
  OCI_CAPTURE_TEST_FIRST_CHILD_ENVIRONMENT="$FIRST_CHILD_ENVIRONMENT" \
  OCI_CAPTURE_TEST_VALIDATOR_ENVIRONMENT="$VALIDATOR_ENVIRONMENT" \
  OCI_CAPTURE_TEST_VALIDATOR_ARGUMENTS="$VALIDATOR_ARGUMENTS" \
  OCI_CAPTURE_TEST_SRUN_ENVIRONMENT="$SRUN_ENVIRONMENT" \
  OCI_CAPTURE_TEST_SRUN_ARGUMENTS="$SRUN_ARGUMENTS" \
  HF_TOKEN=scheduler-injected-authorization \
  HUGGING_FACE_HUB_TOKEN=scheduler-injected-authorization \
  HF_API_TOKEN=scheduler-injected-authorization \
  HUGGINGFACE_TOKEN=scheduler-injected-authorization \
  NVIDIA_API_KEY=scheduler-injected-authorization \
  NGC_API_KEY=scheduler-injected-authorization \
  AWS_ACCESS_KEY_ID=scheduler-injected-authorization \
  AWS_SECRET_ACCESS_KEY=scheduler-injected-authorization \
  AWS_SESSION_TOKEN=scheduler-injected-authorization \
  /bin/bash "$PROBE_SCRIPT"

for environment_evidence in \
  "$FIRST_CHILD_ENVIRONMENT" \
  "$VALIDATOR_ENVIRONMENT" \
  "$SRUN_ENVIRONMENT"; do
  test -s "$environment_evidence" || die "Missing environment boundary evidence: ${environment_evidence##*/}"
  test ! -L "$environment_evidence" || die "Symlink environment evidence: ${environment_evidence##*/}"
  for authorization_variable in \
    HF_TOKEN \
    HUGGING_FACE_HUB_TOKEN \
    HF_API_TOKEN \
    HUGGINGFACE_TOKEN \
    NVIDIA_API_KEY \
    NGC_API_KEY \
    AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY \
    AWS_SESSION_TOKEN; do
    if grep -Eq "^${authorization_variable}=" "$environment_evidence"; then
      die "Scheduler authorization reached ${environment_evidence##*/}: ${authorization_variable}"
    fi
  done
  if grep -Fq scheduler-injected-authorization "$environment_evidence"; then
    die "Scheduler authorization value reached ${environment_evidence##*/}"
  fi
  grep -Fx -- 'SLURM_JOB_ID=424247' "$environment_evidence" >/dev/null ||
    die "Required Slurm job ID missing from ${environment_evidence##*/}"
done

test -s "$VALIDATOR_ARGUMENTS" || die 'Validator boundary was not reached'
test ! -L "$VALIDATOR_ARGUMENTS" || die 'Validator argument evidence is a symlink'
printf '%s\n' \
  "$CONTAINER_PATH" \
  "$CONTAINER_PATH.metadata.txt" \
  "$CONTAINER_PATH.complete" >"$TEST_DIRECTORY/expected-validator.args"
cmp -s "$TEST_DIRECTORY/expected-validator.args" "$VALIDATOR_ARGUMENTS" || die 'Validator arguments changed'

test -s "$SRUN_ARGUMENTS" || die 'srun boundary was not reached'
test ! -L "$SRUN_ARGUMENTS" || die 'srun argument evidence is a symlink'
for expected_argument in \
  --nodes=1 \
  --ntasks=1 \
  --gpus=1 \
  "--container-image=${CONTAINER_PATH}" \
  "$SCRATCH_BASE/oci-capture-424247/oci_hsg_capture_precision_source_evidence.sh"; do
  grep -Fx -- "$expected_argument" "$SRUN_ARGUMENTS" >/dev/null || die "Missing srun argument: $expected_argument"
done
grep -Fx -- 'SLURM_EXPORT_ENV=ALL' "$SRUN_ENVIRONMENT" >/dev/null || die 'srun export mode changed'

printf 'OCI capture authorization boundary checks passed\n'
