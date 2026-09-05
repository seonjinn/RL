#!/bin/bash
# shellcheck disable=SC2016

set -euo pipefail

report_failure() {
  local exit_status=$?
  local line_number=$1
  local command=$2

  printf 'Harness failure: line=%s exit=%s command=%s\n' "${line_number}" "${exit_status}" "${command}" >&2
  exit "${exit_status}"
}
trap 'report_failure "${LINENO}" "${BASH_COMMAND}"' ERR

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly HARNESS=${script_dir}/test_oci_hsg_stage_precision_source_metadata_static.sh
readonly BATCH=${script_dir}/oci_hsg_stage_precision_source_metadata.sbatch
readonly SUBMIT=${script_dir}/submit_oci_hsg_stage_precision_source_metadata.sh
readonly EXPECTED_MANIFEST_SHA256=d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b
readonly EXPECTED_IMAGE_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly EXPECTED_OUTPUT_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw
readonly EXPECTED_IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly -a AUTHORIZATION_VARIABLES=(
  HF_TOKEN
  HUGGING_FACE_HUB_TOKEN
  HF_API_TOKEN
  HUGGINGFACE_TOKEN
  NVIDIA_API_KEY
  NGC_API_KEY
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_SESSION_TOKEN
)

die() {
  echo "$*" >&2
  exit 1
}

require_literal() {
  local literal=$1
  local path=$2

  grep -Fq -- "${literal}" "${path}" || die "Missing required literal in ${path}: ${literal}"
}

forbid_literal() {
  local literal=$1
  local path=$2

  if grep -Fq -- "${literal}" "${path}"; then
    die "Forbidden literal in ${path}: ${literal}"
  fi
}

forbid_literal "/opt/$(printf '%s' homebrew)/bin/python3" "${HARNESS}"

REAL_PYTHON=$(command -v python3) || die 'python3 is required for the executable import-origin probe'
readonly REAL_PYTHON
[[ ${REAL_PYTHON} == /* && -x ${REAL_PYTHON} ]] || die "python3 did not resolve to an executable absolute path: ${REAL_PYTHON}"
"${REAL_PYTHON}" -S -P - <<'PYTHON_RUNTIME_PROBE'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"python3 >= 3.11 is required, found {sys.version.split()[0]}")
if not sys.flags.no_site:
    raise SystemExit("python3 -S did not disable site initialization")
if not sys.flags.safe_path:
    raise SystemExit("python3 -P did not enable safe_path")
PYTHON_RUNTIME_PROBE

expect_failure() {
  local label=$1
  shift

  if "$@" >"${TEST_DIRECTORY}/${label}.stdout" 2>"${TEST_DIRECTORY}/${label}.stderr"; then
    die "Expected failure: ${label}"
  fi
}

expect_failure_status() {
  local label=$1
  local expected_status=$2
  local actual_status
  shift 2

  if "$@" >"${TEST_DIRECTORY}/${label}.stdout" 2>"${TEST_DIRECTORY}/${label}.stderr"; then
    die "Expected failure: ${label}"
  else
    actual_status=$?
  fi
  if (( actual_status != expected_status )); then
    die "Expected ${label} exit ${expected_status}, received ${actual_status}"
  fi
}

sha256_file() {
  local path=$1

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

for required_file in "${BATCH}" "${SUBMIT}"; do
  [[ -x ${required_file} ]] || die "Required executable is missing: ${required_file}"
  bash -n "${required_file}"
done

require_literal '#SBATCH --account=nemotron_n3_post' "${BATCH}"
require_literal '#SBATCH --partition=cpu_datamover' "${BATCH}"
require_literal '#SBATCH --nodes=1' "${BATCH}"
require_literal '#SBATCH --ntasks-per-node=1' "${BATCH}"
require_literal '#SBATCH --time=00:30:00' "${BATCH}"
require_literal "readonly CONTAINER=${EXPECTED_IMAGE}" "${BATCH}"
require_literal "readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_IMAGE_SHA256}" "${BATCH}"
require_literal 'readonly CONTAINER_PYTHON=/opt/nemo_rl_venv/bin/python' "${BATCH}"
require_literal "readonly OUTPUT_ROOT=${EXPECTED_OUTPUT_ROOT}" "${BATCH}"
require_literal "readonly EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256}" "${BATCH}"
require_literal 'readonly RAW_MANIFEST_FILENAME=SHA256SUMS' "${BATCH}"
require_literal 'readonly SCRATCH_ROOT=/raid/scratch' "${BATCH}"
require_literal 'readonly USER_SCRATCH_ROOT=${SCRATCH_ROOT}/sna' "${BATCH}"
require_literal 'readonly SCRATCH_BASE=${USER_SCRATCH_ROOT}/nemo-rl-semantic-precision-metadata' "${BATCH}"
require_literal 'readonly SCRATCH_DIRECTORY=${SCRATCH_BASE}/oci-stage-${SLURM_JOB_ID}' "${BATCH}"
forbid_literal 'readonly SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-' "${BATCH}"
require_literal 'readonly -a SNAPSHOT_BLOBS=(' "${BATCH}"
require_literal 'PYTHONPATH=${SNAPSHOT_ROOT}' "${BATCH}"
require_literal 'PYTHONNOUSERSITE=1' "${BATCH}"
require_literal 'PYTHONSAFEPATH=1' "${BATCH}"
require_literal '"${CONTAINER_PYTHON}" -S -P -c' "${BATCH}"
require_literal '"${CONTAINER_PYTHON}" -S -P - "${SNAPSHOT_ROOT}"' "${BATCH}"
require_literal 'run_stager_site_free_twice()' "${BATCH}"
test "$(grep -Fc -- 'run_stager_site_free_twice' "${BATCH}")" = 2
require_literal 'verified metadata module identity changed' "${BATCH}"
require_literal 'job-owned scratch lifecycle violated during container stage' "${BATCH}"
forbid_literal 'runpy.run_path' "${BATCH}"
forbid_literal 'actual_leaf = pathlib.Path(leaf_file).resolve' "${BATCH}"
forbid_literal '"${CONTAINER_PYTHON}" -S -P "${STAGER}"' "${BATCH}"
require_literal 'os.environ[variable] = "http://127.0.0.1:9"' "${BATCH}"
require_literal 'readonly SRUN=/cm/local/apps/slurm/current/bin/srun' "${BATCH}"
require_literal '--container-image="${CONTAINER}"' "${BATCH}"
require_literal '--container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch' "${BATCH}"
for relative_path in \
  tools/stage_precision_policy_source_metadata.py \
  tools/precision_policy_source_artifacts.py; do
  require_literal "${relative_path}" "${BATCH}"
  require_literal "${relative_path}" "${SUBMIT}"
done
for excluded_path in \
  tools/capture_precision_policy_source_evidence.py \
  nemo_rl/precision_policy/semantic.py \
  nemo_rl/precision_policy/source_formats.py; do
  forbid_literal "${excluded_path}" "${BATCH}"
  forbid_literal "${excluded_path}" "${SUBMIT}"
done
for blocked_module in \
  nemo_rl \
  pydantic \
  typing_extensions \
  tools.capture_precision_policy_source_evidence; do
  require_literal "${blocked_module}" "${BATCH}"
done
require_literal 'blocked_imports=not-attempted-not-loaded' "${BATCH}"
forbid_literal '"${CONTAINER_PYTHON}" -P' "${BATCH}"
for authorization_variable in "${AUTHORIZATION_VARIABLES[@]}"; do
  require_literal "${authorization_variable}" "${BATCH}"
done
forbid_literal '#SBATCH --gpus' "${BATCH}"
forbid_literal '#SBATCH --gres=gpu' "${BATCH}"
forbid_literal '#SBATCH --exclusive' "${BATCH}"
forbid_literal 'sha256sum "${CONTAINER}"' "${BATCH}"
forbid_literal 'export HOME=' "${BATCH}"
forbid_literal '--export=ALL' "${SUBMIT}"
forbid_literal 'HF_TOKEN=' "${SUBMIT}"
forbid_literal 'HUGGING_FACE_HUB_TOKEN=' "${SUBMIT}"

require_literal 'readonly ACTION=${1:-test-only}' "${SUBMIT}"
require_literal 'readonly PRODUCT_WORKTREE=/home/sna/nemorl-task4b-stdlib-artifacts-20260904' "${SUBMIT}"
require_literal 'readonly EXPECTED_PRODUCT_SHA=5378b3312e0a7d80e1f7520bcf231ab4d1d76344' "${SUBMIT}"
require_literal ': "${PRODUCT_WORKTREE:?Set PRODUCT_WORKTREE to the absolute clean product worktree root}"' "${BATCH}"
require_literal ': "${EXPECTED_PRODUCT_SHA:?Set EXPECTED_PRODUCT_SHA to the product worktree HEAD}"' "${BATCH}"
require_literal "readonly OUTPUT_ROOT=${EXPECTED_OUTPUT_ROOT}" "${SUBMIT}"
require_literal 'readonly SBATCH_COMMAND=/cm/local/apps/slurm/current/bin/sbatch' "${SUBMIT}"
require_literal 'readonly OCI_SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf' "${SUBMIT}"
require_literal '| /usr/bin/env -i' "${SUBMIT}"
require_literal 'PATH=/cm/local/apps/slurm/current/bin:/usr/bin:/bin' "${SUBMIT}"
require_literal 'SLURM_CONF="${OCI_SLURM_CONF}"' "${SUBMIT}"
require_literal '--export="${EXPORTS}"' "${SUBMIT}"
forbid_literal '| sbatch' "${SUBMIT}"
forbid_literal 'SEMANTIC_WORKTREE' "${BATCH}"
forbid_literal 'SEMANTIC_WORKTREE' "${SUBMIT}"
forbid_literal 'EXPECTED_REPO_SHA' "${BATCH}"
forbid_literal 'EXPECTED_REPO_SHA' "${SUBMIT}"

test_directory=$(mktemp -d)
TEST_DIRECTORY=$(cd -- "${test_directory}" && pwd -P)
readonly TEST_DIRECTORY
cleanup() {
  rm -rf -- "${TEST_DIRECTORY}"
}
trap cleanup EXIT

readonly TOOLING_SHA=1111111111111111111111111111111111111111
readonly SEMANTIC_SHA=5378b3312e0a7d80e1f7520bcf231ab4d1d76344
readonly OTHER_SHA=3333333333333333333333333333333333333333
readonly TOOL_ROOT=${TEST_DIRECTORY}/tool-root
readonly SEMANTIC_ROOT=${TEST_DIRECTORY}/semantic-root
readonly TEST_OUTPUT_ROOT=${TEST_DIRECTORY}/raw
readonly TEST_SCRATCH_ROOT=${TEST_DIRECTORY}/raid-scratch
readonly TEST_USER_SCRATCH_ROOT=${TEST_SCRATCH_ROOT}/sna
readonly TEST_SCRATCH_BASE=${TEST_USER_SCRATCH_ROOT}/nemo-rl-semantic-precision-metadata
readonly TEST_JOB_SCRATCH_DIRECTORY=${TEST_SCRATCH_BASE}/oci-stage-98765
readonly STUB_BIN=${TEST_DIRECTORY}/bin
readonly CALLS=${TEST_DIRECTORY}/calls
mkdir -p \
  "${TOOL_ROOT}/experiments/pr3652_validation_container/scripts" \
  "${SEMANTIC_ROOT}" \
  "${TEST_SCRATCH_BASE}" \
  "${STUB_BIN}" \
  "${CALLS}"
cp "${BATCH}" "${TOOL_ROOT}/experiments/pr3652_validation_container/scripts/oci_hsg_stage_precision_source_metadata.sbatch"

readonly SUBMIT_PROBE=${TOOL_ROOT}/experiments/pr3652_validation_container/scripts/submit-probe.sh
sed \
  -e 's#^readonly PRODUCT_WORKTREE=.*#readonly PRODUCT_WORKTREE=${TEST_PRODUCT_WORKTREE:?}#' \
  -e 's#^readonly OUTPUT_ROOT=.*#readonly OUTPUT_ROOT=${TEST_OUTPUT_ROOT:?}#' \
  -e 's#^readonly SBATCH_COMMAND=.*#readonly SBATCH_COMMAND=${TEST_SBATCH_COMMAND:?}#' \
  -e 's#^readonly OCI_SLURM_CONF=.*#readonly OCI_SLURM_CONF=${TEST_OCI_SLURM_CONF:?}#' \
  "${SUBMIT}" >"${SUBMIT_PROBE}"
chmod 755 "${SUBMIT_PROBE}" "${TOOL_ROOT}/experiments/pr3652_validation_container/scripts/oci_hsg_stage_precision_source_metadata.sbatch"

cat >"${STUB_BIN}/git" <<'GIT_STUB'
#!/bin/bash
set -euo pipefail

[[ ${1:-} == -C ]] || exit 90
requested_root=$2
shift 2
printf '%s\t%s\n' "${requested_root}" "$*" >>"${TEST_GIT_LOG}"

if [[ ${requested_root} == "${TEST_TOOL_ROOT}" || ${requested_root} == "${TEST_TOOL_ROOT}/"* ]]; then
  root=${TEST_TOOL_ROOT}
  head_sha=${TEST_TOOL_HEAD}
  upstream_sha=${TEST_TOOL_UPSTREAM_SHA}
  dirty=${TEST_TOOL_DIRTY:-}
  upstream_missing=${TEST_TOOL_UPSTREAM_MISSING:-}
elif [[ ${requested_root} == "${TEST_SEMANTIC_ROOT}" ]]; then
  root=${TEST_SEMANTIC_ROOT}
  head_sha=${TEST_SEMANTIC_HEAD}
  upstream_sha=${TEST_SEMANTIC_UPSTREAM_SHA}
  dirty=${TEST_SEMANTIC_DIRTY:-}
  upstream_missing=${TEST_SEMANTIC_UPSTREAM_MISSING:-}
else
  printf 'unexpected git fixture root: %s\n' "${requested_root}" >&2
  exit 91
fi

case ${1:-} in
  rev-parse)
    shift
    case "$*" in
      '--show-toplevel') printf '%s\n' "${root}" ;;
      '--is-inside-work-tree') printf '%s\n' true ;;
      'HEAD') printf '%s\n' "${head_sha}" ;;
      '--abbrev-ref --symbolic-full-name @{upstream}')
        [[ -z ${upstream_missing} ]] || exit 2
        printf '%s\n' origin/test
        ;;
      '@{upstream}')
        [[ -z ${upstream_missing} ]] || exit 2
        printf '%s\n' "${upstream_sha}"
        ;;
      *) exit 92 ;;
    esac
    ;;
  status)
    [[ ${2:-} == --porcelain ]] || exit 93
    [[ -z ${dirty} ]] || printf '%s\n' ' M changed'
    ;;
  cat-file)
    spec=${3:?}
    relative_path=${spec#*:}
    if [[ ${root} == "${TEST_TOOL_ROOT}" ]]; then
      [[ ${relative_path} == experiments/pr3652_validation_container/scripts/oci_hsg_stage_precision_source_metadata.sbatch ]] || exit 96
    else
      case ${relative_path} in
        tools/stage_precision_policy_source_metadata.py | tools/precision_policy_source_artifacts.py) ;;
        *) exit 96 ;;
      esac
    fi
    case ${2:-} in
      -e)
        [[ ${relative_path} != "${TEST_MISSING_BLOB:-}" ]] || exit 1
        ;;
      -t) printf '%s\n' blob ;;
      *) exit 94 ;;
    esac
    ;;
  show)
    cat "${TEST_BATCH_FILE}"
    ;;
  *) exit 95 ;;
esac
GIT_STUB

cat >"${STUB_BIN}/sbatch" <<SBATCH_STUB
#!/bin/bash
set -euo pipefail
printf '%s\n' "\$@" >"${CALLS}/sbatch.args"
env | LC_ALL=C sort >"${CALLS}/sbatch.env"
cat >"${CALLS}/sbatch.stdin"
touch "${CALLS}/sbatch.called"
SBATCH_STUB
chmod 755 "${STUB_BIN}/git" "${STUB_BIN}/sbatch"

export TEST_GIT_LOG=${CALLS}/submit-git.log
export TEST_SBATCH_ARGS=${CALLS}/sbatch.args
export TEST_SBATCH_STDIN=${CALLS}/sbatch.stdin
export TEST_SBATCH_CALLED=${CALLS}/sbatch.called
export TEST_SBATCH_ENV=${CALLS}/sbatch.env
export TEST_TOOL_ROOT=${TOOL_ROOT}
export TEST_SEMANTIC_ROOT=${SEMANTIC_ROOT}
export TEST_PRODUCT_WORKTREE=${SEMANTIC_ROOT}
export TEST_OUTPUT_ROOT
export TEST_TOOL_HEAD=${TOOLING_SHA}
export TEST_TOOL_UPSTREAM_SHA=${TOOLING_SHA}
export TEST_SEMANTIC_HEAD=${SEMANTIC_SHA}
export TEST_SEMANTIC_UPSTREAM_SHA=${SEMANTIC_SHA}
export TEST_BATCH_FILE=${TOOL_ROOT}/experiments/pr3652_validation_container/scripts/oci_hsg_stage_precision_source_metadata.sbatch
export TEST_SBATCH_COMMAND=${STUB_BIN}/sbatch
export TEST_OCI_SLURM_CONF=${TEST_DIRECTORY}/slurm.conf
touch "${TEST_OCI_SLURM_CONF}"

PATH=${STUB_BIN}:/usr/bin:/bin \
  HF_TOKEN=must-not-be-exported \
  SBATCH_ACCOUNT=hostile-account \
  SBATCH_PARTITION=hostile-gpu \
  SBATCH_GPUS=8 \
  SBATCH_GRES=gpu:8 \
  SBATCH_EXCLUSIVE=1 \
  SBATCH_TIME=7-00:00:00 \
  SLURM_CLUSTERS=hostile-cluster \
  SLURM_HINT=nomultithread \
  "${SUBMIT_PROBE}" test-only
test -e "${TEST_SBATCH_CALLED}"
grep -Fx -- '--test-only' "${TEST_SBATCH_ARGS}" >/dev/null
grep -Fx -- "--chdir=${TOOL_ROOT}" "${TEST_SBATCH_ARGS}" >/dev/null
expected_exports="--export=SCRIPT_ROOT=${TOOL_ROOT},EXPECTED_TOOLING_SHA=${TOOLING_SHA},PRODUCT_WORKTREE=${SEMANTIC_ROOT},EXPECTED_PRODUCT_SHA=${SEMANTIC_SHA}"
grep -Fx -- "${expected_exports}" "${TEST_SBATCH_ARGS}" >/dev/null
forbid_literal 'must-not-be-exported' "${TEST_SBATCH_ARGS}"
if grep -Eq '^(SBATCH_|SLURM_CLUSTERS=|SLURM_HINT=)' "${TEST_SBATCH_ENV}"; then
  die 'Hostile scheduler option environment reached sbatch'
fi
cmp -s "${BATCH}" "${TEST_SBATCH_STDIN}" || die 'Submit wrapper did not stream the committed batch blob'

rm -f "${TEST_SBATCH_CALLED}"
expect_failure submit_dirty_tooling env "PATH=${STUB_BIN}:/usr/bin:/bin" TEST_TOOL_DIRTY=1 "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_missing_tooling_upstream env "PATH=${STUB_BIN}:/usr/bin:/bin" TEST_TOOL_UPSTREAM_MISSING=1 "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_divergent_tooling env "PATH=${STUB_BIN}:/usr/bin:/bin" "TEST_TOOL_UPSTREAM_SHA=${OTHER_SHA}" "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_dirty_semantic env "PATH=${STUB_BIN}:/usr/bin:/bin" TEST_SEMANTIC_DIRTY=1 "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_missing_semantic_upstream env "PATH=${STUB_BIN}:/usr/bin:/bin" TEST_SEMANTIC_UPSTREAM_MISSING=1 "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_divergent_semantic env "PATH=${STUB_BIN}:/usr/bin:/bin" "TEST_SEMANTIC_UPSTREAM_SHA=${OTHER_SHA}" "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_clean_different_product_head env \
  "PATH=${STUB_BIN}:/usr/bin:/bin" \
  "TEST_SEMANTIC_HEAD=${OTHER_SHA}" \
  "TEST_SEMANTIC_UPSTREAM_SHA=${OTHER_SHA}" \
  "${SUBMIT_PROBE}" \
  test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_missing_blob env "PATH=${STUB_BIN}:/usr/bin:/bin" TEST_MISSING_BLOB=tools/stage_precision_policy_source_metadata.py "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"
expect_failure submit_missing_leaf_blob env "PATH=${STUB_BIN}:/usr/bin:/bin" TEST_MISSING_BLOB=tools/precision_policy_source_artifacts.py "${SUBMIT_PROBE}" test-only
test ! -e "${TEST_SBATCH_CALLED}"

readonly ARCHIVE_ROOT=${TEST_DIRECTORY}/archive-root
readonly ALTERED_ARCHIVE_ROOT=${TEST_DIRECTORY}/altered-archive-root
readonly ALTERED_LEAF_ARCHIVE_ROOT=${TEST_DIRECTORY}/altered-leaf-archive-root
readonly PRIMARY_FAILURE_ARCHIVE_ROOT=${TEST_DIRECTORY}/primary-failure-archive-root
mkdir -p \
  "${ARCHIVE_ROOT}/tools"
cat >"${ARCHIVE_ROOT}/tools/stage_precision_policy_source_metadata.py" <<'PYTHON_STAGER_FIXTURE'
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from tools import precision_policy_source_artifacts


def _maybe_attempt_deferred_import(pass_index: int) -> None:
    module_name = os.environ.get("TEST_DEFERRED_IMPORT_MODULE")
    requested_pass = os.environ.get("TEST_DEFERRED_IMPORT_PASS")
    if module_name is None or requested_pass != str(pass_index):
        return
    try:
        __import__(module_name)
    except ImportError:
        pass


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--output-root":
        return 72
    output_root = Path(sys.argv[2])
    count_path = Path(os.environ["TEST_STAGE_COUNT"])
    count = int(count_path.read_text()) if count_path.exists() else 0
    count += 1
    count_path.write_text(f"{count}\n")
    _maybe_attempt_deferred_import(count)

    mode = os.environ.get("TEST_STAGE_MODE", "success")
    partial = output_root / ".precision-policy-source-metadata-fixture"
    if count == 1 and mode == "first_fail":
        partial.mkdir()
        shutil.rmtree(partial)
        return 71

    final = output_root / f"sha256-{os.environ['TEST_MANIFEST_SHA256']}"
    if not final.exists():
        shutil.copytree(os.environ["TEST_GOLDEN_TREE"], partial)
        if mode == "wrong_manifest":
            with (partial / "SHA256SUMS").open("a") as manifest:
                manifest.write("malformed\n")
        elif mode == "wrong_topology":
            (partial / "unexpected").write_text("unexpected\n")
        elif mode == "symlink":
            (partial / "linked-checkpoints").symlink_to("checkpoints")
        partial.rename(final)

    if count == 1 and mode == "snapshot_vanishes_after_publish":
        shutil.rmtree(os.environ["PYTHONPATH"])
    elif count == 1 and mode == "whole_scratch_vanishes_after_publish":
        shutil.rmtree(os.environ["TEST_SCRATCH_DIRECTORY"])
    elif count == 1 and mode == "leaf_origin_escaped":
        precision_policy_source_artifacts.__file__ = os.environ[
            "TEST_ESCAPED_LEAF_PATH"
        ]
    elif count == 1 and mode == "leaf_module_replaced":
        sys.modules["tools.precision_policy_source_artifacts"] = object()

    if count == 2:
        assert os.environ.get("https_proxy") == "http://127.0.0.1:9"
        assert os.environ.get("HTTPS_PROXY") == "http://127.0.0.1:9"
        with Path(os.environ["TEST_PYTHON_LOG"]).open("a") as python_log:
            python_log.write("second_stage_network=blocked\n")
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON_STAGER_FIXTURE
printf '%s\n' \
  'FIXTURE_ARTIFACT_IDENTITIES = ()' \
  >"${ARCHIVE_ROOT}/tools/precision_policy_source_artifacts.py"
cp -R "${ARCHIVE_ROOT}" "${ALTERED_ARCHIVE_ROOT}"
printf '%s\n' altered >>"${ALTERED_ARCHIVE_ROOT}/tools/stage_precision_policy_source_metadata.py"
cp -R "${ARCHIVE_ROOT}" "${ALTERED_LEAF_ARCHIVE_ROOT}"
printf '%s\n' altered >>"${ALTERED_LEAF_ARCHIVE_ROOT}/tools/precision_policy_source_artifacts.py"
cp -R "${ARCHIVE_ROOT}" "${PRIMARY_FAILURE_ARCHIVE_ROOT}"
cat >"${PRIMARY_FAILURE_ARCHIVE_ROOT}/tools/stage_precision_policy_source_metadata.py" <<'PYTHON_PRIMARY_FAILURE_FIXTURE'
def main() -> int:
    raise RuntimeError("primary deferred fixture failure")


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON_PRIMARY_FAILURE_FIXTURE
printf '%s\n' semantic-fixture >"${SEMANTIC_ROOT}/must-remain-unchanged"
SEMANTIC_SENTINEL_SHA256=$(sha256_file "${SEMANTIC_ROOT}/must-remain-unchanged")
readonly SEMANTIC_SENTINEL_SHA256

readonly GOLDEN_TREE=${TEST_DIRECTORY}/golden-tree
mkdir -p "${GOLDEN_TREE}"
while IFS= read -r relative_path; do
  mkdir -p "${GOLDEN_TREE}/$(dirname -- "${relative_path}")"
  printf 'payload:%s\n' "${relative_path}" >"${GOLDEN_TREE}/${relative_path}"
done <<'PAYLOAD_PATHS'
checkpoints/kimi_k2/config.json
checkpoints/kimi_k2/model.safetensors.index.json
checkpoints/kimi_k2/safetensors_header_manifest.json
checkpoints/kimi_k25/config.json
checkpoints/kimi_k25/model.safetensors.index.json
checkpoints/kimi_k25/safetensors_header_manifest.json
checkpoints/kimi_k3/config.json
checkpoints/kimi_k3/model.safetensors.index.json
checkpoints/kimi_k3/safetensors_header_manifest.json
checkpoints/nemotron_lightning_nvfp4/config.json
checkpoints/nemotron_lightning_nvfp4/model.safetensors.index.json
checkpoints/nemotron_lightning_nvfp4/safetensors_header_manifest.json
checkpoints/qwen3_bf16/config.json
checkpoints/qwen3_bf16/model.safetensors.index.json
checkpoints/qwen3_bf16/safetensors_header_manifest.json
checkpoints/qwen_a95b_fp8/config.json
checkpoints/qwen_a95b_fp8/model.safetensors.index.json
checkpoints/qwen_a95b_fp8/safetensors_header_byte_lengths.json
checkpoints/qwen_a95b_fp8/safetensors_header_manifest.json
PAYLOAD_PATHS
(
  cd "${GOLDEN_TREE}"
  find checkpoints -type f -print | LC_ALL=C sort | while IFS= read -r path; do
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${path}"
    else
      digest=$(shasum -a 256 "${path}" | awk '{print $1}')
      printf '%s  %s\n' "${digest}" "${path}"
    fi
  done >SHA256SUMS
)
TEST_MANIFEST_SHA256=$(sha256_file "${GOLDEN_TREE}/SHA256SUMS")
readonly TEST_MANIFEST_SHA256

readonly TEST_CONTAINER=${TEST_DIRECTORY}/nightly.sqsh
readonly TEST_ESCAPED_LEAF_PATH=${TEST_DIRECTORY}/escaped-leaf.py
touch "${TEST_CONTAINER}"
printf '%s\n' escaped >"${TEST_ESCAPED_LEAF_PATH}"

write_valid_container_receipts() {
  rm -f -- "${TEST_CONTAINER}.metadata.txt" "${TEST_CONTAINER}.complete"
  cat >"${TEST_CONTAINER}.metadata.txt" <<EOF
source=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
source_cluster=ptyche
source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh
source_smoke_job=2721177
download_job=12345
downloaded_at_utc=2026-09-04T12:34:56Z
sha256=${EXPECTED_IMAGE_SHA256}
EOF
  cat >"${TEST_CONTAINER}.complete" <<EOF
destination=${TEST_CONTAINER}
source=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
sha256=${EXPECTED_IMAGE_SHA256}
completed_at_utc=2026-09-04T12:35:00Z
EOF
}

write_valid_container_receipts

readonly BATCH_PROBE=${TEST_DIRECTORY}/batch-probe.sh
sed \
  -e 's#^export PATH=.*#export PATH=${TEST_BATCH_PATH:?}#' \
  -e 's#^readonly CONTAINER=.*#readonly CONTAINER=${TEST_CONTAINER:?}#' \
  -e 's#^readonly CONTAINER_PYTHON=.*#readonly CONTAINER_PYTHON=${TEST_CONTAINER_PYTHON:?}#' \
  -e 's#^readonly OUTPUT_ROOT=.*#readonly OUTPUT_ROOT=${TEST_OUTPUT_ROOT:?}#' \
  -e 's#^readonly EXPECTED_MANIFEST_SHA256=.*#readonly EXPECTED_MANIFEST_SHA256=${TEST_MANIFEST_SHA256:?}#' \
  -e 's#^readonly SCRATCH_ROOT=.*#readonly SCRATCH_ROOT=${TEST_SCRATCH_ROOT:?}#' \
  -e 's#^readonly USER_SCRATCH_ROOT=.*#readonly USER_SCRATCH_ROOT=${TEST_USER_SCRATCH_ROOT:?}#' \
  -e 's#^readonly SCRATCH_BASE=.*#readonly SCRATCH_BASE=${TEST_SCRATCH_BASE:?}#' \
  -e 's#^readonly SCRATCH_DIRECTORY=.*#readonly SCRATCH_DIRECTORY=${TEST_SCRATCH_DIRECTORY:?}#' \
  -e 's#^readonly SRUN=.*#readonly SRUN=${TEST_SRUN:?}#' \
  "${BATCH}" >"${BATCH_PROBE}"
chmod 755 "${BATCH_PROBE}"

readonly EXIT_WINDOW_DELETE_BATCH_PROBE=${TEST_DIRECTORY}/batch-probe-exit-window-delete.sh
awk '
  { print }
  index($0, "staged_metadata=%s") {
    print "rm -rf -- \"${SCRATCH_DIRECTORY}\""
    insertions += 1
  }
  END {
    if (insertions != 1) {
      exit 97
    }
  }
' "${BATCH_PROBE}" >"${EXIT_WINDOW_DELETE_BATCH_PROBE}"
chmod 755 "${EXIT_WINDOW_DELETE_BATCH_PROBE}"

cat >"${STUB_BIN}/batch-git" <<'BATCH_GIT_STUB'
#!/bin/bash
set -euo pipefail

[[ ${1:-} == -C ]] || exit 80
root=$2
shift 2
printf '%s\t%s\n' "${root}" "$*" >>"${TEST_BATCH_GIT_LOG}"

case ${1:-} in
  rev-parse)
    shift
    case ${1:-} in
      --show-toplevel) printf '%s\n' "${root}" ;;
      --is-inside-work-tree) printf '%s\n' true ;;
      HEAD)
        if [[ ${root} == "${SCRIPT_ROOT}" ]]; then
          printf '%s\n' "${TEST_BATCH_TOOL_HEAD}"
        else
          printf '%s\n' "${TEST_BATCH_SEMANTIC_HEAD}"
        fi
        ;;
      *:*)
        relative_path=${1#*:}
        /usr/bin/git hash-object "${TEST_PRISTINE_ARCHIVE_ROOT}/${relative_path}"
        ;;
      *) exit 81 ;;
    esac
    ;;
  status)
    [[ ${2:-} == --porcelain ]] || exit 82
    if [[ ${root} == "${SCRIPT_ROOT}" ]]; then
      [[ -z ${TEST_BATCH_TOOL_DIRTY:-} ]] || printf '%s\n' ' M tooling'
    else
      [[ -z ${TEST_BATCH_SEMANTIC_DIRTY:-} ]] || printf '%s\n' ' M semantic'
    fi
    ;;
  archive)
    [[ $# == 6 ]] || exit 83
    [[ ${2:-} == --format=tar ]] || exit 84
    [[ ${3:-} == "${TEST_BATCH_SEMANTIC_HEAD}" ]] || exit 85
    [[ ${4:-} == -- ]] || exit 86
    [[ ${5:-} == tools/stage_precision_policy_source_metadata.py ]] || exit 87
    [[ ${6:-} == tools/precision_policy_source_artifacts.py ]] || exit 88
    /usr/bin/tar -cf - -C "${TEST_ARCHIVE_ROOT}" "${5}" "${6}"
    ;;
  hash-object)
    /usr/bin/git hash-object "${2:?}"
    ;;
  *) exit 89 ;;
esac
BATCH_GIT_STUB

cat >"${STUB_BIN}/srun" <<'SRUN_STUB'
#!/bin/bash
set -euo pipefail
env | LC_ALL=C sort >"${TEST_SRUN_ENV}"
printf '%s\n' "$@" >"${TEST_SRUN_LOG}"
touch "${TEST_SRUN_CALLED}"
while (( $# > 0 )) && [[ $1 == --* ]]; do
  shift
done
exec "$@"
SRUN_STUB

cat >"${STUB_BIN}/fake-python" <<'PYTHON_STUB'
#!/bin/bash
set -euo pipefail

[[ ${1:-} == -S && ${2:-} == -P ]] || {
  printf 'Python invocation was not site-free and safe-path: %s\n' "$*" >&2
  exit 74
}
shift 2
case ${1:-} in
  -c)
    printf '%s\n' "${TEST_PYTHON_VERSION:-3.13.7}"
    ;;
  -)
    printf '%s\n' "import_probe=$*" >>"${TEST_PYTHON_LOG}"
    exec "${TEST_REAL_PYTHON}" -S -P "$@"
    ;;
  *)
    printf 'unexpected fake Python invocation: %s\n' "$*" >&2
    exit 73
    ;;
esac
PYTHON_STUB

cat >"${STUB_BIN}/find" <<'FIND_STUB'
#!/bin/bash
set -euo pipefail

if [[ -n ${TEST_FIND_FAIL_TOPOLOGY:-} &&
  ${1:-} == . && ${2:-} == -type && ${3:-} == d ]]; then
  "${TEST_REAL_FIND}" "$@"
  exit 23
fi
if [[ -n ${TEST_FIND_FAIL_SYMLINK_SCAN:-} &&
  ${1:-} == /* && ${2:-} == -type && ${3:-} == l ]]; then
  exit 24
fi
exec "${TEST_REAL_FIND}" "$@"
FIND_STUB

cat >"${STUB_BIN}/wc" <<'WC_STUB'
#!/bin/bash
set -euo pipefail

call_index=0
if [[ -f ${TEST_WC_COUNT} ]]; then
  IFS= read -r call_index <"${TEST_WC_COUNT}"
fi
((call_index += 1))
printf '%s\n' "${call_index}" >"${TEST_WC_COUNT}"
"${TEST_REAL_WC}" "$@"
if [[ -n ${TEST_WC_FAIL_CALL:-} && ${call_index} == "${TEST_WC_FAIL_CALL}" ]]; then
  exit 25
fi
WC_STUB
chmod 755 \
  "${STUB_BIN}/batch-git" \
  "${STUB_BIN}/srun" \
  "${STUB_BIN}/fake-python" \
  "${STUB_BIN}/find" \
  "${STUB_BIN}/wc"

readonly BATCH_PATH=${TEST_DIRECTORY}/batch-path
mkdir -p "${BATCH_PATH}"
ln -s "${STUB_BIN}/batch-git" "${BATCH_PATH}/git"
ln -s "${STUB_BIN}/find" "${BATCH_PATH}/find"
ln -s "${STUB_BIN}/wc" "${BATCH_PATH}/wc"
for command_path in /usr/bin/awk /usr/bin/basename /usr/bin/cat /usr/bin/cmp /usr/bin/cut /usr/bin/diff /usr/bin/dirname /usr/bin/env /usr/bin/grep /usr/bin/head /bin/chmod /bin/mkdir /bin/mv /bin/rm /bin/rmdir /usr/bin/mktemp /usr/bin/sed /usr/bin/sort /usr/bin/stat /usr/bin/tar /usr/bin/test /usr/bin/tr; do
  [[ -x ${command_path} ]] || continue
  ln -sf "${command_path}" "${BATCH_PATH}/$(basename -- "${command_path}")"
done
if command -v sha256sum >/dev/null 2>&1; then
  ln -sf "$(command -v sha256sum)" "${BATCH_PATH}/sha256sum"
else
  cat >"${BATCH_PATH}/sha256sum" <<'SHA256SUM_STUB'
#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --check ]]; then
  shift
  [[ ${1:-} != --strict ]] || shift
  manifest=$1
  while read -r digest path; do
    actual=$(shasum -a 256 "${path}" | awk '{print $1}')
    [[ ${actual} == "${digest}" ]] || exit 1
  done <"${manifest}"
else
  shasum -a 256 "$@"
fi
SHA256SUM_STUB
  chmod 755 "${BATCH_PATH}/sha256sum"
  ln -sf /usr/bin/shasum "${BATCH_PATH}/shasum"
fi

mutate_container_receipt() {
  local mode=$1
  local receipt
  local target

  write_valid_container_receipts
  case ${mode} in
    valid) ;;
    corrupt_metadata) printf '%s\n' corrupt >"${TEST_CONTAINER}.metadata.txt" ;;
    missing_metadata) rm -f -- "${TEST_CONTAINER}.metadata.txt" ;;
    duplicate_metadata) printf '%s\n' 'source_cluster=ptyche' >>"${TEST_CONTAINER}.metadata.txt" ;;
    symlink_metadata)
      receipt=${TEST_CONTAINER}.metadata.txt
      target=${TEST_DIRECTORY}/metadata-symlink-target
      cp "${receipt}" "${target}"
      rm -f -- "${receipt}"
      ln -s "${target}" "${receipt}"
      ;;
    corrupt_completion) printf '%s\n' corrupt >"${TEST_CONTAINER}.complete" ;;
    missing_completion) rm -f -- "${TEST_CONTAINER}.complete" ;;
    duplicate_completion) printf '%s\n' "sha256=${EXPECTED_IMAGE_SHA256}" >>"${TEST_CONTAINER}.complete" ;;
    symlink_completion)
      receipt=${TEST_CONTAINER}.complete
      target=${TEST_DIRECTORY}/completion-symlink-target
      cp "${receipt}" "${target}"
      rm -f -- "${receipt}"
      ln -s "${target}" "${receipt}"
      ;;
    *) die "Unknown receipt mutation: ${mode}" ;;
  esac
}

run_batch() {
  local mode=${1:-success}
  local archive_root=${2:-${ARCHIVE_ROOT}}
  local semantic_head=${3:-${SEMANTIC_SHA}}
  local semantic_dirty=${4:-}
  local python_version=${5:-3.13.7}
  local receipt_mode=${6:-valid}
  local injected_authorization=${7:-}
  local pristine_archive_root=${8:-${ARCHIVE_ROOT}}
  local deferred_import_module=${9:-}
  local deferred_import_pass=${10:-}
  local output_root=${TEST_DIRECTORY}/batch-output
  local scratch_directory=${TEST_JOB_SCRATCH_DIRECTORY}
  local run_calls=${TEST_DIRECTORY}/batch-run-calls

  rm -rf -- "${output_root}" "${scratch_directory}" "${run_calls}"
  mkdir -p "${output_root}/logs" "${run_calls}" "${TEST_SCRATCH_BASE}"
  mutate_container_receipt "${receipt_mode}"
  env -i \
    "PATH=${BATCH_PATH}:/bin:/usr/bin:/sbin" \
    "SCRIPT_ROOT=${TOOL_ROOT}" \
    "EXPECTED_TOOLING_SHA=${TOOLING_SHA}" \
    "PRODUCT_WORKTREE=${SEMANTIC_ROOT}" \
    "EXPECTED_PRODUCT_SHA=${SEMANTIC_SHA}" \
    SLURM_JOB_ID=98765 \
    "TEST_BATCH_PATH=${BATCH_PATH}:/bin:/usr/bin:/sbin" \
    "TEST_CONTAINER=${TEST_CONTAINER}" \
    "TEST_CONTAINER_PYTHON=${STUB_BIN}/fake-python" \
    "TEST_OUTPUT_ROOT=${output_root}" \
    "TEST_MANIFEST_SHA256=${TEST_MANIFEST_SHA256}" \
    "TEST_SCRATCH_ROOT=${TEST_SCRATCH_ROOT}" \
    "TEST_USER_SCRATCH_ROOT=${TEST_USER_SCRATCH_ROOT}" \
    "TEST_SCRATCH_BASE=${TEST_SCRATCH_BASE}" \
    "TEST_SCRATCH_DIRECTORY=${scratch_directory}" \
    "TEST_ESCAPED_LEAF_PATH=${TEST_ESCAPED_LEAF_PATH}" \
    "TEST_SRUN=${STUB_BIN}/srun" \
    "TEST_SRUN_LOG=${run_calls}/srun.log" \
    "TEST_SRUN_ENV=${run_calls}/srun.env" \
    "TEST_SRUN_CALLED=${run_calls}/srun.called" \
    "TEST_BATCH_GIT_LOG=${run_calls}/git.log" \
    "TEST_BATCH_TOOL_HEAD=${TOOLING_SHA}" \
    "TEST_BATCH_SEMANTIC_HEAD=${semantic_head}" \
    "TEST_BATCH_SEMANTIC_DIRTY=${semantic_dirty}" \
    "TEST_PRISTINE_ARCHIVE_ROOT=${pristine_archive_root}" \
    "TEST_ARCHIVE_ROOT=${archive_root}" \
    "TEST_PYTHON_VERSION=${python_version}" \
    "TEST_REAL_PYTHON=${REAL_PYTHON}" \
    "TEST_PYTHON_LOG=${run_calls}/python.log" \
    "TEST_STAGE_COUNT=${run_calls}/stage.count" \
    "TEST_STAGE_MODE=${mode}" \
    "TEST_GOLDEN_TREE=${GOLDEN_TREE}" \
    "TEST_REAL_FIND=$(command -v find)" \
    "TEST_REAL_WC=$(command -v wc)" \
    "TEST_WC_COUNT=${run_calls}/wc.count" \
    "TEST_WC_FAIL_CALL=${TEST_WC_FAIL_CALL:-}" \
    "TEST_FIND_FAIL_TOPOLOGY=${TEST_FIND_FAIL_TOPOLOGY:-}" \
    "TEST_FIND_FAIL_SYMLINK_SCAN=${TEST_FIND_FAIL_SYMLINK_SCAN:-}" \
    "TEST_DEFERRED_IMPORT_MODULE=${deferred_import_module}" \
    "TEST_DEFERRED_IMPORT_PASS=${deferred_import_pass}" \
    "HF_TOKEN=${injected_authorization}" \
    "HUGGING_FACE_HUB_TOKEN=${injected_authorization}" \
    "HF_API_TOKEN=${injected_authorization}" \
    "HUGGINGFACE_TOKEN=${injected_authorization}" \
    "NVIDIA_API_KEY=${injected_authorization}" \
    "NGC_API_KEY=${injected_authorization}" \
    "AWS_ACCESS_KEY_ID=${injected_authorization}" \
    "AWS_SECRET_ACCESS_KEY=${injected_authorization}" \
    "AWS_SESSION_TOKEN=${injected_authorization}" \
    bash "${TEST_BATCH_PROBE_OVERRIDE:-${BATCH_PROBE}}"
}

run_batch success
readonly SUCCESS_OUTPUT=${TEST_DIRECTORY}/batch-output/sha256-${TEST_MANIFEST_SHA256}
test -d "${SUCCESS_OUTPUT}"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2
require_literal 'second_stage_network=blocked' "${TEST_DIRECTORY}/batch-run-calls/python.log"
require_literal "--container-image=${TEST_CONTAINER}" "${TEST_DIRECTORY}/batch-run-calls/srun.log"
require_literal '--container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch' "${TEST_DIRECTORY}/batch-run-calls/srun.log"
forbid_literal '--gpus' "${TEST_DIRECTORY}/batch-run-calls/srun.log"
test ! -e "${TEST_JOB_SCRATCH_DIRECTORY}"
test "$(sha256_file "${SEMANTIC_ROOT}/must-remain-unchanged")" = "${SEMANTIC_SENTINEL_SHA256}"

readonly -a SCRATCH_SIBLINGS=(
  "${TEST_SCRATCH_BASE}/must-preserve-sibling-a"
  "${TEST_SCRATCH_BASE}/must-preserve-sibling-b"
)
for scratch_sibling in "${SCRATCH_SIBLINGS[@]}"; do
  mkdir -p "${scratch_sibling}"
  printf '%s\n' preserve >"${scratch_sibling}/sentinel"
done
assert_scratch_siblings() {
  local scratch_sibling

  for scratch_sibling in "${SCRATCH_SIBLINGS[@]}"; do
    test "$(cat "${scratch_sibling}/sentinel")" = preserve
  done
}
run_batch snapshot_vanishes_after_publish
test -d "${SUCCESS_OUTPUT}"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2
test ! -e "${TEST_JOB_SCRATCH_DIRECTORY}"
assert_scratch_siblings

expect_failure \
  batch_whole_scratch_vanishes \
  run_batch \
  whole_scratch_vanishes_after_publish
require_literal \
  'job-owned scratch lifecycle violated during container stage' \
  "${TEST_DIRECTORY}/batch_whole_scratch_vanishes.stderr"
forbid_literal 'Traceback' "${TEST_DIRECTORY}/batch_whole_scratch_vanishes.stderr"
forbid_literal 'FileNotFoundError' "${TEST_DIRECTORY}/batch_whole_scratch_vanishes.stderr"
test -d "${SUCCESS_OUTPUT}"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 1
test ! -e "${TEST_JOB_SCRATCH_DIRECTORY}"
assert_scratch_siblings

TEST_BATCH_PROBE_OVERRIDE=${EXIT_WINDOW_DELETE_BATCH_PROBE} \
  expect_failure_status \
  batch_exit_window_scratch_vanishes \
  97 \
  run_batch \
  success
require_literal \
  'could not quarantine job scratch' \
  "${TEST_DIRECTORY}/batch_exit_window_scratch_vanishes.stderr"
test -d "${SUCCESS_OUTPUT}"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2
test ! -e "${TEST_JOB_SCRATCH_DIRECTORY}"
assert_scratch_siblings

for module_identity_mode in leaf_origin_escaped leaf_module_replaced; do
  expect_failure \
    "batch_${module_identity_mode}" \
    run_batch \
    "${module_identity_mode}"
  require_literal \
    'verified metadata module' \
    "${TEST_DIRECTORY}/batch_${module_identity_mode}.stderr"
  test -d "${SUCCESS_OUTPUT}"
  assert_scratch_siblings
done

run_batch success "${ARCHIVE_ROOT}" "${SEMANTIC_SHA}" '' 3.13.7 valid scheduler-injected-authorization
test -d "${SUCCESS_OUTPUT}"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2
test -f "${TEST_DIRECTORY}/batch-run-calls/srun.env"
test ! -L "${TEST_DIRECTORY}/batch-run-calls/srun.env"
forbid_literal scheduler-injected-authorization "${TEST_DIRECTORY}/batch-run-calls/srun.env"

for receipt_mode in \
  corrupt_metadata \
  missing_metadata \
  duplicate_metadata \
  symlink_metadata \
  corrupt_completion \
  missing_completion \
  duplicate_completion \
  symlink_completion; do
  expect_failure "batch_${receipt_mode}" run_batch success "${ARCHIVE_ROOT}" "${SEMANTIC_SHA}" '' 3.13.7 "${receipt_mode}"
  test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
done

expect_failure batch_wrong_sha run_batch success "${ARCHIVE_ROOT}" "${OTHER_SHA}"
test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
expect_failure batch_dirty_semantic run_batch success "${ARCHIVE_ROOT}" "${SEMANTIC_SHA}" dirty
test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
expect_failure batch_altered_blob run_batch success "${ALTERED_ARCHIVE_ROOT}"
test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
expect_failure batch_altered_leaf_blob run_batch success "${ALTERED_LEAF_ARCHIVE_ROOT}"
test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"

expect_failure \
  batch_primary_exception \
  run_batch \
  success \
  "${PRIMARY_FAILURE_ARCHIVE_ROOT}" \
  "${SEMANTIC_SHA}" \
  '' \
  3.13.7 \
  valid \
  '' \
  "${PRIMARY_FAILURE_ARCHIVE_ROOT}"
require_literal \
  'primary deferred fixture failure' \
  "${TEST_DIRECTORY}/batch_primary_exception.stderr"
forbid_literal \
  'metadata artifact identity leaf was not loaded' \
  "${TEST_DIRECTORY}/batch_primary_exception.stderr"
test -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
test ! -e "${TEST_DIRECTORY}/batch-run-calls/stage.count"

while IFS='|' read -r blocked_case_id blocked_module; do
  attempted_root=${TEST_DIRECTORY}/blocked-attempt-${blocked_case_id}
  cp -R "${ARCHIVE_ROOT}" "${attempted_root}"
  printf '%s\n' \
    'try:' \
    "    __import__(\"${blocked_module}\")" \
    'except ImportError:' \
    '    pass' \
    >>"${attempted_root}/tools/stage_precision_policy_source_metadata.py"
  expect_failure \
    "batch_blocked_attempt_${blocked_case_id}" \
    run_batch \
    success \
    "${attempted_root}" \
    "${SEMANTIC_SHA}" \
    '' \
    3.13.7 \
    valid \
    '' \
    "${attempted_root}"
  require_literal \
    "blocked imports attempted: ${blocked_module}" \
    "${TEST_DIRECTORY}/batch_blocked_attempt_${blocked_case_id}.stderr"
  test -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
  test ! -e "${TEST_DIRECTORY}/batch-run-calls/stage.count"

  loaded_root=${TEST_DIRECTORY}/blocked-loaded-${blocked_case_id}
  cp -R "${ARCHIVE_ROOT}" "${loaded_root}"
  printf '%s\n' \
    'import sys' \
    "sys.modules[\"${blocked_module}\"] = object()" \
    >>"${loaded_root}/tools/stage_precision_policy_source_metadata.py"
  expect_failure \
    "batch_blocked_loaded_${blocked_case_id}" \
    run_batch \
    success \
    "${loaded_root}" \
    "${SEMANTIC_SHA}" \
    '' \
    3.13.7 \
    valid \
    '' \
    "${loaded_root}"
  require_literal \
    "blocked imports loaded: ${blocked_module}" \
    "${TEST_DIRECTORY}/batch_blocked_loaded_${blocked_case_id}.stderr"
  test -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
  test ! -e "${TEST_DIRECTORY}/batch-run-calls/stage.count"
done <<'BLOCKED_MODULES'
nemo_rl|nemo_rl
pydantic|pydantic
typing_extensions|typing_extensions
capture|tools.capture_precision_policy_source_evidence
BLOCKED_MODULES

expect_failure \
  batch_deferred_import_first_pass \
  run_batch \
  success \
  "${ARCHIVE_ROOT}" \
  "${SEMANTIC_SHA}" \
  '' \
  3.13.7 \
  valid \
  '' \
  "${ARCHIVE_ROOT}" \
  pydantic \
  1
require_literal \
  'blocked imports attempted: pydantic' \
  "${TEST_DIRECTORY}/batch_deferred_import_first_pass.stderr"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 1

expect_failure \
  batch_deferred_import_reuse_pass \
  run_batch \
  success \
  "${ARCHIVE_ROOT}" \
  "${SEMANTIC_SHA}" \
  '' \
  3.13.7 \
  valid \
  '' \
  "${ARCHIVE_ROOT}" \
  tools.capture_precision_policy_source_evidence \
  2
require_literal \
  'blocked imports attempted: tools.capture_precision_policy_source_evidence' \
  "${TEST_DIRECTORY}/batch_deferred_import_reuse_pass.stderr"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2

expect_failure_status batch_first_pass_failure 71 run_batch first_fail
test ! -e "${TEST_DIRECTORY}/batch-output/sha256-${TEST_MANIFEST_SHA256}"
test -z "$(find "${TEST_DIRECTORY}/batch-output" -name '.precision-policy-source-metadata-*' -print -quit)"
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 1

for bad_mode in wrong_manifest wrong_topology symlink; do
  expect_failure "batch_${bad_mode}" run_batch "${bad_mode}"
  test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2
done

TEST_FIND_FAIL_TOPOLOGY=1 expect_failure \
  batch_topology_find_failure \
  run_batch \
  success
TEST_FIND_FAIL_SYMLINK_SCAN=1 expect_failure \
  batch_symlink_find_failure \
  run_batch \
  success
TEST_WC_FAIL_CALL=1 expect_failure \
  batch_metadata_wc_failure \
  run_batch \
  success
test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
TEST_WC_FAIL_CALL=2 expect_failure \
  batch_completion_wc_failure \
  run_batch \
  success
test ! -e "${TEST_DIRECTORY}/batch-run-calls/srun.called"
TEST_WC_FAIL_CALL=3 expect_failure \
  batch_manifest_wc_failure \
  run_batch \
  success
test "$(cat "${TEST_DIRECTORY}/batch-run-calls/stage.count")" = 2

expect_failure batch_old_python run_batch success "${ARCHIVE_ROOT}" "${SEMANTIC_SHA}" '' 3.11.9
test ! -e "${TEST_DIRECTORY}/batch-run-calls/stage.count"

echo 'OCI-HSG metadata staging tooling checks passed'
