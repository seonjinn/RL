#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=qwen3_30ba3b_fixed_vs_always_stable_200step_20260827
readonly SOURCE_ROOT=/home/sna/nemorl-q30-fixed-always-product-20260827
readonly SOURCE_SHA=4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/${EXPERIMENT}
readonly ACCOUNT=nemotron_n3_post
readonly WANDB_GROUP=q30ba3b-fixed-vs-always-stable-200step-20260827
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly CAPTURE_SIZES='[1,2,4,8,12,16,24,32,40,48]'

usage() {
  echo "usage: $0 --assert-harness-clean|--assert-capture-coverage|--emit-manifest VARIANT|--render-sbatch VARIANT|--test-only VARIANT|--submit VARIANT" >&2
  exit 2
}

die() { echo "Q30_FIXED_ALWAYS_FAIL_CLOSED: $*" >&2; exit 1; }

valid_variant() {
  case "$1" in
    dflash-fixed|dflash-always|dspark-fixed|dspark-always) ;;
    *) usage ;;
  esac
}

drafter_for() { printf '%s\n' "${1%%-*}"; }

checkpoint_for() {
  case "$(drafter_for "$1")" in
    dflash) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391 ;;
    dspark) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391 ;;
  esac
}

config_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$1.yaml"
}

input_identity() {
  local variant="$1"
  python3 - \
    "${SCRIPT_DIR}/submit_qwen3_30ba3b_fixed_vs_always_200step.sh" \
    "${SCRIPT_DIR}/check_checkpoint_state_dict.py" \
    "${SCRIPT_DIR}/verify_composed_configs.py" \
    "${SCRIPT_DIR}/configs/${variant}.yaml" <<'PY'
import hashlib
import json
import pathlib
import sys

paths = [pathlib.Path(argument) for argument in sys.argv[1:]]
names = (
    "launcher_sha256",
    "checkpoint_checker_sha256",
    "composition_verifier_sha256",
    "config_sha256",
)
print(json.dumps({
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in zip(names, paths, strict=True)
}, sort_keys=True))
PY
}

submission_record() {
  printf '%s\n' "${DURABLE_ROOT}/submissions/${1}-${SOURCE_SHA}-${HARNESS_SHA}.json"
}

run_id() {
  python3 - "$1" <<'PY'
import sys
import uuid

print(f"q30ba3b-stable-200step-{sys.argv[1]}-k5-{uuid.uuid4().hex}")
PY
}

emit_manifest() {
  local variant="$1" run="$2" record identity
  record="$(submission_record "${variant}")"
  identity="$(input_identity "${variant}")"
  python3 - "${variant}" "${run}" "${HARNESS_SHA}" "${record}" "${identity}" <<PY
import json
import sys

print(json.dumps({
    "variant": sys.argv[1],
    "k": 5,
    "source": {"root": "${SOURCE_ROOT}", "sha": "${SOURCE_SHA}"},
    "harness_sha": sys.argv[3],
    "input_identity": json.loads(sys.argv[5]),
    "container": "${CONTAINER}",
    "slurm": {"account": "${ACCOUNT}", "partition": "batch", "time": "04:00:00", "nodes": 4, "gpus_per_node": 4},
    "gates": ["source-clean", "state-dict", "wandb-auth", "cudagraph", "step1", "step2"],
    "max_steps": 200,
    "wandb_project": "sna-specdec",
    "wandb_group": "${WANDB_GROUP}",
    "wandb_reuse": "never",
    "wandb_run_id": sys.argv[2],
    "submission_record": sys.argv[4],
}, sort_keys=True))
PY
}

assert_capture_coverage() {
  python3 - <<'PY'
import json

capture_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
shape_to_bucket = {
    shape: next(bucket for bucket in capture_sizes if bucket >= shape)
    for shape in range(1, 49)
}
print(json.dumps({"capture_sizes": capture_sizes, "shape_to_bucket": shape_to_bucket}, sort_keys=True))
PY
}

source_guard() {
  test -e "${SOURCE_ROOT}/.git" || die "missing product source ${SOURCE_ROOT}"
  test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}" || die "product source SHA drift"
  test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || die "product source is dirty"
  if git -C "${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "product source has unresolved submodule gitlinks"
  fi
  test -z "$(git -C "${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "product source submodule is dirty"
  test -r "${CONTAINER}" || die "missing immutable container"
}

harness_guard() {
  local worktree_root current_head
  worktree_root="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" || die "harness is not in a git worktree"
  worktree_root="$(cd -- "${worktree_root}" && pwd -P)"
  case "${SCRIPT_DIR}/" in
    "${worktree_root}/"*) ;;
    *) die "harness path is outside its git worktree" ;;
  esac
  current_head="$(git -C "${worktree_root}" rev-parse HEAD)" || die "cannot resolve harness HEAD"
  test "${current_head}" = "${HARNESS_SHA}" || die "harness HEAD changed during invocation"
  test -z "$(git -C "${worktree_root}" status --porcelain=v1 --untracked-files=all)" || die "harness worktree is dirty"
}

preflight() {
  local variant="$1"
  harness_guard
  source_guard
  python3 "${SCRIPT_DIR}/check_checkpoint_state_dict.py" \
    --variant "$(drafter_for "${variant}")" \
    --checkpoint "$(checkpoint_for "${variant}")"
}

write_sbatch() {
  local variant="$1" root="$2" run artifact_dir sbatch_path config checkpoint drafter
  run="$(run_id "${variant}")"
  artifact_dir="${root}/artifacts/${run}"
  sbatch_path="${artifact_dir}/job.sbatch"
  config="${SCRIPT_DIR}/configs/${variant}.yaml"
  checkpoint="$(checkpoint_for "${variant}")"
  drafter="$(drafter_for "${variant}")"
  mkdir -p "${artifact_dir}"
  cp "${config}" "${artifact_dir}/resolved-input-${variant}.yaml"
  cp "${SCRIPT_DIR}/check_checkpoint_state_dict.py" "${artifact_dir}/check_checkpoint_state_dict.py"
  cp "${SCRIPT_DIR}/verify_composed_configs.py" "${artifact_dir}/verify_composed_configs.py"
  cat >"${artifact_dir}/driver.sh" <<DRIVER
#!/usr/bin/env bash
set -euo pipefail
readonly SOURCE_ROOT="${SOURCE_ROOT}"
readonly SOURCE_SHA="${SOURCE_SHA}"
readonly ARTIFACT_DIR="${artifact_dir}"
readonly CONFIG="${artifact_dir}/resolved-input-${variant}.yaml"
readonly CHECKPOINT="${checkpoint}"
readonly DRAFTER="${drafter}"
readonly VARIANT="${variant}"
readonly WANDB_ID="${run}"

die() { echo "Q30_FIXED_ALWAYS_FAIL_CLOSED: \$*" >&2; exit 1; }
source_guard() {
  test -e "\${SOURCE_ROOT}/.git" || die "missing product source \${SOURCE_ROOT}"
  test "\$(git -C "\${SOURCE_ROOT}" rev-parse HEAD)" = "\${SOURCE_SHA}" || die "product source SHA drift"
  test -z "\$(git -C "\${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || die "product source is dirty"
  if git -C "\${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then die "product source has unresolved submodule gitlinks"; fi
  test -z "\$(git -C "\${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "product source submodule is dirty"
}
wait_for_gate() {
  local pattern="\$1" marker="\$2" deadline="\$((SECONDS + 2700))"
  while kill -0 "\${train_pid}" 2>/dev/null; do
    if grep -qE "\${pattern}" "\${train_log}"; then echo "\${marker}" | tee -a "\${ARTIFACT_DIR}/gates.log"; return; fi
    (( SECONDS < deadline )) || { kill -- "-\${train_pid}" 2>/dev/null || true; wait "\${train_pid}" || true; die "timed out waiting for \${marker}"; }
    sleep 10
  done
  wait "\${train_pid}" || die "training ended before \${marker}"
  grep -qE "\${pattern}" "\${train_log}" || die "missing \${marker}"
  echo "\${marker}" | tee -a "\${ARTIFACT_DIR}/gates.log"
}

source_guard
test -n "\${WANDB_API_KEY:-}" || die "WANDB_API_KEY is absent inside the job container"
python3 - <<'PY'
import base64
import json
import os
import urllib.request

token = base64.b64encode(f"api:{os.environ['WANDB_API_KEY']}".encode()).decode()
request = urllib.request.Request(
    "https://api.wandb.ai/graphql",
    data=json.dumps({"query": "query Viewer { viewer { entity } }"}).encode(),
    headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
if not payload.get("data", {}).get("viewer"):
    raise SystemExit("W&B authenticated viewer preflight failed")
PY
echo WANDB_AUTH_GATE_PASS | tee "\${ARTIFACT_DIR}/gates.log"
(cd "\${SOURCE_ROOT}" && NRL_FORCE_REBUILD_VENVS=true uv run --with hydra-core==1.3.2 python3 "\${ARTIFACT_DIR}/verify_composed_configs.py" --source-root "\${SOURCE_ROOT}" --config "\${CONFIG}") | tee "\${ARTIFACT_DIR}/composed-config.json"
python3 "\${ARTIFACT_DIR}/check_checkpoint_state_dict.py" --variant "\${DRAFTER}" --checkpoint "\${CHECKPOINT}" | tee -a "\${ARTIFACT_DIR}/gates.log"
export WANDB_RUN_ID="\${WANDB_ID}"
export WANDB_PROJECT=sna-specdec
export WANDB_MODE=online
train_log="\${ARTIFACT_DIR}/train.log"
setsid bash -c "set -o pipefail; cd '${SOURCE_ROOT}'; NRL_FORCE_REBUILD_VENVS=true uv run --with hydra-core==1.3.2 examples/run_grpo.py --config '${artifact_dir}/resolved-input-${variant}.yaml' checkpointing.checkpoint_dir='${artifact_dir}/checkpoints' ++policy.generation.vllm_kwargs.max_num_seqs=8 ++policy.generation.vllm_kwargs.compilation_config.backend=eager ++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE ++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${CAPTURE_SIZES} logger.log_dir='${artifact_dir}/logs' logger.wandb_enabled=true logger.wandb.project=sna-specdec +logger.wandb.group=${WANDB_GROUP} logger.wandb.name='${run}' 2>&1 | tee '${artifact_dir}/train.log'" &
train_pid=\$!
wait_for_gate 'Capturing CUDA graphs.*100%|Graph capturing finished' CUDAGRAPH_GATE_PASS
wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*200' STEP1_GATE_PASS
wait_for_gate 'Step[[:space:]]+2[[:space:]]*/[[:space:]]*200' STEP2_GATE_PASS
wait "\${train_pid}"
python3 - "\${ARTIFACT_DIR}/run-complete.json" "\${VARIANT}" "\${WANDB_ID}" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "state": "complete",
    "variant": sys.argv[2],
    "wandb_entity": "nvidia",
    "wandb_project": "sna-specdec",
    "wandb_group": "q30ba3b-fixed-vs-always-stable-200step-20260827",
    "wandb_run_id": sys.argv[3],
}, sort_keys=True) + "\n")
PY
DRIVER
  chmod 700 "${artifact_dir}/driver.sh"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=sna-q30-fa-${variant}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:\${PATH}"
command -v scontrol >/dev/null
command -v sinfo >/dev/null
command -v srun >/dev/null
export CONTAINER="${CONTAINER}"
export MOUNTS="/lustre:/lustre,/home:/home"
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export ARTIFACT_DIR="${artifact_dir}"
export BASE_LOG_DIR="${artifact_dir}"
export NRL_FORCE_REBUILD_VENVS=true
export WANDB_API_KEY="\${WANDB_API_KEY:?WANDB_API_KEY must be exported at submission}"
export COMMAND='bash "${artifact_dir}/driver.sh"'
exec bash "${SOURCE_ROOT}/ray.sub"
SBATCH
  chmod 700 "${sbatch_path}"
  printf '%s\n' "${sbatch_path}"
}

write_testonly_receipt() {
  local variant="$1" sbatch_output="$2" identity="$3" receipt="${DURABLE_ROOT}/preflight/${variant}.json"
  mkdir -p "$(dirname "${receipt}")"
  python3 - "${receipt}" "${variant}" "$(config_sha "${variant}")" "${sbatch_output}" "${identity}" <<PY
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "config_sha": sys.argv[3],
    "harness_sha": "${HARNESS_SHA}",
    "input_identity": json.loads(sys.argv[5]),
    "source_sha": "${SOURCE_SHA}",
    "test_only_output": sys.argv[4],
    "variant": sys.argv[2],
}, sort_keys=True) + "\n")
PY
}

require_testonly_receipt() {
  local variant="$1" identity="$2"
  python3 - "${DURABLE_ROOT}/preflight" "${HARNESS_SHA}" "${variant}" "$(config_sha "${variant}")" "${identity}" <<PY
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
variant = sys.argv[3]
receipt = json.loads((root / f"{variant}.json").read_text())
expected = {
    "config_sha": sys.argv[4],
    "harness_sha": sys.argv[2],
    "input_identity": json.loads(sys.argv[5]),
    "source_sha": "${SOURCE_SHA}",
    "variant": variant,
}
if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("test_only_output"):
    raise SystemExit(f"invalid test-only receipt for {variant}")
PY
}

create_submitting_record() {
  local record="$1" variant="$2" run="$3" artifact_dir="$4" sbatch_path="$5" identity="$6"
  python3 - "${record}" "${variant}" "${SOURCE_SHA}" "${HARNESS_SHA}" "${run}" "${artifact_dir}" "${sbatch_path}" "${identity}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "artifact_dir": sys.argv[6],
    "harness_sha": sys.argv[4],
    "input_identity": json.loads(sys.argv[8]),
    "run_id": sys.argv[5],
    "sbatch_path": sys.argv[7],
    "schema_version": 1,
    "source_sha": sys.argv[3],
    "state": "submitting",
    "variant": sys.argv[2],
}
with path.open("x") as record:
    record.write(json.dumps(payload, sort_keys=True) + "\n")
    record.flush()
    os.fsync(record.fileno())
directory = os.open(path.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

submit_and_finalize_record() {
  local record="$1" variant="$2" run="$3" artifact_dir="$4" sbatch_path="$5" identity="$6"
  python3 - "${record}" "${variant}" "${SOURCE_SHA}" "${HARNESS_SHA}" "${run}" "${artifact_dir}" "${sbatch_path}" "${identity}" <<'PY'
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import uuid

MAX_CAPTURE_BYTES = 8192
MAX_SAFE_LINES = 8
SBATCH_TIMEOUT_SECONDS = 120

path = pathlib.Path(sys.argv[1])
submitting = {
    "artifact_dir": sys.argv[6],
    "harness_sha": sys.argv[4],
    "input_identity": json.loads(sys.argv[8]),
    "run_id": sys.argv[5],
    "sbatch_path": sys.argv[7],
    "schema_version": 1,
    "source_sha": sys.argv[3],
    "state": "submitting",
    "variant": sys.argv[2],
}
if json.loads(path.read_text()) != submitting:
    raise SystemExit("submission receipt changed before scheduler acceptance was recorded")


def replace_receipt(payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as receipt:
            receipt.write(json.dumps(payload, sort_keys=True) + "\n")
            receipt.flush()
            os.fsync(receipt.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


captured = bytearray()
output_bytes = [0]
timed_out = False
try:
    process = subprocess.Popen(
        ["sbatch", sys.argv[7]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
except OSError:
    scheduler_exit_status = 127
else:
    assert process.stdout is not None

    def drain_output() -> None:
        while chunk := process.stdout.read(4096):
            output_bytes[0] += len(chunk)
            remaining = MAX_CAPTURE_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])

    drain_thread = threading.Thread(target=drain_output, daemon=True)
    drain_thread.start()
    try:
        scheduler_exit_status = process.wait(timeout=SBATCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
        scheduler_exit_status = 124
    drain_thread.join()

matched_job_ids = re.findall(rb"(?m)^Submitted batch job ([0-9]+)\r?$", bytes(captured))
candidate_job_ids = [match.decode() for match in matched_job_ids[:MAX_SAFE_LINES]]
safe_output = [f"Submitted batch job {job_id}" for job_id in candidate_job_ids]
output_truncated = output_bytes[0] > len(captured) or len(matched_job_ids) > MAX_SAFE_LINES
outcome = {
    **submitting,
    "scheduler_exit_status": scheduler_exit_status,
    "scheduler_output_bytes": output_bytes[0],
    "scheduler_output_truncated": output_truncated,
    "scheduler_safe_output": safe_output,
    "scheduler_timed_out": timed_out,
}
if scheduler_exit_status == 0 and not timed_out and not output_truncated and len(candidate_job_ids) == 1:
    outcome.update({"job_id": candidate_job_ids[0], "state": "accepted"})
    replace_receipt(outcome)
    print(safe_output[0])
else:
    outcome.update({"candidate_job_ids": candidate_job_ids, "state": "ambiguous"})
    replace_receipt(outcome)
    print(
        f"scheduler submission outcome is ambiguous (exit_status={scheduler_exit_status}); reconcile {path} before retrying",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

mode="${1:-}"
case "${mode}" in
  --assert-harness-clean)
    [[ $# -eq 1 ]] || usage
    harness_guard
    ;;
  --assert-capture-coverage)
    [[ $# -eq 1 ]] || usage
    assert_capture_coverage
    ;;
  --emit-manifest|--render-sbatch|--test-only|--submit)
    [[ $# -eq 2 ]] || usage
    variant="$2"
    valid_variant "${variant}"
    case "${mode}" in
      --emit-manifest) emit_manifest "${variant}" "$(run_id "${variant}")" ;;
      --render-sbatch) write_sbatch "${variant}" "${Q30_FIXED_ALWAYS_RENDER_ROOT:?Q30_FIXED_ALWAYS_RENDER_ROOT is required for render}" ;;
      --test-only)
        preflight "${variant}"
        identity="$(input_identity "${variant}")"
        sbatch_output="$(sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1)"
        test "$(input_identity "${variant}")" = "${identity}" || die "harness inputs changed during scheduler validation"
        write_testonly_receipt "${variant}" "${sbatch_output}" "${identity}"
        printf '%s\n' "${sbatch_output}"
        ;;
      --submit)
        preflight "${variant}"
        identity="$(input_identity "${variant}")"
        require_testonly_receipt "${variant}" "${identity}"
        record="$(submission_record "${variant}")"
        sbatch_path="$(write_sbatch "${variant}" "${DURABLE_ROOT}")"
        artifact_dir="$(dirname -- "${sbatch_path}")"
        run="$(basename -- "${artifact_dir}")"
        mkdir -p "$(dirname "${record}")"
        if ! create_submitting_record "${record}" "${variant}" "${run}" "${artifact_dir}" "${sbatch_path}" "${identity}" 2>/dev/null; then
          die "actual ${variant} submission receipt already exists; reconcile it before retrying"
        fi
        harness_guard
        test "$(input_identity "${variant}")" = "${identity}" || die "harness inputs changed before scheduler submission"
        if ! submit_and_finalize_record "${record}" "${variant}" "${run}" "${artifact_dir}" "${sbatch_path}" "${identity}"; then
          die "scheduler did not produce a confirmed ${variant} submission; reconcile the receipt before retrying"
        fi
        ;;
    esac
    ;;
  *) usage ;;
esac
