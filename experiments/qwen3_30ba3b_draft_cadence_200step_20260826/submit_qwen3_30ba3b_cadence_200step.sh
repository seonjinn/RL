#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=qwen3_30ba3b_draft_cadence_200step_20260826
readonly SOURCE_ROOT=/home/sna/nemorl-q30-cadence-product-20260826
readonly SOURCE_SHA=716930391e21c01bc7a79273c45bc407752c9c4a
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/${EXPERIMENT}
readonly ACCOUNT=nemotron_n3_post
readonly WANDB_GROUP=q30ba3b-draft-cadence-200step-20260826
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
usage() {
  echo "usage: $0 --emit-manifest VARIANT|--render-sbatch VARIANT|--test-only VARIANT|--submit VARIANT" >&2
  exit 2
}

die() { echo "Q30_CADENCE_FAIL_CLOSED: $*" >&2; exit 1; }

valid_variant() {
  case "$1" in
    dflash-fixed5|dflash-fixed10|dflash-fixed20|dspark-fixed5|dspark-fixed10|dspark-fixed20) ;;
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

submission_record() {
  printf '%s\n' "${DURABLE_ROOT}/submissions/${1}-${SOURCE_SHA}-${HARNESS_SHA}.json"
}

run_id() {
  python3 - "$1" <<'PY'
import sys
import uuid

print(f"q30ba3b-200step-{sys.argv[1]}-k5-{uuid.uuid4().hex}")
PY
}

emit_manifest() {
  local variant="$1" run="$2" record
  record="$(submission_record "${variant}")"
  python3 - "${variant}" "${run}" "${HARNESS_SHA}" "${record}" <<PY
import json
import sys

print(json.dumps({
    "variant": sys.argv[1],
    "source": {"root": "${SOURCE_ROOT}", "sha": "${SOURCE_SHA}"},
    "harness_sha": sys.argv[3],
    "container": "${CONTAINER}",
    "slurm": {"account": "${ACCOUNT}", "partition": "batch_long", "time": "18:00:00", "nodes": 4, "gpus_per_node": 4},
    "gates": ["source-clean", "state-dict", "wandb-auth", "cudagraph", "step1", "step2", "draft-refit"],
    "max_steps": 200,
    "wandb_project": "sna-specdec",
    "wandb_group": "${WANDB_GROUP}",
    "wandb_reuse": "never",
    "wandb_run_id": sys.argv[2],
    "submission_record": sys.argv[4],
}, sort_keys=True))
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

preflight() {
  local variant="$1"
  source_guard
  python3 "${SCRIPT_DIR}/check_checkpoint_state_dict.py" \
    --variant "$(drafter_for "${variant}")" \
    --checkpoint "$(checkpoint_for "${variant}")"
}

write_sbatch() {
  local variant="$1" root="$2" run artifact_dir sbatch_path config checkpoint drafter interval
  run="$(run_id "${variant}")"
  artifact_dir="${root}/artifacts/${run}"
  sbatch_path="${artifact_dir}/job.sbatch"
  config="${SCRIPT_DIR}/configs/${variant}.yaml"
  checkpoint="$(checkpoint_for "${variant}")"
  drafter="$(drafter_for "${variant}")"
  interval="${variant##*fixed}"
  mkdir -p "${artifact_dir}"
  cp "${config}" "${artifact_dir}/resolved-input-${variant}.yaml"
  cp "${SCRIPT_DIR}/check_checkpoint_state_dict.py" "${artifact_dir}/check_checkpoint_state_dict.py"
  cp "${SCRIPT_DIR}/prepare_vllm_dspark_fap_overlay.py" "${artifact_dir}/prepare_vllm_dspark_fap_overlay.py"
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
readonly WANDB_ID="${run}"

die() { echo "Q30_CADENCE_FAIL_CLOSED: \$*" >&2; exit 1; }
source_guard() {
  test -e "\${SOURCE_ROOT}/.git" || die "missing product source \${SOURCE_ROOT}"
  test "\$(git -C "\${SOURCE_ROOT}" rev-parse HEAD)" = "\${SOURCE_SHA}" || die "product source SHA drift"
  test -z "\$(git -C "\${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || die "product source is dirty"
  if git -C "\${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then die "product source has unresolved submodule gitlinks"; fi
  test -z "\$(git -C "\${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "product source submodule is dirty"
}
wait_for_gate() {
  local pattern="\$1" marker="\$2" timeout_seconds="\$3" deadline=0
  if (( timeout_seconds > 0 )); then deadline="\$((SECONDS + timeout_seconds))"; fi
  while kill -0 "\${train_pid}" 2>/dev/null; do
    if grep -qE "\${pattern}" "\${train_log}"; then echo "\${marker}" | tee -a "\${ARTIFACT_DIR}/gates.log"; return; fi
    if (( timeout_seconds > 0 && SECONDS >= deadline )); then
      kill -- "-\${train_pid}" 2>/dev/null || true
      wait "\${train_pid}" || true
      die "timed out waiting for \${marker}"
    fi
    sleep 10
  done
  wait "\${train_pid}" || die "training ended before \${marker}"
  grep -qE "\${pattern}" "\${train_log}" || die "missing \${marker}"
  echo "\${marker}" | tee -a "\${ARTIFACT_DIR}/gates.log"
}

source_guard
test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp" || die "missing node-local MCore overlay"
echo MCORE_OVERLAY_GATE_PASS | tee "\${ARTIFACT_DIR}/gates.log"
if [[ "\${DRAFTER}" == dspark ]]; then
  test -f "\${Q30_VLLM_OVERLAY}/dspark-fap-vllm-48167-attention-guard.json" || die "missing DSpark vLLM overlay receipt"
  cp "\${Q30_VLLM_OVERLAY}/dspark-fap-vllm-48167-attention-guard.json" "\${ARTIFACT_DIR}/vllm-dspark-fap-overlay-receipt.json"
  /opt/nemo_rl_venv/bin/python - <<'PY'
import os
from pathlib import Path

import vllm

overlay = Path(os.environ["Q30_VLLM_OVERLAY"]).resolve()
Path(vllm.__file__).resolve().relative_to(overlay)
PY
  echo DSPARK_VLLM_OVERLAY_GATE_PASS | tee -a "\${ARTIFACT_DIR}/gates.log"
else
  /opt/nemo_rl_venv/bin/python - <<'PY'
import os
from pathlib import Path

import vllm

overlay = Path(os.environ["Q30_VLLM_OVERLAY"]).resolve()
try:
    Path(vllm.__file__).resolve().relative_to(overlay)
except ValueError:
    pass
else:
    raise SystemExit("DFlash unexpectedly imported the DSpark vLLM overlay")
PY
  echo STOCK_VLLM_GATE_PASS | tee -a "\${ARTIFACT_DIR}/gates.log"
fi
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
(cd "\${SOURCE_ROOT}" && NRL_FORCE_REBUILD_VENVS=true UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync python3 "\${ARTIFACT_DIR}/verify_composed_configs.py" --source-root "\${SOURCE_ROOT}" --config "\${CONFIG}") | tee "\${ARTIFACT_DIR}/composed-config.json"
python3 "\${ARTIFACT_DIR}/check_checkpoint_state_dict.py" --variant "\${DRAFTER}" --checkpoint "\${CHECKPOINT}" | tee -a "\${ARTIFACT_DIR}/gates.log"
export WANDB_RUN_ID="\${WANDB_ID}"
export WANDB_PROJECT=sna-specdec
export WANDB_MODE=online
train_log="\${ARTIFACT_DIR}/train.log"
setsid bash -c "set -o pipefail; cd '${SOURCE_ROOT}'; NRL_FORCE_REBUILD_VENVS=true UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync examples/run_grpo.py --config '${artifact_dir}/resolved-input-${variant}.yaml' logger.log_dir='${artifact_dir}/logs' logger.wandb_enabled=true logger.wandb.project=sna-specdec +logger.wandb.group=${WANDB_GROUP} logger.wandb.name='${run}' 2>&1 | tee '${artifact_dir}/train.log'" &
train_pid=\$!
wait_for_gate 'Capturing CUDA graphs.*100%|Graph capturing finished' CUDAGRAPH_GATE_PASS 2700
wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*200' STEP1_GATE_PASS 2700
wait_for_gate 'Step[[:space:]]+2[[:space:]]*/[[:space:]]*200' STEP2_GATE_PASS 2700
wait_for_gate 'draft_post_update_refit=complete step=${interval}' DRAFT_REFIT_GATE_PASS 0
wait "\${train_pid}"
DRIVER
  chmod 700 "${artifact_dir}/driver.sh"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=sna-q30-c200-${variant}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch_long
#SBATCH --time=18:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:\${PATH}"
command -v scontrol >/dev/null
command -v sinfo >/dev/null
command -v srun >/dev/null
export CONTAINER="${CONTAINER}"
export MOUNTS="/lustre:/lustre,/home:/home,/raid:/raid"
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export SOURCE_ROOT="${SOURCE_ROOT}"
export Q30_DRAFTER="${drafter}"
export Q30_NODE_ROOT="/raid/scratch/sna/q30-cadence-\${SLURM_JOB_ID}"
export Q30_MCORE_SOURCE="\${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q30_MCORE_OVERLAY="\${Q30_NODE_ROOT}/mcore-overlay"
export Q30_VLLM_OVERLAY="\${Q30_NODE_ROOT}/vllm-overlay"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:\${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q30_MCORE_OVERLAY}"; cp -a "\${Q30_MCORE_SOURCE}/megatron" "\${Q30_MCORE_OVERLAY}/"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"; if [[ "\${Q30_DRAFTER}" == dspark ]]; then /opt/nemo_rl_venv/bin/python "${artifact_dir}/prepare_vllm_dspark_fap_overlay.py" --overlay-root "\${Q30_VLLM_OVERLAY}"; fi'
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
  local variant="$1" sbatch_output="$2" receipt="${DURABLE_ROOT}/preflight/${variant}.json"
  mkdir -p "$(dirname "${receipt}")"
  python3 - "${receipt}" "${variant}" "$(config_sha "${variant}")" "${sbatch_output}" <<PY
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "config_sha": sys.argv[3],
    "harness_sha": "${HARNESS_SHA}",
    "source_sha": "${SOURCE_SHA}",
    "test_only_output": sys.argv[4],
    "variant": sys.argv[2],
}, sort_keys=True) + "\n")
PY
}

require_testonly_receipt() {
  local variant="$1"
  python3 - "${DURABLE_ROOT}/preflight" "${HARNESS_SHA}" "${variant}" "$(config_sha "${variant}")" <<PY
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
variant = sys.argv[3]
receipt = json.loads((root / f"{variant}.json").read_text())
expected = {"config_sha": sys.argv[4], "harness_sha": sys.argv[2], "source_sha": "${SOURCE_SHA}", "variant": variant}
if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("test_only_output"):
    raise SystemExit(f"invalid test-only receipt for {variant}")
PY
}

create_submitting_record() {
  local record="$1" variant="$2" run="$3" artifact_dir="$4" sbatch_path="$5"
  python3 - "${record}" "${variant}" "${SOURCE_SHA}" "${HARNESS_SHA}" "${run}" "${artifact_dir}" "${sbatch_path}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "artifact_dir": sys.argv[6],
    "harness_sha": sys.argv[4],
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
  local record="$1" variant="$2" run="$3" artifact_dir="$4" sbatch_path="$5"
  python3 - "${record}" "${variant}" "${SOURCE_SHA}" "${HARNESS_SHA}" "${run}" "${artifact_dir}" "${sbatch_path}" <<'PY'
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
        ["sbatch", sys.argv[7]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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

matched_job_ids = re.findall(
    rb"(?m)^Submitted batch job ([0-9]+)\r?$", bytes(captured)
)
candidate_job_ids = [match.decode() for match in matched_job_ids[:MAX_SAFE_LINES]]
safe_output = [f"Submitted batch job {job_id}" for job_id in candidate_job_ids]
output_truncated = (
    output_bytes[0] > len(captured) or len(matched_job_ids) > MAX_SAFE_LINES
)
outcome = {
    **submitting,
    "scheduler_exit_status": scheduler_exit_status,
    "scheduler_output_bytes": output_bytes[0],
    "scheduler_output_truncated": output_truncated,
    "scheduler_safe_output": safe_output,
    "scheduler_timed_out": timed_out,
}
if (
    scheduler_exit_status == 0
    and not timed_out
    and not output_truncated
    and len(candidate_job_ids) == 1
):
    outcome.update({"job_id": candidate_job_ids[0], "state": "accepted"})
    replace_receipt(outcome)
    print(safe_output[0])
else:
    outcome.update({"candidate_job_ids": candidate_job_ids, "state": "ambiguous"})
    replace_receipt(outcome)
    print(
        f"scheduler submission outcome is ambiguous (exit_status={scheduler_exit_status}); "
        f"reconcile {path} before retrying",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

mode="${1:-}"
case "${mode}" in
  --emit-manifest|--render-sbatch|--test-only|--submit)
    [[ $# -eq 2 ]] || usage
    variant="$2"
    valid_variant "${variant}"
    case "${mode}" in
      --emit-manifest) emit_manifest "${variant}" "$(run_id "${variant}")" ;;
      --render-sbatch) write_sbatch "${variant}" "${Q30_CADENCE_RENDER_ROOT:?Q30_CADENCE_RENDER_ROOT is required for render}" ;;
      --test-only)
        preflight "${variant}"
        sbatch_output="$(sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1)"
        write_testonly_receipt "${variant}" "${sbatch_output}"
        printf '%s\n' "${sbatch_output}"
        ;;
      --submit)
        preflight "${variant}"
        require_testonly_receipt "${variant}"
        record="$(submission_record "${variant}")"
        sbatch_path="$(write_sbatch "${variant}" "${DURABLE_ROOT}")"
        artifact_dir="$(dirname -- "${sbatch_path}")"
        run="$(basename -- "${artifact_dir}")"
        mkdir -p "$(dirname "${record}")"
        if ! create_submitting_record "${record}" "${variant}" "${run}" "${artifact_dir}" "${sbatch_path}" 2>/dev/null; then
          die "actual ${variant} submission receipt already exists; reconcile it before retrying"
        fi
        if ! submit_and_finalize_record "${record}" "${variant}" "${run}" "${artifact_dir}" "${sbatch_path}"; then
          die "scheduler did not produce a confirmed ${variant} submission; reconcile the receipt before retrying"
        fi
        ;;
    esac
    ;;
  *) usage ;;
esac
