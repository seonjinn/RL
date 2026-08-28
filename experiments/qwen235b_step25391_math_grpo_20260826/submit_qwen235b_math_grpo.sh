#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-q30-cadence-product-20260826
readonly SOURCE_SHA=d5c8bfa987025949699f7cfff188b349480bb8b5
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DRAFTER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/checkpoints/qwen3-235ba22b-base-nemotron-b8-s25391/dspark
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen235b_step25391_math_grpo_20260826
readonly ACCOUNT="${ACCOUNT:-nemotron_n3_post}"
readonly MAX_STEPS="${Q235_MAX_STEPS:-20}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly HARNESS_SHA

usage() {
  echo "usage: $0 --emit-manifest|--emit-submission-record|--validate-arm-contract|--render-sbatch|--test-only|--submit ARM" >&2
  exit 2
}

die() { echo "Q235_STEP25391_FAIL_CLOSED: $*" >&2; exit 1; }

valid_arm() {
  case "$1" in baseline|dspark_k3|dspark_k5|dspark_k7) ;; *) usage ;; esac
}

method_for() {
  case "$1" in
    baseline) printf '\n' ;;
    dspark_*) printf 'dspark\n' ;;
  esac
}

k_for() {
  case "$1" in
    baseline) printf '0\n' ;;
    *_k3) printf '3\n' ;;
    *_k5) printf '5\n' ;;
    *_k7) printf '7\n' ;;
  esac
}

checkpoint_for() {
  case "$1" in
    dspark_*) printf '%s\n' "${DRAFTER_ROOT}" ;;
    baseline) printf '\n' ;;
  esac
}

arm_contract_guard() {
  local arm="$1" checkpoint
  [[ "${arm}" == baseline ]] && return
  checkpoint="$(checkpoint_for "${arm}")"
  [[ "${checkpoint}" == "${DRAFTER_ROOT}" ]] || die "unexpected Base DSpark checkpoint: ${checkpoint}"
}

config_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$1.yaml"
}

submission_record() {
  local arm="$1"
  printf '%s/submissions/%s-steps%s-%s-%s-%s.json\n' \
    "${DURABLE_ROOT}" "${arm}" "${MAX_STEPS}" "${SOURCE_SHA}" "$(config_sha "${arm}")" "${HARNESS_SHA}"
}

emit_manifest() {
  local arm="$1" method checkpoint
  method="$(method_for "${arm}")"
  checkpoint="$(checkpoint_for "${arm}")"
  python3 - "${arm}" "${method}" "$(k_for "${arm}")" "${checkpoint}" <<PY
import json
import sys

print(json.dumps({
    "arm": sys.argv[1],
    "method": sys.argv[2] or None,
    "num_speculative_tokens": int(sys.argv[3]),
    "checkpoint": sys.argv[4] or None,
    "source": {"root": "${SOURCE_ROOT}", "sha": "${SOURCE_SHA}"},
    "harness_sha": "${HARNESS_SHA}",
    "container": "${CONTAINER}",
    "max_steps": int("${MAX_STEPS}"),
    "wandb_project": "sna-specdec",
    "wandb_group": "q235-base-dspark-b8-math20-20260828",
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "slurm": {
        "account": "${ACCOUNT}",
        "partition": "batch",
        "qos": "normal",
        "time": "04:00:00",
        "nodes": 32,
        "gpus_per_node": 4,
        "segment": 32,
    },
}, sort_keys=True))
PY
}

source_guard() {
  test -e "${SOURCE_ROOT}/.git" || die "missing source: ${SOURCE_ROOT}"
  test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}" || die "source SHA drift"
  if git -C "${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "source has unresolved submodules"
  fi
  test -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || die "source is dirty"
  test -z "$(git -C "${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "source submodule is dirty"
  test -r "${CONTAINER}" || die "missing container: ${CONTAINER}"
}

checkpoint_guard() {
  local arm="$1" checkpoint method expected_arch expected_bytes
  [[ "${arm}" == baseline ]] && return
  checkpoint="$(checkpoint_for "${arm}")"
  method="$(method_for "${arm}")"
  case "${method}" in dspark) expected_arch=Qwen3DSparkModel; expected_bytes=2546451906 ;; esac
  python3 - "${checkpoint}" "${expected_arch}" "${expected_bytes}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
architecture = config.get("architectures", [None])[0]
if architecture != sys.argv[2]:
    raise SystemExit(f"architecture mismatch: {architecture!r} != {sys.argv[2]!r}")
weight = root / "model.safetensors"
if weight.stat().st_size != int(sys.argv[3]):
    raise SystemExit(f"weight size mismatch: {weight.stat().st_size} != {sys.argv[3]}")
if config.get("hidden_size") != 4096 or config.get("vocab_size") != 151936:
    raise SystemExit("Q235 drafter shape mismatch")
if config.get("block_size") != 8:
    raise SystemExit(f"DSpark block_size mismatch: {config.get('block_size')!r}")
print(f"CHECKPOINT_GATE_PASS {root}")
PY
}

preflight() {
  local arm="$1"
  case "${MAX_STEPS}" in 1|3|20) ;; *) die "Q235_MAX_STEPS must be 1, 3, or 20" ;; esac
  arm_contract_guard "${arm}"
  source_guard
  checkpoint_guard "${arm}"
  python3 -m unittest experiments.qwen235b_step25391_math_grpo_20260826.tests.test_contract
}

run_id() {
  python3 - "$1" <<'PY'
import sys
import uuid

print(f"q235-math-grpo-step25391-{sys.argv[1]}-{uuid.uuid4().hex}")
PY
}

write_sbatch() {
  local arm="$1" root="$2" run artifact config sbatch_path overlay_source post_sync_exports
  run="$(run_id "${arm}")"
  artifact="${root}/artifacts/${run}"
  config="${SCRIPT_DIR}/configs/${arm}.yaml"
  sbatch_path="${artifact}/job.sbatch"
  mkdir -p "${artifact}"
  cp "${config}" "${artifact}/resolved-input-${arm}.yaml"
  cp "${SCRIPT_DIR}/verify_composed_configs.py" "${artifact}/verify_composed_configs.py"
  emit_manifest "${arm}" >"${artifact}/manifest.json"
  post_sync_exports=""
  if [[ "$(method_for "${arm}")" == dspark ]]; then
    overlay_source="${SCRIPT_DIR}/../qwen3_30ba3b_draft_cadence_200step_20260826"
    mkdir -p "${artifact}/patches"
    cp "${overlay_source}/prepare_vllm_dspark_fap_overlay.py" "${artifact}/prepare_vllm_dspark_fap_overlay.py"
    cp "${overlay_source}/patches/vllm-0.25.1-pr48167-runtime.patch" "${artifact}/patches/vllm-0.25.1-pr48167-runtime.patch"
    cp "${overlay_source}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch" "${artifact}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch"
    post_sync_exports="export NRL_VENV_POST_SYNC_SCRIPT='${artifact}/prepare_vllm_dspark_fap_overlay.py'
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
  fi
  cat >"${artifact}/driver.sh" <<DRIVER
#!/usr/bin/env bash
set -euo pipefail
die() { echo "Q235_STEP25391_DRIVER_FAIL_CLOSED: \$*" >&2; exit 1; }
test "\$(git -C '${SOURCE_ROOT}' rev-parse HEAD)" = '${SOURCE_SHA}' || die 'source SHA drift'
test -z "\$(git -C '${SOURCE_ROOT}' status --porcelain=v1 --untracked-files=all)" || die 'source is dirty'
test -f "\${Q235_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp" || die 'missing node-local MCore overlay'
test -n "\${WANDB_API_KEY:-}" || die 'WANDB_API_KEY is absent'
cd '${SOURCE_ROOT}'
NRL_FORCE_REBUILD_VENVS=true UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv \
  uv run --frozen --no-sync python3 '${artifact}/verify_composed_configs.py' \
  --source-root '${SOURCE_ROOT}' --config '${artifact}/resolved-input-${arm}.yaml' \
  | tee '${artifact}/composed-config.json'
export WANDB_RUN_ID='${run}'
NRL_FORCE_REBUILD_VENVS=true UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv \
  uv run --frozen --no-sync examples/run_grpo.py \
  --config '${artifact}/resolved-input-${arm}.yaml' \
  grpo.max_num_steps='${MAX_STEPS}' \
  logger.log_dir='${artifact}/logs' \
  logger.wandb_enabled=True \
  logger.wandb.project=sna-specdec \
  +logger.wandb.group=q235-base-dspark-b8-math20-20260828 \
  logger.wandb.name='${run}' \
  2>&1 | tee '${artifact}/train.log'
grep -qE 'Capturing CUDA graphs.*100%|Graph capturing finished' '${artifact}/train.log'
if [[ '${arm}' == dspark_* ]]; then
  receipt="\${Q235_VLLM_OVERLAY}/dspark-fap-vllm-48167-runtime.json"
  test -f "\${receipt}" || die 'missing DSpark vLLM overlay receipt'
  cp "\${receipt}" '${artifact}/vllm-dspark-fap-overlay-receipt.json'
  python3 - "\${receipt}" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt.get("schema_version") != 3:
    raise SystemExit("invalid DSpark overlay receipt schema")
if receipt.get("patch_sha256") != "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317":
    raise SystemExit("DSpark primary overlay digest mismatch")
if receipt.get("followup_patch_sha256") != "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90":
    raise SystemExit("DSpark causality overlay digest mismatch")
if receipt.get("status") not in {"applied", "already-patched"}:
    raise SystemExit("DSpark primary overlay status invalid")
if receipt.get("followup_status") not in {"applied", "already-patched"}:
    raise SystemExit("DSpark causality overlay status invalid")
PY
fi
grep -qE 'Step[[:space:]]+1[[:space:]]*/' '${artifact}/train.log'
test "\$(grep -Ec 'Step[[:space:]]+[0-9]+[[:space:]]*/' '${artifact}/train.log')" -ge '${MAX_STEPS}'
echo Q235_MATH_GRPO_GATE_PASS | tee '${artifact}/gates.log'
DRIVER
  chmod 700 "${artifact}/driver.sh"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=q235-math-${arm}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=32
#SBATCH --segment=32
#SBATCH --gpus-per-node=4
#SBATCH --output=${artifact}/slurm-%j.out
#SBATCH --error=${artifact}/slurm-%j.err
set -euo pipefail
export CONTAINER='${CONTAINER}'
export MOUNTS='/lustre:/lustre,/home:/home,/raid:/raid'
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export SOURCE_ROOT='${SOURCE_ROOT}'
export Q235_NODE_ROOT="/raid/scratch/sna/q235-math-\${SLURM_JOB_ID}"
export Q235_MCORE_SOURCE="\${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q235_MCORE_OVERLAY="\${Q235_NODE_ROOT}/mcore-overlay"
export Q235_VLLM_OVERLAY="\${Q235_NODE_ROOT}/vllm-overlay"
export NEMO_RL_VENV_DIR="\${Q235_NODE_ROOT}/venvs"
export PYTHONPATH="\${Q235_VLLM_OVERLAY}:\${Q235_MCORE_OVERLAY}:\${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q235_MCORE_OVERLAY}"; cp -a "\${Q235_MCORE_SOURCE}/megatron" "\${Q235_MCORE_OVERLAY}/"; test -f "\${Q235_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"'
${post_sync_exports}
export ARTIFACT_DIR='${artifact}'
export BASE_LOG_DIR='${artifact}'
export NRL_FORCE_REBUILD_VENVS=true
export WANDB_API_KEY="\${WANDB_API_KEY:?WANDB_API_KEY must be exported at submission}"
export COMMAND='bash ${artifact}/driver.sh'
exec bash '${SOURCE_ROOT}/ray.sub'
SBATCH
  chmod 700 "${sbatch_path}"
  printf '%s\n' "${sbatch_path}"
}

write_receipt() {
  local arm="$1" output="$2"
  local receipt="${DURABLE_ROOT}/preflight/${arm}-steps${MAX_STEPS}.json"
  mkdir -p "$(dirname "${receipt}")"
  python3 - "${receipt}" "${arm}" "$(config_sha "${arm}")" "${output}" <<PY
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "arm": sys.argv[2],
    "config_sha": sys.argv[3],
    "harness_sha": "${HARNESS_SHA}",
    "source_sha": "${SOURCE_SHA}",
    "max_steps": int("${MAX_STEPS}"),
    "test_only_output": sys.argv[4],
}, sort_keys=True) + "\n", encoding="utf-8")
PY
}

require_receipt() {
  local arm="$1"
  local receipt="${DURABLE_ROOT}/preflight/${arm}-steps${MAX_STEPS}.json"
  python3 - "${receipt}" "${arm}" "$(config_sha "${arm}")" <<PY
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "arm": sys.argv[2],
    "config_sha": sys.argv[3],
    "harness_sha": "${HARNESS_SHA}",
    "source_sha": "${SOURCE_SHA}",
    "max_steps": int("${MAX_STEPS}"),
}
if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("test_only_output"):
    raise SystemExit(f"invalid test-only receipt: {sys.argv[1]}")
PY
}

mode="${1:-}"
[[ $# -eq 2 ]] || usage
arm="$2"
valid_arm "${arm}"
case "${mode}" in
  --emit-manifest)
    emit_manifest "${arm}"
    ;;
  --emit-submission-record)
    submission_record "${arm}"
    ;;
  --validate-arm-contract)
    arm_contract_guard "${arm}"
    ;;
  --render-sbatch)
    write_sbatch "${arm}" "${Q235_RENDER_ROOT:?Q235_RENDER_ROOT is required}"
    ;;
  --test-only)
    preflight "${arm}"
    output="$(sbatch --test-only "$(write_sbatch "${arm}" "${DURABLE_ROOT}")" 2>&1)"
    write_receipt "${arm}" "${output}"
    printf '%s\n' "${output}"
    ;;
  --submit)
    preflight "${arm}"
    require_receipt "${arm}"
    record="$(submission_record "${arm}")"
    mkdir -p "$(dirname "${record}")"
    (set -o noclobber; : >"${record}.lock") 2>/dev/null || die "submission already exists or is in progress: ${record}"
    trap 'rm -f "${record}.lock"' EXIT
    output="$(sbatch "$(write_sbatch "${arm}" "${DURABLE_ROOT}")")"
    python3 - "${record}" "${arm}" "${output}" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    json.dumps({"arm": sys.argv[2], "job_output": sys.argv[3]}, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    printf '%s\n' "${output}"
    ;;
  *) usage ;;
esac
