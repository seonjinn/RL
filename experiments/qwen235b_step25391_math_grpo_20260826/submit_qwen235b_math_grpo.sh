#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-q235-math-product-20260828
readonly SOURCE_SHA=f6f8605da02675af4361cfc9fd4d5f4d23279ff1
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DSPARK_DRAFTER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/checkpoints/qwen3-235ba22b-base-nemotron-b8-s25391/dspark
readonly EAGLE3_DRAFTER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen235b_step25391_math_grpo_20260826
readonly ACCOUNT="${ACCOUNT:-nemotron_n3_post}"
readonly MAX_STEPS="${Q235_MAX_STEPS:-20}"
readonly PARTITION=batch
readonly QOS=normal
readonly TIME_LIMIT=04:00:00
readonly NODES=32
readonly GPUS_PER_NODE=4
readonly SEGMENT=16
readonly CPUS_PER_WORKER=64
readonly MEMORY=0
readonly SBATCH_TIMEOUT_SECONDS=30
readonly SBATCH_MAX_OUTPUT_BYTES=65536
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly DSPARK_OVERLAY_SOURCE="${SCRIPT_DIR}/../qwen3_30ba3b_draft_cadence_200step_20260826"
HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly HARNESS_SHA

usage() {
  echo "usage: $0 --emit-manifest|--emit-submission-record|--validate-arm-contract|--render-sbatch|--test-only|--submit ARM" >&2
  exit 2
}

die() { echo "Q235_STEP25391_FAIL_CLOSED: $*" >&2; exit 1; }

valid_arm() {
  case "$1" in
    baseline|dspark_k3|dspark_k5|dspark_k7|eagle3_k3|baseline_cg2048|dspark_k3_cg2048|dspark_k5_cg2048|dspark_k7_cg2048|eagle3_k3_cg2048) ;;
    *) usage ;;
  esac
}

base_arm_for() {
  printf '%s\n' "${1%_cg2048}"
}

graph_profile_for() {
  case "$1" in
    *_cg2048) printf 'expanded_2048\n' ;;
    *) printf 'default_small\n' ;;
  esac
}

capture_sizes_source_for() {
  case "$1" in
    *_cg2048) printf 'arm-config-expanded-through-2048\n' ;;
    baseline) printf 'official-performance-recipe\n' ;;
    dspark_*|eagle3_*) printf 'arm-config-k-aware-small-buckets\n' ;;
  esac
}

method_for() {
  case "$(base_arm_for "$1")" in
    baseline) printf '\n' ;;
    dspark_*) printf 'dspark\n' ;;
    eagle3_*) printf 'eagle3\n' ;;
  esac
}

k_for() {
  case "$(base_arm_for "$1")" in
    baseline) printf '0\n' ;;
    *_k3) printf '3\n' ;;
    *_k5) printf '5\n' ;;
    *_k7) printf '7\n' ;;
  esac
}

checkpoint_for() {
  case "$(base_arm_for "$1")" in
    dspark_*) printf '%s\n' "${DSPARK_DRAFTER_ROOT}" ;;
    eagle3_*) printf '%s\n' "${EAGLE3_DRAFTER_ROOT}" ;;
    baseline) printf '\n' ;;
  esac
}

arm_contract_guard() {
  local arm="$1" checkpoint method expected
  [[ "$(base_arm_for "${arm}")" == baseline ]] && return
  checkpoint="$(checkpoint_for "${arm}")"
  method="$(method_for "${arm}")"
  case "${method}" in
    dspark) expected="${DSPARK_DRAFTER_ROOT}" ;;
    eagle3) expected="${EAGLE3_DRAFTER_ROOT}" ;;
    *) die "unsupported speculative method for ${arm}: ${method}" ;;
  esac
  [[ "${checkpoint}" == "${expected}" ]] || die "unexpected ${method} checkpoint: ${checkpoint}"
}

config_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$1.yaml"
}

file_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

submission_identity() {
  local arm="$1" primary_patch_sha="" followup_patch_sha=""
  if [[ "$(method_for "${arm}")" == dspark ]]; then
    primary_patch_sha="$(file_sha "${DSPARK_OVERLAY_SOURCE}/patches/vllm-0.25.1-pr48167-runtime.patch")"
    followup_patch_sha="$(file_sha "${DSPARK_OVERLAY_SOURCE}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch")"
  fi
  python3 - \
    "${arm}" "${ACCOUNT}" "${MAX_STEPS}" "$(config_sha "${arm}")" \
    "$(file_sha "${BASH_SOURCE[0]}")" "$(file_sha "${SCRIPT_DIR}/verify_composed_configs.py")" \
    "$(file_sha "${SCRIPT_DIR}/prepare_vllm_frozen_drafter_overlay.py")" \
    "$(file_sha "${SCRIPT_DIR}/frozen_drafter_sleep_policy.py")" \
    "$(file_sha "${SCRIPT_DIR}/patches/vllm-0.25.1-refit-aware-frozen-drafter-sleep.patch")" \
    "${primary_patch_sha}" "${followup_patch_sha}" \
    "${SOURCE_SHA}" "${CONTAINER}" "$(checkpoint_for "${arm}")" "${HARNESS_SHA}" \
    "${PARTITION}" "${QOS}" "${TIME_LIMIT}" "${NODES}" \
    "${GPUS_PER_NODE}" "${SEGMENT}" "${CPUS_PER_WORKER}" "${MEMORY}" <<'PY'
import json
import sys

content = {
    "config_sha256": sys.argv[4],
    "launcher_sha256": sys.argv[5],
    "verifier_sha256": sys.argv[6],
    "sleep_overlay_helper_sha256": sys.argv[7],
    "sleep_policy_sha256": sys.argv[8],
    "sleep_runtime_patch_sha256": sys.argv[9],
}
if sys.argv[10]:
    content["dspark_primary_patch_sha256"] = sys.argv[10]
    content["dspark_followup_patch_sha256"] = sys.argv[11]
print(json.dumps({
    "arm": sys.argv[1],
    "container": sys.argv[13],
    "content": content,
    "drafter_root": sys.argv[14],
    "harness_commit": sys.argv[15],
    "max_steps": int(sys.argv[3]),
    "slurm": {
        "account": sys.argv[2],
        "cpus_per_worker": int(sys.argv[22]),
        "exclusive": True,
        "gpus_per_node": int(sys.argv[20]),
        "memory": sys.argv[23],
        "nodes": int(sys.argv[19]),
        "partition": sys.argv[16],
        "qos": sys.argv[17],
        "segment": int(sys.argv[21]),
        "time": sys.argv[18],
    },
    "source_sha": sys.argv[12],
}, sort_keys=True))
PY
}

submission_record() {
  local arm="$1" identity="${2:-}" config_digest
  if [[ -n "${identity}" ]]; then
    config_digest="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["content"]["config_sha256"])' "${identity}")"
  else
    config_digest="$(config_sha "${arm}")"
  fi
  printf '%s/submissions/%s-steps%s-%s-%s-%s.json\n' \
    "${DURABLE_ROOT}" "${arm}" "${MAX_STEPS}" "${SOURCE_SHA}" "${config_digest}" "${HARNESS_SHA}"
}

logical_claim() {
  printf '%s/submissions/claims/%s-steps%s.json\n' \
    "${DURABLE_ROOT}" "$1" "${MAX_STEPS}"
}

emit_manifest() {
  local arm="$1" method checkpoint base_arm graph_profile capture_sizes_source
  method="$(method_for "${arm}")"
  checkpoint="$(checkpoint_for "${arm}")"
  base_arm="$(base_arm_for "${arm}")"
  graph_profile="$(graph_profile_for "${arm}")"
  capture_sizes_source="$(capture_sizes_source_for "${arm}")"
  python3 - "${arm}" "${base_arm}" "${graph_profile}" "${capture_sizes_source}" "${method}" "$(k_for "${arm}")" "${checkpoint}" <<PY
import json
import sys

print(json.dumps({
    "arm": sys.argv[1],
    "base_arm": sys.argv[2],
    "graph_profile": sys.argv[3],
    "cudagraph_capture_sizes_source": sys.argv[4],
    "method": sys.argv[5] or None,
    "num_speculative_tokens": int(sys.argv[6]),
    "checkpoint": sys.argv[7] or None,
    "source": {"root": "${SOURCE_ROOT}", "sha": "${SOURCE_SHA}"},
    "harness_sha": "${HARNESS_SHA}",
    "container": "${CONTAINER}",
    "max_steps": int("${MAX_STEPS}"),
    "wandb_project": "sna-specdec",
    "wandb_group": "q235-base-dspark-b8-math20-20260828",
    "cudagraph_mode_source": "official-performance-recipe",
    "slurm": {
        "account": "${ACCOUNT}",
        "partition": "${PARTITION}",
        "qos": "${QOS}",
        "time": "${TIME_LIMIT}",
        "nodes": int("${NODES}"),
        "gpus_per_node": int("${GPUS_PER_NODE}"),
        "segment": int("${SEGMENT}"),
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
  [[ "$(base_arm_for "${arm}")" == baseline ]] && return
  checkpoint="$(checkpoint_for "${arm}")"
  method="$(method_for "${arm}")"
  case "${method}" in
    dspark) expected_arch=Qwen3DSparkModel; expected_bytes=2546451906 ;;
    eagle3) expected_arch=LlamaForCausalLMEagle3; expected_bytes=620791032 ;;
    *) die "unsupported speculative method for ${arm}: ${method}" ;;
  esac
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
if sys.argv[2] == "Qwen3DSparkModel" and config.get("block_size") != 8:
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
  local arm="$1" root="$2" run artifact config sbatch_path dspark_overlay_enabled
  run="$(run_id "${arm}")"
  artifact="${root}/artifacts/${run}"
  config="${SCRIPT_DIR}/configs/${arm}.yaml"
  sbatch_path="${artifact}/job.sbatch"
  mkdir -p "${artifact}"
  cp "${config}" "${artifact}/resolved-input-${arm}.yaml"
  cp "${SCRIPT_DIR}/verify_composed_configs.py" "${artifact}/verify_composed_configs.py"
  cp "${SCRIPT_DIR}/prepare_vllm_frozen_drafter_overlay.py" "${artifact}/prepare_vllm_frozen_drafter_overlay.py"
  cp "${SCRIPT_DIR}/frozen_drafter_sleep_policy.py" "${artifact}/frozen_drafter_sleep_policy.py"
  emit_manifest "${arm}" >"${artifact}/manifest.json"
  mkdir -p "${artifact}/patches"
  cp "${SCRIPT_DIR}/patches/vllm-0.25.1-refit-aware-frozen-drafter-sleep.patch" "${artifact}/patches/"
  dspark_overlay_enabled=0
  if [[ "$(method_for "${arm}")" == dspark ]]; then
    cp "${DSPARK_OVERLAY_SOURCE}/patches/vllm-0.25.1-pr48167-runtime.patch" "${artifact}/patches/vllm-0.25.1-pr48167-runtime.patch"
    cp "${DSPARK_OVERLAY_SOURCE}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch" "${artifact}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch"
    dspark_overlay_enabled=1
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
receipt="\${Q235_VLLM_OVERLAY}/frozen-drafter-sleep-overlay.json"
test -f "\${receipt}" || die 'missing refit-aware vLLM sleep overlay receipt'
cp "\${receipt}" '${artifact}/vllm-frozen-drafter-sleep-overlay-receipt.json'
python3 - "\${receipt}" '${dspark_overlay_enabled}' <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt.get("schema_version") != 1:
    raise SystemExit("invalid refit-aware sleep overlay receipt schema")
if receipt.get("runtime_patch_sha256") != "b61df83aa855edae9e36aef560b03dbd148aa703b326fe42a90c1fdd451564ef":
    raise SystemExit("refit-aware sleep runtime patch digest mismatch")
if receipt.get("policy_module_sha256") != "4cdfb9adbb9dd2ec346460c437fce1a108c20ca8dfdcfa5dec391de136448e59":
    raise SystemExit("refit-aware sleep policy digest mismatch")
if receipt.get("status") not in {"applied", "already-patched"}:
    raise SystemExit("refit-aware sleep overlay status invalid")
prerequisites = receipt.get("prerequisite_patches", [])
expected = []
if sys.argv[2] == "1":
    expected = [
        "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317",
        "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90",
    ]
if [item.get("sha256") for item in prerequisites] != expected:
    raise SystemExit("vLLM prerequisite patch receipt mismatch")
PY
grep -qE 'Step[[:space:]]+1[[:space:]]*/' '${artifact}/train.log'
test "\$(grep -Ec 'Step[[:space:]]+[0-9]+[[:space:]]*/' '${artifact}/train.log')" -ge '${MAX_STEPS}'
echo Q235_MATH_GRPO_GATE_PASS | tee '${artifact}/gates.log'
DRIVER
  chmod 700 "${artifact}/driver.sh"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=q235-math-${arm}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --segment=${SEGMENT}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --exclusive
#SBATCH --mem=${MEMORY}
#SBATCH --output=${artifact}/slurm-%j.out
#SBATCH --error=${artifact}/slurm-%j.err
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:\${PATH}"
export CONTAINER='${CONTAINER}'
export MOUNTS='/lustre:/lustre,/home:/home,/raid:/raid'
export GPUS_PER_NODE=${GPUS_PER_NODE}
export CPUS_PER_WORKER=${CPUS_PER_WORKER}
export NCCL_NVLS_ENABLE=0
export SOURCE_ROOT='${SOURCE_ROOT}'
export Q235_NODE_ROOT="/raid/scratch/sna/q235-math-\${SLURM_JOB_ID}"
export Q235_MCORE_SOURCE="\${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export Q235_MCORE_OVERLAY="\${Q235_NODE_ROOT}/mcore-overlay"
export Q235_VLLM_OVERLAY="\${Q235_NODE_ROOT}/vllm-overlay"
export NEMO_RL_VENV_DIR="\${Q235_NODE_ROOT}/venvs"
export PYTHONPATH="\${Q235_VLLM_OVERLAY}:\${Q235_MCORE_OVERLAY}:\${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH,NRL_FROZEN_DRAFTER_DISCARD_REFIT_TARGET
export SETUP_COMMAND='set -euo pipefail; mkdir -p "\${Q235_MCORE_OVERLAY}"; cp -a "\${Q235_MCORE_SOURCE}/megatron" "\${Q235_MCORE_OVERLAY}/"; test -f "\${Q235_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"'
export Q235_DSPARK_FAP_OVERLAY=${dspark_overlay_enabled}
export NRL_FROZEN_DRAFTER_DISCARD_REFIT_TARGET=1
export NRL_VENV_POST_SYNC_SCRIPT='${artifact}/prepare_vllm_frozen_drafter_overlay.py'
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
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
  local arm="$1" output="$2" identity="$3"
  local receipt="${DURABLE_ROOT}/preflight/${arm}-steps${MAX_STEPS}.json"
  mkdir -p "$(dirname "${receipt}")"
  python3 - "${receipt}" "${identity}" "${output}" <<'PY'
import json
import os
import pathlib
import sys
import uuid

path = pathlib.Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
payload = json.dumps({
    "identity": json.loads(sys.argv[2]),
    "schema_version": 2,
    "test_only_output": sys.argv[3],
}, sort_keys=True) + "\n"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, path)
os.chmod(path, 0o600)
PY
}

require_receipt() {
  local arm="$1" identity="$2"
  local receipt="${DURABLE_ROOT}/preflight/${arm}-steps${MAX_STEPS}.json"
  python3 - "${receipt}" "${identity}" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = json.loads(sys.argv[2])
if (
    receipt.get("schema_version") != 2
    or receipt.get("identity") != expected
    or not receipt.get("test_only_output")
):
    raise SystemExit(f"invalid test-only receipt: {sys.argv[1]}")
PY
}

create_submitting_record() {
  local record="$1" claim="$2" arm="$3" run="$4" artifact="$5" sbatch_path="$6" identity="$7"
  python3 - "${record}" "${claim}" "${arm}" "${run}" "${artifact}" "${sbatch_path}" "${identity}" "${MAX_STEPS}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
claim = pathlib.Path(sys.argv[2])
arm = sys.argv[3]
max_steps = sys.argv[8]
legacy_lock = pathlib.Path(f"{path}.lock")
payload = {
    "arm": arm,
    "artifact_dir": sys.argv[5],
    "identity": json.loads(sys.argv[7]),
    "run_id": sys.argv[4],
    "sbatch_path": sys.argv[6],
    "schema_version": 2,
    "state": "submitting",
}
claim.parent.mkdir(parents=True, exist_ok=True)
try:
    legacy_descriptor = os.open(
        legacy_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
except FileExistsError:
    raise SystemExit(f"legacy submission lock exists; reconcile {legacy_lock}")
with os.fdopen(legacy_descriptor, "w", encoding="utf-8") as stream:
    json.dump({"record": str(path), "state": "compatibility-lock"}, stream, sort_keys=True)
    stream.write("\n")

try:
    claim_descriptor = os.open(
        claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
except FileExistsError:
    raise SystemExit(f"logical-arm submission claim exists; reconcile {claim}")
with os.fdopen(claim_descriptor, "w", encoding="utf-8") as stream:
    json.dump({"arm": arm, "max_steps": int(max_steps), "record": str(path), "state": "claimed"}, stream, sort_keys=True)
    stream.write("\n")

prior_locks = sorted(
    candidate
    for candidate in path.parent.glob(f"{arm}-steps{max_steps}-*.json.lock")
    if candidate != legacy_lock
)
if prior_locks:
    raise SystemExit(
        "prior logical submission lock exists; reconcile "
        + ", ".join(str(candidate) for candidate in prior_locks)
    )
prior_records = sorted(
    candidate
    for candidate in path.parent.glob(f"{arm}-steps{max_steps}-*.json")
    if candidate != path
)
if prior_records:
    raise SystemExit(
        "prior logical submission exists; reconcile "
        + ", ".join(str(candidate) for candidate in prior_records)
    )
try:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    raise SystemExit(f"submission record exists; reconcile {path}")
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")
PY
}

verify_rendered_artifacts() {
  local arm="$1" sbatch_path="$2" identity="$3"
  python3 - "${arm}" "${sbatch_path}" "${BASH_SOURCE[0]}" "${identity}" <<'PY'
import hashlib
import json
import pathlib
import sys

arm = sys.argv[1]
artifact = pathlib.Path(sys.argv[2]).parent
identity = json.loads(sys.argv[4])
content = identity["content"]
paths = {
    "config_sha256": artifact / f"resolved-input-{arm}.yaml",
    "launcher_sha256": pathlib.Path(sys.argv[3]),
    "verifier_sha256": artifact / "verify_composed_configs.py",
    "sleep_overlay_helper_sha256": artifact
    / "prepare_vllm_frozen_drafter_overlay.py",
    "sleep_policy_sha256": artifact / "frozen_drafter_sleep_policy.py",
    "sleep_runtime_patch_sha256": artifact
    / "patches"
    / "vllm-0.25.1-refit-aware-frozen-drafter-sleep.patch",
}
if arm.startswith("dspark_"):
    paths.update({
        "dspark_primary_patch_sha256": artifact
        / "patches"
        / "vllm-0.25.1-pr48167-runtime.patch",
        "dspark_followup_patch_sha256": artifact
        / "patches"
        / "vllm-0.25.1-pr48167-group-causality-followup.patch",
    })
for key, path in paths.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != content.get(key):
        raise SystemExit(f"rendered artifact content drift for {key}: {path}")
PY
}

submit_and_finalize_record() {
  local record="$1" sbatch_path="$2"
  python3 - "${record}" "${sbatch_path}" "${SBATCH_TIMEOUT_SECONDS}" "${SBATCH_MAX_OUTPUT_BYTES}" <<'PY'
import json
import os
import pathlib
import re
import select
import signal
import subprocess
import sys
import time
import uuid

path = pathlib.Path(sys.argv[1])
sbatch_path = sys.argv[2]
timeout_seconds = float(sys.argv[3])
max_output_bytes = int(sys.argv[4])
process = subprocess.Popen(
    ["sbatch", sbatch_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
assert process.stdout is not None
descriptor = process.stdout.fileno()
os.set_blocking(descriptor, False)
deadline = time.monotonic() + timeout_seconds
captured = bytearray()
total_bytes = 0
timed_out = False
eof = False
while not eof or process.poll() is None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 and process.poll() is None and not timed_out:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
    ready, _, _ = select.select([descriptor], [], [], max(0.0, min(0.1, remaining)) if not timed_out else 0.1)
    if ready:
        try:
            chunk = os.read(descriptor, 65536)
        except BlockingIOError:
            chunk = b""
        if chunk:
            total_bytes += len(chunk)
            available = max_output_bytes - len(captured)
            if available > 0:
                captured.extend(chunk[:available])
        else:
            eof = True
    if timed_out and process.poll() is None and time.monotonic() > deadline + 1.0:
        os.killpg(process.pid, signal.SIGKILL)
process.wait()
scheduler_exit_status = process.returncode
output = bytes(captured).decode("utf-8", errors="replace")
output_truncated = total_bytes > len(captured)
job_ids = re.findall(r"(?m)^Submitted batch job ([0-9]+)\r?$", output)
receipt = json.loads(path.read_text(encoding="utf-8"))
receipt.update({
    "scheduler_exit_status": scheduler_exit_status,
    "scheduler_output": output,
    "scheduler_output_bytes": total_bytes,
    "scheduler_output_truncated": output_truncated,
    "scheduler_timed_out": timed_out,
})
accepted = (
    scheduler_exit_status == 0
    and len(job_ids) == 1
    and not output_truncated
    and not timed_out
)
if accepted:
    receipt.update({"job_id": job_ids[0], "state": "accepted"})
else:
    receipt.update({"candidate_job_ids": job_ids, "state": "ambiguous"})
temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(receipt, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, path)
os.chmod(path, 0o600)
if accepted:
    print(f"Submitted batch job {job_ids[0]}")
else:
    print(
        "scheduler submission outcome is ambiguous "
        f"(exit_status={scheduler_exit_status}, timed_out={timed_out}, "
        f"output_truncated={output_truncated}); "
        f"reconcile {path} before retrying",
        file=sys.stderr,
    )
    raise SystemExit(1)
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
    identity="$(submission_identity "${arm}")"
    submission_record "${arm}" "${identity}"
    ;;
  --validate-arm-contract)
    arm_contract_guard "${arm}"
    ;;
  --render-sbatch)
    write_sbatch "${arm}" "${Q235_RENDER_ROOT:?Q235_RENDER_ROOT is required}"
    ;;
  --test-only)
    preflight "${arm}"
    identity="$(submission_identity "${arm}")"
    sbatch_path="$(write_sbatch "${arm}" "${DURABLE_ROOT}")"
    verify_rendered_artifacts "${arm}" "${sbatch_path}" "${identity}"
    output="$(sbatch --test-only "${sbatch_path}" 2>&1)"
    write_receipt "${arm}" "${output}" "${identity}"
    printf '%s\n' "${output}"
    ;;
  --submit)
    preflight "${arm}"
    identity="$(submission_identity "${arm}")"
    require_receipt "${arm}" "${identity}"
    record="$(submission_record "${arm}" "${identity}")"
    claim="$(logical_claim "${arm}")"
    mkdir -p "$(dirname "${record}")"
    sbatch_path="$(write_sbatch "${arm}" "${DURABLE_ROOT}")"
    artifact="$(dirname "${sbatch_path}")"
    run="$(basename "${artifact}")"
    create_submitting_record "${record}" "${claim}" "${arm}" "${run}" "${artifact}" "${sbatch_path}" "${identity}" || die "submission already exists or is in progress: ${record}"
    verify_rendered_artifacts "${arm}" "${sbatch_path}" "${identity}" || die "rendered artifact content drift; reconcile ${record}"
    submit_and_finalize_record "${record}" "${sbatch_path}" || die "scheduler submission was not confirmed; reconcile ${record}"
    ;;
  *) usage ;;
esac
