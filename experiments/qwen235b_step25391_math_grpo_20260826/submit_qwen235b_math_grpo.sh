#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=/home/sna/nemorl-pr11-q30-k57-product-clean-20260823
readonly SOURCE_SHA=d0c4f1110cca28c75b7a1d98ed2d5f197e7d01dc
readonly BRIDGE_REL=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
readonly MEGATRON_REL=3rdparty/Megatron-LM
readonly HELPERS_REL=megatron/core/datasets/helpers_cpp
readonly HELPERS_SHA256=39f37692b828622d8e40d13a683b5d0f511c7c852c7497edce286c7eda28833a
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DRAFTER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/thinking-drafters/nemotron-post-v2-b8-s25391
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen235b_step25391_math_grpo_20260826
readonly ACCOUNT="${ACCOUNT:-nemotron_n3_post}"
readonly MAX_STEPS="${Q235_MAX_STEPS:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly HARNESS_SHA

usage() {
  echo "usage: $0 --emit-manifest|--render-sbatch|--test-only|--submit ARM" >&2
  exit 2
}

die() { echo "Q235_STEP25391_FAIL_CLOSED: $*" >&2; exit 1; }

valid_arm() {
  case "$1" in baseline|dflash_k3|dflash_k5|dspark_k3|dspark_k5) ;; *) usage ;; esac
}

method_for() {
  case "$1" in
    baseline) printf '\n' ;;
    dflash_*) printf 'dflash\n' ;;
    dspark_*) printf 'dspark\n' ;;
  esac
}

k_for() {
  case "$1" in
    baseline) printf '0\n' ;;
    *_k3) printf '3\n' ;;
    *_k5) printf '5\n' ;;
  esac
}

checkpoint_for() {
  case "$1" in
    dflash_*) printf '%s\n' "${DRAFTER_ROOT}/q235-thinking-nemotron-v2-dflash-b8-s25391/exported-checkpoint-25391" ;;
    dspark_*) printf '%s\n' "${DRAFTER_ROOT}/q235-thinking-nemotron-v2-dspark-b8-s25391/exported-checkpoint-25391" ;;
    baseline) printf '\n' ;;
  esac
}

config_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$1.yaml"
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
  local bridge megatron root_state bridge_state megatron_state helpers_sha
  test -e "${SOURCE_ROOT}/.git" || die "missing source: ${SOURCE_ROOT}"
  test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}" || die "source SHA drift"
  if git -C "${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "source has unresolved submodules"
  fi
  bridge="${SOURCE_ROOT}/${BRIDGE_REL}"
  megatron="${bridge}/${MEGATRON_REL}"
  root_state="$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)"
  bridge_state="$(git -C "${bridge}" status --porcelain=v1 --untracked-files=all)"
  megatron_state="$(git -C "${megatron}" status --porcelain=v1 --untracked-files=all)"
  [[ "${root_state}" == " M ${BRIDGE_REL}" ]] || die "unexpected source worktree state: ${root_state}"
  [[ "${bridge_state}" == " M ${MEGATRON_REL}" ]] || die "unexpected Megatron-Bridge worktree state: ${bridge_state}"
  [[ "${megatron_state}" == "?? ${HELPERS_REL}" ]] || die "unexpected Megatron-LM worktree state: ${megatron_state}"
  helpers_sha="$(sha256sum "${megatron}/${HELPERS_REL}" | awk '{print $1}')"
  [[ "${helpers_sha}" == "${HELPERS_SHA256}" ]] || die "helpers_cpp SHA mismatch: ${helpers_sha}"
  test -r "${CONTAINER}" || die "missing container: ${CONTAINER}"
}

checkpoint_guard() {
  local arm="$1" checkpoint method expected_arch expected_bytes
  [[ "${arm}" == baseline ]] && return
  checkpoint="$(checkpoint_for "${arm}")"
  method="$(method_for "${arm}")"
  case "${method}" in
    dflash) expected_arch=DFlashDraftModel; expected_bytes=2390860352 ;;
    dspark) expected_arch=Qwen3DSparkModel; expected_bytes=2546451906 ;;
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
print(f"CHECKPOINT_GATE_PASS {root}")
PY
}

preflight() {
  local arm="$1"
  case "${MAX_STEPS}" in 1|3|20) ;; *) die "Q235_MAX_STEPS must be 1, 3, or 20" ;; esac
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
  local arm="$1" root="$2" run artifact config sbatch_path
  run="$(run_id "${arm}")"
  artifact="${root}/artifacts/${run}"
  config="${SCRIPT_DIR}/configs/${arm}.yaml"
  sbatch_path="${artifact}/job.sbatch"
  mkdir -p "${artifact}"
  cp "${config}" "${artifact}/resolved-input-${arm}.yaml"
  emit_manifest "${arm}" >"${artifact}/manifest.json"
  cat >"${artifact}/driver.sh" <<DRIVER
#!/usr/bin/env bash
set -euo pipefail
die() { echo "Q235_STEP25391_DRIVER_FAIL_CLOSED: \$*" >&2; exit 1; }
bridge='${SOURCE_ROOT}/${BRIDGE_REL}'
megatron="\${bridge}/${MEGATRON_REL}"
test "\$(git -C '${SOURCE_ROOT}' rev-parse HEAD)" = '${SOURCE_SHA}' || die 'source SHA drift'
root_state="\$(git -C '${SOURCE_ROOT}' status --porcelain=v1 --untracked-files=all)"
bridge_state="\$(git -C "\${bridge}" status --porcelain=v1 --untracked-files=all)"
megatron_state="\$(git -C "\${megatron}" status --porcelain=v1 --untracked-files=all)"
[[ "\${root_state}" == ' M ${BRIDGE_REL}' ]] || die "unexpected source worktree state: \${root_state}"
[[ "\${bridge_state}" == ' M ${MEGATRON_REL}' ]] || die "unexpected Megatron-Bridge worktree state: \${bridge_state}"
[[ "\${megatron_state}" == '?? ${HELPERS_REL}' ]] || die "unexpected Megatron-LM worktree state: \${megatron_state}"
helpers_sha="\$(sha256sum "\${megatron}/${HELPERS_REL}" | awk '{print \$1}')"
[[ "\${helpers_sha}" == '${HELPERS_SHA256}' ]] || die "helpers_cpp SHA mismatch: \${helpers_sha}"
export WANDB_RUN_ID='${run}'
cd '${SOURCE_ROOT}'
NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py \
  --config '${artifact}/resolved-input-${arm}.yaml' \
  grpo.max_num_steps='${MAX_STEPS}' \
  logger.log_dir='${artifact}/logs' \
  logger.wandb_enabled=True \
  logger.wandb.project=sna-specdec \
  logger.wandb.name='${run}' \
  2>&1 | tee '${artifact}/train.log'
grep -qE 'Capturing CUDA graphs.*100%|Graph capturing finished' '${artifact}/train.log'
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
export MOUNTS='/lustre:/lustre,/home:/home'
export GPUS_PER_NODE=4
export ARTIFACT_DIR='${artifact}'
export BASE_LOG_DIR='${artifact}'
export NRL_FORCE_REBUILD_VENVS=true
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
    record="${DURABLE_ROOT}/submissions/${arm}-steps${MAX_STEPS}-${SOURCE_SHA}.json"
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
