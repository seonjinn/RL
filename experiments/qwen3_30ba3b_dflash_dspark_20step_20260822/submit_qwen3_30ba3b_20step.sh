#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT="${Q30_20STEP_SOURCE_ROOT:-/home/sna/nemorl-pr11-q30-eagle3-k3-product-clean-20260823}"
readonly SOURCE_SHA=d0c4f1110cca28c75b7a1d98ed2d5f197e7d01dc
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT="${Q30_20STEP_DURABLE_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_lyris14500_k5_k7_20260823}"
readonly ACCOUNT="${Q30_20STEP_ACCOUNT:-nemotron_n3_post}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly HARNESS_SHA
readonly CAPTURE_SIZES_K3='[1,2,4,8,12,16,24,32]'
readonly CAPTURE_SIZES_K5='[1,2,4,8,12,16,24,32,40,48]'
readonly CAPTURE_SIZES_K7='[1,2,4,8,12,16,24,32,40,48,56,64]'
readonly CAPTURE_SIZES_EXPANDED='[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048]'
readonly CAPTURE_SIZES_DFLASH_K5_EXPANDED='[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2046,2048]'

usage() {
  echo "usage: $0 --assert-capture-coverage [VARIANT]|--emit-manifest VARIANT|--render-sbatch VARIANT|--test-only VARIANT|--submit VARIANT" >&2
  exit 2
}

die() { echo "Q30_20STEP_FAIL_CLOSED: $*" >&2; exit 1; }

valid_variant() {
  case "$1" in
    baseline|eagle3-k3|dflash|dspark|dflash-k3|dflash-k5|dflash-k7|dspark-k3|dspark-k5|dspark-k7|\
    baseline-cg2048|eagle3-k3-cg2048|dflash-k3-cg2048|dflash-k5-cg2048|dflash-k7-cg2048|dspark-k3-cg2048|dspark-k5-cg2048|dspark-k7-cg2048) ;;
    *) usage ;;
  esac
}

comparison_arm_for() {
  case "$1" in
    *-cg2048) printf '%s\n' "${1%-cg2048}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

is_expanded_variant() {
  [[ "$1" == *-cg2048 ]]
}

checkpoint_for() {
  case "$(comparison_arm_for "$1")" in
    dflash) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dflash-b8-16n/exported-checkpoint-25391 ;;
    dspark) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/sd1/sd1-direct-q30-base-opb-dspark-b8-16n/exported-checkpoint-25391 ;;
    eagle3-k3) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf ;;
    dflash-k3) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391 ;;
    dflash-k5|dflash-k7) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dflash-b8-16n-migrated-oci-s4400/exported-checkpoint-14500 ;;
    dspark-k3) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391 ;;
    dspark-k5|dspark-k7) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dspark-b8-16n-migrated-oci-s5700/exported-checkpoint-14500 ;;
  esac
}

method_for() {
  case "$(comparison_arm_for "$1")" in
    eagle3-k3) printf '%s\n' eagle3 ;;
    dflash|dflash-k3|dflash-k5|dflash-k7) printf '%s\n' dflash ;;
    dspark|dspark-k3|dspark-k5|dspark-k7) printf '%s\n' dspark ;;
  esac
}

identity_file_for() {
  case "$(comparison_arm_for "$1")" in
    dflash-k3|dspark-k3) printf '%s\n' "${SCRIPT_DIR}/checkpoint_identity_base_s25391.json" ;;
    *) printf '%s\n' "${SCRIPT_DIR}/checkpoint_identity.json" ;;
  esac
}

k_for() {
  case "$(comparison_arm_for "$1")" in
    baseline) printf '%s\n' 0 ;;
    eagle3-k3|dflash-k3|dspark-k3) printf '%s\n' 3 ;;
    dflash|dspark|dflash-k5|dspark-k5) printf '%s\n' 5 ;;
    dflash-k7|dspark-k7) printf '%s\n' 7 ;;
  esac
}

capture_sizes_for() {
  if is_expanded_variant "$1"; then
    if [[ "$(comparison_arm_for "$1")" == dflash-k5 ]]; then
      printf '%s\n' "${CAPTURE_SIZES_DFLASH_K5_EXPANDED}"
    else
      printf '%s\n' "${CAPTURE_SIZES_EXPANDED}"
    fi
    return
  fi
  case "$(k_for "$1")" in
    3) printf '%s\n' "${CAPTURE_SIZES_K3}" ;;
    7) printf '%s\n' "${CAPTURE_SIZES_K7}" ;;
    *) printf '%s\n' "${CAPTURE_SIZES_K5}" ;;
  esac
}

training_mode_for() {
  case "$(comparison_arm_for "$1")" in
    baseline) printf '%s\n' none ;;
    eagle3-k3) printf '%s\n' static ;;
    *) printf '%s\n' always-online ;;
  esac
}

gates_for() {
  case "$(comparison_arm_for "$1")" in
    baseline) printf '%s\n' '["source-clean","cudagraph","step1","step2"]' ;;
    eagle3-k3) printf '%s\n' '["source-clean","checkpoint-contract","cudagraph","step1","step2"]' ;;
    *) printf '%s\n' '["source-clean","state-dict","cudagraph","step1","step2"]' ;;
  esac
}

checkpoint_gate_for() {
  local variant method="$2"
  variant="$(comparison_arm_for "$1")"
  case "${variant}" in
    baseline) printf '%s' '' ;;
    eagle3-k3)
      printf '%s' 'python3 "${ARTIFACT_DIR}/check_eagle3_checkpoint.py" --checkpoint "${CHECKPOINT}" --target-model Qwen/Qwen3-30B-A3B --num-speculative-tokens 3 | tee -a "${ARTIFACT_DIR}/gates.log"'
      ;;
    *)
      printf 'python3 "${ARTIFACT_DIR}/check_checkpoint_state_dict.py" --variant "%s" --checkpoint "${CHECKPOINT}" --identity-file "${CHECKPOINT_IDENTITY}" --verify-content-sha | tee -a "${ARTIFACT_DIR}/gates.log"' "${method}"
      ;;
  esac
}

config_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$(comparison_arm_for "$1").yaml"
}

run_id() {
  local variant="$1" comparison_arm suffix
  comparison_arm="$(comparison_arm_for "${variant}")"
  suffix=""
  if is_expanded_variant "${variant}"; then suffix="-cg2048"; fi
  python3 - "${comparison_arm}" "${suffix}" "${variant}" <<'PY'
import sys
import uuid

labels = {
    "baseline": "baseline-k0",
    "eagle3-k3": "eagle3-k3-base-verifier",
    "dflash": "dflash-k5",
    "dspark": "dspark-k5-b8",
    "dflash-k3": "dflash-k3-base-s25391",
    "dflash-k5": "dflash-k5-lyris14500",
    "dflash-k7": "dflash-k7-lyris14500",
    "dspark-k5": "dspark-k5-lyris14500",
    "dspark-k3": "dspark-k3-base-s25391",
    "dspark-k7": "dspark-k7-lyris14500",
}
label = sys.argv[3] if sys.argv[2] else labels[sys.argv[1]]
print(f"q30ba3b-20step-{label}-{uuid.uuid4().hex}")
PY
}

emit_manifest() {
  local variant="$1" run="$2" comparison_arm checkpoint method k training_mode gates capture_sizes graph_profile
  comparison_arm="$(comparison_arm_for "${variant}")"
  checkpoint=""
  method=""
  if [[ "${comparison_arm}" != baseline ]]; then
    checkpoint="$(checkpoint_for "${variant}")"
    method="$(method_for "${variant}")"
  fi
  k="$(k_for "${variant}")"
  training_mode="$(training_mode_for "${variant}")"
  gates="$(gates_for "${variant}")"
  capture_sizes="$(capture_sizes_for "${variant}")"
  graph_profile="reference"
  if is_expanded_variant "${variant}"; then graph_profile="expanded-2048"; fi
  python3 - "${variant}" "${run}" "${HARNESS_SHA}" "${checkpoint}" "${method}" "${k}" "${training_mode}" "${gates}" "${comparison_arm}" "${graph_profile}" "${capture_sizes}" <<PY
import json
import sys

print(json.dumps({
    "variant": sys.argv[1],
    "comparison_arm": sys.argv[9],
    "source": {"root": "${SOURCE_ROOT}", "sha": "${SOURCE_SHA}"},
    "harness_sha": sys.argv[3],
    "container": "${CONTAINER}",
    "durable_root": "${DURABLE_ROOT}",
    "checkpoint": sys.argv[4] or None,
    "method": sys.argv[5] or None,
    "num_speculative_tokens": int(sys.argv[6]),
    "draft_training_mode": sys.argv[7],
    "target_model": "Qwen/Qwen3-30B-A3B",
    "slurm": {"account": "${ACCOUNT}", "partition": "batch", "qos": "normal", "time": "04:00:00", "nodes": 4, "gpus_per_node": 4},
    "gates": json.loads(sys.argv[8]),
    "cudagraph_profile": sys.argv[10],
    "cudagraph_capture_sizes": json.loads(sys.argv[11]),
    "max_steps": 20,
    "wandb_project": "sna-specdec",
    "wandb_reuse": "never",
    "wandb_run_id": sys.argv[2],
}, sort_keys=True))
PY
}

assert_capture_coverage() {
  local variant="${1:-dflash}" max_shape capture_sizes
  valid_variant "${variant}"
  capture_sizes="$(capture_sizes_for "${variant}")"
  max_shape="$((8 * ($(k_for "${variant}") + 1)))"
  python3 - "${capture_sizes}" "${max_shape}" <<'PY'
import json
import sys

capture_sizes = json.loads(sys.argv[1])
max_shape = int(sys.argv[2])
shape_to_bucket = {shape: next(bucket for bucket in capture_sizes if bucket >= shape) for shape in range(1, max_shape + 1)}
print(json.dumps({"capture_sizes": capture_sizes, "shape_to_bucket": shape_to_bucket}, sort_keys=True))
PY
}

source_guard() {
  local automodel gym bridge megatron root_status bridge_status megatron_status
  test -e "${SOURCE_ROOT}/.git" || die "missing product source ${SOURCE_ROOT}"
  test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${SOURCE_SHA}" || die "product source SHA drift"
  if git -C "${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "product source has unresolved submodule gitlinks"
  fi
  automodel="${SOURCE_ROOT}/3rdparty/Automodel-workspace/Automodel"
  gym="${SOURCE_ROOT}/3rdparty/Gym-workspace/Gym"
  bridge="${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  megatron="${bridge}/3rdparty/Megatron-LM"
  git -C "${SOURCE_ROOT}" diff --ignore-submodules=dirty --quiet || die "product source has tracked changes"
  git -C "${SOURCE_ROOT}" diff --cached --ignore-submodules=dirty --quiet || die "product source has staged changes"
  git -C "${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git diff --ignore-submodules=dirty --quiet && git diff --cached --ignore-submodules=dirty --quiet' || die "product source submodule has tracked changes"
  root_status="$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all | grep -vFx ' M 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge' || true)"
  bridge_status="$(git -C "${bridge}" status --porcelain=v1 --untracked-files=all | grep -vFx ' M 3rdparty/Megatron-LM' || true)"
  megatron_status="$(git -C "${megatron}" status --porcelain=v1 --untracked-files=all | grep -vFx '?? megatron/core/datasets/helpers_cpp' || true)"
  test -z "${root_status}${bridge_status}${megatron_status}" || die "product source has unexpected generated or untracked files"
  test -z "$(git -C "${automodel}" status --porcelain=v1 --untracked-files=all)" || die "Automodel submodule is dirty"
  test -z "$(git -C "${gym}" status --porcelain=v1 --untracked-files=all)" || die "Gym submodule is dirty"
  test -r "${CONTAINER}" || die "missing immutable container"
}

harness_guard() {
  git -C "${SCRIPT_DIR}" diff --quiet -- . || die "experiment harness has tracked changes"
  git -C "${SCRIPT_DIR}" diff --cached --quiet -- . || die "experiment harness has staged changes"
  test -z "$(git -C "${SCRIPT_DIR}" status --porcelain=v1 --untracked-files=all -- .)" || die "experiment harness has untracked files"
}

materialize_config() {
  local input="$1" output="$2"
  python3 - "${input}" "${output}" "${SOURCE_ROOT}" <<'PY'
import pathlib
import sys

input_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
source_root = pathlib.Path(sys.argv[3])
expected_suffix = (
    "/examples/configs/recipes/llm/performance/"
    "grpo-qwen3-30ba3b-4n4g.yaml"
)
original_root = "/home/sna/nemorl-pr11-q30-eagle3-k3-product-clean-20260823"
contents = input_path.read_text()
old_default = f'"defaults": "{original_root}{expected_suffix}"'
new_default = f'"defaults": "{source_root}{expected_suffix}"'
if contents.count(old_default) != 1:
    raise SystemExit(f"unexpected defaults lineage in {input_path}")
output_path.write_text(contents.replace(old_default, new_default))
PY
}

preflight() {
  local variant="$1" comparison_arm checkpoint
  comparison_arm="$(comparison_arm_for "${variant}")"
  harness_guard
  source_guard
  [[ "${comparison_arm}" == baseline ]] && return
  checkpoint="$(checkpoint_for "${variant}")"
  if [[ "${comparison_arm}" == eagle3-k3 ]]; then
    python3 "${SCRIPT_DIR}/check_eagle3_checkpoint.py" --checkpoint "${checkpoint}" --target-model Qwen/Qwen3-30B-A3B --num-speculative-tokens 3
  else
    python3 "${SCRIPT_DIR}/check_checkpoint_state_dict.py" --variant "$(method_for "${variant}")" --checkpoint "${checkpoint}" --identity-file "$(identity_file_for "${variant}")"
  fi
}

write_sbatch() {
  local variant="$1" root="$2" comparison_arm run artifact_dir sbatch_path config checkpoint method identity_file capture_sizes checkpoint_gate
  comparison_arm="$(comparison_arm_for "${variant}")"
  run="$(run_id "${variant}")"
  artifact_dir="${root}/artifacts/${run}"
  sbatch_path="${artifact_dir}/job.sbatch"
  config="${SCRIPT_DIR}/configs/${comparison_arm}.yaml"
  checkpoint=""
  method=""
  identity_file=""
  if [[ "${comparison_arm}" != baseline ]]; then
    checkpoint="$(checkpoint_for "${variant}")"
    method="$(method_for "${variant}")"
    if [[ "${comparison_arm}" != eagle3-k3 ]]; then identity_file="$(identity_file_for "${variant}")"; fi
  fi
  capture_sizes="$(capture_sizes_for "${variant}")"
  checkpoint_gate="$(checkpoint_gate_for "${variant}" "${method}")"
  mkdir -p "${artifact_dir}"
  materialize_config "${config}" "${artifact_dir}/resolved-input-${variant}.yaml"
  if [[ "${comparison_arm}" == eagle3-k3 ]]; then cp "${SCRIPT_DIR}/check_eagle3_checkpoint.py" "${artifact_dir}/check_eagle3_checkpoint.py"; fi
  if [[ "${comparison_arm}" != baseline && "${comparison_arm}" != eagle3-k3 ]]; then cp "${SCRIPT_DIR}/check_checkpoint_state_dict.py" "${artifact_dir}/check_checkpoint_state_dict.py"; fi
  if [[ "${comparison_arm}" != baseline && "${comparison_arm}" != eagle3-k3 ]]; then cp "${identity_file}" "${artifact_dir}/checkpoint_identity.json"; fi
  cp "${SCRIPT_DIR}/verify_df9_configs.py" "${artifact_dir}/verify_df9_configs.py"
  cat >"${artifact_dir}/driver.sh" <<DRIVER
#!/usr/bin/env bash
set -euo pipefail
readonly SOURCE_ROOT="${SOURCE_ROOT}"
readonly SOURCE_SHA="${SOURCE_SHA}"
readonly ARTIFACT_DIR="${artifact_dir}"
readonly CONFIG="${artifact_dir}/resolved-input-${variant}.yaml"
$(if [[ "${comparison_arm}" != baseline ]]; then printf 'readonly CHECKPOINT="%s"' "${checkpoint}"; fi)
$(if [[ "${comparison_arm}" != baseline && "${comparison_arm}" != eagle3-k3 ]]; then printf 'readonly CHECKPOINT_IDENTITY="%s"' "${artifact_dir}/checkpoint_identity.json"; fi)
readonly VARIANT="${variant}"
readonly WANDB_ID="${run}"

die() { echo "Q30_20STEP_FAIL_CLOSED: \$*" >&2; exit 1; }
source_guard() {
  local automodel gym bridge megatron root_status bridge_status megatron_status
  test -e "\${SOURCE_ROOT}/.git" || die "missing product source \${SOURCE_ROOT}"
  test "\$(git -C "\${SOURCE_ROOT}" rev-parse HEAD)" = "\${SOURCE_SHA}" || die "product source SHA drift"
  if git -C "\${SOURCE_ROOT}" submodule status --recursive | grep -qE '^[+-U]'; then die "product source has unresolved submodule gitlinks"; fi
  automodel="\${SOURCE_ROOT}/3rdparty/Automodel-workspace/Automodel"
  gym="\${SOURCE_ROOT}/3rdparty/Gym-workspace/Gym"
  bridge="\${SOURCE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  megatron="\${bridge}/3rdparty/Megatron-LM"
  git -C "\${SOURCE_ROOT}" diff --ignore-submodules=dirty --quiet || die "product source has tracked changes"
  git -C "\${SOURCE_ROOT}" diff --cached --ignore-submodules=dirty --quiet || die "product source has staged changes"
  git -C "\${SOURCE_ROOT}" submodule foreach --quiet --recursive 'git diff --ignore-submodules=dirty --quiet && git diff --cached --ignore-submodules=dirty --quiet' || die "product source submodule has tracked changes"
  root_status="\$(git -C "\${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all | grep -vFx ' M 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge' || true)"
  bridge_status="\$(git -C "\${bridge}" status --porcelain=v1 --untracked-files=all | grep -vFx ' M 3rdparty/Megatron-LM' || true)"
  megatron_status="\$(git -C "\${megatron}" status --porcelain=v1 --untracked-files=all | grep -vFx '?? megatron/core/datasets/helpers_cpp' || true)"
  test -z "\${root_status}\${bridge_status}\${megatron_status}" || die "product source has unexpected generated or untracked files"
  test -z "\$(git -C "\${automodel}" status --porcelain=v1 --untracked-files=all)" || die "Automodel submodule is dirty"
  test -z "\$(git -C "\${gym}" status --porcelain=v1 --untracked-files=all)" || die "Gym submodule is dirty"
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
echo SETUP_GATE_PASS | tee "\${ARTIFACT_DIR}/gates.log"
python3 "\${ARTIFACT_DIR}/verify_df9_configs.py" --capture-sizes '${capture_sizes}' --source-root "\${SOURCE_ROOT}" --config "\${CONFIG}" | tee "\${ARTIFACT_DIR}/df9-compose.json"
${checkpoint_gate}
export WANDB_RUN_ID="\${WANDB_ID}"
train_log="\${ARTIFACT_DIR}/train.log"
setsid bash -c "set -o pipefail; cd '${SOURCE_ROOT}'; NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py --config '${artifact_dir}/resolved-input-${variant}.yaml' ++policy.generation.vllm_kwargs.max_num_seqs=8 ++policy.generation.vllm_kwargs.compilation_config.backend=eager ++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE ++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes} logger.log_dir='${artifact_dir}/logs' logger.wandb_enabled=True logger.wandb.project=sna-specdec logger.wandb.name='${run}' 2>&1 | tee '${artifact_dir}/train.log'" &
train_pid=\$!
wait_for_gate 'Capturing CUDA graphs.*100%|Graph capturing finished' CUDAGRAPH_GATE_PASS
wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*20' STEP1_GATE_PASS
wait_for_gate 'Step[[:space:]]+2[[:space:]]*/[[:space:]]*20' STEP2_GATE_PASS
wait "\${train_pid}"
DRIVER
  chmod 700 "${artifact_dir}/driver.sh"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.q30-20-${variant}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --export=ALL
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export PATH="/cm/local/apps/slurm/25.11/bin:\${PATH}"
export CONTAINER="${CONTAINER}"
export MOUNTS="/lustre:/lustre,/home:/home"
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=64
export ARTIFACT_DIR="${artifact_dir}"
export BASE_LOG_DIR="${artifact_dir}"
export NRL_FORCE_REBUILD_VENVS=true
export COMMAND='bash "${artifact_dir}/driver.sh"'
exec bash "${SOURCE_ROOT}/ray.sub"
SBATCH
  chmod 700 "${sbatch_path}"
  printf '%s\n' "${sbatch_path}"
}

write_testonly_receipt() {
  local variant="$1" sbatch_output="$2" receipt
  receipt="${DURABLE_ROOT}/preflight/${variant}.json"
  mkdir -p "$(dirname "${receipt}")"
  python3 - "${receipt}" "${variant}" "$(config_sha "${variant}")" "${sbatch_output}" <<PY
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "config_sha": sys.argv[3],
    "harness_sha": "${HARNESS_SHA}",
    "source_root": "${SOURCE_ROOT}",
    "source_sha": "${SOURCE_SHA}",
    "test_only_output": sys.argv[4],
    "variant": sys.argv[2],
}, sort_keys=True) + "\\n")
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
expected = {
    "config_sha": sys.argv[4],
    "harness_sha": sys.argv[2],
    "source_root": "${SOURCE_ROOT}",
    "source_sha": "${SOURCE_SHA}",
    "variant": variant,
}
if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("test_only_output"):
    raise SystemExit(f"invalid test-only receipt for {variant}")
PY
}

mode="${1:-}"
case "${mode}" in
  --assert-capture-coverage)
    [[ $# -le 2 ]] || usage
    assert_capture_coverage "${2:-dflash}"
    ;;
  --emit-manifest|--render-sbatch|--test-only|--submit)
    [[ $# -eq 2 ]] || usage
    variant="$2"
    valid_variant "${variant}"
    case "${mode}" in
      --emit-manifest) emit_manifest "${variant}" "$(run_id "${variant}")" ;;
      --render-sbatch) write_sbatch "${variant}" "${Q30_20STEP_RENDER_ROOT:?Q30_20STEP_RENDER_ROOT is required for render}" ;;
      --test-only)
        preflight "${variant}"
        sbatch_output="$(sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1)"
        write_testonly_receipt "${variant}" "${sbatch_output}"
        printf '%s\n' "${sbatch_output}"
        ;;
      --submit)
        preflight "${variant}"
        require_testonly_receipt "${variant}"
        record="${DURABLE_ROOT}/submissions/${variant}-${SOURCE_SHA}-${HARNESS_SHA}.json"
        mkdir -p "$(dirname "${record}")"
        (set -o noclobber; : >"${record}.lock") 2>/dev/null || die "actual ${variant} submission already exists or is in progress"
        trap 'rm -f "${record}.lock"' EXIT
        sbatch_output="$(sbatch "$(write_sbatch "${variant}" "${DURABLE_ROOT}")")"
        python3 - "${record}" "${variant}" "${sbatch_output}" <<PY
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({"job_output": sys.argv[3], "variant": sys.argv[2]}, sort_keys=True) + "\\n")
PY
        printf '%s\n' "${sbatch_output}"
        ;;
    esac
    ;;
  *) usage ;;
esac
