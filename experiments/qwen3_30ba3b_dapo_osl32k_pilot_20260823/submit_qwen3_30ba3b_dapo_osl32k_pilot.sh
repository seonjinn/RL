#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_SHA=d0c4f1110cca28c75b7a1d98ed2d5f197e7d01dc
readonly HARNESS_BASE_SHA=7bca9a95e7bafb85c42cd03912f85113dcf82945
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_dapo_osl32k_pilot_20260823
readonly DATA_SOURCE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/datasets--BytedTsinghua-SIA--DAPO-Math-17k/snapshots/65877096c24ffa7abc4e4fa5edb95cf3413a5674/data/dapo-math-17k.parquet
readonly DATASET=${DURABLE_ROOT}/data/dapo-math-17k-r658770-first64.jsonl
readonly ACCOUNT=nemotron_n3_post
readonly CAPTURE_SIZES='[1,2,4,8,12,16,24,32,40,48,56,64]'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly HARNESS_SHA

usage() {
  echo "usage: $0 --stage-data|--assert-capture-coverage|--emit-manifest VARIANT|--render-sbatch VARIANT|--test-only VARIANT|--submit VARIANT" >&2
  exit 2
}

die() { echo "Q30_DAPO32K_FAIL_CLOSED: $*" >&2; exit 1; }

valid_variant() {
  case "$1" in baseline-k0|dflash-k7|dspark-k7) ;; *) usage ;; esac
}

source_root_for() {
  printf '/home/sna/nemorl-pr11-q30-dapo32k-%s-clean-20260823\n' "$1"
}

checkpoint_for() {
  case "$1" in
    dflash-k7) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dflash-b8-16n-migrated-oci-s4400/exported-checkpoint-14500 ;;
    dspark-k7) printf '%s\n' /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/training/lyris-q30b-nemo-dspark-b8-16n-migrated-oci-s5700/exported-checkpoint-14500 ;;
    *) die "baseline has no checkpoint" ;;
  esac
}

method_for() {
  case "$1" in
    dflash-k7) printf '%s\n' dflash ;;
    dspark-k7) printf '%s\n' dspark ;;
    *) die "baseline has no method" ;;
  esac
}

config_sha() {
  python3 -c 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$1.yaml"
}

run_id() {
  python3 - "$1" <<'PY'
import sys
import uuid
print(f"q30ba3b-dapo-osl32k-pilot-{sys.argv[1]}-{uuid.uuid4().hex}")
PY
}

source_guard() {
  local variant="$1" source_root
  source_root="$(source_root_for "${variant}")"
  test -e "${source_root}/.git" || die "missing product source ${source_root}"
  test "$(git -C "${source_root}" rev-parse HEAD)" = "${SOURCE_SHA}" || die "product source SHA drift"
  test -z "$(git -C "${source_root}" status --porcelain=v1 --untracked-files=all)" || die "product source is dirty"
  if git -C "${source_root}" submodule status --recursive | grep -qE '^[+-U]'; then
    die "product source has unresolved submodule gitlinks"
  fi
  test -z "$(git -C "${source_root}" submodule foreach --quiet --recursive 'git status --porcelain=v1 --untracked-files=all')" || die "product source submodule is dirty"
  test -r "${CONTAINER}" || die "missing immutable container"
}

stage_data() {
  python3 "${SCRIPT_DIR}/verify_dapo_slice.py" --source "${DATA_SOURCE}" --output "${DATASET}" --identity-file "${SCRIPT_DIR}/dataset_identity.json"
}

preflight() {
  local variant="$1" source_root
  source_root="$(source_root_for "${variant}")"
  source_guard "${variant}"
  python3 "${SCRIPT_DIR}/verify_dapo_slice.py" --source "${DATA_SOURCE}" --output "${DATASET}" --identity-file "${SCRIPT_DIR}/dataset_identity.json" --verify-only
  python3 "${SCRIPT_DIR}/verify_pilot_config.py" --source-root "${source_root}" --config "${SCRIPT_DIR}/configs/${variant}.yaml" --capture-sizes "${CAPTURE_SIZES}" --static-only
  if [[ "${variant}" != baseline-k0 ]]; then
    python3 "${SCRIPT_DIR}/check_checkpoint_state_dict.py" --variant "$(method_for "${variant}")" --checkpoint "$(checkpoint_for "${variant}")" --identity-file "${SCRIPT_DIR}/checkpoint_identity.json" --verify-content-sha
  fi
}

assert_capture_coverage() {
  python3 - "${CAPTURE_SIZES}" <<'PY'
import json
import sys
sizes = json.loads(sys.argv[1])
mapping = {shape: next(bucket for bucket in sizes if bucket >= shape) for shape in range(1, 65)}
print(json.dumps({"capture_sizes": sizes, "shape_to_bucket": mapping}, sort_keys=True))
PY
}

emit_manifest() {
  local variant="$1" run="$2" source_root checkpoint method k gates
  source_root="$(source_root_for "${variant}")"
  checkpoint=""
  method=""
  k=0
  gates='["source-clean","data-identity","config-compose","cudagraph","step1","step2","wake-refit","output-length","no-fatal"]'
  if [[ "${variant}" != baseline-k0 ]]; then
    checkpoint="$(checkpoint_for "${variant}")"
    method="$(method_for "${variant}")"
    k=7
    gates='["source-clean","data-identity","config-compose","state-dict","cudagraph","step1","step2","wake-refit","output-length","no-fatal"]'
  fi
  python3 - "${variant}" "${run}" "${source_root}" "${checkpoint}" "${method}" "${k}" "${gates}" "${SCRIPT_DIR}/dataset_identity.json" <<PY
import json
import sys
identity = json.load(open(sys.argv[8]))
print(json.dumps({
  "variant": sys.argv[1],
  "source": {"root": sys.argv[3], "sha": "${SOURCE_SHA}"},
  "harness_base_sha": "${HARNESS_BASE_SHA}",
  "harness_sha": "${HARNESS_SHA}",
  "container": "${CONTAINER}",
  "dataset": {"path": "${DATASET}", "source_revision": identity["source"]["revision"], "source_sha256": identity["source"]["sha256"], "slice_sha256": identity["slice"]["sha256"], "rows": identity["slice"]["rows"], "source_order": identity["slice"]["source_order"], "seed": identity["slice"]["seed"]},
  "checkpoint": sys.argv[4] or None,
  "method": sys.argv[5] or None,
  "num_speculative_tokens": int(sys.argv[6]),
  "capture_sizes": json.loads("${CAPTURE_SIZES}"),
  "topology": {"nodes": 4, "gpus_per_node": 4, "tp": 2, "pp": 1, "ep": 8, "cp": 2},
  "max_input_length": 2048, "max_output_length": 32768, "max_model_len": 40960,
  "global_batch_size": 64, "max_steps": 2,
  "slurm": {"account": "${ACCOUNT}", "partition": "batch", "qos": "normal", "nodes": 4, "gpus_per_node": 4},
  "gates": json.loads(sys.argv[7]),
  "wandb_project": "sna-specdec", "wandb_reuse": "never", "wandb_run_id": sys.argv[2],
  "wandb_url": f"https://wandb.ai/nvidia/sna-specdec/runs/{sys.argv[2]}",
}, sort_keys=True))
PY
}

write_sbatch() {
  local variant="$1" output_root="$2" run artifact_dir sbatch_path config source_root checkpoint method
  run="$(run_id "${variant}")"
  artifact_dir="${output_root}/artifacts/${run}"
  sbatch_path="${artifact_dir}/job.sbatch"
  config="${SCRIPT_DIR}/configs/${variant}.yaml"
  source_root="$(source_root_for "${variant}")"
  checkpoint=""
  method=""
  if [[ "${variant}" != baseline-k0 ]]; then
    checkpoint="$(checkpoint_for "${variant}")"
    method="$(method_for "${variant}")"
  fi
  mkdir -p "${artifact_dir}"
  cp "${config}" "${artifact_dir}/resolved-input-${variant}.yaml"
  cp "${SCRIPT_DIR}/run_pilot_arm.sh" "${artifact_dir}/driver.sh"
  cp "${SCRIPT_DIR}/verify_pilot_config.py" "${artifact_dir}/verify_pilot_config.py"
  cp "${SCRIPT_DIR}/verify_dapo_slice.py" "${artifact_dir}/verify_dapo_slice.py"
  cp "${SCRIPT_DIR}/dataset_identity.json" "${artifact_dir}/dataset_identity.json"
  cp "${SCRIPT_DIR}/summarize_output_lengths.py" "${artifact_dir}/summarize_output_lengths.py"
  if [[ "${variant}" != baseline-k0 ]]; then
    cp "${SCRIPT_DIR}/check_checkpoint_state_dict.py" "${artifact_dir}/check_checkpoint_state_dict.py"
    cp "${SCRIPT_DIR}/checkpoint_identity.json" "${artifact_dir}/checkpoint_identity.json"
  fi
  emit_manifest "${variant}" "${run}" >"${artifact_dir}/manifest.json"
  chmod 700 "${artifact_dir}/driver.sh"
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${ACCOUNT}.q30-dapo32k-${variant}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
export CONTAINER="${CONTAINER}"
export MOUNTS="/lustre:/lustre,/home:/home,/tmp:/tmp"
export GPUS_PER_NODE=4
export SOURCE_ROOT="${source_root}"
export SOURCE_SHA="${SOURCE_SHA}"
export ARTIFACT_DIR="${artifact_dir}"
export CONFIG="${artifact_dir}/resolved-input-${variant}.yaml"
export DATA_SOURCE="${DATA_SOURCE}"
export DATASET="${DATASET}"
export VARIANT="${variant}"
export WANDB_ID="${run}"
export METHOD="${method}"
export CHECKPOINT="${checkpoint}"
export CHECKPOINT_IDENTITY="${artifact_dir}/checkpoint_identity.json"
export BASE_LOG_DIR="${artifact_dir}"
export NRL_FORCE_REBUILD_VENVS=true
export TMPDIR=/tmp
export COMMAND='bash "${artifact_dir}/driver.sh"'
exec bash "${source_root}/ray.sub"
SBATCH
  chmod 700 "${sbatch_path}"
  printf '%s\n' "${sbatch_path}"
}

write_testonly_receipt() {
  local variant="$1" output="$2" receipt
  receipt="${DURABLE_ROOT}/preflight/${variant}.json"
  mkdir -p "$(dirname "${receipt}")"
  python3 - "${receipt}" "${variant}" "$(config_sha "${variant}")" "${output}" <<PY
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"variant": sys.argv[2], "config_sha": sys.argv[3], "test_only_output": sys.argv[4], "harness_sha": "${HARNESS_SHA}", "source_sha": "${SOURCE_SHA}"}, sort_keys=True) + "\n")
PY
}

require_testonly_receipt() {
  local variant="$1"
  python3 - "${DURABLE_ROOT}/preflight/${variant}.json" "${variant}" "$(config_sha "${variant}")" <<PY
import json
import pathlib
import sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {"variant": sys.argv[2], "config_sha": sys.argv[3], "harness_sha": "${HARNESS_SHA}", "source_sha": "${SOURCE_SHA}"}
if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("test_only_output"):
    raise SystemExit(f"invalid test-only receipt: {receipt}")
PY
}

mode="${1:-}"
case "${mode}" in
  --stage-data)
    [[ $# -eq 1 ]] || usage
    stage_data
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
      --render-sbatch)
        render_root="${Q30_DAPO32K_RENDER_ROOT:-$(mktemp -d /tmp/q30-dapo32k-render.XXXXXX)}"
        write_sbatch "${variant}" "${render_root}"
        ;;
      --test-only)
        preflight "${variant}"
        if ! sbatch_output="$(sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1)"; then
          printf 'TEST_ONLY_SCHEDULER_REJECTED: %s\n' "${sbatch_output}" >&2
          die "scheduler rejected test-only ${variant}"
        fi
        write_testonly_receipt "${variant}" "${sbatch_output}"
        printf '%s\n' "${sbatch_output}"
        ;;
      --submit)
        preflight "${variant}"
        require_testonly_receipt "${variant}"
        record="${DURABLE_ROOT}/submissions/${variant}-${HARNESS_SHA}.json"
        mkdir -p "$(dirname "${record}")"
        test ! -e "${record}" || die "actual ${variant} submission already exists"
        test ! -e "${record}.lock" || die "actual ${variant} submission already exists or is in progress"
        (set -o noclobber; : >"${record}.lock") 2>/dev/null || die "actual ${variant} submission already exists or is in progress"
        sbatch_path="$(write_sbatch "${variant}" "${DURABLE_ROOT}")"
        if ! sbatch_output="$(sbatch "${sbatch_path}" 2>&1)"; then
          printf 'ACTUAL_SCHEDULER_REJECTED: %s\n' "${sbatch_output}" >&2
          die "scheduler rejected actual ${variant}; lock retained for reconciliation"
        fi
        python3 - "${record}" "${variant}" "${sbatch_output}" "${sbatch_path}" <<PY
import json
import pathlib
import sys
manifest = json.loads((pathlib.Path(sys.argv[4]).parent / "manifest.json").read_text())
pathlib.Path(sys.argv[1]).write_text(json.dumps({"variant": sys.argv[2], "job_output": sys.argv[3], "sbatch": sys.argv[4], "wandb_run_id": manifest["wandb_run_id"], "wandb_url": manifest["wandb_url"]}, sort_keys=True) + "\n")
PY
        rm -f "${record}.lock"
        printf '%s\n' "${sbatch_output}"
        ;;
    esac
    ;;
  *) usage ;;
esac
