#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=qwen3_30ba3b_draft_cadence_200step_20260826
readonly SOURCE_ROOT=/home/sna/nemorl-q30-cadence-syncfix-product-20260902
readonly SOURCE_SHA=55607a6e784b00058587414ab31aa6ea663a4cfd
readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
readonly DURABLE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/${EXPERIMENT}
readonly ACCOUNT=nemotron_n4_post
readonly WANDB_GROUP=q30ba3b-draft-cadence-200step-20260826
readonly SEGMENT_COUNT=5
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
HARNESS_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
readonly HARNESS_SHA
usage() {
  echo "usage: $0 --emit-manifest VARIANT|--render-sbatch VARIANT|--test-only VARIANT|--submit VARIANT" >&2
  exit 2
}

die() { echo "Q30_CADENCE_FAIL_CLOSED: $*" >&2; exit 1; }

valid_variant() {
  case "$1" in
    baseline|dflash-static|dflash-static-cg2048|dflash-always|dflash-always-cg2048|dflash-fixed5|dflash-fixed5-cg2048|dflash-fixed10|dflash-fixed10-cg2048|dflash-fixed20|dflash-fixed20-cg2048|dflash-fixed20-retry|dflash-adaptive-v2|dflash-adaptive-v2-cg2048|dspark-static|dspark-static-cg2048|dspark-always|dspark-always-cg2048|dspark-always-cg2048-retry|dspark-fixed5|dspark-fixed5-cg2048|dspark-fixed10|dspark-fixed10-cg2048|dspark-fixed20|dspark-fixed20-cg2048|dspark-adaptive-v2|dspark-adaptive-v2-cg2048) ;;
    *) usage ;;
  esac
}

segmented_variant_guard() {
  case "$1" in
    baseline|dflash-static-cg2048|dflash-always-cg2048|dflash-fixed5-cg2048|dflash-fixed10-cg2048|dflash-fixed20-cg2048|dflash-adaptive-v2-cg2048|dspark-static-cg2048|dspark-always-cg2048|dspark-fixed5-cg2048|dspark-fixed10-cg2048|dspark-fixed20-cg2048|dspark-adaptive-v2-cg2048) ;;
    *) die "variant $1 is not approved for segmented execution" ;;
  esac
}

config_key_for() {
  case "$1" in
    dflash-fixed20-retry) printf '%s\n' dflash-fixed20 ;;
    dspark-always-cg2048-retry) printf '%s\n' dspark-always-cg2048 ;;
    *) printf '%s\n' "$1" ;;
  esac
}

drafter_for() {
  case "$1" in
    baseline) printf '%s\n' none ;;
    *) printf '%s\n' "${1%%-*}" ;;
  esac
}

refit_step_for() {
  case "$1" in
    dflash-fixed20-retry) printf '%s\n' 20 ;;
    dspark-always-cg2048-retry) printf '%s\n' 1 ;;
    baseline|*-static|*-static-cg2048|*-adaptive-v2|*-adaptive-v2-cg2048) ;;
    *-always|*-always-cg2048) printf '%s\n' 1 ;;
    *-fixed*-cg2048)
      local cadence="${1#*-fixed}"
      printf '%s\n' "${cadence%%-*}"
      ;;
    *-fixed*) printf '%s\n' "${1##*fixed}" ;;
    *) die "unknown refit schedule for $1" ;;
  esac
}

checkpoint_for() {
  python3 -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["policy"]["draft"]["model_name"])' "${SCRIPT_DIR}/configs/$(config_key_for "$1").yaml"
}

config_sha() {
  python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${SCRIPT_DIR}/configs/$(config_key_for "$1").yaml"
}

submission_record() {
  printf '%s\n' "${DURABLE_ROOT}/submissions/${1}-${SOURCE_SHA}-${HARNESS_SHA}.json"
}

run_id() {
  local variant="$1" k=5
  [[ "${variant}" == baseline ]] && k=0
  printf 'q30ba3b-200step-%s-k%s-segmented-%s-%s\n' \
    "${variant}" "${k}" "${SOURCE_SHA:0:12}" "${HARNESS_SHA:0:12}"
}

checkpoint_root_for() {
  local root="$1" variant="$2"
  printf '%s/checkpoints\n' "$(result_root_for "${root}" "${variant}")"
}

result_root_for() {
  local root="$1" variant="$2"
  printf '%s/runs/%s\n' "${root}" "$(run_id "${variant}")"
}

cadence_runtime_enabled_for() {
  case "$1" in
    baseline) printf 'false\n' ;;
    *-cg2048) printf 'true\n' ;;
    *) die "cadence runtime policy is undefined for unmatched variant $1" ;;
  esac
}

emit_manifest() {
  local variant="$1" run="$2" record
  record="$(submission_record "${variant}")"
  python3 - "${variant}" "${run}" "${HARNESS_SHA}" "${record}" <<PY
import json
import sys

variant = sys.argv[1]
gates = ["source-clean", "wandb-auth", "cudagraph", "step1", "step2"]
if variant != "baseline":
    gates.append("state-dict")
    if "-static" not in variant and "-adaptive-v2" not in variant:
        gates.append("draft-refit")
print(json.dumps({
    "variant": sys.argv[1],
    "source": {"root": "${SOURCE_ROOT}", "sha": "${SOURCE_SHA}"},
    "harness_sha": sys.argv[3],
    "container": "${CONTAINER}",
    "slurm": {"account": "${ACCOUNT}", "partition": "batch", "time": "04:00:00", "nodes": 4, "gpus_per_node": 4},
    "gates": gates,
    "max_steps": 200,
    "wandb_project": "sna-specdec",
    "wandb_group": "${WANDB_GROUP}",
    "wandb_reuse": "one-logical-run",
    "wandb_resume": ["allow", "must", "must", "must", "must"],
    "wandb_run_id": sys.argv[2],
    "result_root": "${DURABLE_ROOT}/runs/" + sys.argv[2],
    "checkpoint_root": "${DURABLE_ROOT}/runs/" + sys.argv[2] + "/checkpoints",
    "completion_receipt": "${DURABLE_ROOT}/runs/" + sys.argv[2] + "/completion-receipt.json",
    "cadence_runtime": {
        "enabled": sys.argv[1] != "baseline",
        "required_checkpoint_steps": [200] if sys.argv[1] != "baseline" else [],
    },
    "checkpointing": {
        "enabled": True,
        "save_optimizer": True,
        "save_period": 200,
        "keep_top_k": 1,
        "metric_name": None,
        "ft_save_period": 20,
        "ft_keep_latest_k": 2,
        "checkpoint_must_save_by": "00:02:45:00",
    },
    "segments": ${SEGMENT_COUNT},
    "afterok_segments": ${SEGMENT_COUNT} - 1,
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

harness_guard() {
  local branch head merge_ref remote remote_branch remote_head
  test -z "$(git -C "${SCRIPT_DIR}" status --porcelain=v1 --untracked-files=all)" || die "harness worktree is dirty"
  branch="$(git -C "${SCRIPT_DIR}" symbolic-ref --quiet --short HEAD)" || die "harness HEAD is detached"
  remote="$(git -C "${SCRIPT_DIR}" config --get "branch.${branch}.remote")" || die "harness branch has no configured remote"
  merge_ref="$(git -C "${SCRIPT_DIR}" config --get "branch.${branch}.merge")" || die "harness branch has no configured upstream"
  [[ "${merge_ref}" == refs/heads/* ]] || die "harness upstream ref is invalid"
  remote_branch="${merge_ref#refs/heads/}"
  head="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)" || die "cannot resolve harness HEAD"
  remote_head="$(git -C "${SCRIPT_DIR}" rev-parse "refs/remotes/${remote}/${remote_branch}")" || die "cannot resolve pushed harness upstream"
  [[ "${head}" == "${remote_head}" ]] || die "harness HEAD is not pushed to its configured upstream"
}

preflight() {
  local variant="$1"
  source_guard
  [[ "${variant}" == baseline ]] && return
  python3 "${SCRIPT_DIR}/check_checkpoint_state_dict.py" \
    --variant "$(drafter_for "${variant}")" \
    --checkpoint "$(checkpoint_for "${variant}")"
}

write_sbatch() {
  local variant="$1" root="$2" run result_root checkpoint_root artifact_dir sbatch_path config_key config config_digest checkpoint drafter cadence_enabled cadence_enabled_python cadence_verify_overrides cadence_train_overrides refit_step post_sync_exports checkpoint_gate refit_gate exclude_directive
  run="$(run_id "${variant}")"
  result_root="$(result_root_for "${root}" "${variant}")"
  checkpoint_root="$(checkpoint_root_for "${root}" "${variant}")"
  artifact_dir="${result_root}/artifacts"
  sbatch_path="${artifact_dir}/job.sbatch"
  config_key="$(config_key_for "${variant}")"
  config="${SCRIPT_DIR}/configs/${config_key}.yaml"
  config_digest="$(config_sha "${variant}")"
  drafter="$(drafter_for "${variant}")"
  cadence_enabled="$(cadence_runtime_enabled_for "${variant}")"
  cadence_enabled_python=False
  cadence_verify_overrides=""
  cadence_train_overrides=""
  if [[ "${cadence_enabled}" == true ]]; then
    cadence_enabled_python=True
    cadence_verify_overrides=" --override cadence_runtime.enabled=true --override '++cadence_runtime.required_checkpoint_steps=[200]'"
    cadence_train_overrides=" cadence_runtime.enabled=true '++cadence_runtime.required_checkpoint_steps=[200]'"
  fi
  refit_step="$(refit_step_for "${variant}")"
  checkpoint=""
  [[ "${drafter}" == none ]] || checkpoint="$(checkpoint_for "${variant}")"
  mkdir -p "${artifact_dir}"
  cp "${config}" "${artifact_dir}/resolved-input-${config_key}.yaml"
  mkdir -p "${artifact_dir}/patches"
  cp "${SCRIPT_DIR}/prepare_mcore_checkpoint_overlay.py" "${artifact_dir}/prepare_mcore_checkpoint_overlay.py"
  cp "${SCRIPT_DIR}/patches/mcore-precision-aware-lazy-state-checkpoint.patch" "${artifact_dir}/patches/mcore-precision-aware-lazy-state-checkpoint.patch"
  if [[ "${drafter}" != none ]]; then
    cp "${SCRIPT_DIR}/check_checkpoint_state_dict.py" "${artifact_dir}/check_checkpoint_state_dict.py"
    cp "${SCRIPT_DIR}/prepare_vllm_dspark_fap_overlay.py" "${artifact_dir}/prepare_vllm_dspark_fap_overlay.py"
    cp "${SCRIPT_DIR}/patches/vllm-0.25.1-pr48167-runtime.patch" "${artifact_dir}/patches/vllm-0.25.1-pr48167-runtime.patch"
    cp "${SCRIPT_DIR}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch" "${artifact_dir}/patches/vllm-0.25.1-pr48167-group-causality-followup.patch"
  fi
  cp "${SCRIPT_DIR}/verify_composed_configs.py" "${artifact_dir}/verify_composed_configs.py"
  cat >"${artifact_dir}/completion_receipt.py" <<PY
#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import sys
import tempfile

EXPECTED = {
    "variant": "${variant}",
    "run_id": "${run}",
    "source_root": "${SOURCE_ROOT}",
    "source_sha": "${SOURCE_SHA}",
    "harness_sha": "${HARNESS_SHA}",
    "config_path": "${artifact_dir}/resolved-input-${config_key}.yaml",
    "config_sha256": "${config_digest}",
    "result_root": "${result_root}",
    "checkpoint_root": "${checkpoint_root}",
    "checkpoint_path": "${checkpoint_root}/step_200",
    "cadence_runtime_enabled": ${cadence_enabled_python},
}
RECEIPT = pathlib.Path(EXPECTED["result_root"]) / "completion-receipt.json"


class InvalidCompletion(RuntimeError):
    pass


SAMPLE_BYTES = 65536
MAX_MANIFEST_FILES = 1024
MAX_MANIFEST_NODES = 4096
MANIFEST_ALGORITHM = "sha256-size-first-last-65536"


def require_file(path: pathlib.Path) -> int:
    if path.is_symlink() or not path.is_file():
        raise InvalidCompletion(f"required file is absent: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise InvalidCompletion(f"required file is empty: {path}")
    return size


def sampled_sha256(path: pathlib.Path, size: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"size={size}\0".encode())
    with path.open("rb") as stream:
        digest.update(stream.read(SAMPLE_BYTES))
        if size > SAMPLE_BYTES:
            stream.seek(max(0, size - SAMPLE_BYTES))
            digest.update(stream.read(SAMPLE_BYTES))
    return digest.hexdigest()


def manifest_file(path: pathlib.Path) -> dict[str, object]:
    size = require_file(path)
    return {
        "path": str(path),
        "size_bytes": size,
        "sample_sha256": sampled_sha256(path, size),
        "algorithm": MANIFEST_ALGORITHM,
    }


def manifest_tree(path: pathlib.Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise InvalidCompletion(f"required directory is absent: {path}")
    entries = []
    pending = [path]
    visited = 0
    while pending:
        current = pending.pop()
        children = []
        for child in current.iterdir():
            visited += 1
            if visited > MAX_MANIFEST_NODES:
                raise InvalidCompletion(f"artifact tree is unexpectedly large: {path}")
            children.append(child)
        children.sort(key=lambda item: item.name)
        for child in children:
            if len(entries) >= MAX_MANIFEST_FILES:
                raise InvalidCompletion(f"artifact tree is unexpectedly large: {path}")
            if child.is_symlink():
                raise InvalidCompletion(f"artifact tree contains a symlink: {child}")
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                size = child.stat().st_size
                entries.append(
                    {
                        "path": child.relative_to(path).as_posix(),
                        "size_bytes": size,
                        "sample_sha256": sampled_sha256(child, size),
                    }
                )
            else:
                raise InvalidCompletion(f"artifact tree contains a special file: {child}")
    entries.sort(key=lambda entry: str(entry["path"]))
    if not any(int(entry["size_bytes"]) > 0 for entry in entries):
        raise InvalidCompletion(f"artifact tree has no nonempty files: {path}")
    return {
        "root": str(path),
        "algorithm": MANIFEST_ALGORITHM,
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
    }


def require_optimizer_payload(manifest: dict[str, object]) -> None:
    files = manifest["files"]
    assert isinstance(files, list)
    metadata_suffixes = {".json", ".txt", ".yaml", ".yml"}
    if not any(
        isinstance(entry, dict)
        and int(entry["size_bytes"]) > 0
        and pathlib.PurePosixPath(str(entry["path"])).suffix not in metadata_suffixes
        for entry in files
    ):
        raise InvalidCompletion("policy optimizer payload is absent")


def load_json(path: pathlib.Path) -> dict[str, object]:
    require_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidCompletion(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise InvalidCompletion(f"JSON root is not an object: {path}")
    return payload


def build_payload() -> dict[str, object]:
    result_root = pathlib.Path(EXPECTED["result_root"])
    checkpoint_root = pathlib.Path(EXPECTED["checkpoint_root"])
    checkpoint = pathlib.Path(EXPECTED["checkpoint_path"])
    if checkpoint_root != result_root / "checkpoints":
        raise InvalidCompletion("checkpoint root is not the result-root checkpoints child")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise InvalidCompletion("terminal checkpoint directory is absent")
    config = pathlib.Path(EXPECTED["config_path"])
    require_file(config)
    if hashlib.sha256(config.read_bytes()).hexdigest() != EXPECTED["config_sha256"]:
        raise InvalidCompletion("resolved config digest mismatch")
    training_info = load_json(checkpoint / "training_info.json")
    if training_info.get("total_steps") != 200:
        raise InvalidCompletion("terminal checkpoint does not record total_steps=200")
    weights = checkpoint / "policy" / "weights"
    weights_evidence = manifest_tree(weights)
    optimizer = checkpoint / "policy" / "optimizer"
    if optimizer.is_dir() and not optimizer.is_symlink():
        optimizer_evidence = manifest_tree(optimizer)
        require_optimizer_payload(optimizer_evidence)
    else:
        iterations = []
        for candidate in weights.glob("iter_*"):
            suffix = candidate.name.removeprefix("iter_")
            if suffix.isdigit() and candidate.is_dir() and not candidate.is_symlink():
                iterations.append((int(suffix), candidate))
        if len(iterations) > 32:
            raise InvalidCompletion("too many policy checkpoint iterations")
        tracker = weights / "latest_checkpointed_iteration.txt"
        if tracker.exists():
            require_file(tracker)
            try:
                tracked_iteration = int(tracker.read_text().strip())
            except ValueError as error:
                raise InvalidCompletion("invalid latest checkpoint iteration") from error
            matching = [path for number, path in iterations if number == tracked_iteration]
            if len(matching) != 1:
                raise InvalidCompletion("latest checkpoint iteration is absent")
            iteration = matching[0]
        elif iterations:
            iteration = max(iterations)[1]
        else:
            raise InvalidCompletion("policy optimizer checkpoint iteration is absent")
        optimizer_evidence = manifest_tree(iteration)
        require_optimizer_payload(optimizer_evidence)
    dataloaders = sorted(checkpoint.glob("train_dataloader*.pt"))
    if not dataloaders:
        raise InvalidCompletion("train dataloader checkpoint is absent")
    dataloader_evidence = [manifest_file(dataloader) for dataloader in dataloaders]
    checkpoint_config = manifest_file(checkpoint / "config.yaml")
    cadence_receipt = None
    checkpoint_runtime = None
    schedule_runtime = None
    evidence_digest = None
    if EXPECTED["cadence_runtime_enabled"]:
        evidence = training_info.get("draft_terminal_evidence")
        if not isinstance(evidence, dict) or not evidence:
            raise InvalidCompletion("cadence terminal evidence is absent")
        cadence_path = checkpoint / "cadence-checkpoint-receipt.json"
        cadence = load_json(cadence_path)
        if (
            cadence.get("successful") is not True
            or cadence.get("checkpoint_path") != str(checkpoint)
            or cadence.get("current_step") != 200
            or cadence.get("cadence_terminal_evidence") != evidence
        ):
            raise InvalidCompletion("cadence checkpoint receipt is not terminal or bound")
        checkpoint_runtime_path = result_root / "checkpoint-runtime.json"
        schedule_runtime_path = result_root / "schedule-runtime.json"
        if not checkpoint_runtime_path.is_file() or not schedule_runtime_path.is_file():
            raise InvalidCompletion("cadence terminal closure files are absent")
        checkpoint_runtime_payload = load_json(checkpoint_runtime_path)
        schedule_runtime_payload = load_json(schedule_runtime_path)
        if checkpoint_runtime_payload != cadence:
            raise InvalidCompletion("cadence terminal closure checkpoint differs from final checkpoint receipt")
        if schedule_runtime_payload.get("current_step") != 200:
            raise InvalidCompletion("cadence terminal closure schedule is not at step 200")
        cadence_receipt = manifest_file(cadence_path)
        checkpoint_runtime = manifest_file(checkpoint_runtime_path)
        schedule_runtime = manifest_file(schedule_runtime_path)
        evidence_digest = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return {
        "schema_version": 1,
        "successful": True,
        **EXPECTED,
        "total_steps": 200,
        "artifacts": {
            "policy_weights": weights_evidence,
            "policy_optimizer": optimizer_evidence,
            "train_dataloaders": dataloader_evidence,
            "checkpoint_config": checkpoint_config,
            "cadence_receipt": cadence_receipt,
            "cadence_checkpoint_runtime": checkpoint_runtime,
            "cadence_schedule_runtime": schedule_runtime,
            "cadence_terminal_evidence_sha256": evidence_digest,
        },
    }


def validate() -> None:
    if not RECEIPT.exists():
        raise SystemExit(3)
    if RECEIPT.is_symlink() or not RECEIPT.is_file():
        raise InvalidCompletion("completion receipt is not a regular file")
    if load_json(RECEIPT) != build_payload():
        raise InvalidCompletion("completion receipt binding differs from terminal artifacts")


def write() -> None:
    if not pathlib.Path(EXPECTED["checkpoint_path"]).exists():
        raise SystemExit(3)
    payload = build_payload()
    if RECEIPT.exists():
        validate()
        return
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".completion-", suffix=".tmp", dir=RECEIPT.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, RECEIPT)
        except FileExistsError:
            validate()
            return
        directory = os.open(RECEIPT.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


try:
    if sys.argv[1:] == ["validate"]:
        validate()
    elif sys.argv[1:] == ["write"]:
        write()
    else:
        raise SystemExit("usage: completion_receipt.py validate|write")
except InvalidCompletion as error:
    print(f"invalid completion receipt: {error}", file=sys.stderr)
    raise SystemExit(2) from error
PY
  chmod 700 "${artifact_dir}/completion_receipt.py"
  checkpoint_gate=""
  refit_gate=""
  if [[ "${drafter}" != none ]]; then
    checkpoint_gate="python3 \"\${ARTIFACT_DIR}/check_checkpoint_state_dict.py\" --variant \"\${DRAFTER}\" --checkpoint \"\${CHECKPOINT}\" | tee -a \"\${GATES_LOG}\""
  fi
  if [[ -n "${refit_step}" ]]; then
    refit_gate="wait_for_gate 'draft_post_update_refit=complete step=${refit_step}' DRAFT_REFIT_GATE_PASS 0"
  fi
  cat >"${artifact_dir}/driver.sh" <<DRIVER
#!/usr/bin/env bash
set -euo pipefail
readonly SOURCE_ROOT="${SOURCE_ROOT}"
readonly SOURCE_SHA="${SOURCE_SHA}"
readonly ARTIFACT_DIR="${artifact_dir}"
readonly CONFIG="${artifact_dir}/resolved-input-${config_key}.yaml"
readonly CHECKPOINT="${checkpoint}"
readonly DRAFTER="${drafter}"
readonly WANDB_ID="${run}"
readonly SEGMENT_INDEX="\${Q30_SEGMENT_INDEX:-0}"
readonly CHECKPOINT_ROOT="${checkpoint_root}"
readonly SEGMENT_DIR="${artifact_dir}/segments/\${SEGMENT_INDEX}"
readonly GATES_LOG="\${SEGMENT_DIR}/gates.log"

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
    if grep -qE "\${pattern}" "\${train_log}"; then echo "\${marker}" | tee -a "\${GATES_LOG}"; return; fi
    if (( timeout_seconds > 0 && SECONDS >= deadline )); then
      kill -- "-\${train_pid}" 2>/dev/null || true
      wait "\${train_pid}" || true
      die "timed out waiting for \${marker}"
    fi
    sleep 10
  done
  wait "\${train_pid}" || die "training ended before \${marker}"
  grep -qE "\${pattern}" "\${train_log}" || die "missing \${marker}"
  echo "\${marker}" | tee -a "\${GATES_LOG}"
}

case "\${SEGMENT_INDEX}" in
  0) export WANDB_RESUME="allow" ;;
  1|2|3|4) export WANDB_RESUME="must" ;;
  *) die "segment index must be between 0 and 4" ;;
esac
mkdir -p "\${SEGMENT_DIR}"
completion_status=0
python3 "\${ARTIFACT_DIR}/completion_receipt.py" validate || completion_status=\$?
case "\${completion_status}" in
  0) echo "Q30_SEGMENT_COMPLETE: validated atomic completion receipt"; exit 0 ;;
  3) ;;
  *) die "completion receipt is malformed" ;;
esac
RESUME_STEP="\$(python3 - "\${CHECKPOINT_ROOT}" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
latest = 0
for checkpoint in root.glob("step_*"):
    if re.fullmatch(r"step_[0-9]+", checkpoint.name) is None:
        continue
    try:
        state = json.loads((checkpoint / "training_info.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        continue
    total_steps = state.get("total_steps")
    if isinstance(total_steps, int):
        latest = max(latest, total_steps)
print(latest)
PY
)"
readonly RESUME_STEP
if (( SEGMENT_INDEX > 0 && RESUME_STEP == 0 )); then
  die "continuation segment has no finalized checkpoint"
fi
source_guard
test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp" || die "missing node-local MCore overlay"
test -f "\${Q30_MCORE_OVERLAY}/mcore-precision-aware-lazy-state-checkpoint.json" || die "missing MCore checkpoint overlay receipt"
cp "\${Q30_MCORE_OVERLAY}/mcore-precision-aware-lazy-state-checkpoint.json" "\${SEGMENT_DIR}/mcore-checkpoint-overlay-receipt.json"
echo MCORE_OVERLAY_GATE_PASS | tee "\${GATES_LOG}"
echo MCORE_CHECKPOINT_OVERLAY_GATE_PASS | tee -a "\${GATES_LOG}"
if [[ "\${DRAFTER}" == dspark ]]; then
  test "\${NRL_VENV_POST_SYNC_TARGET:-}" = nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker || die "DSpark post-sync target is not the synchronous vLLM worker"
  test -f "\${NRL_VENV_POST_SYNC_SCRIPT:-}" || die "DSpark post-sync script is absent"
elif [[ "\${DRAFTER}" == dflash ]]; then
  test -z "\${NRL_VENV_POST_SYNC_SCRIPT:-}" || die "DFlash unexpectedly enabled the DSpark post-sync hook"
  echo STOCK_VLLM_GATE_PASS | tee -a "\${GATES_LOG}"
else
  test "\${DRAFTER}" = none || die "unknown drafter \${DRAFTER}"
  test -z "\${NRL_VENV_POST_SYNC_SCRIPT:-}" || die "baseline unexpectedly enabled a post-sync hook"
  echo NO_SPECDEC_GATE_PASS | tee -a "\${GATES_LOG}"
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
echo WANDB_AUTH_GATE_PASS | tee -a "\${GATES_LOG}"
(cd "\${SOURCE_ROOT}" && NRL_FORCE_REBUILD_VENVS=true UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync python3 "\${ARTIFACT_DIR}/verify_composed_configs.py" --source-root "\${SOURCE_ROOT}" --config "\${CONFIG}" --override checkpointing.enabled=true --override checkpointing.checkpoint_dir=${checkpoint_root} --override checkpointing.metric_name=null --override checkpointing.save_optimizer=true --override checkpointing.save_period=200 --override checkpointing.keep_top_k=1 --override ++checkpointing.ft_save_period=20 --override ++checkpointing.ft_keep_latest_k=2 --override checkpointing.checkpoint_must_save_by=00:02:45:00 --override ++cadence_runtime.result_dir=${result_root}${cadence_verify_overrides}) | tee "\${SEGMENT_DIR}/composed-config.json"
${checkpoint_gate}
export WANDB_RUN_ID="\${WANDB_ID}"
export WANDB_PROJECT=sna-specdec
export WANDB_MODE=online
train_log="\${SEGMENT_DIR}/train.log"
setsid bash -c "set -o pipefail; cd '${SOURCE_ROOT}'; NRL_FORCE_REBUILD_VENVS=true UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv run --frozen --no-sync examples/run_grpo.py --config '${artifact_dir}/resolved-input-${config_key}.yaml' logger.log_dir='\${SEGMENT_DIR}/logs' logger.wandb_enabled=true logger.wandb.project=sna-specdec +logger.wandb.group=${WANDB_GROUP} logger.wandb.name='${run}' checkpointing.enabled=true checkpointing.checkpoint_dir=${checkpoint_root} checkpointing.metric_name=null checkpointing.save_optimizer=true checkpointing.save_period=200 checkpointing.keep_top_k=1 ++checkpointing.ft_save_period=20 ++checkpointing.ft_keep_latest_k=2 checkpointing.checkpoint_must_save_by=00:02:45:00 ++cadence_runtime.result_dir=${result_root}${cadence_train_overrides} 2>&1 | tee '\${SEGMENT_DIR}/train.log'" &
train_pid=\$!
wait_for_gate 'Capturing CUDA graphs.*100%|Graph capturing finished' CUDAGRAPH_GATE_PASS 2700
if [[ "\${DRAFTER}" == dspark ]]; then
  receipt="\${Q30_VLLM_OVERLAY}/dspark-fap-vllm-48167-runtime.json"
  test -f "\${receipt}" || die "missing DSpark vLLM overlay receipt after actor venv sync"
  python3 - "\${receipt}" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt.get("patch_sha256") != "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317":
    raise SystemExit("DSpark vLLM overlay patch digest mismatch")
if receipt.get("status") not in {"applied", "already-patched"}:
    raise SystemExit("DSpark vLLM overlay status is invalid")
if len(receipt.get("patched_files", {})) != 10:
    raise SystemExit("DSpark vLLM overlay does not cover all ten runtime files")
if receipt.get("followup_patch_sha256") != "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90":
    raise SystemExit("DSpark group-causality follow-up digest mismatch")
if receipt.get("followup_status") not in {"applied", "already-patched"}:
    raise SystemExit("DSpark group-causality follow-up status is invalid")
if set(receipt.get("followup_patched_files", {})) != {"vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"}:
    raise SystemExit("DSpark group-causality follow-up coverage is invalid")
PY
  cp "\${receipt}" "\${SEGMENT_DIR}/vllm-dspark-fap-overlay-receipt.json"
  echo DSPARK_VLLM_OVERLAY_GATE_PASS | tee -a "\${GATES_LOG}"
fi
if (( RESUME_STEP == 0 )); then
  wait_for_gate 'Step[[:space:]]+1[[:space:]]*/[[:space:]]*200' STEP1_GATE_PASS 2700
  wait_for_gate 'Step[[:space:]]+2[[:space:]]*/[[:space:]]*200' STEP2_GATE_PASS 2700
  ${refit_gate}
elif (( RESUME_STEP < 200 )); then
  echo "Q30_RESUME_CHECKPOINT_SELECTED step=\${RESUME_STEP}" | tee -a "\${GATES_LOG}"
  wait_for_gate 'Checkpoint loaded' RESUME_CHECKPOINT_LOAD_GATE_PASS 2700
  NEXT_STEP=\$((RESUME_STEP + 1))
  wait_for_gate "Step[[:space:]]+\${NEXT_STEP}[[:space:]]*/[[:space:]]*200" RESUME_NEXT_STEP_GATE_PASS 0
else
  echo "Q30_TERMINAL_CHECKPOINT_SELECTED step=\${RESUME_STEP}" | tee -a "\${GATES_LOG}"
  wait_for_gate 'Checkpoint loaded' TERMINAL_CHECKPOINT_LOAD_GATE_PASS 2700
fi
wait "\${train_pid}"
completion_status=0
python3 "\${ARTIFACT_DIR}/completion_receipt.py" write || completion_status=\$?
case "\${completion_status}" in
  0) echo COMPLETION_RECEIPT_GATE_PASS | tee -a "\${GATES_LOG}" ;;
  3)
    if (( SEGMENT_INDEX == 4 )); then
      die "final segment ended without valid Step-200 completion receipt"
    fi
    echo "Q30_SEGMENT_INCOMPLETE: terminal checkpoint not reached" | tee -a "\${GATES_LOG}"
    ;;
  *) die "terminal checkpoint artifacts cannot be sealed" ;;
esac
DRIVER
  chmod 700 "${artifact_dir}/driver.sh"
  post_sync_exports=""
  if [[ "${drafter}" == dspark ]]; then
    post_sync_exports="export NRL_VENV_POST_SYNC_SCRIPT=\"${artifact_dir}/prepare_vllm_dspark_fap_overlay.py\"
export NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
  fi
  exclude_directive=""
  case "${variant}" in
    dflash-fixed20-retry)
      exclude_directive="#SBATCH --exclude=nvl72047-T16"
      ;;
    dspark-always-cg2048 | dspark-always-cg2048-retry | dspark-fixed5-cg2048 | dspark-fixed10-cg2048 | dspark-fixed20-cg2048 | dspark-adaptive-v2-cg2048)
      exclude_directive="#SBATCH --exclude=nvl72118-T01"
      ;;
  esac
  cat >"${sbatch_path}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=sna-q30-c200-${variant}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
${exclude_directive}
#SBATCH --output=${artifact_dir}/slurm-%j.out
#SBATCH --error=${artifact_dir}/slurm-%j.err
set -euo pipefail
completion_status=0
python3 "${artifact_dir}/completion_receipt.py" validate || completion_status=\$?
case "\${completion_status}" in
  0) echo "Q30_SEGMENT_COMPLETE_BEFORE_RAY: validated atomic completion receipt"; exit 0 ;;
  3) ;;
  *) echo "Q30_CADENCE_FAIL_CLOSED: completion receipt is malformed" >&2; exit 1 ;;
esac
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
export NEMO_RL_VENV_DIR="\${Q30_NODE_ROOT}/venvs"
export PYTHONPATH="\${Q30_VLLM_OVERLAY}:\${Q30_MCORE_OVERLAY}:\${SOURCE_ROOT}:\${PYTHONPATH:-}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=PYTHONPATH
export SETUP_COMMAND='set -euo pipefail; python3 "\${ARTIFACT_DIR}/prepare_mcore_checkpoint_overlay.py" --source-root "\${Q30_MCORE_SOURCE}" --overlay-root "\${Q30_MCORE_OVERLAY}" --patch "\${ARTIFACT_DIR}/patches/mcore-precision-aware-lazy-state-checkpoint.patch"; test -f "\${Q30_MCORE_OVERLAY}/megatron/core/datasets/helpers.cpp"'
${post_sync_exports}
export ARTIFACT_DIR="${artifact_dir}"
export BASE_LOG_DIR="${artifact_dir}"
export NRL_FORCE_REBUILD_VENVS=true
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
export WANDB_API_KEY="\${WANDB_API_KEY:?WANDB_API_KEY must be exported at submission}"
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
  python3 - "${record}" "${variant}" "${SOURCE_SHA}" "${HARNESS_SHA}" "${run}" "${artifact_dir}" "${sbatch_path}" "${SEGMENT_COUNT}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "artifact_dir": sys.argv[6],
    "harness_sha": sys.argv[4],
    "job_ids": [],
    "next_segment": 0,
    "run_id": sys.argv[5],
    "sbatch_path": sys.argv[7],
    "schema_version": 1,
    "source_sha": sys.argv[3],
    "state": "submitting",
    "segments": int(sys.argv[8]),
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
  python3 - "${record}" "${variant}" "${SOURCE_SHA}" "${HARNESS_SHA}" "${run}" "${artifact_dir}" "${sbatch_path}" "${SEGMENT_COUNT}" <<'PY'
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
segment_count = int(sys.argv[8])
submitting = {
    "artifact_dir": sys.argv[6],
    "harness_sha": sys.argv[4],
    "job_ids": [],
    "next_segment": 0,
    "run_id": sys.argv[5],
    "sbatch_path": sys.argv[7],
    "schema_version": 1,
    "source_sha": sys.argv[3],
    "state": "submitting",
    "segments": segment_count,
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


accepted_job_ids: list[str] = []
all_safe_output: list[str] = []
total_output_bytes = 0

for segment_index in range(segment_count):
    arguments = ["sbatch"]
    if accepted_job_ids:
        arguments.append(f"--dependency=afterok:{accepted_job_ids[-1]}")
    arguments.extend(
        [f"--export=ALL,Q30_SEGMENT_INDEX={segment_index}", sys.argv[7]]
    )
    captured = bytearray()
    output_bytes = [0]
    timed_out = False
    try:
        process = subprocess.Popen(
            arguments,
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
    candidate_job_ids = [
        match.decode() for match in matched_job_ids[:MAX_SAFE_LINES]
    ]
    safe_output = [
        f"Submitted batch job {job_id}" for job_id in candidate_job_ids
    ]
    output_truncated = (
        output_bytes[0] > len(captured) or len(matched_job_ids) > MAX_SAFE_LINES
    )
    total_output_bytes += output_bytes[0]
    if (
        scheduler_exit_status != 0
        or timed_out
        or output_truncated
        or len(candidate_job_ids) != 1
    ):
        outcome = {
            **submitting,
            "candidate_job_ids": candidate_job_ids,
            "job_ids": accepted_job_ids,
            "next_segment": segment_index,
            "scheduler_exit_status": scheduler_exit_status,
            "scheduler_output_bytes": output_bytes[0],
            "scheduler_output_truncated": output_truncated,
            "scheduler_safe_output": safe_output,
            "scheduler_timed_out": timed_out,
            "state": "ambiguous",
        }
        replace_receipt(outcome)
        print(
            f"scheduler submission outcome is ambiguous for segment "
            f"{segment_index} (exit_status={scheduler_exit_status}); reconcile "
            f"{path} before retrying",
            file=sys.stderr,
        )
        raise SystemExit(1)

    accepted_job_ids.append(candidate_job_ids[0])
    all_safe_output.extend(safe_output)
    progress = {
        **submitting,
        "job_ids": accepted_job_ids,
        "next_segment": segment_index + 1,
    }
    replace_receipt(progress)
    print(safe_output[0])

outcome = {
    **submitting,
    "job_id": accepted_job_ids[0],
    "job_ids": accepted_job_ids,
    "next_segment": segment_count,
    "scheduler_exit_status": 0,
    "scheduler_output_bytes": total_output_bytes,
    "scheduler_output_truncated": False,
    "scheduler_safe_output": all_safe_output,
    "scheduler_timed_out": False,
    "state": "accepted",
}
replace_receipt(outcome)
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
      --render-sbatch)
        segmented_variant_guard "${variant}"
        write_sbatch "${variant}" "${Q30_CADENCE_RENDER_ROOT:?Q30_CADENCE_RENDER_ROOT is required for render}"
        ;;
      --test-only)
        segmented_variant_guard "${variant}"
        harness_guard
        preflight "${variant}"
        sbatch_output="$(sbatch --test-only "$(write_sbatch "${variant}" "${DURABLE_ROOT}")" 2>&1)"
        write_testonly_receipt "${variant}" "${sbatch_output}"
        printf '%s\n' "${sbatch_output}"
        ;;
      --submit)
        segmented_variant_guard "${variant}"
        harness_guard
        preflight "${variant}"
        require_testonly_receipt "${variant}"
        record="$(submission_record "${variant}")"
        sbatch_path="$(write_sbatch "${variant}" "${DURABLE_ROOT}")"
        artifact_dir="$(dirname -- "${sbatch_path}")"
        run="$(basename -- "$(dirname -- "${artifact_dir}")")"
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
