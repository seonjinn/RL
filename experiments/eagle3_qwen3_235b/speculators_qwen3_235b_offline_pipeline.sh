#!/usr/bin/env bash
set -euo pipefail

# Offline vLLM Speculators EAGLE3 pipeline for Qwen3-235B.
#
# This intentionally does not reuse ModelOpt .pt hidden-state dumps. Speculators
# trains from its own prepared Arrow dataset plus vLLM extract_hidden_states
# .safetensors files:
#
#   conversations JSONL -> prepare_data.py -> hs_*.safetensors -> train.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/eagle3_qwen3_235b"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
SPECULATORS_DIR="${SPECULATORS_DIR:-$ARTIFACT_ROOT/repos/speculators}"
SPECULATORS_GIT_URL="${SPECULATORS_GIT_URL:-https://github.com/vllm-project/speculators.git}"
SPECULATORS_REF="${SPECULATORS_REF:-e8d9da16d9c7a3d3cfca980a721b9681b3105c52}"

MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
SOURCE_CONVERSATIONS="${SOURCE_CONVERSATIONS:-$ARTIFACT_ROOT/data/openmath_reasoning_cot_conversations_50k.jsonl}"
SPECULATORS_JSONL="${SPECULATORS_JSONL:-$ARTIFACT_ROOT/data/openmath_reasoning_cot_conversations_50k_speculators.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/speculators/eagle3_openmath_reasoning_cot_50k}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$OUTPUT_DIR/hidden_states}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
VLLM_TMP_HIDDEN_STATES="${VLLM_TMP_HIDDEN_STATES:-$OUTPUT_DIR/vllm_tmp_hidden_states}"
HIDDEN_STATE_MANIFEST="${HIDDEN_STATE_MANIFEST:-$HIDDEN_STATES_DIR/nrl_hidden_state_manifest.json}"

SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MIN_HIDDEN_STATES="${MIN_HIDDEN_STATES:-0}"
MINIMUM_VALID_TOKENS="${MINIMUM_VALID_TOKENS:-1}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 46 90}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
SPECULATOR_TYPE="${SPECULATOR_TYPE:-eagle3}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
TTT_STEPS="${TTT_STEPS:-3}"
TTT_STEP_LOSS_DECAY="${TTT_STEP_LOSS_DECAY:-1.0}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-256}"
MASK_TOKEN_ID="${MASK_TOKEN_ID:-}"
NUM_LAYERS="${NUM_LAYERS:-1}"
DRAFT_ARCH="${DRAFT_ARCH:-llama}"
DRAFT_HIDDEN_ACT="${DRAFT_HIDDEN_ACT:-}"
FROM_PRETRAINED="${FROM_PRETRAINED:-}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
SPECULATORS_DISABLE_TORCH_COMPILE="${SPECULATORS_DISABLE_TORCH_COMPILE:-false}"
SPECULATORS_FSDP_WRAP_LAYERS="${SPECULATORS_FSDP_WRAP_LAYERS:-true}"
PREPARE_MANIFEST="${PREPARE_MANIFEST:-$OUTPUT_DIR/nrl_prepare_manifest.json}"
WRITE_PREPARE_MANIFEST="${WRITE_PREPARE_MANIFEST:-true}"
VALIDATE_HIDDEN_STATE_COVERAGE="${VALIDATE_HIDDEN_STATE_COVERAGE:-true}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_DP="${VLLM_DP:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$SEQ_LENGTH}"
VLLM_SITE="${VLLM_SITE:-}"
DEFAULT_VLLM_SITE="$ARTIFACT_ROOT/python_site/vllm_0_17_0_extract_py312"
if [[ -z "$VLLM_SITE" && -d "$DEFAULT_VLLM_SITE/vllm" ]]; then
  VLLM_SITE="$DEFAULT_VLLM_SITE"
fi
VLLM_LAUNCH_EXTRA_ARGS="${VLLM_LAUNCH_EXTRA_ARGS:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
CONCURRENCY="${CONCURRENCY:-16}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-240}"
MAX_RETRIES="${MAX_RETRIES:-3}"
DATAGEN_START_INDEX="${DATAGEN_START_INDEX:-}"
DATAGEN_END_INDEX="${DATAGEN_END_INDEX:-}"

RUN_CLONE="${RUN_CLONE:-true}"
RUN_CONVERT="${RUN_CONVERT:-true}"
RUN_PREPARE="${RUN_PREPARE:-true}"
RUN_DATAGEN="${RUN_DATAGEN:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
VALIDATE_OUTPUTS="${VALIDATE_OUTPUTS:-true}"
VALIDATE_SOURCE_CONVERSATIONS="${VALIDATE_SOURCE_CONVERSATIONS:-true}"
FAIL_ON_DUPLICATE_PROMPTS="${FAIL_ON_DUPLICATE_PROMPTS:-true}"
DENYLIST_PROMPTS_FROM="${DENYLIST_PROMPTS_FROM:-}"
INSTALL_SPECULATORS="${INSTALL_SPECULATORS:-true}"
APPLY_COMPAT_PATCHES="${APPLY_COMPAT_PATCHES:-true}"
DRY_RUN="${DRY_RUN:-false}"

HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export SPECULATORS_DISABLE_TORCH_COMPILE
export SPECULATORS_FSDP_WRAP_LAYERS

mkdir -p "$ARTIFACT_ROOT/repos" "$OUTPUT_DIR" "$HIDDEN_STATES_DIR" "$CHECKPOINT_DIR" "$VLLM_TMP_HIDDEN_STATES"

if [[ "$RUN_CLONE" == "true" || "$RUN_CLONE" == "True" ]]; then
  if [[ ! -d "$SPECULATORS_DIR/.git" ]]; then
    git clone --depth 1 "$SPECULATORS_GIT_URL" "$SPECULATORS_DIR"
  fi
  git -C "$SPECULATORS_DIR" fetch --depth 1 origin "$SPECULATORS_REF"
  git -C "$SPECULATORS_DIR" checkout --detach FETCH_HEAD
fi

if [[ "$APPLY_COMPAT_PATCHES" == "true" || "$APPLY_COMPAT_PATCHES" == "True" ]]; then
  compat_patches=(
    "$SCRIPT_DIR/patches/speculators_transformers_kwargs_compat.patch"
    "$SCRIPT_DIR/patches/speculators_transformers_past_key_value_compat.patch"
    "$SCRIPT_DIR/patches/speculators_torch28_distributed_device_compat.patch"
    "$SCRIPT_DIR/patches/speculators_numpy_int_dataset_index_compat.patch"
    "$SCRIPT_DIR/patches/speculators_dynamic_cache_config_compat.patch"
    "$SCRIPT_DIR/patches/speculators_disable_torch_compile_env_compat.patch"
    "$SCRIPT_DIR/patches/speculators_dflash_disable_torch_compile_compat.patch"
    "$SCRIPT_DIR/patches/speculators_fsdp_layer_wrap_cache_compat.patch"
    "$SCRIPT_DIR/patches/speculators_torch28_scatter_index_dtype_compat.patch"
    "$SCRIPT_DIR/patches/speculators_from_pretrained_no_meta_compat.patch"
    "$SCRIPT_DIR/patches/speculators_datagen_index_range_compat.patch"
  )
  for compat_patch in "${compat_patches[@]}"; do
    if [[ -f "$compat_patch" ]]; then
      patch_cmd=(git -C "$SPECULATORS_DIR" apply "$compat_patch")
      check_patch_cmd=(git -C "$SPECULATORS_DIR" apply --check "$compat_patch")
      printf '%q ' "${patch_cmd[@]}"; printf '\n'
      if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" ]]; then
        if "${check_patch_cmd[@]}" >/dev/null 2>&1; then
          "${patch_cmd[@]}"
        else
          echo "# compat patch already applied or not applicable; skipping: $(basename "$compat_patch")"
        fi
      fi
    fi
  done
fi

if [[ -n "$VLLM_SITE" ]]; then
  export PYTHONPATH="$VLLM_SITE:$SPECULATORS_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SPECULATORS_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
fi

needs_speculators_install=false
if [[ "$RUN_PREPARE" == "true" || "$RUN_PREPARE" == "True" || "$RUN_DATAGEN" == "true" || "$RUN_DATAGEN" == "True" || "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "True" ]]; then
  needs_speculators_install=true
fi
if [[ "$INSTALL_SPECULATORS" == "true" || "$INSTALL_SPECULATORS" == "True" ]] && [[ "$needs_speculators_install" == "true" ]]; then
  install_cmd=(python3 -m pip install --no-deps -e "$SPECULATORS_DIR")
  printf '%q ' "${install_cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]] || "${install_cmd[@]}"
fi

echo "# vLLM Speculators Qwen3-235B offline pipeline"
echo "MODEL=$MODEL"
echo "SPECULATORS_DIR=$SPECULATORS_DIR"
echo "SPECULATORS_REF=$(git -C "$SPECULATORS_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "SOURCE_CONVERSATIONS=$SOURCE_CONVERSATIONS"
echo "SPECULATORS_JSONL=$SPECULATORS_JSONL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HIDDEN_STATES_DIR=$HIDDEN_STATES_DIR"
echo "HIDDEN_STATE_MANIFEST=$HIDDEN_STATE_MANIFEST"
echo "TARGET_LAYER_IDS=$TARGET_LAYER_IDS"
echo "SPECULATOR_TYPE=$SPECULATOR_TYPE"
echo "MAX_SAMPLES=$MAX_SAMPLES SAMPLE_OFFSET=$SAMPLE_OFFSET SEQ_LENGTH=$SEQ_LENGTH"
echo "MIN_HIDDEN_STATES=$MIN_HIDDEN_STATES"
echo "DATAGEN_START_INDEX=$DATAGEN_START_INDEX DATAGEN_END_INDEX=$DATAGEN_END_INDEX"
echo "VLLM_SITE=$VLLM_SITE"
echo "VLLM_LAUNCH_EXTRA_ARGS=$VLLM_LAUNCH_EXTRA_ARGS"
echo "VALIDATE_SOURCE_CONVERSATIONS=$VALIDATE_SOURCE_CONVERSATIONS FAIL_ON_DUPLICATE_PROMPTS=$FAIL_ON_DUPLICATE_PROMPTS"
echo "DENYLIST_PROMPTS_FROM=$DENYLIST_PROMPTS_FROM"
echo "RUN_CLONE=$RUN_CLONE RUN_CONVERT=$RUN_CONVERT RUN_PREPARE=$RUN_PREPARE RUN_DATAGEN=$RUN_DATAGEN RUN_TRAIN=$RUN_TRAIN INSTALL_SPECULATORS=$INSTALL_SPECULATORS APPLY_COMPAT_PATCHES=$APPLY_COMPAT_PATCHES"

if [[ "$VALIDATE_SOURCE_CONVERSATIONS" == "true" || "$VALIDATE_SOURCE_CONVERSATIONS" == "True" ]]; then
  validate_source_cmd=(
    python3 "$SCRIPT_DIR/validate_training_conversations.py"
    "$SOURCE_CONVERSATIONS"
    --max-seq-len "$SEQ_LENGTH"
  )
  if [[ "$FAIL_ON_DUPLICATE_PROMPTS" == "true" || "$FAIL_ON_DUPLICATE_PROMPTS" == "True" ]]; then
    validate_source_cmd+=(--fail-on-duplicate-user-prompts)
  fi
  if [[ -n "$DENYLIST_PROMPTS_FROM" ]]; then
    # shellcheck disable=SC2206
    denylist_prompt_files=($DENYLIST_PROMPTS_FROM)
    for denylist_prompt_file in "${denylist_prompt_files[@]}"; do
      validate_source_cmd+=(--denylist-prompts-from "$denylist_prompt_file")
    done
  fi
  printf '%q ' "${validate_source_cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]] || "${validate_source_cmd[@]}"
fi

if [[ "$RUN_CONVERT" == "true" || "$RUN_CONVERT" == "True" ]]; then
  convert_cmd=(
    python3 "$SCRIPT_DIR/convert_conversations_to_speculators_jsonl.py"
    --input "$SOURCE_CONVERSATIONS"
    --output "$SPECULATORS_JSONL"
    --model "$MODEL"
    --seq-length "$SEQ_LENGTH"
    --prepared-output-dir "$OUTPUT_DIR"
    --minimum-valid-tokens "$MINIMUM_VALID_TOKENS"
    --json-out "$ARTIFACT_ROOT/reports/speculators_data_conversion.json"
    --markdown-out "$ARTIFACT_ROOT/reports/speculators_data_conversion.md"
  )
  if [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]]; then
    convert_cmd+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "$SAMPLE_OFFSET" && "$SAMPLE_OFFSET" != "0" ]]; then
    convert_cmd+=(--sample-offset "$SAMPLE_OFFSET")
  fi
  printf '%q ' "${convert_cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]] || "${convert_cmd[@]}"
fi

if [[ "$RUN_PREPARE" == "true" || "$RUN_PREPARE" == "True" ]]; then
  prepare_cmd=(
    python3 "$SPECULATORS_DIR/scripts/prepare_data.py"
    --model "$MODEL"
    --data "$SPECULATORS_JSONL"
    --output "$OUTPUT_DIR"
    --seq-length "$SEQ_LENGTH"
    --minimum-valid-tokens "$MINIMUM_VALID_TOKENS"
    --overwrite
  )
  if [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]]; then
    prepare_cmd+=(--max-samples "$MAX_SAMPLES")
  fi
  printf '%q ' "${prepare_cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]] || "${prepare_cmd[@]}"
  if [[ "$WRITE_PREPARE_MANIFEST" == "true" || "$WRITE_PREPARE_MANIFEST" == "True" ]]; then
    python3 - "$PREPARE_MANIFEST" "$SPECULATORS_JSONL" "$SOURCE_CONVERSATIONS" "$MODEL" "$SEQ_LENGTH" "$TARGET_LAYER_IDS" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


manifest_path = Path(sys.argv[1])
speculators_jsonl = Path(sys.argv[2])
source_conversations = Path(sys.argv[3])
payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "model": sys.argv[4],
    "seq_length": int(sys.argv[5]),
    "target_layer_ids": [int(item) for item in sys.argv[6].split()],
    "speculators_jsonl": str(speculators_jsonl),
    "speculators_jsonl_size": speculators_jsonl.stat().st_size,
    "speculators_jsonl_rows": count_lines(speculators_jsonl),
    "speculators_jsonl_sha256": sha256_file(speculators_jsonl),
    "source_conversations": str(source_conversations),
    "source_conversations_size": source_conversations.stat().st_size if source_conversations.exists() else None,
    "source_conversations_rows": count_lines(source_conversations) if source_conversations.exists() else None,
}
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"# Wrote prepare manifest: {manifest_path}")
PY
    python3 - "$HIDDEN_STATE_MANIFEST" "$PREPARE_MANIFEST" "$MODEL" "$SEQ_LENGTH" "$TARGET_LAYER_IDS" "$MIN_HIDDEN_STATES" "$HIDDEN_STATES_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest_path = Path(sys.argv[1])
prepare_manifest_path = Path(sys.argv[2])
prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "created_by": "speculators_qwen3_235b_offline_pipeline.sh",
    "model": sys.argv[3],
    "seq_length": int(sys.argv[4]),
    "target_layer_ids": [int(item) for item in sys.argv[5].split()],
    "expected_hidden_states": int(sys.argv[6]) if sys.argv[6] else 0,
    "hidden_states_dir": sys.argv[7],
    "prepare_manifest": str(prepare_manifest_path),
    "prepare_manifest_sha256": sha256_file(prepare_manifest_path),
    "speculators_jsonl": prepare_manifest.get("speculators_jsonl"),
    "speculators_jsonl_sha256": prepare_manifest.get("speculators_jsonl_sha256"),
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"# Wrote hidden-state manifest: {manifest_path}")
PY
  fi
fi

check_connector_cmd=(
  python3 -c
  'import pathlib, vllm; root=pathlib.Path(vllm.__file__).parent; text="\n".join(p.read_text(errors="ignore") for p in root.rglob("*.py") if p.is_file()); raise SystemExit(0 if "ExampleHiddenStatesConnector" in text and "extract_hidden_states" in text else "vLLM install lacks ExampleHiddenStatesConnector/extract_hidden_states support")'
)

patch_vllm_ray_runtime_env_cmd=(
  python3 - "$VLLM_SITE"
)

if [[ "$RUN_DATAGEN" == "true" || "$RUN_DATAGEN" == "True" ]]; then
  echo "# Checking vLLM hidden-state extraction support"
  printf '%q ' "${check_connector_cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]] || "${check_connector_cmd[@]}"

  if [[ -n "$VLLM_SITE" ]]; then
    echo "# Patching vLLM Ray worker runtime_env for extracted-site PYTHONPATH"
    printf '%q ' "${patch_vllm_ray_runtime_env_cmd[@]}"; printf ' <<PY\n'
  elif [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
    echo "# dry-run: VLLM_SITE is empty; skipping extracted-site runtime_env patch"
  fi
  if [[ -n "$VLLM_SITE" && "$DRY_RUN" != "true" && "$DRY_RUN" != "True" ]]; then
    "${patch_vllm_ray_runtime_env_cmd[@]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

vllm_site = Path(sys.argv[1])
ray_executor = vllm_site / "vllm/v1/executor/ray_executor.py"
if not ray_executor.exists():
    raise SystemExit(f"vLLM Ray executor is missing: {ray_executor}")
text = ray_executor.read_text(encoding="utf-8")
if "ray_worker_runtime_env" in text:
    print(f"# vLLM Ray runtime_env patch already present: {ray_executor}")
else:
    old = '        self._init_workers_ray(placement_group, runtime_env={"py_executable": "/opt/venv/bin/python"})'
    new = '''        ray_worker_runtime_env = {"py_executable": "/opt/venv/bin/python"}
        ray_worker_env_vars = {}
        for _name in ("PYTHONPATH", "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"):
            if os.environ.get(_name):
                ray_worker_env_vars[_name] = os.environ[_name]
        if ray_worker_env_vars:
            ray_worker_runtime_env["env_vars"] = ray_worker_env_vars
        self._init_workers_ray(placement_group, runtime_env=ray_worker_runtime_env)'''
    if old not in text:
        raise SystemExit(
            "Could not patch vLLM Ray executor runtime_env; expected anchor not found"
        )
    ray_executor.write_text(text.replace(old, new), encoding="utf-8")
    print(f"# Patched vLLM Ray runtime_env: {ray_executor}")
PY
  fi

  vllm_args=(--tensor-parallel-size "$VLLM_TP" --gpu-memory-utilization "$VLLM_GPU_UTIL" --max-model-len "$VLLM_MAX_MODEL_LEN" --port "$VLLM_PORT")
  if [[ "$VLLM_DP" != "1" ]]; then
    vllm_args+=(--data-parallel-size "$VLLM_DP")
  fi
  if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=($VLLM_EXTRA_ARGS)
    vllm_args+=("${extra_args[@]}")
  fi

  launch_cmd=(
    python3 "$SPECULATORS_DIR/scripts/launch_vllm.py"
    "$MODEL"
    --hidden-states-path "$VLLM_TMP_HIDDEN_STATES"
  )
  if [[ -n "$VLLM_LAUNCH_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    launch_extra_args=($VLLM_LAUNCH_EXTRA_ARGS)
    launch_cmd+=("${launch_extra_args[@]}")
  fi
  launch_cmd+=(--target-layer-ids)
  # shellcheck disable=SC2206
  layer_args=($TARGET_LAYER_IDS)
  launch_cmd+=("${layer_args[@]}" -- "${vllm_args[@]}")

  echo "# Launching vLLM hidden-state server"
  printf '%q ' "${launch_cmd[@]}"; printf '\n'
  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
    echo "# dry-run: skipping vLLM launch/datagen"
  else
    "${launch_cmd[@]}" &
    VLLM_PID=$!
    cleanup() {
      kill "$VLLM_PID" 2>/dev/null || true
      wait "$VLLM_PID" 2>/dev/null || true
    }
    trap cleanup EXIT

    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
      sleep 5
    done

    datagen_cmd=(
      python3 "$SPECULATORS_DIR/scripts/data_generation_offline.py"
      --endpoint "http://localhost:${VLLM_PORT}/v1"
      --preprocessed-data "$OUTPUT_DIR"
      --output "$HIDDEN_STATES_DIR"
      --concurrency "$CONCURRENCY"
      --request-timeout "$REQUEST_TIMEOUT"
      --max-retries "$MAX_RETRIES"
    )
    if [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]]; then
      datagen_cmd+=(--max-samples "$MAX_SAMPLES")
    fi
    if [[ -n "$DATAGEN_START_INDEX" ]]; then
      datagen_cmd+=(--start-index "$DATAGEN_START_INDEX")
    fi
    if [[ -n "$DATAGEN_END_INDEX" ]]; then
      datagen_cmd+=(--end-index "$DATAGEN_END_INDEX")
    fi
    if [[ "$VALIDATE_OUTPUTS" == "true" || "$VALIDATE_OUTPUTS" == "True" ]]; then
      datagen_cmd+=(--validate-outputs)
    fi
    printf '%q ' "${datagen_cmd[@]}"; printf '\n'
    "${datagen_cmd[@]}"
    cleanup
    trap - EXIT
  fi
fi

if [[ "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "True" ]]; then
  if [[ "$MIN_HIDDEN_STATES" != "0" && -n "$MIN_HIDDEN_STATES" ]]; then
    python3 - "$HIDDEN_STATE_MANIFEST" "$PREPARE_MANIFEST" "$MODEL" "$SEQ_LENGTH" "$TARGET_LAYER_IDS" "$MIN_HIDDEN_STATES" "$HIDDEN_STATES_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest_path = Path(sys.argv[1])
prepare_manifest_path = Path(sys.argv[2])
if not prepare_manifest_path.exists():
    raise SystemExit(f"ERROR: prepare manifest is missing: {prepare_manifest_path}")
if not manifest_path.exists():
    raise SystemExit(f"ERROR: hidden-state manifest is missing: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "model": sys.argv[3],
    "seq_length": int(sys.argv[4]),
    "target_layer_ids": [int(item) for item in sys.argv[5].split()],
    "expected_hidden_states": int(sys.argv[6]),
    "hidden_states_dir": sys.argv[7],
    "prepare_manifest": str(prepare_manifest_path),
    "prepare_manifest_sha256": sha256_file(prepare_manifest_path),
}
mismatches = [
    f"{key}: actual={manifest.get(key)!r} expected={value!r}"
    for key, value in expected.items()
    if manifest.get(key) != value
]
if mismatches:
    raise SystemExit(
        "ERROR: hidden-state manifest does not match current prepared data: "
        + "; ".join(mismatches)
    )
print(f"# Hidden-state manifest validated: {manifest_path}")
PY
    hidden_state_count="$(find "$HIDDEN_STATES_DIR" -type f -name 'hs_*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
    echo "# Hidden-state files available: $hidden_state_count; required: $MIN_HIDDEN_STATES"
    if (( hidden_state_count < MIN_HIDDEN_STATES )); then
      echo "ERROR: not enough hidden-state files for training yet" >&2
      exit 2
    fi
    if [[ "$VALIDATE_HIDDEN_STATE_COVERAGE" == "true" || "$VALIDATE_HIDDEN_STATE_COVERAGE" == "True" ]]; then
      python3 - "$HIDDEN_STATES_DIR" "$MIN_HIDDEN_STATES" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
pattern = re.compile(r"^hs_(\d+)\.safetensors$")
indices: set[int] = set()
bad_names: list[str] = []
empty_files: list[str] = []
duplicate_numeric_names: list[str] = []

for path in root.glob("hs_*.safetensors"):
    match = pattern.match(path.name)
    if not match:
        bad_names.append(path.name)
        continue
    idx = int(match.group(1))
    if idx in indices:
        duplicate_numeric_names.append(path.name)
    indices.add(idx)
    if path.stat().st_size <= 0:
        empty_files.append(path.name)

missing = [idx for idx in range(expected) if idx not in indices]
extras = sorted(idx for idx in indices if idx >= expected)
problems: list[str] = []
if bad_names:
    problems.append(f"malformed hidden-state filenames: {bad_names[:10]}")
if duplicate_numeric_names:
    problems.append(f"duplicate numeric hidden-state filenames: {duplicate_numeric_names[:10]}")
if empty_files:
    problems.append(f"empty hidden-state files: {empty_files[:10]}")
if missing:
    problems.append(f"missing hidden-state indices count={len(missing)} first={missing[:20]}")
if extras:
    problems.append(f"unexpected hidden-state indices >= {expected}: count={len(extras)} first={extras[:20]}")
if problems:
    raise SystemExit("ERROR: hidden-state coverage validation failed: " + "; ".join(problems))
print(f"# Hidden-state coverage validated: 0..{expected - 1}")
PY
    fi
  fi

  train_cmd=(
    torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS"
    "$SPECULATORS_DIR/scripts/train.py"
    --verifier-name-or-path "$MODEL"
    --data-path "$OUTPUT_DIR"
    --hidden-states-path "$HIDDEN_STATES_DIR"
    --save-path "$CHECKPOINT_DIR"
    --draft-vocab-size "$DRAFT_VOCAB_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --total-seq-len "$SEQ_LENGTH"
    --speculator-type "$SPECULATOR_TYPE"
    --draft-arch "$DRAFT_ARCH"
    --num-layers "$NUM_LAYERS"
    --ttt-steps "$TTT_STEPS"
    --ttt-step-loss-decay "$TTT_STEP_LOSS_DECAY"
    --block-size "$BLOCK_SIZE"
    --max-anchors "$MAX_ANCHORS"
    --target-layer-ids
  )
  # shellcheck disable=SC2206
  train_layer_args=($TARGET_LAYER_IDS)
  train_cmd+=("${train_layer_args[@]}" --on-missing raise --checkpoint-freq 1 --save-best)
  if [[ -n "$MASK_TOKEN_ID" ]]; then
    train_cmd+=(--mask-token-id "$MASK_TOKEN_ID")
  fi
  if [[ -n "$DRAFT_HIDDEN_ACT" ]]; then
    train_cmd+=(--draft-hidden-act "$DRAFT_HIDDEN_ACT")
  fi
  if [[ -n "$FROM_PRETRAINED" ]]; then
    train_cmd+=(--from-pretrained "$FROM_PRETRAINED")
  fi
  printf '%q ' "${train_cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]] || "${train_cmd[@]}"
fi

echo "# Speculators pipeline complete"
