#!/usr/bin/env bash
set -euo pipefail

# Export a trained ModelOpt Eagle3 checkpoint and convert it to vLLM's
# one-checkpoint Eagle3 format.
#
# Required:
#   TRAINED_CKPT=/path/to/modelopt/checkpoint
#   EXPORT_DIR=/path/to/exported_hf_draft
#   VLLM_DRAFT_DIR=/path/to/vllm_draft
#   VERIFIER_CONFIG_DIR=/path/to/local/Qwen3-235B-A22B-Thinking-2507-snapshot
#
# Note: ModelOpt's converter opens "$VERIFIER_CONFIG_DIR/config.json" directly,
# so pass a local verifier snapshot/config directory rather than a bare HF id.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
SPECDEC_DIR="$MODELOPT_DIR/examples/speculative_decoding"
COMPARE_SCRIPT="$ROOT_DIR/experiments/eagle3_qwen3_235b/compare_eagle3_configs.py"
EXPORT_VALIDATE_SCRIPT="$ROOT_DIR/experiments/eagle3_qwen3_235b/validate_eagle3_export_artifacts.py"
TRAINING_CKPT_VALIDATE_SCRIPT="$ROOT_DIR/experiments/eagle3_qwen3_235b/validate_eagle3_training_checkpoint.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-false}"

TRAINED_CKPT="${TRAINED_CKPT:?set TRAINED_CKPT to the trained ModelOpt checkpoint}"
EXPORT_DIR="${EXPORT_DIR:?set EXPORT_DIR to the exported draft checkpoint directory}"
VLLM_DRAFT_DIR="${VLLM_DRAFT_DIR:?set VLLM_DRAFT_DIR to the vLLM draft output directory}"
VERIFIER_CONFIG_DIR="${VERIFIER_CONFIG_DIR:?set VERIFIER_CONFIG_DIR to a local verifier config directory}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
REFERENCE_ARCH="${REFERENCE_ARCH:-$ROOT_DIR/experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json}"
BASE_MODEL="${BASE_MODEL:-}"
RUN_CONFIG_COMPARE="${RUN_CONFIG_COMPARE:-true}"
RUN_TRAINING_CKPT_VALIDATION="${RUN_TRAINING_CKPT_VALIDATION:-true}"
RUN_EXPORT_ARTIFACT_VALIDATION="${RUN_EXPORT_ARTIFACT_VALIDATION:-true}"
EXPORT_CONFIG_COMPARE_JSON="${EXPORT_CONFIG_COMPARE_JSON:-}"
VLLM_CONFIG_COMPARE_JSON="${VLLM_CONFIG_COMPARE_JSON:-}"
if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
  DEFAULT_TRAINING_CKPT_VALIDATION_JSON="$ARTIFACT_ROOT/reports/eagle3_training_checkpoint.json"
  DEFAULT_TRAINING_CKPT_VALIDATION_MARKDOWN="$ARTIFACT_ROOT/reports/eagle3_training_checkpoint.md"
else
  DEFAULT_TRAINING_CKPT_VALIDATION_JSON="$TRAINED_CKPT/training_checkpoint_validation.json"
  DEFAULT_TRAINING_CKPT_VALIDATION_MARKDOWN="$TRAINED_CKPT/training_checkpoint_validation.md"
fi
TRAINING_CKPT_VALIDATION_JSON="${TRAINING_CKPT_VALIDATION_JSON:-$DEFAULT_TRAINING_CKPT_VALIDATION_JSON}"
TRAINING_CKPT_VALIDATION_MARKDOWN="${TRAINING_CKPT_VALIDATION_MARKDOWN:-$DEFAULT_TRAINING_CKPT_VALIDATION_MARKDOWN}"
EXPORT_ARTIFACTS_JSON="${EXPORT_ARTIFACTS_JSON:-}"
EXPORT_ARTIFACTS_MARKDOWN="${EXPORT_ARTIFACTS_MARKDOWN:-}"

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -f "$SPECDEC_DIR/scripts/export_hf_checkpoint.py" ]]; then
  echo "Missing export script under $SPECDEC_DIR/scripts" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -f "$SPECDEC_DIR/scripts/convert_to_vllm_ckpt.py" ]]; then
  echo "Missing vLLM conversion script under $SPECDEC_DIR/scripts" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -f "$TRAINING_CKPT_VALIDATE_SCRIPT" ]]; then
  echo "Missing training checkpoint validator: $TRAINING_CKPT_VALIDATE_SCRIPT" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -e "$TRAINED_CKPT" ]]; then
  echo "TRAINED_CKPT is not visible: $TRAINED_CKPT" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -f "$VERIFIER_CONFIG_DIR/config.json" ]]; then
  echo "VERIFIER_CONFIG_DIR must contain config.json: $VERIFIER_CONFIG_DIR" >&2
  exit 1
fi

needs_reference=false
if [[ "$RUN_CONFIG_COMPARE" != "false" && "$RUN_CONFIG_COMPARE" != "False" ]]; then
  needs_reference=true
fi
if [[ "$RUN_EXPORT_ARTIFACT_VALIDATION" != "false" && "$RUN_EXPORT_ARTIFACT_VALIDATION" != "False" ]]; then
  needs_reference=true
fi
if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && "$needs_reference" == "true" && ! -f "$REFERENCE_ARCH" ]]; then
  echo "REFERENCE_ARCH is not visible: $REFERENCE_ARCH" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" ]]; then
  mkdir -p "$EXPORT_DIR" "$VLLM_DRAFT_DIR"
  cd "$SPECDEC_DIR"
fi

validate_training_cmd=(
  "$PYTHON_BIN" "$TRAINING_CKPT_VALIDATE_SCRIPT"
  --checkpoint-dir "$TRAINED_CKPT"
  --modelopt-dir "$MODELOPT_DIR"
  --reference-arch "$REFERENCE_ARCH"
  --require-modelopt-state-load
  --fail-on-error
)
if [[ -n "$BASE_MODEL" ]]; then
  validate_training_cmd+=(--expected-base-model "$BASE_MODEL")
fi
if [[ -n "$TRAINING_CKPT_VALIDATION_JSON" ]]; then
  validate_training_cmd+=(--json-out "$TRAINING_CKPT_VALIDATION_JSON")
fi
if [[ -n "$TRAINING_CKPT_VALIDATION_MARKDOWN" ]]; then
  validate_training_cmd+=(--markdown-out "$TRAINING_CKPT_VALIDATION_MARKDOWN")
fi

export_cmd=(
  "$PYTHON_BIN" scripts/export_hf_checkpoint.py
  --model_path "$TRAINED_CKPT"
  --export_path "$EXPORT_DIR"
)

if [[ "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
  export_cmd+=(--trust_remote_code)
fi

if [[ "$RUN_TRAINING_CKPT_VALIDATION" != "false" && "$RUN_TRAINING_CKPT_VALIDATION" != "False" ]]; then
  printf '%q ' "${validate_training_cmd[@]}"
  printf '\n'
fi
printf '%q ' "${export_cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  printf '%q ' "$PYTHON_BIN" scripts/convert_to_vllm_ckpt.py --input "$EXPORT_DIR" --verifier "$VERIFIER_CONFIG_DIR" --output "$VLLM_DRAFT_DIR"
  printf '\n'
  if [[ "$RUN_CONFIG_COMPARE" != "false" && "$RUN_CONFIG_COMPARE" != "False" ]]; then
    printf '%q ' "$PYTHON_BIN" "$COMPARE_SCRIPT" --draft-config "$EXPORT_DIR" --verifier-config "$VERIFIER_CONFIG_DIR" --reference-arch "$REFERENCE_ARCH"
    [[ -n "$EXPORT_CONFIG_COMPARE_JSON" ]] && printf '%q ' --json-out "$EXPORT_CONFIG_COMPARE_JSON"
    printf '\n'
    printf '%q ' "$PYTHON_BIN" "$COMPARE_SCRIPT" --draft-config "$VLLM_DRAFT_DIR" --verifier-config "$VERIFIER_CONFIG_DIR" --reference-arch "$REFERENCE_ARCH"
    [[ -n "$VLLM_CONFIG_COMPARE_JSON" ]] && printf '%q ' --json-out "$VLLM_CONFIG_COMPARE_JSON"
    printf '\n'
  fi
  if [[ "$RUN_EXPORT_ARTIFACT_VALIDATION" != "false" && "$RUN_EXPORT_ARTIFACT_VALIDATION" != "False" ]]; then
    printf '%q ' "$PYTHON_BIN" "$EXPORT_VALIDATE_SCRIPT" --export-dir "$EXPORT_DIR" --vllm-draft-dir "$VLLM_DRAFT_DIR" --verifier-config-dir "$VERIFIER_CONFIG_DIR" --reference-arch "$REFERENCE_ARCH"
    [[ -n "$EXPORT_CONFIG_COMPARE_JSON" ]] && printf '%q ' --export-config-compare-json "$EXPORT_CONFIG_COMPARE_JSON"
    [[ -n "$VLLM_CONFIG_COMPARE_JSON" ]] && printf '%q ' --vllm-config-compare-json "$VLLM_CONFIG_COMPARE_JSON"
    [[ -n "$EXPORT_ARTIFACTS_JSON" ]] && printf '%q ' --json-out "$EXPORT_ARTIFACTS_JSON"
    [[ -n "$EXPORT_ARTIFACTS_MARKDOWN" ]] && printf '%q ' --markdown-out "$EXPORT_ARTIFACTS_MARKDOWN"
    printf '%q ' --fail-on-error
    printf '\n'
  fi
  echo "# run from: $SPECDEC_DIR"
  exit 0
fi
if [[ "$RUN_TRAINING_CKPT_VALIDATION" != "false" && "$RUN_TRAINING_CKPT_VALIDATION" != "False" ]]; then
  "${validate_training_cmd[@]}"
fi
"${export_cmd[@]}"

convert_cmd=(
  "$PYTHON_BIN" scripts/convert_to_vllm_ckpt.py
  --input "$EXPORT_DIR"
  --verifier "$VERIFIER_CONFIG_DIR"
  --output "$VLLM_DRAFT_DIR"
)

printf '%q ' "${convert_cmd[@]}"
printf '\n'
"${convert_cmd[@]}"

if [[ "$RUN_CONFIG_COMPARE" != "false" && "$RUN_CONFIG_COMPARE" != "False" ]]; then
  compare_export_cmd=(
    "$PYTHON_BIN" "$COMPARE_SCRIPT"
    --draft-config "$EXPORT_DIR"
    --verifier-config "$VERIFIER_CONFIG_DIR"
    --reference-arch "$REFERENCE_ARCH"
  )
  if [[ -n "$EXPORT_CONFIG_COMPARE_JSON" ]]; then
    compare_export_cmd+=(--json-out "$EXPORT_CONFIG_COMPARE_JSON")
  fi

  compare_vllm_cmd=(
    "$PYTHON_BIN" "$COMPARE_SCRIPT"
    --draft-config "$VLLM_DRAFT_DIR"
    --verifier-config "$VERIFIER_CONFIG_DIR"
    --reference-arch "$REFERENCE_ARCH"
  )
  if [[ -n "$VLLM_CONFIG_COMPARE_JSON" ]]; then
    compare_vllm_cmd+=(--json-out "$VLLM_CONFIG_COMPARE_JSON")
  fi

  printf '%q ' "${compare_export_cmd[@]}"
  printf '\n'
  "${compare_export_cmd[@]}"

  printf '%q ' "${compare_vllm_cmd[@]}"
  printf '\n'
  "${compare_vllm_cmd[@]}"
fi

if [[ "$RUN_EXPORT_ARTIFACT_VALIDATION" == "false" || "$RUN_EXPORT_ARTIFACT_VALIDATION" == "False" ]]; then
  exit 0
fi

validate_export_cmd=(
  "$PYTHON_BIN" "$EXPORT_VALIDATE_SCRIPT"
  --export-dir "$EXPORT_DIR"
  --vllm-draft-dir "$VLLM_DRAFT_DIR"
  --verifier-config-dir "$VERIFIER_CONFIG_DIR"
  --reference-arch "$REFERENCE_ARCH"
  --fail-on-error
)
if [[ -n "$EXPORT_CONFIG_COMPARE_JSON" ]]; then
  validate_export_cmd+=(--export-config-compare-json "$EXPORT_CONFIG_COMPARE_JSON")
fi
if [[ -n "$VLLM_CONFIG_COMPARE_JSON" ]]; then
  validate_export_cmd+=(--vllm-config-compare-json "$VLLM_CONFIG_COMPARE_JSON")
fi
if [[ -n "$EXPORT_ARTIFACTS_JSON" ]]; then
  validate_export_cmd+=(--json-out "$EXPORT_ARTIFACTS_JSON")
fi
if [[ -n "$EXPORT_ARTIFACTS_MARKDOWN" ]]; then
  validate_export_cmd+=(--markdown-out "$EXPORT_ARTIFACTS_MARKDOWN")
fi

printf '%q ' "${validate_export_cmd[@]}"
printf '\n'
exec "${validate_export_cmd[@]}"
