#!/usr/bin/env bash
set -euo pipefail

# Offline Eagle3 draft training for Qwen3-235B using the current local
# Model-Optimizer recipe interface. Run from the repository root:
#
#   HIDDEN_STATES_DIR=/path/to/hidden_states \
#   OUTPUT_DIR=/path/to/output \
#   bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELOPT_DIR="${MODELOPT_DIR:-$ROOT_DIR/Model-Optimizer}"
SPECDEC_DIR="$MODELOPT_DIR/examples/speculative_decoding"
DRY_RUN="${DRY_RUN:-false}"
ARCH_ENV_FILE="${ARCH_ENV_FILE:-}"

if [[ -n "$ARCH_ENV_FILE" ]]; then
  if [[ ! -f "$ARCH_ENV_FILE" ]]; then
    echo "ARCH_ENV_FILE does not exist: $ARCH_ENV_FILE" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$ARCH_ENV_FILE"
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:?set HIDDEN_STATES_DIR to the offline hidden-state dump directory}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3_235b_thinking_eagle3_modelopt}"

TRAINING_SEQ_LEN="${TRAINING_SEQ_LEN:-16384}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
SAVE_STEPS="${SAVE_STEPS:-512}"
EAGLE_TTT_STEPS="${EAGLE_TTT_STEPS:-3}"
EAGLE_LOSS_DECAY_FACTOR="${EAGLE_LOSS_DECAY_FACTOR:-0.9}"
EAGLE_USE_TORCH_COMPILE="${EAGLE_USE_TORCH_COMPILE:-false}"
EAGLE_DECODER_TYPE="${EAGLE_DECODER_TYPE:-llama}"
AUX_LAYERS="${EAGLE_TRAIN_AUX_LAYERS:-${AUX_LAYERS:-[1,46,90]}}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
USE_FAKE_BASE_FOR_OFFLINE="${USE_FAKE_BASE_FOR_OFFLINE:-true}"
MAX_STEPS="${MAX_STEPS:-}"
DATA_SAMPLE_SIZE="${DATA_SAMPLE_SIZE:-}"

# Qwen3-235B-A22B Eagle3 draft architecture defaults. These mirror the public
# NVIDIA Qwen3-235B Eagle3 draft shape where applicable, while using the
# Thinking-2507 verifier's rope_theta by default.
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-12288}"
HEAD_DIM="${HEAD_DIM:-128}"
RMS_NORM_EPS="${RMS_NORM_EPS:-1e-6}"
ROPE_THETA="${ROPE_THETA:-5000000}"
USE_LAST_LAYERNORM="${USE_LAST_LAYERNORM:-true}"
USE_INPUT_LAYERNORM_IN_FIRST_LAYER="${USE_INPUT_LAYERNORM_IN_FIRST_LAYER:-true}"
USE_AUX_HIDDEN_STATE="${USE_AUX_HIDDEN_STATE:-true}"
DISABLE_EXPORT_ROPE_SCALING="${DISABLE_EXPORT_ROPE_SCALING:-true}"

NUM_NODES="${NUM_NODES:-1}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" && ! -x "$SPECDEC_DIR/launch_train.sh" ]]; then
  echo "Missing ModelOpt launcher: $SPECDEC_DIR/launch_train.sh" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "True" ]]; then
  mkdir -p "$OUTPUT_DIR"
  cd "$SPECDEC_DIR"
fi

cmd=(
  ./launch_train.sh
  --config ../../modelopt_recipes/general/speculative_decoding/eagle3.yaml
  --num_nodes "$NUM_NODES"
  model.model_name_or_path="$BASE_MODEL"
  model.trust_remote_code="$TRUST_REMOTE_CODE"
  model.use_fake_base_for_offline="$USE_FAKE_BASE_FOR_OFFLINE"
  data.offline_data_path="$HIDDEN_STATES_DIR"
  training.output_dir="$OUTPUT_DIR"
  training.training_seq_len="$TRAINING_SEQ_LEN"
  training.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE"
  training.num_train_epochs="$NUM_TRAIN_EPOCHS"
  training.learning_rate="$LEARNING_RATE"
  training.save_steps="$SAVE_STEPS"
  training.answer_only_loss="$ANSWER_ONLY_LOSS"
  eagle.eagle_decoder_type="$EAGLE_DECODER_TYPE"
  eagle.eagle_ttt_steps="$EAGLE_TTT_STEPS"
  eagle.eagle_loss_decay_factor="$EAGLE_LOSS_DECAY_FACTOR"
  eagle.eagle_use_torch_compile="$EAGLE_USE_TORCH_COMPILE"
  eagle.eagle_freeze_base_model=true
  eagle.eagle_architecture_config.num_attention_heads="$NUM_ATTENTION_HEADS"
  eagle.eagle_architecture_config.num_key_value_heads="$NUM_KEY_VALUE_HEADS"
  eagle.eagle_architecture_config.intermediate_size="$INTERMEDIATE_SIZE"
  eagle.eagle_architecture_config.head_dim="$HEAD_DIM"
  eagle.eagle_architecture_config.rms_norm_eps="$RMS_NORM_EPS"
  eagle.eagle_architecture_config.rope_theta="$ROPE_THETA"
  eagle.eagle_architecture_config.rope_scaling.rope_type=default
  eagle.eagle_architecture_config.rope_scaling.rope_theta="$ROPE_THETA"
  eagle.eagle_architecture_config.use_aux_hidden_state="$USE_AUX_HIDDEN_STATE"
  eagle.eagle_architecture_config.use_input_layernorm_in_first_layer="$USE_INPUT_LAYERNORM_IN_FIRST_LAYER"
  eagle.eagle_architecture_config.use_last_layernorm="$USE_LAST_LAYERNORM"
  "eagle.eagle_architecture_config.eagle_aux_hidden_state_layer_ids=$AUX_LAYERS"
)

if [[ -n "$DATA_SAMPLE_SIZE" ]]; then
  cmd+=(data.sample_size="$DATA_SAMPLE_SIZE")
fi

if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(training.max_steps="$MAX_STEPS")
fi

if [[ "$DISABLE_EXPORT_ROPE_SCALING" == "true" || "$DISABLE_EXPORT_ROPE_SCALING" == "True" ]]; then
  cmd+=("eagle.eagle_export_rope_scaling={}")
else
  cmd+=("eagle.eagle_export_rope_scaling.original_max_position_embeddings=$TRAINING_SEQ_LEN")
fi

if [[ "$NUM_NODES" != "1" ]]; then
  if [[ -z "$HEAD_NODE_IP" ]]; then
    echo "HEAD_NODE_IP is required when NUM_NODES != 1" >&2
    exit 1
  fi
  cmd+=(--head_node_ip "$HEAD_NODE_IP")
fi

printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  echo "# run from: $SPECDEC_DIR"
  exit 0
fi
exec "${cmd[@]}"
