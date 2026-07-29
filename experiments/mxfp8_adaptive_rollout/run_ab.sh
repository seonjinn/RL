#!/bin/bash

set -euo pipefail

: "${NEMO_RL_REPO_ROOT:?Set NEMO_RL_REPO_ROOT to the canonical shared checkout}"
if [[ "$NEMO_RL_REPO_ROOT" != /* || ! -d "$NEMO_RL_REPO_ROOT" ]]; then
  echo "NEMO_RL_REPO_ROOT must be an existing absolute directory" >&2
  exit 2
fi
REPO_ROOT=$(cd "$NEMO_RL_REPO_ROOT" && pwd -P)
if [[ "$REPO_ROOT" != "$NEMO_RL_REPO_ROOT" || "$REPO_ROOT" == *","* ]]; then
  echo "NEMO_RL_REPO_ROOT must be canonical and contain no comma" >&2
  exit 2
fi
export NEMO_RL_REPO_ROOT="$REPO_ROOT"
EXPERIMENT_DIR="$REPO_ROOT/experiments/mxfp8_adaptive_rollout"
LAUNCHER="$EXPERIMENT_DIR/run_ab.sh"
PARSER="$EXPERIMENT_DIR/parse_results.py"
PERFORMANCE_CONFIG="$EXPERIMENT_DIR/configs/grpo_qwen3_30ba3b_4n4g.yaml"
TRACE_CONFIG_REL="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-adaptive-trace.yaml"
PROFILE_DEFAULT="$EXPERIMENT_DIR/cluster/oci-hsg.env"

VLLM_REPOSITORY="https://github.com/seonjinn/vllm.git"
VLLM_COMMIT="73da4b2c233a45eced4def252e90dbe2079194db"
VLLM_BASE_COMMIT="5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1"
VLLM_VERSION="0.20.2"
FLASHINFER_VERSION="0.6.8.post1"
MODEL="Qwen/Qwen3-30B-A3B"
TP_SIZE=1
BOOTSTRAP_CONFIG_NAME="qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json"
QUALIFIED_CONFIG_NAME="qwen3_30ba3b_tp1_v0202_qualified.json"
NUM_SAMPLES=2048
SEED=42

usage() {
  echo "usage: $0 <trace|shmoo|original|adaptive|ab|parse> [profile]" >&2
  echo "  ACTION=test-only (default) or ACTION=submit" >&2
  echo "  Override the default account with SLURM_ACCOUNT=nemotron_sw_pre." >&2
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 2
  fi
}

require_new_path() {
  if [[ -e "$1" ]]; then
    echo "refusing to overwrite existing output: $1" >&2
    exit 2
  fi
}

write_lines_new() {
  local output=$1
  shift
  require_new_path "$output"
  (
    set -o noclobber
    printf '%s\n' "$@" >"$output"
  )
}

require_container_visible_path() {
  local path=$1
  local mount
  local source
  local destination
  local -a mounts

  IFS=',' read -r -a mounts <<<"$CONTAINER_MOUNTS"
  for mount in "${mounts[@]}"; do
    source=${mount%%:*}
    destination=${mount#*:}
    destination=${destination%%:*}
    if [[ "$source" == "$destination" &&
          ( "$path" == "$destination" || "$path" == "$destination/"* ) ]]; then
      return 0
    fi
  done
  echo "path is not visible at the same absolute location in the container: $path" >&2
  exit 2
}

load_profile() {
  local profile=$1
  require_file "$profile"
  # shellcheck disable=SC1090
  source "$profile"
  : "${SLURM_ACCOUNT:?profile must set SLURM_ACCOUNT}"
  : "${PARTITION:?profile must set PARTITION}"
  : "${QOS:?profile must set QOS}"
  : "${NUM_NODES:?profile must set NUM_NODES}"
  : "${GPUS_PER_NODE:?profile must set GPUS_PER_NODE}"
  : "${SLURM_SWITCHES:?profile must set SLURM_SWITCHES}"
  : "${NEMO_RL_EXPERIMENT_ROOT:?profile must set NEMO_RL_EXPERIMENT_ROOT}"
  : "${CONTAINER_IMAGE:?set CONTAINER_IMAGE to an immutable staged .sqsh}"
  : "${CONTAINER_MOUNTS:?profile must set CONTAINER_MOUNTS}"
  if [[ "$NEMO_RL_EXPERIMENT_ROOT" != /* ||
        "$NEMO_RL_EXPERIMENT_ROOT" == *"/../"* ||
        "$NEMO_RL_EXPERIMENT_ROOT" == *","* ]]; then
    echo "NEMO_RL_EXPERIMENT_ROOT must be an absolute normalized path without comma" >&2
    exit 2
  fi
  export NEMO_RL_EXPERIMENT_ROOT
  EXPERIMENT_ROOT="$NEMO_RL_EXPERIMENT_ROOT"
  require_container_visible_path "$REPO_ROOT"
  require_container_visible_path "$EXPERIMENT_ROOT"
  if [[ "$NUM_NODES" != "4" || "$GPUS_PER_NODE" != "4" ]]; then
    echo "Qwen 4n4g requires NUM_NODES=4 and GPUS_PER_NODE=4" >&2
    exit 2
  fi
}

check_container() {
  require_file "$CONTAINER_IMAGE"
  if [[ -L "$CONTAINER_IMAGE" ]]; then
    echo "use the immutable .sqsh path, not a convenience symlink" >&2
    exit 2
  fi
  case "$CONTAINER_IMAGE" in
    *.sqsh) ;;
    *)
      echo "CONTAINER_IMAGE must name a .sqsh file" >&2
      exit 2
      ;;
  esac
}

check_submission_checkout() {
  git -C "$REPO_ROOT" -c fetch.recurseSubmodules=false \
    pull --ff-only --recurse-submodules=no
  git -C "$REPO_ROOT" submodule update --init --recursive
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]]; then
    echo "tracked checkout must be clean before submission" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$REPO_ROOT" rev-list '@{upstream}..HEAD')" ]]; then
    echo "HEAD must be pushed to its upstream before submission" >&2
    exit 2
  fi
}

submit_job() {
  local mode=$1
  local run_id=$2
  local repeat=$3
  local dependency=${4:-}
  local action=${ACTION:-test-only}
  local -a args
  local output

  args=(
    --account="$SLURM_ACCOUNT"
    --partition="$PARTITION"
    --qos="$QOS"
    --nodes="$NUM_NODES"
    --ntasks-per-node=1
    --gres="gpu:$GPUS_PER_NODE"
    --segment="$NUM_NODES"
    --switches="$SLURM_SWITCHES"
    --time="$WALLTIME"
    --job-name="mxfp8-${mode}-${repeat}"
    --output="$EXPERIMENT_ROOT/slurm/%x-%j.out"
    --export="ALL,MODE=$mode,RUN_ID=$run_id,REPEAT=$repeat,PROFILE_PATH=$PROFILE_PATH,NEMO_RL_REPO_ROOT=$REPO_ROOT,NEMO_RL_EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi

  case "$action" in
    test-only)
      args+=(--test-only)
      output=$(sbatch "${args[@]}" "$LAUNCHER")
      echo "$output" >&2
      ;;
    submit)
      output=$(sbatch --parsable "${args[@]}" "$LAUNCHER")
      output=${output%%;*}
      echo "submitted mode=$mode repeat=$repeat job=$output" >&2
      echo "$output"
      ;;
    *)
      echo "ACTION must be test-only or submit" >&2
      exit 2
      ;;
  esac
}

submit_suite() {
  local mode=$1
  local repeats=${REPEATS:-3}
  local suite_id=${SUITE_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
  local suite_root="$EXPERIMENT_ROOT/runs/$suite_id"
  local submission_manifest="$EXPERIMENT_ROOT/submissions/${suite_id}_submission.env"
  local dependency=""
  local job_id=""
  local repeat
  local schedule=$mode

  if (( repeats < 3 )); then
    echo "REPEATS must be at least 3" >&2
    exit 2
  fi
  if [[ ! "$suite_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "SUITE_ID is not filesystem-safe: $suite_id" >&2
    exit 2
  fi
  if [[ -f "$EXPERIMENT_ROOT/not-applicable.json" && "$mode" == "ab" ]]; then
    echo "Qwen performance skipped: $EXPERIMENT_ROOT/not-applicable.json"
    return 0
  fi
  mkdir -p "$EXPERIMENT_ROOT/slurm" "$EXPERIMENT_ROOT/submissions" \
    "$EXPERIMENT_ROOT/runs"
  if ! mkdir "$suite_root"; then
    echo "suite output already exists; choose a new SUITE_ID: $suite_root" >&2
    exit 2
  fi

  if [[ "$mode" != "ab" ]]; then
    submit_job "$mode" "$suite_id/measured-${mode}-r${REPEAT:-1}" \
      "${REPEAT:-1}"
  else
    schedule="(original,adaptive)x${repeats};in-job-warmup-discarded"
    for ((repeat = 1; repeat <= repeats; repeat++)); do
      job_id=$(submit_job original \
        "$suite_id/measured-original-r${repeat}" "$repeat" "$dependency")
      [[ -n "$job_id" ]] && dependency=$job_id
      job_id=$(submit_job adaptive \
        "$suite_id/measured-adaptive-r${repeat}" "$repeat" "$dependency")
      [[ -n "$job_id" ]] && dependency=$job_id
    done
  fi

  write_lines_new "$submission_manifest" \
    "suite_id=$suite_id" \
    "action=${ACTION:-test-only}" \
    "account=$SLURM_ACCOUNT" \
    "repeats=$repeats" \
    "schedule=$schedule" \
    "nemo_rl_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    "vllm_commit=$VLLM_COMMIT" \
    "container=$CONTAINER_IMAGE"
}

run_in_allocation() {
  local run_dir="$EXPERIMENT_ROOT/runs/$RUN_ID"
  local configured_log_dir="$run_dir/configured_logs"
  local runtime_trace_dir="$run_dir/runtime_dispatch"

  check_container
  if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_ID must contain one filesystem-safe suite/run pair: $RUN_ID" >&2
    exit 2
  fi
  if [[ "$MODE" == "original" || "$MODE" == "adaptive" ]]; then
    unset VLLM_MXFP8_DENSE_CONFIG_FILE
    if [[ -n "${VLLM_MXFP8_DENSE_SHAPE_TRACE:-}" ||
          -n "${VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR:-}" ||
          -n "${VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS:-}" ||
          -n "${VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4:-}" ]]; then
      echo "performance A/B environment contains a forbidden MXFP8 override" >&2
      exit 2
    fi
  fi
  export CONTAINER_SHA256
  CONTAINER_SHA256=$(sha256sum "$CONTAINER_IMAGE" | awk '{print $1}')
  export CONTAINER="$CONTAINER_IMAGE"
  export GPUS_PER_NODE
  export HF_HOME
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
  export XDG_CACHE_HOME="/tmp/mxfp8_adaptive_${SLURM_JOB_ID}"
  export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
  export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
  export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
  export FLASHINFER_CACHE_DIR="$XDG_CACHE_HOME/flashinfer"
  export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
  export BASE_LOG_DIR="$EXPERIMENT_ROOT/slurm"
  export NVTE_CUDA_ARCHS=100
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
  export NRL_FORCE_REBUILD_VENVS=false

  mkdir -p "$EXPERIMENT_ROOT/slurm" "$CACHE_ROOT/uv"
  if ! mkdir "$run_dir"; then
    echo "run output already exists; refusing rerun: $RUN_ID" >&2
    exit 2
  fi
  if [[ "$MODE" == "original" || "$MODE" == "adaptive" ]]; then
    mkdir "$configured_log_dir" "$runtime_trace_dir"
    export MOUNTS="$CONTAINER_MOUNTS,$configured_log_dir:/mxfp8_configured_logs,$runtime_trace_dir:/mxfp8_runtime_trace"
  else
    export MOUNTS="$CONTAINER_MOUNTS"
  fi
  printf -v COMMAND 'cd %q && bash %q __container %q %q %q' \
    "$REPO_ROOT" "$LAUNCHER" "$MODE" "$RUN_ID" "$REPEAT"
  export COMMAND
  exec bash "$REPO_ROOT/ray.sub"
}

runtime_preflight() {
  local python_bin=$1
  local vllm_root
  local actual_vllm_commit
  local actual_versions

  vllm_root=$(
    "$python_bin" -c \
      'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parents[1])'
  )
  actual_vllm_commit=$(git -C "$vllm_root" rev-parse HEAD)
  if [[ "$actual_vllm_commit" != "$VLLM_COMMIT" ]]; then
    echo "expected vLLM commit $VLLM_COMMIT, got $actual_vllm_commit" >&2
    exit 2
  fi
  actual_versions=$(
    "$python_bin" -c \
      'import flashinfer, vllm; print(f"{vllm.__version__} {flashinfer.__version__}")'
  )
  if [[ "$actual_versions" != "$VLLM_VERSION $FLASHINFER_VERSION" ]]; then
    echo "expected vLLM/FlashInfer $VLLM_VERSION $FLASHINFER_VERSION, got $actual_versions" >&2
    exit 2
  fi
  printf '%s\n' "$vllm_root"
}

resolve_config() {
  local python_bin=$1
  local config=$2
  local output=$3
  shift 3
  local -a command=(
    "$python_bin" "$PARSER" resolve-config
    --config "$config"
    --output "$output"
  )
  local override
  local resolved_sha256
  for override in "$@"; do
    command+=(--override "$override")
  done
  require_new_path "$output"
  require_new_path "$output.sha256"
  resolved_sha256=$("${command[@]}")
  write_lines_new "$output.sha256" "$resolved_sha256"
}

make_metadata() {
  local python_bin=$1
  local arm=$2
  local repeat=$3
  local config_hash=$4
  local resolved_config=$5
  local output=$6
  local warmup_steps=${7:-0}

  "$python_bin" "$PARSER" make-metadata \
    --arm "$arm" \
    --repeat "$repeat" \
    --nemo-rl-commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    --vllm-commit "$VLLM_COMMIT" \
    --container-digest "sha256:$CONTAINER_SHA256" \
    --config-hash "$config_hash" \
    --checkpoint "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --seed "$SEED" \
    --num-nodes "$NUM_NODES" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --num-samples "$NUM_SAMPLES" \
    --warmup-steps "$warmup_steps" \
    --resolved-config "$resolved_config" \
    --output "$output"
}

record_not_applicable() {
  local python_bin=$1
  local reason=$2
  local run_dir=$3

  require_new_path "$run_dir/not-applicable.json"
  require_new_path "$EXPERIMENT_ROOT/not-applicable.json"
  "$python_bin" "$PARSER" not-applicable \
    --reason "$reason" \
    --output "$run_dir/not-applicable.json"
  "$python_bin" "$PARSER" not-applicable \
    --reason "$reason" \
    --output "$EXPERIMENT_ROOT/not-applicable.json"
  echo "not-applicable: $reason"
  echo "fallback: Nemotron 3 Ultra TP4"
}

run_trace() {
  local python_bin=$1
  local vllm_root=$2
  local run_dir=$3
  local trace_config="$REPO_ROOT/$TRACE_CONFIG_REL"
  local trace_dir="$EXPERIMENT_ROOT/artifacts/qwen_trace"
  local inventory="$EXPERIMENT_ROOT/artifacts/qwen3_tp1_inventory.json"
  local bootstrap_path
  local actual_bootstrap_sha256
  local inventory_error="$run_dir/inventory.stderr"
  local -a trace_files=()
  local -a trace_args=()
  local trace_file
  local -a overrides

  : "${BOOTSTRAP_CONFIG_SHA256:?Set BOOTSTRAP_CONFIG_SHA256 from the built image}"
  require_file "$trace_config"
  bootstrap_path="$vllm_root/vllm/model_executor/kernels/linear/mxfp8/tactic_configs/$BOOTSTRAP_CONFIG_NAME"
  require_file "$bootstrap_path"
  actual_bootstrap_sha256=$(sha256sum "$bootstrap_path" | awk '{print $1}')
  if [[ "$actual_bootstrap_sha256" != "$BOOTSTRAP_CONFIG_SHA256" ]]; then
    echo "bootstrap config SHA256 does not match the built package" >&2
    exit 2
  fi
  mkdir -p "$EXPERIMENT_ROOT/artifacts"
  if ! mkdir "$trace_dir"; then
    echo "trace output already exists; refusing rerun: $trace_dir" >&2
    exit 2
  fi
  require_new_path "$inventory"
  require_new_path "$inventory.sha256"
  require_new_path "$inventory_error"
  require_new_path "$run_dir/run.log"

  overrides=(
    "grpo.max_num_steps=${TRACE_STEPS:-1}"
    "grpo.seed=$SEED"
    "checkpointing.enabled=false"
    "logger.log_dir=$run_dir/configured_logs"
    "logger.wandb.name=mxfp8-qwen-trace"
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_CONFIG_FILE=$BOOTSTRAP_CONFIG_NAME"
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_SHAPE_TRACE=1"
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR=$trace_dir"
    "policy.generation.vllm_cfg.enforce_eager=true"
  )
  export NEMO_RL_MXFP8_TRACE_DIR="$trace_dir"
  resolve_config "$python_bin" "$trace_config" "$run_dir/resolved_config.json" \
    "${overrides[@]}"
  make_metadata "$python_bin" trace 1 "$BOOTSTRAP_CONFIG_SHA256" \
    "$run_dir/resolved_config.json" "$run_dir/metadata.json"

  "$python_bin" "$REPO_ROOT/examples/run_grpo.py" \
    --config "$trace_config" "${overrides[@]}" 2>&1 | tee "$run_dir/run.log"

  while IFS= read -r -d '' trace_file; do
    trace_files+=("$trace_file")
  done < <(
    find "$trace_dir" -maxdepth 1 -type f \
      \( -name 'adaptive_dispatch_*_*.jsonl' \
         -o -name 'dense_shapes_*_*.jsonl' \) \
      -print0 | sort -z
  )
  if (( ${#trace_files[@]} == 0 )); then
    record_not_applicable "$python_bin" \
      "no dense MXFP8 trace files were emitted" "$run_dir"
    return 0
  fi
  for trace_file in "${trace_files[@]}"; do
    trace_args+=(--trace "$trace_file")
  done
  set +e
  "$python_bin" "$vllm_root/tools/mxfp8/offline_shmoo.py" inventory \
    --bootstrap-manifest "$bootstrap_path" \
    "${trace_args[@]}" \
    --output "$inventory" 2>"$inventory_error"
  local inventory_status=$?
  set -e
  if (( inventory_status != 0 )); then
    if grep -Fq "zero eligible dense MXFP8 trace records" "$inventory_error"; then
      record_not_applicable "$python_bin" \
        "trace files contain zero eligible dense MXFP8 records" "$run_dir"
      return 0
    fi
    cat "$inventory_error" >&2
    return "$inventory_status"
  fi
  write_lines_new "$inventory.sha256" "$(sha256sum "$inventory")"
  echo "inventory=$inventory"
}

run_shmoo() {
  local python_bin=$1
  local vllm_root=$2
  local run_dir=$3
  local repeat_count=${SHMOO_REPEATS:-3}
  local inventory="$EXPERIMENT_ROOT/artifacts/qwen3_tp1_inventory.json"
  local observations="$EXPERIMENT_ROOT/artifacts/qwen3_tp1_observations.jsonl"
  local qualified="$EXPERIMENT_ROOT/artifacts/$QUALIFIED_CONFIG_NAME"
  local qualification="$EXPERIMENT_ROOT/artifacts/qwen3_tp1_qualification.json"

  if (( repeat_count < 3 )); then
    echo "SHMOO_REPEATS must be at least 3" >&2
    exit 2
  fi
  if [[ -f "$EXPERIMENT_ROOT/not-applicable.json" ]]; then
    echo "Qwen shmoo skipped: $EXPERIMENT_ROOT/not-applicable.json"
    return 0
  fi
  require_file "$inventory"
  require_new_path "$observations"
  require_new_path "$qualified"
  require_new_path "$qualification"
  require_new_path "$qualified.sha256"
  require_new_path "$run_dir/resolved_config.json"
  require_new_path "$run_dir/metadata.json"
  require_new_path "$run_dir/shmoo.log"

  write_lines_new "$run_dir/resolved_config.json" \
    '{' \
    "  \"base_seed\": ${SHMOO_BASE_SEED:-1234}," \
    "  \"iterations\": ${SHMOO_ITERATIONS:-80}," \
    "  \"repeat_count\": $repeat_count," \
    "  \"warmup\": ${SHMOO_WARMUP:-10}," \
    "  \"workspace_mb\": ${SHMOO_WORKSPACE_MB:-256}" \
    '}'
  make_metadata "$python_bin" shmoo 1 "$(sha256sum "$inventory" | awk '{print $1}')" \
    "$run_dir/resolved_config.json" "$run_dir/metadata.json"

  "$python_bin" "$vllm_root/tools/mxfp8/offline_shmoo.py" shmoo \
    --inventory "$inventory" \
    --output "$observations" \
    --repeat-count "$repeat_count" \
    --base-seed "${SHMOO_BASE_SEED:-1234}" \
    --warmup "${SHMOO_WARMUP:-10}" \
    --iterations "${SHMOO_ITERATIONS:-80}" \
    --workspace-mb "${SHMOO_WORKSPACE_MB:-256}" \
    --minimum-cosine-similarity 0.999 \
    --vllm-version "$VLLM_VERSION" \
    --flashinfer-version "$FLASHINFER_VERSION" \
    --container-sha256 "$CONTAINER_SHA256" 2>&1 | tee "$run_dir/shmoo.log"

  "$python_bin" "$vllm_root/tools/mxfp8/offline_shmoo.py" promote \
    --inventory "$inventory" \
    --observations "$observations" \
    --output "$qualified" \
    --qualification-output "$qualification" \
    --repeat-count "$repeat_count" \
    --minimum-cosine-similarity 0.999 \
    --minimum-speedup-vs-default 1.02 \
    --qualification-scope nemo_rl_qwen3_30ba3b_mxfp8_rollout \
    --vllm-version "$VLLM_VERSION" \
    --vllm-base-commit "$VLLM_BASE_COMMIT" \
    --flashinfer-version "$FLASHINFER_VERSION" \
    --compute-capability 10.0 \
    --gpu-family GB200 \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE"

  "$python_bin" "$vllm_root/tools/mxfp8/offline_shmoo.py" validate \
    --manifest "$qualified" \
    --inventory "$inventory" \
    --observations "$observations" \
    --repeat-count "$repeat_count" \
    --minimum-cosine-similarity 0.999 \
    --minimum-speedup-vs-default 1.02 \
    --qualification-scope nemo_rl_qwen3_30ba3b_mxfp8_rollout \
    --vllm-version "$VLLM_VERSION" \
    --flashinfer-version "$FLASHINFER_VERSION" \
    --compute-capability 10.0 \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --check
  "$python_bin" "$PARSER" validate-qualified --manifest "$qualified"
  write_lines_new "$qualified.sha256" "$(sha256sum "$qualified")"
}

run_performance_arm() {
  local python_bin=$1
  local vllm_root=$2
  local arm=$3
  local run_id=$4
  local repeat=$5
  local run_dir="$EXPERIMENT_ROOT/runs/$run_id"
  local runtime_trace_dir="$run_dir/runtime_dispatch"
  local warmup_steps=${WARMUP_STEPS:-1}
  local measured_steps=${MEASURE_STEPS:-3}
  local total_steps
  local config_hash="none"
  local qualified_path
  local actual_qualified_sha256
  local pair_dir
  local pair_name
  local peer_metadata
  local run_start_ns
  local coverage_output="$run_dir/tactic_coverage.json"
  local trace_file
  local -a runtime_trace_files=()
  local -a runtime_trace_args=()
  local -a overrides

  if [[ -f "$EXPERIMENT_ROOT/not-applicable.json" ]]; then
    echo "Qwen performance skipped: $EXPERIMENT_ROOT/not-applicable.json"
    return 0
  fi
  if [[ ! "$warmup_steps" =~ ^[0-9]+$ || "$warmup_steps" -lt 1 ||
        ! "$measured_steps" =~ ^[0-9]+$ || "$measured_steps" -lt 1 ]]; then
    echo "WARMUP_STEPS and MEASURE_STEPS must be positive integers" >&2
    exit 2
  fi
  total_steps=$((warmup_steps + measured_steps))
  pair_dir=$(dirname "$run_dir")
  pair_name="measured-r${repeat}"
  overrides=(
    "grpo.max_num_steps=$total_steps"
    "grpo.seed=$SEED"
    "checkpointing.enabled=false"
    "logger.log_dir=/mxfp8_configured_logs"
    "logger.wandb.name=mxfp8-qwen-$pair_name"
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_SHAPE_TRACE=1"
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR=/mxfp8_runtime_trace"
  )

  case "$arm" in
    original)
      unset VLLM_MXFP8_DENSE_CONFIG_FILE
      peer_metadata="$pair_dir/measured-adaptive-r${repeat}/metadata.json"
      ;;
    adaptive)
      : "${QUALIFIED_CONFIG_SHA256:?Set QUALIFIED_CONFIG_SHA256 from the rebuilt image}"
      qualified_path="$vllm_root/vllm/model_executor/kernels/linear/mxfp8/tactic_configs/$QUALIFIED_CONFIG_NAME"
      require_file "$qualified_path"
      actual_qualified_sha256=$(sha256sum "$qualified_path" | awk '{print $1}')
      if [[ "$actual_qualified_sha256" != "$QUALIFIED_CONFIG_SHA256" ]]; then
        echo "qualified config SHA256 does not match the built package" >&2
        exit 2
      fi
      "$python_bin" "$PARSER" validate-qualified --manifest "$qualified_path"
      export VLLM_MXFP8_DENSE_CONFIG_FILE="$QUALIFIED_CONFIG_NAME"
      config_hash="$QUALIFIED_CONFIG_SHA256"
      overrides+=(
        "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_CONFIG_FILE=$QUALIFIED_CONFIG_NAME"
      )
      peer_metadata="$pair_dir/measured-original-r${repeat}/metadata.json"
      ;;
    *)
      echo "invalid performance arm: $arm" >&2
      exit 2
      ;;
  esac

  if [[ -n "${VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS:-}" ||
        -n "${VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4:-}" ]]; then
    echo "legacy inline tactic variables are forbidden in this A/B" >&2
    exit 2
  fi
  if [[ ! -d "$run_dir" ]]; then
    echo "reserved run directory is missing: $run_dir" >&2
    exit 2
  fi
  resolve_config "$python_bin" "$PERFORMANCE_CONFIG" \
    "$run_dir/resolved_config.json" "${overrides[@]}"
  make_metadata "$python_bin" "$arm" "$repeat" "$config_hash" \
    "$run_dir/resolved_config.json" "$run_dir/metadata.json" "$warmup_steps"

  if [[ -f "$peer_metadata" ]]; then
    if [[ "$arm" == "original" ]]; then
      "$python_bin" "$PARSER" validate-pair \
        --original "$run_dir/metadata.json" \
        --adaptive "$peer_metadata" \
        --expected-config-file "$QUALIFIED_CONFIG_NAME" \
        --expected-config-sha256 "$QUALIFIED_CONFIG_SHA256"
    else
      "$python_bin" "$PARSER" validate-pair \
        --original "$peer_metadata" \
        --adaptive "$run_dir/metadata.json" \
        --expected-config-file "$QUALIFIED_CONFIG_NAME" \
        --expected-config-sha256 "$QUALIFIED_CONFIG_SHA256"
    fi
  elif [[ "$arm" == "adaptive" ]]; then
    echo "adaptive arm requires matched original metadata: $peer_metadata" >&2
    exit 2
  fi

  require_new_path "$run_dir/run.log"
  run_start_ns=$(
    "$python_bin" -c 'import time; print(time.monotonic_ns())'
  )
  {
    "$python_bin" "$PARSER" emit-context \
      --metadata "$run_dir/metadata.json"
    "$python_bin" "$REPO_ROOT/examples/run_grpo.py" \
      --config "$PERFORMANCE_CONFIG" "${overrides[@]}"
    "$python_bin" -c \
      'import sys, time; print(f"MXFP8_RUN_WALL_TIME_S {(time.monotonic_ns() - int(sys.argv[1])) / 1e9:.9f}")' \
      "$run_start_ns"
  } 2>&1 | tee "$run_dir/run.log"

  if [[ "$arm" == "adaptive" ]]; then
    while IFS= read -r -d '' trace_file; do
      runtime_trace_files+=("$trace_file")
    done < <(
      find "$runtime_trace_dir" -type f \
        -name 'adaptive_dispatch_*_*.jsonl' -print0 | sort -z
    )
    if (( ${#runtime_trace_files[@]} == 0 )); then
      echo "adaptive runtime produced no tactic trace files" >&2
      exit 2
    fi
    for trace_file in "${runtime_trace_files[@]}"; do
      runtime_trace_args+=(--trace "$trace_file")
    done
    require_new_path "$coverage_output"
    "$python_bin" "$PARSER" validate-runtime \
      --manifest "$qualified_path" \
      "${runtime_trace_args[@]}" \
      --expected-config-sha256 "$QUALIFIED_CONFIG_SHA256" \
      --output "$coverage_output" | tee -a "$run_dir/run.log"
  fi
}

run_in_container() {
  local mode=$1
  local run_id=$2
  local repeat=$3
  local python_bin=${PYTHON_BIN:-/opt/nemo_rl_venv/bin/python}
  local run_dir="$EXPERIMENT_ROOT/runs/$run_id"
  local vllm_root

  require_file "$python_bin"
  require_file "$PARSER"
  require_file "$PERFORMANCE_CONFIG"
  if [[ ! -d "$run_dir" ]]; then
    echo "reserved run directory is missing: $run_dir" >&2
    exit 2
  fi
  if [[ "$mode" == "original" || "$mode" == "adaptive" ]]; then
    if [[ ! -d "$run_dir/configured_logs" ||
          ! -d "$run_dir/runtime_dispatch" ||
          -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 \
            ! -name configured_logs ! -name runtime_dispatch -print -quit)" ||
          -n "$(find "$run_dir/configured_logs" "$run_dir/runtime_dispatch" \
            -mindepth 1 -print -quit)" ]]; then
      echo "reserved performance run directory is not pristine: $run_dir" >&2
      exit 2
    fi
  elif find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "reserved run directory is not empty: $run_dir" >&2
    exit 2
  fi
  mkdir -p "$EXPERIMENT_ROOT/artifacts"
  vllm_root=$(runtime_preflight "$python_bin")
  write_lines_new "$run_dir/runtime.env" \
    "vllm_repository=$VLLM_REPOSITORY" \
    "vllm_commit=$VLLM_COMMIT" \
    "vllm_version=$VLLM_VERSION" \
    "flashinfer_version=$FLASHINFER_VERSION" \
    "container_sha256=$CONTAINER_SHA256" \
    "slurm_job_id=${SLURM_JOB_ID:-unknown}"

  case "$mode" in
    trace)
      run_trace "$python_bin" "$vllm_root" "$run_dir"
      ;;
    shmoo)
      run_shmoo "$python_bin" "$vllm_root" "$run_dir"
      ;;
    original | adaptive)
      run_performance_arm \
        "$python_bin" "$vllm_root" "$mode" "$run_id" "$repeat"
      ;;
    *)
      echo "unsupported container mode: $mode" >&2
      exit 2
      ;;
  esac
}

if [[ "${1:-}" == "__container" ]]; then
  shift
  EXPERIMENT_ROOT=${NEMO_RL_EXPERIMENT_ROOT:?Set NEMO_RL_EXPERIMENT_ROOT}
  run_in_container "$@"
  exit 0
fi

MODE=${1:-${MODE:-}}
PROFILE_PATH=${2:-${PROFILE_PATH:-$PROFILE_DEFAULT}}
if [[ -z "$MODE" ]]; then
  usage
  exit 2
fi
if [[ "$MODE" == "parse" ]]; then
  shift
  exec python3 "$PARSER" "$@"
fi
export PROFILE_PATH
load_profile "$PROFILE_PATH"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : "${RUN_ID:?RUN_ID is required in an allocation}"
  : "${REPEAT:?REPEAT is required in an allocation}"
  run_in_allocation
  exit 0
fi

case "$MODE" in
  trace | shmoo | original | adaptive | ab)
    check_container
    check_submission_checkout
    submit_suite "$MODE"
    ;;
  *)
    usage
    exit 2
    ;;
esac
