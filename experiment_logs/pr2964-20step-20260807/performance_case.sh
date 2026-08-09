#!/bin/bash

render_case() {
  local model=$1
  local dispatcher=$2
  local run_root=$3
  local max_num_steps=${MAX_NUM_STEPS_OVERRIDE:-20}

  local config
  local wandb_name
  local -a model_overrides=()
  case "${model}" in
    qwen3-30ba3b)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
      num_nodes=4
      segment_size=4
      model_overrides+=(logger.tensorboard_enabled=true)
      ;;
    qwen3-235b)
      config=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml
      num_nodes=16
      segment_size=
      model_overrides+=(
        ++policy.generation.vllm_kwargs.disable_custom_all_reduce=true
        ++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false
      )
      ;;
    nemotron3-super)
      config=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n8g.yaml
      num_nodes=32
      segment_size=
      model_overrides+=(
        ++policy.generation.vllm_kwargs.disable_custom_all_reduce=true
        ++policy.generation.vllm_kwargs.moe_backend=triton
        policy.generation.vllm_cfg.enforce_eager=true
        logger.tensorboard_enabled=true
      )
      ;;
    *)
      printf 'Unsupported model: %s\n' "${model}" >&2
      return 2
      ;;
  esac

  local -a dispatcher_overrides=()
  case "${dispatcher}" in
    baseline)
      dispatcher_overrides+=(policy.megatron_cfg.moe_token_dispatcher_type=alltoall)
      ;;
    hybridep)
      dispatcher_overrides+=(
        policy.megatron_cfg.moe_token_dispatcher_type=flex
        ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
        ++policy.megatron_cfg.moe_hybridep_num_sms=32
        "++policy.megatron_cfg.env_vars.NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN='8'"
        "++policy.megatron_cfg.env_vars.NUM_OF_TOKENS_PER_CHUNK_COMBINE_API='128'"
        "++policy.megatron_cfg.env_vars.NVLINK_DOMAIN_SIZE='8'"
        "++policy.megatron_cfg.env_vars.USE_MNNVL='0'"
      )
      ;;
    *)
      printf 'Unsupported dispatcher: %s\n' "${dispatcher}" >&2
      return 2
      ;;
  esac
  if [[ "${dispatcher}" == "hybridep" && "${model}" != "qwen3-235b" ]]; then
    dispatcher_overrides+=(
      ++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true
    )
  fi

  local checkpointing_enabled=${CHECKPOINTING_ENABLED_OVERRIDE:-false}
  local -a checkpoint_overrides=()
  case "${checkpointing_enabled}" in
    false)
      checkpoint_overrides+=(checkpointing.enabled=false)
      ;;
    true)
      : "${CHECKPOINT_DIR_OVERRIDE:?CHECKPOINT_DIR_OVERRIDE is required when checkpointing is enabled}"
      : "${CHECKPOINT_SAVE_PERIOD_OVERRIDE:?CHECKPOINT_SAVE_PERIOD_OVERRIDE is required when checkpointing is enabled}"
      : "${CHECKPOINT_MUST_SAVE_BY_OVERRIDE:?CHECKPOINT_MUST_SAVE_BY_OVERRIDE is required when checkpointing is enabled}"
      : "${CHECKPOINT_METRIC_NAME_OVERRIDE:?CHECKPOINT_METRIC_NAME_OVERRIDE is required when checkpointing is enabled}"
      : "${CHECKPOINT_KEEP_TOP_K_OVERRIDE:?CHECKPOINT_KEEP_TOP_K_OVERRIDE is required when checkpointing is enabled}"
      : "${CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE:?CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE is required when checkpointing is enabled}"
      : "${CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE:?CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE is required when checkpointing is enabled}"
      checkpoint_overrides+=(
        checkpointing.enabled=true
        "checkpointing.checkpoint_dir=${CHECKPOINT_DIR_OVERRIDE}"
        "checkpointing.save_period=${CHECKPOINT_SAVE_PERIOD_OVERRIDE}"
        "checkpointing.checkpoint_must_save_by=${CHECKPOINT_MUST_SAVE_BY_OVERRIDE}"
        "checkpointing.metric_name=${CHECKPOINT_METRIC_NAME_OVERRIDE}"
        "checkpointing.keep_top_k=${CHECKPOINT_KEEP_TOP_K_OVERRIDE}"
        "++checkpointing.ft_keep_latest_k=${CHECKPOINT_FT_KEEP_LATEST_K_OVERRIDE}"
        "checkpointing.save_optimizer=${CHECKPOINT_SAVE_OPTIMIZER_OVERRIDE}"
      )
      ;;
    *)
      printf 'CHECKPOINTING_ENABLED_OVERRIDE must be true or false: %s\n' "${checkpointing_enabled}" >&2
      return 2
      ;;
  esac

  wandb_name=$(basename "${run_root}")
  driver_args=(
    /opt/nemo_rl_venv/bin/python
    examples/run_grpo.py
    --config "${config}"
    "grpo.max_num_steps=${max_num_steps}"
    "cluster.num_nodes=${num_nodes}"
    cluster.gpus_per_node=8
  )
  if [[ -n "${segment_size}" ]]; then
    driver_args+=("cluster.segment_size=${segment_size}")
  fi
  local override
  for override in "${model_overrides[@]-}"; do
    if [[ -n "${override}" ]]; then
      driver_args+=("${override}")
    fi
  done
  for override in "${dispatcher_overrides[@]}"; do
    driver_args+=("${override}")
  done
  for override in "${checkpoint_overrides[@]}"; do
    driver_args+=("${override}")
  done
  driver_args+=(
    "logger.log_dir=${run_root}/training"
    logger.wandb_enabled=true
    logger.wandb.project=sna-hybridep-b200
    "logger.wandb.name=${wandb_name}"
    logger.monitor_gpus=true
  )
}
