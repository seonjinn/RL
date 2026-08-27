#!/bin/bash

# ===== BEGIN CONFIG =====
NUM_NODES=3
GPUS_PER_NODE=4
SEGMENT_SIZE=1
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=1
NUM_MINUTES=240
JOB_REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"disproportionate_resource_requirement","description":"Async GRPO has long GPU-idle phases during Ray init and model loading"}}'
# ===== END CONFIG =====

CONFIG_REL=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml
NANO_MODEL_PATH=${NANO_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16}
MODEL_OVERRIDES=(
  "policy.model_name=${NANO_MODEL_PATH}"
  cluster.num_nodes=3
  cluster.gpus_per_node=4
  cluster.segment_size=1
  policy.generation.colocated.enabled=false
  policy.generation.colocated.resources.num_nodes=1
  policy.generation.colocated.resources.gpus_per_node=4
  policy.generation.vllm_cfg.tensor_parallel_size=1
  policy.generation.vllm_cfg.pipeline_parallel_size=1
  policy.generation.vllm_cfg.expert_parallel_size=1
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5
  policy.generation.vllm_cfg.use_tqdm=false
  policy.generation.vllm_cfg.precision=fp8
  +policy.generation.vllm_cfg.is_mx=true
  +policy.generation.vllm_cfg.quantization_ignore_patterns=[conv1d,mtp,in_proj,out_proj,q_proj,k_proj,v_proj,o_proj,fc1_latent_proj,fc2_latent_proj,shared_experts]
  +policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm
  policy.megatron_cfg.tensor_model_parallel_size=1
  policy.megatron_cfg.pipeline_model_parallel_size=1
  policy.megatron_cfg.context_parallel_size=1
  policy.megatron_cfg.expert_model_parallel_size=8
  policy.megatron_cfg.sequence_parallel=false
  policy.train_global_batch_size=16
  loss_fn.force_on_policy_ratio=false
  loss_fn.use_importance_sampling_correction=true
)

source "$(dirname -- "${BASH_SOURCE[0]}")/pr3865_refit_ab_common.sh"
