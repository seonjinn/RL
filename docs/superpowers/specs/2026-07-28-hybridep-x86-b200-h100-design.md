# HybridEP x86 B200 and H100 Validation Design

## Goal

Validate that the latest DeepEP `hybrid-ep` commit works through NeMo-RL and
Megatron-Bridge on x86_64 B200 and H100 clusters, then measure its effect
against the unchanged recipe-native dispatcher.

## Upstream Evidence

The design follows NVIDIA-NeMo/Megatron-Bridge `main` at
`64d8909918e3bd5df96c0041b3015f9a607eac16`:

- `apply_flex_dispatcher_backend` accepts HybridEP on CUDA compute
  capabilities 8, 9, and 10, independent of host CPU architecture.
- the Qwen3-30B-A3B H100 performance recipe uses `flex` plus `hybridep`,
  32 dispatcher SMs, and an eight-rank NVLink domain;
- the B200 Qwen performance recipes use HybridEP and an eight-rank NVLink
  domain.

DeepEP `hybrid-ep` currently resolves to
`f725d29699f5bda9ba789456bb9579af69844685`. Its build accepts SM90 and SM100
through `TORCH_CUDA_ARCH_LIST`.

## Source Isolation

Use the dedicated NeMo-RL branch
`sna/hybridep-x86-b200-h100-20260728` based on the already validated GB200
branch at `4c14b04266a0b3ed8ec6121fae387d77d869bf1d`.

Keep the existing Megatron-Bridge and Megatron-LM pointers unchanged:

- Megatron-Bridge:
  `483749cb773415f7608525838607dcefc62e4307`
- Megatron-LM:
  `4d04e7625c5e84f984a9f01aef58cb006b0aa7ac`

Changing Bridge or Megatron-LM would confound the hardware-portability test.
Use upstream Bridge `main` only as the configuration reference.

## Dependency and Build Contract

Change every Linux x86_64 NeMo-RL DeepEP dependency from
`29d31c095796f3c8ece47ee9cdcc167051bbeed9` to
`f725d29699f5bda9ba789456bb9579af69844685`, then regenerate `uv.lock`.

Build an immutable wheel on each target architecture:

- H100: `TORCH_CUDA_ARCH_LIST=9.0`
- B200: `TORCH_CUDA_ARCH_LIST=10.0`
- both: `HYBRID_EP_MULTINODE=1`

Record the source commit, wheel SHA256, container SHA256, build job ID, and
runtime import paths. Do not treat a successful build as a runtime pass; run
a one-node import and GPU smoke before distributed training.

## Workload and Configuration

Use Qwen3-30B-A3B because it is the smallest existing NeMo-RL MoE performance
workload that exercises expert dispatch and has native eight-GPU x86 recipes.

The full comparison starts from the unchanged
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml`.
The HybridEP arm adds only:

```yaml
policy:
  megatron_cfg:
    moe_token_dispatcher_type: flex
    moe_flex_dispatcher_backend: hybridep
    moe_hybridep_num_sms: 32
    env_vars:
      NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: "8"
      NVLINK_DOMAIN_SIZE: "8"
      USE_MNNVL: "0"
```

The baseline preserves the recipe-native dispatcher. Model, data, batch
sizes, packing, seed, optimizer, parallelism, and rollout settings remain
identical within each cluster.

## Execution Gates

Run each gate independently on GCP-NRT B200 and CW-DFW H100.

1. Build and import gate:
   wheel build, one-node import, visible-GPU, and extension-load checks.
2. Compatibility gate:
   matched baseline and HybridEP jobs with two nodes, eight GPUs per node,
   and three GRPO steps.
3. Performance gate:
   after both compatibility arms pass, matched baseline and HybridEP jobs
   using the native four-node, eight-GPU recipe for 20 steps.

Every submission must run `sbatch --test-only`, use the highest eligible
FairShare account, come from a clean pushed checkout, and be monitored for at
least five minutes.

## Metrics and Correctness

For completed 20-step jobs, compare matched steps 5–20:

- mean and median total step time;
- ratio-of-sums end-to-end tokens/second/GPU;
- policy-training time and tokens/second/GPU;
- policy and reference LogProb time and tokens/second/GPU;
- generation time and tokens/second/GPU;
- reward, generation KL error, validation accuracy, and response length.

Success requires terminal `COMPLETED 0:0`, all intended steps, finite metrics,
the logged `flex` plus `hybridep` runtime configuration, and no fatal CUDA,
NCCL, RDMA, actor-loss, OOM, NaN, or Inf signature. Performance comparisons
are within-cluster only; B200 and H100 absolute numbers are not directly
normalized against each other.

## CW Storage Review

Inspect only the user's project paths. Record top-level size, modification
time, ownership, active-job references, checkpoint/container provenance, and
whether an item has another verified copy. Produce deletion-review candidates
in the report, but do not delete, truncate, move, or overwrite any data
without explicit user approval.

## Reporting

Extend the existing GitLab Pages HybridEP report with:

- one compatibility and performance table per x86 cluster;
- within-cluster baseline-versus-HybridEP graphs;
- exact source, submodule, wheel, container, config, and job provenance;
- bounded failure diagnostics for unsuccessful gates;
- the CW storage review table.

Keep detailed logs out of Git and link to their cluster paths. Preserve the
existing GB200 findings as a separate hardware section.

