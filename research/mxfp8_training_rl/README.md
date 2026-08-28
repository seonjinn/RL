# MXFP8 training in RL

This experiment validates MXFP8 training compute with BF16 parameter storage
(`fp8_param: false`) and MXFP8 vLLM rollout. It covers Qwen3-30B-A3B and
Nemotron-3 Nano on GB200.

Run a two-step smoke test first:

```bash
MODEL=qwen30 MAX_STEPS=2 ACTION=test-only ./submit_oci_hsg.sh
MODEL=qwen30 MAX_STEPS=2 ACTION=submit ./submit_oci_hsg.sh

MODEL=nano MAX_STEPS=2 ACTION=test-only ./submit_oci_hsg.sh
MODEL=nano MAX_STEPS=2 ACTION=submit ./submit_oci_hsg.sh
```

After both jobs complete, use `MAX_STEPS=20` for performance measurements. The
steady-state summary should average steps 2 through 19.

## GB200 smoke results

Both two-step runs completed with exit code 0 on commit `55ab424a`. The branch
includes the Qwen/Nano MXFP8 MoE padding fix from PR #3630 and the current heads
of PRs #3653 and #3654.

| Model | Job | Step 2 loss | Step 2 generation KL error | Step 2 average reward | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B | `6605779` | 0.0040 | 0.0065 | 0.4990 | [W&B](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/6imeq68l) |
| Nemotron-3 Nano | `6605880` | 0.0000 | 0.0184 | 0.0000 | [W&B](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/317z7a10) |

These runs validate the combined MXFP8 training, MXFP8 rollout, refit, and
training-step path with `fp8_param: false`. They are not performance results:
the first run on each node built Transformer Engine from source, only two steps
were measured, and the container did not provide the native
`nccl.m2n.reshard` operation, so refit used the exact Python fallback. The
launcher now defaults `NVTE_CUDA_ARCHS=100` to avoid compiling unrelated GPU
architectures in later GB200 runs.

The launcher requires `REPO`, `CONTAINER`, `HF_HOME`, `WANDB_HOME`,
`RESULT_ROOT`, and `SLURM_ACCOUNT`. It stores source in `/home`, worker virtual
environments and JIT caches in `/raid/scratch`, and durable logs in `/lustre`.
The node-local environment and compile caches are keyed by the source commit,
so jobs at the same commit reuse them instead of creating per-run copies. Set
`NRL_FORCE_REBUILD_VENVS=true` only when a clean environment is required.

## Per-module Transformer Engine precision

NeMo-RL loads a Megatron per-module recipe from
`policy.megatron_cfg.te_precision_config_file`. Megatron matches the full module
path against the YAML matchers in order; the first match wins.

The validated Nano configuration keeps the global Megatron MXFP8 context
enabled, selects routed expert FC1/FC2 first, and then explicitly disables
quantization for every other TE module. The matcher order is part of the
configuration because the first enabled match wins:

```yaml
configs:
  bf16:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe:
      override_quantized_autocast: true
  mxfp8:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe:
      fp8_quantization_recipe: mxfp8
      override_quantized_autocast: true

matchers:
  routed_experts_fc1_mxfp8:
    config: mxfp8
    type: glob
    pattern: "*mlp.experts.linear_fc1"
    enabled: true
  routed_experts_fc2_mxfp8:
    config: mxfp8
    type: glob
    pattern: "*mlp.experts.linear_fc2"
    enabled: true
  all_other_modules_bf16:
    config: bf16
    type: glob
    pattern: "*"
    enabled: true
```

The Nano recipe also sets `first_last_layers_bf16: true`, keeps zero layers at
the start in BF16, and keeps the final eight layers in BF16. Therefore the
effective training scope is routed expert FC1/FC2 in MXFP8, except for the
final eight transformer layers. Parameter storage remains BF16 because
`fp8_param: false`.

This final configuration completed a two-step GB200 smoke test with exit code
0. Step 2 completed policy training, both policy/reference logprob passes, and
NCCL Reshard refit. The steady-state refit time was 1.03 seconds.

| Model | Job | Commit | Steps | Result |
| --- | --- | --- | ---: | --- |
| Nemotron-3 Nano | `6608045` | `64ca8034` | 2 | [W&B](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/78ve8sq4) |

## Active 20-step measurements

All performance summaries must use steps 2 through 19. The following jobs use
MXFP8 rollout with FlashInfer TRTLLM, CUDA Graph, and NCCL Reshard refit.

| Training precision | Model | Job | W&B |
| --- | --- | --- | --- |
| MXFP8 global context | Qwen3-30B-A3B | `6607378` | [run](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/3momykqd) |
| MXFP8 routed experts | Nemotron-3 Nano | `6609465` | [run](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/ki8fetv6) |
| BF16 | Qwen3-30B-A3B | `6609413` | [run](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/ya7t7pwg) |
| BF16 | Nemotron-3 Nano | `6609265` | [run](https://wandb.ai/nvidia/nemo-rl-mxfp8-training/runs/6zylbrsl) |

## BF16 rollout training-precision A/B

This comparison keeps rollout in BF16 and changes only training compute
precision. Both Qwen Async jobs use FlashInfer TRTLLM, CUDA Graph, and NCCL
Reshard refit. The Nano pair uses the same disaggregated synchronous recipe and
also uses NCCL Reshard refit.

| Model | Training | Rollout | Job |
| --- | --- | --- | --- |
| Qwen3-30B-A3B | BF16 | BF16 | `6610546` |
| Qwen3-30B-A3B | MXFP8, `fp8_param: false` | BF16 | `6610549` |
| Nemotron-3 Nano | BF16 | BF16 | `6610273` |
| Nemotron-3 Nano | MXFP8 routed experts, `fp8_param: false` | BF16 | `6610274` |

The first Qwen submissions (`6610265` and `6610266`) were canceled during
setup because their inherited config forced the on-policy ratio to one. The
replacement jobs above explicitly set `force_on_policy_ratio: false`, so they
compute both policy and reference logprobs like the MXFP8 rollout controls.

Average steps 2 through 19 and report E2E, generation, policy training,
policy/reference logprob, and refit time. Use the logged E2E and generation
tokens/s/GPU metrics rather than reconstructing throughput from averaged time.

This recipe controls TE compute precision. It does not enable MXFP8 parameter
storage. `fp8_param: true` changes parameter and all-gather storage and requires
a separate refit path for native MXFP8 data and E8M0 scales.

## `fp8_param` modes

| Setting | Compute | Parameter communication and storage | Current NeMo-RL status |
| --- | --- | --- | --- |
| `false` | Selected TE GEMMs use MXFP8 | Keep high-precision model parameters and derive MXFP8 compute tensors as needed; parameter all-gather remains high precision | Used by the Qwen and Nano smoke tests in this experiment |
| `true` | Selected TE GEMMs use MXFP8 | Keep the TE compute parameter in MXFP8 and all-gather it in FP8; the optimizer still needs its high-precision state | Requires native MXFP8 source, scale, optimizer-buffer, checkpoint, and vLLM refit support |

`fp8_param: true` does not make the complete optimizer state FP8. Its main
benefit is reducing compute-parameter storage and parameter all-gather traffic.
Its cost is a stricter dependency on Transformer Engine, distributed optimizer,
checkpoint, and refit layouts. Per-module BF16/MXFP8 compute selection is
supported with `fp8_param: false`; per-module mixed parameter storage is not yet
validated for this NeMo-RL rollout path.
