# BF16 Training with NVFP4 Rollout Design

## Goal

Support a plain BF16 Megatron policy with real NVFP4 vLLM rollout generation in
W4A16 and W4A4 modes. The trainer must not use ModelOpt fake quantization or
QAT. Refit converts the BF16 policy weights into the deployment representation
owned by the generation backend.

The first validated model is Qwen3-30B-A3B using the existing four-node
performance recipe. W4A16 is implemented and validated first. W4A4 follows
with an explicit activation-scale artifact because BF16 training does not
produce QAT input-quantizer statistics.

## Non-Goals

- Do not change BF16 policy forward, backward, optimizer, or checkpoint state.
- Do not route the policy through `MegatronQuantPolicyWorker`.
- Do not make W4A4 input scales from dummy weights or silently substitute a
  weight-only W4A16 execution path.
- Do not add DTensor-policy, SGLang, TRTLLM, expert-parallel vLLM, or sparse
  delta-weight support in the first implementation.
- Do not optimize convergence or select a production calibration dataset in
  this change. The calibration artifact and its provenance are explicit inputs.

## User-Facing Recipes

Create two recipes beside the existing MXFP8 rollout recipe:

- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.yaml`
- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.yaml`

Both inherit
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` and
override only rollout-specific behavior:

```yaml
loss_fn:
  force_on_policy_ratio: false
  use_importance_sampling_correction: true

policy:
  quant_cfg: null
  generation:
    quant_cfg: examples/modelopt/quant_configs/nvfp4_experts_weightonly.yaml
    real_quant: true
    real_quant_ignore:
      - lm_head
      - '*output_layer*'
      - '*mlp.gate*'
      - '*router*'
      - '*self_attention*'
      - '*self_attn*'
```

The W4A4 recipe uses the same block with
`quant_cfg: examples/modelopt/quant_configs/nvfp4_experts.yaml`.

The W4A4 recipe additionally sets
`generation.real_quant_calibration_path` to a safetensors artifact generated
from the same base checkpoint and recorded calibration setup. Configuration
validation rejects a missing path, a missing tensor, an unexpected tensor, or
an artifact whose model identity does not match the configured policy model.
The artifact is produced by a standalone calibration command; neither rollout
recipe selects `MegatronQuantPolicyWorker`.

`force_on_policy_ratio` is disabled because BF16 policy logprobs and NVFP4
rollout logprobs are not mathematically identical. Real importance-sampling
correction remains enabled for the functional and performance runs.

## Worker Selection

The recipe deliberately selects different existing worker families:

```text
policy.quant_cfg = null
    -> MegatronPolicyWorker

generation.quant_cfg != null and generation.real_quant = true
    -> VllmQuantGenerationWorker
```

The policy worker exports ordinary BF16 Hugging Face weights and metadata. The
generation worker owns NVFP4 quantization, packing, destination layout, and
vLLM's layerwise reload lifecycle. This follows the receiver-conversion model
introduced for BF16-to-MXFP8 rollout rather than the QARL actor-export model.

## Refit Data Flow

### W4A16

```text
BF16 Megatron shard
    -> Hugging Face name and BF16 source metadata
    -> transport and reshard to the destination-local logical weight
    -> block-16 NVFP4 quantization
    -> packed E2M1 weight + E4M3 block scale + global scale
    -> vLLM checkpoint-layout loader
    -> vLLM layerwise post-load conversion
    -> stable runtime storage and CUDA graph references
```

W4A16 activations remain in the native model dtype, so no input-activation
scale is transferred or recalibrated.

The serializer preserves ModelOpt's shared-scale rules. Fused QKV and gated
gate/up families are quantized as groups and share the required global scale;
quantizing each sibling independently is invalid even if every output tensor
has the expected dtype and shape.

### W4A4

W4A4 uses the same weight conversion and adds one calibrated input scale per
quantized projection. Input `amax` values are loaded from a versioned
safetensors artifact and converted through Megatron-Bridge's canonical
`compute_nvfp4_input_scale()` helper. The resulting scales remain fixed during
the run. The artifact records:

- source model identifier and checkpoint revision;
- quantization recipe identifier;
- calibration dataset identifier, sample count, and sequence length;
- named projection input-`amax` tensors.

The refit updates value-dependent packed weights and weight scales. It does not
overwrite fixed W4A4 input scales on every step. A future periodic calibration
policy requires a separate accuracy design because it changes rollout behavior
during training.

The standalone calibration command loads the configured BF16 checkpoint,
applies the selected W4A4 recipe to a temporary calibration model, runs the
declared dataset/sample/sequence-length workload, and writes only named input
`amax` values plus provenance. It does not produce a training checkpoint and is not
part of policy initialization.

## Legacy Refit

For colocated CUDA-IPC or collective refit without an explicit
`refit_transport`, extend `VllmQuantInternalWorkerExtension` to distinguish two
real-quant source representations:

- prepacked ModelOpt tensors from `MegatronQuantPolicyWorker` for existing QARL;
- BF16 logical weights from `MegatronPolicyWorker` for rollout-only NVFP4.

Source representation is inferred from policy metadata and validated before
the first refit. The QARL path remains unchanged. The rollout-only path builds
a stable mapping from BF16 Hugging Face names to ModelOpt destination tensor
families, performs receiver-side quantization, and then enters the existing
vLLM layerwise reload finalization.

Fused MoE gate and up projections share W13 destination storage. Their updates
form one reload group: both projections and their scales must arrive before the
group can be finalized. Down projection W2 is a separate group. Incomplete or
duplicate groups fail before the transport buffer is acknowledged.

## NCCL-Reshard Extension

The existing blanket rejection of `generation.real_quant=true` is replaced by
format-specific capability validation. The supported initial combinations are:

- BF16 policy storage to W4A16 NVFP4 rollout storage;
- BF16 policy storage to W4A4 NVFP4 rollout storage when a valid input-scale
  artifact is configured.

Add destination-owned transform codecs:

```text
bf16 -> modelopt_nvfp4_w4a16
bf16 -> modelopt_nvfp4_w4a4
```

Each codec describes an ordered output family rather than assuming only
`weight` and `weight_scale`. The family can contain:

- packed weight;
- block weight scale;
- global weight scale;

W4A4 input activation scales are static destination state loaded from the
calibration artifact during setup after canonical conversion from input `amax`.
They participate in destination manifest validation and ModelOpt reload
completeness but are not resent by NCCL on every refit.

The NCCL plan reshares the logical BF16 tensor to the destination-local shard
before running the receiver codec. This keeps ModelOpt/vLLM packing details out
of the trainer and avoids transmitting full unsharded tensors. The plan adds a
group-level finalization scope for fused W13, while preserving parameter-level
finalization for independent weights.

NCCL receives transformed components into checkpoint-layout scratch tensors,
not directly into post-processed runtime parameters. Bulk and miscellaneous
updates share one ModelOpt layerwise reload lifecycle, and model finalization
runs once after all component groups are complete.

The policy-side NCCL code exposes BF16 source metadata and participates in the
existing agreement handshake. It does not quantize weights and does not modify
training computation. The generation backend owns destination component names,
shapes, dtypes, packing, and finalization.

## Validation and Error Handling

Setup fails before GPU collectives begin when any of the following is true:

- `policy.quant_cfg` is non-null in a rollout-only recipe;
- the policy backend is not Megatron or generation backend is not vLLM;
- real-quant mode is not W4A16 or W4A4;
- W4A4 has no valid input-scale artifact;
- the source BF16 K dimension cannot form complete NVFP4 block-16 groups after
  destination sharding;
- vLLM expert parallelism is enabled for fused ModelOpt MoE refit;
- source and destination workers disagree on transform id, component family,
  shape, dtype, or finalization scope.

Runtime refit fails loudly on missing, duplicate, or nonfinite component
tensors. vLLM finalization and a device-wide completion fence occur before an
IPC buffer acknowledgment or NCCL refit completion is returned.

## Test Strategy

### Unit tests

- Recipe composition selects a plain BF16 policy worker and real-quant vLLM
  generation worker for both modes.
- Rollout-only mode does not invoke QARL policy export.
- W4A16 BF16 conversion emits the exact packed value, block-scale, and
  global-scale family expected by vLLM.
- W4A4 rejects absent or mismatched input-scale artifacts and loads a complete
  artifact without changing it during repeated refits.
- The calibration command emits deterministic names and provenance for a fixed
  checkpoint, recipe, dataset, and seed.
- Existing prepacked QARL W4A16 and W4A4 tests remain unchanged and pass.
- Fused gate/up refit finalizes only after the complete W13 group arrives.
- NCCL transform agreement serializes and validates all NVFP4 components.
- The validator accepts the two supported BF16-to-NVFP4 combinations and
  continues to reject unsupported source or backend combinations.

### GPU smoke tests

On GCP-NRT B200, run Qwen3-30B-A3B with the four-node performance topology:

1. W4A16 with legacy refit for two GRPO steps.
2. W4A4 with legacy refit and a recorded scale artifact for two GRPO steps.
3. W4A16 with NCCL-Reshard for two GRPO steps.
4. W4A4 with NCCL-Reshard for two GRPO steps.

Each run must complete two refits and two rollout/training steps, use real
importance sampling, emit no incomplete-reload or manifest errors, and log
reward, KL, importance ratio, generation time, refit time, and E2E step time.

### Correctness comparison

Compare matched prompts and initial weights against BF16 rollout and existing
QARL NVFP4 references. Required checks are finite outputs, identical parameter
coverage, bounded logprob differences, no missing refit tensors, and stable
mean reward over the short smoke window. A longer accuracy run is required
before treating frozen W4A4 input scales as a production recipe.

## Delivery Order

1. Add recipe-composition tests and the two rollout-only performance recipes.
2. Add W4A16 receiver conversion and legacy-refit tests.
3. Run and record the W4A16 legacy GPU smoke.
4. Add the standalone W4A4 calibration command, artifact validation, and
   receiver conversion.
5. Run and record the W4A4 legacy GPU smoke.
6. Generalize NCCL transform components and group finalization.
7. Run the two matched NCCL GPU smokes and compare transport timing.

This order gives a functional oracle through legacy refit before changing the
NCCL planning and transfer path.
