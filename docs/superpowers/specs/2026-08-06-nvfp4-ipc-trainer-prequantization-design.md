# NVFP4 Trainer-Side Prequantization for Colocated IPC

## Summary

Extend NeMo-RL's colocated CUDA-IPC refit path so one setup-time transform
handshake can drive trainer-side prequantization for both MXFP8 and NVFP4.
For BF16 training with W4A16 or W4A4 rollout, the trainer will serialize the
eligible BF16 weights into ModelOpt checkpoint-layout tensors before they enter
the persistent CUDA-IPC staging buffers. The vLLM receiver will load those
components directly and skip its current BF16-to-NVFP4 conversion.

The production scope is colocated Megatron policy plus vLLM generation with
`refit_transport=null`. A separate experimental branch will evaluate the same
NVFP4 transform with NCCL-Reshard after the IPC implementation is correct and
measured. NCCL support will be proposed only if its isolated performance gain
is meaningful.

## Goals

- Preserve BF16 policy training while using W4A16 or W4A4 for rollout.
- Generalize the existing `refit_prequantize` handshake so MXFP8 and NVFP4 use
  one transport-facing contract.
- Quantize each selected logical weight once on a trainer GPU per refit.
- Send packed NVFP4 values and scale components through the existing common IPC
  iterator and persistent staging buffers.
- Reuse the receiver's existing prepacked ModelOpt load path.
- Match the current receiver-side NVFP4 serializer bit for bit.
- Bound temporary GPU memory to one NVFP4 refit group instead of one model.
- Preserve all existing behavior when `refit_prequantize=false`.

## Non-Goals

- Changing training precision or enabling NVFP4 QAT.
- Recomputing W4A4 activation calibration during training.
- Enabling NVFP4 prequantization for DTensor, SGLang, or TensorRT-LLM.
- Shipping production NCCL-Reshard NVFP4 prequantization in the same change.
- Changing the Marlin runtime kernel or quantization scope.
- Quantizing weights excluded by the generation recipe.

## Current Behavior

PR 3294 implements trainer-side MXFP8 prequantization. vLLM reports the
MXFP8-eligible names, the Megatron export iterator converts each BF16 tensor to
E4M3 values plus E8M0 block scales, and CUDA IPC carries those smaller tensors.

The BF16-policy-to-NVFP4 path does not use that transform. The plain Megatron
worker exports BF16 tensors into the IPC staging buffers. The ModelOpt vLLM
receiver groups complete W13 or W2 weights, calls
`serialize_bf16_nvfp4_group()`, and then loads the resulting checkpoint-layout
components. Consequently, persistent buffers remove allocation churn but do
not remove the full BF16 staging copy or receiver-side quantize/pack work.

The existing transform descriptors make this ownership explicit:

- BF16 to MXFP8 describes E4M3 values and E8M0 scales as wire components.
- BF16 to NVFP4 currently describes unmodified BF16 as the wire component and
  packed NVFP4 only as receiver-owned destination state.

## Selected Architecture

### Format-Independent Handshake

Keep `RefitTransformRequest` as the setup-time protocol:

```text
Generation destination
  -> request(source_format="bf16", target_format=<format>, parameter_names=...)
Policy source
  -> validate request and build a deterministic transform plan
  -> return transformed wire metadata
Generation destination
  -> validate and commit the transformed manifest
```

Supported target formats for colocated IPC will be:

- `mxfp8_e4m3_e8m0`
- `nvfp4_w4a16`
- `nvfp4_w4a4`

The handshake runs during refit initialization, before training and before any
weight transfer. Metadata construction uses shape and dtype descriptors; it
must not execute quantization kernels or allocate model-sized tensors.

### Common Export Iterator

Separate logical HF export from wire-format transformation:

```text
Megatron Bridge HF iterator
  -> transform-plan iterator
     -> unchanged BF16 record, or
     -> MXFP8 value plus scale records, or
     -> NVFP4 packed value plus scale records
  -> existing CUDA-IPC/ZMQ stream
```

IPC remains unaware of quantization formats. It consumes only `(name, tensor)`
records, copies their bytes into its ping-pong staging buffers, and sends the
matching names and CUDA IPC handles. This keeps persistent-buffer reuse and the
sender/receiver protocol common across MXFP8, W4A16, and W4A4.

The core transform module owns format names, metadata descriptions, plan
validation, and deterministic component ordering. ModelOpt-specific execution
is imported lazily only when an NVFP4 request is active, preserving the plain
Megatron worker's optional-dependency boundary.

### NVFP4 Grouping

NVFP4 serialization is group-aware:

- An expert gate projection and its matching up projection form one W13 group.
- An expert down projection forms one W2 group.
- Eligible dense projections form single-weight groups.

The iterator will retain only the tensors required to complete the current
group. Once complete, it calls the same canonical ModelOpt-compatible serializer
used by the receiver baseline and immediately yields the packed components.
It must fail before transfer if a requested group is incomplete or duplicated.

W13 gate and up weights share the group weight amax. Quantizing them as unrelated
per-tensor records would change `weight_scale_2` and is not permitted. W2 remains
independent. Quantization occurs after Megatron Bridge has materialized the
logical HF tensor, preserving the current TP/EP gather semantics.

### W4A16 Wire Format

Each selected logical weight produces:

- packed `uint8` weight, containing two 4-bit values per byte;
- E4M3 `weight_scale`, one scale per 16 source values;
- FP32 `weight_scale_2` at the ModelOpt-defined group granularity.

Ignoring the small second-level scale, the wire representation uses about
4.5 bits per source value instead of 16 BF16 bits, a 71.9 percent payload
reduction or approximately 3.56 times fewer bytes for quantized weights.

### W4A4 Wire Format

W4A4 uses the same packed weight family and additionally emits the FP32
`input_scale` required for activation quantization. The trainer loads the same
frozen calibration artifact already validated by the receiver baseline. It
does not collect new activation statistics and does not update scales between
refits.

Artifact identity, model identity, quantization config, and expected parameter
names must be validated once during setup. The small input-scale components are
emitted on every refit so the manifest remains stable; receiver-side caching can
be considered separately only after correctness and performance are established.

### Receiver Behavior

After the second metadata handshake, the receiver classifies the source as a
prepacked ModelOpt manifest rather than BF16. It therefore skips
`_load_bf16_weights()` and its call to `serialize_bf16_nvfp4_group()`.

The existing prepacked path remains responsible for:

- filtering ignored scale records;
- batching fused ModelOpt MoE components when required;
- resolving or replaying vLLM loader routes;
- loading stable Marlin-backed parameter objects;
- detaching temporary IPC-backed tensor references before acknowledgement.

No CPU round trip is introduced. Packed tensors are produced on the trainer GPU,
copied into CUDA staging buffers, reconstructed by CUDA IPC on the colocated vLLM
process, and loaded on the same GPU.

## Configuration and Validation

Use the existing opt-in key:

```yaml
policy:
  generation:
    vllm_cfg:
      refit_prequantize: true
```

The destination quantization config determines whether the request targets
MXFP8, W4A16, or W4A4. A separate NVFP4-only boolean is not introduced.

For NVFP4, validation requires:

- Megatron policy backend;
- BF16 source storage for selected weights;
- vLLM generation backend with `real_quant=true`;
- W4A16 or W4A4 ModelOpt generation config;
- colocated generation;
- `refit_transport=null`;
- a valid frozen calibration artifact for W4A4.

Unsupported combinations fail with an actionable configuration error instead
of silently reverting to receiver quantization. MXFP8's existing supported
transport combinations must remain unchanged.

## Memory and Lifetime

Trainer-side prequantization does not retain a second model copy. The source BF16
tensor is owned by the Bridge iterator; temporary packed components are retained
only until their group has been copied into an IPC staging buffer. Persistent IPC
buffers remain bounded by the configured buffer size.

Peak-memory reporting will compare prequantization disabled and enabled. The
acceptance condition is no model-sized growth and no increase that prevents the
matched B200 recipe from completing. Group-scoped temporary allocation is an
expected trade-off and will be recorded separately from persistent staging.

## Correctness Strategy

### Unit and Component Tests

- Verify W4A16 trainer output names, shapes, dtypes, and order.
- Verify W4A4 output includes the frozen input scale family.
- Compare every packed trainer component bit for bit with the current
  receiver-side serializer for the same BF16 input.
- Verify W13 gate/up shared-amax behavior and W2 independent behavior.
- Verify mixed ignored and quantized weights produce one valid manifest.
- Verify missing calibration, incomplete groups, duplicate members, unsupported
  source dtype, and unsupported transport fail before transfer.
- Verify `refit_prequantize=false` preserves the existing BF16 receiver path.
- Verify the existing MXFP8 prequantization tests remain unchanged and passing.

### GPU Smoke and Training Correctness

Run a five-step GCP-NRT B200 smoke before the 20-step comparison. For matched
prequantization OFF and ON runs, verify:

- the same model, data, seed, topology, quantization scope, and container;
- successful weight updates through every refit;
- finite logits, logprobs, losses, rewards, and KL metrics;
- no divergence in packed component snapshots;
- no new CUDA memory growth across steps.

If the 20-step result is performance-positive and intended for upstreaming,
run a 100-step reward and loss comparison before opening a production PR.

## Performance Evaluation

### Colocated IPC

Use Qwen3-30B-A3B BF16 training with expert-only W4A16 and W4A4 rollout on the
same GCP-NRT B200 allocation. Compare only `refit_prequantize=false` versus
`true` on one source commit. Average steady-state W&B steps 3 through 20.

Report:

- transfer and update time;
- total refit time;
- generation, policy training, logprob, and E2E step time;
- logged generation, policy training, logprob, and E2E tokens/sec/GPU;
- peak allocated and reserved GPU memory;
- mean rollout reward and relevant loss/KL metrics.

The primary success condition is lower transfer/update and total refit time
without correctness regression. E2E improvement is expected but not assumed.

### NCCL-Reshard Follow-Up

Create a separate experimental branch after the IPC implementation passes. The
NCCL path cannot reuse IPC staging, but it can reuse the transform descriptors,
canonical NVFP4 serializer, calibration contract, and receiver manifest.

The prototype must account for local-shard ownership and any cross-rank reduction
needed to reproduce W13 shared amax and ModelOpt second-level scales. It must not
approximate a global scale with independent local values.

Run matched prequantization OFF and ON tests with the same non-colocated
NCCL-Reshard recipe. Start with five steps and promote to 20 steps only after
correctness passes. Adopt the NCCL path only if it improves total refit by at
least 5 percent or E2E step time by at least 1 percent, without higher peak
memory that changes the supported topology. Otherwise retain receiver-side
NVFP4 conversion and record the negative result.

## Delivery Structure

1. Keep `sna/nvfp4-ipc-integration-ab` unchanged as the runtime baseline.
2. Implement the colocated IPC extension in a new isolated worktree and branch.
3. Commit unit and component tests with the implementation using TDD.
4. Push one reproducible GCP-NRT experiment harness with source commit,
   container, recipe overrides, W&B project, and log directories recorded.
5. Derive a separate NCCL experiment branch only after the IPC gate passes.
6. Split any upstream submissions by transport: colocated IPC production change
   first, optional NCCL-Reshard change only after its performance gate passes.

## Acceptance Criteria

- MXFP8, W4A16, and W4A4 can use the same colocated IPC transform handshake.
- W4A16 and W4A4 trainer outputs are bit-identical to the receiver baseline.
- W4A4 uses the same frozen calibration artifact and input scales.
- The receiver skips BF16-to-NVFP4 serialization when prequantization is enabled.
- Existing BF16 and MXFP8 paths pass their targeted unit tests.
- Five-step W4A16 and W4A4 GPU smoke runs complete.
- Matched 20-step prequantization OFF and ON runs are reported with W&B links,
  complete step windows, timing, throughput, memory, reward, and loss metrics.
- NCCL-Reshard remains a separate experiment and is upstreamed only when its
  measured result crosses the predefined performance threshold.
