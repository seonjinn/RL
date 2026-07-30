# Transformer Engine FP64 CUDA Graph Weak-Reference Design

Date: 2026-07-29

## Objective

Add lossless FP64 support to Transformer Engine's CUDA Graph static-buffer
weak-reference path, then validate packed Nemotron Nano partial CUDA Graph
training for attention, Mamba, and MoE router scopes.

The change must not cast router probabilities, change router arithmetic, disable
Transformer Engine input/output buffer reuse, or rebuild Transformer Engine's
native CUDA extensions.

## Confirmed Failure

Ptyche job `2473345` completed three eager optimizer updates and failed on the
first actual Transformer Engine graph capture at step four.

The Nano recipe inherits `moe_router_dtype=fp64`. Megatron Core returns the
FP64 router probabilities as a partial `moe_router` or
`moe_router+moe_preprocess` graph output. Transformer Engine then calls
`make_weak_ref()` on the static outputs because Megatron Core enables
`_reuse_graph_input_output_buffers`.

The pinned nightly image contains Transformer Engine
`2.15.0+42b84005`. Its `_torch_dtype_to_np_typestr_dict` lacks
`torch.float64`, so `_WeakRefTensor.__cuda_array_interface__` raises
`TypeError: Unsupported dtype: torch.float64`.

PR 5672's packed-sequence adapter and the Megatron Core graph-bank lifecycle do
not cause this exception.

## Selected Implementation

Use the upstream-quality Transformer Engine fix:

```python
_torch_dtype_to_np_typestr_dict = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.float64: "<f8",
    ...
}
```

The NumPy/CUDA array-interface typestring `<f8` represents a little-endian
64-bit floating-point value. `make_weak_ref()` continues to use the same CUDA
data pointer, shape, and storage. No Tensor conversion, arithmetic change, or
precision reduction is introduced.

Implement the change from exact NVIDIA Transformer Engine tag `v2.15`
commit `42b840051647eef89761a16dfdff87e82bb253ab`. Use a fresh worktree because
the existing local Transformer Engine checkout has unrelated user changes.
Create and push branch `sj/fp64-cuda-graph-weakref-20260729` to a
`seonjinn/TransformerEngine` fork.

## Tests

### Pure Python Contract

Add a focused test that constructs an FP64 `_WeakRefTensor` and verifies:

- `__cuda_array_interface__["typestr"] == "<f8"`;
- shape and pointer are unchanged;
- the supported-dtype error remains for an actually unsupported dtype.

This test does not require a GPU and proves the dtype-registry behavior.

### CUDA Pointer and Storage Contract

On one GB200 GPU, call `make_weak_ref()` on a CUDA FP64 Tensor and verify:

- returned dtype and shape exactly match the source;
- returned `data_ptr()` exactly matches the source;
- mutations through either Tensor are visible through the other Tensor;
- no copy or cast occurs.

### Graphed Forward and Backward Contract

Use `make_graphed_callables(..., _reuse_graph_input_output_buffers=True)` with
a differentiable FP64 graph output. Verify capture, replay, and backward
complete and compare eager versus graph:

- output dtype is FP64;
- output maximum absolute and relative error are zero for the deterministic
  fixture;
- input-gradient maximum absolute and relative error are zero;
- buffer reuse remains enabled.

### Megatron Core Integration

Run the existing packed graph-bank GPU suite plus focused Nano-like MoE tests
for:

- `moe_router`;
- `moe_router+moe_preprocess`;
- `attn+mamba+moe_router+moe_preprocess`.

For fixed inputs, router top-k indices and routing masks must be exact. Router
probabilities, layer outputs, and input gradients must meet the existing eager
versus graph FP64 tolerances without dtype changes.

## NeMo-RL Runtime Integration

Keep the immutable nightly image
`nemo_rl_nightly_20260729_2472184.sqsh`, whose recorded SHA256 is
`cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91`.

The image's Megatron policy worker environment resolves
`transformer_engine/pytorch/utils.py` through the container-local uv archive at:

```text
/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e/transformer_engine/pytorch/utils.py
```

Materialize the exact patched `utils.py` from the pushed Transformer Engine
commit on shared Lustre, set its mode to `0444`, and bind-mount it read-only
over that single archive file for every Slurm container process.

Before training, a committed preflight must verify:

- `transformer_engine.__version__ == "2.15.0+42b84005"`;
- the mounted file SHA256 matches the experiment manifest;
- the FP64 dtype entry equals `<f8`;
- a CUDA FP64 `make_weak_ref()` preserves dtype, shape, and pointer.

Any mismatch fails before Ray startup. This keeps the original native
Transformer Engine libraries and avoids a native build while executing the
reviewed Transformer Engine source change on every node.

Record both the original container SHA256 and the Transformer Engine overlay
commit/file SHA256 in the experiment manifest and HTML report. Never mutate or
overwrite the original `.sqsh`.

## Twenty-Step Validation

After unit and one-GPU integration tests pass, submit six matched Nano runs:

| Role | CUDA Graph modules |
|---|---|
| Baseline | disabled |
| Dense/hybrid anchor | `[attn,mamba]` |
| Whole MoE | `[moe]` |
| Router | `[moe_router]` |
| Router and preprocess | `[moe_router,moe_preprocess]` |
| Final composition | `[attn,mamba,moe_router,moe_preprocess]` |

Every run uses:

- 20 optimizer steps;
- three successful eager warmup updates;
- sequence packing enabled;
- `cuda_graph_max_packed_seqs=16`;
- two cached schedule banks;
- checkpointing disabled;
- the same immutable model/checkpoint, topology, seed, and rollout settings;
- W&B project `sna-cg-study`.

Steps four and five may contain first-use captures for distinct schedule keys.
Use steps six through twenty as the primary steady-state performance window.
If later steps introduce a new graph key, eviction, recapture, or eager
fallback, report that event and exclude the run from steady-state claims rather
than silently changing the measurement window.

## Correctness and Performance Gates

All runs must complete 20 optimizer updates without NaN/Inf, CUDA illegal
access, NCCL failure, skipped optimizer updates, cache eviction, or eager
fallback.

Fixed-input integration parity is the numerical correctness gate. The
end-to-end RL metrics are a second, stochastic health check:

- loss and KL absolute delta at most `2e-3`; relative delta at most `2e-3`
  when the baseline magnitude is at least `1e-6`;
- gradient-norm relative delta at most `1%`;
- generation KL error absolute delta at most `2e-4` and relative delta at most
  `10%`;
- reward, accuracy, and token probability error remain finite and follow the
  baseline trajectory without systematic drift.

Report median and p95 step time plus tokens/sec/GPU for:

- E2E;
- Generation;
- Policy Training;
- Policy and Reference Logprobs.

An implementation is correctness-acceptable when median Policy Training and
E2E throughput are at least `0.98x` baseline. Claim a performance improvement
only when median Policy Training throughput is at least `1.05x` baseline and
E2E throughput regresses by no more than `2%`. A single 20-step pass is a
validation result, not a convergence claim.

## Failure Handling

- If the pure Python or pointer/storage tests fail, do not run NeMo-RL.
- If forward/backward parity fails, preserve logs and reject the TE patch.
- If the overlay preflight detects a version or SHA mismatch, do not patch an
  unknown nightly dynamically; stage and identify the intended image first.
- If only router scopes fail after the FP64 weak-reference test passes, debug
  the Megatron Core partial-MoE boundary rather than changing router precision.
- Keep the accuracy-neutral Megatron Core buffer-reuse-disable path only as a
  separately measured fallback; it is not part of the selected implementation.
