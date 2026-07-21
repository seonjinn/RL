# PR5672 Nano Packed CUDA Graph Extension Design

## Goal

Extend the current PR5672 packed-THD Transformer Engine (TE) CUDA-graph
adapter so that it has a safe, explicit behavior on the Nemotron-3 Nano
30B-A3B hybrid model. The extension must preserve the Qwen packed-sequence
path, add a graph-compatible Mamba packed-metadata boundary, and prevent the
known packed context-parallel attention capture failure from invalidating a
training run.

The deliverable is a reviewable Megatron-LM branch plus a NeMo-RL branch that
pins it, focused tests, and reproducible 5-, 20-, and 40-step experiments.

## Evidence and Constraints

- The implementation base is the current head of NVIDIA/Megatron-LM PR5672,
  `6ff66f0a000ee65efa4f322c17871a3938f33427`, rather than the older local
  port. It flattens the attention-facing `PackedSeqParams` tensors for TE and
  reconstructs them inside the captured callable.
- Nano uses packed THD data with TP2, PP2, CP2, and EP8. Its hybrid layer
  layout contains attention, Mamba, and MoE layers; it has no dense MLP-only
  layer. Therefore `mlp` is not a Nano validation scope.
- The current packed CP2 attention capture fails inside the installed
  Transformer Engine context-parallel backward code at a tail `fill_(0)` with
  `cudaErrorStreamCaptureUnsupported`. This code is external to Megatron-LM.
- The current Mamba path passes `PackedSeqParams` directly to
  `GraphableMegatronModule._te_cuda_graph_replay`, which accepts only tensors
  and `None`. It consequently fails before graph replay.
- The completed Nano `moe_router` and `moe_router,moe_preprocess` jobs reach
  TE graph creation but fail while TE weak-references a `torch.float64`
  router output. The installed TE weak-reference conversion supports no
  `float64` dtype. This is independent of packed metadata and reproduces for
  both router scopes.
- The existing `sj/cudagraph-on-ultra` branch supplies useful replay-safety
  behavior: no silent overflow truncation, bucket collision checks, and
  correct eager behavior when a graph cannot be reused. It does not change
  the Transformer Engine CP backward operation or Mamba's TE replay ABI.
- All experiment recipes use three CUDA-graph warmup steps and disable
  training-checkpoint saving. Model-conversion cache reuse is permitted.

## Scope and Non-Goals

In scope:

- Rebase the Megatron-LM extension on the fixed PR5672 commit above.
- Retain the local packed RoPE/CP correction only if it is absent after the
  rebase, with a regression test proving its need.
- Port only the replay-safety guards from `sj/cudagraph-on-ultra` that fit the
  PR5672 API.
- Add a Mamba-specific packed-metadata graph adapter.
- Make packed CP attention capability deterministic before an unsafe graph
  capture is attempted.
- Add NeMo-RL configuration validation, Nano launch coverage, and the
  comparison experiment matrix.

Out of scope:

- Modifying or vendoring Transformer Engine to make its CP THD backward tail
  zeroing capture-safe.
- Capturing full dropless MoE dispatch, all-to-all, expert compute, generation,
  or the whole iteration.
- Capturing a production FP64 MoE router. The model's FP64 router numerical
  contract is retained; lowering its dtype merely to make TE graph capture
  work is not a production fix.
- Treating `moe_act` as a CUDA-graph scope. It is activation-recompute
  configuration, not a capturable module.
- Enabling the `moe` scope for Nano's packed EP all-to-all workload.
- Saving training checkpoints for smoke, performance, or accuracy runs.

## Architecture

```text
NeMo-RL packed batch
  └─ real PackedSeqParams
       ├─ attention TE graph adapter (PR5672)
       │    ├─ dynamic tensor fields: cu_seqlens{_q,_kv}{,_padded}
       │    └─ static metadata: THD format, max lengths, CP topology
       └─ Mamba graph adapter (new)
            ├─ dynamic tensor fields: Mamba-consumed packed tensors,
            │    including seq_idx when present
            └─ static metadata: THD layout and CP topology values

Per graph bucket / model chunk
  ├─ fixed-shape packed metadata buffers
  ├─ immutable static-metadata signature
  └─ graph-or-eager decision
       ├─ compatible input: copy dynamic tensors and replay
       ├─ unsupported packed CP attention: exclude attention graph before capture
       └─ overflow or signature change: preserve eager behavior or recapture,
          never truncate or reuse a stale graph
```

The graph callable must receive tensors and `None` only. The adapter owns
flattening and reconstruction of the data-class boundary; model layers keep
receiving a normal `PackedSeqParams` instance. This keeps the public layer API
unchanged and avoids teaching the generic graph module about Mamba-specific
semantics.

## Components

### 1. PR5672 rebase and replay-safe bucket ownership

Create a dedicated Megatron-LM branch from PR5672 head. Apply the local RoPE
CP fix only if the rebase test demonstrates the upstream head still needs it.
Do not cherry-pick the older complete Ultra implementation because it
duplicates and conflicts with PR5672's adapter.

Port these Ultra properties through the current PR5672 interfaces:

- Use PR5672/TE's existing per-callable, per-model-chunk sample ownership for
  packed metadata buffers; do not add a second global buffer registry. Each
  Mamba callable stores an immutable static signature and TE owns its matching
  tensor input surfaces, so PP2/VPP calls cannot alias a Python data class.
- A metadata-shape overflow raises an actionable configuration error. It does
  not truncate a cumulative-length tensor or replay a differently shaped
  graph.
- An eager call preserves the original `PackedSeqParams` object and manual
  overlap hooks. Graph state is marked reusable only after capture completes.
- A same-sized hidden-state bucket with incompatible packed metadata fails
  clearly rather than sharing a graph.

### 2. Mamba packed-metadata adapter

Add Mamba-specific capture and replay preparation next to the existing
`PackedSeqParams` CUDA-graph helpers. It will:

1. Split the Mamba-required `PackedSeqParams` fields into tensor graph inputs
   and an immutable metadata signature.
2. Capture a closure that rebuilds `PackedSeqParams` from those graph tensors
   before calling `MambaLayer.forward`.
3. On replay, validate the static signature, copy only dynamic tensors into
   the selected bucket buffers, and invoke the captured callable.
4. Preserve eager Mamba behavior whenever the input has no compatible packed
   signature.

`seq_idx` is dynamic because its values vary with document boundaries, even
when the hidden-state bucket is unchanged. `total_tokens` is deliberately not
reconstructed at the graph boundary: after `seq_idx` is materialized, Mamba
does not consume it, while `PackedSeqParams.__post_init__` would otherwise
allocate and recompute `seq_idx` during capture. The adapter must not pass a
`PackedSeqParams` object through Transformer Engine's tensor-only replay
interface.

### 3. Packed CP attention capability gate

The current installed TE cannot capture Nano's packed CP2 attention backward.
Megatron-LM cannot safely fix that external capture operation by changing the
PR5672 metadata adapter alone.

Before any graph is captured, the Nano recipe path checks the tuple
`(NVIDIA-Nemotron-3-Nano model, packed THD, local_cp_size > 1, attention
scope, TE capability)`. This gate is deliberately Nano-specific: it must not
block existing packed CP attention experiments on other model families. If the
known-unsafe TE capability is present, it removes only attention from the
effective graph scope when another requested scope remains; otherwise it
raises one clear configuration error describing the required TE capability.
It must never begin capture, invalidate a stream, and fail later with an
illegal-memory or stream-capture error. A future TE release or upstream TE
patch can flip this gate and enable the same PR5672 attention adapter without
changing NeMo-RL's public configuration.

### 4. NeMo-RL integration and scopes

NeMo-RL preserves the existing packed boundary and loss semantics from the
PR5672 adapter branch. It adds validation for the effective Nano scopes:

- A requested `moe_router` or `moe_router,moe_preprocess` scope with an FP64
  router fails during preflight with the TE dtype limitation. The effective
  scope must never include router capture in that configuration.
- A separate FP32-router recipe may be used only as a non-production
  diagnostic after it explicitly records the changed router precision.
  `moe_preprocess` still requires `moe_router`.
- `mamba` becomes valid after the Mamba adapter passes its graph test.
- `attn` is enabled only when the TE capability gate permits it.
- `mlp`, `moe`, and `moe_act` fail early for this packed Nano recipe with an
  explanation rather than attempting an unsupported graph.

The launcher records both requested and effective scopes, Megatron-LM commit,
TE version/capability result, warmup count, packed-sequence maximum, and graph
bucket list in the experiment directory. This makes an eager fallback visible
in performance reports.

## Error Handling

- Never overwrite real loss boundaries with endpoint-padded graph metadata.
- Never reuse a graph after static metadata changes, a parameter-storage move,
  or an incomplete capture.
- Never silently change a requested attention-only run to a no-graph run. If
  no other requested scope survives the capability gate, fail before model
  execution.
- Report the exact incompatible field, configured maximum, effective scope,
  and required remediation in every validation error.
- Existing no-CG behavior remains the correctness baseline.

## Tests and Validation

### Megatron-LM unit and GPU tests

1. Extend PR5672 packed helper tests for one compatible and one incompatible
   bucket signature, overflow rejection, and eager-object preservation.
2. Add Mamba helper tests proving that two different `cu_seqlens`/`seq_idx`
   values with the same bucket reconstruct distinct runtime
   `PackedSeqParams` without passing a data class to TE replay.
3. Add a distributed GPU Mamba packed forward/backward parity test: eager and
   graph paths use equal inputs, compare output and parameter gradients, then
   repeat with different document boundaries in the same graph bucket.
4. Add a packed CP attention preflight test that verifies an unsafe TE reports
   the configured fallback or error before `make_graphed_callables` runs.
5. Retain the packed RoPE CP regression test when it is still needed after the
   PR5672 rebase.

### NeMo-RL tests and experiments

1. Add launcher/config tests for scope validation and persisted provenance.
2. Assert that a production FP64 router scope fails in preflight before TE
   graph creation. Run the no-CG baseline and `mamba` for five steps after its
   MCore GPU test passes. Each run uses the packed 4-node x 4-GPU topology,
   warmup 3, and no checkpoints.
3. Run matched no-CG and each production-passing scope for 20 steps with identical data,
   seed, topology, and packed-sequence limit. Report E2E, generation,
   logprob, policy-training time, and their token throughputs over post-warmup
   steps only.
4. Run a 40-step no-CG versus each production-passing graph scope comparison using the
   same seed and fixed validation set. Compare reward, accuracy, policy loss,
   KL, clipping diagnostics, NaN/invalid counts, and completion rate.
5. Once TE makes packed CP attention capture-safe, rerun the exact attention,
   attention-plus-router, and baseline matrix. The prior gated result is not
   presented as an attention performance result.

## Acceptance Criteria

- The branch contains PR5672 latest functionality plus no unrelated Ultra
  code.
- Unit tests demonstrate safe dynamic packed metadata handling and Mamba
  replay without a non-tensor TE argument.
- Nano's production-valid scopes either complete the five-step smoke test or
  fail at a deterministic preflight before CUDA graph capture. FP64 router
  graph requests are rejected before `make_graphed_callables`.
- 20-step performance rows are comparable and identify requested versus
  effective scope; 40-step rows report convergence diagnostics against no-CG.
- A merge is considered only after the focused MCore tests, NeMo-RL launcher
  tests, and the corresponding Nano smoke test pass. The MCore branch is
  pushed first; the NeMo-RL branch pins its immutable commit.
