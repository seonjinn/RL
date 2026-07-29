# Mamba and MoE TE Partial CUDA Graph Design

Date: 2026-07-29

## Objective

Implement and validate Transformer Engine partial CUDA Graph training for packed
Nemotron hybrid models across attention, Mamba, and MoE capture regions. The
implementation must:

- start from the latest NeMo-RL and Megatron-LM main branches;
- preserve the packed-THD adapter contract from Megatron-LM PR 5672;
- preserve the latest Megatron-LM CUDA Graph correctness and buffer-lifetime
  fixes, including PR 5975;
- support variable pipeline microbatch counts produced by NeMo-RL sequence
  packing through a bounded graph-bank cache;
- demonstrate numerical correctness before reporting performance.

This change targets Transformer Engine training graphs. It does not replace
Megatron-LM local or full-iteration CUDA Graph implementations.

## Current Behavior and Gaps

NeMo-RL owns training-level policy:

- validate the requested capture scope;
- build fixed-shape packed THD inputs;
- wait for three successful optimizer updates by default;
- capture before a subsequent forward/backward schedule;
- reject model offload that would invalidate captured addresses.

Megatron-LM owns model-level mechanics:

- discover graphable `TransformerLayer` and `MambaLayer` instances;
- select attention, Mamba, full MoE, router, and preprocess regions;
- construct TE graph sample arguments;
- dispatch replay using `current_microbatch`;
- maintain graph-safe MoE router and dispatcher outputs.

PR 5672 adds a Tensor/static adapter for attention `PackedSeqParams`. Attention
uses cumulative sequence-length tensors and must not receive Mamba-only
`total_tokens` or `seq_idx`.

Mamba sequence packing additionally needs a token-to-sequence `seq_idx` Tensor.
The existing experimental branch flattens and rebuilds Mamba packed inputs, but
it is based before the latest buffer-lifetime changes and passes more cumulative
sequence tensors than Mamba consumes.

Existing Nano evidence proves one five-step combined-scope capture/replay smoke.
It does not prove eager/graph numerical parity or steady-state performance. A
separate run failed when the PP microbatch count changed from five to three
after capture.

## Considered Designs

### Fixed Pipeline Microbatch Count

Pad every packed optimizer step to one configured number of PP microbatches.

Advantages:

- one graph bank;
- simple replay and cleanup;
- no graph switching.

Disadvantages:

- wastes policy-training compute on dummy packed bins;
- the worst-case count can be much larger than the typical count;
- changes the performance characteristic being measured;
- requires careful zero-loss masking and normalization.

### Delete and Recapture on Every Count Change

Keep one graph bank and recapture whenever `num_microbatches` changes.

Advantages:

- bounded memory;
- straightforward ownership.

Disadvantages:

- repeated capture stalls can dominate GRPO steps;
- alternating counts can thrash indefinitely;
- capture overhead contaminates performance measurements.

### Bounded Graph-Bank Cache

Cache TE graph banks by schedule geometry and activate the matching bank before
the forward/backward schedule. Evict least-recently-used inactive banks when the
configured limit is exceeded.

Advantages:

- supports the actual variable packing workload;
- avoids dummy policy compute;
- recurring geometries replay without recapture;
- memory has an explicit upper bound.

Disadvantages:

- requires explicit graph ownership and activation APIs;
- every cached bank consumes CUDA Graph memory;
- first use of a new geometry still incurs capture cost.

The bounded graph-bank cache is the selected design.

## Source and Branch Layout

Use isolated worktrees and separate branches:

- NeMo-RL: latest upstream main, branch
  `experiment/pr5672-mamba-moe-graph-cache-20260729`;
- Megatron-Bridge: latest upstream main, branch
  `sna/pr5672-mamba-moe-graph-cache-20260729`;
- Megatron-LM: latest NVIDIA main, branch
  `sj/pr5672-mamba-moe-graph-cache-20260729`;
- apply the latest PR 5672 changes to Megatron-LM, then implement the Mamba and
  graph-bank extensions;
- push the Megatron-LM branch to the personal fork, update and push
  Megatron-Bridge's nested Megatron-LM pointer, then update NeMo-RL's
  Megatron-Bridge pointer.

Do not reuse or merge the old experimental Megatron-LM branch wholesale.
Forward-port only reviewed changes needed by this design.

## Megatron-LM Design

### Packed Input Schemas

Keep independent attention and Mamba graph-boundary schemas.

Attention dynamic Tensor fields:

- `cu_seqlens_q`;
- `cu_seqlens_kv`;
- `cu_seqlens_q_padded`;
- `cu_seqlens_kv_padded`.

Mamba dynamic Tensor fields:

- `seq_idx` for all packed Mamba captures;
- `cu_seqlens_q` and `cu_seqlens_q_padded` when context parallelism is active.

Mamba does not graph unused KV cumulative-length fields. `total_tokens` remains
outside the graph boundary so reconstruction cannot regenerate or allocate a
new `seq_idx` during capture. Static metadata is stored on the layer and must
match on replay.

Both adapters fail before replay when Tensor field presence, shape, dtype, or
static metadata differs from capture. Boundary values may change while the
contract remains fixed.

### Mamba Capture Boundary

`CudaGraphModule.mamba` captures a complete `MambaLayer`. It is a peer of
heterogeneous transformer layers, not a new region inside
`TransformerLayer`.

Capture:

1. TE receives only Tensor positional and keyword inputs.
2. `MambaLayer._te_cuda_graph_capture` rebuilds a temporary
   `PackedSeqParams`.
3. The rebuilt object carries the original `seq_idx` Tensor without invoking
   data-dependent reconstruction.
4. `MambaMixer` forwards `seq_idx` to the packed Mamba training kernel.

Replay performs the inverse flattening and validates the captured contract.

### TE Graph Banks

Add an explicit graph-bank value object and one graph-bank manager per model
rather than retaining multiple stock helpers or letting callers mutate
`layer.cuda_graphs` without ownership tracking. Stock helpers do not own the
lists they capture, and their deletion method operates on whichever bank is
currently installed. They are therefore unsafe as cache entries.

A graph bank contains:

- the schedule geometry used for capture;
- the captured callable list for each graphable layer;
- enough provenance to verify that scope, layer layout, and model identity
  match the active helper;
- the packed-input/static-metadata signature and MoE dispatcher attribute
  schema.

Required manager operations:

- capture a fresh helper into a manager-owned bank;
- atomically activate one compatible bank across all graphable layers;
- uninstall the active bank without resetting it;
- reset an inactive bank during eviction or shutdown;
- install manual DDP hooks independently of which bank is active.

Only one bank is active on a model at a time. Switching is allowed only between
optimizer steps and before the pipeline forward/backward schedule begins.
Layer graph lists are cleared before every new capture; otherwise their
non-empty state would dispatch replay instead of capture. Every new bank uses a
fresh one-shot `TECudaGraphHelper`.

Capture is exception-safe: it snapshots the active bank, serializes capture
across ranks, and restores the prior bank and process-global capture flags if
the new capture fails. Activation validates the exact graph count instead of
allowing modulo replay to hide a wrong schedule.

Deletion explicitly resets TE graphs when supported and otherwise drops all
strong references before CUDA cache cleanup.

### MoE Correctness

Use latest-main MoE partial-graph mechanics in which router/dispatcher state
crossing a graph boundary is represented as explicit graph output Tensors.
Do not restore the old side-channel weak-reference dictionary.

Validate:

- full `moe`;
- `moe_router`;
- `moe_router+moe_preprocess`;
- shared-expert overlap disabled and enabled;
- `moe_act` selective recompute disabled and enabled;
- fp64 router probabilities retain dtype and values;
- router top-k indices remain identical to eager execution.

Graph-bank switching occurs only after the previous forward, backward, delayed
wgrad, and communication work is drained. Before evicting a bank, the manager
asserts that MoE dispatcher Tensor stores are empty and drops stale dispatcher
references to the evicted bank while preserving the structural attribute-name
schema. Manual DDP hooks are shared by banks because they depend on the model
and scope, not captured graph identity.

The installed nightly Transformer Engine is used directly. No native TE rebuild
is part of the normal workflow. Any required compatibility patch must be
version-gated, tested in the worker environment, and removed when the installed
TE contains the fix.

## NeMo-RL Design

### Cache Key and Policy

The cache varies only the PP schedule:

`TECudaGraphScheduleKey(num_microbatches)`.

For PP equal to one, every request normalizes to key `1`, because one graph per
layer is reused across microbatches. Sequence length, microbatch size, model,
capture scope, packed-sequence Tensor shapes, optimizer, process groups, and
precision are immutable for a worker and are validated before cache lookup.
They are not variable cache dimensions because packed static metadata is stored
on the layer and shared by every bank.

Add one user-facing setting:

`policy.megatron_cfg.cuda_graph_max_cached_schedules`

The exemplar default is `2`. The value must be positive when packed TE graphs
and PP greater than one are enabled. PP equal to one naturally uses one bank.
Eviction requires Transformer Engine 2.10 or newer so a graph can be explicitly
reset and the cache is a physical as well as logical memory bound.

### Lifecycle

1. Count only successful optimizer updates toward the configured three-step
   warmup.
2. Before each forward/backward schedule, compute the complete geometry key.
3. Reconfigure MCore's process-global microbatch calculator for the requested
   schedule, including cache hits.
4. If a cached bank exists, activate it.
5. If no bank exists after warmup:
   - snapshot and uninstall the active bank;
   - create a fresh helper and capture the missing schedule;
   - extract the captured layer lists into a manager-owned bank;
   - store and atomically activate the new bank;
   - evict the least-recently-used inactive bank if over capacity.
6. A failed capture does not create a cache entry and restores the previously
   active bank.
7. A failed optimizer update does not advance warmup or cache state.
8. Worker shutdown resets all cached banks exactly once.

Cache misses and evictions are logged explicitly. Performance aggregation
excludes capture-miss steps. Telemetry records the schedule key, capture count,
replay count, hit count, eviction count, and eager-fallback count. Silent eager
fallback is never permitted after graph warmup.

### Safety Gates

Fail before model execution when:

- the selected scope discovers no matching layers;
- packed Tensor shapes exceed configured token or sequence-count capacity;
- a cached bank belongs to different layer topology or capture scope;
- graph switching is requested inside an open forward/backward step;
- MoE dispatcher state or delayed wgrad work is still live during a switch;
- colocated generation would offload captured model storage.

## Test Strategy

### CPU and Contract Tests

NeMo-RL:

- scope normalization for all supported Mamba/MoE combinations;
- cache-key construction and LRU eviction;
- three-successful-step warmup;
- failed-update and failed-capture rollback;
- activation sequence `5 -> 3 -> 5`;
- PP1 requests normalize to one schedule key;
- MCore's global microbatch calculator is updated on cache hits;
- cleanup resets every bank once;
- config validation and exemplar default.

Megatron-LM:

- attention and Mamba adapters flatten only their declared Tensor fields;
- Mamba reconstruction preserves the supplied `seq_idx`;
- changed shape, dtype, field set, or static metadata fails;
- explicit `[mamba]` discovery and empty-match failure;
- detached bank activation cannot cross model or scope provenance.

### Megatron-LM GPU Correctness

Compare eager and TE graph execution using identical weights and inputs:

- explicit `mamba`;
- `attn+mamba`;
- `moe`, `moe_router`, and `moe_router+moe_preprocess`;
- Mamba/MoE and attention/Mamba/MoE combinations;
- CP2 and PP2 with more than one microbatch;
- graph-bank activation sequence `5 -> 3 -> 5`;
- varying packed boundaries with fixed Tensor shapes;
- shared-expert overlap and `moe_act` recompute axes.

Compare forward outputs, loss, all parameter gradients, and optimizer-updated
weights. Router indices must match exactly. BF16 floating-point comparisons use
the existing MCore graph-test tolerances unless a stricter kernel-specific
tolerance is already defined.

### NeMo-RL Integration

Use the Nano performance recipe with sequence packing and checkpointing
disabled.

Smoke:

- five steps;
- explicit capture and replay events;
- fixed-rollout PP schedule `[5, 5, 5, 5, 3]`;
- every graph row captures or activates geometry three on step five without
  eager fallback;
- all 15 valid graph scopes complete:
  - `attn`, `mamba`, `attn+mamba`;
  - `moe`, `attn+moe`, `mamba+moe`, `attn+mamba+moe`;
  - `moe_router`, `attn+moe_router`, `mamba+moe_router`,
    `attn+mamba+moe_router`;
  - `moe_router+moe_preprocess`,
    `attn+moe_router+moe_preprocess`,
    `mamba+moe_router+moe_preprocess`,
    `attn+mamba+moe_router+moe_preprocess`.

Performance:

- no-CG baseline plus `mamba`, `attn+mamba`, `mamba+moe`,
  `attn+mamba+moe`, `mamba+moe_router`, `attn+mamba+moe_router`,
  `moe_router+moe_preprocess`, `mamba+moe_router+moe_preprocess`, and the full
  combined scope;
- at least 20 total steps and at least 10 steady post-capture samples;
- exclude initialization, warmup, capture, and cache-miss steps;
- report median and p95 E2E step time and tokens/s/GPU;
- report generation, policy training, and combined policy/reference-logprob
  time and tokens/s/GPU;
- report peak GPU memory and graph-bank count.
- require zero silent eager fallbacks and exactly one capture per recurring
  schedule key when cache capacity is two.

Accuracy:

- same checkpoint, dataset, seed, topology, and fixed input batches;
- compare policy loss, reward, generation KL error, policy/reference logprob
  error, gradient norm, and parameter deltas;
- run a longer paired check after the 20-step gate before making a convergence
  claim.

Write run artifacts under the experiment directory and use the W&B project
`sna-cg-study`.

## Completion Criteria

The implementation is complete only when:

- all contract tests pass;
- MCore eager/graph forward, backward, and optimizer parity passes;
- all required attention/Mamba/MoE combinations complete five-step smoke
  replay;
- variable PP microbatch switching works without silent eager fallback;
- paired 20-step performance results and accuracy metrics are recorded;
- the static HTML report distinguishes correctness, smoke, performance, and
  unsupported cases;
- the Megatron-LM, Megatron-Bridge, and NeMo-RL branches and both submodule
  pointers are pushed with reproducible commit provenance.
