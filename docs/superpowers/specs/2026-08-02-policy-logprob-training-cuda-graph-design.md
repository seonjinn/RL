# Policy Training and Logprob CUDA Graph Design

Date: 2026-08-02

Status: Approved for implementation on 2026-08-03.

## Objective

Support correctness-preserving CUDA Graph replay for both Megatron policy
training and teacher-forced policy/reference logprob evaluation in NeMo-RL.
The target workloads use packed THD inputs and include dense Transformer,
Mamba, and dropless MoE layers in Nemotron 3 Nano, Super, and Ultra and Qwen
MoE models.

The final implementation must:

- preserve the existing Transformer Engine partial training graph path;
- add an independent forward-only partial graph domain for policy and
  reference logprobs;
- retain dynamic dispatcher, expert, combine, and other unsupported work in
  eager execution;
- reuse captured graphs across GRPO iterations without retaining stale
  parameter, FP8, route, or packed-sequence state;
- fall back or fail only at a collective pre-forward boundary, never inside a
  partially replayed model; and
- demonstrate eager-versus-graph parity before performance results are
  accepted.

Generation remains owned by vLLM and is outside this design.

## Relationship to Existing Designs

This document extends, rather than replaces:

- `2026-07-31-nemotron-thd-te-cuda-graph-correctness-design.md`;
- `2026-07-31-packed-seq-aux-loss-te-graph-design.md`; and
- `2026-08-02-qwen-moe-router-cuda-graph-validation-design.md`.

All physical-versus-logical THD geometry, structural-mask, router accounting,
dispatcher replay, scope capability, and correctness requirements from those
documents remain mandatory.

The earlier design's zero-fallback rule continues to apply to policy training.
This document permits a counted eager fallback for an unseen logprob graph key
only before any rank starts the forward schedule. A performance-eligible run
must have zero measured fallbacks after warmup.

## Source Baseline

The design was verified against this isolated stack:

| Repository | Branch | Commit |
|---|---|---|
| NeMo-RL | `experiment/thd-cg-hybrid-nemotron-20260731` | `b1e7a1d15e61c1f80883e2b18696ad61b6a74ec2` |
| Megatron-Bridge | `sna/thd-cg-hybrid-nemotron-20260731` | `69c29747e85328d7a5ba39f8cbea844d60314b11` |
| Megatron-LM | `sj/thd-cg-hybrid-nemotron-20260731` | `5d320e339003f5c2820b1ca0a163e1ca44dfb31e` |

The implementation must first merge the latest upstream main branches using
the repository's normal dependency order, then revalidate every code anchor
and capability gate in this document.

## Confirmed Current Behavior

The current graph implementation is training-only.

- `nemo_rl/models/megatron/setup.py` maps
  `policy.megatron_cfg.cuda_graph_impl=transformer_engine` and
  `cuda_graph_modules` into MCore configuration.
- `MegatronPolicyWorker` constructs `TECudaGraphHelper`,
  `TECudaGraphBankManager`, and `TECudaGraphLifecycle` only for a worker that
  owns an optimizer.
- `train()` creates fixed packed inputs, warms up three successful optimizer
  updates, captures a missing schedule bank, and replays the selected partial
  graph scopes.
- `get_logprobs()` explicitly uninstalls the active training bank, sets
  `model.eval()`, enters `torch.no_grad()`, requests
  `for_cuda_graph_training=False`, and executes `forward_only=True` eagerly.
- `GraphableMegatronModule._should_call_te_cudagraph()` requires
  `self.training`, so an eval-mode logprob forward cannot replay a TE graph.
- `TECudaGraphHelper` currently builds its sample schedule with
  `forward_only=False`.
- `use_reference_model()`, `move_model()`, and `move_optimizer()` reset graph
  banks because storage may relocate.

The current `cuda_graph_modules` setting therefore selects training module
coverage. It does not select NeMo-RL stages and does not graph logprob or vLLM
generation.

MCore's `FullCudaGraphWrapper` proves that a `forward_only=True` schedule can
be captured as a separate validation graph, and Megatron RL uses that wrapper
for old-logprob computation. It is useful reference code, but its process-wide
single validation slot, dict-only static loader, and lack of packed schedule
keys make it unsuitable for direct use by NeMo-RL.

## Considered Approaches

### Selected: dual-mode TE partial graph domains

Keep the existing TE partial training graph and extend the MCore helper and
bank surfaces with an explicit forward-only execution kind. Training and
logprob banks have independent keys, warmup counters, installed callables,
and metrics.

This approach preserves the proven partial scope boundaries. In particular,
`moe_router` and `moe_preprocess` may be graphed while dynamic dispatch,
expert compute, combine, and postprocessing remain eager.

### Optional: outer full-forward logprob graph

Capture the complete `megatron_forward_backward(..., forward_only=True)` call
with a keyed replacement for `FullCudaGraphWrapper`.

This may be useful for dense models or MoE with verified fixed drop-and-pad
capacity. It is not the default because dropless MoE route counts can change
collective and grouped-GEMM geometry. Capturing that dynamic eager tail would
violate the existing partial-MoE safety boundary.

### Rejected as the primary path: MCore local inference graphs

`cuda_graph_impl=local` supports layer/block graphs for MCore's dynamic
inference context. Teacher-forced NeMo-RL logprob instead supplies
`PackedSeqParams` and no inference context. The current local path therefore
falls back to eager before graph creation, loses the requested partial scope
control, and cannot coexist with the single
`cuda_graph_impl=transformer_engine` training setting.

## Selected Architecture

### Independent execution domains

The worker owns three logical graph domains:

```text
TE partial graph owner
  training
    role: policy
    mode: train + autograd
    schedule: forward and backward

  logprob
    role: policy-old
    mode: eval + no_grad
    schedule: forward-only

  logprob
    role: reference
    mode: eval + no_grad
    schedule: forward-only
```

Only one bank is installed at a time. Switching domains uninstalls replay
surfaces but does not destroy cached banks. A different valid packed signature
selects or warms a different bank. Destruction occurs only on LRU eviction or
when an owned Tensor address, topology, or static execution contract
invalidates the bank.

Each domain is the explicit pair `(execution_kind, role)`: training uses
`(TRAINING, policy)`, policy-old logprob uses
`(FORWARD_ONLY_EVAL, policy)`, and reference logprob uses
`(FORWARD_ONLY_EVAL, reference)`. Neither component may be inferred from
`self.training`, `torch.is_grad_enabled`, or the presence of an optimizer.

### MCore execution contract

Add a typed execution-kind value to the TE helper, capture context, replay
context, and graph bank. The exact public name may follow the current MCore
style, but it has these values:

```python
class TECudaGraphExecutionKind(Enum):
    TRAINING = auto()
    FORWARD_ONLY_EVAL = auto()
```

For `TRAINING`:

- the model and graphable leaves must be in training mode;
- the capture schedule uses `forward_only=False`;
- TE forward and backward callables follow the normalized PP/VPP schedule;
- RNG and FP8 training state follow the existing graph contract.

For `FORWARD_ONLY_EVAL`:

- the model and graphable leaves must be in eval mode;
- capture and replay run under `torch.no_grad()`;
- the capture schedule uses `forward_only=True` and contains no backward
  entries;
- dropout and other train/eval-dependent behavior are frozen as eval behavior;
- TE FP8/NVFP4 state is fingerprinted independently from training state; and
- graph replay is allowed only while the explicit forward-only context is
  active.

Replace the implicit `self.training` replay decision with an explicit context
check that also validates module mode. An eval graph must never replay in
training mode, and a training graph must never replay in eval mode.

The helper continues to call Transformer Engine's
`make_graphed_callables()`. A focused capability test must first prove that the
pinned TE version can capture and replay the selected eval/no-grad callable
with no backward use. If the API creates an unusable training-only autograd
surface, forward-only TE partial graphs remain unsupported; the implementation
must not emulate eval by leaving the model in training mode.

### Scope semantics

Logprob uses an explicit scope list independent of the training list.

| Scope | Forward-only graph boundary |
|---|---|
| `attn` | Transformer attention partial leaf |
| `mlp` | Dense MLP partial leaf |
| `mamba` | Mamba layer graphable boundary |
| `moe_router` | Router and supported non-overlapped shared-expert prefix |
| `moe_preprocess` | Dispatcher preprocessing; requires `moe_router` |
| `moe` | Whole MoE only with verified fixed expert capacity |
| empty list | Whole layer only when every contained operation has fixed geometry |

The same incompatibility and dispatcher capability checks used by training
apply independently to the logprob scope. A model-incompatible request fails
at setup rather than producing zero coverage.

For dropless MoE, the recommended logprob scope ends at
`moe_router,moe_preprocess`. Dispatch communication, expert compute, combine,
and postprocess remain eager.

## Configuration

Add a Pydantic policy-level config rather than overloading MCore's single
`cuda_graph_impl` enum. This is a new user-facing config block, so its defaults
live on the `BaseModel`; the exemplar YAML mirrors them for discoverability.
Unknown keys fail validation because a misspelled graph-safety option must not
be silently ignored.

```python
class LogprobCudaGraphConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    implementation: Literal["transformer_engine"] = "transformer_engine"
    modules: list[str] = Field(default_factory=list)
    warmup_steps: PositiveInt = 3
    mb_tokens: PositiveInt | None = None
    max_packed_sequences: PositiveInt | None = None
    cache_size: PositiveInt = 2
    roles: list[Literal["policy", "reference"]] = Field(
        default_factory=lambda: ["policy", "reference"]
    )
    unseen_key_policy: Literal["eager", "error"] = "eager"
```

`PolicyConfig`, which remains a legacy `TypedDict`, gains only a field that
references the new v2 model:

```python
logprob_cuda_graph: NotRequired[LogprobCudaGraphConfig]
```

At the single config boundary, an absent legacy field is parsed once as
`LogprobCudaGraphConfig()`. Consumers receive the validated model and use
attribute access without call-site defaults. Enabling training graphs does
not implicitly enable logprob graphs, and enabling logprob graphs does not
change vLLM generation.

Required setup validation:

- Megatron backend enabled;
- `implementation=transformer_engine`;
- `warmup_steps == 3` for the initial implementation;
- sequence packing enabled and dynamic batching disabled;
- fused attention selected;
- non-`None` fixed `mb_tokens` when enabled;
- non-`None` `max_packed_sequences >= 2` when enabled;
- non-empty, duplicate-free roles;
- explicit, valid graph modules;
- `moe_preprocess` requires `moe_router`;
- `moe` and `moe_router` are mutually exclusive;
- logprob and training capacities satisfy CP/SP alignment;
- the installed TE version passes the forward-only capability gate; and
- every requested precision, dispatcher, route mode, and graph scope is
  supported.

Do not infer logprob token capacity from `logprob_batch_size`. Batch size does
not fix packed token occupancy.

## Fixed Logprob Geometry

### Physical input contract

Every graphable logprob microbatch has one canonical physical geometry:

- `mb_tokens` physical tokens before the established CP/SP transform;
- `max_packed_sequences` model-facing sequences, including one dummy
  sequence;
- `max_packed_sequences + 1` cumulative-length entries;
- fixed structural padding-mask extent;
- fixed token, position, label, and loss-mask extents;
- fixed PP input/output Tensor signatures; and
- fixed per-scope packed metadata signatures.

Real token and sequence occupancy may change. The physical Tensor shapes,
dtypes, devices, layouts, strides, and static metadata may not.
The maximum real sequence count is therefore `max_packed_sequences - 1`.

The logprob packer must reuse the existing training geometry helpers where the
contract is identical. It must not fork a second implementation of CP zigzag,
SP slicing, dummy-sequence insertion, structural masking, or cumulative-length
padding.

### Microbatch normalization

Normalize a logprob call to one of a bounded set of graph schedules before
model execution.

- Split input into fixed-capacity packed microbatches.
- Pad the last microbatch with dummy data.
- Normalize the number of pipeline microbatches when the configured schedule
  permits it.
- Copy dynamic Tensor values into bank-owned static buffers.
- Slice and mask static logprob outputs back to the original batch structure
  after replay.

`ProcessedMicrobatch` receives a typed static-buffer copy adapter. Do not
depend on `FullCudaGraphWrapper.StaticBufferLoader`, which accepts only dict
inputs and owns process-global buffers.

### Graph key

Use a frozen typed key. At minimum it contains:

```text
execution kind: training | forward_only_eval
role: policy | reference
model/topology fingerprint
model storage generation
normalized number of microbatches
microbatch token capacity
packed-sequence capacity
padded sequence length and max-sequence metadata
PP/VPP schedule fingerprint
TP/PP/CP/EP sizes
requested module scopes
dispatcher capability fingerprint
router-replay mode
precision and FP8/NVFP4 recipe fingerprint
static packed metadata fingerprint
```

Dynamic Tensor values such as token IDs, masks, cumulative lengths, and
explicit route IDs are copied into static inputs and are not key values. Their
Tensor signatures are part of the key.

Every distributed rank derives the key before forward. Ranks collectively
compare a stable key digest. A mismatch raises before capture, replay, or any
model collective.

## Lifecycle

Each `(execution_kind, role, graph key)` domain has an independent lifecycle:

```text
unseen key
  -> eager warmup call 1
  -> eager warmup call 2
  -> eager warmup call 3
  -> collective capture
  -> replay
```

Only a successful, finite call advances warmup. A skipped policy-old logprob
call does not advance its lifecycle. Policy and reference calls do not advance
one another's counters.

The logprob manager owns a bounded LRU cache. The initial default is two total
forward-only banks, sufficient for one canonical policy key and one canonical
reference key. Eviction occurs only at a drained boundary with no graph or
collective in flight.

Capture is collective. If any rank fails warmup, signature validation, memory
preflight, or capture, all ranks abandon the candidate and retain the previous
valid banks.

## Parameter and Buffer Address Stability

CUDA Graph validity depends on addresses, not parameter values. Optimizer
updates and in-place policy/reference value copies are allowed; allocation,
replacement, or device movement of an owned Tensor invalidates its banks.

Add a typed storage fingerprint that records at least:

- Tensor data pointer;
- shape, stride, dtype, layout, and device;
- parameter and persistent-buffer identity;
- gradient-buffer identity for training graphs;
- FP8/NVFP4 extra-state Tensor identities; and
- distributed parameter-buffer generation.

Validate the relevant fingerprint before every replay.

### Logprob stage transition

When training CUDA Graphs are enabled:

- policy parameters stay on GPU;
- graph-owned gradient buffers stay allocated at stable addresses;
- `prepare_for_lp_inference()` must not offload or replace them;
- optimizer offload is initially disabled for the combined graph mode; and
- graph bank reset is selective rather than unconditional.

This intentionally trades GPU memory for graph reuse. If the topology cannot
fit stable parameter, gradient, graph-pool, and logprob static buffers, setup
fails the combined mode rather than capturing every iteration and reporting a
misleading speedup.

Optimizer offload may be re-enabled later only after a test proves that no
captured graph owns optimizer-state addresses and that offload does not replace
parameter or gradient buffers.

### Reference weights

Policy and reference logprob banks are separate even when they use the same
physical model object.

Reference activation follows this order:

1. verify the policy storage fingerprint;
2. copy reference parameter values into existing CUDA storage;
3. install reference FP8/NVFP4 state without replacing graph-owned storage;
4. verify the reference storage fingerprint;
5. select or warm/capture the reference bank;
6. run reference logprob;
7. copy policy values and state back in place; and
8. verify the original policy and training fingerprints before continuing.

Ordinary parameter values are not part of a graph key. A pointer or static
state change increments `model_storage_generation` and invalidates every bank
that owns the changed storage.

The first supported precision is BF16. Reference logprob CUDA Graph remains
disabled for FP8/NVFP4 until focused tests prove that `_extra_state`
restoration preserves all graph-owned addresses and metadata semantics. The
implementation must not recapture reference and training graphs every GRPO
step as a substitute for address stability.

An optional persistent reference-model instance with its own storage and bank
is a later memory-for-speed optimization, not an initial requirement.

## Router Replay and MoE

Policy-old logprob may request Router Replay. Reference logprob intentionally
uses its own routing.

For an R3-enabled policy logprob graph, routed expert IDs must be either:

- explicit fixed-signature graph Tensor inputs; or
- copied into graph-bank-owned persistent buffers before replay.

The graph must consume their current values. Capturing a Python attribute,
temporary Tensor address, or route selected during warmup is forbidden.

Until that contract and exact route-parity tests pass:

- R3 plus `moe_router` or `moe_preprocess` logprob graph fails setup;
- R3-off router graph tests are allowed; and
- R3-on attention/Mamba graph tests are allowed only if routing remains
  entirely eager.

The dispatcher capability matrix from the training design applies to
forward-only graphs. There is no assumption that eval mode makes dynamic MoE
communication graph-safe.

## Fallback and Error Handling

### Allowed logprob fallback

With `unseen_key_policy=eager`, an unseen but valid logprob key may execute the
entire forward eagerly when:

- every rank selects fallback before model entry;
- no partial graph bank is installed;
- fallback reason and key digest are counted; and
- the call does not mutate warmup/capture state unless it is an intentional
  warmup call.

Repeated eligible keys may become capture candidates at a later drained
boundary. A performance measurement window starts only after all expected keys
are captured and fallback count stops increasing.

### Fail-closed conditions

Raise collectively before forward for:

- token or packed-sequence capacity overflow when repacking is impossible;
- cross-rank key disagreement;
- Tensor signature or static metadata mismatch for an installed bank;
- unsupported precision, scope, dispatcher, R3 combination, or TE runtime;
- storage fingerprint change;
- graph-pool memory preflight failure;
- reference state that reallocates graph-owned storage; or
- a requested graph scope with no graphable leaves.

There is no rank-local or mid-layer eager fallback. Once a graphable forward
starts, every rank completes the same selected schedule or raises after a
collective-safe synchronization point.

## Telemetry

Report metrics separately for training, policy logprob, and reference
logprob:

- eligible calls;
- eager warmup calls;
- captures and capture failures;
- replay calls and graph coverage;
- cache hits, misses, evictions, and active key digest;
- eager fallback count and reason;
- logical, padded, and capacity tokens;
- real and capacity packed-sequence counts;
- padding utilization;
- static-buffer copy time;
- capture time;
- replay time; and
- storage-generation invalidations.

`policy_and_reference_logprobs` remains the aggregate stage timer, but the
worker additionally emits policy-only and reference-only time and
tokens/second/GPU. A combined Logprob speedup claim must report which roles
actually replayed graphs.

Training coverage must not include logprob calls, and logprob coverage must
not include training calls.

## Correctness Strategy

### Focused unit and contract tests

Write failing tests before production changes for:

- execution-kind normalization and train/eval mode mismatch;
- independent role warmup counters and LRU banks;
- fixed logprob THD packing and dummy-output removal;
- `ProcessedMicrobatch` static-buffer copying;
- collective graph-key equality;
- storage fingerprint stability and invalidation;
- reference value swap with stable BF16 addresses;
- FP8/NVFP4 fail-closed capability gate;
- unseen-key whole-call eager fallback;
- R3 route Tensor copying and stale-route rejection;
- policy/reference metric separation; and
- training banks surviving a logprob stage transition without recapture.

### MCore/TE distributed parity

For each supported scope, run eager and forward-only graph calls with identical
weights, packed inputs, routes, and RNG state.

Required rows include:

- dense `attn`, `mlp`, and `attn,mlp`;
- hybrid `attn,mamba`;
- R3-off `moe_router`;
- R3-off `moe_router,moe_preprocess`;
- combined `attn,mamba,moe_router,moe_preprocess`;
- supported shared-expert off/on variants;
- TP2/CP2/PP2 and model-required EP topology; and
- alternating logical occupancies within one fixed physical signature.

Compare valid-token:

- logits and token logprobs;
- layer outputs;
- router top-k IDs exactly;
- expert counts exactly;
- router probabilities;
- structural padding behavior;
- FP8/NVFP4 metadata when supported; and
- output reuse across at least 20 replays.

### NeMo-RL stage parity

For one frozen rollout batch, run eager and graph modes from identical policy
and reference states. Compare:

- policy-old token logprobs, max/mean absolute error, relative error, and ULP
  envelope;
- reference token logprobs with the same statistics;
- `token_mult_prob_error` and sequence-level multiplicative error;
- `num_masked_seqs_by_logprob_error`;
- `gen_kl_error` and policy/reference KL;
- policy loss, gradient norm, and parameter delta;
- router top-k and expert-count parity;
- reward and valid-token masks; and
- NaN/Inf status.

Capture and warmup calls are excluded from steady-state performance
aggregation but included in correctness checks.

Run five-step smokes before paired 20-step runs. A 20-step performance result
is accepted only when expected training and logprob banks report cache hits,
zero post-warmup fallback, and zero post-capture recapture. The selected best
scope then runs a 100-step accuracy and storage-lifetime soak.

## Performance Reporting

Report matched eager and graph runs for:

- E2E step time and tokens/second/GPU;
- generation time and tokens/second/GPU;
- policy-only logprob time and tokens/second/GPU;
- reference-only logprob time and tokens/second/GPU;
- combined policy/reference logprob time and tokens/second/GPU;
- policy-training time and tokens/second/GPU; and
- static-copy, capture, padding, and graph-pool overhead.

Use the same source commits, container digest, model snapshot, topology,
packed geometry, rollout batch, measurement window, and aggregation statistic
for each pair. Stochastic independent runs are stability evidence, not
fixed-input parity evidence.

The existing HTML report gains separate training and logprob graph coverage,
role-specific cache metrics, and storage invalidation columns.

## Implementation Boundaries

### Megatron-LM

Expected changes are limited to:

- explicit TE graph execution kind and context;
- forward-only PP/VPP capture schedule construction;
- separate train/eval callable installation and replay validation;
- forward-only packed-argument descriptors for Transformer, Mamba, and
  supported partial MoE leaves;
- bank fingerprints and telemetry; and
- focused unit/distributed tests.

Do not broaden dynamic decode inference support as part of this work.

### Megatron-Bridge

Bridge owns the reviewed Megatron-LM gitlink and any typed config plumbing
required to construct the new MCore helper contract. Standalone Bridge tests
exercise forward-only eval capture before NeMo-RL integration.

### NeMo-RL

Expected changes are limited to:

- typed `logprob_cuda_graph` policy config and validation;
- fixed logprob THD geometry using shared packing helpers;
- role-specific lifecycle and bank management;
- collective key/fallback decisions;
- stable model/reference state transitions;
- selective graph invalidation;
- `get_logprobs()` capture/replay integration;
- metrics and report collection; and
- persistent experiment scripts and correctness tests.

Generation workers and vLLM source are not modified.

## Delivery Order

1. Approve and commit this design.
2. Write and commit a file-by-file implementation plan.
3. Add failing MCore capability tests for eval/no-grad TE callables.
4. Implement the MCore execution-kind and forward-only partial graph path.
5. Run dense, packed THD, Mamba, and partial-MoE MCore parity tests.
6. Push Megatron-LM and pin Megatron-Bridge.
7. Add failing NeMo-RL config, geometry, lifecycle, storage, and telemetry
   tests.
8. Implement policy-old BF16 Logprob graph with R3 off.
9. Prove training-bank reuse across Logprob transitions.
10. Implement BF16 reference Logprob graph with stable in-place state swaps.
11. Add Mamba and R3-off partial-MoE scopes.
12. Add explicit R3 route inputs and parity tests.
13. Evaluate FP8/NVFP4 address stability before enabling those modes.
14. Commit and push NeMo-RL and nested gitlinks in dependency order.
15. Run persistent five-step, 20-step, and 100-step experiment gates and
    update the HTML report.

## Completion Criteria

This work is complete only when:

- policy training continues to use the existing TE partial graph scopes;
- policy-old and reference Logprob each replay a separate forward-only partial
  graph bank;
- the combined mode has training and Logprob cache hits with no per-step
  recapture;
- packed THD inputs with alternating occupancy replay one valid physical
  signature;
- unsupported inputs select a counted whole-call fallback or fail before
  forward according to config;
- R3 never reuses a captured or stale route;
- dense, Mamba, router, preprocess, and supported combined scopes pass fixed-
  input parity;
- five-step and paired 20-step NeMo-RL runs pass all correctness gates;
- the best correct scope passes a 100-step accuracy and storage-lifetime soak;
- the report separates PolicyTraining, policy Logprob, reference Logprob,
  Generation, and E2E metrics; and
- Megatron-LM, Megatron-Bridge, and NeMo-RL commits and runtime provenance are
  pinned and reproducible.
