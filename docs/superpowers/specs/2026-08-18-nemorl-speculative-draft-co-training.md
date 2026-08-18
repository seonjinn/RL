# Speculative Draft Co-Training

Status: approved for implementation

## Motivation

NeMo RL can co-train a single-pass EAGLE-3 draft model with a Megatron
policy. A fixed draft model becomes stale as RL updates move the policy, which
reduces speculative-decoding acceptance and rollout throughput. Draft
co-training keeps the draft model aligned with the live policy and refits both
models into the generation backend after each policy update.

This design extends the Megatron path with multi-pass EAGLE-3, DFlash, and
DSpark without coupling training code to one vLLM object layout or one
attention-kernel ABI.

## Scope

This work covers:

- Megatron policy training only;
- EAGLE-3 with one or more sequential draft passes;
- DFlash anchor-and-mask block drafting;
- DSpark Markov-only block drafting;
- synchronous and split policy training APIs;
- collective, IPC, and NCCL-reshard refit transports;
- vLLM 0.25.1 and 0.27.1 runtime layouts; and
- Qwen3-8B DFlash and DSpark correctness and performance validation.

The Automodel training path is out of scope. DSpark confidence-head scheduling
is also out of scope because the supported vLLM versions do not consume those
weights during serving.

## Design Principles

1. Method-specific behavior has one owner. Batch preparation, loss metadata,
   checkpoint export, and serving validation belong to a draft-method object,
   not conditionals spread through policy workers.
2. Runtime compatibility is capability-driven. vLLM version strings are useful
   diagnostics, but object access and loading behavior are selected from the
   capabilities exposed by the active runner.
3. Training math is independent of microbatch partitioning. Draft loss
   numerators and per-pass or per-slot counts remain explicit until global
   normalization is known.
4. Refit treats the target and draft as separate components. Each component has
   its own ownership, manifest, coverage, and finalization contract.
5. Correctness uses public, composable attention interfaces. Optimized kernels
   remain replaceable behind the same structured attention plan.
6. Performance claims require end-to-end measurements that include draft
   training and refit, not generation-only throughput.

## Architecture

### Configuration and Method Registry

New user-facing draft configuration uses discriminated Pydantic models:

- `Eagle3DraftConfig`;
- `DFlashDraftConfig`; and
- `DSparkDraftConfig`.

The selected method is resolved once into a `DraftTrainingMethod`. The training
method owns only behavior exercised by the training process:

```python
class DraftTrainingMethod(Protocol):
    def prepare_batch(self, batch: BatchedDataDict) -> DraftBatchPlan: ...
    def forward(self, context: DraftForwardContext) -> DraftOutput: ...
    def loss_stats(self, output: DraftOutput) -> DraftLossStats: ...
```

Checkpoint export is added to each method only when that method is integrated
with training. Serving capabilities remain in the later
`DraftRuntimeAdapter`; training workers do not inspect a remote vLLM runner.

All required values are validated before model construction. In particular,
DFlash and DSpark require their block width, anchors per sequence, mask token,
target hidden-state taps, `sample_from_anchor` behavior, and matching serving
method. Invalid parallelism or attention capabilities fail during setup rather
than during the first batch or refit.

### Immutable Batch Plans

`DraftBatchPlan` is an internal dataclass containing device tensors only. A
block plan describes anchor rows, anchor positions, visible trunk lengths,
block slots, valid labels, and loss-count bins. An EAGLE plan describes pass
indices, inference-aligned RoPE positions, valid labels, and pass weights.

The same plan is consumed by attention and loss code. Anchor selection and
bucket construction use vectorized device operations; the training hot path
must not call `.cpu()`, `.item()`, or loop over batch rows.

Plans are deterministic per sample from the configured seed, optimizer step,
stable sample identity, and slot index. They do not depend on microbatch order.
Synchronous and split training therefore reproduce identical anchors without
requiring the split worker to receive the complete global batch. If stable
sample identities are unavailable, `begin_train_step` must receive an explicit
precomputed plan payload; silently falling back to row order is not allowed.

### Attention

Attention consumes an explicit structured plan instead of mutating each
decoder layer's core-attention object.

For DFlash and DSpark, each query slot attends to:

- the target trunk strictly before its anchor; and
- every valid slot in its own block, bidirectionally.

DFlash and DSpark share the visibility rule but not their query and label
contracts. DFlash constructs one anchor-conditioning query plus `gamma` mask
queries, excludes the conditioning query from the loss, and emits `gamma`
tokens. DSpark with `sample_from_anchor=true` constructs `gamma` queries and
emits `gamma` tokens: slot zero predicts the token after the anchor, and each
later slot adds a teacher-forced Markov bias conditioned on the preceding
token. The batch plan records query count, emitted-token count, teacher label,
loss validity, and Markov input for every slot. Other
`sample_from_anchor` modes require an explicit serving-parity test before they
are enabled.

For multi-pass EAGLE-3, each pass attends to:

- the causal target trunk; and
- prior same-anchor draft branches with the pass-specific RoPE offset used by
  inference.

The initial backend uses a public structured-mask implementation with a dense
FP32 oracle. A backend interface permits a later grouped FlashAttention or
Triton implementation without changing method or worker code. Kernel selection
is based on measured latency and memory. Private positional FlashAttention
forward/backward calls are not part of the model implementation.

Multi-pass state is explicit and append-only. The supported pass count is
bounded by configuration validation. The implementation must demonstrate that
peak retained KV memory does not grow quadratically with the number of passes;
otherwise activation recomputation or a packed-cache kernel is required before
enabling larger pass counts.

### Streaming Vocab-Parallel Distillation

`StreamingVocabParallelSoftCE` consumes target and draft logits as a packed
token stream. It processes bounded sequence tiles and gathers target rows
inside each tile, avoiding a full `[batch, anchors * slots, vocab]` teacher
temporary.

The primitive returns unnormalized statistics:

```python
@dataclass(frozen=True)
class DraftLossStats:
    numerators: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
```

`numerators`, `counts`, and `weights` have the same pass-or-slot shape. TP
collectives compute vocabulary-wide log-sum-exp values. DP collectives reduce
the numerator and count bins. The final draft loss is
`sum(global_numerators * weights) / sum(global_counts * weights)`. Policy
gradients keep the policy denominator; draft-tagged gradients use the draft
denominator. Tests use unequal weights so an unweighted-numerator bug cannot
pass accidentally.

### Synchronous and Split Training

Both training APIs use the same `DraftStepState`.

- Synchronous training prepares the full plan and global count vector before
  forward/backward.
- Split training stores the plan slices and accumulates raw numerators and
  local count vectors. Step finalization performs the DP reduction and applies
  the draft denominator independently of the policy denominator.

Metrics reduce numerator and denominator pairs. A policy-token metric
normalizer is never used as a fallback for a draft metric.

Draft parameters retain NeMo RL's existing separate gradient-norm tag and
clipping behavior. The optimizer builder may additionally assign them a
separate learning rate and weight decay without changing policy clipping or
no-draft optimizer semantics.

### Checkpoint and Refit Contracts

Checkpoint export produces a logical full-weight schema independent of vLLM's
current TP placement. The draft model does not export a private LM head or mask
embedding when the serving model shares those tensors from the live target.

Refit represents an update as two components:

```python
@dataclass(frozen=True)
class ModelUpdateManifest:
    target: WeightComponentManifest
    draft: WeightComponentManifest | None
```

Each component declares its ordered names, byte count, owner ranks, loader,
post-load finalizer, and coverage policy. Non-owner vLLM PP ranks skip draft
loading but still participate in the transport protocol. A draft-owning rank
fails if required weights are absent or unconsumed.

Owner-rank behavior does not imply that every method supports PP. The runtime
adapter contains an explicit runner, method, version, and PP support matrix and
rejects unsupported combinations during setup. In particular, DSpark PP is not
advertised where the upstream runner or loader rejects it. Owner/non-owner
refit tests run only for combinations supported by that vLLM runner.

NCCL reshard sends a draft tensor directly only when both sides expose stable,
exact-layout live storage. Tensors requiring vLLM loader transformations or
post-load derived state use the packed misc path. Loader finalization completes
before reusable transport buffers can be overwritten. A partially failed refit
marks the generation worker unusable.

The implementation extends the reload lifecycle introduced by the native BF16
FlashInfer work instead of bypassing vLLM loaders.

### vLLM Compatibility

`DraftRuntimeAdapter` is resolved once during generation-worker setup. It
records:

- runner family and draft-model accessor;
- target and draft owner ranks;
- method and attention capabilities;
- weight-loader and post-load hooks;
- LM-head and mask-embedding sharing rules; and
- DSpark Markov-head placement.

The 0.25.1 adapter supports legacy `drafter.model` and `speculator.model`
layouts. The 0.27.1 adapter prefers `get_draft_model()` and the newer replicated
DSpark Markov-head layout. Feature code does not scatter version comparisons;
an unsupported capability combination produces one setup-time error containing
the detected vLLM version and runner family.

The dependency remains pinned to vLLM 0.25.1 until a separate dependency-only
pull request moves the default to 0.27.1. Compatibility tests continue to cover
both adapters after the bump.

## Failure Handling and Observability

Setup errors list every incompatible draft, parallelism, serving, and transport
setting in one message. Runtime refit errors include the component, rank
ownership, missing and unexpected weight names, and finalization phase.

Each refit reports target and draft bytes, transferred and loaded tensor counts,
coverage, owner-rank skips, transport time, loader time, and finalization time.
Training reports per-method forward/backward time, loss count bins, draft grad
norm, and peak allocated memory in benchmark mode.

## Pull Request Sequence

Each pull request contains its own tests and documentation and targets roughly
1,500 or fewer production lines where practical. Line count is not the primary
goal: every pull request includes only the production code required by its own
acceptance tests. A later method must not cause speculative hooks, config
fields, or abstractions to be added to an earlier pull request.

1. Introduce typed training config and method registration by adapting the
   current single-pass EAGLE path with exact legacy YAML and default parity. Do
   not introduce serving runtime types.
2. Add streaming vocab-parallel soft cross entropy and explicit draft loss
   statistics, connected to the existing EAGLE path so the primitive has an
   immediate production consumer.
3. Add `DraftStepState` and prove current single-pass EAGLE synchronous/split
   gradient and metric parity.
4. Add only the optional draft learning-rate and weight-decay optimizer group;
   reuse the existing separate gradient clipping.
5. Add the internal DFlash core: vectorized block plans, structured attention,
   model forward/backward oracle, and checkpoint round trip, without policy or
   vLLM wiring.
6. Add DFlash training and export integration for synchronous and split APIs.
7. Add Markov-only DSpark by reusing the DFlash method and attention path,
   including both training APIs and Qwen3-8B validation.
8. Add bounded multi-pass EAGLE-3 with both training APIs, explicit attention
   state and RoPE plans, and pass-count memory gates.
9. After the native BF16 reload lifecycle lands on `main`, add the dual-version
   vLLM draft runtime, component manifest, and collective/IPC refit adapter.
10. Add component-aware NCCL draft reshard on top of the runtime adapter and the
    merged reload lifecycle, including the BF16-to-MXFP8 regression gate.
11. Bump the default vLLM dependency, lock files, and container to 0.27.1
    without removing 0.25.1 adapter coverage. Semantic compatibility fixes
    belong to pull request 9 rather than the dependency bump.

Independent pull requests may be developed in parallel, but publication order
must preserve these contracts. A dependent draft PR identifies its base commit
and is rebased onto upstream `main` when its prerequisites merge.

Pull request descriptions follow the concise structure used by NeMo RL pull
request #3477:

- Summary;
- Why;
- Performance, only when the pull request makes a measured performance claim;
  and
- Validation.

Detailed design rationale, raw benchmark records, and large test matrices live
in linked documents or artifacts rather than the pull request description.
Dependencies and unsupported combinations are stated briefly in Summary rather
than adding a long implementation walkthrough.

## Validation

### Unit and Distributed Correctness

- Dense FP32 attention forward and Q/K/V gradient parity for MHA and GQA.
- Empty trunk, full-chunk trunk, remainder trunk, duplicate anchors, masked
  samples, and anchors near the sequence end.
- EAGLE pass counts 1, 2, 4, and the configured maximum, including RoPE
  positions and gradients into the target trunk and every prior pass.
- Full-batch, microbatch, synchronous, and split gradient and metric parity.
- TP 1 and 2, DP 1 and 2, and every explicitly supported PP layout.
- Real optimizer steps proving independent learning rate, weight decay, and
  gradient clipping.
- Checkpoint round trips and exact refit weight-consumption coverage.
- vLLM PP owner and non-owner rank behavior for collective, IPC, and NCCL
  reshard transports.
- Targeted BF16-to-MXFP8 refit regression coverage from the existing
  receiver-side conversion path after relevant branches merge or rebase.

### Qwen3-8B GPU Matrix

Qwen3-8B is the required integration model for DFlash and DSpark. The matrix is
run once with vLLM 0.25.1 and once with vLLM 0.27.1:

- draft construction and one optimizer step;
- checkpoint export and generation-worker refit;
- greedy output equivalence with speculative decoding disabled;
- accepted tokens per verifier call and per-slot acceptance;
- generation tokens per second and request latency;
- draft forward/backward time and peak memory; and
- refit p50/p95 with target/draft byte and phase breakdown.

Performance comparisons hold checkpoint, prompts, seeds, sampling parameters,
block width, topology, software image, warmup, and GPU clocks constant. Results
include at least five repetitions and retain raw per-request and per-refit data.
An optimization is accepted only if correctness is unchanged and refit-inclusive
rollout performance is non-regressing. A claim of improvement over another
implementation requires a statistically positive result under the same matrix.
The Qwen3-8B BF16 matrix does not replace the targeted BF16-to-MXFP8 refit
regression required for changes overlapping the receiver-side conversion path.

The primary execution site is OCI-Hsg using the `nemotron_n3_post` account on
the `batch` partition. Lyris GB200 and Pre-Tyche GB200 are fallback sites after
an account, partition, container, and checkpoint preflight. Cluster switches do
not mix performance samples in one comparison table; each reported A/B uses the
same cluster, hardware placement, and container.

## Review Gate

Before human review, every pull request must:

1. pass focused unit and applicable distributed/GPU tests;
2. run the repository's `review-pr-team` self-review workflow on the OCI-Hsg primary GPU-capable host;
3. post its output, findings, and dispositions to the PR, and resolve every high-confidence finding with regression tests where applicable;
4. include a devil's-advocate verdict;
5. post test commands, environment, raw performance artifact links, and known
   unsupported configurations; and
6. contain no code whose first consumer is a later pull request; and
7. remain a draft until the self-review evidence is posted, then request human review.
