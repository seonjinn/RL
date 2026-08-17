# R3 Router CUDA Graph Input Design

## Objective

Support Router Replay together with Transformer Engine partial CUDA Graph
training scopes that own `moe_router`. Preserve the expert IDs selected by the
rollout backend while allowing Megatron-Core to recompute differentiable router
scores, router probabilities, expert outputs, loss, and gradients from the
current policy state.

The initial target is Nemotron 3 Nano with packed THD input, TP2, PP2, CP2,
EP8, HybridEP, BF16, three successful optimizer warmups, and the
`attn,mamba,moe_router` scope. The design must remain model-agnostic enough for
the supported Qwen MoE rows, but it does not enable `moe_preprocess` or whole
MoE capture.

## Current Problem

Router Replay currently installs routed expert IDs as eager runtime state.
When `moe_router` is captured, those IDs are not explicit graph inputs. A graph
can therefore retain ordinary top-k routing, a warmup route, or a temporary
Tensor address. NeMo-RL correctly rejects Router Replay combined with a
graph-owned router.

R3-off measurements demonstrate why the rejection must remain fail-closed:

- Nano `moe_router` job `2598985` reached 100% post-warmup graph coverage with
  no fallback or eviction, but reported route-sensitive logprob outliers;
- Nano `attn,mamba,moe_router` job `2598992` reached approximately 2.0x
  cache-hit policy throughput, but reported an unmasked token multiplier of
  `68.2613602`; and
- the outliers were visible between vLLM rollout logprobs and eager Megatron
  previous-policy logprobs, so successful graph execution alone cannot provide
  cross-backend routing correctness.

The implementation must make replayed expert IDs ordinary current-value graph
inputs and prove that the same payload reaches eager previous-policy logprob
and graph-backed policy training.

## Non-Goals

- Do not capture HybridEP dispatch communication, expert compute, combine, or
  postprocess as part of this change.
- Do not enable Nano `moe_preprocess`; its fixed-capacity geometry remains a
  separate capability gate.
- Do not make reference-policy logprob use rollout routes. Reference routing
  remains independent.
- Do not change the rollout backend's router scores or probabilities.
- Do not key graph banks by route values or create one graph per route layout.
- Do not address the separate attention-only THD mask staging optimization.
- Do not treat independent stochastic rollouts as eager-versus-graph parity.

## Selected Approach

Represent routed expert IDs as fixed-capacity Tensor inputs to the graph-owned
router path. NeMo-RL carries the existing R3 payload into the Megatron training
microbatch. Megatron-Core slices and pads the payload with the same
token/sequence/CP contract used by the hidden states, copies current values
into graph-bank-owned static input surfaces before every replay, and consumes
those values inside the captured router computation.

The router still computes logits from the current hidden states. It skips
top-k ID selection when replay IDs are present, gathers current router scores
at those IDs, and applies the existing differentiable probability and scaling
semantics. This preserves gradients to router weights while fixing expert
identity to the rollout route.

### Rejected alternatives

1. **Leave the router eager.** This is safe and remains a fallback
   configuration, but it cannot preserve the measured combined-scope graph
   coverage or performance.
2. **Create a graph bank per route value.** Route values are high-cardinality
   data rather than execution geometry. Keying on them would cause unbounded
   bank growth and recapture.
3. **Capture a Python attribute or temporary route Tensor.** This can replay a
   stale route or pointer and is forbidden.
4. **Detach or replace router probabilities.** This would remove the intended
   router gradient and change training semantics.

## Route Payload Contract

### Logical payload

NeMo-RL may retain the existing full-model payload with a layer axis for
transport and trace attribution. Before entering a graph-owned router,
Megatron-Core resolves that axis through the validated model-layer assignment
and exposes exactly one integer Tensor slice to the router. Its logical rows
correspond exactly to the token rows owned by that layer after packing and
context-parallel partitioning. The trailing dimension is the configured router
top-k. Each valid row contains the rollout-selected expert IDs in their
preserved slot order. The captured graph never consumes the unsliced
full-model payload.

The contract includes typed static metadata:

- payload schema version;
- logical token count and fixed token capacity;
- router top-k and number of experts;
- model-layer identity and MoE payload-axis identity;
- packed-sequence and context-parallel ownership signature;
- route source role, which must be rollout policy replay; and
- route content digest for diagnostics only, never for graph-bank selection.

Route values are dynamic inputs. Capacity, dtype, rank, top-k, layer identity,
and partition ownership are graph-key material.

### Physical representation

The graph input uses the fixed token capacity already selected for packed
training. Valid logical rows contain rollout-selected expert IDs. Every
structural padding row contains the canonical valid dummy route
`[0, 1, ..., topk - 1]`, matching NeMo-RL's packed-route padding contract.
Router computation still executes for fixed-capacity padding rows, so an
out-of-range sentinel must never reach the captured `gather` operation. The
logical token count and existing packed-token validity contract distinguish
real route rows from structural dummy rows; this design does not add a second
independent route-validity mask.

The existing all-`-1` missing-route sentinel remains valid in general eager R3
processing. It is not accepted by `r3_router_cuda_graph_input_v1`: every
logical row must have a complete rollout route before graph entry. Supporting
data-dependent missing-route fallback inside the captured router is a separate
capability.

The implementation must not infer valid rows from expert value `0`; expert
zero is valid. It must not use structural zeros to distinguish non-MoE layers.
The existing model layer-pattern assignment remains the authority for which
payload axes are MoE axes.

### Validation

Before graph launch, every rank validates:

- exact integer dtype and Tensor rank;
- exact fixed capacity, logical count, top-k, and layer identity;
- all valid IDs are in `[0, num_experts)`;
- IDs within a valid top-k row are unique;
- every logical row is populated, contains no missing-route sentinel, and
  every structural tail row uses the canonical dummy route;
- packed token identity and CP ownership match the hidden-state microbatch;
- the R3 payload exists for every graph-owned router layer; and
- the payload belongs to the current training microbatch and storage
  generation.

Malformed or missing state raises an explicit exception before manual hooks,
graph entry, or CUDA graph launch. It must not silently recompute top-k, reuse a
previous payload, or fall back from inside graph replay.

## Ownership and Lifetime

The caller owns the source R3 Tensor. The active graph bank owns a stable
static route-input Tensor with the same physical signature. Replay copies the
source values into that surface before graph launch. The graph captures only
the bank-owned address.

The bank fingerprint includes the route schema and physical signature, but not
the route values. Activation snapshots and rollback cover the route surface
alongside existing packed THD and MoE metadata. Bank uninstall clears active
route ownership. Reset releases it only after the bank is drained.

No long-lived manager may retain an attached autograd reference to the source
route payload. Route IDs do not require gradients. Router logits and gathered
scores remain attached to the current forward graph.

## NeMo-RL Data Flow

1. vLLM returns routed expert IDs with generated tokens.
2. Rollout processing validates the model layer pattern and stores the R3
   payload with token identity.
3. Eager previous-policy logprob consumes the payload through the existing R3
   path.
4. Training sharding carries the same content-bound payload into each policy
   microbatch.
5. Packing and CP slicing transform token rows and route rows together.
6. Megatron-Core copies the current route rows into the selected graph bank and
   replays the graph-owned router.
7. Training telemetry records the source, eager, and graph route digests plus
   exact expert-ID parity and payload validation counts.

Reference logprob does not consume this payload. A route trace is complete
only when the rollout producer, eager previous-policy consumer, and graph
training consumer all report the same current microbatch identity.

## Router Computation Semantics

With no replay payload, routing is unchanged. With a replay payload:

1. compute router logits from the current hidden states;
2. validate and read the supplied expert IDs;
3. gather the current logits or scores at those IDs;
4. apply the existing score function, normalization, scaling, and auxiliary
   bookkeeping supported by eager R3; and
5. return the same routed-expert and probability structure expected by the
   dispatcher and router-gradient path.

The implementation must share the eager R3 helper rather than duplicate its
probability semantics. CUDA Graph changes address/lifetime handling, not the
mathematical Router Replay definition.

## Capability and Configuration Gate

The versioned runtime capability is
`r3_router_cuda_graph_input_v1`. NeMo-RL may allow Router Replay with a
graph-owned `moe_router` only when all of the following are true:

- the Megatron-Core runtime advertises
  `r3_router_cuda_graph_input_v1`;
- Router Replay validation is enabled;
- the model exposes a validated layer-pattern-to-payload-axis mapping;
- the dispatcher/scope pair is in the tested capability matrix;
- packed THD and CP geometry have fixed capacity; and
- the precision and router configuration are covered by parity tests.

If any condition is absent, setup fails with a message naming the missing
capability. Existing R3-on attention/Mamba scopes and R3-off router scopes
remain unchanged.

`r3_router_cuda_graph_input_v1` covers BF16, HybridEP, Nano TP2/PP2/CP2/EP8,
`moe_router`, and `attn,mamba,moe_router`. It does not imply
`moe_preprocess`, whole MoE, FP8, NVFP4, or reference-logprob graph support.

## Telemetry

Add exact counters or immutable run evidence for:

- route payloads produced, eagerly consumed, and graph consumed;
- route rows copied and validated;
- missing, stale, malformed, out-of-range, duplicate, or CP-mismatched rows;
- exact eager-versus-graph expert-ID and expert-count parity;
- graph launches with explicit route input; and
- route payload schema and capability version.

Counters must be reduced consistently across TP/CP/PP/DP ranks. Any unsafe
counter greater than zero blocks result promotion. Route digests are for
attribution and must not expose prompt or generated content.

## Error and Recovery Semantics

- Validate on CPU metadata and device Tensor properties before graph entry.
- Use explicit exceptions for correctness contracts; do not rely on `assert`.
- Do not mutate or detach source route Tensors to recover from invalid state.
- On a failed bank activation, restore the previous bank and route surface
  transactionally.
- On an aborted training step, clear current-microbatch route ownership before
  another step can activate a bank.
- An unseen valid physical signature may select a different existing bank or
  follow the existing outer scheduler policy. It must not switch to eager from
  inside a partially entered replay.

## Testing Strategy

### Unit tests

Write failing tests before production changes for:

- fixed-capacity route packing, CP slicing, padding, and token-identity
  preservation;
- valid expert zero, canonical structural dummy routes, missing-route
  rejection, top-k uniqueness, and range validation;
- missing, stale, wrong-layer, wrong-capacity, and wrong-schema rejection;
- graph-key inclusion of route signature and exclusion of route values;
- bank-owned static copy with changing route values and a stable address;
- transactional activation/rollback and abort cleanup;
- eager and graph paths sharing router probability semantics; and
- setup accepting only the advertised R3-plus-router capability.

Every negative test verifies that graph entry, manual hooks, and graph launch
counts remain zero.

### Megatron-Core distributed parity

Add a TP2/PP2/CP2/EP8 Nano-like row with three warmups and at least 20
same-capacity, changed-route replays. Compare:

- route IDs and expert counts exactly;
- router logits and gathered probabilities within BF16 tolerance;
- layer output, loss, and valid-token logits;
- input gradients and every local parameter-gradient shard;
- simulated parameter deltas; and
- graph coverage, cache reuse, and zero unsafe events.

Include shared-expert enabled and disabled variants where the existing fixture
supports them. Alternating route values must reuse the same bank without
recapture.

### NeMo-RL frozen-batch parity

Use one frozen rollout batch and identical policy/reference state and RNG. Run
eager R3 and graph R3 on the same live policy workers without an optimizer or
scheduler step. Require:

- identical packed input and route digests on every rank;
- exact route IDs and expert counts;
- output, loss, gradient, and parameter-delta equivalence;
- identical `token_mult_prob_error`, policy KL, generation KL, masks, and
  rewards; and
- nonzero requested graph calls, cache hits, zero recapture, zero fallback,
  and zero unsafe route counters.

Independent GRPO jobs are stability evidence, not this parity gate.

## Performance and Acceptance

After unit and distributed parity pass:

1. run a five-step Nano smoke with Router Replay on and
   `cuda_graph_modules=[moe_router]`;
2. run a five-step `attn,mamba,moe_router` smoke;
3. promote each correctness-clean row to 20 steps;
4. compare cache-hit policy training against a same-state, same-batch eager R3
   control; and
5. run a 100-step soak for the selected combined scope.

Promotion requires:

- exact source, runtime, model, topology, batch, state, RNG, and route identity;
- three warmups followed by nonzero capture and cache hits;
- 100% requested-path post-capture coverage;
- zero fallback, eviction caused by under-sized configured capacity, recapture,
  or unsafe route events;
- correctness within the established BF16 parity envelope; and
- cache-hit policy throughput higher than eager after including route-copy and
  control overhead.

E2E throughput is reported separately because generation runs on different
workers and stochastic responses change workload. A policy-core speedup does
not imply an E2E speedup until a fixed-rollout paired comparison passes.

## Rollout and Compatibility

Land the change in reviewable layers:

1. Megatron-Core route payload type, validation, eager semantic reuse, static
   graph input, bank ownership, and unit/distributed tests.
2. NeMo-RL typed payload plumbing, CP/packing identity, capability gate,
   telemetry, and unit tests.
3. Frozen-batch parity harness and persistent experiment leaves.
4. Nano five-step and 20-step evidence, followed by the 100-step soak.

Keep capability disabled by default until the corresponding runtime artifact
passes distributed parity. Older Megatron-Core revisions continue to fail
setup for R3 plus graph-owned router instead of silently changing behavior.

## Success Criteria

The feature is complete only when:

- no graph captures a Python route attribute or caller-owned route address;
- every replay consumes current route values through a stable graph-owned
  surface;
- eager and graph R3 use the same differentiable router-score semantics;
- all malformed, missing, stale, or mismatched payloads fail before launch;
- fixed-batch output, route, loss, and gradient parity passes at Nano topology;
- `moe_router` and `attn,mamba,moe_router` complete 20 steps with Router Replay
  on, full post-capture coverage, and zero unsafe events; and
- the selected combined scope preserves a correctness-clean cache-hit policy
  speedup over eager.
