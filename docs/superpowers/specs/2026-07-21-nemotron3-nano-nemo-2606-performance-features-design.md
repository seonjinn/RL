# Nemotron 3 Nano NeMo 26.06 Performance Features Design

**Date:** 2026-07-21

**Model:** NVIDIA Nemotron 3 Nano 30B-A3B

**NeMo-RL integration base:** `23b6c3feb5dc229d61b6633d7d8f8537f485dcdc`

**Official NeMo-RL main reviewed:** `9f701f069af4424f96d44901c4b7e505bb5a34d1`

**Megatron-Bridge base:** `f15b45e59fbeab33e58123812930473fa2c3be35`

**Megatron-Core base:** `cf2f07d7b1315c96c05554c670c43207c6783e5e`

**Implementation branch:** `sna/nemo-2606-full-cg-a2a-integration-20260713`

## Objective

Enable and measure the three principal NeMo 26.06 MoE performance features in
Nemotron 3 Nano policy training and current-policy Logprob:

1. CuTeDSL fused grouped GEMM;
2. expert-parallel A2A overlap; and
3. fixed-shape full-iteration CUDA Graph.

The work must distinguish feature availability from measured speedup. A feature
is available only after its runtime path and expected kernels are observed. A
speedup is reported only from controlled, replicated MXFP8 comparisons with the
same model, topology, batch shape, data, image, and dependency revisions.

## Current State and Proven Constraints

The integrated NeMo-RL branch already exposes CuTeDSL, A2A overlap, and
full-iteration CUDA Graph configuration. It also has graph support for
synchronous PolicyTraining and current-policy Logprob. The existing support was
developed around Qwen-style SwiGLU MoE models and does not make Nano immediately
runnable.

### CuTeDSL validation is incorrectly GLU-specific for Nano

`_validate_cutedsl_config()` currently requires
`moe_mlp_glu_interleave_size=32` whenever
`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`. Nemotron 3 Nano uses non-gated weighted
squared-ReLU. The pinned Megatron-Core already supports that fused TE operation
chain through `ScaledSReLU` when `use_fused_weighted_squared_relu=true`.

The GLU interleave requirement is therefore valid for the Qwen/SwiGLU path but
not for Nano. The invariant that applies to both paths is that the selected
activation must be supported by the fused Transformer Engine chain.

### Full-CG repeats the same activation mismatch

The full-CG validator also unconditionally requires GLU interleave 32. Nano must
instead satisfy the weighted squared-ReLU fused-path condition. All other MoE
graph requirements remain unchanged: MXFP8, CuTeDSL, HybridEP flex dispatcher,
expert tensor parallel size one, positive static expert-rank capacity, PagedStash,
fixed shapes, no sequence packing, no dynamic batching, context parallel size
one, and a non-colocated resident policy lifecycle.

### Pinned Megatron-Core cannot run Nano A2A overlap

The pinned combined-1F1B implementation accepts `GPTModel` only. Nemotron 3
Nano is a `HybridModel`, so enabling A2A overlap currently fails before useful
training. Megatron-Core pull request 4942 is locally fetchable as commit
`36454d9ca` and adds HybridModel fine-grained callables and schedule-plan
support. It is not an ancestor of the pinned Megatron-Core revision and must be
integrated explicitly before Nano A2A testing.

### Existing Nano recipes do not form a valid performance baseline

The checked-in 4-node Nano QA recipe is a one-step ModelOpt NVFP4 W4A16
functional test. A separate 16-node recipe uses BF16, async GRPO, sequence
packing, and context parallel size eight. Neither is a valid baseline for these
features. CuTeDSL and the current full-CG implementation require MXFP8, and
full-CG requires fixed-shape synchronous execution without packing or context
parallelism.

No cluster run has yet produced Nemotron 3 Nano correctness or performance
evidence for any of the three features.

## Considered Approaches

### Approach A: Enable all three features in one existing recipe

This is rejected. It confounds numerical format and execution topology changes,
hides which feature caused a failure, and cannot attribute a speedup. The
existing Nano recipes also violate full-CG requirements.

### Approach B: Reuse the Qwen GLU validator unchanged

This is rejected. Setting a meaningless GLU interleave value for a non-gated
squared-ReLU model would encode an architecture mismatch and could give a false
impression that the fused Nano activation path was validated.

### Approach C: Staged, architecture-aware integration

This is the selected approach. First prove Nano MXFP8 eager CuTeDSL. Then add
HybridModel schedule-plan support and prove eager A2A. Finally add the complete
fixed-shape Full-CG configuration and test the composed feature cells. This
order follows the runtime dependencies: host-free CuTeDSL kernels make capture
possible, and full-iteration graph replay removes CPU launch overhead that
otherwise limits A2A overlap.

## Architecture

### Shared fused-MoE activation validation

Split validation into two stages. The existing early preflight continues to
check config-visible common requirements. A new typed validator runs after
Megatron-Bridge has produced the model provider and NeMo-RL has applied its MoE,
precision, and performance overrides. The late validator inspects the resolved
`model_cfg` and recognizes the two supported fused activation families:

- GLU path: `moe_mlp_glu_interleave_size == 32`;
- Nano weighted-sReLU path: `activation_func == squared_relu`,
  `gated_linear_unit == false`, `use_fused_weighted_squared_relu == true`, and
  no contradictory GLU interleave configuration.

The CuTeDSL and full-CG paths must call the same resolved-model validator so
their activation requirements cannot drift. Looking only at the YAML boolean is
not sufficient evidence of Nano's actual activation. The helper does not weaken
common requirements such as grouped GEMM, Transformer Engine op fuser, MXFP8,
or ETP1. Unsupported or ambiguous activation configurations fail during setup
with an actionable message.

### Phase A: Nano MXFP8 eager CuTeDSL

Create a dedicated, minimized Nano performance recipe inheriting a checked-in
exemplar. It must use:

- Nemotron 3 Nano 30B-A3B;
- MCore MXFP8 for both OFF and ON cells;
- identical policy, generation, reference, topology, batch, and sequence shape;
- grouped GEMM, weighted squared-ReLU, and Transformer Engine op fuser in both
  cells;
- only `NVTE_CUTEDSL_FUSED_GROUPED_MLP` changes from `0` to `1` in the ON cell;
  and
- eager execution, A2A overlap disabled, and full-CG disabled.

The first topology is 4 nodes with 4 GB200 GPUs per node because the existing
functional Nano recipe already establishes that resource envelope. A smaller
2-node topology may be used only as a correctness smoke test, never silently
substituted for the reported performance topology. The resource split, TP, EP,
DP, microbatch size, and global batch size must resolve to at least two
microbatches per data-parallel rank. Combined-1F1B has no overlapping middle
phase with only one microbatch, so the launcher rejects that shape before A2A
jobs are submitted.

Runtime acceptance requires successful model setup, three policy updates,
finite losses, and profiler or bounded-log evidence of the intended fused
grouped GEMM and weighted-sReLU path. Configuration flags alone are not proof.

### Phase B: HybridModel A2A overlap

Integrate the minimal Megatron-Core change set that provides HybridModel
fine-grained callables and `build_schedule_plan()`. Preserve Megatron-Bridge's
submodule hierarchy and publish every dependency commit before a remote job is
submitted so clusters never depend on unreachable local Git objects.

NeMo-RL's existing typed controls remain the public interface:

- `overlap_moe_expert_parallel_comm`;
- `high_priority_a2a_comm_stream`; and
- `delay_wgrad_compute`.

Add tests that construct a HybridModel-compatible schedule plan, preserve the
existing GPTModel path, and compose with the current NeMo-RL train adapter.
Validate eager A2A OFF/ON before any graph combination. Runtime acceptance
requires NCCL A2A activity plus temporal overlap with expert compute in a trace;
successful training alone proves correctness, not overlap.

The initial Nano topology uses pipeline parallel size one. Virtual pipeline
parallelism from NeMo-RL pull request 1126 is therefore not a dependency of this
work and remains outside scope.

### Phase C: Full-iteration CUDA Graph

Add a separate fixed-shape Nano recipe and an explicit fixed-length data path
rather than modifying the async or NVFP4 recipes. Recipe settings alone cannot
make RL trajectories static: current preprocessing pads only to each step's
observed maximum length.

Extend the existing `MegatronConfig` with an optional
`fixed_sequence_length: int | None`. Its default is documented as `null` in the
GRPO exemplar, and the Nano H-bundle recipes set it to
`${policy.max_total_sequence_length}`. When set, the Megatron policy worker
passes the target and its tokenizer pad ID into the unpacked microbatch path.
That path right-pads `input_ids` with the tokenizer pad ID and every supported
sequence-aligned tensor with its semantic neutral value. This applies
identically to current-policy Logprob and PolicyTraining. It preserves
`input_lengths`, rejects source sequences longer than the target, rejects
packing/dynamic/CP or unsupported multimodal data, and reports the configured
target as the schedule sequence length. Direct worker inputs that are not
exactly normalized to the target fail before capture.

The fixed-length helper operates on the explicit GRPO tensor schema and handles
at least `input_ids`, `token_mask`, `prev_logprobs`, `advantages`,
`generation_logprobs`, `mtp_loss_mask`, and routed-expert data. New
sequence-aligned fields must declare their pad semantics rather than being
silently ignored. Tests prove padded positions remain masked and eager loss is
unchanged.

Factor the PagedStash schedule runner out of the graph-only builder so the eager
H-baseline uses the same static expert capacity, fixed-address stash, all-rank
overflow reduction, and dropless retry contract as the Full-CG cells. Merely
setting `moe_paged_stash=true` in an eager recipe is insufficient: without a
runner consuming the over-budget signal, routed tokens can be dropped silently.
The eager wrapper executes the raw schedule without graph capture, checks the
capacity result on every rank, and either retries dropless with an explicit
counter or fails closed. Correctness smoke tests may exercise the retry, but
performance windows require the retry counter to remain zero.

The complete path must satisfy every current fail-closed graph invariant:

- synchronous PolicyTraining;
- current-policy Logprob only;
- non-colocated vLLM generation with resident policy storage;
- sequence packing and dynamic batching disabled;
- context parallel size one;
- HybridEP flex dispatcher and positive preprocessing SM allocation;
- static expert-rank capacity and PagedStash;
- CuTeDSL fused grouped GEMM with the Nano weighted-sReLU activation path; and
- no storage-invalidating policy offload or colocated refit lifecycle.

All experiment cells disable reference-policy KL work identically with
`reference_policy_kl_penalty=0` and
`skip_reference_policy_logprobs_calculation=true`. Reference Logprob is not a
supported full-CG operation and leaving it enabled would either fail or change
the E2E workload between cells. Current-policy Logprob remains enabled and is
measured separately.

The captured boundary remains the Megatron forward/backward schedule and the
supported current-policy Logprob forward path. Optimizer step, scheduler step,
generation, reward computation, weight transfer/refit, and outer-loop
orchestration remain eager. Reference Logprob is disabled for this controlled
workload. Reports must describe these boundaries and mark unsupported
components as not applicable.

Runtime acceptance requires one capture followed by at least two replays for
PolicyTraining and current-policy Logprob, stable storage signatures, no
unexpected recapture, and `cudaGraphLaunch` correlated with the intended NVTX
ranges. Both eager H and Full-CG cells must expose all-rank capacity-overflow,
dropless retry, and PagedStash fallback counters. The measured window must
contain zero overflow, retry, fallback, or graph-reset events. A successful
retry is correctness recovery, not valid steady-state performance evidence.

## Experiment Matrix

After each individual gate passes, run this controlled matrix. `H` denotes the
Full-CG dependency bundle: HybridEP flex, static expert capacity, PagedStash,
fixed shape, and the matching non-colocated resident lifecycle.

| Cell | CuTeDSL | H bundle | Full-CG | A2A | Purpose |
| --- | --- | --- | --- | --- | --- |
| `c0h0g0a0` | Off | Off | Off | Off | MXFP8 eager baseline |
| `c1h0g0a0` | On | Off | Off | Off | CuTeDSL-only effect |
| `c1h1g0a0` | On | On | Off | Off | Dependency-matched fixed-shape eager baseline |
| `c1h1g0a1` | On | On | Off | On | Eager A2A incremental effect |
| `c1h1g1a0` | On | On | On | Off | Full-CG incremental effect |
| `c1h1g1a1` | On | On | On | On | Full-CG plus A2A composition |

Cells that violate a proven dependency are not run merely to complete a cube.
In particular, Full-CG without CuTeDSL is not a supported Nano cell. Full-CG
speedup is `c1h1g1a0 / c1h1g0a0`, not a comparison against a different
dispatcher or storage policy. A2A has both eager and graph-augmented adjacent
comparisons. Each performance cell uses at least three independent replicas
with alternating submission order. Warm-up steps are excluded using the same
rule in every cell.

## Metrics and Attribution

Collect per-step and post-warm-up aggregates for:

- E2E RL step time and samples or tokens per second per GPU;
- Generation time and throughput;
- current-policy Logprob time and throughput;
- reference Logprob marked not applicable because it is disabled identically;
- PolicyTraining time and throughput;
- refit or weight-transfer time; and
- other orchestration time as the residual, with the residual formula recorded.

Report paired speedup and percent change with replica dispersion. CuTeDSL is
compared as `c1h0g0a0 / c0h0g0a0`: both cells retain the same MXFP8, grouped
GEMM, TE op-fuser, and weighted-sReLU configuration, and only the CuTeDSL
environment switch changes. Feature-local speedups use the dependency-matched
adjacent cells. The total composed comparison against `c0h0g0a0` is reported as
the complete optimized-stack effect and explicitly includes the required H
bundle; it is not mislabeled as the sum of three isolated effects. No Qwen
result or NeMo pretraining TFLOPS figure is presented as a Nano NeMo-RL result.

## Source, Container, and Cluster Reproducibility

Before cluster submission:

1. merge the reviewed official NeMo-RL main revision;
2. commit and push the NeMo-RL branch to the authorized user fork;
3. push the Megatron-Bridge and Megatron-Core dependency commits to reachable
   refs and record their full SHAs;
4. stage an immutable NeMo-RL nightly container with SHA-256 provenance; and
5. record resolved config, source SHAs, image digest, model revision, dataset
   revision, SLURM script, environment allow-list, and profiler command.

Pre-Tyche is the primary cluster. Lyris and AWS are backup sites after source,
container, checkpoint, and dataset equivalence is established. OCI-HSG may be
used when its queue is materially shorter. Scheduling is checked before each
submission, and every new running job is monitored for at least five minutes.

## Test Strategy

Implementation follows test-driven development:

1. unit tests for resolved GLU, weighted-sReLU, unsupported, and contradictory
   fused activation validation;
2. setup and Full-CG tests proving their late model-config validation shares the
   same contract;
3. Megatron-Core HybridModel schedule-plan tests and GPTModel regression tests;
4. NeMo-RL train-adapter composition tests for A2A with eager and graph paths;
5. unit tests for fixed-target padding, pad values, unchanged valid-token loss,
   overflow rejection, and identical Logprob/PolicyTraining shapes;
6. eager and graph PagedStash schedule tests proving identical all-rank
   overflow detection, dropless retry behavior, and counters;
7. resolved-config tests for every Nano experiment cell, including reference
   policy work disabled and at least two microbatches per DP rank;
8. a three-update GPU correctness smoke test; and
9. replicated performance and profiler jobs only after all functional gates.

Local verification includes focused tests, affected unit suites, Ruff, Pyrefly,
YAML/config resolution, shell syntax, and lockfile consistency. Tests requiring
CUDA, Transformer Engine, Megatron-Core, Ray, or model checkpoints run in the
pinned Linux container on a supported cluster.

## Reporting and Failure Handling

Maintain a static HTML experiment report with:

- a timestamped status dashboard;
- immutable source and environment provenance;
- job and log links;
- resolved config and submission script snapshots;
- concise symptom, root cause, fix, and verification entries for each failed
  attempt; and
- final component timing and throughput tables.

Failures are classified before retrying. Configuration, dependency, source
reachability, image, model/data staging, scheduling, OOM, correctness, and
performance-regression failures each have a bounded diagnostic path. A failed
cell is not silently replaced with a smaller model, shorter sequence, different
precision, or different topology.

## Delivery Sequence

1. Merge the latest official NeMo-RL main and re-run the existing integration
   suite.
2. Implement the shared activation-aware validator with tests.
3. Add and validate the Nano MXFP8 eager CuTeDSL recipe and OFF/ON harness.
4. Integrate and publish HybridModel schedule-plan support; validate eager A2A.
5. Add the fixed-shape Nano Full-CG recipe and validate capture/replay.
6. Run the replicated factorial matrix on GB200.
7. Publish the HTML report and a concise result summary with measured effects,
   limitations, and reproducible artifact links.

This design intentionally prioritizes a trustworthy Nano CuTeDSL result before
the larger dependency upgrade. A2A and Full-CG proceed in parallel only where
their source edits and cluster resources are independent; the combined job is
submitted only after both individual functional gates pass.
