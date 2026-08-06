# Qwen MoE Router CUDA Graph Validation Design

## Objective

Determine whether Transformer Engine partial CUDA Graph execution changes
Qwen MoE policy correctness or performance, first on Qwen3-30B-A3B and then
on Qwen3-235B-A22B. Separate CUDA Graph effects from vLLM-to-Megatron routing
differences, weight-refit differences, and stochastic rollout divergence.

## Fixed Experiment Constraints

- Use the latest NeMo-RL `main` merged into the existing experiment branch.
- Preserve the current reviewed Bridge, Megatron-LM, and Transformer Engine
  partial-CUDA-Graph implementation until a test demonstrates that a source
  change is required.
- Run on OCI-HSG `batch` with four GPUs per node.
- Use the digest-pinned nightly container and an exact source/runtime
  attestation created from the submitted commit.
- Use sequence packing, three successful optimizer warmup steps, and no
  checkpoints.
- Store W&B runs under project `sna-cg-study` and logs under the existing
  `nemotron_thd_te_graph_20260731` experiment hierarchy.
- Run a five-step smoke before every new model/scope combination. Promote only
  successful and finite smokes to 20-step performance runs.
- Treat independent 20-step rollouts as convergence and stability evidence,
  not bitwise eager-versus-graph parity.

## Current Evidence

The current Nano result contains a rare vLLM-versus-eager-Megatron logprob
outlier, but that comparison explicitly disabled Transformer Engine replay.
It does not prove that `moe_router` graph replay computes an incorrect result.

Qwen3-30B-A3B already has a four-node, four-GPU-per-node performance recipe and
an official Router Replay recipe. Historical runs completed no-CG and
attention-only 20-step comparisons, while router-plus-preprocess failed on a
packed residual-shape mismatch. No Qwen run has completed on the newest graph
source.

Qwen3-235B-A22B already completed a separate 16-node, four-GPU-per-node,
64-GPU, 20-step non-CG job on OCI-HSG. This proves that the model snapshot,
runtime, and topology are available. It is not a matched baseline because it
used a different source branch and W&B project.

## Safety Boundary

Do not use `Router Replay enabled + moe_router CUDA Graph enabled` as
correctness evidence in the current implementation.

Router Replay installs routed expert indices immediately before a model
forward. The current Transformer Engine graph is captured before Router Replay
is armed, and routed expert indices are not explicit graph inputs. The graph
can therefore retain ordinary top-k routing or a stale route tensor address.

The unsafe arm becomes eligible only after all of the following are true:

1. Routed expert indices are explicit per-replay graph inputs or are copied
   in place into graph-owned persistent buffers.
2. The captured path contains no Python-state decision that changes between
   capture and replay.
3. A fixed-input eager-versus-graph test proves exact expert-ID and expert-count
   parity and acceptable output, loss, gradient, and parameter-delta parity.
4. Packed token occupancy, context-parallel ownership, and unseen shapes either
   select a matching graph bank or fail closed to eager execution.

## Staged Experiment Matrix

### Phase 1: Qwen3-30B-A3B smoke

Submit these five-step jobs concurrently from one source/container provenance:

| Arm | Router Replay | Training CUDA Graph | Purpose |
| --- | --- | --- | --- |
| A | Off | Off | Matched eager baseline |
| B | Off | `moe_router` | Isolate router-scope graph behavior |
| C | On | Off | Measure cross-backend route-replay correction |
| E | On | `attn` | Graph control with the router remaining eager |

Arm D, Router Replay on with `moe_router` graph capture, is deliberately not
submitted.

Each smoke must enable Router Replay validation and trace verification for the
R3 arms. It must reject missing routes, duplicate top-k expert IDs, invalid
expert IDs, context-parallel identity failures, NaN/Inf metrics, graph replay
fallbacks that were not declared, or incomplete optimizer steps.

### Phase 2: Qwen3-30B-A3B performance

Promote passing smoke arms to matched 20-step runs. The first pass uses one
repeat to identify plumbing or numerical failures. If the result is finite and
the baseline and candidate differ, run three matched repeats before making a
convergence claim.

After `moe_router` passes, add `attn,moe_router` as a separate attribution arm.
Do not add `moe_preprocess` until the router-only path passes and packed residual
shapes are verified at every pipeline stage.

### Phase 3: Qwen3-235B-A22B smoke

Use the checked-in 16-node, four-GPU-per-node performance recipe. Before
training, run the routed-expert completeness diagnostic because the R3 payload
contains one route record per token, MoE layer, and top-k slot.

Submit five-step A, B, and C arms. Arm E may run after the R3 payload preflight
passes. Promote A and B to 20 steps first; promote C and E only after their R3
trace is complete and finite. Do not run the unsafe D arm.

## Persistent Launch Surface

Extend the existing experiment harness rather than creating ad hoc submission
commands.

- Add a `qwen3_235b` model selector for the 16n4g recipe.
- Extend the model allowlists and performance-matrix cases.
- Add an explicit Router Replay dimension or persistent R3-on launch leaves;
  distinguish R3 state in run names, metadata, and pairing keys.
- Preserve one shell entry point per condition so every command can be rerun.
- Keep `--test-only` classification and SLURM `sbatch --test-only` as mandatory
  gates.
- Store job ID, exact command, source commits, container digest, model snapshot,
  topology, scope, R3 state, and output path in the experiment ledger.

## Data Flow and Reporting

Every completed run produces a provenance JSON and a canonical per-run JSONL
record. The collector pairs a candidate only with a baseline that has identical
model, topology, dispatcher, step count, repeat, source commits, Transformer
Engine revision, container digest, and run group.

The HTML report must show:

- total step time and E2E tokens/s/GPU;
- generation, policy-training, and policy/reference-logprob time and
  tokens/s/GPU;
- graph calls, eligible calls, coverage, cache hits, captures, and fallbacks;
- reward, generation KL, policy KL, token multiplicative probability error,
  JS divergence, loss, and gradient norm;
- Router Replay trace status and eager-versus-graph parity status;
- included measurement steps and valid sample counts; and
- complete source, image, model, cluster, and job provenance.

Use one aggregation window and statistic for all models in a comparison. Do not
compare the existing Nano manual steps-11-to-19 arithmetic mean directly with
the generic steps-6-to-20 median until Nano is re-exported through the same
ledger.

Qwen3-235B currently disables TensorBoard in its base recipe. The smoke must
verify whether the latest runtime has resolved that issue. If TensorBoard still
fails, add a W&B-history-to-canonical-JSONL adapter rather than enabling a known
broken logger or fabricating missing parity fields.

## Correctness Gates

A run is performance-eligible only if all applicable checks pass:

- every requested optimizer step completes;
- all reported timing and correctness metrics are finite;
- CUDA Graph coverage and fallback counts agree with the declared scope;
- Router Replay trace contains every rollout route and preserves token identity
  through packing and context-parallel slicing;
- no sequence exceeds the baseline logprob-error envelope without being
  reported explicitly;
- post-refit weight fingerprints and a deterministic forward probe match the
  policy source when the probe is available; and
- fixed-input parity artifacts are generated by an actual test, never inferred
  from stochastic reward or KL metrics.

For fixed-input parity, require exact top-k expert IDs and expert counts.
Compare router probabilities, layer outputs, logits, loss, input gradients, and
parameter deltas with dtype-appropriate absolute and relative tolerances.

## Failure Handling

- A five-step failure blocks the corresponding 20-step run.
- A shape or graph-bank miss falls back to eager only when the fallback is
  explicit, counted, and included in coverage; silent reuse is a correctness
  failure.
- A single extreme token logprob error triggers token-level route, score-margin,
  refit, and neighboring-token diagnostics before rerunning.
- A Ray, NCCL, or CUDA error is classified from the first causal error. Later
  distributed teardown failures are not recorded as the root cause.
- Cancel jobs that allocate idle GPUs after the normal initialization window or
  cannot make forward progress.

## Verification and Stop Rules

Before submission:

1. Merge the latest main branch and resolve the exact nested gitlinks.
2. Run launcher, classifier, collector, and renderer tests locally where
   platform-compatible.
3. Commit and push all required source and persistent experiment files.
4. Build a clean source snapshot and fresh runtime attestation.
5. Run local command rendering and remote SLURM `--test-only` for every leaf.

After submission, monitor all jobs for at least five minutes and record the
first failure or successful initialization evidence.

The initial campaign stops after one 20-step Qwen3-30B-A3B A/B comparison and
one 20-step Qwen3-235B-A22B A/B comparison, plus the safe R3 controls that pass
smoke. Additional scopes or three-repeat convergence runs require reviewing the
initial correctness and performance report.
