# MXFP8 MoE Workload-Replayed Tactic Audit

## Objective

Determine whether the FlashInfer TRTLLM MXFP8 MoE autotuner selects robust
FC1/GEMM1 and FC2/GEMM2 tactics for the expert-token distributions produced by
real NeMo-RL rollouts. Promote a different tactic only when it improves
measured performance, preserves correctness, and remains safe under CUDA Graph
capture and replay.

The first target is Qwen3-30B-A3B on GB200 with the current vLLM 0.25.1 and
FlashInfer environment used by the NeMo-RL MXFP8 performance recipe. A
successful result can then be repeated for Nemotron3 Ultra with its own model,
parallelism, and runtime fingerprint.

## Current Behavior and Gap

Dense MXFP8 linear GEMMs and routed MoE expert GEMMs use separate execution
paths. The existing dense adaptive lookup therefore does not tune MoE FC1 or
FC2.

The installed FlashInfer TRTLLM MoE implementation already:

- exposes legal MoE tactics through its runner;
- tunes FC1/GEMM1 and FC2/GEMM2 choices;
- profiles token-count buckets up to vLLM's maximum batched-token setting;
- uses CUDA Graph and cold-L2 profiling modes; and
- supports serialization of chosen tactics into a versioned JSON cache.

vLLM invokes this tuner during kernel warmup and loads the resulting cache for
serving. The important limitation is that the tuner constructs synthetic
random top-k routing inputs. It does not qualify the chosen tactic against the
expert-token occupancy distributions observed in the target rollout. A tuned
entry is therefore optimal for the synthetic profile used by the tuner, not
necessarily for the target workload.

## Considered Approaches

### 1. Workload-replayed, bucket-level tactic cache (recommended)

Collect real expert-token distributions, replay representative distributions
offline, and select one robust tactic per existing token bucket. Persist the
result through FlashInfer's cache format and leave the runtime lookup unchanged.

Advantages:

- no GPU-to-CPU synchronization in the request path;
- no new CUDA Graph branch or dynamic allocation;
- reuses FlashInfer's legal-tactic and cache interfaces;
- deterministic, reproducible, and easy to disable;
- unseen buckets retain the backend fallback.

Trade-off: one tactic must perform well across the routing distributions mapped
to the same token bucket.

### 2. Exact expert-histogram runtime lookup

Key the selected tactic by the full per-expert token histogram.

This can specialize more aggressively, but the key space is large and the
runtime would need to inspect dynamic routing results before dispatch. That can
introduce synchronization, complicate CUDA Graph capture, and make cache
coverage impractical. This approach is not selected.

### 3. Validation-only audit

Measure the stock autotuner without generating a replacement cache.

This is the lowest-risk option and remains the fallback if no stable opportunity
is found. It does not capture a demonstrated improvement, so it is insufficient
as the primary experiment.

## Experiment Architecture

The workflow has five independently testable stages.

### 1. Routing-signature collection

Instrument the FlashInfer TRTLLM MXFP8 MoE boundary during representative
rollouts. Record only execution metadata:

- model and layer family;
- global and local expert counts;
- top-k;
- total input token count;
- per-local-expert token-count histogram;
- hidden and intermediate dimensions;
- FC1/GEMM1 and FC2/GEMM2 stage identity;
- TP, EP, and DP topology;
- CUDA Graph capture or replay state; and
- vLLM, FlashInfer, CUDA, GPU, weight-layout, and quantization fingerprints.

Do not record prompts, token IDs, hidden-state values, or model outputs. Merge
identical signatures and retain call count and sampled GPU time. The collector
must be disabled by default and must not be enabled in performance measurement
runs.

### 2. Representative workload construction

Rank collected signatures by call-weighted GPU time. Select the smallest set of
token buckets and expert histograms that covers at least 95% of observed MoE
GPU time. Preserve rare signatures in the trace artifact, but do not shmoo the
combinatorial routing space.

Construct synthetic top-k IDs and weights that reproduce each selected
per-expert token count. This preserves the grouped-GEMM workload while avoiding
storage of rollout data. Include at least one balanced, one median-skew, and one
high-skew distribution for each high-weight token bucket when those classes are
observed.

### 3. Offline tactic shmoo

For every selected profile:

1. Ask the installed FlashInfer runner for legal tactics.
2. Measure the stock cached or heuristic tactic and every legal candidate.
3. Measure FC1/GEMM1 and FC2/GEMM2 components separately where the runner API
   exposes separate tactic IDs; otherwise retain the pair and report both
   component kernel times from profiling.
4. Use CUDA Graph replay, cold-L2 inputs, three warmups, and at least ten timed
   repetitions.
5. Record median, p95, coefficient of variation, failures, and kernel names.

Select one robust candidate per existing FlashInfer token bucket using the
call-weighted distribution of the replayed profiles. A candidate is eligible
only when it:

- improves the weighted median by at least 2% over the stock selection;
- has a coefficient of variation no greater than 3%;
- does not regress any high-weight replay profile by more than 1%; and
- passes every micro-correctness and CUDA Graph gate.

If no candidate qualifies, retain the stock tactic for that bucket.

### 4. Cache generation and runtime dispatch

Write qualified entries using FlashInfer's versioned autotune JSON format. Use
the existing vLLM cache-directory and cache-loading mechanism instead of adding
a second MoE lookup implementation.

The cache identity must include all factors already validated by the FlashInfer
and vLLM cache metadata plus the model revision, topology, MoE dimensions,
quantization format, and CUDA Graph mode. A metadata mismatch invalidates the
cache. A missing entry uses the stock FlashInfer cached tactic or heuristic;
cache misses are not errors.

The runtime path must not inspect an expert histogram, launch a profiler, read a
JSON file per call, allocate new tensors, or synchronize with the host.

### 5. End-to-end validation

Run the shipped Qwen3-30B-A3B MXFP8 performance recipe on Ptyche GB200. Hold the
dense linear backend, model revision, quantization scope, topology, generation
settings, container, and node count constant. Change only the MoE tactic cache:

- baseline: stock FlashInfer TRTLLM MoE autotune cache;
- candidate: workload-replayed qualified MoE cache.

Run a two-step smoke first. After it passes, run eight steps and report
steady-state steps 3 through 8. Do not add a scheduling dependency between the
two arms.

## Correctness Gates

### Micro-level correctness

For every promoted tactic and representative routing distribution:

- keep routing IDs and routing weights identical;
- compare FC1 activation output and final FC2 reduced output with the stock
  tactic;
- reject NaN or Inf output;
- use the upstream FlashInfer MXFP8 MoE numerical tolerance for
  `torch.testing.assert_close` and record cosine similarity and maximum error;
- repeat capture and replay with stable output bounds; and
- verify that tactic selection does not modify routing or expert counts.

The stock MXFP8 backend is the immediate reference. A BF16 reference is added
for a representative subset to detect a shared MXFP8 integration error.

### vLLM correctness

Run fixed-prompt, fixed-seed greedy generation with and without the candidate
cache. Require valid token IDs, expected output lengths, no engine or CUDA Graph
errors, and matching deterministic outputs unless an accepted upstream MXFP8
numerical tolerance explains a boundary-token difference.

### Evaluation-level correctness

Run the same matched GSM8K set for the stock and candidate caches. Report exact
match, paired disagreements, and a paired confidence interval. The candidate
must show no statistically significant accuracy regression. A passing
microbenchmark alone is not sufficient for promotion.

For NeMo-RL smoke and measurement runs, also require successful refit, rollout,
logprob, and training phases with finite rewards, losses, and KL metrics.

## Metrics and Reporting

Primary performance metrics:

- MoE FC1/GEMM1 and FC2/GEMM2 GPU time;
- generation time;
- generated tokens per second per GPU; and
- total RL step time.

Secondary metrics:

- call-weighted share of signatures for which the selected tactic changes;
- distribution of microbenchmark speedups;
- cache hit and fallback rates;
- realized token counts and rollout lengths;
- refit, logprob, and training time; and
- run-to-run variation across measured steps.

The report must distinguish microbenchmark opportunity from end-to-end gain.
It must also state whether a result uses stock synthetic-routing autotuning or
the workload-replayed cache.

## Failure Handling

- Invalid or crashing tactics are excluded and recorded with the failing
  signature.
- Cache metadata mismatch invalidates the candidate cache and uses the stock
  path.
- Missing token buckets fall back without failing the request.
- Any CUDA Graph, numerical, token-validity, GSM8K, or NeMo-RL phase failure
  blocks promotion.
- If the end-to-end result does not exceed run-to-run variance, conclude that
  the stock autotuner is sufficient for this workload.

## Ownership and Upstream Boundary

Prefer existing public interfaces first.

- FlashInfer owns legal MoE tactics, stage-specific execution, profiling input
  semantics, autotune cache serialization, and tactic execution.
- vLLM owns warmup coverage, cache-path configuration, distributed cache
  loading, and fallback policy.
- NeMo-RL owns workload generation, experiment configuration, and rollout-level
  correctness and performance validation.

If representative routing inputs cannot be supplied through current FlashInfer
APIs, the smallest upstreamable FlashInfer change is an offline profiling entry
point that accepts a synthetic expert-token histogram and emits a normal
autotune-cache entry. vLLM should not duplicate the FlashInfer MoE tactic search.

## Acceptance Criteria

The study is complete when it produces:

1. a provenance-stamped routing-signature artifact covering at least 95% of
   observed MoE GPU time;
2. a stock-versus-replayed FC1/FC2 tactic audit for every selected bucket;
3. a versioned cache containing only qualified improvements;
4. passing micro, CUDA Graph, vLLM, and matched GSM8K correctness results;
5. a valid eight-step NeMo-RL baseline/candidate comparison; and
6. an HTML and Markdown report containing raw data, normalized plots, cache hit
   rates, correctness results, and an explicit keep-or-reject conclusion.
