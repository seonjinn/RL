# DynamicSD Under Synchronous RL Rollout: Where a Batch-Size-Aware K Schedule Pays Off

vLLM 0.24 ships DynamicSD: a user-supplied batch-size -> K lookup
(`num_speculative_tokens_per_batch_size`) that the scheduler applies per step.
A synchronous GRPO rollout traverses the whole batch-size axis every step
(launch at N x G concurrency, drain to a long tail), so it is the natural
stress test. We asked: profiled per-model K schedules vs the best fixed K vs
no speculation, on the exact shapes of the NeMo-RL GB200 SyncRL recipes
(temperature 1.0, top_p 1.0, 32 generations per prompt, barrier per step),
with RedHatAI EAGLE3 Thinking drafters.

---

## Fixed-K3 is a 2x lever on 30B-A3B; DynamicSD matches but does not beat it

| Setting (TP) | baseline | fixed K3 | dynamic (capture-aware) |
|---|---|---|---|
| 30B-A3B math (openmath/math500/dapo, TP1) | 1.00x | **1.86-2.00x** | 1.71-1.90x |
| 30B-A3B swe_verified (TP1) | 1.00x | **1.85x** | 1.68x |
| 32B openmath / dapo (TP2) | 1.00x | **1.12-1.16x** | 1.06-1.10x |
| 32B math500 (TP2) | 1.00x | 1.13x | **1.16x** |
| 32B swe_verified (TP2) | 1.00x | 0.96x | **1.08x** |
| 235B all benches (TP4) | 1.00x | 0.31-0.44x | 0.47-0.53x (swe 0.30x) |
| 30B-A3B 40K long-tail (TP2, 32K gen) | 1.00x | **1.19x** | 0.63x |

Speedups are mean rollout-step wall-time ratios over 4 steps (2 for 40K).
On 235B, where speculation is a net loss everywhere, the dynamic schedule
consistently *mitigates* the damage vs fixed-K3 (0.47-0.53x vs 0.31-0.44x on
math benches) but cannot cross 1.0x - the right call there is no speculation
at all, which only an operator (or a K=0-everywhere schedule) can make.

**vLLM 0.24 caveat, resolved by a 0.25 pilot rerun:** 0.24 has a single
`uniform_decode_query_len = 1 + max K`, so any step where DynamicSD selects
K &lt; max falls off the FULL cudagraph path onto piecewise graphs, while a
fixed-K engine always runs FULL. PR #45953 (v0.25 + V2 model runner,
`VLLM_USE_V2_MODEL_RUNNER=1` required for Qwen3-MoE) captures a graph per K.
Rerunning the pilot points on 0.25 with per-K FULL capture verified:
30B openmath fixed 2.19x / dynamic 2.01x, 32B swe fixed 0.92x / dynamic
0.96x, 40K dynamic 0.67x. Dynamic gained ~13% wall time everywhere, but
fixed gained the same, so **the rankings are version-independent**: fixed-K3
still leads on 30B math, dynamic still leads fixed on 32B SWE (though both
dip below baseline on 0.25), and the 40K depth-collapse persists. Bonus
finding: vLLM 0.25.0 crashes at engine init (ZeroDivisionError in the
drafter's per-K cudagraph manager, `gpu/cudagraph_utils.py`) for any
DynamicSD schedule - we run with a local one-line guard patch; upstream
report pending.

At 4K generations the step spends most wall time at high concurrency, and the
per-BS optimum there is exactly K=3, so a fixed K=3 already sits on the
optimum; the dynamic schedule can only match it (1.87-1.90x vs 2.00x, the gap
is schedule-switching overhead plus the BS 86-127 capture-capped K4/K3 band).

**The one clear DynamicSD win is Qwen3-32B on SWE prompts: fixed-K3 is a net
loss (0.96x) while the derived schedule, which turns speculation off at
BS 256, converts it into a 1.08x gain.** This is DynamicSD working as
designed: not "more speculation", but insurance against speculating in
compute-bound regimes.

---

## The cudagraph capture cliff dominates naive schedules

K=5 profiling collapses at BS=128 (22.2k -> 8.0k tok/s on 30B-A3B): once
bs x (K+1) exceeds `max_cudagraph_capture_size` (512), decode falls back to
eager mode. Our first derived table carried K=5 into BS 86-127 and the
"dynamic" rollout ran *slower than fixed* (37.4s vs 25.4s per step). Profiled
grid points alone cannot see between-point cliffs; the table derivation now
caps K analytically (`bs x (K+1) <= capture budget`), which recovered dynamic
from 1.36x to 1.90x on openmath. **Any DynamicSD deployment must encode
hardware execution-mode boundaries, not just measured throughput points.**

## Deeper K is not the memory-bound answer

K=7 raises mean acceptance length to 4.11 (from 3.73 at K=5) but never beats
K=5 tokens/s, and at BS=1 plain K=3 is fastest (607 vs 590/556 tok/s):
per-position acceptance decays (0.82 / 0.66 / 0.53 / ...) faster than the
extra draft positions add value. The profiled schedules never select K>5.

## Probabilistic drafting adds nothing here

vLLM 0.24's `draft_sample_method="probabilistic"` left acceptance length
unchanged (2.99 vs 3.01 at K=3, temperature 1.0) and cost 3-10% tokens/s from
draft-logit caching. Greedy drafting remains the right default for these
EAGLE3 heads.

## MoE at scale inverts the sign

Qwen3-235B-A22B (TP4): despite healthy acceptance (AL 3.05 at temp 1.0),
fixed-K3 runs at **0.44x** - verifying K+1 tokens multiplies MoE expert
dispatch, which is already compute-bound at BS 64. The profiled schedule
correctly zeroes K at BS >= 64, yet the dynamic rollout still landed at 0.51x;
the residual gap is under diagnosis.

## Long-tail exposes the wrong index variable

The 40K preset was expected to be DynamicSD's showcase; it is its clearest
failure, and the logs say why. **EAGLE3 acceptance collapses with generation
depth**: shallow-phase AL is 2.3-2.5, but by the time median depth reaches
~10K tokens the cumulative draft acceptance rate is 2.9-3.7% (per-position
0.08 / 0.003 / 0.000). Fixed-K3 wins the shallow phase (step 0: 77s vs
baseline 106s) and evaporates in the deep phase (step 1: 186s vs 206s),
netting 1.19x. The dynamic schedule does worse (0.63x) because during the
drain the batch falls into its K=5 bands, paying 5-token drafting overhead at
~0% acceptance exactly where it was tuned to be aggressive - the schedule was
profiled at shallow depth (OSL 2048) and is indexed by batch size only.
**At long generation lengths the binding variable is sequence depth, not
batch size, and `num_speculative_tokens_per_batch_size` cannot express a
depth-aware schedule.** Raising max OSL further (64K) would widen, not close,
this gap; the fix is a depth-conditioned K (or drafters trained for deep
thinking contexts).

We prototyped that fix as a ~15-line scheduler patch on vLLM 0.25 (cap K to 0
once mean generated depth exceeds a threshold; `patches/`). The concept is
not enough on its own: the first run fell into a *third* capture-coverage
trap (a runtime K=0 has query_len 1, which the per-K capture list does not
include), and after adding the K=0 shape the V2 dispatcher mis-matched
speculative batches and slowed even the shallow phase (191s vs 116s step 0).
Depth-aware K therefore needs a dispatch-aware upstream implementation, not a
scheduler-only monkey-patch. On the same 0.25 stack the 40K standings are:
fixed-K3 **1.33x**, dynamic 0.68x, depth-capped dynamic 0.35-0.42x.

---

## Nemotron3 MTP: built-in drafting flips the big-MoE verdict

Nemotron3 Super 120B-A12B (FP8, TP4) and Ultra 550B-A55B (NVFP4, TP4) ship a
single in-checkpoint MTP module; vLLM reuses it chained for K>1
(`{"method": "mtp", "num_speculative_tokens": K}`). Two surprises:

1. **Chained reuse does not decay the way an external head does**: K=3 mean
   acceptance length is 2.96 (Super) / 3.00 (Ultra) at temperature 1.0 -
   equal to the separately-trained EAGLE3 drafters on Qwen3.
2. **The big-MoE sign flips.** Same-shape sync rollouts (temp 1.0):

| Setting | baseline | fixed K3 (MTP) | dynamic |
|---|---|---|---|
| Super 120B openmath | 37.2s | 1.50x | **1.53x** |
| Super 120B swe_verified | 48.3s | **1.47x** | 1.43x |
| Ultra 550B openmath | 67.8s | **1.75x** | 1.75x (schedule is K3-everywhere) |
| Ultra 550B swe_verified | 70.7s | **1.56x** | 1.55x |

Qwen3-235B-A22B with an external EAGLE3 drafter was a net loss (0.31-0.44x);
Nemotron3-Ultra at 2.4x the parameter count gains 1.56-1.75x. **"SpecDec
does not pay at MoE scale" was a statement about external drafters, not about
speculation** - the in-checkpoint MTP head shares the target's backbone and
quantization, eliminating the dispatch-heavy separate-drafter overhead.
DynamicSD adds nothing for Ultra because its profiled optimum is K=3 at every
batch size (the schedule degenerates to fixed-K); Super's schedule has a
K1/K2 range at BS 128, whose dynamic run needed one more vLLM patch (Mamba
per-K capture assert, ledger #7). With that fix, Super openmath is the first
math setting where dynamic edges out fixed-K (1.53x vs 1.50x) - the K1 range
at BS 128 pays for itself.

## How much long tail actually forms

Per-request drain data answers when DynamicSD can matter at all. At the
recipe-standard 4K cap on math prompts there is **no tail**: thinking-style
outputs exceed the cap, so p50 = p90 = max = 4096 - every sequence is
truncated at the same length, the batch stays full to the end (last-10% tail
= 0-3% of step wall), and the batch-size axis never moves. **Fixed-K wins
structurally in the standard recipes because there is nothing dynamic to
adapt to.** A real tail only appears at 32K caps: p50 10K vs max 24K, the
last 10% of sequences consume ~20% of the wall, and half the wall runs at
under half occupancy. Measurement caveat: with `SamplingParams(n=G)` vLLM
reports finish times per parent prompt (all G copies together), which hides
the tail; the harness now submits G explicit copies per prompt to expose
per-generation drain.

## MTP survives depth; long-context gains die elsewhere

Super 120B long-tail runs (32K / 64K max_tokens, openmath): **MTP acceptance
holds at depth** - AL stays 2.0-2.6 at median generated depth 4-7K, where
EAGLE3 had already collapsed to per-position 0.08. The in-checkpoint head
tracks the target distribution independent of depth. Yet the 32K rollout
shows **no net speedup (fixed 1.01x, dynamic 1.00x vs 1.50x at 4K)**: with
acceptance intact, the remaining suspect is verification cost - the Mamba
hybrid must roll back / recompute state for speculative verification, and
that overhead appears to grow with context until it cancels the acceptance
gains. This is a different failure mode from EAGLE3's acceptance collapse.

A depth-controlled sweep (forced OSL, K3/K0 tokens-per-second ratio) resolves
the mechanism - it is a **depth x batch-size interaction**:

| forced OSL | BS 1 | BS 8 | BS 32 |
|---|---|---|---|
| 2K | 2.08x | 1.66x | 1.61x |
| 8K | 2.73x | 1.77x | 1.43x |
| 16K | 3.03x | 2.02x | 1.12x |
| 32K | **3.19x** | **2.10x** | **1.07x** |

Deeper context makes speculation MORE valuable at low concurrency (decode is
more memory-bound) and worthless at BS >= 32 (long-context verify compute).
The 32K rollout nets 1.0x because most wall time sits in the deep/high-BS
cell. Neither a fixed K nor a batch-size-only schedule can express this
diagonal - **a correct depth x BS schedule would speculate hardest exactly in
the drain tail (3.19x at BS1/32K)**, which is the quantitative case for the
dispatch-aware depth-conditioned K feature (ledger #5/#6). The 64K redo
(4 steps, per-generation timing) shows SpecDec net-negative at that scale
(fixed 0.65x, dynamic 0.62x; SD runs spend 81-85% of wall in the last-10%
tail at depths beyond the measured 32K sweet spot).

## Statistical robustness and output quality (P0 gates)

**Seed repeats (3 seeds x 4 key settings, mean +/- std of step-wall speedup,
warmup step excluded):**

| Setting | fixed-K3 | dynamic |
|---|---|---|
| Qwen3-30B-A3B openmath | 2.192x +/- 0.007 | 2.044x +/- 0.036 |
| Qwen3-32B swe_verified | 0.922x +/- 0.014 | 0.950x +/- 0.014 |
| Nemotron3-Super openmath | 1.576x +/- 0.045 | 1.568x +/- 0.046 |
| Nemotron3-Ultra openmath | 1.758x +/- 0.016 | 1.757x +/- 0.028 |

Every headline gap is far outside seed noise: fixed>dynamic on 30B math is
real (0.15 gap vs 0.04 std), dynamic>fixed on 32B SWE is real (0.03 gap vs
0.014 std), and the MTP ties are true ties.

**Output quality:** bitwise greedy parity is not a valid gate on this stack -
the baseline engine itself returns 2-28 distinct outputs among 32 greedy
copies of one prompt (batch-position numerics; worst on FP8 MoE). At the
answer level (majority boxed answer over 32 greedy copies, balanced-brace
parser), SD variants agree with baseline in 6/6 stable prompts across both
model families; the single differing prompt is one where the baseline itself
is unstable (11/32 majority). Combined with the losslessness of rejection
sampling, speculation does not alter answer quality. (Amusingly, the graded
probe also caught OpenMathInstruct-2 ground-truth noise: for prompt 0 the
dataset says "2" but the equation has no valid solution - all three variants
of both models correctly answer "no solution".)

## E2E validation: standalone predictions transfer, Amdahl-exactly

Real NeMo-RL GRPO (unmodified `grpo-qwen3-30ba3b-4n4g.yaml`, 4 nodes x 4
GB200, 10 steps, baseline vs EAGLE3-K3 via
`vllm_kwargs.speculative_config`):

| Phase (steady mean) | baseline | eagle3-K3 | speedup |
|---|---|---|---|
| generation | 68.5s | 41.2s | **1.66x** |
| policy_training | 94.6s | 92.1s | 1.0x |
| logprobs | 43.1s | 45.9s | 1.0x |
| core step | 206.2s | 179.1s | **1.15x** |

An independent same-config rerun reproduced these numbers to within 0.7%
(baseline 68.7s, eagle3 40.9s), so the E2E generation timings are stable
across runs. With generation at 33% of the step, Amdahl predicts
1/(0.67 + 0.33/1.66) = 1.151x - the measured E2E speedup to three decimals.
**Speculation's end-to-end effect is exactly its generation-phase gain
diluted by the generation fraction; training and logprob phases are
untouched.** The generation-phase 1.66x sits below the standalone 2.19x
because the E2E engine is the NeMo-RL container's older vLLM and the
generation timer includes engine wake/sleep overheads around each rollout.
CUDA graphs were active (FULL decode + PIECEWISE mixed captures confirmed,
vLLM 0.20.0); fixed-K always hits the FULL path on 0.20, so the comparison is
fair. DynamicSD cannot yet be enabled inside NeMo-RL - the schedule key needs
vLLM >= 0.24 (plus the ledger #2 crash fix) - and on this 4K-cap recipe our
standalone data predicts no additional gain over fixed-K anyway (no drain
tail forms); it becomes relevant for SD-breakeven workloads (32B-SWE-like)
and 32K+ generation recipes once NeMo-RL's vLLM catches up.

## DynamicSD inside NeMo-RL: engine-level first light

Can `num_speculative_tokens_per_batch_size` run inside NeMo-RL at all? Yes -
**we booted it** (run C4): a triple-patched vLLM 0.25.0 wheel (ZeroDivision
guard, Qwen3MoE added to the V2-runner auto-enable list, torchcodec
exception guard) injected via the job's `SYSTEM_PYDEPS_SITE` overlay, with
the K-schedule passed as a Hydra override. The engine initialized cleanly:
`v0.25.0`, `Using V2 Model Runner`, schedule accepted, zero errors. Two
integration landmines are documented for whoever does this next: NeMo-RL's
tensorboard hparams logger rejects nested-list config values (disable it or
flatten), and a global PYTHONPATH overlay leaks into the Megatron training
worker (transformer_engine cublasLt symbol clash from overlay nvidia libs;
transformers double-registration) - full 10-step A/B/C timing on 0.25 inside
NeMo-RL therefore needs a rebuilt worker venv. A third landmine closed the
last shortcut: even with `use_system_env=false` and the prepared
`RL-dynsd-vllm025` uv lock, generation workers resolve the container-baked
`/opt/ray_venvs` (vLLM 0.20) while only the training venv rebuilds from the
lock (a full TransformerEngine source build, ~1h, for nothing). We then pursued the
forced-rebuild path to its end: lock-hash staleness (uv caches path-dep
hashes; `uv lock --refresh-package` needed after re-patching a wheel),
force-rebuild's rmtree failing on lustre leftovers, a deep-ep source-build
flake (node-dependent CUDA_HOME), and finally a shared-venv write race that
poisoned the worker env (`No module named 'ray'` crash loops). Seven
documented landmines in total; the track was closed there. The definitive
clean path is baking a new NeMo-RL container with vLLM >= 0.24 + the ledger
patches - at which point DynamicSD is a config line. The measured E2E numbers
remain the 0.20-stack fixed-K3 results (1.66x generation, 1.15x step,
replicated to 0.7%), and standalone 0.25 data bounds DynamicSD within a few
percent below fixed-K3 on this recipe. The
30B E2E numbers above (vLLM 0.20 stack, 1.66x generation / 1.15x step for
fixed-K3) remain the reference; standalone 0.25 data predicts dynamic would
land within a few percent of fixed there. Separately, 235B E2E was abandoned
after 5 attempts - cross-node TP8 engine init on this fabric failed under
both SHARP-reservation and SHARP-off NCCL paths; the standalone 235B verdict
(external-drafter SpecDec is a net loss) stands on 3-seed evidence.

## Drafter choice for SWE workloads: single-turn vs agentic multi-turn

Literature says suffix decoding dominates SWE-agent workloads (Snowflake:
2.5x on SWE-Bench, 1.7x over ngram). We tested the shortlist on
Qwen3-30B-A3B at temperature 1.0 in two regimes.

**Single-turn SWE prompts, sync-rollout concurrency (128 seqs):**

| drafter | speedup | AL |
|---|---|---|
| EAGLE3 K=3 | **1.94x** | 2.48 |
| suffix | 0.52x | 2.22 |
| ngram (K8, lookup 5-8) | 0.42x | 3.43 |

Model-free drafting is a heavy net loss at RL batch sizes even with high
acceptance (ngram's AL 3.43 beats EAGLE3's!). Seed repeats confirm the
ranking far outside noise (fixed-K3 1.85x +/- 0.08, suffix 0.50x +/- 0.02,
ngram 0.41x +/- 0.01, n=3). We then falsified the two obvious excuses for
suffix: at low concurrency (8 seqs, the serving regime of the published
numbers) it still loses (0.51x), and under greedy decoding it gets *worse*
(0.34x, AL unchanged at 2.26). The failure is structural on this
model/prompt mix: suffix drafts deep (default tree depth 24) but accepts
only ~1.3 extra tokens per draft, so verify waste dominates at any
temperature or batch size. The published SWE-Bench wins presuppose much
higher copy density (strong cross-request response-cache hits) than
single-turn issue-to-patch generation provides; tighter
`max_spec_factor`/`min_token_prob`/`num_speculative_tokens` tuning might
close some of the gap but was out of scope.

**Agentic multi-turn (teacher-forced replay of real OpenHands SWE
trajectories, 8 copies, matched turns):**

| drafter | speedup | AL |
|---|---|---|
| EAGLE3 K=3 | 0.88x | 1.79 (out-of-distribution on tool-call turns) |
| suffix | 0.50x | **3.10, rising 2.81 -> 3.74 with turn depth** |
| ngram | 0.54x | 1.19 |

The suffix copy-density thesis is confirmed at the acceptance level - and it
still loses, because agentic turns are the wrong shape for speculation:
median assistant turns are tens of tokens (tool calls) while the recorded
prefix grows 3K -> 27K, so turn time is prefill- and overhead-dominated and
the decode segment a drafter can accelerate is tiny. **In agentic SWE
pipelines the binding constraint is not acceptance but the short generation
segments** - consistent with our earlier finding that agentic E2E masks
SpecDec value behind tool execution. Speculation pays in agent loops only
where turns produce long generations (reasoning-heavy steps), which is also
where EAGLE3 (in-distribution) rather than suffix carries the win.

## Real NemoGym environment validation (PR #3243 eval mode)

All prior sections simulate rollout; this one measures the real thing.
NVIDIA-NeMo/RL PR #3243 (`run_grpo_rollout_benchmark.py`) converts a GRPO
recipe into an eval-only run: vLLM async engines plus a NemoGym environment
actor, no training workers, one GB200 node. We ran the SWE1-pivot recipe
(`grpo_qwen3_30ba3b_thinking_swe1.yaml`, Qwen3-30B-A3B-Thinking-2507, 100
val prompts x 4 generations, 32 prompts/step, temperature 1.0) on Lyris with
two stacks: the PR's own vLLM 0.20 pin and a worktree of the validated vLLM
0.25.1 eagle3-fullcg stack with the PR cherry-picked on top. Two integration
fixes were needed: NemoGym subprocess venvs inherit the parent's
`openai==2.44.0` pin, which is unsatisfiable against nemo-gym's
`openai<=2.7.2` (clamped at the injection point in `global_config.py`), and
the capture-size fix below.

Rollout-collection wall (sum of per-batch progress walls, engine init
excluded), vLLM 0.25.1:

| Variant | Wall | vs baseline | Graph coverage (steps / tokens) |
|---|---:|---:|---|
| Baseline (no SD) | 264-279 s | 1.00x | 100% / 100% |
| EAGLE3 K3, default capture | 448-529 s | 0.56x | 52.6% / much lower |
| EAGLE3 K3, dense capture to 512 | 305 s | 0.87x | 100% / 100% |
| K3 + DynamicSD `[[1,8,3],[9,32,2],[33,512,1]]`, dense capture | 336 s | 0.79x | 100% / 100% |

Rewards were 0.185-0.2275 across all variants (noise band), confirming
losslessness. The same workload on vLLM 0.20 gave baseline 524 s - the
0.20 -> 0.25.1 upgrade alone is worth 1.88x, larger than any SpecDec effect.

**The dominant slowdown was a capture cliff inside NeMo-RL, not acceptance.**
The engine resolved `max_cudagraph_capture_size=64`, so speculative verify
steps (uniform-decode shape = BS x (K+1) tokens) fell off the captured path
for BS > 16 while the baseline's BS <= 64 decode stayed fully covered. We
measured this from the PR's `inflight_batch_sizes` telemetry: 47.4% of K3
engine steps - and the large-BS majority of tokens - ran uncaptured.
Passing an explicit dense `cudagraph_capture_sizes` list up to 512 restored
100% coverage and recovered K3 from 0.56x to 0.87x. (An attempt to set
`compilation_config.max_cudagraph_capture_size` directly dies in vLLM
0.25.1 with `TypeError: cannot pickle 'pydantic_core.ArgsKwargs'` - the
explicit list is the working syntax.) Acceptance is not the binding
constraint: the same drafter measures 64.8% acceptance (MAL 2.94) on math
GRPO and MAL ~2.2 on SWEBench standalone.

Two portable lessons. First, **DynamicSD schedules are calibrated to a
capture configuration, not just to a model**: the schedule above was derived
under the broken capture regime (back off to K1 at BS >= 33, where K3 was
eager); once capture covers K3 everywhere, plain fixed K3 beats it (305 vs
336 s). Second, even at 100% coverage SpecDec does not beat no-SD on this
workload: SWE1 rollouts are prefill-heavy (long recorded-trajectory
contexts) with short tool-call decodes at engine BS ~64, so the K+1 verify
FLOPs never amortize - consistent with the BS x K grids and the replay
study. SpecDec pays in this environment only at low per-engine concurrency
or long-decode regimes.

The verdict is Gym-version-invariant. Re-running the trio after bumping the
NeMo Gym submodule from v0.4.0 to latest main (f0c460f, including the
PR #1825 rollout-timeline instrumentation) reproduces the same ordering:
baseline 286 s, K3 + dense capture 319 s (0.90x), DynamicSD 333 s (0.86x),
rewards 0.20-0.22, prefix-cache hits still 0.0%. The binding structure -
22:1 prefill dominance and the concurrent-identical-prompt cache miss -
lives in the NeMo-RL submission pattern and the workload shape, not in Gym.

A recalibration follow-up refutes the obvious rescue. The engine telemetry
shows 52-55% of active engine steps sit at BS 1-8 (the regime where our
grids give K3 a 1.5-2x win) while producing only ~12% of tokens, suggesting
a tail-only schedule `[[1,8,3],[9,16,2],[17,512,0]]` should clamp the
high-BS loss at zero and keep the tail gain (naive ceiling ~1.3x). vLLM
0.25.1 accepts K=0 ranges, the schedule ran - and lost to everything:
374 s vs fixed-K3 305 s and dynsd-K1 336 s. **Batch-size-adaptive K
optimizes per-step throughput, but a barriered sync rollout is a makespan
problem**: the batch drains into the low-BS tail only as fast as its
longest trajectories move, and those trajectories benefit from speculation
during the high-BS phase too. Turning speculation off at high BS delays the
very transition the schedule was waiting for. The right adaptive axis for
sync rollout is per-trajectory (length/age-aware, as in DAS's speculation
policy), not per-batch-size. A cross-run suffix-match oracle on the SWE1
outputs (independent temp-1.0 runs of the same 100 prompts) caps the DAS
corpus-drafter ceiling at ~1.15x token rate for this single-step env - the
rollout-corpus ingredient of DAS needs true multi-turn copy density (our
replay measured suffix AL 3.10 there) to pay.

## True agentic SWE2 on GB200: pipeline built, timeline decomposed

The SWE1 verdict does not extend to true multi-turn agentic SWE, and we can
now say so with measurements instead of extrapolation. We stood up the full
SWE2 stack (OpenHands agents inside per-instance SWE-bench apptainer
containers, driven through NemoGym and PR #3243's eval mode) on Lyris GB200
- previously assumed impossible without x86 infrastructure. Seven distinct
failure layers had to be root-caused; the load-bearing one was a single
hardcoded `jq-linux-amd64` download in Gym's OpenHands setup script that
made the entry script kill its tmux pane on aarch64, masquerading as a
universal command-timeout. With arm64 SIF prepull, a /dev/fuse bind for
apptainer-in-enroot, a synthetic /swe_util layer (official arm64 images
lack the OpenHands bake), and openai-2.7.2 tool-schema normalization, 3/3
SWE-bench_Verified rollouts complete end-to-end (5-8 turns, all producing
patches, eval harness scoring).

The rollout-timeline instrumentation (Gym PR #1825 + nv-OpenHands PR #19)
then gives the decomposition SWE1 could only approximate. Over 3 instances
(agent + eval wall 434 s): **LLM generation 45.7%**, agent init 24.5%,
framework overhead between turns 17.2%, final evaluation 11.2%, tool
execution 0.5%. Per-turn LLM latency is p50 6.6 s / p90 28.4 s (n=18) at
~2.3K completion tokens per turn - decode-heavy, the opposite regime from
SWE1's 22:1 prefill dominance. Amdahl over the 45.7% LLM share puts the
SpecDec E2E ceiling at ~1.21x with the measured 1.63x eagle3 generation
speedup (1.27x at the standalone-SWE 1.85x) - a positive headroom, in
contrast to SWE1's measured 0.87x. Caveat: 3 instances x <=8 turns, single
generation each; acceptance on SWE2 outputs not yet measured directly.

## Do our numbers match the upstream DynamicSD PRs?

Yes, when compared apples-to-apples. PR #32374 reports +7.5% over no-SD at
BS 128 and +23% over fixed-K SD at BS 128/256 (MT-Bench serving, Llama-8B
class); PR #45953 motivates per-K full graphs. We measure: +8% over no-SD in
the saturation regime (32B SWE dynamic 1.08x), +4-50% over fixed-K where
fixed-K is miscalibrated for the batch size (32B SWE, 235B), and -13%
dynamic step wall from per-K FULL graphs. The apparent disagreement in our
headline math rows comes from the reference: the PRs compare against a fixed
K that hurts at high batch sizes (compare our fixed-K5, which a naive tuner
might pick from small-batch profiling: dynamic beats it ~3x), while we
compare against the profile-optimal fixed K. **DynamicSD's benefit
presupposes that the optimal K actually moves with runtime batch size** -
true for fluctuating serving traffic and for workloads whose K-optimum
crosses zero, not for cap-truncated sync rollouts that pin the batch at one
size.

## Key takeaway

**On these RL-rollout shapes, EAGLE3 with a well-chosen fixed K is the
workhorse (up to 2.0x per-step wall time on Qwen3-30B-A3B at temperature 1.0),
and DynamicSD's value is asymmetric: it cannot beat a fixed K that already
sits on the optimum, but it converts speculation from a liability into a gain
where the optimum crosses zero (Qwen3-32B SWE at BS 256).** The derived
schedules reproduce the Cohere-reported structure (dense = monotonically
decreasing K, MoE = non-monotonic), and the practical lesson is that schedule
quality is bounded by profiling fidelity: capture-mode cliffs, deep-context
drift, and MoE dispatch costs all have to be encoded, or the schedule
confidently picks poisoned points.

Data: `data/` (profile grid, rollout summaries, drain curves). Plots:
`plots/`. Live page: `docs/dynamic_sd_sync_rollout_results_latest.html`.
Every vLLM patch/change and its measured perf impact is tracked in
[`../PATCH_LEDGER.md`](../PATCH_LEDGER.md) for upstream PR conversion.
Open items: 235B/40K dynamic regression diagnosis, 3 transient-node retries,
suffix-decoding composition.
