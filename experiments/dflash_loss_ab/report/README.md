# TV Loss Does Not Help DFlash: kl_div Wins the 4-Way, and the 235B Pipeline Works

## Why this experiment

Every off-the-shelf drafter we tested (eagle3, PARD, public DFlash) loses most of its
acceptance on agentic SWE rollouts, so we are training our own DFlash drafter for
Qwen3-235B-A22B-Thinking. Before spending compute on the main run, two questions had
to be answered cheaply: (1) which training loss maximizes acceptance - the bebop-mtp
paper claims TV loss beats cross-entropy, and spec-sampling theory says acceptance
equals 1 - TVD, so the claim is plausible; (2) does the speculators online-training
pipeline work at 235B scale on GB200 at all. Both were answered with 10K UltraChat
screening runs (~25 min per arm).

## The 4-way loss comparison

Qwen3-30B-A3B-Thinking-2507 target, identical data (UltraChat 10K, seq 8192), 5 epochs,
identical hyperparameters; only `--loss-fn` varies. Final-epoch validation:

| Loss | val EAL | pos-1 acc | pos-2 acc | pos-3 acc |
|---|---|---|---|---|
| **kl_div** (speculators default) | **1.211** | 0.702 | 0.488 | 0.351 |
| ce (DFlash paper) | 1.188 | 0.695 | 0.481 | 0.344 |
| nla (-log acceptance) | 1.180 | 0.692 | 0.479 | 0.342 |
| tv (bebop-mtp claim) | **0.571** | 0.417 | 0.293 | 0.211 |

Two findings, one expected and one not:

1. **Raw TV loss is catastrophic, not merely worse.** EAL halves. This confirms the
   vanishing-gradient analysis: TV = 1 - sum(min(p,q)) has zero gradient on every token
   where one distribution dominates, so most of the batch stops teaching the drafter.
   The bebop-mtp result does not transfer to DFlash as a drop-in loss swap.
2. **Among healthy losses, the choice barely matters.** kl_div, ce, and nla land within
   3% of each other, with kl_div marginally ahead. Even nla - which fixes TV's gradient
   and directly optimizes log-acceptance - does not beat plain distribution matching.
   At this scale the binding constraint is data and capacity, not the loss surface.

**Decision: kl_div for the 235B main training.** Caveats: single seed, 5 epochs,
UltraChat-only validation; the 2-3% gaps between healthy losses are within noise, but
the tv collapse is far outside it.

## 235B pipeline screening

`dflash_235b_screen.sbatch`: 2 GB200 nodes - node 0 serves the 470 GB target with TP4
and streams hidden states (layers 1/23/46/68/91 of 94) through the FileBackend on
lustre; node 1 trains the drafter on 4 GPUs, requesting on-demand generation via
`--vllm-endpoint`. Completed end-to-end in 33 minutes (7 min warm-JIT server boot,
26 min for 5 epochs including online generation), rc=0, per-epoch checkpoints saved.

| Target | val EAL | pos-1 acc | arm wall time |
|---|---|---|---|
| 30B-Thinking (kl_div) | 1.211 | 0.702 | ~24 min |
| 235B-Thinking (kl_div) | 1.128 | 0.681 | ~26 min |

The 235B drafter lands slightly below the 30B one at equal data - expected, since the
same-size drafter is modeling a harder target distribution - but the pipeline cost is
nearly identical thanks to TP4 serving. Extrapolating 26 min / 10K samples, a
DFlash-paper-scale run (~800K samples) is roughly 35 GPU-hours x 8 GPUs: one
`batch_long` job, no offline hidden-state storage (online mode used zero disk beyond
checkpoints).

## Real-rollout 4-way: 10% SWE data converts a useless drafter into a 1.29x one

The leakage concern above made the val-EAL gap (+54% for the SWE mix) untrustworthy on
its own, so both 10K drafters were dropped into the real NemoGym SWE2 rollout on Lyris
(GB200, vLLM 0.25, K=9, identical overrides to the public-drafter runs):

| Drafter | SWE2 rollout tok/s | vs baseline |
|---|---|---|
| none (baseline) | 209 | 1.00x |
| public DFlash (800K generic) | 310 | 1.48x |
| ours, UltraChat 10K | 206 | 0.99x |
| ours, UltraChat + 10% SWE trajectories 10K | **269** | **1.29x** |

Three conclusions. First, **a small generic drafter is worthless on agentic SWE** - the
UltraChat-only drafter lands at baseline parity: whatever it accepts is exactly eaten by
SpecDec overhead. Second, **swapping 10% of the same 10K budget to SWE trajectories
buys +31% rollout throughput** (206 -> 269) - the domain-data hypothesis survives the
real benchmark, not just the leaky val split. Third, the mix drafter already recovers
~60% of the public drafter's gain with 80x less data, and the public drafter had zero
SWE data - which bounds how much generic scale alone can matter.

### Contamination split: how much of the gain is memorization?

The mix-own drafter was trained on trajectories from the *same three astropy instances*
used in the eval - a contamination the user flagged. A third arm isolates it:
**mix-public** replaces our trajectories with SWE-smith public trajectories (synthetic
tasks on disjoint repos, astropy filtered out, SWE-agent scaffold), same 10% share and
budget, evaluated on the same (now fully unseen) astropy instances:

| Drafter (all 10K, kl_div, K9) | tok/s | vs baseline |
|---|---|---|
| UltraChat only | 206 | 0.99x |
| + 10% public SWE (unseen instances, foreign scaffold) | 220 | 1.05x |
| + 10% own rollouts (seen instances, matched scaffold) | 269 | 1.29x |

The clean cross-domain generalization effect is +7% (206 -> 220); the remaining +49
tok/s of the contaminated arm comes from the combination of instance familiarity and
**scaffold match** (our trajectories share the deployment's OpenHands system prompt and
tool-call format; SWE-smith uses the SWE-agent scaffold). For general SWE serving
claims, 220 is the honest number at this scale. For the RL-rollout use case both
components are legitimately available - GRPO re-rolls the same instance pool with the
same scaffold every step, so "contamination" is exactly the adaptation that deployment
provides.

## K re-tuning and the ratio sweep: match beats mass

Two follow-up sweeps completed the picture. First, re-tuning K for our drafter
(the public drafter's optimum was K=9): K3=311, **K5=313**, K7=270, K9=269 tok/s.
Our 10K drafter's acceptance decays faster with draft position, so shorter drafts cut
wasted draft cost - at K5 the 58-conversation matched-scaffold drafter **beats the
public 800K drafter's best (313 vs 310, 1.50x vs baseline)**.

Second, scaling the contamination-free SWE share (SWE-smith, SWE-agent scaffold) at a
fixed 10K budget, evaluated on the astropy OpenHands rollout at K9: 0% -> 206, ~3% ->
220, 50% -> 213. **Foreign-scaffold SWE data saturates by ~3%** - even though val EAL
kept climbing (1.211 -> 1.238 -> 2.680, and 3.274 at 100% SWE), the rollout number did
not move. The val metric inflates in-distribution: a textbook case of a fit-time metric
decoupling from downstream impact. Meanwhile 58 matched-scaffold conversations (~3%
share) moved the rollout from 206 to 269@K9 / 313@K5.

**The dominant axis is deployment match (OpenHands scaffold + instance pool), not SWE
token mass.** Also validated: speculators resume-from-checkpoint works across chained
jobs (`Resuming training on epoch 5` after a dependency-chained restart), so 4h-wall
chunked training and the 7d `batch_long` partition are both viable for the 235B run.

## Key takeaway

**Loss engineering is not where 235B drafter acceptance will come from - kl_div is
already at the practical ceiling of this axis, and the acceptance-aligned alternatives
(tv, nla) either collapse or tie.** The lever that remains is training data that matches
the deployment distribution (SWE trajectories at 16K context), which is exactly the
axis where every public drafter fails. The validated 2-node online pipeline makes that
main run a single job away.
