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

## Key takeaway

**Loss engineering is not where 235B drafter acceptance will come from - kl_div is
already at the practical ceiling of this axis, and the acceptance-aligned alternatives
(tv, nla) either collapse or tie.** The lever that remains is training data that matches
the deployment distribution (SWE trajectories at 16K context), which is exactly the
axis where every public drafter fails. The validated 2-node online pipeline makes that
main run a single job away.
