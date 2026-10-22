# SpecDec pays off where decode sets the step time, and only there

## Why This Experiment

The DFlash/DSpark draft co-training stack (11 PRs) puts speculative decoding with trained drafters into NeMo-RL. The question is how much of the GRPO step this saves on latest main, with the shipped performance recipes and only the SpecDec settings added, on Qwen3-30B-A3B, Nemotron 3.5 Nano, Qwen3.5-35B-A3B and Qwen3-235B-A22B, for math GRPO and for SWE RL. We also want the best K per drafter, because block size 8 sets the largest K, not the best one.

Workspace: branch `sna/draft-cotrain-specdec-perf-20261020` on github.com/seonjinn/RL. The code is at `d3ab5836f3` (upstream main `6fcd05f01c` merged). Later commits on the branch add only this report. It holds the merged PRs #3701, #3703, #3706, #3707, #3711 and #3715 through main, and the open PR heads #3723 (`bfbd6d7b21`), #3726 (`c39e31e538`), #3744 (`5dfe9ebe06`), #3750 (`7da56c0ebb`) and #3757 (`5a85b30b05`). Two fixes from these runs are on the branch: the draft gate accepts any TP size (`b2264932df`), and the venv build retries after a race between nodes (`844da82424`). Live dashboard: https://claude.ai/artifact/GsAcZhvXmCfRq8ETRWutd1.

Every run: OCI-HSG GB200, nightly container, ray.sub, vLLM `moe_backend: flashinfer_trtllm`, CUDA graphs on, 20 steps, no checkpoints, W&B project `sna-async-grpo-gb200`. Each arm is compared with the control of its own cohort over the steps 3-20 that both logged (SWE: 1-20). Speedup is control time over arm time.

---

## The ceiling decides the gain

If generation took zero time, the step would shrink by the control's generation share. That gives a generation-free E2E ceiling for each workload. The table lists the best arm per model, workload, mode and drafter, and the share of the ceiling saving it reached.

| Model | Workload | Mode | Drafter | Best K | Gen share of control | E2E ceiling | E2E | Gen | Accept len | Share of ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Q30 | 4K | sync | DFlash | 7 | 0.38 | 1.60x | 1.21x | 1.74x | 3.86 | 0.47 |
| Q30 | 4K | sync | DSpark | 6 | 0.38 | 1.61x | 1.21x | 1.81x | 3.85 | 0.46 |
| Q30 | DAPO 4K | sync | DFlash | 7 | 0.35 | 1.53x | 1.15x | 1.63x | 3.85 | 0.38 |
| Q30 | DAPO 4K | sync | DSpark | 8 | 0.35 | 1.53x | 1.14x | 1.55x | 4.14 | 0.35 |
| Q30 | DAPO 40K | sync | DFlash | 5 | 0.61 | 2.57x | 1.52x | 2.28x | 3.28 | 0.56 |
| Q30 | DAPO 40K | sync | DSpark | 5 | 0.61 | 2.57x | 1.52x | 2.27x | 3.19 | 0.56 |
| Q30 | 32K | sync | DSpark | 8 | 0.81 | 5.33x | 1.55x | 1.75x | 3.57 | 0.44 |
| Q30 | 32K | sync | DFlash | 7 | 0.81 | 5.33x | 1.30x | 1.39x | 2.88 | 0.28 |
| Q30 | 32K | async | DSpark | 8 | 0.43 | 1.76x | 1.60x | 3.13x | 3.56 | 0.87 |
| Q30 | 32K | async | DFlash | 7 | 0.43 | 1.76x | 1.40x | 2.05x | 2.78 | 0.66 |
| Q30 | DAPO 32K | sync | DSpark | 8 | 0.82 | 5.41x | 1.55x | 1.77x | 3.64 | 0.43 (12 paired) |
| Q30 | DAPO 32K | sync | DFlash | 7 | 0.82 | 5.41x | 1.33x | 1.43x | 2.99 | 0.30 (12 paired) |
| Q30 | DAPO 32K | async | DFlash | 7 | 0.54 | 2.17x | 1.41x | 2.16x | 2.98 | 0.54 |
| Q30 | DAPO 32K | async | DSpark | 8 | 0.54 | 2.17x | 1.38x | 2.08x | 3.61 | 0.51 (12 paired) |
| Q235 | 4K | sync | DFlash | 5 | 0.52 | 2.09x | 1.29x | 1.62x | 2.78 | 0.43 |
| Q235 | 4K | sync | DSpark | 6 | 0.52 | 2.09x | 1.28x | 1.63x | 2.97 | 0.42 |
| Q235 | 4K | async | DFlash | 7 | 0.28 | 1.38x | 1.32x | 2.64x | 2.86 | 0.88 |
| Q235 | 4K | async | DSpark | 8 | 0.28 | 1.38x | 1.32x | 2.41x | 3.02 | 0.87 |
| N35 | 4K | sync | DSpark | 4 | 0.49 | 1.97x | 1.19x | 1.55x | 3.27 | 0.32 |
| N35 | 4K | sync | DFlash | 3 | 0.49 | 1.97x | 1.17x | 1.36x | 2.43 | 0.30 |
| Q35 | 4K | sync | DSpark | 4 | 0.18 | 1.22x | 1.05x | 1.24x | 2.66 | 0.24 |
| Q35 | 4K | sync | DFlash | 3 | 0.18 | 1.22x | 0.94x | 0.74x | 2.59 | below 0 |
| Q30, N35, Q35 | 4K, DAPO 4K | async | both | 7-8 | 0.00-0.01 | 1.00-1.01x | 0.99-1.04x | 0.91-1.11x | 3.1-4.1 | n/a |

![Best-K E2E speedup per workload against its generation-free ceiling. The gray bar runs from 1.0x to the ceiling.](plots/ceiling.png)

**The gain follows the generation share, not the drafter.** Long outputs (Q30 32K, DAPO 32K and DAPO 40K) and the large model (Q235) give 1.3-1.6x E2E. 4K sync math gives 1.14-1.21x on Q30 and N35, and only 1.05x on Qwen3.5-35B, where generation is 18% of the step. Async 4K math is train-bound: exposed generation is about 1% of the step, so there is nothing left to save, and the measured 0.99-1.04x is noise around 1.0x. Async runs with a small exposed share (Q235 4K at 0.28, Q30 32K at 0.43) reach 0.66-0.88 of their ceiling, because training hides the rest of generation. DAPO 32K async exposes 54% of the step as generation and reaches about 0.5. Sync runs reach 0.3-0.6, because a 1.4-2.3x generation speedup cannot remove all of generation. On 32K sync DSpark beats DFlash (1.55x against 1.30-1.33x) with acceptance length 3.6 against 2.9-3.0. On DAPO 32K async the two tie. Qwen3.5-35B DFlash is the one arm below 1.0x: its hybrid GDN layers give 46 KV cache groups, and generation gets slower (0.74x). Every arm in this table runs on stock vLLM; the GDN metadata-reuse backport lifts Q35 DFlash K3 to 1.02x, but it is not part of the SpecDec settings.

Training, logprob and refit times match the control within 1-4% in every frozen arm, as they should: a frozen drafter adds no training work, and the drafter refit is small.

---

## Best K sits 2-4 below the block size

| Model | Mode | DFlash K tried | Best | DSpark K tried | Best |
|---|---|---|---|---|---|
| Q30 | sync | 3, 5, 7 | 5 (1.20x; K3 1.19x, K7 1.19x) | 4, 6, 8 | 6 (1.21x; K4 1.20x, K8 1.19x) |
| Q30 | async | 3, 5, 7 | 7 (1.04x, noise) | 4, 6, 8 | 8 (1.04x, noise) |
| Q235 | sync | 3, 5, 7 | 5 (1.29x; K7 1.26x; K3 pending) | 4, 6, 8 | 6 (1.28x; K8 1.27x; K4 pending) |
| Q235 | async | 7 (3, 5 pending) | 7 (1.32x) | 8 (4, 6 pending) | 8 (1.32x) |
| N35 | sync | 3, 7 | 3 (1.17x; K7 1.10x) | 4, 8 | 4 (1.19x; K8 1.10x) |
| Q35 | sync | 3, 7 | 3 (0.94x; K7 0.90x) | 4, 8 | 4 (1.05x; K8 0.97x) |

Each row compares K inside one cohort, so all its K share a control. Q30 uses the K-sweep cohort (the first cohort gave DFlash K7 1.22x and DSpark K8 1.20x). Q35 uses the stock-vLLM cohort.

![E2E speedup by K for 4K sync math. N35 joins two cohorts (dashed).](plots/k_sweep.png)

Per-token acceptance at temperature 1.0 is 0.25-0.41, so the last draft slots rarely land but still cost verify tokens. On Q30 the K curve is flat (within 2% from K3 to K8). On Q235 the middle K wins by 2-3%. On N35 and Q35 the smallest K wins by 4-9%. On Q35 only DSpark K4 beats the control; DFlash slows the step at both K. The N35 K3/K4 and K7/K8 arms ran in different cohorts, so their gap has about 2% of baseline drift in it. **The block-size K is never the clear winner. Use a K 2-4 below the block size.**

---

## SWE RL: measure model calls, not step time

| Model | Arms | Model-call speedup | E2E | Accept len | Steps |
|---|---|---|---|---|---|
| Q30 small (3 nodes) | frozen DFlash K7 / DSpark K8 | 1.05x / 1.24x | 1.06x / 1.04x | n/a | 20 |
| Q30 scale (9 nodes) | frozen DFlash K7 / DSpark K8 | 1.03x / 1.25x | 1.07x / 1.04x | 1.40 / 1.87 | 20 |
| Q30 scale2 (9 nodes) | frozen DFlash K3 / DSpark K4 | 1.00x / 1.30x | 0.90x / 0.90x | 1.28 / 1.76 | 20 |
| Q30 single controller (V2) | frozen DSpark K4 | 1.30x | 1.17x | 2.12 | 20 |
| Q30 scale2 online TP4 | DSpark K4 lr 1e-4 | 1.30x | 1.05x | 1.74 | 18 |
| Q235 (32 nodes) | frozen DFlash K7 | 1.14x | 1.16x | 1.38 | 11 (all baseline steps) |

A SWE step is mostly tool execution and agent tail time, so E2E moves with tool time, not with decode. The scale2 frozen arms show it: DSpark K4 decodes model calls 1.30x faster but the step is 0.90x, because tool bursts in steps 4-8 hit those arms and not the control. **DSpark gives a steady 1.24-1.30x on model calls; DFlash on the Thinking drafter gives 1.00-1.05x on Q30 and 1.14x on Q235**, with acceptance length 1.3-1.4 against 1.7-2.1 for DSpark. A Q235 model call spends more of its time in decode, so the same acceptance gives more. SWE runs skip the logprob pass, so the dashboard leaves the logprob columns of SWE rows empty.

---

## Online co-training

The first online arms never trained the drafter: on the Megatron path the draft parameters inherited the policy lr (3e-7). With the draft lr at 1e-4 and the policy at TP1, ours and the teammate's implementation give the same result.

| Arm (Q30 TP1) | Impl | Gen | Step | Accept len | Paired |
|---|---|---|---|---|---|
| DFlash K7 | teammate | 1.71x | 1.15x | 3.62 | 18 |
| DFlash K7 | ours | 1.68x | 1.14x | 3.62 | 18 |
| DSpark K8 | teammate | 1.67x | 1.13x | 4.00 | 18 |
| DSpark K8 | ours | 1.60x | 1.13x | 4.01 | 18 |

The DSpark K8 generation ratios differ (1.60x against 1.67x) with the same acceptance length, because the two baselines differ: arm generation is 38.3 s against 37.4 s and baseline generation 61.2 s against 62.3 s, inside the 1.3-2.7% drift between jobs. Drafter training adds 9-12% to policy training time in both. SWE online arms OOMed at TP2, which the old draft gate forced. At the baseline TP4 layout DSpark K4 passes. Over 18 paired steps it keeps the frozen model-call gain (1.30x), and training takes 13% longer than the control and 9% longer than the frozen arm.

---

## Suspicious outcomes

| Outcome | Why it is suspicious | Status |
|---|---|---|
| Async 4K arms show 1.03-1.04x E2E | Exposed generation is about 1% of the step, so the ceiling is 1.01x. The gain is noise or a side effect, not SpecDec. | explained |
| SWE scale2 frozen arms 0.90x E2E with 1.30x model calls | Tool bursts in steps 4-8 of the arms, not the control | explained (tool drift) |
| SWE online DSpark K4 TP4 1.05x E2E, above the frozen arm (0.90x) | Model calls are 1.30x in both and training is 9% longer than frozen, so the E2E gap is tool time in different windows | explained, use model call 1.30x |
| SWE logprob 1.13-1.33x longer in arms | SWE skips the logprob pass, and the control timer logs 0.5-1.2 ms per step | fixed: the dashboard leaves a ratio empty when the control phase is under 0.5 s |
| DSpark draft loss differs between implementations by about 0.26 | Teammate TV term is sum\|p-q\|, ours is 0.5*sum\|p-q\| | explained, values not comparable |
| Two baselines 1.3-2.7% apart | Same config, different jobs | compare inside a cohort only |
| SC (V2) single pair | One baseline and one arm | needs a repeat |
| Q235 SWE fits 11 steps in 4 h; E2E 1.16x above model call 1.14x | Async step times alternate 60-80 s and 870-1430 s. Steps 1-6 give 1.02x and steps 7-11 give 1.30x, so part of the E2E gain is rollout wait | use model call 1.14x |
| 8150978 step-1 generation 238x | Async step 1 pairs a buffered step with a fresh one | excluded by the paired window |
| DAPO 32K async baseline waits at steps 3 and 5 (412 s exposed generation against 2 s) | Its slower generation has not caught up yet. From step 7 all runs alternate in phase. Steps 7-20 give DFlash 1.43x against 1.41x over steps 3-20, and steps 7-14 give 1.38x for both drafters | explained, gain holds in steady state |
| DAPO 32K sync pairs on 12 steps | The baseline (8146971) hit the 4 h limit at step 14 | above the 8-step floor |
| Q35 DFlash slower than control at K3 and K7 | 46 KV cache groups from the GDN layers; the GDN metadata-reuse backport gives 1.4-1.8x on rollout and lifts K3 to 1.02x | root cause known, fix not in stock vLLM |
| Q235 sync DFlash K3 and DSpark K4 died at startup | megatron-core editable build race between nodes of one job | fixed in 844da82424, reruns pending |

---

## Key Takeaway

**SpecDec speeds up GRPO by roughly the share of the step that is exposed decode: 1.3-1.6x E2E for long outputs and Q235, 1.14-1.21x for 4K sync math on Q30 and N35, 1.05x on Qwen3.5-35B, and nothing for train-bound async 4K.** DFlash and DSpark tie on 4K math and on DAPO 32K async. DSpark wins on 32K sync (1.55x against 1.30-1.33x) and on SWE model calls. K below the block size is as good or slightly better, so the block-size K is a safe default but not the best. The next lever is not a better drafter on 4K async math, which has no generation left to hide, but long-output and agentic workloads where decode or model calls set the step time.