# SpecDec pays off where decode sets the step time, and only there

## Why This Experiment

The DFlash/DSpark draft co-training stack (11 PRs) adds speculative decoding with trained drafters to NeMo RL. This experiment measures how much of the GRPO step this saves on latest main. Each run uses the shipped performance recipe and adds only the SpecDec settings. The models are Qwen3-30B-A3B, Nemotron 3.5 Lightning 30B-A3B, Qwen3.5-35B-A3B and Qwen3-235B-A22B. The workloads are math GRPO and SWE RL. We also find the best K for each drafter. Block size 8 sets the largest K, but not the best K.

Workspace: branch `sna/draft-cotrain-specdec-perf-20261020` on github.com/seonjinn/RL. Live dashboard: https://claude.ai/artifact/GsAcZhvXmCfRq8ETRWutd1. Static run tables: [index.html](index.html), written by `report_html.py` from [data/runs.json](data/runs.json). `build_runs.py` writes runs.json from the W&B history CSVs.

Code state:

- All runs used the code at `d3ab5836f3`, which merges upstream main `6fcd05f01c`.
- This code holds the merged PRs #3701, #3703, #3706, #3707, #3711 and #3715 through main. It also holds the open PR heads #3723 (`bfbd6d7b21`), #3726 (`c39e31e538`), #3744 (`5dfe9ebe06`), #3750 (`7da56c0ebb`) and #3757 (`5a85b30b05`).
- Two fixes from these runs are on the branch. The draft gate accepts any TP size (`b2264932df`). The venv build tries again after a race between nodes (`844da82424`).
- After the runs, the branch merged upstream main `d643cc4c11` (`f5632d3d95`). This merge adds only MSC checkpoint storage (#4743). `nemo_rl.utils.checkpoint` now imports `multistorageclient` at module load. The vLLM venvs of the 2026-10-20 nightly do not have this package, so this code needs a container built after #4743.
- On 2026-10-22 the open PRs moved to main `d643cc4c11` with review fixes. The new heads are #3723 (`6542b23aac`), #3726 (`c608fd49c9`), #3744 (`9caf100c8b`), #3750 (`725b740614`) and #3757 (`c9c181ba18`). The branch merged them at `30cdfe9e1d`. Then it merged main `f0639c6042`, a one-line CONTRIBUTING.md fix.
- On 2026-10-23 a check of branch head `b7fdc46a93` (main `b2de3de156`) found all 11 PRs. The merge commits of #3701, #3703, #3706, #3707, #3711 and #3715 are ancestors. The heads of #3723, #3726, #3744 and #3750 are ancestors. The #3757 head `6df57d6039` is a squash, so its commits are not ancestors. Of the 114 files that it changes, 99 match the branch exactly. Branch fixes from these runs cause 14 of the other 15 differences: DFlash sliding-window attention (`03a3bbc896`, `4b8e538eb7`), the bare cuda device check (`333b3c6ce7`), draft optimizer warmup (`dd209da65b`), async metric windows (`8f1959fe65`) and the HybridEP packed length under PP (`a857fed389`). Upstream #4719 in main removes one Dynamo test key and causes the last difference. Thus the branch holds all of #3757 plus these fixes.
- These changes do not touch the measured code paths, so the runs stay valid.

Run settings: OCI-HSG GB200, nightly container, ray.sub, vLLM `moe_backend: flashinfer_trtllm`, CUDA graphs on, 20 steps, no checkpoints, W&B project `sna-async-grpo-gb200`. We compare each arm with the control of its own cohort. The comparison uses the steps 3-20 that both runs logged (SWE: steps 1-20). Speedup is control time divided by arm time.

---

## Workloads and datasets

A workload label names the prompt dataset and the maximum sequence length (prompt plus response). OpenMathInstruct-2 and DAPO-Math-17k are math datasets. The reward checks the final answer. In SWE RL, an agent fixes a repository issue in a container. The task tests give the reward. Q235 math runs use the 8K length of their recipe, not 4K.

| Workload | Dataset | Max length (tokens) | Models | Config |
|---|---|---|---|---|
| OpenMath 4K | OpenMathInstruct-2 | 4,096 | Q30, N35, Q35 | `grpo-qwen3-30ba3b-4n4g.yaml`, `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml` |
| OpenMath 8K | OpenMathInstruct-2 | 8,192 | Q235 | `grpo-qwen3-235b-16n4g.yaml` (sync), `grpo-qwen3-235b-32n4g-async-1off.yaml` (async) |
| OpenMath 32K | OpenMathInstruct-2 | 32,768 | Q30 | `configs/qwen3_30ba3b/seq32k.yaml` |
| DAPO 4K | DAPO-Math-17k | 4,096 | Q30 | `configs/qwen3_30ba3b/dapo.yaml` |
| DAPO 32K | DAPO-Math-17k | 32,768 | Q30 | `configs/qwen3_30ba3b/dapo_seq32k_*.yaml` |
| DAPO 40K | DAPO-Math-17k | 40,960 | Q30 | `configs/qwen3_30ba3b/dapo40k.yaml` |
| SWE RL | SWE blend `balanced_language.jsonl` (SWE-Gym, SWE-bench, SWE-rebench tasks), OpenHands agent in NeMo Gym | 131,072 | Q30 and Q235 Thinking-2507 | `examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml` |

Models: Q30 is Qwen3-30B-A3B, Q235 is Qwen3-235B-A22B, N35 is Nemotron-3.5-Lightning-30B-A3B, Q35 is Qwen3.5-35B-A3B.

---

## The ceiling decides the gain

Assume that generation takes zero time. Then the step time decreases by the generation share of the control. This gives a generation-free E2E ceiling for each workload. The table shows the best arm for each model, workload, mode and drafter. The last column shows the fraction of the ceiling gain that the arm gets.

| Model | Workload | Mode | Drafter | Best K | Gen share of control | E2E ceiling | E2E | Gen | Accept len | Share of ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Q30 | OpenMath 4K | sync | DFlash | 7 | 0.38 | 1.60x | 1.21x | 1.74x | 3.86 | 0.47 |
| Q30 | OpenMath 4K | sync | DSpark | 6 | 0.38 | 1.61x | 1.21x | 1.81x | 3.85 | 0.46 |
| Q30 | DAPO 4K | sync | DFlash | 7 | 0.35 | 1.53x | 1.15x | 1.63x | 3.85 | 0.38 |
| Q30 | DAPO 4K | sync | DSpark | 8 | 0.35 | 1.53x | 1.14x | 1.55x | 4.14 | 0.35 |
| Q30 | DAPO 40K | sync | DFlash | 5 | 0.61 | 2.57x | 1.52x | 2.28x | 3.28 | 0.56 |
| Q30 | DAPO 40K | sync | DSpark | 5 | 0.61 | 2.57x | 1.52x | 2.27x | 3.19 | 0.56 |
| Q30 | OpenMath 32K | sync | DSpark | 8 | 0.81 | 5.33x | 1.55x | 1.75x | 3.57 | 0.44 |
| Q30 | OpenMath 32K | sync | DFlash | 7 | 0.81 | 5.33x | 1.30x | 1.39x | 2.88 | 0.28 |
| Q30 | OpenMath 32K | async | DSpark | 8 | 0.43 | 1.76x | 1.60x | 3.13x | 3.56 | 0.87 |
| Q30 | OpenMath 32K | async | DFlash | 7 | 0.43 | 1.76x | 1.40x | 2.05x | 2.78 | 0.66 |
| Q30 | DAPO 32K | sync | DSpark | 8 | 0.82 | 5.41x | 1.55x | 1.77x | 3.64 | 0.43 (12 paired) |
| Q30 | DAPO 32K | sync | DFlash | 7 | 0.82 | 5.41x | 1.33x | 1.43x | 2.99 | 0.30 (12 paired) |
| Q30 | DAPO 32K | async | DSpark | 8 | 0.54 | 2.17x | 1.46x | 2.46x | 3.62 | 0.58 |
| Q30 | DAPO 32K | async | DFlash | 7 | 0.54 | 2.17x | 1.41x | 2.16x | 2.98 | 0.54 |
| Q235 | OpenMath 8K | sync | DFlash | 5 | 0.52 | 2.09x | 1.29x | 1.62x | 2.78 | 0.43 |
| Q235 | OpenMath 8K | sync | DSpark | 6 | 0.52 | 2.09x | 1.28x | 1.63x | 2.97 | 0.42 |
| Q235 | OpenMath 8K | async | DSpark | 4 | 0.27 | 1.37x | 1.39x | 3.03x | 2.70 | 1.06 |
| Q235 | OpenMath 8K | async | DFlash | 3 | 0.27 | 1.37x | 1.36x | 2.94x | 2.46 | 0.99 |
| N35 | OpenMath 4K | sync | DSpark | 4 | 0.49 | 1.97x | 1.19x | 1.55x | 3.27 | 0.32 |
| N35 | OpenMath 4K | sync | DFlash | 3 | 0.49 | 1.97x | 1.17x | 1.36x | 2.43 | 0.30 |
| Q35 | OpenMath 4K | sync | DSpark | 4 | 0.18 | 1.22x | 1.05x | 1.24x | 2.66 | 0.24 |
| Q35 | OpenMath 4K | sync | DFlash | 3 | 0.18 | 1.22x | 0.94x | 0.74x | 2.59 | below 0 |
| Q30, N35, Q35 | OpenMath 4K, DAPO 4K | async | both | 7-8 | 0.00-0.01 | 1.00-1.01x | 0.99-1.04x | 0.91-1.11x | 3.1-4.1 | n/a |

![Best-K E2E speedup per workload against its generation-free ceiling. The gray bar runs from 1.0x to the ceiling.](plots/ceiling.png)

**The gain follows the generation share, not the drafter.**

- Long outputs (Q30 OpenMath 32K, DAPO 32K and DAPO 40K) and the large model (Q235) give 1.3-1.6x E2E.
- OpenMath 4K sync gives 1.14-1.21x on Q30 and N35. It gives only 1.05x on Qwen3.5-35B, where generation is 18% of the step.
- Async OpenMath 4K is train-bound. Exposed generation is about 1% of the step, so SpecDec has nothing to save. The measured 0.99-1.04x is noise around 1.0x.
- Async runs with a small exposed share (Q235 OpenMath 8K at 0.27, Q30 32K at 0.43) get 0.66-1.06 of their ceiling. Training hides the rest of generation. Q235 async DFlash K3 and DSpark K4 remove almost all exposed generation. DSpark K4 is 6% past the ceiling, because its policy training and logprob phases are also 3% and 2% faster than the control. A frozen drafter does not change these phases, so that part is noise between jobs. DAPO 32K async shows 54% of the step as generation and gets 0.54-0.58.
- Sync runs get 0.3-0.6 of their ceiling. A 1.4-2.3x generation speedup cannot remove all of generation.
- On 32K sync, DSpark is faster than DFlash (1.55x against 1.30-1.33x). Its acceptance length is 3.6 against 2.9-3.0.
- On DAPO 32K async, DSpark is about 3% faster (1.46x against 1.41x, generation 2.46x against 2.16x). The two drafters tie on steps 7-14. DSpark is faster on steps 15-20 (1.67x against 1.51x), where the rollout waits are longer.
- Qwen3.5-35B DFlash is the only arm below 1.0x. Its hybrid GDN layers give 46 KV cache groups, and generation becomes slower (0.74x).
- Every arm in this table uses stock vLLM. The GDN metadata-reuse backport lifts Q35 DFlash K3 to 1.02x. This backport is not part of the SpecDec settings.

In every frozen arm, the policy training, logprob and refit times are within 1-4% of the control. This is the expected result. A frozen drafter adds no training work, and the drafter refit is small.

---

## Best K sits 2-4 below the block size

| Model | Mode | DFlash K tried | Best | DSpark K tried | Best |
|---|---|---|---|---|---|
| Q30 | sync | 3, 5, 7 | 5: 1.20x (K3 1.19x, K7 1.19x) | 4, 6, 8 | 6: 1.21x (K4 1.20x, K8 1.19x) |
| Q30 | async | 3, 5, 7 | 7: 1.04x (noise) | 4, 6, 8 | 8: 1.04x (noise) |
| Q235 | sync | 5, 7 | 5: 1.29x (K7 1.26x) | 6, 8 | 6: 1.28x (K8 1.27x) |
| Q235 | async | 3, 5, 7 | 3: 1.36x (K5 1.35x, K7 1.33x) | 4, 6, 8 | 4: 1.39x (K6 1.38x, K8 1.33x) |
| N35 | sync | 3, 7 | 3: 1.17x (K7 1.10x) | 4, 8 | 4: 1.19x (K8 1.10x) |
| Q35 | sync | 3, 7 | 3: 0.94x (K7 0.90x) | 4, 8 | 4: 1.05x (K8 0.97x) |

Each row compares K values in one cohort, so all K values in a row share one control. The Q235 K3/K4 sync arms are follow-ups (see below). Q30 uses the K-sweep cohort. In the first cohort, DFlash K7 gave 1.22x and DSpark K8 gave 1.20x. Q35 uses the stock-vLLM cohort.

![E2E speedup by K for OpenMath 4K sync. N35 joins two cohorts (dashed).](plots/k_sweep.png)

At temperature 1.0, the per-token acceptance is 0.25-0.41. Thus the target rarely accepts the last draft slots, but these slots still cost verify tokens.

- On Q30 the K curve is flat (within 2% from K3 to K8).
- On Q235 sync the middle K wins by 2-3%. On Q235 async, the two smaller K values tie and the block-size K is 3-5% slower. DFlash gives 1.36x, 1.35x and 1.33x at K3, K5 and K7. DSpark gives 1.39x, 1.38x and 1.33x at K4, K6 and K8. The acceptance length goes up with K (DFlash 2.46 to 2.86, DSpark 2.70 to 3.03), but the exposed generation speedup goes down at the block-size K (DFlash 2.94x to 2.51x, DSpark 3.19x at K6 to 2.38x at K8).
- On N35 and Q35 the smallest K wins by 4-9%.
- On Q35 only DSpark K4 is faster than the control. DFlash makes the step slower at both K values.

The N35 K3/K4 and K7/K8 arms ran in different cohorts. Thus their gap includes about 2% of baseline drift. **The block-size K is never the clear winner. Use a K 2-4 below the block size.**

---

## SWE RL: measure model calls, not step time

| Model | Arms | Model-call speedup | E2E | Accept len | Steps |
|---|---|---|---|---|---|
| Q30 small (3 nodes) | frozen DFlash K7 / DSpark K8 | 1.05x / 1.24x | 1.06x / 1.04x | n/a | 20 |
| Q30 scale (9 nodes) | frozen DFlash K7 / DSpark K8 | 1.03x / 1.25x | 1.07x / 1.04x | 1.40 / 1.87 | 20 |
| Q30 scale2 (9 nodes) | frozen DFlash K3 / DSpark K4 | 1.00x / 1.30x | 0.90x / 0.90x | 1.28 / 1.76 | 20 |
| Q30 single controller (V2) | frozen DSpark K4 | 1.30x | 1.17x | 2.12 (3 steps) | 20 |
| Q30 V2, SWE-trained drafter | frozen DFlash K3 / K7 | 1.83x / 2.06x | 1.06x / 0.98x | 2.48 / 3.22 | 20 |
| Q30 V2, SWE-trained drafter | online DFlash K3 / K7, lr 1e-4 | 1.78x / 1.95x | 0.94x / 1.00x | 2.42 / 2.81 | 20 |
| Q30 V2, SWE-trained drafter | frozen DSpark K4 / K8 | 1.91x / 2.00x | 1.10x / 1.03x | 2.75 / 3.67 | 20 |
| Q30 V2, SWE-trained drafter | online DSpark K4 / K8, lr 1e-4 | 1.84x / 1.82x | 1.08x / 1.08x | 2.78 / 2.74 | 20 |
| Q30 V2, SWE-trained drafter, eval timeout 600 s | frozen DSpark K8 | 1.95x | 1.01x | 3.38 | 20 |
| Q30 V2, SWE-trained drafter, ready-first sampler | frozen DSpark K8 | 2.01x | **1.23x** | 3.31 | 20 |
| Q30 V2, SWE-trained drafter, ready-first, eval timeout 450 s, four timeout prompts dropped | frozen DSpark K8 | 1.98x | **1.22x** | 3.26 | 20 |
| Q30 V2 window 2 | frozen DFlash K3 / frozen DSpark K4 / online DFlash K3 / online DSpark K4 | 1.01x / 1.33x / 1.32x / 1.29x | 1.00x / 1.09x / 1.09x / 1.06x | 1.02 / 1.22 / 1.65 / 1.29 (3-5 steps) | 20 |
| Q30 scale2 online TP4 | DFlash K3 / DSpark K4 / DSpark K8 / DFlash K7, lr 1e-4 | 1.28x / 1.31x / 1.22x / 1.27x | 1.05x / 1.07x / 1.00x / 0.99x | 1.66 / 1.76 / 1.84 / 1.69 | 20 |
| Q235 (32 nodes) | frozen DFlash K7 | 1.14x | 1.16x | 1.38 | 11 (all baseline steps) |

Model calls take about one fifth of a SWE trajectory. Thus a 1.3x model-call gain saves about 5% of the trajectory. This gain is smaller than the drift between time windows (next table). Thus the E2E column is not a SpecDec measure on SWE.

**DSpark gives a steady 1.24-1.33x on model calls. DFlash on the Thinking drafter gives 1.00-1.05x on Q30 and 1.14x on Q235.** The DFlash acceptance length is 1.3-1.4, against 1.7-2.1 for DSpark. A Q235 model call spends more of its time in decode, so the same acceptance gives more.

**Online co-training fixes the weak DFlash drafter on SWE.** The DFlash K3 acceptance length starts at the frozen value (1.28 at step 1). From step 6 it is 1.76-1.89. Thus model calls go from 1.00x frozen to 1.28x online. The V2 window-2 pair repeats this result: 1.01x frozen against 1.32x online. Online DFlash at the block-size K (K7) gives 1.27x. Online DSpark K4 drops at step 2 (1.45). From step 10 it is above the frozen arm (1.9-2.1 against 1.6-2.1).

SWE runs skip the logprob pass. The SWE2 loss sets `force_on_policy_ratio: true`. Then the legacy path (`grpo.py`) and the V2 path (`AdvantageStageConfig.policy_logprobs_required`) both skip the prev-logprob pass, and the KL penalty is 0. The legacy timer logs 0.5-1.2 ms per step. The V2 path logs no timer, so the dashboard shows 0 s for V2 SWE rows. The dashboard leaves the logprob ratio of all SWE rows empty. The V2 path logs nonzero SpecDec counters in only 3-5 of 20 steps. Thus the V2 acceptance lengths are small samples.

### Final eval, not tool time, moves the SWE step

| Q30 run | Model calls | Tool exec | Agent other | Final eval | Final eval, steps 4-8 | Setup | Trajectory | Step |
|---|---|---|---|---|---|---|---|---|
| scale2 control | 66.7 | 4.6 | 30.8 | 86.4 | 71.0 | 30.2 | 312.9 | 367.4 |
| scale2 frozen DFlash K3 | 67.8 | 5.8 | 28.7 | 111.5 | 180.1 | 30.3 | 340.6 | 408.2 |
| scale2 frozen DSpark K4 | 51.4 | 6.0 | 30.5 | 125.4 | 193.4 | 30.7 | 328.7 | 406.7 |
| scale2 online DFlash K3 | 53.2 | 5.7 | 30.6 | 75.8 | 41.3 | 31.1 | 273.6 | 351.1 |
| scale2 online DSpark K4 | 51.8 | 6.0 | 27.9 | 77.3 | 32.3 | 30.3 | 264.6 | 342.5 |
| scale2 online DSpark K8 | 55.2 | 5.6 | 30.4 | 76.6 | 33.6 | 31.2 | 277.9 | 368.8 |
| scale2 online DFlash K7 | 52.5 | 6.6 | 28.2 | 92.3 | 68.6 | 31.1 | 287.5 | 369.4 |
| V2 control | 69.0 | 5.2 | 33.9 | 85.7 | 56.0 | 31.2 | n/a | 442.0 |
| V2 frozen DSpark K4 | 51.0 | 4.7 | 31.1 | 85.0 | 55.0 | 31.6 | n/a | 377.1 |
| V2 window-2 control | 68.5 | 4.9 | 28.0 | 90.1 | 53.6 | 30.9 | n/a | 398.3 |
| V2 window-2 frozen DFlash K3 | 68.5 | 5.7 | 28.5 | 80.0 | 61.9 | 30.2 | n/a | 399.5 |
| V2 window-2 frozen DSpark K4 | 51.2 | 5.5 | 27.7 | 76.8 | 39.6 | 30.6 | n/a | 367.0 |
| V2 window-2 online DFlash K3 | 52.0 | 5.2 | 28.4 | 81.3 | 54.3 | 30.8 | n/a | 366.6 |
| V2 window-2 online DSpark K4 | 52.4 | 5.0 | 28.4 | 72.0 | 41.0 | 31.2 | n/a | 374.6 |

Units are seconds per trajectory, mean of steps 1-20 ([data](data/swe_time_breakdown.csv), from `swe_breakdown.py`). Agent other is the OpenHands run time minus model calls, tool execution and setup. Setup is container spin-up plus runtime create, connect and init. Ray queue time is 1.1-1.4 s in all runs. The V2 path does not log the trajectory time. Its step is the trainer step, which also includes policy training (29-33 s).

SpecDec changes one column. DSpark and online DFlash cut model calls from 67-69 s to 51-55 s. Tool execution is 4.6-6.0 s in every run, so tool time does not explain an E2E gap. Agent other and setup are also flat. Final eval is the column that moves:

- Outside steps 4-8, final eval is 87-103 s in all scale2 runs.
- In steps 4-8, final eval is 180-193 s in the two frozen arms and 71 s in the control. The online arms take 32-41 s.
- The two frozen arms ran in the same window (18:46-21:38 on 2026-10-21). The online DFlash K3 and DSpark K8 arms ran together on 2026-10-22 (05:54-08:21).

Final eval runs the tests of each patch in a container on shared nodes. Thus the data agrees with a window effect, not an arm effect. **The scale2 E2E column (0.90x frozen, 1.05-1.07x online) compares time windows, not drafters.** The V2 pair ran in windows that overlap, and its final eval times match (85.7 s against 85.0 s). But its 1.17x step ratio does not repeat. The window-2 repeat gives 1.09x, and that interval includes 1.

**Window 2 confirms the window effect.** The V2 window-2 control and three arms ran at one commit (15dbc21d1a) in one window on 2026-10-22. Final eval is 72-90 s over steps 1-20 and 41-62 s in steps 4-8 in all four runs. No arm has a final-eval spike when all arms share a window. The step ratios are 1.00x for frozen DFlash K3, 1.09x for online DFlash K3 and 1.06x for online DSpark K4. The frozen DSpark K4 repeat (8176229) started at 22:09, after the other four runs ended. Its final eval is 76.8 s over steps 1-20 and 39.6 s in steps 4-8, in the same range. Its step ratio is 1.09x.

### The model-call gain is real, the step gain is mostly noise

Agent runs differ from run to run. An agent can take more turns, a command can fail, or a final eval can time out. Thus we check two things:

1. Does the workload differ between arms?
2. Is the gain larger than the step-to-step spread?

Each step pairs the same prompts in the arm and in its control. The 90% interval comes from a bootstrap resample of the 20 paired steps. Each ratio is the control time over the arm time. Thus a ratio above 1 means that the arm is faster.

| Q30 run | Turns | Gen tokens | Masked | Model-call ratio [90%] | Steps faster | Model-call steps 11-20 [90%] | Step ratio [90%] | Steps faster |
|---|---|---|---|---|---|---|---|---|
| scale2 control | 11.3 | 14,953 | 4.1% | 1 | | 1 | 1 | |
| scale2 frozen DFlash K3 | 12.0 | 15,248 | 3.6% | 1.01x [0.99, 1.03] | 12/20 | 1.00x [0.97, 1.03] | 0.90x [0.73, 1.11] | 7/20 |
| scale2 frozen DSpark K4 | 11.2 | 15,034 | 4.8% | 1.31x [1.28, 1.34] | 20/20 | 1.31x [1.25, 1.36] | 0.90x [0.74, 1.10] | 10/20 |
| scale2 online DFlash K3 | 11.4 | 15,291 | 3.6% | 1.30x [1.24, 1.36] | 20/20 | 1.42x [1.38, 1.45] | 1.05x [0.92, 1.23] | 9/20 |
| scale2 online DSpark K4 | 11.3 | 15,190 | 3.6% | 1.31x [1.26, 1.36] | 20/20 | 1.40x [1.36, 1.44] | 1.07x [0.95, 1.24] | 9/20 |
| scale2 online DSpark K8 | 11.5 | 15,079 | 3.4% | 1.22x [1.18, 1.26] | 19/20 | 1.30x [1.26, 1.33] | 1.00x [0.84, 1.21] | 7/20 |
| scale2 online DFlash K7 | 11.4 | 14,643 | 4.1% | 1.27x [1.20, 1.35] | 18/20 | 1.44x [1.39, 1.48] | 0.99x [0.71, 1.48] | 10/20 |
| V2 control | 11.6 | 15,497 | 3.8% | 1 | | 1 | 1 | |
| V2 frozen DSpark K4 | 11.6 | 14,882 | 3.8% | 1.32x [1.27, 1.36] | 20/20 | 1.32x [1.28, 1.35] | 1.17x [1.04, 1.35] | 14/20 |
| V2 window-2 control | 11.6 | 15,261 | 4.7% | 1 | | 1 | 1 | |
| V2 window-2 frozen DFlash K3 | 12.2 | 15,318 | 4.2% | 1.01x [0.99, 1.04] | 10/20 | 1.01x [0.97, 1.05] | 1.00x [0.69, 1.35] | 10/20 |
| V2 window-2 frozen DSpark K4 | 11.2 | 14,987 | 3.9% | 1.33x [1.29, 1.36] | 20/20 | 1.33x [1.27, 1.40] | 1.09x [0.78, 1.50] | 12/20 |
| V2 window-2 online DFlash K3 | 11.4 | 15,016 | 4.1% | 1.32x [1.25, 1.39] | 19/20 | 1.43x [1.38, 1.47] | 1.09x [0.77, 1.49] | 11/20 |
| V2 window-2 online DSpark K4 | 11.5 | 14,968 | 3.0% | 1.29x [1.25, 1.33] | 19/20 | 1.35x [1.32, 1.40] | 1.06x [0.73, 1.48] | 11/20 |

Values are means of steps 1-20, except the steps 11-20 column ([data](data/swe_variation.csv), from `swe_variation.py`). Masked is the share of trajectories that the agent server drops from training.

The workload is the same in every run. Each step has 11.2-12.2 turns, 14.6-15.5K generated tokens and 3.0-4.8% masked trajectories. No run has truncated trajectories or failed vLLM generations. More turns make model calls slower (per-step correlation -0.47 in the scale2 control). But the turn counts match, so the workload does not explain the gain. Three independent controls give 223.7-226.1 tok/s. Three independent frozen DSpark K4 runs give 295.7-297.2 tok/s. **Model-call tok/s repeats within 1.1% across runs. DSpark is faster in each of the 20 paired steps. Thus its 1.3x is a real SpecDec gain.** Frozen DFlash K3 stays inside the noise in both windows (1.01x).

**The online drafters are faster at the end of the run than the 20-step mean shows.** In steps 11-20, online DFlash K3 gives 1.42x in scale2 and 1.43x in window 2. Online DFlash K7 gives 1.44x. Online DSpark K4 gives 1.40x in scale2 and 1.35x in window 2. Frozen DSpark K4 gives 1.31-1.33x in all three of its runs. The frozen arms have the same ratio in steps 1-20 and in steps 11-20. Thus the late gain does not come from the time of the run. **Two independent windows give the same online DFlash K3 gain (1.30x and 1.32x over steps 1-20).**

Step time is different. Async step times alternate between short and long steps. The per-step spread is 99-123% of the mean. In every run, the slowest final evals reach the 1200 s eval timeout. **Only the first V2 frozen DSpark K4 step ratio is above 1 at 90%. Its interval is 1.04x to 1.35x.** The 90% intervals of all other step ratios include 1. This includes window 2, where all arms share one window and the intervals are 0.69x-1.49x wide. A 1.3x model-call gain moves the step by about 5%. Thus 20 steps cannot resolve the SWE step gain. The window-2 frozen DSpark K4 repeat confirms this. Its model-call ratio repeats (1.33x against 1.32x), but its step ratio drops to 1.09x [0.78, 1.50].

### A drafter trained on SWE rollouts doubles model-call speed

The DFlash drafter above learned from general Thinking outputs. Its SWE acceptance length is 1.28 at K3. A new DFlash drafter (`q30t-swe-regen-rp25swa8k-b8-s28050`) learned from SWE trajectories that Q30 generated again. It has block size 8 and 8K sliding-window attention. A new V2 control and the frozen K3 and K7 arms ran in one window on 2026-10-23 at commit 333b3c6ce7. The online K3 and K7 arms (draft lr 1e-4) ran later on the same day (05:30-07:57) at the same commit. A DSpark drafter (`q30t-swe-regen-dspark-rp25swa8k-b8-n32-s7010`) learned from the same SWE trajectories. It also has block size 8 and 8K sliding-window attention. Its four arms ran together on 2026-10-23 (17:29-19:56), 15 h after the control. The frozen K8 arm ran at 333b3c6ce7 plus its config. The other three arms ran at 8bb9b95f4c, which also trains the DSpark body with the drafter attention window.

| Q30 V2 run | Accept len | Model calls (s) | Model-call ratio [90%] | Steps faster | Slowest agent run (s) | Step (s) | Step ratio [90%] |
|---|---|---|---|---|---|---|---|
| control | | 67.7 | 1 | | 268 | 381.0 | 1 |
| frozen DFlash K3 | 2.48 | 36.9 | 1.83x [1.78, 1.87] | 20/20 | 220 | 358.6 | 1.06x [0.99, 1.19] |
| frozen DFlash K7 | 3.22 | 33.6 | 2.06x [2.03, 2.10] | 20/20 | 214 | 389.0 | 0.98x [0.77, 1.16] |
| online DFlash K3 | 2.42 | 38.1 | 1.78x [1.75, 1.81] | 20/20 | 269 | 403.9 | 0.94x [0.83, 1.06] |
| online DFlash K7 | 2.81 | 34.9 | 1.95x [1.90, 2.00] | 20/20 | 242 | 380.1 | 1.00x [0.84, 1.19] |
| frozen DSpark K4 | 2.75 | 34.4 | 1.91x [1.89, 1.93] | 20/20 | 200 | 348.0 | 1.10x [1.02, 1.23] |
| frozen DSpark K8 | 3.67 | 33.4 | 2.00x [1.96, 2.03] | 20/20 | 220 | 371.2 | 1.03x [0.95, 1.17] |
| online DSpark K4 | 2.78 | 36.2 | 1.84x [1.82, 1.87] | 20/20 | 190 | 353.8 | 1.08x [0.95, 1.26] |
| online DSpark K8 | 2.74 | 37.2 | 1.82x [1.76, 1.87] | 20/20 | 201 | 352.0 | 1.08x [0.94, 1.30] |

Model calls are seconds per trajectory. The slowest agent run is the largest OpenHands run time in a step. All columns are means of steps 1-20. The acceptance lengths are token-weighted over the steps with nonzero V2 counters (frozen DFlash: 5 steps for K3 and 3 steps for K7, online DFlash: 4 steps for K3 and 5 steps for K7, frozen DSpark: 4 steps for K4 and 2 steps for K8, online DSpark: 3 steps for K4 and 4 steps for K8).

**The SWE-trained drafter doubles the acceptance length and the model-call speed.** At K3, the acceptance length goes from 1.28 to 2.48, and model calls go from 1.01x to 1.83x. K7 gives 2.06x. On SWE, the block-size K beats K3. This is the opposite of the math result. The workload matches the control: 11.4-11.7 turns, 15.1-15.6K generated tokens and 4.1-4.2% masked trajectories.

**Four final-eval timeouts set the step, so the step does not move.** In steps 2, 7, 10 and 15, one final eval reaches the 1200 s timeout in all three runs. The runs use the same prompts in each step. These four steps take 54-63% of the 20-step time, and SpecDec cannot shorten a test that times out. Outside these four steps, K3 gives 1.21x. K7 gives 0.90x, because it has a fifth timeout at step 20 (912 s step). The slowest agent run drops from 268 s to 214-220 s (1.22-1.25x). **On SWE, the step waits for the final-eval timeout, not for the model calls. The next SWE lever is a shorter eval timeout or a step that does not wait for the slowest eval, not a better drafter.**

**Online co-training at draft lr 1e-4 makes the SWE-trained drafter slightly worse.** Online K3 gives 1.78x against 1.83x frozen. Online K7 gives 1.95x against 2.06x frozen. In steps 11-20, the gap is larger at K7: 1.89x against 2.04x. At K3 it is 1.76x against 1.80x. The acceptance length drops from 2.48 to 2.42 at K3 and from 3.22 to 2.81 at K7. The draft loss goes up during the run. At K3 it is 2.62 in steps 1-10 and 3.00 in steps 11-20. At K7 it is 2.60 and 2.80. The Thinking drafter did not fit SWE outputs, so online training lifted it from 1.01x to 1.32x. This drafter already fits SWE outputs, and lr 1e-4 moves it away from that fit. The online workload matches the control: 11.1-11.3 turns, 15.2K generated tokens and 3.4-4.1% masked trajectories. Policy training takes 32.6 s per step in both online arms, against 30.5-31.4 s in the frozen arms.

The online step ratios are 0.94x and 1.00x, and both intervals include 1. The online arms have the same four timeout steps as the control. On the other 15 steps, model calls stay fast (38 s and 35 s per trajectory, against 69 s in the control). But exposed generation is 192 s in both online arms, against 163 s in the control and 128-135 s in the frozen arms. The slowest final eval sets this gap: 191 s and 178 s against 111 s in the control. The online arms ran in a later window than the control. Thus this is the final-eval window effect from scale2, not a drafter effect.

**The SWE-trained DSpark drafter matches the SWE-trained DFlash drafter.** Frozen DSpark K4 gives 1.91x, against 1.83x for DFlash K3. Frozen DSpark K8 gives 2.00x, against 2.06x for DFlash K7. Thus the training data matters more than the drafter type. The block-size K beats the small K again (2.00x against 1.91x). In steps 11-20 the frozen arms give 1.90x at K4 and 1.98x at K8, so they do not drift. The K8 acceptance length (3.67) comes from 2 steps only. The workload matches the control: 11.3-11.5 turns, 14.8-15.0K generated tokens and 3.6-4.1% masked trajectories.

**Frozen DSpark K4 is the second SWE run with a step ratio above 1 at 90% (1.10x [1.02, 1.23]).** The four DSpark arms have the same four timeout steps as the control. These steps take 61-66% of the 20-step time. Outside them, the DSpark arms give 1.17-1.25x on the step, and frozen DFlash K3 gives 1.21x. Exposed generation outside these steps is 119-129 s, against 159 s in the control. The slowest final eval outside these steps is 121-132 s, against 115 s in the control. Thus the later window has no final-eval spike. But frozen DSpark K8 gives 1.03x [0.95, 1.17], and both online intervals include 1. Thus the step gain is not confirmed.

**Online co-training at lr 1e-4 also makes the SWE-trained DSpark drafter worse, more at K8.** Online K4 gives 1.84x against 1.91x frozen. Online K8 gives 1.82x against 2.00x frozen. In steps 11-20 the gap grows: 1.80x against 1.90x at K4, and 1.75x against 1.98x at K8. The draft loss goes up from 0.98 in steps 1-10 to 1.04 in steps 11-20 at K4, and from 1.00 to 1.08 at K8. The DSpark loss adds CE and TV terms, so its value does not compare with the DFlash loss. Policy training takes 32.1-32.3 s per step in the online arms, against 29.5-29.8 s in the frozen arms (+8-9%). The DFlash and DSpark results agree. **Keep a drafter that already fits SWE outputs frozen.**

### The ready-first sampler turns the model-call gain into a step gain

The SWE step waits for the slowest final eval of its prompts. Thus two rollout settings can help. A shorter final-eval timeout (600 s, against 1200 s) cuts the wait. The ready-first sampler of the V2 path does not wait. It trains on the groups that finish first and moves a slow group to a later step (`max_staleness_versions: 1`, `max_lookahead_versions: 1`). Each setting ran as a baseline and a frozen SWE-trained DSpark K8 arm. All four runs ran in one window (02:07-03:51 on 2026-10-24). The in_order control ran on 2026-10-23.

| Q30 V2 run | Compared with | Model calls (s) | Model-call ratio [90%] | Slowest agent run (s) | Exposed gen (s) | Step (s) | Step ratio [90%] | Steps faster | Step spread |
|---|---|---|---|---|---|---|---|---|---|
| in_order control | | 67.7 | 1 | 268 | 342.1 | 381.0 | 1 | | 105% |
| in_order DSpark K8 | in_order control | 33.4 | 2.00x [1.96, 2.03] | 220 | 333.5 | 371.2 | 1.03x [0.95, 1.17] | 13/20 | 121% |
| eval 600 s control | in_order control | 66.2 | 1.03x [1.01, 1.04] | 259 | 232.7 | 270.7 | 1.41x [1.04, 1.76] | 12/20 | 80% |
| eval 600 s DSpark K8 | eval 600 s control | 34.4 | 1.95x [1.90, 2.00] | 241 | 228.6 | 267.2 | 1.01x [0.79, 1.31] | 10/20 | 88% |
| ready-first control | in_order control | 63.3 | 1.05x [1.03, 1.08] | 288 | 157.1 | 213.3 | 1.79x [1.12, 2.49] | 12/20 | 18% |
| ready-first DSpark K8 | ready-first control | 30.9 | 2.01x [1.99, 2.04] | 177 | 119.0 | 173.1 | **1.23x [1.19, 1.28]** | **20/20** | 18% |
| eval 450 s control | ready-first control | 66.6 | 0.98x [0.97, 1.00] | 254 | 127.7 | 180.6 | 1.18x [1.06, 1.33] | 13/20 | 31% |
| eval 450 s DSpark K8 | eval 450 s control | 31.7 | 1.98x [1.94, 2.02] | 202 | 92.3 | 148.7 | **1.22x [1.08, 1.38]** | 14/20 | 35% |

The step spread is the standard deviation of the step time over its mean. The data is in [swe_variation.csv](data/swe_variation.csv) and [swe_time_breakdown.csv](data/swe_time_breakdown.csv).

**With the ready-first sampler, the SWE step gain is confirmed.** Frozen DSpark K8 gives 1.23x on the step, and it is faster in each of the 20 steps. The 90% interval is 1.19x to 1.28x. The model-call ratio is the same as with the in_order sampler (2.01x against 2.00x), and it does not drift (2.03x in steps 11-20). Ready-first cuts the step spread from 105-121% to 18%. Thus a step no longer depends on which prompts time out, and 20 steps can resolve the gain. Exposed generation drops from 157 s to 119 s, and that drop is most of the 40 s step gain.

**The 600 s timeout speeds up the baseline, but the SpecDec step gain stays in the noise.** The eval 600 s control is 1.41x faster than the in_order control. The 1200 s timeouts in steps 2, 7, 10 and 15 now stop at 600 s. But the step still waits for them, and the step spread stays at 80-88%. The DSpark K8 arm has a fifth 600 s timeout at step 12. Its step ratio is 1.01x [0.79, 1.31].

Ready-first moves the slow groups but does not remove them. One final eval still reaches 1200 s in 3-4 steps of each ready-first run, at other steps than in the in_order runs. The ready-first control gives 1.79x against the in_order control, but the interval is wide (1.12x to 2.49x). The two runs ran in different windows, and ready-first does not keep the same prompts in a step. Thus this ratio is a sampler estimate, not a SpecDec result.

Ready-first changes the training data, so we checked its health. Neither run aborted, dropped, evicted or replaced a group. No sample was out of bounds. The probability ratio is 1.0, and about 31 of 32 samples per step are valid. The ready-first loss sets `force_on_policy_ratio: false`. Thus these runs do the logprob pass (7.9-8.2 s per step), and the in_order runs skip it. This cost is in both ready-first runs, so it does not change the K8 ratio. It is in the 1.79x sampler ratio. The token-probability error of the K8 arm is 1.03 in 19 of 20 steps and 5.34 at step 18 (see Suspicious outcomes).

**A third pair repeats the ready-first step gain.** It adds a 450 s eval timeout to ready-first. It also drops the four prompts that reach the 1200 s timeout in every in_order run (jobs 8228299 and 8228300, 2026-10-24). Frozen DSpark K8 gives 1.98x on model calls and 2.01x in steps 11-20. It gives 1.22x [1.08, 1.38] on the step and is faster in 14 of 20 steps. The seconds map one to one. Model calls save 35.0 s per trajectory, exposed generation saves 35.4 s and the step saves 31.9 s. Each run has one step with a long final eval (570 s at step 9 in the control, 450 s at step 14 in the arm). Thus the step spread is 31-35%, against 18% in the first pair. The masked share drops from 2.5-3.9% to 0.3-0.5%. The V2 counters are nonzero in 15 of 20 steps, so the K8 acceptance length (3.26) is a good sample.

The 450 s control is 1.18x [1.06, 1.33] faster than the ready-first control. Final eval drops from 76.8 s to 33.2 s per trajectory. The two controls use different prompts in 4 of 20 steps, so this ratio is a config estimate. Against the ready-first control, the 450 s config and K8 together give 1.43x (213.3 s to 148.7 s). The 450 s control has a token-probability error of 7.57 at step 19, and the 450 s K8 arm has no spike (median 1.03). Thus the 5.34 spike of the first K8 arm is not a SpecDec effect.

**Two more drafters repeat the sampler result, and neither beats DSpark K8 on the step.** DFlash K7 and a DSpark K8 drafter with 12 target taps (against 5) ran with the same three sampler configs on 2026-10-24. They ran in a later window than their controls. Thus the model-call ratio, the acceptance length and the model-call speed are the robust metrics here.

| Drafter | Sampler | Acceptance length | Model-call tok/s | Model-call ratio | Step ratio [90%] | Steps faster |
|---|---|---|---|---|---|---|
| DSpark K8, 5 taps | eval 600 s | 3.38 | 452.5 | 1.95x | 1.01x [0.79, 1.31] | 10/20 |
| DFlash K7 | eval 600 s | 2.94 | 466.3 | 2.01x | 1.06x [0.81, 1.41] | 11/20 |
| DSpark K8, 12 taps | eval 600 s | 3.17 | 460.0 | 1.99x | 1.00x [0.87, 1.14] | 12/20 |
| DSpark K8, 5 taps | ready-first | 3.31 | 478.4 | 2.01x | 1.23x [1.19, 1.28] | 20/20 |
| DFlash K7 | ready-first | 3.10 | 488.9 | 2.06x | 1.19x [1.13, 1.25] | 18/20 |
| DSpark K8, 12 taps | ready-first | 3.19 | 470.2 | 1.98x | 1.21x [1.15, 1.26] | 19/20 |
| DSpark K8, 5 taps | ready-first + eval 450 s | 3.26 | 461.2 | 1.98x | 1.22x [1.08, 1.38] | 14/20 |
| DFlash K7 | ready-first + eval 450 s | 3.09 | 479.2 | 2.05x | 1.29x [1.15, 1.44] | 16/20 |
| DSpark K8, 12 taps | ready-first + eval 450 s | 3.14 | 462.7 | 1.98x | 1.24x [1.10, 1.41] | 15/20 |

The acceptance length is the draft-weighted mean (1 + accepted tokens / drafts) over the steps with nonzero V2 counters. The data is in [swe_variation.csv](data/swe_variation.csv).

- The 12-tap DSpark drafter accepts 4-6% fewer tokens per draft than the 5-tap drafter. Its model-call speed stays within 2% of the 5-tap drafter, and its step ratios overlap. More target taps do not help this drafter.
- DFlash K7 accepts fewer tokens than both DSpark drafters, but it has the highest model-call speed (466-489 tok/s). Thus a DFlash draft costs less time than a DSpark draft at these K values.
- All three drafters give 1.95-2.06x on model calls. The eval 600 s sampler hides that gain in the step for all three drafters. The two ready-first samplers show 1.19-1.29x for all three drafters.

### OpenHands startup is as long as the K8 model calls

With K8, model calls take 31.5 s per trajectory. The OpenHands startup before the first model call takes 51 s. `swe_agent_timeline.py` splits each OpenHands run of the 450 s pair into phases from the agent logs ([control](data/swe_agent_timeline_baseline.csv), [K8](data/swe_agent_timeline_k8.csv)).

| Phase | What runs | Control mean | Control p50 / p90 | K8 mean |
|---|---|---|---|---|
| spin-up | generation start to apptainer start | 0.6 | | 0.6 |
| pre | `run_infer.sh`, two `poetry run` calls, imports from the 2.4G Lustre venv, up to `create_runtime` | 23.5 | 21.2 / 36.2 | 20.5 |
| connect | `create_runtime` and `connect_to_runtime`: action server start, `/alive` poll every 2 s | 17.9 | 18.9 / 21.0 | 15.7 |
| init | `initialize_runtime` | 14.5 | 4.6 / 16.1 | 14.2 |
| model calls | | 66.6 | 59.9 / 113.4 | 31.5 |
| tool execution | | 5.0 | 1.9 / 8.5 | 4.2 |
| controller | agent loop between turns | 2.5 | 2.4 / 3.3 | 2.2 |
| post | controller end to container exit | 4.2 | 3.5 / 7.3 | 4.2 |
| OpenHands run | sum of the phases | 134.9 | | 93.2 |

Units are seconds per trajectory (640 control and 646 K8 trajectories).

- Startup (spin-up, pre, connect and init) is 56.5 s in the control and 51.0 s in K8. With K8, startup is 1.6x the model-call time.
- Pre grows with the number of agents that start together. Its correlation with the number of starts within 30 s is 0.77. At 45 or more starts per minute, pre is 37-45 s. Connect stays at about 18 s.
- In each run, all eight trajectories of one prompt (`singapore__renovate-527`) reach the 600 s init timeout. Without them, init is 7.1 s in the control and 6.9 s in K8.
- The controller adds 0.3 s per turn. It is not a lever.

In this pair, the model-call saving maps one to one into the step. If a startup saving maps the same way, a 30 s cut takes the K8 step from 148.7 s to about 119 s. This is an estimate. The levers, with the expected saving per trajectory:

| Lever | Phase | Expected saving (s) |
|---|---|---|
| Node-local or squashfs copy of the OpenHands setup tree | pre, connect | 15-30 |
| Pool of warm action servers | connect | about 18 |
| Remove the second `poetry run` and the `git checkout` from `run_infer.sh` | pre | 5-10 |
| Fix `singapore__renovate-527`, or set the init timeout to 120 s | init | about 7 |
| Poll `/alive` every 0.2 s, not every 2 s | connect | about 1 |
| Stagger the agent starts | pre | about 2 |

The node-local copy helps only if Lustre metadata contention sets the pre time. Job 8247019 was to test this, but its result is not valid. The bench runs the startup commands at concurrency 1, 8, 32 and 64, once from Lustre and once from `/dev/shm` ([bench](../agent_startup_bench/README.md)). All levers change Gym or OpenHands code.

### Five trajectory fixes together give 1.11x on the step

A ladder of six arms tests the trajectory fixes one at a time. All arms use DSpark K8 with ready-first and eval 450 s. Each arm adds one item to the arm above it. One commit holds all items, and config knobs turn them on ([ladder data](data/swe_traj_ladder.csv)).

| Arm | Item added | Step (s) | This item | All items so far | Trajectory (s) | Phase that moves |
|---|---|---|---|---|---|---|
| L0 | Control | 150.0 | | | 130.1 | |
| L1 | O1 precompiled OpenHands bytecode | 144.1 | 1.04x [0.95, 1.15] | 1.04x [0.95, 1.15] | 122.9 | pre 20.7 to 18.4 s, connect 16.0 to 12.4 s |
| L2 | O8 least-loaded vLLM endpoint placement | 144.6 | 1.00x [0.90, 1.12] | 1.04x [0.93, 1.15] | 126.4 | none |
| L3 | O2 direct venv python, O3 0.1 s `/alive` poll | 149.8 | 0.97x [0.87, 1.08] | 1.00x [0.94, 1.08] | 123.8 | init 14.3 to 13.3 s |
| L4 | O4/O5 fast deep reset | 144.9 | 1.03x [0.94, 1.13] | 1.04x [0.95, 1.13] | 112.0 | init 13.3 to 2.2 s |
| L5 | O7 0.2 s eval poll | 135.1 | 1.07x [0.98, 1.19] | 1.11x [0.99, 1.25] | 109.1 | final eval 34.1 to 28.9 s |

Step ratios pair 20 steps and show a 90% bootstrap interval. "This item" compares an arm with the arm above it. "All items so far" compares it with L0. Trajectory is the mean sum of phases from `swe_agent_timeline.py`, over 653-672 trajectories per arm.

- **Three items cut a phase.** O1 cuts pre and connect by 5.9 s. O4/O5 cut the deep reset in init from 12.1 s to 1.1 s. O7 cuts the final eval by 5.2 s.
- O8, O2 and O3 do not move a phase. Their step ratios are noise.
- Together, the items cut the trajectory by 21 s (16%) and the step by 15 s. Exposed generation drops from 94.0 s to 77.4 s. The ready-first step waits for full groups of eight rollouts. The mean time to the last rollout of a group drops from 133.7 s to 121.7 s.
- Each single item is inside the step noise of about 10%. Only the total comes near the bar: 15 of 20 steps are faster, and the low bound is 0.99.
- L0 ran before L1 to L5, and L1 to L5 ran at the same time. The item ratios of L2 to L5 compare arms of one window. L1 and the totals compare across windows.

After the fixes, startup before the first model call is 35.4 s, down from 52.4 s. Model calls take 33.7 s. Pre (18.7 s) and connect (12.8 s) are now the largest phases outside the model calls and the final eval.

### On the faster baseline, SpecDec gains more per trajectory but less per step

The fixes cut time outside the model calls. SpecDec cuts the model calls. Thus the model calls are a larger part of the faster trajectory, and the SpecDec gain should grow. A projection from the ladder gave 1.24-1.31x on the step, against 1.22x before the fixes. A 2x2 tests this in one window. It runs L0 and L5, each with no drafter and with DSpark K8. Two replicates (R1, R2) run all four arms for 20 steps each ([arms](data/swe_sd2x2_arms.csv), [ratios](data/swe_sd2x2_ratios.csv)). The tables show both replicates together: 40 steps and 1311-1341 trajectories per arm.

| Arm | Fixes | Drafter | Step (s) | Exposed gen (s) | Model calls (s) | Final eval (s) | Trajectory mean (s) | Trajectory p50 (s) | Trajectory p90 (s) |
|---|---|---|---|---|---|---|---|---|---|
| b0 | L0 | none | 181.9 | 126.8 | 64.0 | 33.5 | 160.4 | 137.0 | 222.0 |
| d0 | L0 | DSpark K8 | 145.9 | 90.5 | 33.0 | 33.4 | 129.7 | 104.1 | 184.1 |
| b5 | L5 | none | 165.5 | 110.2 | 66.5 | 30.5 | 139.0 | 122.4 | 197.8 |
| d5 | L5 | DSpark K8 | 147.1 | 92.3 | 32.0 | 33.8 | 108.6 | 86.3 | 156.4 |

| Comparison | Step [90%] | Steps faster | R1 step | R2 step | Model calls | Trajectory mean | Trajectory p50 | Trajectory p90 |
|---|---|---|---|---|---|---|---|---|
| SpecDec at L0 (b0 / d0) | 1.25x [1.15, 1.35] | 33/40 | 1.33x | 1.17x | 1.94x | 1.24x | 1.32x | 1.21x |
| SpecDec at L5 (b5 / d5) | 1.13x [1.02, 1.24] | 28/40 | 1.11x | 1.14x | 2.08x | 1.28x | 1.42x | 1.26x |
| Fixes without drafter (b0 / b5) | 1.10x [0.99, 1.23] | 24/40 | 1.13x | 1.07x | 0.96x | 1.15x | 1.12x | 1.12x |
| Fixes with DSpark K8 (d0 / d5) | 0.99x [0.91, 1.07] | 23/40 | 0.94x | 1.04x | 1.03x | 1.19x | 1.21x | 1.18x |
| Fixes and DSpark K8 (b0 / d5) | 1.24x [1.13, 1.34] | 31/40 | 1.25x | 1.22x | 2.00x | 1.48x | 1.59x | 1.42x |

The step interval comes from a paired bootstrap that resamples steps within each replicate. Model calls and trajectory columns are means or quantiles over all trajectories from `swe_agent_timeline.py`.

- **Per trajectory, the SpecDec gain grows on the faster baseline, as the projection says.** The mean gain goes from 1.24x to 1.28x, the median from 1.32x to 1.42x and the p90 from 1.21x to 1.26x. Both replicates agree.
- **Per model call, the gain also grows a little: 1.94x to 2.08x.** The fixes do not change the model calls without a drafter (0.96x).
- **Per step, the gain does not grow. It falls from 1.25x to 1.13x.** The L5 gain stays above 1 (low bound 1.02x), so SpecDec still helps on the faster baseline. R1 alone gave 1.33x and 1.11x. R2 gave 1.17x and 1.14x. Thus one 20-step run moves by about 10%, and the pooled numbers are the ones to use.
- **With the drafter, the fixes do not cut the step (0.99x).** They cut the trajectory mean by 21 s (129.7 s to 108.6 s) and the p90 by 28 s. But exposed generation stays at 90.5 s and 92.3 s. Without the drafter, the same fixes cut exposed generation by 16.6 s. So with DSpark, the trajectory time no longer sets the exposed generation. The cause is not known yet. The candidates are the slowest trajectories in a step, the group gate of the ready-first sampler, and rollouts that async carries into the next step.
- The fixes and DSpark K8 together give 1.24x [1.13, 1.34] on the step and 1.48x on the trajectory mean.

**Answer: on the faster baseline, the SpecDec gain is higher per trajectory (1.28x) and per model call (2.08x), but lower per step (1.13x against 1.25x).** To turn the trajectory gain into a step gain, find what sets the exposed generation of the DSpark arms.

---

## Online co-training

The first online arms did not train the drafter. On the Megatron path, the draft parameters inherited the policy lr (3e-7). With the draft lr at 1e-4 and the policy at TP1, our implementation and the teammate's implementation give the same result.

| Arm (Q30 TP1) | Impl | Gen | Step | Accept len | Paired |
|---|---|---|---|---|---|
| DFlash K7 | teammate | 1.71x | 1.15x | 3.62 | 18 |
| DFlash K7 | ours | 1.68x | 1.14x | 3.62 | 18 |
| DSpark K8 | teammate | 1.67x | 1.13x | 4.00 | 18 |
| DSpark K8 | ours | 1.60x | 1.13x | 4.01 | 18 |
| DFlash K7, draft lr 1e-5 | ours | 1.70x | 1.14x | 3.80 | 18 |

The DSpark K8 generation ratios differ (1.60x against 1.67x) with the same acceptance length. The cause is a difference between the two baselines. Arm generation is 38.3 s against 37.4 s, and baseline generation is 61.2 s against 62.3 s. Both differences are inside the 1.3-2.7% drift between jobs. Drafter training adds 9-12% to policy training time in both implementations.

**The draft lr sets the early dip, not the gain.**

- At lr 1e-4, the DFlash acceptance length drops from 3.92 at step 1 to 3.25 at step 3. It recovers to 3.66 by step 20. The draft loss peaks at 2.30 at step 2 and ends at 1.90.
- At lr 1e-5, the acceptance length stays at 3.72-3.94 for all 20 steps. The draft loss is 1.70-1.94.
- Both stay below the untrained drafter (3.86 in the earlier run, where the draft inherited lr 3e-7).

The drafter already fits math outputs well, so 20 steps of math rollouts add little. Step time is the same (1.14x). Generation is 25% of the arm step, and the 5% higher acceptance length cuts generation by only 1.4%.

SWE online arms ran out of memory at TP2. The old draft gate forced TP2. At the baseline TP4 layout, all three online SWE arms pass. Over 20 paired steps, DSpark K4 keeps the frozen model-call gain (1.31x against 1.30x). In steps 11-20 it is ahead of the frozen arm (1.40x against 1.31x). Its training takes 13% longer than the control and 9% longer than the frozen arm. The V2 window-2 rerun repeats the online DFlash K3 result (1.32x, and 1.43x in steps 11-20). Online DFlash K7 gives 1.27x and 1.44x in steps 11-20.

Online training does not help a drafter that already fits the workload. With the SWE-trained DFlash drafter at lr 1e-4, online K3 and K7 give 1.78x and 1.95x on model calls, against 1.83x and 2.06x frozen. With the SWE-trained DSpark drafter, online K4 and K8 give 1.84x and 1.82x, against 1.91x and 2.00x frozen. The draft loss goes up over the run in all four online arms. A lower draft lr (1e-5) is the next check. Until then, keep a SWE-trained drafter frozen.

### A schedule removes most of the drafter-training cost

`policy.draft.update_schedule` selects the steps on which the drafter trains. `every_step` is the old behavior. `interval` trains on steps 1, N+1, 2N+1, and so on. `acceptance` trains while the acceptance length improves by more than `min_gain` (0.02) and stops after `patience` (2) steps without gain. It starts again when the acceptance length drops by more than `drift_tolerance` (0.05). The SWE arm also sets `max_skip_steps` to 4. All arms use Q30 DSpark K8 with draft lr 1e-4.

| Workload | Schedule | Updates | Step (s) | Policy train (s) | Accept len | Model-call tok/s | Step vs every step | Train overhead removed |
|---|---|---|---|---|---|---|---|---|
| Math sync | Frozen | 0 / 18 | 215.7 | 133.5 | 4.16 | | 1.10x [1.10, 1.11] | 100% |
| Math sync | Every step | 18 / 18 | 238.1 | 153.3 | 4.16 | | | |
| Math sync | Interval 5 | 3 / 18 | 220.7 | 138.1 | 4.17 | | 1.08x [1.07, 1.09] | 77% |
| Math sync | Acceptance | 0 / 18 | 220.2 | 137.0 | 4.16 | | 1.08x [1.08, 1.09] | 82% |
| Math async | Frozen | 0 / 18 | 382.7 | 309.1 | 4.15 | | 1.10x [1.10, 1.10] | 100% |
| Math async | Every step | 18 / 18 | 421.5 | 346.6 | 4.14 | | | |
| Math async | Interval 5 | 3 / 18 | 394.0 | 318.9 | 4.15 | | 1.07x [1.06, 1.08] | 74% |
| Math async | Acceptance (partial) | 1 / 12 | 389.8 | 316.2 | 4.16 | | 1.09x [1.07, 1.10] | 81% |
| SWE | Frozen | 0 / 20 | 143.0 | 29.6 | 3.26 | 474.3 | 1.06x [0.94, 1.17] | 100% |
| SWE | Every step | 20 / 20 | 151.0 | 31.4 | 2.80 | 419.8 | | |
| SWE | Interval 5 | 4 / 20 | 151.0 | 30.9 | 3.07 | 440.4 | 1.00x [0.89, 1.10] | 27% |
| SWE | Acceptance | 13 / 20 | 151.6 | 31.1 | 2.84 | 422.9 | 1.00x [0.88, 1.11] | 19% |

Math pairs steps 3-20 and SWE pairs steps 1-20. Updates count the steps in this window. The brackets give the 90% paired bootstrap interval. The data is in [data/q30_draft_schedule.csv](data/q30_draft_schedule.csv) and [data/q30_draft_schedule_steps.csv](data/q30_draft_schedule_steps.csv).

**On math, the schedule recovers most of the frozen speed.** Every-step training adds 19.8 s to the sync policy-training phase. Interval 5 removes 77% of this overhead and the acceptance schedule removes 82%. Async math shows the same pattern: interval 5 removes 74% and gives 1.07x against 1.10x for frozen. The async acceptance arm was still running at the time of this report. Over its first 12 window steps it gives 1.09x. The math arms measure cost only. The draft lr follows the 50-step policy warmup, so the acceptance length is 4.16 in all arms.

**A skipped step is not free.** A sync step without a drafter update still adds 2.0-3.5 s of policy training over frozen, against 19.8 s on an update step. In async math the values are 1.5-4.9 s and 37.5 s. Thus a skipped step keeps 4-18% of the drafter-training cost. On a skipped step, the draft parameters still get zero gradients through the DDP buckets, and the optimizer still runs an lr-0 step with a state snapshot and restore. If the schedule removes the draft parameters from the backward pass and the optimizer on skipped steps, this residual can go away.

**On SWE, the schedule protects drafter quality, not step time.** Drafter training adds only 1.8 s to a 151 s SWE step, so all step intervals include 1.0x. The important effect is on the drafter. Every-step training at lr 1e-4 lowers the acceptance length by 14% (3.26 to 2.80) and the model-call throughput by 11%. Interval 5 lowers the acceptance length by only 6%. The acceptance schedule trained on 13 of 20 steps, because the step-to-step noise of the SWE acceptance length is larger than its thresholds. Its acceptance length is almost the same as with every-step training. This agrees with the result above: online training does not help a drafter that already fits the workload. For a SWE-trained drafter, keep the drafter frozen. Use `interval` when the drafter must follow the policy. Use `acceptance` only with thresholds above the workload noise.

---

## Suspicious outcomes

| Outcome | Why it is suspicious | Status |
|---|---|---|
| Async OpenMath 4K arms show 1.03-1.04x E2E | Exposed generation is about 1% of the step, so the ceiling is 1.01x. The gain is noise or a side effect, not SpecDec. | explained |
| SWE scale2 frozen arms 0.90x E2E with 1.30x model calls | Final eval in steps 4-8 takes 180-193 s in both frozen arms, which ran in one window, against 71 s in the control. Tool execution is 4.6-6.0 s in all arms. | explained (final-eval window drift). In V2 window 2, final eval is 41-62 s in steps 4-8 in all four runs. |
| SWE online DFlash K3 acceptance 1.66, far above the frozen arm (1.28) | Per-step values start at 1.28 and climb to 1.76-1.89 by step 6. The drafter adapts to SWE outputs during training. | real effect, online gain |
| SWE online DSpark K4 TP4 1.07x E2E, above the frozen arm (0.90x) | Model calls are 1.30-1.31x in both, and training is 9% longer than frozen. The E2E gap is final eval in different windows (32 s against 193 s in steps 4-8). | explained, use model call 1.31x |
| SWE logprob 1.13-1.33x longer in arms | SWE skips the logprob pass, and the control timer logs 0.5-1.2 ms per step. | fixed: the dashboard leaves a ratio empty when the control phase is under 0.5 s |
| DSpark draft loss differs between implementations by about 0.26 | The teammate TV term is sum\|p-q\|. Ours is 0.5*sum\|p-q\|. | explained, values not comparable |
| Two baselines 1.3-2.7% apart | Same config, different jobs. | compare inside a cohort only |
| SC (V2) single pair | One baseline and one arm. | window 2 repeats the control (223.7 against 226.1 tok/s), the online DFlash gain and the frozen DSpark K4 model-call gain (1.33x against 1.32x). The step ratio does not repeat (1.09x against 1.17x). |
| V2 SpecDec counters are zero in 15-17 of 20 steps | The V2 path logs the vLLM SpecDec counters in only some steps. Window-2 frozen DFlash K3 gives 1.02 from 3 steps, against 1.28 in scale2, with the same 1.01x model calls. | not used: V2 rows give no acceptance length. Use model-call tok/s. |
| Q235 SWE fits 11 steps in 4 h, and E2E (1.16x) is above model call (1.14x) | Async step times alternate between 60-80 s and 870-1430 s. Steps 1-6 give 1.02x and steps 7-11 give 1.30x, so part of the E2E gain is rollout wait. | use model call 1.14x |
| Online SWE-trained DFlash K7 acceptance 2.81, below frozen 3.22 | The online drafter starts from the frozen weights. The draft loss goes from 2.60 in steps 1-10 to 2.80 in steps 11-20, and model calls drop to 1.89x in steps 11-20 against 2.04x frozen. | real effect, lr 1e-4 moves a drafter that already fits SWE |
| SWE-trained DFlash K7 step 0.98x with 2.06x model calls | Four steps with a 1200 s final-eval timeout take 54-63% of the time in all three runs. K7 has a fifth timeout at step 20. | explained, use model call 2.06x |
| SWE-trained frozen DSpark K4 step 1.10x [1.02, 1.23] | The arm ran 15 h after the control. Outside the four timeout steps, all four DSpark arms give 1.17-1.25x. Frozen K8 gives 1.03x [0.95, 1.17] in the same window. | not confirmed, use model call 1.91x |
| SWE-trained frozen DSpark K8 acceptance 3.67 | The V2 counters are nonzero in only 2 of 20 steps. | small sample, use model call 2.00x |
| Ready-first DSpark K8 token-probability error 5.34 at step 18 | The other 19 steps give 1.03, the same as the control. The generation KL error at step 18 is 0.0018. | not SpecDec: the 450 s control (no SpecDec) has a 7.57 spike at step 19, and the 450 s K8 arm has none. A rare ready-first spike. |
| Ready-first control 1.79x step against the in_order control | The two runs ran in different windows, and ready-first does not keep the same prompts in a step. The ready-first runs also do a logprob pass of about 8 s. | sampler estimate only, the SpecDec ratio pairs inside one window |
| Eval 600 s DSpark K8 step 1.01x with 1.95x model calls | The arm has a fifth 600 s timeout at step 12. | explained, the step still waits for the slowest eval |
| 8150978 step-1 generation 238x | Async step 1 pairs a buffered step with a fresh one. | excluded by the paired window |
| DAPO 32K async baseline waits at steps 3 and 5 (412 s exposed generation against 2 s) | Its slower generation has not caught up yet. From step 7 all runs alternate in phase. Steps 7-20 give DFlash 1.43x and DSpark 1.48x, against 1.41x and 1.46x over steps 3-20. | explained, gain holds in steady state |
| DAPO 32K sync pairs on 12 steps | The baseline (8146971) hit the 4 h limit at step 14. | above the 8-step floor |
| Q35 DFlash slower than control at K3 and K7 | The GDN layers give 46 KV cache groups. The GDN metadata-reuse backport gives 1.4-1.8x on rollout and lifts K3 to 1.02x. | root cause known, fix not in stock vLLM |
| Q235 sync DFlash K3 and DSpark K4 died at startup | megatron-core editable build race between nodes of one job. | fixed in 844da82424, reruns are follow-ups |
| Q235 async token-probability error spikes in 2 of 6 SpecDec arms | DFlash K3 gives 319 at step 16 and DSpark K6 gives 237.5 at step 8. All other steps of the six arms give 1.05-2.35, and the control maximum is 1.09. The spike is not monotonic in K. The generation KL error stays at 0.0068-0.0069 and the reward at 0.57 in all arms. | not root-caused. One step per run, no effect on the step time or the reward. |

---

## Follow-ups still queued

These jobs had not finished on OCI-HSG on 2026-10-25. The results above do not include them. Q235 SWE DSpark K8 is the one drafter that the Q235 SWE comparison does not have yet. The Q235 async K sweep (jobs 8146563-8146568) finished on 2026-10-25, and the tables above include it. Job 8247019 (agent startup bench) ran, but its result is not valid. It needs a fix and a rerun.

| Job | Run | Adds |
|---|---|---|
| 8150700, 8151041 | q235-sd1020-ksweep-frozen-dflash-k3-r2, dspark-k4-r2 | Q235 sync smallest K |
| 8149970 | swe235-sd1020-frozen-dspark-k8-v10 | Q235 SWE DSpark |

---

## Key Takeaway

**SpecDec speeds up GRPO by about the share of the step that is exposed decode.** It gives 1.3-1.6x E2E for long outputs and Q235. It gives 1.14-1.21x for OpenMath 4K sync on Q30 and N35, and 1.05x on Qwen3.5-35B. It gives no gain for train-bound async OpenMath 4K. DFlash and DSpark tie on OpenMath 4K. DSpark wins on long outputs (32K sync 1.55x against 1.30-1.33x, DAPO 32K async 1.46x against 1.41x) and on SWE model calls. On SWE, online co-training lifts the DFlash acceptance length from 1.28 to about 1.85 in 6 steps. It also lifts the model-call speedup from 1.00x to 1.28x over 20 steps and to 1.42x in steps 11-20. A second window repeats this (1.32x and 1.43x). Online DSpark K4 reaches 1.40x in steps 11-20, against 1.31x frozen. A DFlash drafter trained on SWE rollouts gives 1.83x at K3 and 2.06x at K7 on model calls. A DSpark drafter trained on the same rollouts gives 1.91x at K4 and 2.00x at K8. Online training at lr 1e-4 lowers all four (1.78x, 1.95x, 1.84x and 1.82x), so online co-training helps only an off-domain drafter. With the in_order sampler, the SWE step stays inside the noise, because final-eval timeouts set it. **With the ready-first sampler, frozen DSpark K8 gives 1.23x [1.19, 1.28] on the SWE step, faster in 20 of 20 steps.** Ready-first cuts the step spread from about 105% to 18%, so the 2x model-call gain shows in the step. A second pair with a 450 s eval timeout repeats it (1.22x [1.08, 1.38]), and the 35 s model-call saving becomes a 32 s step saving. With K8, OpenHands startup (51 s per trajectory) is longer than the model calls (32 s). Thus the next SWE lever is the agent startup, not the drafter. Five trajectory fixes cut the startup and give about 1.1x on the step. On that faster baseline, SpecDec gains more per trajectory (1.28x against 1.24x) but less per step (1.13x against 1.25x). The fixes and DSpark K8 together give 1.24x on the step. On math, a K below the block size is as good or slightly better. On SWE with a SWE-trained drafter, the block-size K beats the small K for both DFlash and DSpark. Thus the block-size K is a safe default, but not the best K. The next lever is not a better drafter for 4K async math, because that workload has no generation left to hide. The next lever is long-output and agentic workloads, where decode or model calls set the step time.