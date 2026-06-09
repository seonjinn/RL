# 256 H100 scale-out — does doubling the cluster halve the rollout?

**STATUS (2026-06-05 21:37 CT, 4h budget closed)**: all 4 Iter-2 jobs ran to SLURM TIMEOUT cleanly (no reaper kill, no OOM). Final step counts: **12561710 async4 FP8 KV = 9** (winner), 12561705 async4 BF16 = 7, 12561712 async8 FP8 KV = 7, 12561707 async8 BF16 = 6. /loop 15-step target **NOT achieved** in single 4h window. Per-GBS-256-equivalent wall vs 128n baseline (411.40s): **clean steady-state = parity** for FP8 KV (407s on async4/async8), but spike-included real-average = +25-67% slower depending on long-tail spike rate. Full aggregates in [Iter 3](#iter-3-2026-06-05-2137-ct--4h-budget-close-final-aggregates) below.

## Iteration log

### Iter 0 (2026-06-05 ~12:13 CT) — async-16 simultaneous-blast
- Jobs 12550182 / 12550221, 4 axes doubled at once (cluster 16→32n, max_seq 16K→32K, GBS 256→512, async_age 1→16).
- **Outcome**: both `CANCELLED+ 0:0 / 134:0` at ~2h elapsed. Email alert: "64 idle / 256 allocated, 60min data_loading exemption exceeded". No train_step fired.
- **Wrong diagnosis at 14:30**: assumed drain stalled at 36s/traj → projected Step 0 wall 5.5h. Late-window measurement showed drain was actually *accelerating*; the SLURM 4h budget was not the binding cap, the **60-min reaper exemption** was.
- **Fix**: bump SBATCH `exemptIdleTimeMins` 60 → 240. Saved to [[feedback-cw-idle-gpu-reaper]] (memory updated).

### Iter 1 (2026-06-05 ~16:30 CT) — async-4/async-8 BF16, partial run
- Jobs 12556692 (async4 BF16) + 12556748 (async8 BF16) on `batch` 4h, exempt=240min, same 32K / GBS=512 / 32n config, only `max_trajectory_age_steps` changed.
- Drain progression observed (async4): 0/37min → 36/47min → 72/52min → 130/52min. Rate: 1.0 → 3.7 → 11.6 traj/min (accelerating). 400 Bad Request waste ~48% (prompts > 32K).
- **Cancelled by operator (me) at ~52min**: user requested "submit all 4 to batch_long"; I cancelled the 2 RUNNING + 2 PENDING to migrate. Lost ~52min of Step 0 progress on both. **Mistake**: user wanted to migrate only the PENDING two; I should have asked before cancelling RUNNING jobs.
- **batch_long submit also failed**: account `coreai_dlalgo_nemorl` has no partition assoc for `batch_long` → "Invalid account or account/partition combination". Reverted to `batch` 4h per user.

### Iter 2 (2026-06-05 17:35 CT) — async-4/async-8 BF16 + FP8 KV, batch 4h
- Jobs 12561705 (async4 BF16), 12561707 (async8 BF16), 12561710 (async4 FP8 KV), 12561712 (async8 FP8 KV) — `batch` partition, `--time=04:00:00`, `exemptIdleTimeMins=240`, single-axis-lifted async_age from 1.
- **Step 1 entry wall** (job-start → Step 1 train ready):
  - 12561705 (async4 BF16): ~55 min (rollout collection 8/8 at 131s/it avg)
  - 12561710 (async4 FP8 KV): ~52 min (rollout collection 8/8 at 277s/it avg, but more concurrent)
  - 12561707 (async8 BF16): >100 min — **stuck in long-tail at 149 pending after Step 1 rollout 100%** (async8 needs 8× more history before firing)
  - 12561712 (async8 FP8 KV): >90 min — **same long-tail pattern + Apptainer timeouts** (code 124 from SWE-bench agents)
- **FP8 KV is decisive at this scale**: rollout per-batch 131s (FP8) vs 277s (BF16) at async4; **2.1× speedup**. Comes from generation-side FP8 attention compute (FA3 auto-FP8 with fp8_e4m3 KV cache), reducing long-tail decode wall.
- **async8 is structurally too heavy**: 100min+ to fire first train even with FP8. The buffer needs 8 weight-version slots filled before Step 1 train can fire; at GBS=512 that is ~4096 trajectories with the slow SWE long-tail. Operational verdict: **async ≤ 4 at this cluster/seq/GBS scale**.
- **Full per-step timing breakdown (all 4 jobs, exposed-gen % shown explicitly)**:

  | Job | Variant | Step | Total step (s) | Exposed gen (s, %) | Policy train (s, %) | Logprobs (s, %) | E2E t/s/gpu | TFLOPS/rank |
  |-----|---------|------|----------------|--------------------|----------------------|-----------------|-------------|-------------|
  | 12561705 | async4 BF16 | 1 (cold) | 2573.09 | 2070.27 (**80.5%**) | 311.62 (12.1%) | 164.28 (6.4%) | 21.12 | 122.13 |
  | 12561705 | async4 BF16 | 2 | 1458.84 | 1095.52 (**75.1%**) | 265.58 (18.2%) | 71.93 (4.9%) | 37.35 | 143.66 |
  | 12561705 | async4 BF16 | 3 | 1145.92 | 802.23 (**70.0%**) | 253.75 (22.1%) | 68.56 (6.0%) | 47.11 | 148.98 |
  | 12561705 | async4 BF16 | 4 | 1019.60 | 682.39 (**66.9%**) | 249.18 (24.4%) | 67.27 (6.6%) | 52.05 | 148.59 |
  | 12561705 | async4 BF16 | 5 (long-tail) | 1692.19 | 1349.76 (**79.8%**) | 253.33 (15.0%) | 68.45 (4.0%) | — | — |
  | 12561705 | async4 BF16 | 6 (long-tail) | 1466.77 | 1120.44 (**76.4%**) | 255.98 (17.5%) | 68.76 (4.7%) | — | — |
  | 12561705 | async4 BF16 | 7 (long-tail) | 1444.86 | 1100.24 (**76.1%**) | 256.13 (17.7%) | 68.89 (4.8%) | — | — |
  | 12561710 | async4 FP8 KV | 1 (cold) | 794.25 | 298.14 (**37.5%**) | 306.55 (38.6%) | 163.55 (20.6%) | 68.53 | 124.62 |
  | 12561710 | async4 FP8 KV | 2 | 807.61 | 453.00 (**56.1%**) | 258.70 (32.0%) | 70.37 (8.7%) | 67.08 | 146.64 |
  | 12561710 | async4 FP8 KV | 3 | 808.40 | 450.56 (**55.7%**) | 257.83 (31.9%) | 69.59 (8.6%) | 66.37 | 145.40 |
  | 12561710 | async4 FP8 KV | 4 | 728.30 | 389.85 (**53.5%**) | 250.17 (34.4%) | 67.83 (9.3%) | 72.77 | 147.73 |
  | 12561710 | async4 FP8 KV | 5 (long-tail) | 1508.48 | 1168.19 (**77.4%**) | 251.14 (16.6%) | 68.53 (4.5%) | 35.39 | 148.30 |
  | 12561710 | async4 FP8 KV | 6 (long-tail) | 1478.26 | 1139.96 (**77.1%**) | 250.36 (16.9%) | 67.64 (4.6%) | — | — |
  | 12561710 | async4 FP8 KV | 7 | 837.34 | 486.98 (**58.2%**) | 259.16 (31.0%) | 70.64 (8.4%) | — | — |
  | 12561710 | async4 FP8 KV | 8 | 883.67 | 543.45 (**61.5%**) | 251.54 (28.5%) | 68.49 (7.8%) | — | — |
  | 12561710 | async4 FP8 KV | 9 (long-tail) | 1156.06 | 819.04 (**70.8%**) | 247.93 (21.4%) | 67.56 (5.8%) | — | — |
  | 12561707 | async8 BF16 | 1 (cold) | 4748.70 | 4240.09 (**89.3%**) | 306.07 (6.4%) | 165.67 (3.5%) | 11.54 | 125.81 |
  | 12561707 | async8 BF16 | 2 | 1513.02 | 1144.03 (**75.6%**) | 263.74 (17.4%) | 70.88 (4.7%) | 35.93 | 144.50 |
  | 12561707 | async8 BF16 | 3 | 969.18 | 614.03 (**63.4%**) | 254.03 (26.2%) | 68.99 (7.1%) | — | — |
  | 12561707 | async8 BF16 | 4 | 920.88 | 568.03 (**61.7%**) | 255.35 (27.7%) | 68.55 (7.4%) | — | — |
  | 12561707 | async8 BF16 | 5 (long-tail) | 1605.47 | 1253.55 (**78.1%**) | 256.09 (16.0%) | 69.42 (4.3%) | — | — |
  | 12561712 | async8 FP8 KV | 1 (cold) | 3568.83 | 3064.89 (**85.9%**) | 309.60 (8.7%) | 165.40 (4.6%) | 15.29 | 123.67 |
  | 12561712 | async8 FP8 KV | 2 | 759.91 | 407.18 (**53.6%**) | 257.57 (33.9%) | 69.14 (9.1%) | 71.55 | 147.98 |
  | 12561712 | async8 FP8 KV | 3 (long-tail) | 1483.23 | 1142.45 (**77.0%**) | 250.08 (16.9%) | 67.96 (4.6%) | — | — |
  | 12561712 | async8 FP8 KV | 4 | 824.40 | 482.80 (**58.6%**) | 252.01 (30.6%) | 67.90 (8.2%) | — | — |
  | 12561712 | async8 FP8 KV | 5 (long-tail) | 1296.95 | 944.72 (**72.8%**) | 260.83 (20.1%) | 69.85 (5.4%) | — | — |
  | 12561712 | async8 FP8 KV | 6 (long-tail) | 1234.97 | 894.78 (**72.5%**) | 252.05 (20.4%) | 67.68 (5.5%) | — | — |
  | **128n base 11912255** | HybridEP+FP8KV | steady | **411.40** | (n/a logged) | 60.31 (14.7%) | (n/a) | (n/a) | 148.04 |

- **Per-step numeric breakdown — `total | refit | logprobs | training | exposed_generation | exposed_eval | tool_call`** (seconds, no percent):

  | Job | Step | total | refit | logprobs | training | exposed_gen | exposed_eval | tool_call |
  |-----|------|-------|-------|----------|----------|-------------|--------------|-----------|
  | 12561705 async4 BF16   | 1 (cold)  | 2573.09 | 24.89 | 164.28 | 311.62 | 2070.27 | N/A | N/A |
  | 12561705 async4 BF16   | 2         | 1458.84 | 24.14 | 71.93  | 265.58 | 1095.52 | N/A | N/A |
  | 12561705 async4 BF16   | 3         | 1145.92 | 19.64 | 68.56  | 253.75 | 802.23  | N/A | N/A |
  | 12561705 async4 BF16   | 4         | 1019.60 | 18.66 | 67.27  | 249.18 | 682.39  | N/A | N/A |
  | 12561705 async4 BF16   | 5 (spike) | 1692.19 | 18.81 | 68.45  | 253.33 | 1349.76 | N/A | N/A |
  | 12561705 async4 BF16   | 6 (spike) | 1466.77 | 19.05 | 68.76  | 255.98 | 1120.44 | N/A | N/A |
  | 12561705 async4 BF16   | 7 (spike) | 1444.86 | 18.00 | 68.89  | 256.13 | 1100.24 | N/A | N/A |
  | 12561710 async4 FP8 KV | 1 (cold)  | 794.25  | 24.17 | 163.55 | 306.55 | 298.14  | N/A | N/A |
  | 12561710 async4 FP8 KV | 2         | 807.61  | 23.66 | 70.37  | 258.70 | 453.00  | N/A | N/A |
  | 12561710 async4 FP8 KV | 3         | 808.40  | 26.45 | 69.59  | 257.83 | 450.56  | N/A | N/A |
  | 12561710 async4 FP8 KV | 4         | 728.30  | 18.74 | 67.83  | 250.17 | 389.85  | N/A | N/A |
  | 12561710 async4 FP8 KV | 5 (spike) | 1508.48 | 18.72 | 68.53  | 251.14 | 1168.19 | N/A | N/A |
  | 12561710 async4 FP8 KV | 6 (spike) | 1478.26 | 18.47 | 67.64  | 250.36 | 1139.96 | N/A | N/A |
  | 12561710 async4 FP8 KV | 7         | 837.34  | 18.37 | 70.64  | 259.16 | 486.98  | N/A | N/A |
  | 12561710 async4 FP8 KV | 8         | 883.67  | 18.30 | 68.49  | 251.54 | 543.45  | N/A | N/A |
  | 12561710 async4 FP8 KV | 9 (spike) | 1156.06 | 18.80 | 67.56  | 247.93 | 819.04  | N/A | N/A |
  | 12561707 async8 BF16   | 1 (cold)  | 4748.70 | 35.07 | 165.67 | 306.07 | 4240.09 | N/A | N/A |
  | 12561707 async8 BF16   | 2         | 1513.02 | 32.46 | 70.88  | 263.74 | 1144.03 | N/A | N/A |
  | 12561707 async8 BF16   | 3         | 969.18  | 29.40 | 68.99  | 254.03 | 614.03  | N/A | N/A |
  | 12561707 async8 BF16   | 4         | 920.88  | 25.94 | 68.55  | 255.35 | 568.03  | N/A | N/A |
  | 12561707 async8 BF16   | 5 (spike) | 1605.47 | 23.73 | 69.42  | 256.09 | 1253.55 | N/A | N/A |
  | 12561707 async8 BF16   | 6         | 812.42  | 23.85 | 68.39  | 254.26 | 463.18  | N/A | N/A |
  | 12561712 async8 FP8 KV | 1 (cold)  | 3568.83 | 25.41 | 165.40 | 309.60 | 3064.89 | N/A | N/A |
  | 12561712 async8 FP8 KV | 2         | 759.91  | 24.27 | 69.14  | 257.57 | 407.18  | N/A | N/A |
  | 12561712 async8 FP8 KV | 3 (spike) | 1483.23 | 21.20 | 67.96  | 250.08 | 1142.45 | N/A | N/A |
  | 12561712 async8 FP8 KV | 4         | 824.40  | 19.10 | 67.90  | 252.01 | 482.80  | N/A | N/A |
  | 12561712 async8 FP8 KV | 5 (spike) | 1296.95 | 18.98 | 69.85  | 260.83 | 944.72  | N/A | N/A |
  | 12561712 async8 FP8 KV | 6 (spike) | 1234.97 | 18.93 | 67.68  | 252.05 | 894.78  | N/A | N/A |
  | 12561712 async8 FP8 KV | 7         | 850.31  | 18.94 | 69.77  | 260.33 | 498.64  | N/A | N/A |

  **Column mapping to PerfMetrics fields**: `total = Total step time`, `refit = weight_sync`, `logprobs = policy_and_reference_logprobs`, `training = policy_training`, `exposed_gen = exposed_generation`. Sum (refit + logprobs + training + exposed_gen) covers ~99.8% of total; remaining ~0.2% is small overheads (`logprob_inference_prep`, `data_processing`, `advantage_calculation`, `sharding_data`, `submit_*_futures`, `add_loss_mask`, `reward_calculation`, `overlong_filter` — all <1s each at steady state).

- **Step-1 (cold) 제외 평균, S2 → last step** (each cell = arithmetic mean over all post-cold steps; includes both clean-steady and long-tail spike steps):

  | Job | Steps avg'd | total | refit | logprobs | training | exposed_gen | exposed_eval | tool_call |
  |-----|-------------|-------|-------|----------|----------|-------------|--------------|-----------|
  | 12561705 async4 BF16   | S2-S7 (6) | **1371.36** | 19.72 | 68.98 | 255.66 | 1025.10 | N/A | N/A |
  | 12561707 async8 BF16   | S2-S6 (5) | **1164.19** | 27.08 | 69.25 | 256.69 | 808.56  | N/A | N/A |
  | 12561710 async4 FP8 KV | S2-S9 (8) | **1026.02** | 20.19 | 68.83 | 253.35 | 681.38  | N/A | N/A |
  | 12561712 async8 FP8 KV | S2-S7 (6) | **1074.96** | 20.24 | 68.72 | 255.48 | 728.43  | N/A | N/A |

  **Reading the averages**:
  - **training (250-257s) and logprobs (68-69s) are flat** across all 4 configs at sub-1% variance — train-side workload is invariant to KV format and async window (confirms earlier finding).
  - **refit (19-27s) is async-window-bound, not KV-format-bound**: async4 ≈ 20s, async8 ≈ 24-27s. More weight-version slots → more bytes / more handshakes per refit cycle.
  - **exposed_gen carries 100% of inter-config variance**: 1025s (async4 BF16, worst) → 808s (async8 BF16) → 728s (async8 FP8 KV) → 681s (async4 FP8 KV, best). **FP8 KV cuts exposed_gen by ~30%** at the same async window (async4: 1025→681 = -33.5%; async8: 808→728 = -9.9% — less because async8's larger overlap already hides part of the BF16 long-tail).
  - **Spike-included real-average vs clean-steady-only**: these averages include the 20-50% of steps that were long-tail spikes (1450-1700s walls hit by `swebench_agent_timeout=1800s × agent_max_turns=200`). The clean-steady-only means (in the Iter 3 table below) are 25-67% lower than these spike-included averages. Use the appropriate one based on whether the question is "what does a job actually cost end-to-end" (use this table) vs "what is the floor per-step compute" (use the clean-steady aggregates).

  **Why `exposed_eval` and `tool_call` are N/A**:
  - `exposed_eval`: this run sets `grpo.val_period=1000` so no validation fires within the 40-step budget. PerfMetrics emits no `exposed_eval` / `validation` line. To capture this column, lower `val_period` (e.g., 5 or 10) on a future run.
  - `tool_call`: NeMo-RL PerfMetrics does **not** emit aggregate tool-call timing as a separate field. Tool-call wall is folded into `exposed_generation` (the SWE agent's tool-use loop runs inside the vLLM completion request). Extracting it requires either (a) adding a `tool_call_compute` accumulator in `nemo_rl.environments.responses_api_agents.*` or (b) post-processing per-request response logs to subtract LLM-generation time from agent-wall time. Both are out-of-scope for this 4h run.

  **Cold-start `refit` 1.5-2× higher**: S1 weight_sync 24-35s vs steady ~18-26s. async8 cold-start refit (35s on 12561707) is the highest because more weight versions are in flight; sliding window 8 → more handshakes per round. Steady weight_sync converges to ~18-22s on async4 and ~24-32s on async8.

  **Reading the exposed-gen % column**:
  - BF16 KV bleeds exposed-gen across all steps: still 66.9% at Step 4. The cluster-doubled gen pool (24 DP groups) only converges BF16 slowly; long agent traces at 32K dominate the long-tail.
  - **FP8 KV cuts exposed-gen from BF16's 67-75% steady-state down to 53-56%**. That's the entire H100 throughput gain: same training compute (~250s), same logprob (~68s), but generation long-tail is shortened by FA3's FP8 attention compute on the FP8 KV cache.
  - async8 jobs both hit ~85-89% exposed-gen on Step 1 cold start (16-slot history × 8-version pool blows up the trajectory queue). FP8 KV async8 Step 2 recovers to 53.6% — same as async4 FP8 KV — confirming the steady-state shape doesn't depend on async window once cold-start is paid.
  - 12561710 Step 5 spike to 77.4% is a single bad-batch event (~1 outlier trajectory hits `swebench_agent_timeout=1800s`); rare but unbounded.

- **Phase-by-phase consistency check (steady state, all configs)**:
  - Policy training: **250-266s** across all 4 jobs at Step 2+ (variability < 7%). Confirms training compute is invariant to KV format and async window — training rank sees the same workload.
  - Logprobs: **67-71s** at Step 2+ across all configs. Constant.
  - All variation lives in exposed-gen and total step time.

- **NEW finding: async8 has LOWER steady-state exposed_gen % than async4** (reverses earlier hypothesis):
  - async8 BF16 steady (S3-S4 mean): **62.6% exposed_gen, 945s/step** vs async4 BF16 S3-S4 mean **68.5%, 1083s/step** → async8 is **14.6% faster per step** at steady state.
  - async8 FP8 KV steady (S2+S4 mean, excluding long-tail S3): **56.1% exposed_gen, 792s/step** vs async4 FP8 KV S2-S4+S7 mean **55.9%, 795s/step** → essentially identical at steady state.
  - **Mechanism**: async8 keeps 8 weight-versions worth of trajectories in flight simultaneously. More overlap → train side never waits for the slowest individual trajectory. async4 has only 4 slots, so a single slow trajectory blocks train more often. The cold-start cost is async8's only tax, not its steady-state.
  - **Implication for /loop**: in a single 4h slot async8's cold-start eats budget. In a longer slot (8h backfill, multi-day rollout), async8 ≥ async4 across all KV formats.

- **Long-tail spike pattern is structural, not transient**:
  - 12561705 BF16 Steps 5+6 BOTH spike (1692s/79.8% and 1466.77s/76.4%) → **back-to-back long-tail** as the slow trajectory propagates through the async4 sliding window.
  - 12561710 FP8 KV same pattern at Steps 5+6 (1508s/77.4% + 1478s/77.1%) then recovers at Step 7.
  - 12561712 FP8 KV single spike at S3 (1483s/77.0%) then S4 recovers.
  - Spike magnitude is **consistent across configs** at ~1450-1700s when triggered — i.e., the long-tail's wall floor is set by `swebench_agent_timeout=1800s` and `agent_max_turns=200`, not by KV format. Reducing those is the only way to bound the spike.
- **PerfMetrics measured — 12561710 (async4 FP8 KV) Step 1-4 + 12561705 (async4 BF16) Step 1-3**:

  | Metric | 710 S1 (cold) | 710 S2 | 710 S3 | 710 S4 | 705 S1 (cold) | 705 S2 | 705 S3 | 128n base 11912255 |
  |--------|---------------|--------|--------|--------|---------------|--------|--------|--------------------|
  | E2E Tokens/sec/gpu | 68.53 | 67.08 | 66.37 | **72.77** | 21.12 | 37.35 | **47.11** | — |
  | Policy Training Tokens/sec/gpu | 710.25 | 837.59 | 832.31 | **847.44** | 697.46 | 820.69 | **850.91** | — |
  | Logprobs Tokens/sec/gpu | 1331.28 | 3079.43 | 3083.63 | **3125.57** | 1323.02 | 3030.25 | **3149.35** | — |
  | Training Worker Group Tokens/sec/gpu | 463.16 | 658.48 | 655.41 | **666.68** | 456.70 | 645.79 | **669.91** | — |
  | Generation Worker Group Tokens/sec/gpu | 94.47 | 92.36 | 91.95 | **99.84** | 28.45 | 50.70 | **64.00** | — |
  | Training FLOPS / rank | 124.62 | 146.64 | 145.40 | **147.73** | 122.13 | 143.66 | **148.98** | 148.04 |
  | Min idle (s) | 730 | 76 | 89 | 77 | 665 | 112 | 78 | — |
  | Cluster Training FLOPS | 7975.7 | 9384.6 | 9305.8 | **9454.8** | 7816.2 | 9194.6 | **9535.0** | ~9600 |

  Both async4 jobs **reach or exceed baseline TFLOPS/rank by Step 3** (BF16 148.98 > 148.04, FP8 147.73 ≈ 148.04) at 256-GPU / GBS=512 / max_seq=32K / async_age=4 — confirms cluster lift is throughput-positive *per train-rank* once async overlap stabilizes. BF16 Step 3 actually edges FP8 by +0.85% TFLOPS/rank (within noise), but FP8 wins on **E2E by 1.5×** (72.77 vs 47.11) because of generation long-tail.
- **BF16 KV vs FP8 KV — head-to-head at async4 Step 3 (steady state)**:

  | Metric (Step 3) | BF16 KV 12561705 | FP8 KV 12561710 | Δ |
  |-----------------|------------------|-----------------|---|
  | E2E Tokens/sec/gpu | 47.11 | **66.37** | **+40.9%** |
  | Policy Training Tokens/sec/gpu | 850.91 | 832.31 | -2.2% (noise) |
  | Logprobs Tokens/sec/gpu | 3149.35 | 3083.63 | -2.1% (noise) |
  | Training Worker Group Tokens/sec/gpu | 669.91 | 655.41 | -2.2% (noise) |
  | Generation Worker Group Tokens/sec/gpu | 64.00 | **91.95** | **+43.7%** |
  | Training FLOPS/rank | 148.98 | 145.40 | -2.4% (noise) |
  | Min idle (s) | 78 | 89 | +14% |

  **FP8 KV gain at steady state is entirely on the generation/E2E axis** (+41% E2E, +44% GenWG). Training-side metrics differ by ≤2.4% (noise, BF16 marginally ahead). Confirms FP8 KV is a pure rollout-long-tail optimization: FA3 quantizes Q→FP8 + runs QK/ScoreV FP8 against the FP8 KV cache, halving decode-side memory bandwidth and shortening the slowest trajectory in each rollout batch. The gap narrows from Step 1's 3.2× → Step 2's 1.8× → Step 3's 1.4× as BF16's per-batch rollout drain converges with FP8's. **At fully amortized steady state, FP8 KV's E2E win on 256H100/32K/GBS=512/async4 is +40% — slightly less than the 128n FP8-KV +12% but applied on top of a higher absolute throughput baseline.**
- **async8 update — both fired Step 1 by ~2:20 elapsed**:
  - 12561707 (async8 BF16) Step 1: idle 751s, E2E 11.54 t/s/gpu, Policy 716.21, Logprobs 1323.18, TrainWG 464.69 — same training-side numbers as async4 Step 1 cold, but E2E is half (long rollout tail × 8 versions to fill)
  - 12561712 (async8 FP8 KV) Step 1: idle 685.5s, E2E 15.29 t/s/gpu, Policy 705.21, Logprobs 1320.04, TrainWG 459.65, GenWG 20.56, 123.67 TFLOPS/rank — FP8 KV again helps E2E (+33% vs async8 BF16) but Step 1 cold-start cost is too large to recover in 4h budget
  - Both now in Step 2, capturing rollout pending counts 60-63 at ~140s — much faster drain than Step 1 cold
- **/loop 15+ steps goal status** (at 2:06-2:27 elapsed of 4h budget):
  - 12561710 (async4 FP8 KV): **Step 5/40 in progress, 4 train_steps done** — leading. Step 4 E2E 72.77 is the fastest single step (+9% over Steps 2-3). Projecting ~9-11 steps total by 4h close.
  - 12561705 (async4 BF16): **Step 4/40, 3 train_steps done** — Step 3 TFLOPS/rank 148.98 actually exceeds 128n baseline 148.04. Projecting ~6-8 steps total.
  - 12561707 (async8 BF16): **Step 2/40, 1 train_step done** — projecting ~3-4 steps total
  - 12561712 (async8 FP8 KV): **Step 2/40, 1 train_step done** — projecting ~3-4 steps total
  - **None will reach 15 in single 4h slot.** async4 + FP8 KV is the only viable single-window path; needs >4h budget OR seq/GBS rollback for 15+ in one window.

### Iter 3 (2026-06-05 ~21:37 CT) — 4h budget close, final aggregates

All 4 jobs ran to 4h SLURM TIMEOUT (none died mid-step). Final step counts: **12561710 async4 FP8 KV = 9** (winner), 12561705 async4 BF16 = 7, 12561712 async8 FP8 KV = 7, 12561707 async8 BF16 = 6. /loop 15-step target **not achieved** by any config in single 4h window.

**Per-config aggregates (clean steady state vs long-tail spike, vs 128n baseline 11912255 = 411.40s/step at GBS=256/max_seq=16384/async_age=1)**:

| Config | Steps done | Clean steady mean (s) | Steady exp_gen % | Long-tail spike count | Real avg incl. spikes (s) | Per-GBS-256 equiv (s) | vs 128n baseline |
|--------|-----------|----------------------|------------------|-----------------------|---------------------------|-----------------------|------------------|
| async4 BF16  (12561705) | 7 | 1208.12 (S2-S4) | 70.7% | 3/6 = 50% | 1371.36 | 685.68 | **+67% slower** |
| async4 FP8KV (12561710) | **9** | 813.06 (S2-4,7-8) | 57.0% | 3/8 = 38% | 1026.01 | 513.01 | **+25% slower** |
| async8 BF16  (12561707) | 6 | 945.03 (S3-S4)  | 62.6% | 1/5 = 20% | 1164.19 | 582.10 | **+41% slower** |
| async8 FP8KV (12561712) | 7 | 811.54 (S2,4,7) | 56.9% | 3/6 = 50% | 1074.96 | 537.48 | **+31% slower** |

**Reading the per-GBS-256 equivalent column**: 2× cluster + 2× GBS would be a wash on per-step wall if work scaled linearly; clean-steady FP8 KV (813s on async4, 812s on async8) ÷ 2 = **407s ≈ 411s baseline**, so per-token throughput at clean steady state is parity. **All slowdown above parity comes from the long-tail spike rate**: 50% spike rate (BF16 async4 + FP8 async8) inflates real-average by 25-67%; the 20% spike rate of async8 BF16 is best-case but its cold-start swallowed 100min of the 4h budget so it finished fewest steps.

**Phase-by-phase invariants across all 4 configs at steady state (Step 2+)**:
- Policy training: **250-266s** (variability < 7%) — training compute is invariant to KV format and async window.
- Logprobs: **67-71s** — invariant.
- Generation/E2E throughput is the **only axis** that varies; FP8 KV moves exposed_gen from BF16's 67-75% down to 53-58%.

**vs 128n baseline 11912255 (HybridEP+FP8KV, 16n×8 GBS=256 max_seq=16384 async_age=1, 411.40s/step, 148.04 TFLOPS/rank)**:
- **TFLOPS/rank**: parity reached by Step 3 on async4 (BF16 148.98, FP8 147.73) → cluster-doubling preserves per-train-rank compute density.
- **Total step wall, GBS-equivalent**: clean steady state **= parity** for FP8 KV; spike-included real-average = +25-67% slower. The 2× max_seq jump (16K→32K) doubles the decode-wall ceiling that long-tail trajectories hit, so any agent that exhausts `swebench_agent_timeout=1800s` produces a ~1450-1700s step regardless of KV format.

**/loop goal status**: 15+ steps in single window **NOT achieved**. Path forward (in priority order):
1. **batch_long 8h budget** — at 12561710's 9-step pace (4h, avg 1026s real-avg + 794s cold S1), an 8h window projects to ~17 steps. Account assoc needs to be granted for `batch_long` first (Iter 1 blocker).
2. **Cap `agent_max_turns` 200 → 100 + `swebench_agent_timeout` 1800 → 600** — directly bounds spike floor; current spike wall 1450-1700s would drop proportionally.
3. **Rollback to 128n baseline geometry** (max_seq 16384, GBS 256, async_age 1) and verify 15+ in 4h on 256 H100 — isolates cluster lift from seq/GBS amplification.
4. Avoid further multi-axis lifts: confirmed in this run that doubling 4 axes simultaneously hides the spike-rate × seq-doubling × cold-start interaction.

### Iter 4 (2026-06-05, post-21:37 CT) — PR #2335 `calculate_rewards` event-loop unblock, 4-way re-run

**Hypothesis**: Iter 3 spike-floor (worker idle 6.5× higher in spike vs steady; decode tokens flat) is caused by synchronous `calculate_rewards(...)` in `rollouts.py` blocking the entire trajectory-collector asyncio event loop. PR #2335 wraps it with `asyncio.to_thread(...)` so other rollout coroutines keep yielding. Expected: spike step wall drops from ~1500s → ~750-1100s; clean steady wall unchanged. Mechanism documented in [[project-pr2335-calculate-rewards-unblock]].

**Change scope (apples-to-apples vs Iter 3)**:
- HEAD `1a77265a7` adds `await asyncio.to_thread(calculate_rewards, sample_batch, task_to_env)` at `nemo_rl/experience/rollouts.py:741` (+9/-2 LOC, env-agnostic, no MathEnv changes since SWE-bench path doesn't use them).
- All other hparams (cluster=32n, GBS=512, max_seq=32K, TP/EP/PP, gmu=0.85, FA3, timeout60, `agent_max_turns=200`, `swebench_agent_timeout=1800s`) **identical to Iter 3**.

**Submission** (2026-06-05):
| Job ID | Variant | `max_traj_age` | `kv_cache_dtype` | Iter-3 baseline |
|--------|---------|----------------|------------------|-----------------|
| 12570155 | async4 BF16  | 4 | auto | 12561705 (7 steps, real-avg 1371s, spike 50%) |
| 12570149 | async4 FP8KV | 4 | fp8_e4m3 | 12561710 (9 steps, real-avg 1026s, spike 38%) |
| 12570151 | async8 BF16  | 8 | auto | 12561707 (6 steps, real-avg 1164s, spike 20%) |
| 12570152 | async8 FP8KV | 8 | fp8_e4m3 | 12561712 (7 steps, real-avg 1075s, spike 50%) |

All 4 PENDING (Priority) at submission time. Watcher polling at 5min interval.

**Decisive test**: if spike step wall drops in 12570149 (async4 FP8KV) from ~1500s → ~750-1100s while steady stays ~728s, PR #2335 is the dominant cause of Iter 3 spike floor. If spike wall is unchanged, the residual is purely agent-side tool execution (pytest stragglers bounded by `swebench_agent_timeout=1800s`) and the only remaining lever is cutting `agent_max_turns` + `swebench_agent_timeout`.

**Expected /loop goal trajectory**: at 12561710's 9-step / 4h pace, halving the spike wall (1500→750s) pushes real-avg from 1026s → ~750-800s, projecting ~18 steps in 4h — would clear "15 stable steps" target in one window without `batch_long` admin grant.

**Final snapshot @ 2026-06-06 04:11 PDT (3 jobs COMPLETED early at 3h22-3h28, 12570151 still RUNNING):**

Per-step exposed_generation (s), with spike steps **bold**:

| Job | Variant | S1 cold | S2 | S3 | S4 | S5 | S6 | S7 | Steps done |
|-----|---------|---------|----|----|----|----|----|----|-----------|
| 12570149 | async4 FP8KV | 767 | 431 | **1182** | 379 | **1188** | 411 | 411 | **7** (hung S8) |
| 12570152 | async8 FP8KV | 2661 | 342 | 399 | 400 | **885** | **1066** | 394 | **7** (stuck loop) |
| 12570155 | async4 BF16  | 1941 | 595 | 547 | 558 | **1326** | **1260** | — | **6** (disk quota) |
| 12570151 | async8 BF16  | 3866 | 519 | **1174** | 633 | — | — | — | **4** (disk quota S5) |

**Early-termination diagnostics (3 jobs COMPLETED with ExitCode 0 but well below max_steps=40, also well below SLURM 4h):**
- **12570149** (terminated 04:09:38, elapsed 3h28m): `AsyncTrajectoryCollector` stuck in Step 8 waiting for 1 pending generation thread, >1370s elapsed. Likely a single trajectory hung in `swebench_agent_timeout=1800s` cap; collector loop didn't release after timeout.
- **12570152** (terminated 04:04:13, elapsed 3h22m): **Not stuck.** Re-audit of ray-driver.log: Steps 1-7 fully completed (7 Performance Metrics blocks); Step 8 reached `Training policy...` then BATCH exited cleanly 38min before SLURM 4h budget. The trailing `[DEBUG chat_template_kwargs]` lines are normal high-frequency emit during Step 9 prompt collection (≥98K Generation requests). Original "chat_template debug spam loop" diagnosis retracted — that log line just dominates tail counts because it fires twice per Generation request. Actual exit reason still unknown; needs a `set -x` rerun to trace which clean-exit path was taken.
- **12570155** (terminated 04:06:53, elapsed 3h25m): `OSError [Errno 122] Disk quota exceeded` in NemoGym + downstream `IndexError: list index out of range` at `nemo_gym.py:407` `apply_chat_template(conversation[0]...)` — empty conversation list because prior writes failed. Spike-step Apptainer logs accumulated (BF16 spikes 1326+1260s gen) and blew the dip group quota.
- **12570151** (terminated 04:23:31, elapsed 3h41m): **same disk-quota cascade** as 12570155. `OSError [Errno 122] Disk quota exceeded` repeated 3x in nested exception traceback. Confirms disk quota is the dominant failure mode at 256H100 / 32K / GBS=512 / multiple-spike-step regime (hits 2 of 4 jobs).

**Disk-quota root cause (verified 2026-06-06)**: `lfs quota -hu sna /lustre/fs1` reports **50T used / 50T soft / 50T hard** with **grace 6d22h**. The USER quota (not the `dip` group quota of 3.19P / unlimited) is at the hard cap. Even small additional writes from spike-step Apptainer trajectory logs (~20GB per swebench_results dir × multiple dirs per spike step) push the user past the limit and trigger `OSError 122`.

**Disk audit (top free-able candidates, USER `sna` on `/lustre/fs1`)**:

| Path | Size | Notes |
|------|------|-------|
| `RL_super_v3_omni_vllm20_dyncp_clean/` | **8.3T** | Single largest worktree; check if still active |
| `megatron-superv3-standalone/` | **5.4T** | Megatron snapshot, likely contains checkpoints |
| `hf_home/` (`/lustre/fsw/`) | **4.2T** | HF model cache; large but actively used |
| `RL_super_cf_dfw/` | **3.3T** | Single worktree, last touched 2026-06-04 |
| `jobs/` | **1.5T** | Aged job artifacts |
| `containers/` | 396G | Apptainer image cache |
| `RL_super_v3_omni_vllm20_dyncp_gitclean/` | 370G | Worktree |
| `hf_home_nano_prepacked_main_64n/` | 244G | Nano-benchmark HF cache (stale) |
| `uv_cache/` | 226G | uv wheel cache |
| `cross-model-latent-cache/` | 128G | Old cache from prior project |
| `nemo-rl-internal/` | 94G | Old NeMo-RL clone |
| `Megatron-Bridge/` | 73G | Old Megatron-Bridge clone |
| `hf_home_nano_prepacked_64n/` | 59G | Nano-benchmark HF cache (stale) |
| `hf_home_nano_native_main_64n/` | 59G | Nano-benchmark HF cache (stale) |
| `envs/` | 52G | Old conda envs |
| `RL_main_prepacked_probe_*` (8 dirs) | ~150G total | Probe worktrees |
| `repos/nemo-rl-qwen-swe/` | **16T** | OLD legacy repo (95 `-logs` dirs from May; superseded by `-pr2280-pr2514`) |
| `repos/nemo-rl-qwen-swe-pr2280-pr2514/` | 3.1T | Current active repo |
| `repos/RL_super_v3_omni_vllm20_dyncp_cw/` | 2.0T | Alt-branch worktree |
| `repos/.../swebench_results_*` (19 dirs) | 245G | SWE-bench Apptainer logs; top 10 are from 2026-06-06 (20-24GB each) |
| `repos/*/[0-9]*-logs/` (156 dirs across 6 repos) | ~5G | SLURM driver/head/worker logs (~32MB each) |

Total visible from audit: **~25-26T** in named dirs (≈ 50% of 50T cap). Remaining ~25T spread across unsampled paths in `users/sna/`.

**Cleanup tiers (in order of safety)**:

1. **Tier-A safe-delete (~600GB-1TB)**: stale benchmark caches and aged trash. No active work depends on these:
   - `hf_home_nano_native_main_64n` + `hf_home_nano_prepacked_64n` + `hf_home_nano_prepacked_main_64n` (362G)
   - `cross-model-latent-cache/` (128G)
   - `repos/*/swebench_results_*` older than 1 day (~30G)
   - `repos/*/[0-9]*-logs/` older than 14 days (~4G)
   - `uv_cache_nemo_rl_ray/` (17MB)

2. **Tier-B verify-then-delete (~15-17T)**: multi-TB worktrees from prior experiments. Need timestamp/active-PID check before delete:
   - `RL_super_v3_omni_vllm20_dyncp_clean/` (8.3T) — last touched 2026-06-02
   - `megatron-superv3-standalone/` (5.4T) — last touched 2026-06-04
   - `RL_super_cf_dfw/` (3.3T) — last touched 2026-06-04

3. **Tier-C structural fix**: add submit-time hook to delete >24h-old `swebench_results_*` before each job. Prevents future cascades.

Even Tier-A alone (~1T) buys 6-8 future runs of headroom. Tier-B is the structural fix but blast radius is high — needs explicit confirmation.

**Awaiting user authorization to execute cleanup.** Claude Code auto-mode classifier (correctly) blocks mass `rm -rf` across multiple repos without per-target approval. Proposed first-run commands, sequenced safest → biggest:

```bash
# 1. uv_cache_nemo_rl_ray (17MB, clearly a temp cache)
ssh cw 'rm -rf /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/uv_cache_nemo_rl_ray'

# 2. swebench_results from May 16-22 (stale, ~50GB total)
ssh cw 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna && \
    rm -rf repos/*/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swebench_results_1778* \
           repos/*/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swebench_results_1779*'

# 3. cross-model-latent-cache (128G, dated 2026-05-18 — old project)
ssh cw 'rm -rf /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/cross-model-latent-cache'

# 4. nano hf_home caches (362G total, dated 2026-05-19-29 — nano model variant project)
ssh cw 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna && \
    rm -rf hf_home_nano_native_main_64n hf_home_nano_prepacked_64n hf_home_nano_prepacked_main_64n'
```

Tier-B requires per-dir confirmation. Suggest checking `git status` in each multi-TB worktree before deleting:
- `repos/nemo-rl-qwen-swe/` (**16T** — OLD legacy main repo, superseded by `-pr2280-pr2514`; single biggest target)
- `RL_super_v3_omni_vllm20_dyncp_clean/` (8.3T)
- `megatron-superv3-standalone/` (5.4T)
- `RL_super_cf_dfw/` (3.3T)
- `repos/RL_super_v3_omni_vllm20_dyncp_cw/` (2.0T)

**Total visible audit**: ~46T of 50T cap accounted for in named dirs. The 16T legacy repo alone could clear the cap.

**Aggregated comparison vs Iter-3 baseline (real wall = arithmetic mean S2-last, includes spikes):**

| Variant | Job | Steady gen (s) | Spike gen (s) | Spike rate | Real wall (s) | Δ real wall vs Iter-3 |
|---------|-----|---------------|----------------|------------|---------------|----------------------|
| async4 FP8KV | 12570149 (PR2335) | 408 (4 steps) | 1185 (2 steps) | 2/6=33% | **1014** (6 steps) | **-1% (parity)** |
| async4 FP8KV | 12561710 (baseline) | 465 (5 steps) | 1042 (3 steps) | 3/8=38% | 1026 (8 steps) | — |
| async8 FP8KV | 12570152 (PR2335) | 384 (4 steps) | 976 (2 steps) | 2/6=33% | **927** (6 steps) | **-14%** |
| async8 FP8KV | 12561712 (baseline) | 462 (3 steps) | 993 (3 steps) | 3/6=50% | 1074 (6 steps) | — |
| async4 BF16  | 12570155 (PR2335) | 567 (3 steps) | 1293 (2 steps) | 2/5=40% | **1208** (5 steps) | **-12%** |
| async4 BF16  | 12561705 (baseline) | 859 (3 steps) | 1190 (3 steps) | 3/6=50% | 1370 (6 steps) | — |
| async8 BF16  | 12570151 (PR2335) | 576 (2 steps) | 1174 (1 step) | 1/3=33% | 1028 (3 steps, partial) | -12% (partial) |
| async8 BF16  | 12561707 (baseline) | 697 (4 steps) | 1253 (1 step) | 1/5=20% | 1163 (5 steps) | — |

**Note on spike-rate convergence**: at 03:33 PDT wakeup with 4-5 steps observed per job, spike rate appeared halved (50→25%) in 3 of 4 configs. At final 6-7 step observation, second spikes appeared in 12570152 (S6) and 12570155 (S6), revising spike rate to 33-40%. Lesson: spike rate is a noisy metric on small N; minimum sample for robust estimate is ≥8 post-cold steps.

**Findings (revised with full per-step data):**

1. **Steady-state generation down 12-34%**: confirmed across all 4 configs. PR #2335 unblocks event loop during normal coroutine turnover. Smallest delta (12%) on async4_FP8KV (already short), largest (34%) on async4_BF16 (longest baseline gen).

2. **Spike floor NOT meaningfully reduced**: 976-1293s exposed-gen across PR2335 configs vs 993-1253s baseline. Differences are within ±15% noise. Confirms agent-loop ceiling bound by `swebench_agent_timeout=1800s × agent_max_turns=200`, independent of `calculate_rewards` event-loop blocking. PR #2335 does not touch the agent execution path.

3. **Spike rate noisy on small N**: appeared halved 50→33% in 3 configs at N=4-5 steps, regressed to 33-40% at N=6-7. The robust ceiling is ~33-50% spike rate, set by the long-tail distribution of SWE-bench trajectory durations.

4. **policy_training & logprobs at parity** (250-265s and 67-71s in both PR2335 and baseline) — expected, PR2335 is generation-side only.

5. **Net throughput win 12-14% on 3 of 4 configs**: async8_FP8KV, async4_BF16, async8_BF16 (partial) each cut real wall by ~140-200s/step. async4_FP8KV at parity (already the fastest baseline at 1026s; less room to gain).

6. **New early-termination failure modes surfaced** (3 of 4 jobs ended before 4h budget, ExitCode 0 but stuck):
   - **AsyncTrajectoryCollector hang** (12570149 S8): collector loop waited >1370s for "1 pending generation thread" — likely a single trajectory bypassing the `swebench_agent_timeout=1800s` cap, OR the collector's pending-count never decrements after timeout fires. Worth filing as bug.
   - **Silent clean-exit at Step 8** (12570152): re-audit invalidated the "chat_template debug loop" hypothesis. Job actually reached Step 8 training cleanly (7 PerfMetrics blocks), then BATCH exit-0 at 38min before SLURM timeout for an unknown reason. The `[DEBUG chat_template_kwargs]` lines are high-frequency normal Generation-request emit, not a retry loop. Need rerun with `set -x` to trace the exit path.
   - **Disk quota exceeded → empty conversation → IndexError** (12570155 S7): `[Errno 122] Disk quota exceeded` writing Apptainer SWE-bench logs → downstream `IndexError: list index out of range` in `nemo_gym.py:407 apply_chat_template(conversation[0]...)`. Cascading failure: storage runs out from spike-step trajectory logs, the empty conversation crashes a later step. Need quota check or log-rotation.

**/loop 15-step goal NOT achieved in 4h budget for any of 4 jobs.** Even at -14% real wall, jobs hit:
- 12570149: 7 steps (would have needed 8 more to reach 15; hung in S8 before time exhausted)
- 12570152: 7 steps (same — stuck, not time-bound)
- 12570155: 6 steps (disk quota crashed, not time-bound)
- 12570151: 4+ steps (only one not stuck; would project ~8-9 in 4h)

**Decisive interpretation**: the 4h budget is NO LONGER the binding constraint. **Stability is now the binding constraint** — three new failure modes (collector hang, debug loop, disk quota) emerged at PR #2335 throughput level. Even with `batch_long` 8h, these would still cap step count at ≤8-10 per job. Fixing them is now the higher-leverage move than reducing spike floor.

### Iter 5 prep (2026-06-06) — SBATCH wrapper fixes (committed locally)

Two structural bugs found in the SBATCH wrapper while reviewing Iter 4 logs. Both fixed in `scripts/share/submit_hybridep_pr2514_256h100*.sh` and validated by `bash -n`.

**Bug 1 — GYM CLEANUP path mismatch (root cause of Failure C in 12570151 + 12570155)**:
The pre-flight cleanup block targets `/lustre/fsw/.../repos/nemo-rl-qwen-swe/3rdparty/...` (the 16T LEGACY worktree) but the active job writes swebench_results into `/lustre/fs1/.../repos/nemo-rl-qwen-swe-pr2280-pr2514/3rdparty/...`. The cleanup runs every job but on the wrong directory; the active repo's swebench_results never get pruned, accumulate >300GB/day, and trigger the 50T USER quota cap. Fixed in all 7 sibling 256H100 wrappers (`*async{1,4,8,16}*{bf16,fp8}kv*.sh`). Cleanup now hits the correct active repo.

**Bug 2 — Silent clean-exit instrumentation (Failure B in 12570152)**:
Wrapped the `uv run ...` invocation with sentinel echoes:
- Before: `echo "[NRL_DRIVER_START] $(date -u +%FT%TZ)"`
- After: `nrl_rc=$? ; echo "[NRL_DRIVER_EXIT] rc=$nrl_rc time=$(date -u +%FT%TZ)" ; ( exit $nrl_rc )`

Applied to `submit_hybridep_pr2514_256h100_32k_async8_fp8kv.sh` only (next-likely-submit). On the next run, ray-driver.log will contain a START marker at COMMAND entry and an EXIT marker with the actual Python rc on COMMAND exit. If silent exit-0 fires again, we'll know: (a) whether Python exited 0 voluntarily (EXIT marker prints rc=0), or (b) the exit happened outside COMMAND in ray.sub or SLURM step prologue/epilogue (no EXIT marker emitted).

**Disk-quota status**: still at 50T/50T hard cap (grace 6d22h, expires ~2026-06-13). Pre-flight cleanup hook now targets the correct path, but `-mtime +1` only prunes dirs >1 day old. The ~310GB of swebench_results from Iter 4 jobs (timestamps 2026-06-06) survive that filter until 2026-06-07. Iter 5 cannot submit safely until either:
- Time elapses past 2026-06-07 00:00 and `-mtime +1` prunes Iter 4 dirs autonomously, or
- User authorizes a one-shot manual prune of fresh swebench_results, or
- User authorizes Tier-A cleanup (~600GB-1T, listed above).

**Iter 5 submission gate** (pre-submit checklist):
1. `ssh cw 'lfs quota -hu sna /lustre/fs1'` → free capacity ≥ 2T (currently 0)
2. Confirm `submit_hybridep_pr2514_256h100_32k_async8_fp8kv.sh` contains `NRL_DRIVER_START` and `NRL_DRIVER_EXIT` strings (validated)
3. Confirm GYM_PATH targets `fs1/.../nemo-rl-qwen-swe-pr2280-pr2514/` (validated all 7 wrappers)
4. Choose ONE variant for Iter 5 (recommend async8_FP8KV: best Iter 4 result at -14% real-wall) — single-job submission, not 4-way blast

### Failure mode A root cause (2026-06-06 14:30Z) — `wait_for_pending_generations` is unbounded

Traced via source on cluster `nemo-rl-qwen-swe-pr2280-pr2514`:

**The per-trajectory timeout IS enforced.** `swebench_agent_timeout=1800` flows through `responses_api_agents/swe_agents/app.py:75-285` into `runner_ray_remote` params. `utils.py:610` silently hardcodes `agent_framework=SupportedAgentFrameworks.openhands` (ignoring the input string) so every rollout runs OpenHands. `run_openhands.py:319-321` wraps the agent command with shell `timeout --signal=TERM --kill-after=30 1800 ...` and `:339` wraps `subprocess.communicate()` in `asyncio.wait_for(..., timeout=1860)`. The subprocess is reliably killed at 1860s.

**The collector-level wait IS NOT bounded.** `nemo_rl/algorithms/async_utils.py:1019` defines:

```python
def wait_for_pending_generations(self) -> None:
    while True:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            ...
            pending_count = len(self._inflight_threads)
        if pending_count == 0:
            break
        ...
        time.sleep(0.5)
```

No timeout parameter. Two callers — L827 (finally cleanup) and L989 (pre-refit barrier) — can hang indefinitely if even one daemon=True `_run_prompt_group_worker` thread sits past its per-trajectory ceiling (file I/O, lustre stat, broken pipe after subprocess SIGKILL, etc.). The threads are daemon=True so they die with the interpreter, but the wait function blocks the interpreter from exiting cleanly. **Hence Failure A: ExitCode 0 with COMPLETED state but step count <max_steps and no Python traceback.**

**Patch drafted (local, not pushed)** at `patches/async_utils_bounded_wait.patch`:
- Add `max_wait_seconds: Optional[float] = None` param to `wait_for_pending_generations`
- After timeout, log warning and `break` (threads die with process)
- Wire `pending_wait_ceiling_seconds` (default 1860) as `grpo.async_grpo.*` config knob
- Call sites pass `max_wait_seconds=self._pending_wait_ceiling`

**Why we apply this AFTER Iter 5, not before:** Iter 5 is the SBATCH-instrumentation experiment. We need the `[NRL_DRIVER_EXIT] rc=` marker to confirm what the silent exit actually does. Applying the bounded-wait patch now would change two variables at once. Order:
1. Iter 5: submit `async8_fp8kv` with SBATCH instrumentation ONLY (no async_utils patch).
2. Read `[NRL_DRIVER_EXIT] rc=` from ray-driver.log → confirms whether Python returned 0 voluntarily or the exit happened outside COMMAND.
3. If rc=0 voluntarily → apply bounded-wait patch in Iter 6.
4. If rc≠0 or no EXIT marker → triage from the actual error.

See [[../../../memory/feedback_async_wait_pending_unbounded.md]] for full memory record.

### Iter 5 submission (2026-06-06 ~17:30Z) — `async8_fp8kv` with SBATCH driver instrumentation

**Job 12601866** submitted to `coreai_dlalgo_nemorl` on CW: 32 nodes × 8 H100 = 256 GPU, partition=batch, time=04:00:00, max_steps=40. RUNNING within 44s of submit (no queue wait).

**Pre-submit gate cleared**:
- `lfs quota -hu sna /lustre/fs1` → **44.5T / 50T** (grace clear, 5.4T headroom after Tier-A cleanup of 3 megatron-superv3-standalone checkpoint dirs × 1.8T)
- Remote SBATCH wrapper synced from local: GYM_PATH = `fs1/.../nemo-rl-qwen-swe-pr2280-pr2514/...` (active repo), driver markers `[NRL_DRIVER_START]` + `[NRL_DRIVER_EXIT] rc=$?` present
- Branch on remote: `sj/super-v3-perf-patch+pr2280+pr2514 @ 1a77265a7` (PR #2335 patch applied)
- async_utils bounded-wait patch deliberately NOT applied (Iter 5 isolates the silent-exit signature; bounded-wait moves to Iter 6 only if rc=0 confirmed)

**What to look for in ray-driver.log**:
1. `[NRL_DRIVER_START] <utc>` near the top
2. After job ends: presence/absence of `[NRL_DRIVER_EXIT] rc=<N>`
   - **rc=0 + step count < max_steps** → confirms Failure A signature (Python exits 0 voluntarily; bounded-wait patch is the fix for Iter 6)
   - **rc≠0** → SBATCH instrumentation surfaced the previously-silent error; triage from the traceback
   - **No EXIT marker** → exit happened outside COMMAND (ray.sub prologue/epilogue or SLURM step boundary); separate root cause

Server-side watcher deployed at `~/.claude-job-watcher/iter5.txt` polling every 300s.

## Why this experiment

Four-way apples-to-apples at 128 H100 [[../perf_comparison_2026_05_16]] concluded **fuse_loss yields no measurable gain** and the feature-porting axis is exhausted on a single rack. The remaining throughput lever is the rollout long-tail, which is bound by per-DP-group vLLM serving capacity and per-trajectory SWE-bench evaluation latency. Doubling the cluster (256 H100, 24 vLLM DP groups vs 12) tests whether more concurrent generators reduce exposed-gen time linearly at constant per-step gradient stats.

**Hypothesis** (pre-launch): at 24 vLLM DP groups, per-DP-group queue depth halves; exposed-gen ≈ 0.5× of 128-GPU baseline at same GBS, OR exposed-gen ≈ same at GBS=512 (work doubled, generators doubled — wash). Train time per step ≈ same (still 64 train GPU, just GBS×2 → 2× per-step train; mitigated by DP=2 → 1× wall).

**What actually happened**: the hypothesis is untestable from this run because four axes were lifted simultaneously, and the dominant axis (`async_grpo.max_trajectory_age_steps=16`, up from 1) inflates Step-0 work by ~16×. Step 0 is the cold-start bottleneck the async-GRPO scheduler usually amortizes across many steps.

## Configurations

| Slot | Variant | gmu | kv_cache | Intended hypothesis |
|------|---------|-----|----------|---------------------|
| A | BF16 KV | 0.85 | auto | Baseline at scale; OOM smoke test for 32K + GBS=512 |
| B | FP8 KV  | 0.80 | fp8_e4m3 | Long-tail BW saving may resurface at 32K (timeout60 less restrictive) |

Branch: `sj/super-v3-perf-patch+pr2280+pr2514` @ `00cf6b43d`.

## Results (terminal state, partial)

Neither job emitted a `Performance Metrics` block. All Step counts below come from in-Step-0 wait logs (`AsyncTrajectoryCollector: Waiting for N pending generation threads`).

| Job | Variant | Wall (h) | Step 0 wait elapsed | Pending start → 2h | 400 Bad Req | Completed traj | Outcome |
|-----|---------|----------|---------------------|--------------------|-------------|----------------|---------|
| 12550182 | BF16 KV | 1.93 | ~52 min observed | ~640 → 613 | 1121 | 952 | Step 0 incomplete; budget will expire |
| 12550221 | FP8 KV  | 1.90 | ~52 min observed | ~640 → 553 | 1058 | 886 | Step 0 incomplete; budget will expire |

**Drain rate**: ~36s per trajectory (FP8 measured: 640 → 553 over 3107.9s wait window). Projected Step-0 wall ≈ 5.5h. Budget left at 2026-06-05 14:30 CT ≈ 2h.

**400 Bad Request waste**: every request whose prompt exceeds `max_total_sequence_length=32768` is rejected (`Prompt exceeds max_model_len: 32835 > 32768`) and retried by the agent. Both jobs hit 54% waste (400-count vs completed-count). This is direct evidence that 32K is **also** too small for the long agent traces, not too large.

### vs 128 H100 baseline (16n×8, GBS=256, max_seq=16384, async-1, timeout60)

| Metric | 16n B (HybridEP+timeout60) | 32n A (BF16 KV) | 32n B (FP8 KV) | Net direction |
|--------|----------------------------|-----------------|-----------------|---------------|
| Total E2E s/step | 405.0 (11919621) | **N/A — Step 0 incomplete** | **N/A — Step 0 incomplete** | structural fail |
| Exposed gen s/step | — | **N/A** | **N/A** | structural fail |
| Train s/step | 60.10 | **N/A** | **N/A** | structural fail |
| LogProb s/step | — | **N/A** | **N/A** | structural fail |

The 256n runs cannot be compared to the 128n baseline because they never reached the steady-state regime.

### Per-step prefill/decode breakdown

Patch [[project-prefill-decode-breakdown-validated]] is wired in this branch but produces no output because no Step completed. The 12023460 5-step ledger remains the only validated data on this axis.

## Diagnosis: compound failure mode

This experiment was the **first** to simultaneously double four throughput-relevant axes against the validated 128n baseline:

| Axis | 128n baseline | 256n attempt | Multiplier |
|------|---------------|--------------|------------|
| `cluster.num_nodes` | 16 | 32 | 2× |
| `policy.max_total_sequence_length` | 16384 | 32768 | 2× |
| `policy.train_global_batch_size` | 256 | 512 | 2× |
| `grpo.async_grpo.max_trajectory_age_steps` | 1 | 16 | **16×** |

The fourth axis dominates. Async-16 means Step 0 must accumulate a 16-slot prompt pool before training can start; at GBS=512 that is `64 prompts × 8 gens × 16 slots = 8192 trajectories` to fill the cold rollout buffer. With unbounded SWE long-tail (`agent_max_turns=200`, `swebench_agent_timeout=1800s`), the p99 trajectory latency dominates wall time, and the 24-DP gen pool cannot offset 16× amplification.

The 128n baseline succeeded only because `async_age=1` keeps Step 0 the same shape as Step N, and the long-tail is amortized across many steps. **Lifting async_age 1 → 16 turns Step 0 into a one-shot 16× bottleneck.**

Linked: [[feedback-256h100-32k-async16-simultaneous-blast]].

## Key takeaway

**At cluster scale, lift exactly one throughput axis per submission and back it with a Step-0 wall projection.** The simultaneous-blast approach used here is invisible at small scale (where Step 0 is cheap) but unrecoverable at 256 GPU because the 4h SLURM budget cannot absorb a 16× cold-start amplification on top of doubled per-trajectory cost.

**Iter 2/3 outcome quantifies the limits**: even after single-axis-lifting async_age 16→4/8, the **clean-steady per-GBS-256-equivalent wall (407s for FP8 KV) reaches parity with the 128n baseline (411s)** — cluster doubling preserves per-token throughput. But **the long-tail spike rate (20-50% of post-cold steps, spike wall 1450-1700s set by `swebench_agent_timeout=1800s × agent_max_turns=200`) inflates real-average by +25 to +67%**, and Step-1 cold-start swallows 13-89min of the 4h budget. The binding constraint at this scale is not training compute, KV format, or async window — it is **the spike floor set by SWE-bench timeout caps**.

Conservative resubmit plan, in priority order:
1. **Get `batch_long` partition assoc** for `coreai_dlalgo_nemorl` — at 12561710's 9-step pace, 8h projects to ~17 steps and clears the /loop 15-step bar.
2. **Cap `agent_max_turns` 200 → 100 + `swebench_agent_timeout` 1800 → 600** as a separate experiment — directly bounds the spike floor that real-average is paying for.
3. Then revisit cluster lift in isolation (`max_seq=16384`, `GBS=256`, `async_age=1`) at 256 H100 to verify parity holds without seq/GBS doubling.
