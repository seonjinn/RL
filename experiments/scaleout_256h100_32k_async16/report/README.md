# 256 H100 scale-out — does doubling the cluster halve the rollout?

**STATUS (2026-06-05 14:30 CT)**: jobs **structurally cannot meet `/loop` goal within budget**. Both 12550182 (BF16 KV) and 12550221 (FP8 KV) are still stuck in Step 0 trajectory collection after ~2h wall, with ~2h SLURM budget remaining and an empirical Step-0 drain projection of ~5.5h. No `Performance Metrics` block has been emitted; no `train_step` has fired. The runs will time out at ~Step 0. Lessons captured below; conservative resubmit prepared.

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

**At cluster scale, lift exactly one throughput axis per submission and back it with a Step-0 wall projection.** The simultaneous-blast approach used here is invisible at small scale (where Step 0 is cheap) but unrecoverable at 256 GPU because the 4h SLURM budget cannot absorb a 16× cold-start amplification on top of doubled per-trajectory cost. Conservative resubmit plan: scale cluster only first (`max_seq=16384`, `GBS=256`, `async_age=1`), validate Step 0 ≤ 1.5× baseline, then lift one axis at a time.
