# Goal

Drive Qwen3-235B-A22B-Thinking-2507 SWE async GRPO throughput optimization on 16n8g CW H100 (sm_90). For each candidate optimization (HybridEP, MXFP8 rollout, blockwise FP8 refit), debug to completion, record every problem and resolution in the throughput tracker, and accumulate cross-conversation insights as memory so the same mistake is never repeated. Verify each execution config sustains ≥15 of 20 steps and record Training / LogProb / Generation / E2E throughput vs prior baselines.

## Success Criteria

The literal goal as written ("HybridEP + MXFP8 rollout 적용한 execution config 가 20-step 중 15+ step 잘 작동") is partially infeasible on H100 hardware (see Rule 6). Amended criteria per user redirect 2026-05-16 ("MXFP8가 hardware 적으로 지원안되면 다른것부터 먼저 다 해주세요"):

- [x] HybridEP-only execution config sustains ≥15/20 steps (row 3, job 11811510 reached 22/20)
- [x] HybridEP + skip-prev-logprobs (row 4, job 11819947) reached ≥15/20 — net E2E gain ≈ 0 (gen +13%, logprob 0)
- [x] HybridEP + blockwise FP8 refit (row 4b, PR #2037 port) — BLOCKED on H100: 1536/TP8=192 not div 128 (gmu=0.7→KV underflow at TP=4, refit OOM at gmu=0.90)
- [x] MXFP8 leg closed with HW-permanent block documented (BLOCKED: Blackwell-only E8M0 tensor cores; H100 emulation NaN)
- [x] **HybridEP + fuse_loss (row 5, job 11825251) reached ≥15/20 steps** — step 15 binding lands at 378.85s clean, policy_training 60.76s (-9.2% vs HybridEP-only 66.8s); aggregate 10-clean-sample 3-stage E2E -2.3% vs row 3 baseline
- [x] Throughput tracker row 5 synced with fuse_loss measured medians (Training 60.5 / LogProb 17.9 / Generation 299.2 / E2E 377.6 — all deltas vs baseline filled in summary + stage-by-stage tables)

## Context

- **Project**: Nemo-RL_Qwen3_Roadmap (Qwen3-235B async GRPO SWE)
- **Cluster**: CW (`cw-dfw-cs-001-vscode-02`)
- **Repos on cluster**:
  - main perf-patch: `/lustre/fsw/portfolios/coreai/users/sna/repos/nemo-rl-qwen-swe` @ `sj/super-v3-perf-patch`
  - FP8 refit worktree: `/lustre/fsw/.../repos/nemo-rl-qwen-swe-fp8-refit` @ `sj/super-v3-fp8-refit`
  - fuse_loss worktree (stalled): `/lustre/fsw/.../repos/nemo-rl-qwen-swe-fuseloss-port` (no port commits)
- **Tracker**: `experiments/perf_comparison_2026_05_16/report/throughput_tracker.html`
- **Container**: `nemo-rl:7684dc2-45115915.squashfs` (Row 4) and `nemo-rl:4641794-51006907.squashfs` (May-13 for MXFP8 experiments)
- **Validation commands**:
  - Step count: `ssh cw-dfw-cs-001-vscode-02 'grep -c "Total step time:" /lustre/.../NN-logs/ray-driver.log'`
  - Step timings: `ssh cw-dfw-cs-001-vscode-02 'grep "Total step time:" /lustre/.../NN-logs/ray-driver.log | tail -20'`
  - Job status: `ssh cw-dfw-cs-001-vscode-02 'squeue -h -j NN -o "%T %M"'`

## Rules

1. Read `progress.txt` first to see prior iterations and avoid repeated approaches.
2. Run validation commands before making changes (baseline) — current run state always lives in cluster logs.
3. Job monitoring must use background pollers writing to status files (`/tmp/job_NNN_status.txt`), NEVER foreground `sleep + ssh` (memory `feedback-no-foreground-sleep-for-monitoring`).
4. After each change, run validation commands again and update the throughput tracker.
5. Append iteration summary to `progress.txt` (see format below).
6. **MXFP8 on H100 is permanently BLOCKED at the hardware level.** Do NOT attempt MXFP8 perf validation on this cluster. The only valid path is GB200 (OCI-Hsg / Lyris). User-approved closure 2026-05-16.
7. If all amended success criteria pass, write `GOAL COMPLETE`.
8. If stuck after 2 attempts on the same issue, write `BLOCKED: <reason>` and stop.

## Iteration Log Format

Append to `progress.txt`:

```
## Iteration N — YYYY-MM-DD HH:MM UTC
**Action**: <what changed and why>
**Result**: <pass/fail, error message, step count, throughput numbers>
**Next**: <what the next iteration should try>
```
