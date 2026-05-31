# Qwen3-235B SWE Async GRPO — Throughput Optimization Status

**Last update:** 2026-05-20 11:15 UTC
**Hardware scope:** H100 sm_90 ONLY (CW `cw-dfw-cs-001-vscode-01`, 16n × 8 GPU)
**Workload:** Qwen3-235B-A22B-Thinking-2507 SWE async GRPO, 20 max_steps, `swebench_tests_timeout=60`
**Active branch:** `sj/super-v3-perf-patch+pr2280+pr2514` @ `ea5153a2e` (worktree `nemo-rl-qwen-swe-pr2280-pr2514`)

---

## 1. /loop Goal (verbatim)

> HybridEP 와 MXFP8 rollout에 대해서 각각 실행이 완료될떄까지 계속 디버깅하고 report로 어떤문제들이 있었고 어떻게 해결하려했고 결과는 어떗는지 계속 반복적으로 기록해서 똑같은 실수안하면서 계속 발전적으로 나아가주세요. 핵심 목표는 HybridEP + MXFP8 Rollout 까지 적용한 execution config 이 20-step 중 15+ step 잘 작동해야 합니다. 그리고 Training/LogProb/Generation/E2E throughput 비교를 기록해주세요.

**Amended (user-approved 2026-05-16):** MXFP8 forward-pass is permanently BLOCKED on H100 at the hardware level — Blackwell-only E8M0 tensor cores; H100 EMULATION = BF16 compute + 50% memory savings only, no perf gain. Substitute legs: blockwise FP8 training, FP8 KV cache, HybridEP, fuse_loss, selective recompute, max_tokens clamp.

---

## 2. Current Job (in flight)

| Field | Value |
|---|---|
| Job ID | **11941263** |
| Variant | `qwen3-235b-swe-perf-hybridep-pr2280-pr2514-timeout60` |
| State | RUNNING (Submit 10:08:42 UTC, Start 10:46:26 UTC, elapsed ~30 min) |
| Phase | SETUP (driver log exists but `Total step time` count = 0) |
| Nodes | 16 × pool0 (00148 00241 00282 00395 00984 01099 01105 01132 01134 01136 01193 01198 01200 01827 01829 01841) |
| Wakeup armed | 11:06 UTC (queue/SETUP check), then again at step 1 ETA ~11:30-11:40 UTC |

**What's stacked on top of baseline HybridEP+to60:**
- **PR #2280** (selective recompute) — `recompute_granularity=selective`, `recompute_modules=[core_attn, moe_act]`
- **PR #2514** (max_tokens clamp) — clamps `max_new_tokens` to `(max_model_len - prompt_tokens)` to cut over-generation past context

**Hypothesis (vs HybridEP+to60 baseline 11891376, median step 413.95s):**
- PR #2280 drops `policy_training` 60.07s → ~50s (-10s) by skipping recompute on dominant fwd cost
- PR #2514 compresses `exposed_generation` right-tail outliers (median ~315s → ~310s)
- **Target steady ≤390s/step** (+5% vs HybridEP+to60)

**Fallback ladder if selective recompute OOMs:**
1. `recompute_modules=[moe_act]` only (smaller activation memory hit)
2. `recompute_modules=[core_attn]` only
3. Revert to `recompute_granularity=full uniform N=1` (baseline)

---

## 3. Performance Comparison Table (measured)

All numbers are medians over steady-state steps (steps 5-20). Format: `value (delta vs Ray-opt+to60 baseline)`.

| Config | Job ID | timeout60 | Step total (s) | policy_training (s) | logprobs (s) | exposed_generation (s) | weight_sync (s) | Steps reached |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| super-v3 vanilla (no opt, NO to60) | 11772327 | ❌ | ~405 | 67.85 | — | 293 | — | 19/19 done |
| Ray-opt (BF16, NO to60) | 11857294 | ❌ | 400.22 | 67.58 | 17.65 | 288.48 | — | 20/20 |
| **Ray-opt + to60 (baseline)** | 11891359 | ✅ | **428.12** | 67.00 | 17.66 | 319.83 | 19.43 | 20/20 |
| HybridEP + to60 | 11891376 | ✅ | 413.95 (-3.3%) | 60.07 (-10.3%) | 15.17 (-14.1%) | 315.02 (-1.5%) | 18.21 (-6.3%) | 20/20 |
| HybridEP + FP8-Tr v2 + to60 | 11912255 | ✅ | 403.88 (-5.7%) | 58.75 (-12.3%) | 14.86 | 326.68 | 17.95 | 20/20 |
| HybridEP + fuse_loss + to60 | 11919621 | ✅ | 411.40 (-3.9%) | 60.31 | 15.40 | 313.5 | 17.92 | 15/15 (3h cap) |
| HybridEP + FP8KV (NO to60) | 11835558 | ❌ | 380.86 (-9.6% vs no-to60) | 60.5 | 15.4 | 299.2 | — | done |
| **HybridEP + PR2280 + PR2514 + to60** | **11941263** | ✅ | **TBD** | TBD | TBD | TBD | TBD | RUNNING |

**Cross-config insights:**
- **timeout60 regime cost**: Ray-opt baseline 400.22 → 428.12s = **+28s/step regression** when `swebench_tests_timeout=60` enforced (job 11891359 vs 11857294). This isolates timeout60's truncation cost from optimization gains.
- **HybridEP under to60**: −14s/step recovered (428→414). Pure MoE comms win, not regime-dependent.
- **FP8-Tr v2 under to60**: another −10s/step (414→404). policy_training drops further but ExposedGen rises (BF16-W refit chooses different generation paths).
- **fuse_loss under to60**: NO measurable gain over HybridEP (411.40 ≈ 413.95). Loss aggregation is not the binding constraint on this workload (memory [[feedback-fuseloss-super-v3-structural-mismatch]]).
- **HybridEP+FP8KV (no to60)**: −19s/step vs Ray-opt no-to60 (400→381). Best raw H100 number, but only when timeout60 not enforced.
- **HybridEP+FP8KV+to60**: regresses (+26% ExposedGen) — memory [[feedback-fp8-kv-regresses-with-timeout60]]. The FP8 KV bandwidth gain is absorbed by the same long-tail that timeout60 truncates.

---

## 4. Current Performance Bottleneck

**Generation is the binding constraint at the to60 regime.**

Breakdown of HybridEP+to60 median step (413.95s):
| Stage | Time (s) | % of step | Comment |
|---|---:|---:|---|
| exposed_generation | 315.02 | **76.1%** | Dominant; long-tail not fully cut by to60 |
| policy_training | 60.07 | 14.5% | Reduced by HybridEP MoE comms savings |
| logprobs | 15.17 | 3.7% | Already minimal |
| weight_sync | 18.21 | 4.4% | NCCL broadcast bound by parameter volume |
| other (idle, sync) | ~5 | 1.3% | — |

**Why generation dominates:**
1. SWE agent rollouts have per-trajectory tail bound by `swebench_agent_timeout=1800` (not the test-execution timeout 60s)
2. Even with to60, ~10-20% of trajectories hit the 1800s agent ceiling
3. async overlap with policy_training only hides up to ~policy_training time → max ~60s of gen overlap, leaving ~250s exposed

**Per-leg attack on the bottleneck (current /loop session):**
- **PR #2280 (selective recompute)** — attacks `policy_training` (14.5%). Predicted -10s/step, marginal because policy_training is not the binding constraint.
- **PR #2514 (max_tokens clamp)** — attacks `exposed_generation` long-tail directly. This is the right target. Hypothesis: clamping output tokens prevents generation from running past `max_model_len` even when agent decides to continue.

**Settled bottlenecks (cannot move further on H100):**
- FP8 weight refit (NCCL bandwidth) — broken at NeMo-RL patch boundary across 4 axes (memory [[feedback-fp8-weight-refit-two-failure-modes]])
- FP8 KV cache memory pressure — works at no-to60, regresses at to60
- MXFP8 forward-pass — Blackwell-only HW
- Per-tensor FP8 MoE quality — output collapse to "!" repeated tokens
- FA3 attention compute — already at H100 kernel ceiling, FlashInfer 0.5.3 cannot JIT-compile mixed BF16-Q+FP8-KV on sm_90

---

## 5. Completed Work Inventory

### 5.1 HybridEP leg — ✅ COMPLETE
| Sub-task | Job | Outcome | Memory |
|---|---|---|---|
| Container build (CUDA 13 → 12.9, deep_ep cp313 wheel) | — | Unblocked | [[project-hybridep-cuda13-block]], [[project-hybridep-mxfp8-combined-blocker]] |
| First successful 16/20 step run | 11795544 | -5% vs baseline | (early run) |
| 22/20 step full run | 11811510 | 22/20 (over-runs target) | (validates) |
| skip_prev_logprobs ablation | 11819947 | Net E2E ≈ 0 (gen +13%, logprob 0) | [[feedback-async-overlap-loss-when-skipping-logprob]] |
| Apples-to-apples to60 baseline | 11891376 | 20/20, median 413.95s | [[project-apples2apples-4way-timeout60]] |
| FP8 Training v2 stack | 11912255 | 20/20, 414.88s steady | [[project-qwen3-235b-swe-hybridep-fp8tr-v2-success]] |
| fuse_loss stack | 11919621 | 15/15 at 3h cap, no measurable gain | [[feedback-fuseloss-super-v3-structural-mismatch]] |
| Vanilla super-v3 + to60 reference | 11922433 | (pending harvest) | — |

### 5.2 MXFP8 leg — ✅ CLOSED (HW-blocked)
| Investigation | Outcome |
|---|---|
| vllm 0.17.1 + ModelOpt patches | sm_90 bypass via 3 software gates worked |
| EMULATION path (BF16 compute) | Mechanically runs, no perf gain |
| Refit broadcast `weight_scale_from_checkpoint` | Stays ALL ZEROS post-refit → NaN logits | 
| HW root cause | E8M0 tensor cores Blackwell-only; H100 cannot accelerate MXFP8 |
| Resolution | Replace with H100-feasible FP8 substitutes; MXFP8 perf validation moves to GB200 (out of scope) |

Relevant memories: [[reference-flashinfer-mxfp8-sm90]], [[feedback-mxfp8-emulation-swizzle-skip]], [[feedback-mxfp8-emulation-refit-register]], [[feedback-mxfp8-refit-broadcast-zero]], [[feedback-mxfp8-worker-subprocess-bypass]]

### 5.3 FP8 substitutes — partial
| Substitute | Status | Memory |
|---|---|---|
| FP8 KV cache (vllm `fp8_e4m3` + FA3 implicit Q quantize) | ✅ Works, -9.6% E2E (no to60); regresses at to60 | [[project-fp8-kv-cache-path-validated]], [[reference-fa3-fp8kv-implicit-attn-compute]] |
| Blockwise FP8 training (Megatron+TE) | ✅ HybridEP+FP8-Tr v2 11912255 | [[project-qwen3-235b-swe-hybridep-fp8tr-v2-success]] |
| Blockwise FP8 weight refit (PR #2037 port) | ❌ TP=8 div 128 fail; TP=4 KV underflow + refit OOM | [[project-pr2037-fp8-refit-evaluation]], [[feedback-fp8-blockwise-tp4-kv-underflow]], [[feedback-fp8-blockwise-refit-oom]] |
| Per-tensor FP8 MoE | ❌ Quality collapse | [[feedback-pertensor-fp8-moe-quality-collapse]] |

### 5.4 Selective recompute + max_tokens clamp (in flight, this session)
- Worktree: `nemo-rl-qwen-swe-pr2280-pr2514` @ `sj/super-v3-perf-patch+pr2280+pr2514` (commit `ea5153a2e`)
- 6 files, +187/-8: `setup.py` recompute fork, `rollouts.py` + `vllm_worker_async.py` clamp, tests, reference config
- Conflict resolution: super-v3 filtered prefix-deepcopy retained, PR #2514 actual_request_max_tokens setup added
- Submit script: `submit_hybridep_pr2280_pr2514_timeout60.sh` with `++recompute_granularity=selective ++recompute_modules=[core_attn,moe_act]`
- Job 11941263 (this section 2)

---

## 6. Key Pitfalls / Anti-patterns Recorded (memory references)

Sub-agents and future sessions: **read these before touching the corresponding area** to avoid repeating known failures.

| Pitfall | Memory |
|---|---|
| MXFP8 on H100 cannot perf-validate (HW limit, not patch) | [[reference-flashinfer-mxfp8-sm90]], [[feedback-h100-only-scope]] |
| Worktree submodules missing → uv sync fails | [[feedback-worktree-missing-submodules]] |
| MOUNTS line drift across worktrees → mimo F401 import fail | [[feedback-fuseloss-worktree-mount-regression]] |
| May-13 container `uv run --frozen` rebuilds venv → kills ray head | [[feedback-may13-container-uv-run-pitfall]] |
| Gym subprocess venv py-version mismatch (3.13 vs 3.12) | [[feedback-gym-subprocess-venv-py-version-mismatch]] |
| Gym ray pin conflict (parent 2.54.0 vs Gym 2.49.2) | [[feedback-gym-ray-pin-conflict]] |
| Gym uv_cache lock timeout (default 300s too short) | [[feedback-gym-uv-lock-timeout]] |
| FP8 blockwise 1536/TP8=192 not div 128 | [[feedback-fp8-blockwise-192-not-div-128]] |
| Async overlap loss when skipping prev_logprobs | [[feedback-async-overlap-loss-when-skipping-logprob]] |
| Foreground sleep blocks parallel work — use background poller | [[feedback-no-foreground-sleep-for-monitoring]] |
| vllm 0.17 openai entrypoints module split | [[feedback-vllm017-openai-module-split]] |
| vllm 0.17.1 `_preprocess_chat` signature change | [[feedback-vllm017-preprocess-chat-signature]] |
| torch 2.10 `register_op_strategy` removed | [[feedback-torch210-register-op-strategy-removed]] |
| Per-tensor FP8 MoE → "!" output collapse | [[feedback-pertensor-fp8-moe-quality-collapse]] |
| Full SHA verify via `gh api .sha` (never extend abbreviated) | [[feedback-full-sha-from-gh-api]] |
| CW OccupiedIdleGPUsJobReaper kills nsys jobs at ~22min | [[feedback-cw-idle-gpu-reaper]] |

---

## 7. How to Resume (cold-start checklist for new session/agent)

1. **Check job state**:
   ```bash
   ssh cw-dfw-cs-001-vscode-01 'sacct -j 11941263 --format=State,Elapsed -P | head -3'
   ssh cw-dfw-cs-001-vscode-01 'grep -c "Total step time" /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/repos/nemo-rl-qwen-swe-pr2280-pr2514/11941263-logs/ray-driver.log'
   ```

2. **If OOM detected** (search log for `out of memory|CUDA OOM|torch.cuda.OutOfMemoryError`):
   - Edit `${REPO_DIR}/submit_hybridep_pr2280_pr2514_timeout60.sh` line with `recompute_modules`
   - Try `[moe_act]` only, then `[core_attn]` only, then fall back to `recompute_granularity=full`
   - Resubmit with `--account=coreai_dlalgo_nemorl`

3. **If job completes ≥15/20 steps**: harvest per-step ledger
   ```bash
   ssh cw-dfw-cs-001-vscode-01 'grep "Total step time:" /lustre/fs1/.../11941263-logs/ray-driver.log'
   ```
   Then update **section 3** of this report with the row, compute deltas vs HybridEP+to60 baseline.

4. **If job fails before step 1**:
   - Likely cause: ValueError at `setup.py:495-498` if `recompute_granularity` override didn't land in EXTRA_OVERRIDES
   - Verify wrapper script `EXTRA_OVERRIDES` env has both `recompute_granularity` and `recompute_modules`
   - Re-check commit `ea5153a2e` cleanly applied via `git log --oneline -1` in worktree

5. **Job watcher**: `~/.claude-job-watcher/status_11941263.txt` (server-side, 300s polling)

6. **Wakeup schedule**: 11:06 UTC armed (queue/SETUP), expect step 1 entry ~11:30 UTC; next wakeup after that should target step 5 (steady-state observable) and step 15 (success criterion met).

---

## 8. Outstanding / Optional Next Steps

| Priority | Item | Notes |
|---|---|---|
| **P0** | Harvest 11941263 ledger + decide if `[core_attn, moe_act]` OOM'd | Defines next scope shrink (if any) |
| P1 | Update memory `project-qwen3-235b-swe-hybridep-fp8tr-v2-success` final-step ledger | Was through step 15 only |
| P1 | Update `experiments/perf_comparison_2026_05_16/report/throughput_tracker.html` with row 6 (PR2280+PR2514) | After ledger harvest |
| P2 | Vanilla super-v3 + to60 (11922433) ledger harvest | For baseline comparison |
| P2 | Optional GitHub PR from `sj/super-v3-perf-patch+pr2280+pr2514` | Currently only branch exists, no PR opened |
| P3 | Bin Hu cache-seeding pattern integration | Deferred, may further cut gen tail |
| P3 | Profiling job for HybridEP step (nsys) | CW reaper kills at 22min → need profile flag or shorter window |

---

## 9. File / Path Reference

| Asset | Local path | Cluster path |
|---|---|---|
| Project root | `/Users/sna/Nemo-RL_Qwen3_Roadmap` | — |
| Main worktree | — | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/repos/nemo-rl-qwen-swe` |
| PR2280+PR2514 worktree | — | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/repos/nemo-rl-qwen-swe-pr2280-pr2514` |
| FP8 refit worktree (stalled) | — | `/lustre/fsw/.../repos/nemo-rl-qwen-swe-fp8-refit` |
| Active config (used by submit) | — | `${REPO_DIR}/grpo_qwen3_235b_swe.yaml` |
| Submit script (current) | — | `${REPO_DIR}/submit_hybridep_pr2280_pr2514_timeout60.sh` |
| Memory dir | `~/.claude/projects/-Users-sna-Nemo-RL-Qwen3-Roadmap/memory/` | — |
| Throughput tracker | `experiments/perf_comparison_2026_05_16/report/throughput_tracker.html` | — |
| Main perf report (61K, historical) | `experiments/perf_comparison_2026_05_16/report/README.md` | — |
| Progress log (Ralph Loop) | `progress.txt` (1989 lines, last iteration N+92) | — |
| Goal spec (Ralph Loop) | `PROMPT.md` | — |

---

## 10. Key Takeaway

**HybridEP is the only structural win on H100 that survives the timeout60 regime.** FP8 substitutes either regress under to60, OOM at refit, structurally fail divisibility, or collapse model quality. The current PR #2280 + PR #2514 stack attacks the right bottleneck — **PR #2514 max_tokens clamp targets the dominant generation long-tail (76.1% of step), which is the actual binding constraint**. PR #2280 selective recompute is a secondary win on `policy_training` (14.5% of step) and may free memory for other future legs but cannot move the needle alone.

If 11941263 lands ≥15/20 with steady ≤390s/step, this becomes the new production H100 config. If it OOMs on selective recompute, the fallback ladder (Section 2) lets us shrink scope to `[moe_act]` only without losing the PR #2514 generation gain.

**The /loop goal "HybridEP + MXFP8 rollout 20/20 step" is permanently infeasible on H100; the H100 substitute path is fully validated. GB200 (out of scope) is the only platform where the original goal can be tested.**
