# Online PARD-2 Impact Current Read - 2026-06-15

This note separates proven online-drafter behavior from pending 235B proof runs.

## Proven Functional Evidence

| Scope | Evidence | Current read |
| --- | --- | --- |
| Qwen8 official PARD-2 comparison | `3288181` baseline, `3288182` static PARD-2, `3288183` online PARD-2 all completed on OCI-HSG. Metrics are in `docs/qwen8_pard2_official_online_impact_20260613.md`. | Online refit ran for 9 post-step rows and changed acceptance from `1.836` to `2.553`, a `+0.717` point gain over static PARD-2. |
| Qwen30 long-output PARD-2 online comparison | Static baseline `3265386` compared against online jobs `3265387`, `3265388`, and `3274811`. Metrics are in `docs/qwen30ba3b_pard2_online_long_output_win2048_comparison_20260611.md`. | Online refit ran for 1-4 rows across 19 parsed steps. It did not improve acceptance over static PARD-2; generation-worker TPS was roughly flat at `1.0015x`, `0.9932x`, and `0.9919x` vs static. |
| 235B SWE-RL completed 10-step set | `3299487` baseline, `3299489` PARD K5, `3299491` Eagle-3 K3 all completed on OCI-HSG. Metrics are in `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_20260615.md`. | This is the best completed 235B SWE-RL proof so far, but it does not show a speculative throughput win. It also does not settle 235B PARD-2 because the PARD-2 cell in that matrix failed before usable metrics. |

## Performance Read

| Comparison | Result |
| --- | --- |
| Online PARD-2 vs static PARD-2 generation-worker TPS | `0.9696x` |
| Online PARD-2 vs static PARD-2 E2E TPS | `0.8087x` |
| Qwen30 online PARD-2 vs static PARD-2 generation-worker TPS | `1.0015x`, `0.9932x`, `0.9919x` |
| Qwen30 online PARD-2 acceptance delta vs static | `-2.625`, `-1.381`, `-1.085` percentage points |
| Static PARD-2 vs matched baseline generation-worker TPS | `0.6071x` |
| Online PARD-2 vs matched baseline generation-worker TPS | `0.5887x` |
| 235B SWE-RL PARD K5 vs matched baseline generation-worker TPS | `0.2679x` |
| 235B SWE-RL Eagle-3 K3 vs matched baseline generation-worker TPS | `0.4857x` |

Interpretation: the completed Qwen8 run proves that online PARD-2 refit is wired into NeMo-RL and can improve acceptance, but it does not show a throughput win. The Qwen30 long-output comparison adds a larger-model online-refit check and shows roughly flat throughput with lower acceptance than static PARD-2. The completed 235B SWE-RL run proves that baseline/PARD/Eagle-3 can finish a 10-step SWE-RL job, but the completed speculative variants were slower than baseline. The current blocker for a stronger claim is not online-refit mechanics; it is producing 235B MathRL and 235B SWE-RL PARD-2 proof runs with useful throughput.

## Pending 235B Gates

| Gate | Jobs | Why it matters |
| --- | --- | --- |
| Lyris SWE-RL Raymatch baseline proof | `2129203` | r30 retry after patching Ray defaults and moving heavy build/cache paths to node-local `/tmp`; running with stdout/runtime logs present, but no parsed step metrics yet. |
| Lyris SWE-RL PARD/PARD-2 dependent proof | `2129271`, `2129272` | r30 PARD/PARD-2 jobs waiting on `afterok:2129203`; re-submitted after disabling shell trace in `ray.sub`. |
| OCI-HSG SWE-RL PARD-2 proof | `3308774` | Current 235B SWE-RL PARD-2 attempt under `nemotron_n3_post`. |
| OCI-HSG MathRL latest-main proof | `3315380`, `3315381`, `3315382` | Current guarded 235B MathRL baseline/PARD K3/PARD K5 attempts under `nemotron_n3_post`. |

As of `2026-06-15 08:29 PDT`, Lyris r28 baseline `2126895` is superseded negative evidence: it failed with `sacct_exit=1:0` after TransformerEngine built successfully, because the Ray cluster was started with `Ray 2.49.2` and `Python 3.12.13` while the driver process used `Ray 2.54.0` and `Python 3.13.13`. The r29 baseline `2128989` fixed the Ray/Python mismatch and reached 64/64 connected actors, but stalled at TransformerEngine source build in the user persistent cache while the Lyris user inode quota was over soft quota and close to hard limit; it was cancelled and superseded. The active Lyris proof chain is now r30: baseline `2129203` is `RUNNING/None` after starting at `2026-06-15T07:57:46`, uses `PERSISTENT_CACHE=/tmp/qwen235b_swerl_specdec_r30`, and the TE build is still active after advancing to fresh `ninja`/`c++`/`cc1plus` work; PARD `2129271` and PARD-2 `2129272` wait on `afterok:2129203` and have stored batch scripts verified without `bash -x`. OCI-HSG SWE-RL PARD-2 `3308774` and MathRL jobs `3315380`, `3315381`, and `3315382` remain `PENDING/Priority` under `nemotron_n3_post`; their scheduler start estimates are volatile, so the active gate snapshot is the authoritative source for current starts. A 235B MathRL or 235B SWE-RL PARD-2 success claim should still wait for completed steps and parsed metrics from one of these gates.

Current monitoring artifacts:

- `docs/nemorl_235b_active_gates_latest_20260615.md`
- `docs/nemorl_235b_gate_runtime_report_latest_20260615.md`
- `docs/nemorl_235b_active_gates_history_20260615.csv`
- `docs/nemorl_235b_active_gates_changes_latest_20260615.md`
- `scripts/inspect_nemorl_235b_gate_first_logs_20260615.sh`
- `scripts/monitor_nemorl_235b_gates_until_runtime_20260615.sh`
- `scripts/fetch_and_parse_nemorl_235b_ready_gate_metrics.py`

Current 10-step example runbook:

- `docs/nemorl_10_step_examples_current_20260615.md`
- `docs/nemorl_10_step_repro_runbook_20260615.md`
- `docs/pard_online_goal_completion_audit_20260615.md`
