# NeMo-RL SpecDec Matrix Coverage

Date: 2026-07-02

This is the current evidence map for the requested NeMo-RL performance-recipe
matrix. A cell is complete only when a matched baseline and SpecDec run use the
same model, mode, max OSL, sampling configuration, CUDA Graph setting, resource
shape, and steady-state step span.

All performance values below use CUDA Graphs (`enforce_eager=false`),
temperature 1.0, top-p 1.0, and exclude cold Step 1 when 20 steps completed.

## Default Performance-Recipe OSL

| Model | Mode | Baseline | PARD K=1/7/9/16 | Eagle K>3 | Best completed E2E speedup | Best completed generation speedup | Status |
|---|---|---|---|---|---:|---:|---|
| Qwen3-30B-A3B | sync | complete, Steps 2-20 | complete | K=5/7/9 complete | 0.97x, Eagle K5 | 0.91x, Eagle K5 | Complete; all tested SpecDec rows regress |
| Qwen3-30B-A3B | async-1off | complete, Steps 2-20 | complete, including replicate cohort | K=5/7/9 complete | 1.07x, Eagle K9 | 1.07x, Eagle K9 | Complete; PARD has material run-to-run variance |
| Qwen3-32B | sync | complete, Steps 2-20 | complete | K=5/7/9 complete | 1.21x, Eagle K5 | 1.33x, Eagle K5 | Complete |
| Qwen3-32B | async-1off | complete, Steps 2-20 | complete | K=7/9 complete | 1.09x, PARD K1 | 1.09x, PARD K1 | Complete for requested PARD set and Eagle K>3 |
| Qwen3-235B-A22B | sync | queued as `2264443` | K1 TP8 smoke `2261383` failed before engine initialization because the launcher emitted invalid `method=pard`; draft-TP1 K1/7/9/16 held | K7/K9 complete; preliminary matched Steps 2-4 | waiting final baseline | waiting final baseline | Incomplete |
| Qwen3-235B-A22B | async-1off | baselines `2261942` and `2264402` queued | draft-TP1 K1/7/9/16 held | K7 `2264403` queued | waiting baseline | waiting baseline | Incomplete |

Qwen3-235B sync Eagle runs completed 20/20, but the matched baseline currently
has only clean Steps 2-4. The preliminary matched comparison is:

| Method | Matched steps | E2E throughput speedup | Generation throughput speedup | E2E step-time speedup | Generation-time speedup | Acceptance | Mean accept length | W&B |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Eagle K7 | 2-4 | 1.195x | 1.286x | 1.162x | 1.284x | 18.20% | 2.27 | [run](https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/bra22f2d) |
| Eagle K9 | 2-4 | 1.147x | 1.194x | 1.169x | 1.198x | 14.19% | 2.28 | [run](https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/y99qvnc9) |

These speedups are preliminary because the matched baseline failed after Step 4
with the TP8 logits all-gather timeout. They are not substituted for a matched
Steps 2-20 result. The completed Eagle absolute Step 2-20 metrics are:

| Method | E2E step time | Generation time | E2E throughput/GPU | Generation throughput/GPU | Acceptance | Mean accept length | W&B |
|---|---:|---:|---:|---:|---:|---:|---|
| Eagle K7 | 230.74 s | 114.60 s | 108.72 tok/s | 196.38 tok/s | 17.23% | 2.21 | [run](https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/bra22f2d) |
| Eagle K9 | 262.32 s | 131.11 s | 95.81 tok/s | 174.21 tok/s | 13.41% | 2.21 | [run](https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/y99qvnc9) |

No final Qwen3-235B Steps 2-20 speedup is publishable until a matched baseline
completes.

## Long-Context OSL 32768

| Model | Mode | Baseline | PARD K=1/7/9/16 | Eagle K>3 | Current conclusion |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | sync | fails at Step 2 wake-up | held after baseline failure | K5/K7 fail at Step 2; K9 cancelled | Colocated policy memory prevents vLLM weight/KV remap; allocator cleanup level 2 did not fix it |
| Qwen3-30B-A3B | async-1off | missing | missing | missing | Needs a matched non-colocated async performance-recipe cohort |
| Qwen3-32B | sync | complete, Steps 2-20 | partial K1/7/9/16 | partial K5/7/9 | All partial SpecDec rows regress; full runs timed out after 5-13 steady-state steps |
| Qwen3-32B | async-1off | missing | missing | missing | Needs a matched async performance-recipe cohort |
| Qwen3-235B-A22B | sync | missing | missing | missing | Default-OSL stability and baseline must complete first |
| Qwen3-235B-A22B | async-1off | missing | missing | missing | Default-OSL exact async baseline stability must complete first |

The Qwen3-32B sync 32K baseline completed with 168.03 E2E tok/s/GPU and
187.45 generation tok/s/GPU. Partial SpecDec E2E speedups range from 0.39x to
0.72x; none currently improves the matched baseline.

## Active Scheduling

Runnable Lyris jobs:

| Job | Model/mode | Method | Expected start (PDT) |
|---:|---|---|---|
| 2261942 | Qwen3-235B async-1off | matched no-AR-RMS baseline | 2026-07-03 02:17 |
| 2264402 | Qwen3-235B async-1off | exact baseline | 2026-07-03 02:37 |
| 2264403 | Qwen3-235B async-1off | exact Eagle K7 | 2026-07-03 04:06 |
| 2264443 | Qwen3-235B sync | exact no-AR-RMS baseline | 2026-07-03 05:07 |

All remaining runnable jobs use 32 nodes, four GPUs per node, `--segment=16`, no `--gres`, Lustre
artifacts, the nightly container, CUDA Graphs, and the
`sna-nemorl-specdec-lyris` W&B project.

Failed PARD smoke `2261383` produced no performance sample. Its W&B run is
[`vl2euq7r`](https://wandb.ai/nvidia/sna-nemorl-specdec-lyris/runs/vl2euq7r).
The required one-variable correction is `method=pard` to
`method=draft_model`; this is the stock-vLLM PARD contract already enforced by
the canonical NeMo-RL SpecDec launcher.

## Required Next Cohorts

1. Collect the four runnable Qwen3-235B jobs and publish matched Step 2-20 rows
   where both sides complete.
2. Run Qwen3-30B-A3B and Qwen3-32B async-1off 32K matched baseline, PARD
   K1/7/9/16, and Eagle K5/7/9 using their existing non-colocated async
   performance recipes.
3. Validate a Qwen3-30B-A3B sync 32K non-colocated resource overlay. This is a
   new matched setup because the exact colocated recipe cannot wake vLLM after
   policy training at 32K.
4. Resolve Qwen3-235B draft TP1 through a V1-native proposal broadcast design,
   or run a separately matched target/draft TP4 control. Deleting the vLLM TP
   mismatch guard is not safe.
5. Add engine-interval SpecDec acceptance metrics to async-1off W&B logging.
   The root cause and metrics-only design are documented in
   `nemorl_async_specdec_metrics_root_cause_20260702.md`.

## Sources

- `lyris_qwen30_sync_pard_strict_matched_metrics_20260702.csv`
- `lyris_qwen30_async1off_strict_matched_live_metrics_20260702.csv`
- `lyris_qwen32_sync_eagle3_matched_live_metrics_20260702.csv`
- `lyris_qwen32_sync_pard_tp2_noarrms_matched_live_metrics_20260702.csv`
- `lyris_qwen32_async1off_eagle3_matched_live_metrics_20260702.csv`
- `lyris_qwen235b_sync_eagle3_absolute_metrics_20260702.csv`
- `pretyche_qwen32_sync_osl32k_matched_live_metrics_20260702.csv`
- `latest_lyris_qwen30_sync_osl32k_matched_step20_20260702_jobs.csv`
