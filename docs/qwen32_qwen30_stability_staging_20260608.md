# Qwen3-32B / Qwen3-30B-A3B Stability Staging

Date: 2026-06-08

## Why these models are the staging tier

Qwen3-32B and Qwen3-30B-A3B use far fewer nodes than Qwen3-235B-A22B, so they
are the practical place to sweep Full-GRPO stability, batch/token limits, and
vLLM memory reservation settings. Qwen3-235B should be reserved for final
confirmation once the smaller-model envelope is stable.

## Confirmed results

| Model | Runtime / shape | Jobs | Result |
| --- | --- | --- | --- |
| Qwen3-30B-A3B | latest-main / vLLM 0.20, GBS512, worker-batch approx 32, mem80/bt16k | baseline `3198446`, PARD K3 `3198447` | Both completed 20/20. Step 2-20 avg: baseline total `84.34s`, generation `15.77s`, E2E `141.17`; PARD total `81.58s`, generation `10.82s`, E2E `146.21`; generation speedup `1.46x`, E2E `1.04x`, acceptance `69.45%`. |
| Qwen3-30B-A3B | latest-main / vLLM 0.20, GBS2048, mem80/bt16k | baseline `3207492`, PARD K3 `3207978` | Both completed 20/20. Step 2-20 avg: baseline total `248.85s`, generation `54.11s`, E2E `188.61`; PARD total `229.80s`, generation `31.49s`, E2E `204.23`; total/E2E speedup `1.083x`, generation speedup `1.719x`, acceptance `69.10%`. |
| Qwen3-32B | latest-main / vLLM 0.20, mem80/bt16k | baseline `3197980`, PARD K3 `3197981` | Both completed 5/5. Step 1-5 generation `118.01s -> 71.49s` (`1.65x`), total `261.31s -> 217.89s` (`1.20x`). |
| Qwen3-32B | latest-main / vLLM 0.20, mem80/bt16k, GBS512/2048 | `3210222`-`3210225` | GBS512 baseline/PARD both completed 20/20. Step 2-20 PARD speedups: total `1.156x`, generation `1.512x`, E2E `1.156x`. GBS2048 baseline/PARD also completed 20/20; Step 2-20 speedups: total `1.227x`, generation `1.704x`, E2E `1.228x`, generation throughput `1.705x`, acceptance `63.86%`. |
| Qwen3-32B | historical fixed/offline always-on | K1/K3 completed 20-step rows in `docs/eagle3_focus_nemorl_alwayson_metrics.csv` | Confirms the model family can sustain 20-step NeMo-RL loops, but this is not the same latest-main public-PARD K3 path as the current 235B validation. |

## Known unstable envelope

| Model | Condition | Outcome |
| --- | --- | --- |
| Qwen3-30B-A3B | GBS2048 with higher vLLM reservation (`gpu_memory_utilization=0.90`, `max_num_batched_tokens=32768`) | Failed around Step 3 during vLLM `wake_up()` with CuMem OOM. The conservative mem80/bt16k shape fixed this for fixed256 20-step. |
| Qwen3-30B-A3B | long OSL 16K diagnostic | Step 1 matched positive, then Step 2 hit CuMem OOM. Long-OSL stability remains open. |

## Current 235B transfer

The validated Qwen3-235B-A22B Full-GRPO path is non-colocated TP4:

- 32 train nodes plus 4 generation-only nodes.
- 4 GPUs per node.
- generation TP4, draft TP4 for PARD.
- GBS256, fixed decode 256 for the 20-step stability test.
- `max_model_len=8192`, `max_num_seqs=32`, `max_num_batched_tokens=8192`,
  `gpu_memory_utilization=0.70`.

Current 20-step jobs:

| Job | Mode | Status |
| --- | --- | --- |
| `3210070` | Qwen3-235B public PARD K3 | Failed after Step 16 with 600s NCCL watchdog collective timeouts, not OOM. vLLM `0.20.0` and PARD draft config confirmed. Matched Step 2-16 vs baseline retry `3210159`: total `60.53s` vs `88.00s` (`1.454x`), generation `30.93s` vs `56.78s` (`1.836x`), E2E `11.26` vs `7.70 tok/s/GPU` (`1.461x`), generation throughput `196.70` vs `107.01 tok/s/GPU` (`1.838x`); acceptance `57.70%`. |
| `3210069` | Original baseline | Failed before Step 1 due checkpoint conversion race, not OOM. |
| `3210159` | Baseline retry with unique checkpoint dir | Failed after Step 16 with the same 600s NCCL watchdog timeout pattern as `3210070`. Step 2-16 avg total `88.00s`, generation `56.78s`, E2E `7.70 tok/s/GPU`, generation worker `107.01 tok/s/GPU`. This shows the 20-step stability issue is not PARD-specific. |
| `3210513` | Qwen3-235B public PARD K3 timeout retry | Failed after Step 16. Top-level Megatron NCCL timeout was `1800s`, but Megatron Bridge/MCore subgroup watchdogs still showed `Timeout(ms)=600000`. Step 2-16 avg total `60.24s`, generation `30.52s`, E2E `11.30 tok/s/GPU`, generation worker `199.35 tok/s/GPU`, acceptance `58.35%`. |
| `3210580` | Baseline timeout retry | Running at Step 20. It started before the later subgroup-timeout patch, so it validates the top-level timeout path only. Current Step 2-19 avg total `88.25s`, generation `57.52s`, E2E `7.61 tok/s/GPU`, generation worker `104.93 tok/s/GPU`. |

## Staging rule

Use Qwen3-30B-A3B or Qwen3-32B for broad sweeps:

- Raise or lower `max_new_tokens`, `max_num_batched_tokens`, and
  `gpu_memory_utilization`.
- Validate 20-step stability before submitting the same envelope to 235B.
- Treat mem80/bt16k as the conservative default for fixed256 stability.
- Treat long-OSL 16K as unproven until it passes more than Step 1 without CuMem
  OOM.
- Track the Qwen3-32B GBS512/2048 20-step jobs in
  `docs/qwen32_gbs512_2048_fullgrpo20_status_20260608.md`.

Promote a configuration to Qwen3-235B only after the smaller model completes at
least 5 steps for shape sanity, and preferably 20 steps if the question is
stability rather than first-step performance.
