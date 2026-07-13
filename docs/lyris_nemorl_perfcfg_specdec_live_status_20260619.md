# Lyris NeMo-RL Performance Config SpecDec Status

Updated: 2026-06-19 13:55 PDT

## Key Findings

- Qwen3-30B-A3B and Qwen3-32B performance-config OSL=4096 Step20 runs have usable metrics; Eagle-3 is the only consistently positive SpecDec method in this NeMo-RL matrix.
- Qwen3-235B-A22B is still not stable for recipe-default OSL=8192 Step20: no matched baseline completed a usable step. OSL=1024 step1 canaries now complete, but the step3 fallback times out in step2 with NCCL ALLGATHER_BASE watchdog / EngineDeadError.
- Current `squeue` check found no active matching Lyris Nemo-RL jobs; latest states are terminal in `sacct`.

## Methodology

- Workload: NeMo-RL GRPO performance recipes from latest-main merged worktree with SpecDec added.
- Sampling: temperature=1.0, top_p=1.0.
- Hardware: Lyris GB200, 4 GPUs/node, segment-aware submission, no `--gres=gpu:4`.
- OSL: Qwen30/Qwen32 recipe-default Max OSL=4096; Qwen235B recipe-default Max OSL=8192; Qwen235B debug canaries use Max OSL=1024.
- Speedups are matched only within the same model, mode, and Max OSL baseline.

## Qwen30/Qwen32 Recipe OSL=4096 Step20

| Model | Mode | Method | Job | Steps | State | E2E step | E2E throughput | Gen time | Gen tok/s/GPU | Gen speed | Accept | Mean len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | sync | baseline | 2152132 | 19/20 last 20 | COMPLETED | 421.7s | 1.00x | 278.5s | 1500.1 | 1.00x | n/a | n/a |
| Qwen3-30B-A3B | sync | Eagle-3 | 2152193 | 19/20 last 20 | COMPLETED | 299.4s | 1.39x | 127.8s | 3226.3 | 2.15x | 63.9% | 2.87 |
| Qwen3-30B-A3B | sync | Suffix | 2152134 | 19/20 last 20 | COMPLETED | 415.2s | 1.01x | 240.0s | 1731.1 | 1.15x | 11.3% | 1.84 |
| Qwen3-30B-A3B | sync | PARD K=5 | 2152135 | 19/20 last 20 | COMPLETED | 427.5s | 0.98x | 280.9s | 1478.0 | 0.99x | 36.6% | 2.29 |
| Qwen3-30B-A3B | sync | baseline fuse_loss=false | 2152320 | 19/20 last 20 | COMPLETED | 449.7s | 0.94x | 299.3s | 1395.1 | 0.93x | n/a | n/a |
| Qwen3-30B-A3B | async-1off | baseline | 2152136 | 19/20 last 20 | COMPLETED | 425.6s | 1.00x | n/a | 1998.4 | 1.00x | n/a | n/a |
| Qwen3-30B-A3B | async-1off | Eagle-3 | 2152194 | 19/20 last 20 | COMPLETED | 338.2s | 1.25x | n/a | 2510.4 | 1.26x | n/a | n/a |
| Qwen3-30B-A3B | async-1off | Suffix | 2152138 | 19/20 last 20 | COMPLETED | 477.2s | 0.90x | n/a | 1794.7 | 0.90x | n/a | n/a |
| Qwen3-30B-A3B | async-1off | PARD K=5 | 2152139 | 19/20 last 20 | COMPLETED | 450.6s | 0.95x | n/a | 1894.6 | 0.95x | n/a | n/a |
| Qwen3-30B-A3B | async-1off | baseline fuse_loss=false | 2152321 | 19/20 last 20 | COMPLETED | 416.6s | 1.03x | n/a | 2052.3 | 1.03x | n/a | n/a |
| Qwen3-32B | sync | baseline | 2152140 | 19/20 last 20 | COMPLETED | 536.1s | 1.00x | 331.8s | 1269.6 | 1.00x | n/a | n/a |
| Qwen3-32B | sync | Eagle-3 | 2152195 | 19/20 last 20 | COMPLETED | 395.2s | 1.33x | 192.6s | 2150.3 | 1.69x | 45.2% | 2.33 |
| Qwen3-32B | sync | Suffix | 2152499 | 19/20 last 20 | COMPLETED | 592.5s | 0.89x | 386.1s | 1070.8 | 0.84x | 16.9% | 1.94 |
| Qwen3-32B | sync | PARD K=5 | 2152143 | 19/20 last 20 | COMPLETED | 562.2s | 0.95x | 370.6s | 1138.2 | 0.90x | 31.8% | 2.00 |
| Qwen3-32B | sync | PARD-2 | 2152532 | 19/20 last 20 | COMPLETED | 873.4s | 0.61x | 677.2s | 623.9 | 0.49x | 1.0% | 1.05 |
| Qwen3-32B | sync | baseline fuse_loss=false | 2152343 | 19/20 last 20 | COMPLETED | 544.1s | 0.99x | 331.2s | 1272.9 | 1.00x | n/a | n/a |
| Qwen3-32B | async-1off | baseline | 2152144 | 19/20 last 20 | COMPLETED | 227.5s | 1.00x | n/a | 1910.5 | 1.00x | n/a | n/a |
| Qwen3-32B | async-1off | Eagle-3 | 2152196 | 19/20 last 20 | COMPLETED | 189.0s | 1.19x | n/a | 2281.9 | 1.19x | n/a | n/a |
| Qwen3-32B | async-1off | Suffix | 2152146 | 19/20 last 20 | COMPLETED | 494.9s | 0.46x | n/a | 873.3 | 0.46x | n/a | n/a |
| Qwen3-32B | async-1off | PARD K=5 | 2152147 | 19/20 last 20 | COMPLETED | 244.6s | 0.94x | n/a | 1794.2 | 0.94x | n/a | n/a |
| Qwen3-32B | async-1off | PARD-2 | 2152218 | 19/20 last 20 | COMPLETED | 339.3s | 0.70x | n/a | 1344.2 | 0.70x | n/a | n/a |
| Qwen3-32B | async-1off | baseline fuse_loss=false | 2152344 | 19/20 last 20 | COMPLETED | 238.6s | 0.97x | n/a | 1864.5 | 0.98x | n/a | n/a |

## Qwen235B Current State

- OSL=1024 baseline canaries: 2 completed step1 runs; 1 step3 fallback completed step1 then timed out in step2.
- OSL=8192 step20 matrix: 20 jobs cancelled without allocation after the dependency/fallback path; no usable baseline step exists yet.
- Earlier OSL=8192 smoke/fallback attempts: 24 failed/timed out/cancelled rows before a completed step.

| Job | Max OSL | Steps | State | E2E step | Gen time | E2E tok/s/GPU | Gen tok/s/GPU | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2153816 | 1024 | 1/1 last 1 | COMPLETED | 268.6s | 151.6s | 16.7 | 29.6 | OSL1024 step1 canary; completed one step, shutdown warnings only. |
| 2153824 | 1024 | 1/1 last 1 | COMPLETED | 260.0s | 143.3s | 17.2 | 31.3 | OSL1024 backfill step1 canary; clean completed one step. |
| 2154865 | 1024 | 1/3 last 2 | TIMEOUT | 257.2s | 141.6s | 17.5 | 31.7 | OSL1024 step3 fallback; completed step1, timed out in step2 with NCCL ALLGATHER_BASE watchdog / EngineDeadError. |

## Qwen235B Terminal Status Highlights

| Job | OSL | Max steps | State | Elapsed | Nodes | Note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 2152615 | 8192 | 3 | FAILED | 01:38:01 | 32 | OSL8192 no-SHARP alltoall flex-none baseline; failed before first parsed completed step. |
| 2152682 | 8192 | 3 | CANCELLED by 2001147693 | 01:20:19 | 32 | OSL8192 baseline cancelled after about 1h20 in Step 1 policy training; no Total step time. |
| 2153775 | 1024 | 3 | TIMEOUT | 01:15:04 | 32 | OSL1024 step3 attempt timed out; superseded by corrected step1 canaries. |
| 2153816 | 1024 | 1 | COMPLETED | 00:20:25 | 32 | OSL1024 step1 completed; shutdown warning after completion. |
| 2153824 | 1024 | 1 | COMPLETED | 00:19:58 | 32 | OSL1024 step1 completed cleanly. |
| 2154865 | 1024 | 3 | TIMEOUT | 01:15:21 | 32 | OSL1024 step3 completed step1 then timed out in step2 on NCCL ALLGATHER_BASE. |

## Sources

- `docs/lyris_nemorl_perfcfg_latest_summary_20260619.csv`
- `docs/lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv`
- `docs/lyris_nemorl_qwen235b_osl1024_canary_summary_20260619.csv`
- `docs/lyris_nemorl_perfcfg_sacct_20260619.raw`
- `docs/lyris_nemorl_squeue_20260619.raw`
