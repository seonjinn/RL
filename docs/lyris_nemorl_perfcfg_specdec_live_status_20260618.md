# Lyris NeMo-RL Performance Config SpecDec Live Status: Current Recipe OSL Step20 Matrix

Updated: 2026-06-22 01:57:25

## Methodology

- Workload: NeMo-RL GRPO performance recipes from latest-main merged worktree.
- Sampling: temperature=1.0, top_p=1.0.
- Current recipe-default rows use Qwen3-30B/32B OSL=4096 and Qwen3-235B OSL=8192.
- Hardware: Lyris GB200, 4 GPUs/node, no `--gres`, segment-aware submission.
- Speedups are baseline-matched by model, RL mode, and Max OSL.

## Queue Snapshot

- Completed top-level jobs: 22
- Running top-level jobs: 0
- Pending top-level jobs: 3
- Visible rows: 32; hidden superseded/no-metric rows: 42

## Key Finding

This page now shows only the current recipe-default OSL Step20 matrix; baseline speedups are matched by model, mode, and Max OSL. Qwen30/Qwen32 rows are producing usable metrics; Qwen30 PARD-2 is unsupported with the public AMD Qwen3 PARD-2 drafters because their target dims do not match; Qwen235B is not stable yet, with recipe-OSL attempts failing before a usable step, corrected OSL=1024 CUDA_DEVICE_MAX_CONNECTIONS=1 canaries queued, an OSL=8192 matrix submitted behind the first canary, and a failure-only fallback canary/matrix queued behind afternotok/afterok dependencies.

## Current recipe-default OSL reruns (Qwen30/32=4096, Qwen235B=8192)

| Max OSL | Model | Mode | Method | State | Steps | E2E step | E2E speed | Gen tok/s/GPU | Gen speed | Accept | Mean len | Notes |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | Qwen3-30B-A3B | sync | baseline | COMPLETED | 19/20 last 20 | 421.7s | 1.00x | 1500.1 | 1.00x | n/a | n/a | live partial |
| 4096 | Qwen3-30B-A3B | sync | Eagle-3 | COMPLETED | 19/20 last 20 | 299.4s | 1.39x | 3226.3 | 2.15x | 63.9% | 2.87 | live partial; patched Eagle-3 retry with SpecDec context guard; recipe OSL unchanged |
| 4096 | Qwen3-30B-A3B | sync | Suffix | COMPLETED | 19/20 last 20 | 415.2s | 1.01x | 1731.1 | 1.15x | 11.3% | 1.84 | usable metrics; driver ended with NCCL shutdown warning |
| 4096 | Qwen3-30B-A3B | sync | PARD K=5 | COMPLETED | 19/20 last 20 | 427.5s | 0.98x | 1478.0 | 0.99x | 36.6% | 2.29 | live partial |
| 4096 | Qwen3-30B-A3B | sync | PARD-2 | FAILED | 0/20 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; unsupported with available AMD Qwen3 PARD-2 drafters: Qwen30 target features are 8192-dim, PARD2-Qwen3-8B expects 16384, and PARD2-Qwen3-14B expects 20480 |
| 4096 | Qwen3-30B-A3B | sync | baseline fuse_loss=false | COMPLETED | 19/20 last 20 | 449.7s | 0.94x | 1395.1 | 0.93x | n/a | n/a | live partial; paired baseline rerun with policy.sequence_packing.fuse_loss=false; compare against the recipe-default fuse_loss=true baseline with the same model/mode/OSL |
| 4096 | Qwen3-30B-A3B | async-1off | baseline | COMPLETED | 19/20 last 20 | 425.6s | 1.00x | 1998.4 | 1.00x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU |
| 4096 | Qwen3-30B-A3B | async-1off | Eagle-3 | COMPLETED | 19/20 last 20 | 338.2s | 1.25x | 2510.4 | 1.26x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU; patched Eagle-3 retry with SpecDec context guard; recipe OSL unchanged |
| 4096 | Qwen3-30B-A3B | async-1off | Suffix | COMPLETED | 19/20 last 20 | 477.2s | 0.90x | 1794.7 | 0.90x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU |
| 4096 | Qwen3-30B-A3B | async-1off | PARD K=5 | COMPLETED | 19/20 last 20 | 450.6s | 0.95x | 1894.6 | 0.95x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU |
| 4096 | Qwen3-30B-A3B | async-1off | baseline fuse_loss=false | COMPLETED | 19/20 last 20 | 416.6s | 1.03x | 2052.3 | 1.03x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU; paired baseline rerun with policy.sequence_packing.fuse_loss=false; compare against the recipe-default fuse_loss=true baseline with the same model/mode/OSL |
| 4096 | Qwen3-32B | sync | baseline | COMPLETED | 19/20 last 20 | 536.1s | 1.00x | 1269.6 | 1.00x | n/a | n/a | live partial |
| 4096 | Qwen3-32B | sync | Eagle-3 | COMPLETED | 19/20 last 20 | 395.2s | 1.33x | 2150.3 | 1.69x | 45.2% | 2.33 | live partial; patched Eagle-3 retry with SpecDec context guard; recipe OSL unchanged |
| 4096 | Qwen3-32B | sync | Suffix | TIMEOUT | 16/20 last 18 | 788.6s | 0.68x | 733.0 | 0.58x | 7.9% | 1.56 | live partial |
| 4096 | Qwen3-32B | sync | Suffix | COMPLETED | 19/20 last 20 | 592.5s | 0.89x | 1070.8 | 0.84x | 16.9% | 1.94 | live partial |
| 4096 | Qwen3-32B | sync | PARD K=5 | COMPLETED | 19/20 last 20 | 562.2s | 0.95x | 1138.2 | 0.90x | 31.8% | 2.00 | live partial |
| 4096 | Qwen3-32B | sync | PARD-2 | TIMEOUT | 16/20 last 18 | 802.7s | 0.66x | 688.1 | 0.54x | 1.0% | 1.05 | live partial; PARD-2 retry after static patched-vLLM source validation fallback; recipe OSL unchanged |
| 4096 | Qwen3-32B | sync | PARD-2 | COMPLETED | 19/20 last 20 | 873.4s | 0.61x | 623.9 | 0.49x | 1.0% | 1.05 | live partial; PARD-2 retry reusing the global patched vLLM site to avoid per-run vLLM import/build failure; recipe OSL unchanged |
| 4096 | Qwen3-32B | sync | baseline fuse_loss=false | COMPLETED | 19/20 last 20 | 544.1s | 0.99x | 1272.9 | 1.00x | n/a | n/a | live partial; qwen32 fuse_loss=false retry using ++policy.sequence_packing.fuse_loss=false; recipe OSL unchanged |
| 4096 | Qwen3-32B | async-1off | baseline | COMPLETED | 19/20 last 20 | 227.5s | 1.00x | 1910.5 | 1.00x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU |
| 4096 | Qwen3-32B | async-1off | Eagle-3 | COMPLETED | 19/20 last 20 | 189.0s | 1.19x | 2281.9 | 1.19x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU; patched Eagle-3 retry with SpecDec context guard; recipe OSL unchanged |
| 4096 | Qwen3-32B | async-1off | Suffix | COMPLETED | 19/20 last 20 | 494.9s | 0.46x | 873.3 | 0.46x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU |
| 4096 | Qwen3-32B | async-1off | PARD K=5 | COMPLETED | 19/20 last 20 | 244.6s | 0.94x | 1794.2 | 0.94x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU |
| 4096 | Qwen3-32B | async-1off | PARD-2 | COMPLETED | 19/20 last 20 | 339.3s | 0.70x | 1344.2 | 0.70x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU; PARD-2 retry using shared patched vLLM source site cache; recipe OSL unchanged |
| 4096 | Qwen3-32B | async-1off | baseline fuse_loss=false | COMPLETED | 19/20 last 20 | 238.6s | 0.97x | 1864.5 | 0.98x | n/a | n/a | usable metrics; driver ended with NCCL shutdown warning; async log omits generation_time_s; using generation worker tok/s/GPU; qwen32 fuse_loss=false retry using ++policy.sequence_packing.fuse_loss=false; recipe OSL unchanged |
| 8192 | Qwen3-235B-A22B | sync | baseline | FAILED | 0/20 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; failed during Step 1 policy training with 3600s NCCL watchdog timeouts across EP alltoall, CP send/recv, and PP groups; no usable baseline step |
| 8192 | Qwen3-235B-A22B | sync | baseline | FAILED | 0/3 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; qwen235B corrected no-SHARP alltoall flex-none smoke with quoted CUDA_DEVICE_MAX_CONNECTIONS env override and fuse_loss=false |
| 8192 | Qwen3-235B-A22B | sync | baseline | CANCELLED by 2001147693 | 0/3 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; qwen235B OSL8192 no-SHARP alltoall flex-none smoke with quoted CUDA_DEVICE_MAX_CONNECTIONS env override and fuse_loss=true; cancelled after ~1h20 in Step 1 policy training with no Total step time |

## Historical OSL=1024 short/debug runs

| Max OSL | Model | Mode | Method | State | Steps | E2E step | E2E speed | Gen tok/s/GPU | Gen speed | Accept | Mean len | Notes |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | Qwen3-235B-A22B | sync | baseline_osl1024_repro | FAILED | 0/3 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; OSL=1024 repro canary after 2152682 stuck; no SHARP, segment=16, alltoall, fuse_loss=false; failed before Step 1 because CUDA_DEVICE_MAX_CONNECTIONS was exported as an empty string |
| 1024 | Qwen3-235B-A22B | sync | baseline_osl1024_repro | PENDING | 0/3 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; corrected OSL=1024 repro canary with CUDA_DEVICE_MAX_CONNECTIONS=1; no SHARP, segment=16, alltoall, flex backend none, fuse_loss=false |
| 1024 | Qwen3-235B-A22B | sync | baseline_osl1024_repro | PENDING | 0/1 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; shorter corrected OSL=1024 Step-1 canary with CUDA_DEVICE_MAX_CONNECTIONS=1; no SHARP, segment=16, alltoall, flex backend none, fuse_loss=false |
| 1024 | Qwen3-235B-A22B | sync | baseline_osl1024_repro | PENDING | 0/1 | n/a | waiting baseline | n/a | waiting baseline | n/a | n/a | waiting for first parsed NeMo-RL step; gb200-backfill corrected OSL=1024 Step-1 canary with CUDA_DEVICE_MAX_CONNECTIONS=1; no SHARP, segment=16, alltoall, flex backend none, fuse_loss=false |

