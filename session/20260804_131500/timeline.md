# Timeline

## 2026-08-04 13:15 PDT

- User approved GCP-NRT for the MXFP8 measurements.
- Selected Qwen3-30B-A3B, four B200 nodes, and a 20-step matched A/B.
- Kept PR 3477 performance work isolated from the feature branch.

## 2026-08-04 17:27 PDT

- Scheduler preflight passed and jobs `496441` and `496442` started.
- Both failed at W&B initialization before model setup.
- Verified the launcher wrote a 41-byte hostname from a one-line `.netrc`
  entry instead of the 86-byte password field.
- Reproduced the parser failure with a synthetic one-line entry before fixing it.
- API verification showed the exported 40-character key is invalid while the
  86-character `.netrc` token is valid; the launcher now prefers `.netrc`.

## 2026-08-04 17:45 PDT

- Jobs `496459` and `496460` passed W&B and reached vLLM model initialization.
- Both failed at `assert self.moe_kernel is not None` in vLLM 0.25 ModelOpt.
- Cherry-picked `553b985aa` and `c12c0eb7a`, which initialize the modular MoE
  kernel and restore the quantized value/scale tensor shapes.
- Python compilation and `git diff --check` passed for both fixes.

## 2026-08-04 17:54 PDT

- Jobs `496495` and `496496` failed while force-rebuilding the existing shared
  Lustre worker environments with `OSError: [Errno 116] Stale file handle`.
- Switched to a new versioned cache root and disabled in-place force rebuilds.

## 2026-08-04 18:33 PDT

- Jobs `496508` and `496509` passed worker-environment setup, vLLM model load,
  and the previous `moe_kernel` failure point.
- Both matched arms reached step 4 without a traceback or OOM.
- W&B runs: legacy `37hdhdbt`; NCCL-Reshard `i4xg5s7k`.

## 2026-08-04 20:10 PDT

- Both arms recorded all 20 steps and all requested W&B samples for steps 3-20.
- PR 3477 reduced transfer/update from 8.697 s to 4.186 s and total refit from
  8.214 s to 3.955 s.
- E2E step time decreased from 308.450 s to 304.135 s and throughput increased
  from 830.875 to 842.150 tokens/s/GPU.
- Legacy job `496508` completed with exit 0. NCCL job `496509` completed the
  workload but exited 1 during Ray interpreter shutdown after W&B finalization.
- Added `experiments/pr3477_refit_ab/RESULTS.md`.
