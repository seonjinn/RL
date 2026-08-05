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
