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
