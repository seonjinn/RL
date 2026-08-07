# Task 10 Report: MXFP8 MoE Tactic-Audit Results

## Delivered

- Added a fail-closed collector for exactly steps 3-8, successful
  refit/rollout/logprob/train phases, realized token counts, finite
  reward/loss/KL, and manifest invariants.
- Added 600-DPI PNG and vector-PDF report plots for FC1/GEMM1 and FC2/GEMM2
  micro speedup, tactic/cache shares, stock-normalized end-to-end metrics, and
  per-step variation.
- Added Markdown and HTML reports with explicit KEEP/REJECT logic. Missing
  inputs render REJECT/incomplete evidence and do not report performance
  numbers.
- Generated the committed not-yet-executed template. Task 11 will replace it
  with measured artifacts.

## Verification

```text
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py

5 passed, 16 warnings in 2.20s

.venv/bin/ruff check \
  experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py

All checks passed.

.venv/bin/pyright \
  experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py

0 errors, 1 seaborn source-resolution warning
```

The four template PNGs are 4140x1380 at 600 DPI. The PDF counterparts are
single-page vector PDF 1.4 files. HTML and Markdown contain the required
REJECT/incomplete state, steps 3-8, phase, token, finite-metric, 95%,
FC1/GEMM1, FC2/GEMM2, cache hit/fallback, GSM8K, raw-table, and source-hash
terms.

## Commit

Implementation commit: `4cebcb2b0c337a3a57fadbeb5506d5680788b209`
