# Task 10 Report: MXFP8 MoE Tactic-Audit Review Fixes

## Delivered

- Collector now parses the GRPO producer labels `Loss`, `Generation KL Error`,
  `Avg Reward`, and `Mean Generation Length` together with the established
  timing and throughput fields. It requires exactly measured steps 3-8,
  explicit positive per-step realized token evidence, exit success, and
  successful `refit`, `rollout`, `logprob`, and `train` phases.
- Validation submission now writes `run_evidence.json` with arm, exit, phase,
  metadata, and observed Training Results evidence. It intentionally records
  unavailable exact token counts as `null`; the collector fails closed instead
  of inferring tokens from mean generation length.
- Qualification decisions, stock/candidate cache selections, trace summary,
  profile coverage, and hashes bind exact successful shmoo rows. Failed or
  zero-timing rows cannot select a tactic. NSys FC1/GEMM1 and FC2/GEMM2 rows
  are keyed by signature, cache key, arm, component, and selected tactic.
- Executed failed correctness evidence renders `REJECT` with raw data;
  insufficient/malformed evidence renders `INCOMPLETE` or `NOT YET EXECUTED`.
  Promotion requires at least two comparable runs per arm and measured
  run-to-run variation. Within-run steps are displayed separately.
- HTML is structured with tables and embedded figures. Executed reports carry
  raw steps 3-8 reward/loss/KL tables, four figure links, cache/trace/decision
  hashes, source fingerprints, and an explicit KEEP/REJECT conclusion.
- Regenerated the committed no-artifacts template. It is explicitly `NOT YET
  EXECUTED`, contains no performance claims, and embeds four unavailable-data
  figures for Task 11 to replace with measured artifacts.

## Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py

24 passed, 16 macOS pytest temporary-directory cleanup warnings in 12.69s

PYTHONPATH=. .venv/bin/ruff check \
  experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py

All checks passed.

PYTHONPATH=. .venv/bin/pyright \
  experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py

0 errors; 1 seaborn source-resolution warning.
```

All four template PNGs are 4140x1473 at 600 DPI. Their PDF counterparts are
single-page vector PDF 1.4 files. Visual inspection confirms the template
plots clearly state `NOT YET EXECUTED` and contain no fabricated values.
The regenerated HTML has four `<figure>/<img>` entries and no `<pre>` block;
the Markdown and HTML include steps 3-8, phase status, realized-token,
finite-metric, 95% coverage, FC1/GEMM1, FC2/GEMM2, cache hit/fallback, GSM8K,
and trace/qualification provenance requirements.

## Commit

Review-fix implementation: `8f02405429af48879426c81d1a2e62cce00a6beb`

## Fix Round 2

- The synchronous GRPO producer emits explicit successful `refit`, `rollout`,
  `logprob`, and `train` markers for each completed step. The validation
  launcher derives `run_evidence.json` from those markers and the producer's
  `train_data_step*.jsonl` `token_loss_mask` fields. It records only positive,
  directly measured generated-token totals for steps 3-8.
- `qualify_cache.py` now emits authoritative `qualification_decisions.json`
  alongside its real cache manifest. It includes selected and stock tactics,
  promotion decisions, signature bindings, and cache/trace/profile/shmoo
  fingerprints. The report checks these bindings and excludes failed/zero
  shmoo rows before any tactic metric is accepted.
- Trace-set provenance now uses the same sorted member-digest JSON algorithm
  as `CacheProvenance._sha256_file_set`. Report source hashes list each raw
  trace artifact. Run evidence must carry cache-manifest runtime fingerprints.
- Paired GSM8K output now records `matched_examples=1319`; collection requires
  provenance matching, all paired count fields summing to 1319, accuracy,
  McNemar, and CI data. Executed reports show replay coverage and paired GSM8K
  comparison values.
- Existing execution artifacts that are malformed, mismatched, or missing
  component evidence render `INCOMPLETE`; only a complete failed gate renders
  `REJECT`. Duplicate paths/run IDs, wrong arm labels, and non-comparable
  repetition manifests cannot satisfy the stability gate. Per-step plots now
  show every repetition, labelled by arm/run/step.

## Fix Round 2 Verification

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py

60 passed, 16 macOS pytest temporary-directory cleanup warnings

PYTHONPATH=. .venv/bin/ruff check <changed audit modules and tests>
All checks passed.

PYTHONPATH=. .venv/bin/pyright <changed audit modules>
0 errors; 1 pre-existing seaborn source-resolution warning.
```

Regenerated template report: all four PNGs are 4140x1473 at 600 DPI; HTML is
structured and has four embedded figure elements. The template is explicitly
`NOT YET EXECUTED` and makes no measured-performance claim.
