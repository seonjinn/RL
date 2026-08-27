# Qwen3-30B-A3B cadence report

`cadence_report.py` collects the six Q30 cadence runs from W&B group
`q30ba3b-draft-cadence-200step-20260826` and writes deterministic JSON plus a
self-contained HTML report. It aggregates the closed window from step 3
through step 200. A missing step leaves the run visible but labels it
`preliminary`.

The report uses only canonical W&B throughput fields:

- `performance/generation_tokens_per_sec_per_gpu`
- `performance/tokens_per_sec_per_gpu`

It never derives throughput from averaged timing or token-count data. Always
and fixed-10 arms are compared only with their same-drafter static arm. These
are cadence-relative comparisons; the matrix has no matched no-SpecDec
baseline, so the output does not claim a SpecDec-versus-baseline speedup.

## Offline verification

The checked-in fixture has the portable input schema accepted by
`--history-json`. Run it without importing or contacting W&B:

```bash
python3 experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/cadence_report.py \
  --history-json experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests/fixtures/history.json \
  --json-output /tmp/q30-cadence-report.json \
  --html-output /tmp/q30-cadence-report.html
```

## W&B collection

Install the optional `wandb` Python package in the environment and provide
authentication through the environment using W&B's normal supported mechanism.
Do not place credentials in a command, fixture, report, or source file.

```bash
python3 experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/cadence_report.py \
  --entity sna \
  --project sna-specdec \
  --group q30ba3b-draft-cadence-200step-20260826 \
  --json-output /tmp/q30-cadence-report.json \
  --html-output /tmp/q30-cadence-report.html
```

Online collection uses `scan_history(min_step=3, max_step=201)`, which gives
the required closed 3-200 window. Outputs are written by atomic replacement.

## Local checks

```bash
python3 -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests/test_cadence_report.py
ruff check experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting
python3 -m py_compile experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/cadence_report.py
```
