# Qwen3-30B-A3B cadence report

`cadence_report.py` collects the six Q30 cadence runs from W&B group
`q30ba3b-draft-cadence-200step-20260826` and writes deterministic JSON plus a
self-contained HTML report. It aggregates the closed window from step 3
through step 200. A missing step leaves the run visible but labels it
`preliminary`.

Launcher names follow
`q30ba3b-200step-{variant}-k5-{32-character-hex-id}`. The collector prefers an
allowlisted `variant` in run metadata (including `config.variant` or
`config.cadence_variant`) and otherwise accepts only that exact launcher-name
contract. If a variant has retries, the collector deterministically keeps the
attempt with the latest W&B `created_at` value, breaking ties by run ID.

The report uses only canonical W&B throughput fields:

- `performance/generation_tokens_per_sec_per_gpu`
- `performance/tokens_per_sec_per_gpu`

It never derives throughput from averaged timing or token-count data. Always
and fixed-10 arms are compared only with their same-drafter static arm. These
are cadence-relative comparisons; the matrix has no matched no-SpecDec
baseline, so the output does not claim a SpecDec-versus-baseline speedup.

Mean accepted length uses only `train/vllm/spec_acceptance_length` and
`vllm/spec_acceptance_length`. Aggregate accepted-token counters are used only
as an optional numerator when deriving acceptance rate; they are never treated
as an accepted length.

Cadence decision reasons come from each completed run's terminal
`cadence/schedule-runtime.json`, not guessed W&B history keys. That artifact is
constructed from the sealed decision ledger and terminal evidence by the
cadence runtime. Pass the durable experiment root with
`--cadence-artifact-root`; JSON and HTML outputs include the resolved artifact
path, terminal step, and decision count as provenance. Runs without a terminal
artifact remain visible with no cadence reason counts.

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
  --cadence-artifact-root /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_draft_cadence_200step_20260826 \
  --json-output /tmp/q30-cadence-report.json \
  --html-output /tmp/q30-cadence-report.html
```

Online collection uses `scan_history(min_step=3, max_step=201)`, which gives
the required closed 3-200 window. Outputs are written by atomic replacement.

## Local checks

```bash
python3 -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests/test_cadence_report.py
ruff check experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting
pyright experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/cadence_report.py
python3 -m py_compile experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/cadence_report.py
```
