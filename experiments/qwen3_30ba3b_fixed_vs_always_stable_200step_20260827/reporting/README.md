# Stable Q30 fixed-versus-always report

`fixed_vs_always_report.py` collects the four W&B runs in group
`q30ba3b-fixed-vs-always-stable-200step-20260827` and writes deterministic JSON
plus a self-contained HTML report.

The report uses a closed step window from 3 through 200. It merges repeated
W&B history records by step, omits null and non-finite observations per metric,
and discloses the included steps, missing steps, and valid count for every
displayed metric. A comparison is `ready` only when both W&B runs are
`finished`, every closed-window step 3–200 is present, and all four comparison
metrics have a valid observation at every one of those 198 steps. Failed,
running, missing-step, or sparse-metric histories remain visible as
`preliminary` and do not receive speedup claims.

Throughput comes only from the canonical logged fields:

- `performance/generation_tokens_per_sec_per_gpu`
- `performance/tokens_per_sec_per_gpu`

It is never reconstructed from averaged time or token counts. Generation-time,
E2E step-time, generation-throughput, and E2E-throughput ratios compare each
always-online arm only against its same-drafter fixed arm. Fixed means the
generation drafter remains enabled but its training is frozen; it is not a
no-SpecDec baseline.

## W&B collection

Provide W&B authentication through the environment's normal supported
mechanism. Do not put credentials in a command, fixture, report, or source
file.

```bash
uv run --no-project --with wandb python \
  experiments/qwen3_30ba3b_fixed_vs_always_stable_200step_20260827/reporting/fixed_vs_always_report.py \
  --entity nvidia \
  --project sna-specdec \
  --group q30ba3b-fixed-vs-always-stable-200step-20260827 \
  --json-output /tmp/q30-fixed-vs-always.json \
  --html-output /tmp/q30-fixed-vs-always.html
```

Online collection uses `scan_history(min_step=3, max_step=201)`, where the W&B
maximum is exclusive, to implement the required closed 3–200 window. If a
variant has retries, the collector selects the latest `created_at`, breaking
ties with the run ID. Outputs use atomic replacement.

## Offline input

For offline validation, pass a JSON file shaped as
`{"runs": [{"id": "...", "name": "...", "history": [{"_step": 3, ...}]}]}`
with `--history-json`. Exact launcher names follow
`q30ba3b-stable-200step-{drafter}-{fixed|always}-k5-{32-hex-id}`.
