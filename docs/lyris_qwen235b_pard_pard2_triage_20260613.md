# Lyris Qwen3-235B PARD/PARD-2 Triage - 2026-06-13

Refreshed at `2026-06-13T09:51:16+02:00`.

Source: `docs/lyris_qwen235b_standalone_live_diagnostics_20260613.csv`

## Current Read

- PARD now has 3 final SWE rows in the current K9/K11 sweep: full batch 2 PARD K9, full batch 2 PARD K11, and full batch 8 PARD K9.
- The PARD final rows are still weak relative to Suffix and Eagle-3: `6.18-19.98` tok/s/GPU with `16.25%-18.12%` acceptance, and no matched final baseline row yet.
- PARD-2 K9/K11 still has zero final SWE rows in this sweep.
- No PARD/PARD-2 row shows a parsed latest error in the diagnostics CSV; remaining rows have `tail_status=ok`.
- The current blocker for PARD-2 is acceptance/performance, not launcher failure.

## Aggregate

| Dataset | Method | Jobs | Final rows | Running | Mean gen tok/s | Mean acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench full | PARD K9 | 5 | 2 | 4 | 98.80 | 7.8% |
| SWE-Bench full | PARD K11 | 5 | 1 | 4 | 101.90 | 8.2% |
| SWE-Bench full | PARD-2 K9 | 5 | 0 | 5 | 59.22 | 4.6% |
| SWE-Bench full | PARD-2 K11 | 5 | 0 | 5 | 62.12 | 4.1% |
| SWE-Bench-Verified | PARD K9 | 5 | 0 | 5 | 101.74 | 6.9% |
| SWE-Bench-Verified | PARD K11 | 5 | 0 | 5 | 129.38 | 8.3% |
| SWE-Bench-Verified | PARD-2 K9 | 5 | 0 | 5 | 160.42 | 5.1% |
| SWE-Bench-Verified | PARD-2 K11 | 5 | 0 | 5 | 62.56 | 4.3% |

## Interpretation

PARD K9/K11 is now measurable as final rows, but the accepted-token signal is too low to call it competitive on Qwen3-235B SWE OSL32K. PARD-2 remains a negative/immature signal in this configuration: it is running cleanly, but still has no final rows and low live acceptance.
