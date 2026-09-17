# GPU gate receipt

## Verification before submission

- Focused refit suite: Ptyche job `2845677`, 277 passed, 9 skipped, 70
  warnings in 1,081.80 seconds. The skips are pre-existing hardware/test
  constraints documented by their test markers.
- Async deep-refit lifecycle: Ptyche job `2845786`, 6 passed, 117 deselected,
  17 warnings in 23.08 seconds.
- Local static validation: Ruff check, Ruff format check, Python compileall,
  and `git diff --check` passed for all touched implementation and test files.

## Three-step GPU gate

| Arm | Job | State at receipt update | Source SHA |
|---|---:|---|---|
| Baseline, legacy level-1 | `2845797` | Pending | `26366cd97657c09181a988de33b736bcf311e638` |
| DFlash K7 B8, deep refit | `2845799` | Pending | `26366cd97657c09181a988de33b736bcf311e638` |

Artifact directories:

- `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-specdec-deep-refit-20260917/Qwen3-235B-Baseline-Legacy-3step-20260917T235620Z`
- `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-specdec-deep-refit-20260917/Qwen3-235B-DFlashK7-B8-DeepRefit-3step-20260917T235620Z`

Final metrics and acceptance-gate decisions remain pending job completion.
