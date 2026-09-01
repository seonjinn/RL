# Task 4 report: Q30 calibration and schedule selection

## Status

Complete. The Q30 package now accepts explicitly validated calibration summary
rows for one drafter, checks the exact required BS/K grid independently for
every repetition, and returns the best fixed K plus a contiguous monotone
schedule covering batch sizes 1 through 128. This task does not render jobs,
submit work, or generate live calibration results.

## Commit

- `8ae3c45de` — `exp: calibrate Q30 dynamic K schedules` (signed off)
- `6c9e95819` — `docs: report Q30 calibration selector` (signed off)
- `0216da303` — `fix: optimize Q30 monotone calibration globally` (signed off)
- This reviewed report update is committed separately.

## Objective and selection policy

Each result contributes its own natural-EOS throughput:
`output_tokens / elapsed_seconds`. Repetition values are not pooled by summing
tokens or durations. Instead, each BS/K cell uses the median of its per-result
throughputs. The best fixed K maximizes the equal-weight arithmetic mean of
those cell medians across the nine required batch sizes. Exact throughput ties
select the smaller K.

The monotone schedule globally maximizes the sum of BS/K cell-median
throughputs across all nine sampled batch sizes. Dynamic programming retains
the best cumulative path for every `(batch index, ending K)` state while only
allowing transitions to equal or smaller K. Equal-score paths use the
lexicographically smaller K tuple from low to high batch size. Adjacent equal-K
grid points are merged, gaps between sampled batch sizes are assigned to the
preceding fitted range, and the final range ends at BS128.

## TDD evidence

### RED: table-driven schedule and best fixed K

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k calibration_selects_hand_calculated
```

Result: exit 2 during collection with `ModuleNotFoundError: No module named
'experiments.vllm_028_q30_sync_dynamicsd.calibrate'`. This was the intended
failure before `calibrate.py` existed.

### GREEN: literal schedule

The same command exited 0 with `1 passed, 72 deselected in 0.05s`. The
hand-calculated throughput table selected:

```text
[[1,8,5],[9,32,3],[33,64,2],[65,128,0]]
```

It also selected K5 as the best fixed K. The BS2 raw winner was K7 but the
monotone fit retained K5, and the BS64 K2/K3 tie selected K2.

### RED: grid, validation, repetition, and capability gates

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 'calibration_'
```

Result: exit 1 with `16 failed, 8 passed, 66 deselected`. The missing gates
accepted duplicate and incomplete grids, incomplete repetitions, invalid and
unvalidated rows, mixed drafters, and out-of-grid K8 values; malformed values
also leaked internal exceptions instead of contract `ValueError`s.

### GREEN: strict selector boundary

After adding one validation pass, the same command exited 0 with `24 passed,
66 deselected in 0.07s`. The passing cases cover:

- the exact BS `{1,2,4,8,16,32,64,96,128}` and K `{0,1,2,3,5,7}` grid for
  every repetition;
- duplicate `(repetition, BS, K)` rejection and incomplete repetition
  rejection;
- explicit validated-row status, strict integer fields, positive finite
  timing, and positive token counts;
- smaller-K ties for both schedule and best fixed selection;
- DFlash capability through K7 and DSpark capability through K8, while
  rejecting DSpark K8 from the current calibration grid through K7;
- one requested drafter per selector call.

## Review remediation: globally optimal monotone path

The first implementation selected each batch independently and then clamped
later K values to the preceding fitted K. That greedy fit could discard the
globally best path based only on the first batch.

### RED: greedy counterexample

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k global_monotone_throughput_optimum
```

Result: exit 1 with `1 failed, 90 deselected`. The exact table assigned BS1
throughput K0=10 and K7=9, every later sampled BS throughput K0=10 and K7=100,
and all other K values throughput 1. The greedy implementation returned
`[[1,128,0]]`, scoring 90, instead of the globally optimal K7 path, scoring
`9 + 8 * 100 = 809`.

### GREEN: dynamic-programming optimum and preserved behavior

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 'global_monotone_throughput_optimum or calibration_selects_hand_calculated or calibration_ties_choose'
```

Result: exit 0 with `3 passed, 88 deselected`. The counterexample now returns
`[[1,128,7]]`; the original required hand schedule remains unchanged; and the
equal-throughput case still returns the lexicographically smaller all-K0 path.

## Final verification

- `python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py` — exit 0,
  `91 passed`.
- `python3 -m pytest -q tests/test_vllm028_nemotron_bf16_matrix.py
  tests/test_vllm028_mrv2_patch_canary.py tests/test_vllm028_mrv2_patch_matrix.py`
  — exit 0, `88 passed`.
- `ruff check experiments/vllm_028_q30_sync_dynamicsd
  tests/test_vllm028_q30_sync_dynamicsd.py` — exit 0, `All checks passed!`.
- `pyright experiments/vllm_028_q30_sync_dynamicsd
  tests/test_vllm028_q30_sync_dynamicsd.py` — exit 0, `0 errors, 0 warnings,
  0 informations`.
- `git diff --cached --check` before the implementation commit — exit 0.

## Files changed

- `experiments/vllm_028_q30_sync_dynamicsd/calibrate.py`
- `tests/test_vllm028_q30_sync_dynamicsd.py`
- `.superpowers/sdd/2026-09-01-vllm028-q30-sync-dynamicsd-osl1k/task-4-report.md`

## Concerns

- The passing pytest suites emitted the existing non-failing temporary
  directory cleanup warnings (`OSError: [Errno 66] Directory not empty`) from
  the shared staging-test environment.
- `validated=True` is an explicit boundary marker on the summary row; the
  live-results collector remains responsible for creating rows only after the
  existing Task 3 worker-result validator succeeds.
- No independent reviewer was dispatched because this task explicitly
  prohibited subagents; staged-diff inspection and the full local gates were
  used instead.
