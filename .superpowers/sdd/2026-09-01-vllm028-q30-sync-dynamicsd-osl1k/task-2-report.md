# Task 2 report: Q30 workload and method contract

## Status

Complete. The package defines frozen, typed, strictly validated
`ExperimentContract` and `MethodPlan` records plus deterministic calibration
and barrier matrix builders. No runner, calibration-selection, rendering, or
submission behavior was added.

## Commits

- `dc6039f02` — `exp: define Q30 DynamicSD benchmark contract` (signed off)

## TDD evidence

### RED: workload contract

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py
```

Result: exit 2 during collection with
`ModuleNotFoundError: No module named 'experiments.vllm_028_q30_sync_dynamicsd'`.
This was the intended failure because the Task 2 package did not exist.

### GREEN: workload contract

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py
```

Result: exit 0, `2 passed in 0.01s`.

### RED: method and matrix contracts

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py
```

Result: exit 2 during collection with
`ImportError: cannot import name 'MethodPlan'`. This was the intended failure
before adding the method contract and matrix builders.

### GREEN: complete matrix

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py
```

Result: exit 0, `10 passed in 0.03s`.

### RED/GREEN: explicit container and external-engine identity

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py
```

RED result: exit 1, `1 failed, 9 passed`; the failure was the intended
`AttributeError` for the missing `container_path`. After adding the pinned
container and external coordination fields, the same command exited 0 with
`10 passed in 0.02s`.

## Final verification

- `python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py` — exit 0,
  `10 passed in 0.02s`.
- `python3 -m pytest -q tests/test_vllm028_nemotron_bf16_matrix.py tests/test_vllm028_mrv2_patch_canary.py tests/test_vllm028_mrv2_patch_matrix.py`
  — exit 0, `88 passed in 2.46s`.
- `ruff check experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 0, `All checks passed!`.
- `pyright experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 0, `0 errors, 0 warnings, 0 informations`.
- `git diff --cached --check` before the implementation commit — exit 0.

## Files changed

- `experiments/vllm_028_q30_sync_dynamicsd/__init__.py`
- `experiments/vllm_028_q30_sync_dynamicsd/contract.py`
- `tests/test_vllm028_q30_sync_dynamicsd.py`
- `.superpowers/sdd/2026-09-01-vllm028-q30-sync-dynamicsd-osl1k/task-2-report.md`

## Concerns

- The 88-test foundation run passed but emitted non-failing Pytest temporary
  directory cleanup warnings (`OSError: [Errno 66] Directory not empty`) from
  existing staging tests.
- Barrier fixed-K rows intentionally leave `verifier_k=None`; Task 4 selects
  and freezes the measured best K. DynamicSD rows likewise identify the
  controller without inventing a schedule before calibration.
- The requested no-subagent constraint prevented dispatching a separate review
  agent; verification and a local diff review were performed instead.
