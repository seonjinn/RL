# Task 2 report: Q30 workload and method contract

## Status

Complete after review remediation. The package defines frozen, typed, strictly validated
`ExperimentContract` and `MethodPlan` records plus deterministic calibration
and barrier matrix builders. No runner, calibration-selection, rendering, or
submission behavior was added.

## Commits

- `dc6039f02` — `exp: define Q30 DynamicSD benchmark contract` (signed off)
- `138dd21a6` — `docs: report Q30 contract task` (signed off)
- `669eec2f2` — `fix: harden Q30 benchmark contracts` (signed off)

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

## Review remediation TDD evidence

### RED/GREEN: defensive mapping copies

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k defensively_freezes
```

RED result: exit 1, `1 failed, 10 deselected in 0.03s`; mutating the caller's
source dictionary changed `contract.drafter_paths`. GREEN result: exit 0,
`1 passed, 10 deselected in 0.02s` after copying both mappings into independent
`MappingProxyType` views during initialization.

### RED/GREEN: calibrated fixed-K barrier materialization

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 'materialize_only or incomplete_or_non_grid'
```

RED result: exit 1, `4 failed, 11 deselected in 0.04s`; the builder rejected
the missing `best_fixed_k` keyword. GREEN result: exit 0,
`4 passed, 11 deselected in 0.02s`. Exact two-drafter mappings now materialize
only fixed rows, reject non-grid/incomplete/Boolean values, and retain K0's
diagnostic controller identity.

### RED/GREEN: DSpark adaptive compatibility opt-in

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k 'barrier_arm_requires or keep_each_method'
```

RED result: exit 1, `2 failed, 14 deselected in 0.04s`; the default matrix
still contained the adaptive arm and the opt-in keyword was absent. GREEN
result: exit 0, `2 passed, 14 deselected in 0.01s`; the default now excludes
the arm and `include_dspark_adaptive=True` adds it explicitly.

### RED/GREEN: deterministic seeds and natural EOS

Command:

```text
python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py -k unique_per_request
```

RED result: exit 1, `1 failed, 16 deselected in 0.04s` with `AttributeError`
for the missing `seed_for_request`. GREEN result: exit 0,
`1 passed, 16 deselected in 0.02s`. The contract now pins base seed
`20260901`, derives 2,048 unique seeds by global request index, validates the
index bounds, and pins `ignore_eos=False`.

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

Post-review verification:

- `python3 -m pytest -q tests/test_vllm028_q30_sync_dynamicsd.py` — exit 0,
  `17 passed in 0.02s`.
- `python3 -m pytest -q tests/test_vllm028_nemotron_bf16_matrix.py tests/test_vllm028_mrv2_patch_canary.py tests/test_vllm028_mrv2_patch_matrix.py`
  — exit 0, `88 passed in 2.46s`.
- `ruff check experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 0, `All checks passed!`.
- `pyright experiments/vllm_028_q30_sync_dynamicsd tests/test_vllm028_q30_sync_dynamicsd.py`
  — exit 0, `0 errors, 0 warnings, 0 informations`.

## Files changed

- `experiments/vllm_028_q30_sync_dynamicsd/__init__.py`
- `experiments/vllm_028_q30_sync_dynamicsd/contract.py`
- `tests/test_vllm028_q30_sync_dynamicsd.py`
- `.superpowers/sdd/2026-09-01-vllm028-q30-sync-dynamicsd-osl1k/task-2-report.md`

## Concerns

- The 88-test foundation run passed but emitted non-failing Pytest temporary
  directory cleanup warnings (`OSError: [Errno 66] Directory not empty`) from
  existing staging tests.
- Default barrier fixed-K rows remain unresolved (`verifier_k=None`) until
  Task 4 selection, but callers can now materialize both fixed rows explicitly
  from a validated `best_fixed_k` mapping. Dynamic/adaptive/baseline rows never
  inherit those fixed K values.
- The requested no-subagent constraint prevented dispatching a separate review
  agent; verification and a local diff review were performed instead.
