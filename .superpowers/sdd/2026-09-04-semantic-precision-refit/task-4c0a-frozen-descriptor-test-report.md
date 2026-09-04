# Task 4C0a: Frozen-descriptor test portability

## Implementation

Updated `tests/unit/precision_policy/test_semantic.py` so
`test_descriptor_records_are_frozen_and_slotted` parametrizes concrete
descriptor instances, their declared frozen field names, and replacement
values. `FormatDescriptor.format_id` and `ComponentDescriptor.dtype` are now
mutated, while the existing `__dict__` absence assertion remains unchanged.
The test continues to require `FrozenInstanceError` and does not broaden the
accepted exception or alter production dataclasses.

The remote RED was already captured by Ptyche job 2720948; no remote access
was attempted.

## Test evidence

- Focused GREEN:
  `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy/test_semantic.py -k descriptor_records_are_frozen_and_slotted`
  Result: `2 passed, 174 deselected in 0.11s`.
- Precision-policy GREEN:
  `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy`
  Result: `487 passed in 4.27s`.
- Ruff check: `All checks passed!`
- Ruff format check: `1 file already formatted`.
- `git diff --check`: passed with no output.

## Files changed

- `tests/unit/precision_policy/test_semantic.py`
- `.superpowers/sdd/2026-09-04-semantic-precision-refit/task-4c0a-frozen-descriptor-test-report.md`

## Self-review

The diff is limited to the requested test behavior and keeps both declared
field names explicit in the parameter table. Both parameters exercise actual
frozen slots, and all required gates pass with pristine output.

## Concerns

None.
