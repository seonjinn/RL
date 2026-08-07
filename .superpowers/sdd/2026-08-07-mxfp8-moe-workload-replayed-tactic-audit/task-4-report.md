# Task 4 Report

## Delivered

- Added frozen, typed schemas for routing traces, tactic pairs, tactic
  measurements, and replay profiles.
- Canonical routing signature keys use SHA256 over sorted, ASCII JSON and omit
  `sampled_gpu_time_us`.
- Added strict known-field parsing, JSON round trips, and validation for the
  trace invariants in the task brief.

## TDD Evidence

- RED: `PYTHONPATH="$PWD" .venv/bin/pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py`
  failed during collection with `ModuleNotFoundError: No module named
  'experiments.mxfp8_moe_tactic_audit.schema'` before `schema.py` existed.
- GREEN: the same targeted test command passed with `22 passed` after the
  implementation.

## Verification

```text
PYTHONPATH="$PWD" .venv/bin/pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
.venv/bin/ruff check experiments/mxfp8_moe_tactic_audit/schema.py tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
.venv/bin/pyright experiments/mxfp8_moe_tactic_audit/schema.py
```

All commands completed successfully: 22 tests passed, Ruff reported no
findings, and Pyright reported 0 errors, warnings, or information messages.

## Follow-up Review Fix

- Normalized direct `RoutingSignature` construction to store
  `expert_counts` as a tuple before validation, so frozen instances cannot
  retain caller-owned mutable lists.
- Added a regression test covering list normalization, source-list mutation,
  tuple immutability, and exact JSON round trips.
- Added a named Task 3 JSONL fixture and integration-style assertion that the
  complete Task 3 metadata, shape, timing, and histogram field set is accepted.
- RED: the new regression test failed because direct construction stored the
  caller-provided list.
- GREEN: `PYTHONPATH="$PWD" .venv/bin/pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py`
  passed with 24 tests after the fix.

## Environment Note

The requested `uv run pytest ...` invocation could not resolve the repository
workspace because `nemo-gym` is declared as a workspace source but is not a
workspace member. Verification therefore used the worktree `.venv` directly;
`PYTHONPATH="$PWD"` exposes the uninstalled `experiments` namespace.
