# Task 2 Report: Tail-Gated Scheduler and NeMo Validation

## Status

Implemented on `sna/nemorl-vllm024-tail-gating`.

## Changes

- Added `TailGatedScheduler`, a vLLM `Scheduler` subclass that gates only the
  next proposal K, preserves already scheduled draft tokens, forwards
  `throttle_prefills`, and emits patched scheduler-output telemetry fields.
- Recorded accepted drafts before delegating to the upstream scheduler and
  reset the rollout latch only after `get_num_unfinished_requests()` reports
  zero, including skipped waiting requests.
- Added pure NeMo validation for active tail gates. It requires an external
  Eagle/Eagle-3 drafter, a positive static K, supported mode and positive
  threshold/check count, no stock DynamicSD schedule or internal vLLM data
  parallelism, and a roofline configuration path when required.
- Registered the required scheduler class without changing target or draft
  sampling settings.

## TDD Record

- RED: `uv run --no-sync pytest tests/unit/models/generation/test_vllm_tail_gate_scheduler.py tests/unit/models/generation/test_vllm_generation.py -q`
  exited `1` before implementation because
  `nemo_rl.models.generation.vllm.tail_gate_scheduler` did not exist.
- GREEN: focused tail-gate tests passed after implementation.

## Verification

- `uv run --no-sync pytest -o addopts='' --noconftest tests/unit/models/generation/test_vllm_tail_gate_scheduler.py tests/unit/models/generation/test_vllm_generation.py -k tail_gate -q`
  passed: `11 passed`.
- `uv run --no-sync ruff check nemo_rl/models/generation/__init__.py nemo_rl/models/generation/vllm/tail_gate_scheduler.py tests/unit/models/generation/test_vllm_tail_gate_scheduler.py tests/unit/models/generation/test_vllm_generation.py`
  passed.
- `git diff --check` passed.
- The exact requested full command was attempted. It entered the generation
  suite but did not complete because a Ray worker created a fresh `.venv` and
  failed with `ModuleNotFoundError: No module named 'ray'`.

## Concern

`tail_gate_scheduler.py` is not added to the existing `pyrefly.toml` explicit
include list. Updating that file is outside the Task 2 ownership boundary.
