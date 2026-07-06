# Task 1 Report: Request-Plan Core

## Scope

- Created `experiments/vllm_024_dynamicsd/sync_rollout_core.py`
- Created `experiments/vllm_024_dynamicsd/profiles/swe_sync_32k.json`
- Created `experiments/vllm_024_dynamicsd/profiles/swe_sync_64k.json`
- Modified `tests/test_vllm024_dynamicsd.py`
- Updated `.superpowers/sdd/task-1-report.md`

## Requirements Source

Command:

```bash
sed -n '1,120p' /Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/vllm024-dynamicsd/.superpowers/sdd/task-1-brief.md
```

Summary:

- Implement `LengthBucket`, `RequestPlan`, `ResolvedRequest`,
  `load_request_plan()`, `resolve_request_plan()`,
  `validate_context_window()`, and `summarize_barrier_tail()`.
- Add the `swe_sync_32k.json` and `swe_sync_64k.json` profiles.
- Use strict TDD with a focused RED run first, then focused and full GREEN runs.

## Red Step

Added focused request-plan tests first in `tests/test_vllm024_dynamicsd.py` for:

- deterministic 8:4:3:1 allocation across 4k/8k/16k/32k buckets
- exact planned token budget of `589824`
- stable plan hashing
- explicit context-overflow errors

The brief suggested `pytest -q tests/test_vllm024_dynamicsd.py -k request_plan`,
but the local `pytest` command in this environment is a wrapper that reported
`Pytest: No tests collected`. I used the real runner directly.

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k request_plan
```

Output summary:

- `3` tests failed
- failure mode was expected: `FileNotFoundError` while loading
  `experiments/vllm_024_dynamicsd/sync_rollout_core.py`

Representative output:

```text
FAILED tests/test_vllm024_dynamicsd.py::test_resolve_request_plan_is_deterministic_and_exact
FAILED tests/test_vllm024_dynamicsd.py::test_request_plan_hash_is_stable_for_equivalent_json
FAILED tests/test_vllm024_dynamicsd.py::test_request_plan_validates_context_overflow
3 failed, 41 deselected in 0.04s
```

## Implementation

Implemented the minimal production code required by the tests:

- frozen dataclasses for `LengthBucket`, `RequestPlan`, and `ResolvedRequest`
- canonical JSON hashing with normalized bucket ordering and sorted keys
- deterministic weighted prompt allocation with stable remainder handling
- exact fixed-length request expansion with per-sample seeds
- explicit context-window validation
- two fixed-length SWE request-plan profiles for `32k` and `64k`
- a minimal `summarize_barrier_tail()` helper for later rollout summaries

One integration fix was needed after the first GREEN attempt:

- Python 3.14 dataclasses interacted badly with postponed annotations under the
  repo's custom import loader, so the new module does not use
  `from __future__ import annotations`.

## Green Step

Focused command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k request_plan
```

Focused output summary:

```text
3 passed, 41 deselected in 0.05s
```

Full command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Full output summary:

```text
44 passed in 4.13s
```

## Self-Review

Commands:

```bash
git diff --check -- tests/test_vllm024_dynamicsd.py experiments/vllm_024_dynamicsd/sync_rollout_core.py experiments/vllm_024_dynamicsd/profiles/swe_sync_32k.json experiments/vllm_024_dynamicsd/profiles/swe_sync_64k.json
git status --short
```

Output summary:

- `git diff --check` produced no output
- only the Task 1 files plus this report are staged for commit

Review notes:

- The new tests prove the exact 8:4:3:1 prompt split and total planned tokens.
- The production code is intentionally narrow and does not invent rollout
  behavior beyond the Task 1 brief.
- `rollout_batch_index` is accepted for the required interface but is not used
  yet because the brief does not define batch-dependent behavior.

## Concerns

- The bare `pytest` command in this environment is not the real pytest runner.
  Future tasks in this worktree should use `python3 -m pytest` unless that
  wrapper is fixed.
- `summarize_barrier_tail()` is present with a minimal shape, but later tasks
  may need to lock down its exact output schema once the downstream caller is
  defined.

## Commit

- Commit hash: `af07475db506cd335a9a12036285178094d374e1`
