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

- Commit hash: `91fe2362949369298b53e06d1dc365e60d1102fc`
- Note: this report line is intentionally left outside the signed-off commit so
  it can record the final hash without changing it again.

## Review Fix Cycle

### Review Scope

Addressed all requested follow-up items:

- consume `rollout_batch_index` so rollout batches do not repeat seed ranges
- assert the full `8:4:3:1` prompt allocation, not partial slices
- reject invalid JSON types instead of coercing with `bool()` and `int()`
- strengthen the hash-stability test to reorder once and use `tmp_path`
- correct `swe_sync_64k.json` to `4K/8K/16K/64K` with weights `8/4/3/1`
- change `summarize_barrier_tail().tail_gap_s` to `max - median`

### Review RED

Added the failing regression coverage first in `tests/test_vllm024_dynamicsd.py`.

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k "request_plan or barrier_tail or swe_sync_64k_profile"
```

Output summary:

- `7` tests failed
- failures matched the reviewed defects: repeated seed ranges, permissive JSON
  coercion, incorrect 64k profile values, and `tail_gap_s` using `max - min`

Representative output:

```text
FAILED tests/test_vllm024_dynamicsd.py::test_resolve_request_plan_uses_rollout_batch_index_for_unique_seed_ranges
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_json_types[ignore_eos-true-ignore_eos must be a boolean]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_json_types[max_tokens-4096.5-max_tokens must be an integer]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_json_types[weight-8.5-weight must be an integer]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_json_types[max_model_len-36864.5-max_model_len must be an integer]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_reads_expected_swe_sync_64k_profile
FAILED tests/test_vllm024_dynamicsd.py::test_summarize_barrier_tail_uses_max_minus_median
7 failed, 3 passed, 41 deselected in 0.19s
```

### Review Implementation

Made the smallest production changes required by the failing tests:

- derived the first seed as
  `seed_start + rollout_batch_index * (len(prompt_ids) * samples_per_prompt)`
- kept the request-plan work allocation unchanged while shifting only seeds
- introduced strict JSON field validators for integer and boolean fields
- corrected `swe_sync_64k.json` to `4096/8192/16384/65536`
- changed `tail_gap_s` to `max_s - median_s`

### Review GREEN

Focused command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k "request_plan or barrier_tail or swe_sync_64k_profile"
```

Focused output summary:

```text
10 passed, 41 deselected in 0.05s
```

Full command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Full output summary:

```text
51 passed in 2.86s
```

### Review Files Changed

- `experiments/vllm_024_dynamicsd/sync_rollout_core.py`
- `experiments/vllm_024_dynamicsd/profiles/swe_sync_64k.json`
- `tests/test_vllm024_dynamicsd.py`
- `.superpowers/sdd/task-1-report.md`

### Review Self-Check

- The stronger allocation test now checks both prompt counts (`8/4/3/1`) and
  expanded request counts (`32/16/12/4`).
- The reordered-hash test now uses `tmp_path` and actually writes a reversed
  bucket order once before loading.
- The JSON-type regression tests cover the exact reviewed failure modes:
  string booleans and fractional integers.

## Name Validation Follow-Up

### Follow-Up Scope

Addressed the remaining warning that `RequestPlan.name` was not strictly
validated at runtime.

### Follow-Up RED

Added a failing regression for invalid `name` values in
`tests/test_vllm024_dynamicsd.py`.

Command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k invalid_name_values
```

Output summary:

- `4` tests failed
- failures matched the missing runtime checks for numeric, list, object, and
  empty-string `name` values

Representative output:

```text
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_name_values[123-TypeError-name must be a string]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_name_values[value1-TypeError-name must be a string]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_name_values[value2-TypeError-name must be a string]
FAILED tests/test_vllm024_dynamicsd.py::test_load_request_plan_rejects_invalid_name_values[-ValueError-name must be a non-empty string]
4 failed, 51 deselected in 0.17s
```

### Follow-Up Implementation

Made the minimal production change in
`experiments/vllm_024_dynamicsd/sync_rollout_core.py`:

- added `_require_non_empty_string()`
- validated `name` through the canonical payload path so loading and hashing
  share the same rule

### Follow-Up GREEN

Focused command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py -k invalid_name_values
```

Focused output summary:

```text
4 passed, 51 deselected in 0.05s
```

Full command:

```bash
python3 -m pytest -q tests/test_vllm024_dynamicsd.py
```

Full output summary:

```text
55 passed in 5.01s
```
