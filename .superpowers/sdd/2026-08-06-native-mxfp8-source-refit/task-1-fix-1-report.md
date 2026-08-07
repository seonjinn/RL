# Task 1 Fix 1 Report: Native MXFP8 Source Validation

## Status

Implemented the round 1 review fixes for the canonical native MXFP8 storage adapter.
The implementation API is unchanged:

```python
extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components
```

Plan files were not modified. Existing unrelated untracked `docs/superpowers/`
content was left untouched.

## Changes

- `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
  - Added concise Google-style docstrings to `NativeMXFP8Components` and
    `extract_native_mxfp8_components`.
  - Converts absent, `None`, and non-tensor `rowwise_data` or
    `rowwise_scale_inv` metadata into component-specific `ValueError`s.
  - Retains `uint8` dtype validation for both native components.
  - Validates exact weight-byte storage before reinterpretation.
  - Validates that `rowwise_scale_inv` is 2-D and has at least the flattened
    logical scale shape `(prod(shape[:-1]), K / 32)` before cropping.
  - Preserves compact scale cropping and the byte-preserving E4M3 view.

- `tests/unit/models/policy/test_mxfp8_refit_source.py`
  - Added missing-component and non-tensor validation coverage for both native
    metadata keys.
  - Added undersized weight and scale coverage, including non-2-D scale input.
  - Added exact byte-preservation assertions for E4M3 weights.
  - Added row and column crop value assertions for a padded 2-D source.
  - Added the same value-preservation and crop assertions for a 3-D grouped
    expert-shaped source.

## TDD And Test Results

Tests were extended before the production validation changes. The required
project-level command was run first:

```text
uv run pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py
```

It exited 1 during project setup because this checkout's `pyproject.toml`
references `nemo-gym` as a workspace source without that workspace member:

```text
Failed to parse entry: `nemo-gym`
`nemo-gym` references a workspace in `tool.uv.sources` ... but is not a workspace member
```

The dependency-light pytest attempt also stopped during collection because the
normal package import graph requires unavailable project dependencies (`transformers`
and `ray`). A small in-process harness loaded the owned module directly and
ran the actual focused test file while bypassing unrelated package initializers:

```text
uv run --no-project --with torch --with pytest python <isolated pytest harness>
```

Result:

```text
14 passed in 0.02s
```

## Static And Runtime Checks

All checks below passed:

```text
uv run --no-project --with ruff ruff format --check \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py \
  tests/unit/models/policy/test_mxfp8_refit_source.py
2 files already formatted

uv run --no-project --with ruff ruff check \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py \
  tests/unit/models/policy/test_mxfp8_refit_source.py
All checks passed!

uv run --no-project python -m py_compile \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py \
  tests/unit/models/policy/test_mxfp8_refit_source.py
exit 0

git diff --check
exit 0
```

The direct dependency-light PyTorch smoke used a 3-D `(2, 3, 64)` source,
checked the E4M3 byte reinterpretation, and checked the exact leading `(6, 2)`
scale crop reshaped to `(2, 3, 2)`:

```text
native MXFP8 adapter runtime smoke: PASS
```

An optional `pyrefly` invocation was also attempted, but it reported missing
`torch` and `pytest` imports because it inspected the checkout's existing
project environment rather than the temporary `uv --with` environment. No
full NeMo-RL dependency installation was attempted.

## Concerns

- The normal project-level pytest command remains blocked by the repository's
  existing `nemo-gym` workspace resolution issue.
- Full package pytest collection remains unavailable in this macOS worktree
  without the NeMo-RL dependency graph; focused behavior was verified through
  the isolated PyTorch/pytest harness instead.
- `pyrefly` remains unavailable as a meaningful project check until the
  checkout environment includes its configured runtime imports.
