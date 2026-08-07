# Task 2 Fix 1 Report

## Status

Implemented both review fixes in the requested worktree.

## Fixes

- `weight_scale` components now accept only the canonical serialized aliases
  `torch.uint8` and `uint8`; supported-but-wrong dtypes such as
  `torch.bfloat16` are rejected.
- Pre-grouped `experts.gate_up_proj` metadata now requires a 3-D
  `[E, 2*intermediate, hidden]` shape with an even fused dimension before
  splitting logical or component shapes. Component leading dimensions are
  validated against the fused layout as well.

## TDD Evidence

Added focused regression tests for the supported-but-wrong scale dtype and odd
fused gate/up dimension. Each test failed with `DID NOT RAISE` before the
production fix and passed afterward.

## Verification

- Isolated harness: `65 passed`
- Ruff check: passed
- Ruff format check: passed
- Python compile check: passed
- `git diff --check`: passed

The normal project pytest environment was unavailable because the worktree
environment lacks `ray` and `torch`; no full `uv sync` was run. The passing
isolated harness loaded the real owned module and test file with temporary CPU
dependencies and lightweight package stubs.

## Concerns

- Full project pytest remains unverified locally due to missing project
  dependencies outside this task's scope.
