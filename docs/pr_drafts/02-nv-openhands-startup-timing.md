# [SWE] Add OpenHands startup phase metrics

## Summary

Record the main SWE startup phases in the existing evaluation metrics output.

## Why

SWE rollout startup includes framework import, runtime connection, workspace initialization, generation, and final evaluation. A single total time does not show which phase changed.

## High-level implementation

- Use monotonic timers around runtime creation, connection, and initialization.
- Record framework startup from the launcher timestamp to the first runtime phase.
- Write numeric fields to the existing metrics file.
- Do not change rollout commands or retry behavior.

## Performance impact

This PR is for observability and has no intended speedup. The acceptance limit is at most 0.5% median and 1% p95 rollout overhead.

## Validation

- Verify all fields are present on success and partial failure.
- Verify every duration is non-negative.
- Compare enabled and disabled runs for output parity and timing overhead.
