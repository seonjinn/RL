# [SWE] Support an immutable workspace cache

## Summary

Allow the SWE entry script to create a private writable workspace from an optional immutable cache.

## Why

The current hard-link copy crosses filesystem boundaries and fails. Every rollout then copies the full `/testbed` tree into `/workspace`, which takes about 22 seconds.

## High-level implementation

- Accept an optional read-only cache path.
- Validate its instance ID, source image identity, and base commit.
- Create a private workspace with reflink copy when supported.
- Fall back to the current full copy when the cache is absent or invalid.
- Never expose writable shared inodes to multiple rollouts.

## Performance impact

This change needs a caller-provided cache mount. The warm initialization target is 21.8 s to at most 5 s, saving at least 16.8 s per rollout. This is a target, not a measured result.

## Validation

- Run two concurrent rollouts and prove that file changes do not leak.
- Verify the base commit, generated patch, reward, and trajectory match the current path.
- Test cache miss, stale cache, unsupported reflink, and interrupted population cases.
