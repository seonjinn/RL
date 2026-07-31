# [SWE] Stage and mount immutable workspace caches

## Summary

Prepare an immutable per-instance workspace cache in Gym and mount it for the OpenHands cache path added by the companion PR.

## Why

The OpenHands cache protocol needs a safe host-side cache owner. Gym already knows the source image, instance ID, base commit, and container mounts.

## High-level implementation

- Key each cache by source image identity, instance ID, and base commit.
- Populate it under a lock in a temporary directory.
- Verify a manifest, then publish with an atomic rename.
- Mount the cache read-only at a private container path.
- Keep `/workspace` private and writable for every rollout.

## Performance impact

This PR enables the performance target of the companion OpenHands PR; the gain is not counted twice.

- Warm Initialize target: 21.8 s to at most 5 s.
- Phase-normalized target: 123.6 s to at most 106.8 s, about 13.6% lower.
- Full-job gate: at least 8% lower n=80 allocation-to-result wall time.

These values are projected until paired full-job runs complete.

## Validation

- Run cold-cache and warm-cache cases.
- Verify locking, atomic publication, stale-cache rejection, and fallback.
- Require identical patches and rewards, no file leakage, and no new failure class.

## Dependency

Depends on the nv-OpenHands immutable workspace-cache PR.
