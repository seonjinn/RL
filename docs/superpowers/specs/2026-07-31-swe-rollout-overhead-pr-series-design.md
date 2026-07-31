# SWE Rollout Overhead PR Series Design

## Goal

Split the SWE rollout startup work into four small pull requests with clear ownership, simple behavior, and independent validation.

## Scope

The measurements were collected on Lyris GB200 with vLLM 0.25.1, Qwen3-30B-A3B-Thinking-2507, Gym main at `fbde772`, and nv-OpenHands at `sdd/multilingual-fixes@5f018005`.

The vLLM 0.25.1 upgrade improved generation throughput, but it did not remove the CPU and filesystem startup costs in OpenHands. Before the node-local patch, Connect, Initialize, and Framework took 15.0, 22.1, and 21.8 seconds per rollout. The node-local patch reduced Connect to 11.1 seconds and Framework to 16.4 seconds. Initialize stayed at 21.8 seconds because the workspace was still copied for every rollout.

## PR boundaries

1. Gym stages OpenHands and miniforge on node-local storage.
2. nv-OpenHands records startup phase metrics without changing behavior.
3. nv-OpenHands accepts an immutable workspace cache and creates a private writable workspace.
4. Gym prepares and mounts the immutable per-instance cache used by PR 3.

PRs 1 and 2 are independent. PR 3 can be tested with a manually mounted cache. PR 4 depends on PR 3 and provides automatic Gym integration.

## Performance claims

- PR 1 has measured phase-level results: Connect improves by 26.0%, Framework by 24.8%, and their combined time falls by 9.3 seconds per rollout.
- PR 2 is an observability change. It must not add more than 0.5% median or 1% p95 rollout overhead.
- PRs 3 and 4 share one projected performance target. Warm Initialize should fall from 21.8 seconds to at most 5 seconds. The target is at least 8% lower n=80 allocation-to-result wall time.
- The 24-rollout aggregate excludes the 216-second node-local staging cost. Its modeled break-even is about 24 rollouts.
- A historical n=80 run reported 7.7% lower job wall time, but its raw logs must be recovered or reproduced before the result is used as validated PR evidence.

## Safety rules

- Keep all optimizations opt-in until full-job validation passes.
- Never share writable hard-linked workspace files between rollouts.
- Key caches by source image identity, instance ID, and base commit.
- Populate caches under a lock and publish them with an atomic rename.
- Mount the immutable cache at a private path, not over `/workspace`.
- Preserve the existing copy path as the fallback.

## Validation

Use paired ABBA runs with at least four n=24 pairs and four n=80 pairs. Record setup, allocation-to-result wall time, phase timings, failures, drain time, generation throughput, reward, and valid rollout rate. Require no workspace leakage, no new failure class, no more than a one percentage point valid-rate loss, and no more than a 2% generation-throughput loss.

## Draft PR bodies

- `docs/pr_drafts/01-gym-node-local-openhands-staging.md`
- `docs/pr_drafts/02-nv-openhands-startup-timing.md`
- `docs/pr_drafts/03-nv-openhands-immutable-workspace-cache.md`
- `docs/pr_drafts/04-gym-workspace-cache-integration.md`
