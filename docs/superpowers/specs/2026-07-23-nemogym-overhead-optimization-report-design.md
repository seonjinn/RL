# NeMo-Gym SWE Overhead Optimization Report Design

**Goal:** Correct the existing SWE rollout overhead reports and document code-grounded, experimentally gated optimizations for the remaining initialization and framework costs.

## Scope

Update these existing pages:

- `docs/nemogym_init_framework_fixes.html`
- `docs/nemogym_swe_efficiency_report.html`

The work is documentation and analysis only. It does not change Gym, OpenHands, launcher, or runtime behavior.

## Evidence Model

Every performance statement will use one of three labels:

- **Measured:** backed by a committed raw artifact in this worktree.
- **Handoff-reported:** recorded in `HANDOFF_CODEX.md` but missing a committed raw artifact.
- **Projected:** derived from a measured ledger or an unvalidated code-path hypothesis.

One-time setup costs must be included in full-job comparisons. Phase-normalized comparisons may exclude stochastic LLM and evaluation differences, but must be labeled as cost models rather than observed job-wall results.

## Corrected Ledger

The n=24 paired data in `experiments/dflash_loss_ab/report/data/patch_ab_n24.csv` is the canonical committed source.

| Phase, seconds per rollout | Unpatched | Patched | Delta |
|---|---:|---:|---:|
| Connect/action server | 15.0 | 11.1 | -3.9 |
| Initialize/workspace copy | 22.1 | 21.8 | -0.3 |
| Framework/run_infer | 21.8 | 16.4 | -5.4 |
| LLM | 49.2 | 53.8 | +4.6 |
| Evaluation | 19.3 | 20.5 | +1.2 |
| Phase sum | 127.4 | 123.6 | -3.8 |

Observed n=24 rollout wall excluding mirror preparation is 3,059 to 2,968 seconds. Including the 216-second mirror preparation cost changes the patched total to 3,184 seconds, a 4.09% regression. The approximately 24-rollout break-even is a phase-normalized model based on 9.3 seconds of stable connect plus framework savings, not an observed break-even.

The n=80 claim of 948 seconds or 7.7% net job-wall savings will be retained only as handoff-reported, with a note that job IDs and raw CSV are pending.

## Root Causes

### Workspace initialization

`instance_swe_entry.sh` attempts:

```bash
if ! cp -al /testbed /workspace/$WORKSPACE_NAME 2>/dev/null; then
    cp -r /testbed /workspace/$WORKSPACE_NAME
fi
```

Hard links cannot cross from the read-only instance squashfs to the writable workspace filesystem, so every rollout takes the full-copy fallback. `NRL_SKIP_GIT_RESET=1` changed initialization from 20.1 to 19.9 seconds, disproving the earlier git-cleanup diagnosis.

### Framework startup

Every rollout launches `run_infer.sh`, which starts a new Python process for `run_infer.py`. The process eagerly imports the OpenHands evaluation, agent, controller, configuration, and runtime stack before it creates the runtime. The remaining framework phase is therefore a once-per-rollout process and import cold start, not inter-turn overhead.

### Action-server connection

`LocalRuntime` starts a fresh `python -u -m openhands.runtime.action_execution_server` process. That process imports its stack, constructs `ActionExecutor`, initializes bash, plugins, and environment state, and is terminated at rollout completion. Existing warm-server globals are process-local and disappear when the one-shot `run_infer.py` process exits.

## Optimization Candidates

### 1. Prebuilt OpenHands squashfs

**Status:** High-confidence, low-risk candidate.

Use the existing `NRL_OH_SQUASHFS` branch to replace the 216-second small-file mirror with one large sequential read and local extraction. Bake rewritten container paths and checked-hash bytecode into a versioned artifact.

**Why it should help:** Lustre handles one large file substantially better than thousands of metadata-heavy Python and venv files.

**Acceptance gate:** setup at most 36 seconds, at least 180 seconds below the current setup; byte-level manifest parity; no change to rollout phases or valid-rollout rate.

### 2. Per-instance immutable cache plus private reflink workspace

**Status:** High-confidence design, filesystem support unverified.

Populate an immutable node-local cache keyed by instance-image digest, instance ID, and base commit. Create a private rollout workspace with `cp --reflink=always -a` when supported. Use per-key locking, temporary directories, atomic rename, and a manifest before publishing `.ready`.

**Why it should help:** the first rollout pays extraction; later rollouts create copy-on-write metadata clones rather than copying the repository contents.

**Correctness requirement:** never use writable hard-linked clones. A plain `cp -al` cache would allow one rollout to mutate shared inodes and contaminate other rollouts.

**Acceptance gate:** warm initialization at most 5 seconds; n=80 full-wall improvement at least 8%; clean base commit, no tracked/untracked file leakage, and identical evaluation output for every rollout.

### 3. Pre-imported forkserver

**Status:** Medium-risk candidate requiring instrumentation first.

Create separate pre-imported forkservers for the run-infer harness and action server. Start the forkserver after module imports but before controller, runtime, HTTP client, thread, logger-handler, or rollout state construction. Fork a fresh child for each rollout.

**Why it should help:** Python module import execution and filesystem lookups are paid once while per-rollout process isolation remains.

**Acceptance gate:** framework improves by at least 3 seconds and full wall by at least 2%; deterministic event ordering and output parity; no inherited thread, socket, logger, or environment state.

### 4. Persistent controller with one-use prewarmed servers

**Status:** Higher-risk follow-up.

Keep a controller alive across rollouts, prewarm action servers, lease each server to exactly one rollout, and destroy it afterward. Do not initially recycle a used `ActionExecutor`.

**Why it should help:** server import and boot overlap with preceding work without reusing stateful executors.

**Acceptance gate:** connect p50 at most 5 seconds, p95 at least 30% better, and sentinel tests prove no process, filesystem, environment, secret, tmux, plugin, todo, or download state crosses rollout boundaries.

### 5. Small isolated candidates

- Invoke the venv Python directly instead of `poetry run`; expected benefit is below one second and must be measured alone.
- Precompile checked-hash `.pyc` files in the versioned squashfs; test independently from delivery changes.
- Treat asynchronous final evaluation as a separate throughput project because delayed rewards and trajectory age can affect RL semantics.

## Failed Attempts and Retracted Hypotheses

The HTML will include a table with symptom, evidence, root cause, and retry condition for each item:

- `NRL_SKIP_GIT_RESET=1`: removed resets but saved approximately 0.2 seconds; workspace copy is the real initialization cost.
- Merging initialization commands from seven to four: no measurable gain; initialization is work-bound rather than round-trip-bound.
- Polling every 0.5 seconds instead of 2 seconds: no measurable gain; action-server import and boot dominate.
- Moving the event store to `/tmp`: no measurable gain; measured inter-turn framework overhead is already near zero.
- Direct `NRL_WS_CACHE` binds over `/workspace`: invalid experiment because the bind collided with OpenHands workspace ownership and produced non-git workspaces.
- Writable hardlink cache: rejected by design because it violates rollout isolation.
- Full action-server reuse: deferred until an explicit reset protocol is proven.

## Validation Protocol

Use an ABBA job-level crossover with at least four independent pairs at n=24 and four at n=80. Include allocation-to-result wall time, setup, failed rollouts, and drain time.

Run two layers:

1. Deterministic replay for systems-cost isolation.
2. Temperature-1 live rollout for operational, reward, and completion validation.

Common gates:

- One-sided 95% paired-bootstrap confidence interval shows at least 3% lower full job wall including setup.
- Valid-rollout-rate lower confidence bound is no worse than -1 percentage point.
- Generation throughput lower confidence bound is no worse than -2%.
- Reward lower confidence bound is no worse than -0.01 absolute.
- No new timeout, OOM, Ray, stale-cache, workspace-isolation, or process-leak failure class.

## Page Structure

`nemogym_init_framework_fixes.html` will become the detailed canonical page:

1. Evidence status and provenance.
2. Corrected observed full-job ledger.
3. Root-cause code paths.
4. Ranked optimization candidates.
5. Validation matrix.
6. Failed attempts and retracted hypotheses.
7. Remaining measured and theoretical ceiling.

`nemogym_swe_efficiency_report.html` will be corrected and kept as the broader study:

1. Replace the git-cleanup diagnosis with workspace-copy evidence.
2. Separate measured, handoff-reported, and projected numbers.
3. Replace the obsolete 145-to-70-second projection with the latest 123.6-second ledger, 78.6-second realistic bound, and 74.3-second zero-overhead bound.
4. Link to the detailed fixes page for implementation analysis and validation gates.

## Remaining Ceiling

From the n=24 patched ledger, Connect + Initialize + Framework is 49.3 seconds of a 123.6-second phase total. Removing the identified approximately 45 seconds gives a projected 78.6 seconds, a 36.4% wall reduction or 1.57x throughput ceiling. Removing all 49.3 seconds is the absolute 74.3-second, 1.66x bound. Neither is a measured result.
