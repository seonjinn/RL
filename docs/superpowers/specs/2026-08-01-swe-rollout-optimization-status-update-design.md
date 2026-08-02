# SWE Rollout Optimization Status Update Design

## Goal

Make the existing NeMo-Gym initialization and framework optimization report the
canonical execution-status page for the remaining SWE rollout overhead work.
Keep the PR draft page focused on proposed upstream changes and avoid presenting
code-verification jobs as rollout performance evidence.

## Canonical ownership

- `docs/nemogym_init_framework_fixes.html` owns the human-readable current
  status, next experiment, optimization queue, and promotion gates.
- `public/reports/nemogym_init_framework_fixes.html` is a byte-identical public
  mirror of the canonical report.
- `experiments/swe_rollout_latest_main_ab/attempts.md` owns the dated operational
  history, including environment failures and live SLURM observations.
- `docs/swe_rollout_pr_drafts.html` continues to own PR scope, provenance, and
  upstream-overlap decisions. It is not an experiment-status dashboard.
- `public/index.html` is out of scope because it has unrelated uncommitted
  changes and already links to the optimization report.

## Status section

Add a dated `Current execution status` section near the top of the optimization
report. It contains one compact table with these rows:

1. Frozen latest-main source arm A at `1afc767c`: source prepared; rollout
   performance pending.
2. Arm B, A plus NeMo-RL PR #3390 at `41374086`: source prepared; rollout
   performance pending.
3. Arm C, B plus NeMo-RL PR #3283 at `6f8ca0b6`: Linux/GPU correctness gate
   passed in OCI-HSG job `5755875`; rollout performance pending.
4. Arm D, C plus node-local OpenHands staging: historical vLLM 0.25.1 evidence
   exists, but the refreshed latest-main arm and rollout gate are pending.
5. Progressive candidates: not started and not running.

The section must state explicitly that the 2026-07-31 job was a code and CUDA
Graph validation job, not a SWE rollout benchmark. Therefore it has no dataset,
trajectory-duration, token-count, ReplayBuffer, throughput, or phase-timing
result. It must also record that the 2026-08-01 OCI-HSG live check found no
active `swe-ab`, SWE rollout, or NeMoGym performance job.

## One-variable experiment sequence

The optimization report records the enforced progression:

1. Complete the reproducible launcher, provenance manifest, result parser, and
   parser tests.
2. Run A/B/C/D with one prompt, one generation, and concurrency one.
3. Run matched n=24 ABBA comparisons for only the arms that pass correctness.
4. Include one-time staging, allocation-to-result wall time, failures, and drain
   in the primary result. Report summed rollout phases separately.
5. Promote a meaningful and correct result to matched n=80 reproduction.
6. Test each progressive candidate through static/unit, n=1, n=24, and
   conditional n=80 gates without combining candidates.

The candidate order remains: prebuilt runtime artifact, private copy-on-write
workspace, pre-imported forkserver, one-use prewarmed action servers, trajectory
payload compaction, and nv-OpenHands episode affinity.

## Evidence and labeling rules

- `Completed` means a named validation gate has passed with a job ID or committed
  artifact.
- `Pending` means no result exists for that gate.
- `Historical measured` remains separate from refreshed latest-main evidence.
- Environment, mount, dependency, or harness failures are operational attempts,
  not performance regressions.
- Projected savings must never be shown as measured results.
- Do not add speedups across unmatched runs or across different sample counts.

## Attempts log update

Append a dated entry to `attempts.md` without rewriting its existing uncommitted
history. Record:

- the exact `sacct` state for job `5755875`;
- the bounded `squeue` observation that no matching performance job was active;
- the distinction between the passed correctness gate and the unsubmitted
  rollout canaries;
- the next required artifact and gate.

Do not copy full remote logs into the repository.

## Validation

- Parse the canonical and public HTML with Python's `HTMLParser`.
- Require the new section ID, all four arm labels, job `5755875`, and the
  one-variable progression text.
- Resolve every local link in the two optimization pages.
- Require the canonical and public optimization pages to be byte-identical.
- Scan the touched HTML and Markdown for credentials, private keys, and
  user-specific local paths.
- Verify the implementation changes only the canonical report, its public
  mirror, and the append-only attempts log.

## Non-goals

- Do not submit GPU jobs in this documentation update.
- Do not claim a refreshed node-local speedup.
- Do not change the PR draft page, Pages landing page, CI configuration, or
  broader efficiency report.
- Do not implement the launcher, parser, workspace cache, forkserver, or server
  pool in this change.
