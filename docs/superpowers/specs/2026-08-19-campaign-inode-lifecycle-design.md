# Campaign Inode Lifecycle Design

## Objective

Bound the number of shared-filesystem entries created by CUDA Graph campaign
submission and result collection without weakening source provenance, scheduler
preflight, distributed parity evidence, or runtime attestation.

The immediate target is the typed MCore and Bridge matrix workflow under
`experiments/cuda_graph/nemotron_thd_te_graph_20260731`. The same lifecycle
rules apply to later correctness, smoke, performance, and soak launchers that
reuse these artifacts.

## Current Problem

The typed submitters publish a fresh UUID-named source archive and immutable
intent before deciding whether the operation is a scheduler dry-run or an
actual submission. A single MCore archive contains 3,752 filesystem entries.
The standard `TEST_ONLY -> SBATCH_TEST_ONLY -> actual` workflow therefore
leaves two archives and two intents, or 7,506 entries, before the job starts.

Distributed parity also retains every collection, node, and final rank JSON.
A successful 16-rank R3 row leaves 69 rank-exchange entries; a failed run leaves
a variable partial tree. Neither source submissions nor rank exchange has a
retention policy.

Source-bound runtime stages are intentionally larger. Each new root source SHA
creates a recursive source snapshot and two Python environments. These are not
per-job leaks, but repeated fix-and-restage cycles accumulate until the inode
quota becomes the campaign's limiting resource.

The observed campaign reached 24,972,129 of 26,214,400 allowed files (95.26%)
while using only 30.97% of its byte quota. A later cleanup deleted the entire
campaign root, including the current valid runtime stage and attestation. The
replacement design must prevent both unbounded accumulation and unsafe global
deletion.

## Non-Goals

- Do not weaken exact source SHA, directory digest, intent, profile, runtime,
  container, or attestation verification.
- Do not remove failed-run evidence without first publishing a bounded,
  immutable replacement.
- Do not convert the staged Python environments to SquashFS in this change.
- Do not automatically delete an active job's files or any artifact referenced
  by the current profile.
- Do not infer liveness only from timestamps.
- Do not make login-node scans proportional to the entire campaign tree on
  every submission.
- Do not fix the separate CUDA-capture-safe route-value validation bug.

## Selected Architecture

Use four cooperating mechanisms:

1. a content-addressed candidate snapshot cache;
2. mode-specific transactional submission artifacts;
3. bounded distributed-rank evidence finalization; and
4. an allowlist-driven retention tool with an inode admission gate.

The submitter and retention tool share typed artifact metadata. No cleanup path
may operate on a campaign directory that lacks the expected ownership marker
and schema version.

## Artifact Classes

### Candidate snapshot

A candidate snapshot is immutable source content shared by submissions. Its
identity is:

```text
(candidate_kind, candidate_commit, snapshot_sha256)
```

The final path is deterministic:

```text
RUN_LOG_ROOT/source-snapshots/<kind>/<commit>/<snapshot_sha256>/
```

The directory contains the archived source, `.candidate-sha`, and
`.snapshot-sha256`. Values are verified before reuse. Source content is never
selected by path alone.

Publication uses a temporary sibling directory followed by same-filesystem
atomic no-clobber publication. If another process wins the race, the loser
deletes its temporary tree and verifies the winner byte-for-byte through the
existing snapshot verifier. A mismatched, writable, symlinked, partial, or
markerless destination fails closed.

Snapshot creation returns whether the caller created the cache entry. This is
used only for transactional rollback before a scheduler accepts a real job. A
shared verified snapshot is never removed merely because one submission fails.

### Submission intent

Every actual scheduler submission has one small immutable intent. The intent
binds the row, profile, runtime feature, candidate snapshot path and digest,
and source commits.

`TEST_ONLY` and `SBATCH_TEST_ONLY` use a temporary artifact root. The scheduler
dry-run sees the same rendered source and intent layout as an actual submit,
but the entire temporary root is removed on every exit path. Neither dry-run
may leave entries under the durable `RUN_LOG_ROOT`.

An actual submit writes its intent under:

```text
RUN_LOG_ROOT/submission-intents/<kind>/<commit>/<submission_id>.json
```

If `sbatch` rejects the request, the submitter removes the intent it created.
It also removes a newly created snapshot only when the snapshot is still
unreferenced, unchanged, and owned by that failed transaction. Failure to
prove those conditions preserves the snapshot and reports a bounded orphan
for later retention; it never performs an unsafe delete.

### Rank exchange and evidence bundle

Ranks continue to exchange JSON files on the shared filesystem because the
distributed test spans nodes. The exchange directory is explicitly transient
and bound to `(scheduler_job_id, restart_count, intent_sha256)`.

After the distributed command returns, the outer batch process finalizes the
tree into one immutable evidence bundle. The bundle is a deterministic tar
archive containing sorted paths and normalized metadata, plus a small JSON
index with:

- run identity and job ID;
- candidate, row, intent, and profile digests;
- expected and observed rank files;
- per-node exit status;
- completeness status; and
- archive SHA256.

Success and failure both produce a bundle. Missing-rank failures record the
missing set in the index. After the bundle and index are atomically published
and verified, the transient rank tree is removed.

If evidence publication fails, the rank tree remains quarantined for the
retention tool. It is never silently removed. A successful final result JSON
continues to exist at its current fixed content-bound path.

Catchable `HUP`, `INT`, and `TERM` execute the same finalization path. `SIGKILL`
and node loss cannot guarantee cleanup; stale transient trees are therefore a
first-class retention input rather than an assumed impossible state.

### Runtime stages and source snapshots

Runtime stages retain their existing source-bound key and read-only verifier.
The current profile's stage and attestation are protected. New runtime stages
remain allowed because source changes require new code and environments.

Retention removes only superseded namespaces that pass all safety checks. This
change does not deduplicate virtual-environment files or alter runtime loading.

## Submission Mode Semantics

### `TEST_ONLY=1`

- Uses a private temporary artifact root.
- Does not contact the scheduler.
- Does not create durable source snapshots, intents, logs, rank directories,
  results, or runtime artifacts.
- Removes the temporary root on success, error, and catchable signal.

### `SBATCH_TEST_ONLY=1`

- Uses a private temporary artifact root.
- Executes the literal scheduler `--test-only` request.
- Does not create entries under durable `RUN_LOG_ROOT`.
- Removes the temporary root after scheduler return on success or failure.

### Actual submission

- Reuses a verified content-addressed snapshot when present.
- Creates exactly one immutable intent before scheduler contact.
- Creates one Slurm log if accepted.
- Rolls back the unique intent after scheduler rejection.
- Never creates a second source snapshot for the same verified content.

For an already cached candidate, submission-time durable inode growth before
job start is bounded to the intent, necessary parent directories, and Slurm
log. It is independent of source-tree size.

## Retention Policy

The retention tool is dry-run by default and requires an explicit `--apply`.
It computes a protected set before selecting any deletion candidate.

Always protected:

- every artifact referenced by the canonical current cluster profile;
- every source, intent, rank tree, log, stage, or attestation referenced by a
  `PENDING`, `RUNNING`, or `COMPLETING` scheduler job;
- the newest verified candidate snapshot for every commit referenced by an
  active or retained submission;
- ownership marker, retention manifest, and cleanup audit records; and
- any path whose schema, ownership, symlink, device, or canonical-root check
  fails.

Per candidate, retain:

- the newest two terminal successful run evidence bundles and result records;
- the newest two terminal failed run evidence bundles and logs; and
- every explicit pin named in a retention manifest.

Superseded runtime namespaces are eligible only when no protected profile,
active job, retained result, or immutable intent refers to them. Deletion is
bottom-up, restricted to exact allowlisted artifact roots, and records the
path, class, identity, entry count, byte count, reason, and timestamp in an
append-only audit manifest.

The retention tool refuses:

- the campaign root itself;
- `RUN_LOG_ROOT`, `runtime-validation`, `source-checkouts`, or
  `source-snapshots` as a direct deletion target;
- paths outside the canonical campaign root;
- symlink traversal;
- filesystem-device changes;
- an incomplete protected-set calculation; and
- deletion while scheduler liveness cannot be established.

## Inode Admission Gate

Before publishing a new durable candidate snapshot or runtime stage, the
launcher reads the user file quota through a typed adapter. The adapter returns
the file limit, files used, and source command evidence.

Policy:

- below 80%: allow normal publication;
- 80% through 90%: require a completed retention dry-run report and print the
  bounded projected inode delta;
- above 90%: fail closed before creating the large artifact;
- missing, malformed, inconsistent, or unavailable quota evidence: fail
  closed for large artifact creation, while cached-snapshot submissions may
  proceed if their projected durable delta is below a fixed small budget.

The threshold is evaluated against projected post-operation use, not current
use alone. Projection uses measured entry counts stored with candidate
snapshots and runtime-stage manifests. It must not recursively scan an entire
environment at submit time.

## Bridge and MCore Unification

MCore and Bridge submitters call the same artifact-lifecycle implementation.
The implementation accepts multiple archive sources but exposes one contract:

```python
prepare_candidate_submission(
    *,
    archive_sources: tuple[ArchiveSource, ...],
    artifact_root: Path,
    mode: SubmissionMode,
    candidate_kind: Literal["mcore", "bridge"],
    candidate_sha: str,
    intent_payload: Mapping[str, object],
) -> SubmissionTransaction
```

`SubmissionTransaction` owns the temporary root, unique intent, and rollback
state. It exposes explicit `commit_scheduler_acceptance()` and `close()`
methods and is also a context manager. There is no implicit deletion in an
object destructor.

Both submitters must use the same command builder for test-only and actual
modes so scheduler validation remains representative.

## Failure Semantics

- Archive or digest failure removes only the unpublished temporary tree.
- Concurrent publication verifies and reuses an identical winner.
- Scheduler rejection removes the unique intent and leaves shared snapshots
  intact.
- Job failure finalizes bounded rank evidence before returning failure.
- Evidence-finalization failure preserves the transient tree in quarantine.
- Retention uncertainty preserves data and fails closed.
- Quota uncertainty prevents large new publications.
- No code path recursively deletes the campaign root.

Cleanup failures never replace the primary scheduler or test exit status. They
are reported as separate lifecycle failures and block promotion.

## Observability

Every submit prints machine-readable lifecycle records for:

- submission mode;
- snapshot cache hit, created, or concurrent reuse;
- snapshot entry count and projected inode delta;
- intent created, scheduler accepted, or rolled back;
- rank evidence finalized or quarantined;
- retention protected/candidate counts; and
- quota used, limit, projected use, and policy decision.

Records contain hashes and paths but no prompt, response, token, or model
content.

## Testing Strategy

All behavior changes use strict RED-GREEN TDD.

### Candidate cache and submission transaction

- same source prepared twice yields one snapshot path and unchanged snapshot
  inode;
- content or marker mismatch at the deterministic path rejects reuse;
- concurrent identical publishers converge on one verified snapshot;
- archive failure leaves no temporary directory;
- intent failure after snapshot publication leaves no temporary intent;
- scheduler rejection removes the unique intent;
- actual accepted submission preserves one intent;
- `TEST_ONLY` leaves no durable entries and never contacts scheduler; and
- `SBATCH_TEST_ONLY` contacts scheduler exactly once with `--test-only` and
  leaves no durable entries.

### Rank evidence

- successful 16-rank/two-node-phase exchange becomes one verified bundle plus
  index and no transient tree;
- first-node failure produces a complete failure bundle;
- collection failure produces a complete failure bundle;
- missing ranks are represented explicitly;
- bundle publication failure preserves quarantine;
- catchable signals run finalization and retain conventional exit status; and
- stale SIGKILL-style trees are classified by retention without assuming the
  owning job is dead.

### Retention

- current profile references are always protected;
- pending/running/completing job references are always protected;
- latest two successes and failures per candidate are retained;
- pinned artifacts are retained;
- superseded terminal artifacts are selected deterministically;
- campaign-root and category-root targets are rejected;
- symlink/path/device escapes are rejected;
- scheduler-query failure causes no deletion;
- dry-run changes nothing; and
- apply mode deletes only the dry-run candidate set and writes an audit record.

### Quota gate

- projected use below 80% permits creation;
- projected use crossing 80% requires retention evidence;
- projected use above 90% rejects before directory creation;
- malformed or missing quota evidence rejects large creation;
- cached small submissions remain bounded under the fixed small-delta rule;
  and
- quota parsing rejects booleans, negative values, overflow, and unit mismatch.

### Integration

- MCore and Bridge real submit harnesses share the lifecycle path;
- full TEST_ONLY and SBATCH_TEST_ONLY harnesses leave durable inode count
  unchanged;
- two actual submissions of one candidate reuse the snapshot and add only
  bounded intent/log/result artifacts;
- submit failure leaves no orphaned transaction artifacts; and
- the existing source, profile, runtime, container, and typed-row gates remain
  unchanged.

## Promotion Gate

The inode lifecycle change is promotable only when:

- all unit and launcher integration tests pass;
- RED evidence exists for every fixed leak path;
- a filesystem-backed test demonstrates zero durable inode delta for both
  dry-run modes;
- a repeated-actual test demonstrates one shared snapshot inode;
- success and failure rank trees finalize to bounded artifacts;
- retention dry-run never includes protected paths;
- the current typed campaign can execute without relaxing provenance or
  runtime attestation; and
- an independent reviewer finds no blocker, high, or medium issue.

Remote cleanup and new GPU jobs are separate operational approvals. This
design does not authorize deleting existing remote data.
