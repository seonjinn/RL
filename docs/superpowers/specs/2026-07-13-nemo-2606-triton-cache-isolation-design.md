# NeMo 26.06 Triton Cache Isolation Design

- **Approved approach:** 2026-07-13
- **Primary cluster:** Pre-Tyche
- **Source branch:** `sna/nemo-2606-cutedsl-a2a-factorial-20260712`
- **Model:** `Qwen/Qwen3-30B-A3B`

## Purpose

Remove the shared-filesystem Triton compilation-cache boundary from the official
four-node Qwen3-30B-A3B CuTeDSL benchmark. The change must preserve the official
training workload, both CuTeDSL arms, the pinned container and dependencies, and
the five-warmup plus twenty-measured update contract.

This is an initialization reliability change. Triton cache preparation and JIT
compilation remain outside the accepted measurement window, and the cache
location must not become a benchmark feature variable.

## Failure Evidence and Confidence Boundary

Official job `2369788` completed generation for Step 1, then failed during the
first reference-logprob MoE forward. Rank 0 entered Transformer Engine's fused
MoE chunk-sort path, Triton autotuning called
`fn_cache_manager.get_group(metadata_filename)`, and `json.load` raised
`JSONDecodeError` at line 1 column 1.

The current harness gives each Slurm job a unique runtime directory, so this was
not cross-job cache contamination. Within one job, however, all sixteen policy
ranks on four nodes receive the same Lustre-backed `TRITON_CACHE_DIR`. Jobs
`2369786` and `2369787` used separate job caches and completed the same boundary
in opposite CuTeDSL orders, demonstrating that the model and shared workload do
not fail deterministically there.

Triton 3.6.0 reads group metadata without an exception guard. Its normal writer
serializes into a unique temporary file and uses `os.replace`, so the available
evidence proves only that one reader observed invalid group JSON. It does not
prove an ordinary concurrent-writer race. Cross-client visibility, a prior
corrupt writer, and an abnormal filesystem or I/O event remain possible. The
design therefore removes the unnecessary Lustre boundary and retains bounded
post-failure diagnostics instead of claiming a more specific mechanism.

## Considered Approaches

### 1. Job-scoped node-local cache — selected

Use the same absolute cache path on every allocated node under the existing
job-scoped `/tmp` namespace. Because `/tmp` is node-local, four nodes receive
four independent caches while the four local policy ranks can reuse compiled
artifacts.

This removes cross-node and Lustre cache sharing without multiplying compilation
by all sixteen ranks. The path includes the Slurm job ID and restart suffix, so
requeues and concurrent jobs cannot reuse partial state.

### 2. Rank-scoped node-local cache — fallback only

Give every policy rank an independent cache. This maximizes isolation but can
compile identical Triton kernels sixteen times and materially increase startup
cost. Use it only if the selected node-local, four-rank cache still reproduces
corruption.

### 3. Triton upgrade or runtime monkey patch — rejected for this experiment

Newer upstream code can treat malformed group metadata as a cache miss. Changing
the pinned Triton build or monkey-patching its cache manager would change the
software-under-test and could hide the underlying storage failure. Dependency
upgrades belong to a separate compatibility experiment.

## Runtime Interface

The existing-Ray multinode path gains one canonical node-local runtime root:

```text
/tmp/${USER}/nemo2606-factorial/${RUN_ID}
```

Both the worker virtual-environment root and Triton cache are derived from it:

```text
worker_venvs/
triton_cache/
```

The harness exports `TRITON_CACHE_DIR` before creating Ray actors. NeMo-RL's
existing worker-environment propagation then gives every actor the same path
string. The path resolves on each actor's local node and never points into the
shared result or checkpoint directory.

Single-node and non-existing-Ray modes retain their current run-local container
runtime path. This change targets the demonstrated multinode shared-Lustre
boundary and does not broaden unrelated runtime behavior.

The benchmark manifest records only the cache scope (`job_node_local`), not a
private absolute path or hostname. CuTeDSL ON and OFF must report the same scope.

## Failure Evidence Preservation

The harness must not publish raw cache contents or internal paths. When the Ray
driver exits nonzero, the Slurm-side experiment wrapper runs a bounded epilogue
with one overlapping, CPU-only task per allocated node before it stops the Ray
containers or removes node-local state. Each task scans only its own
job-scoped Triton root and writes one small summary under the shared result
directory. A fixed node index names the summary; hostnames are not retained.

The epilogue is outside every timing arm and does not run after a successful
driver exit. It has a 60-second aggregate timeout. A missing, timed-out, or
truncated node summary is recorded explicitly and never delays normal failure
classification indefinitely.

For every cache metadata candidate visible to a diagnostic task, record:

- a SHA-256 digest of the relative filename rather than the filename;
- file type, byte size, inode, and nanosecond modification time;
- whether bounded bytes parse as JSON;
- a bounded prefix digest and byte count, not raw bytes;
- Triton version, job/restart identity, rank when available, and cache scope;
- truncation reasons when file-count or byte limits are reached.

Each node summary is capped at 256 candidates and 1 MiB of bytes read. The
summary merger must reject symlinks, paths outside the cache root, non-regular
files, non-finite numeric fields, duplicate node indexes, and unbounded
listings. The public report keeps only the failure boundary, counts, hashes,
and interpretation. Credentials, hostnames, IP addresses, absolute paths, and
raw JSON never enter tracked evidence.

Because a failed actor may overwrite or repair a metadata file before the
epilogue observes it, the summary is supporting evidence rather than a guarantee
of capturing the instantaneous invalid bytes. The original traceback remains
the authoritative failure boundary.

## Test-Driven Implementation Contract

Implementation starts with failing tests in
`tests/test_nemo2606_multinode_factorial_harness.py` and the smallest relevant
report test. The tests must prove:

1. existing-Ray multinode mode derives both worker venvs and Triton cache from a
   job/restart-scoped node-local root;
2. the exported Triton path cannot resolve under the shared result, checkpoint,
   or container runtime roots;
3. ON and OFF arms receive identical cache-scope evidence;
4. non-existing-Ray behavior remains unchanged;
5. synthetic malformed, empty, valid, symlinked, and oversized cache inputs
   produce bounded, sanitized diagnostics;
6. the Slurm-side failure epilogue uses one CPU-only task per node, cannot run
   on successful exits, and writes diagnostics before removing node-local state;
7. manifests and reports never expose internal absolute paths, hostnames, IPs,
   credentials, or raw cache contents.

The first new test must fail against commit `9c8962a5b` because the multinode
path still exports a Lustre-backed `CONTAINER_RUNTIME_DIR/triton_cache`. Only
after that RED result may the harness implementation change.

## Remote Validation

After local tests, lint, Bash syntax, and review pass:

1. commit and push only the feature branch;
2. fast-forward the Pre-Tyche worktree and verify recursive source cleanliness;
3. run the exact `--test-only` scheduler preflight;
4. run one four-node, three-update functional gate and verify all actors report
   `job_node_local` cache scope with no Triton metadata error;
5. submit a new three-replica official matrix from one source SHA, alternating
   ON/OFF order and retaining the designated profile replica;
6. require all three replicas, both arms, workload-equivalence checks, kernel
   attribution, and the existing collector contract before a speedup claim.

Jobs `2369786` and `2369787` remain useful execution evidence, but they cannot be
combined with a replacement replica from a different source SHA. A clean matrix
is required after the cache-isolation commit.

## Acceptance and Fallback

The change is accepted only when the functional gate and all three official
replicas complete without Triton cache parse errors, every manifest reports the
same node-local scope, and performance collection passes unchanged.

If node-local sharing among four ranks reproduces the error, preserve the new
diagnostics and move to rank-scoped node-local caches in a separate reviewed
change. Do not add retries, delete cache files during training, patch Triton, or
replace the container without a new root-cause review.

No CuTeDSL, full-iteration CUDA Graph, or A2A-overlap performance conclusion is
changed by this reliability design. Full-CG and A2A remain behind their existing
implementation and attribution gates.
