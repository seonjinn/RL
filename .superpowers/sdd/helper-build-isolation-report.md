# MCore helper-build isolation implementation report

Date: 2026-07-13 PDT

## Outcome

The experiment harness now prevents MCore dataset-helper builds from writing to
the canonical shared checkout. Official existing-Ray submissions and the direct
A/B launcher force non-editable local package installs before the matrix payload
starts. The payload then builds one ABI-suffixed helper in a unique tracked-only
MCore overlay, proves that later Make calls are no-ops, and publishes bounded
runtime evidence.

No NeMo-RL production file, Megatron-Bridge file, or Megatron-LM file changed.

The post-implementation review found one direct-Pyxis topology gap: creating
the archive after the repository was remapped required nested worktree Git
metadata that is only valid at the host path. The corrected harness creates
the pinned archive in the outer shell and makes the container bootstrap
extract-only.

## TDD evidence

The six required regressions were added before implementation and run with the
macOS-compatible existing project test environment:

```text
6 failed, 64 deselected in 2.51s
```

The failures were the intended missing-feature failures:

- ambient `UV_NO_EDITABLE=0` reached every functional/factorial export payload;
- the matrix had no non-editable fail-closed contract;
- no tracked overlay, matching `python3-config`, serial helper build, `make -q`,
  or import-origin check existed;
- canonical helper artifacts were not explicitly rejected before and after
  bootstrap;
- the bounded `mcore_runtime` evidence schema was absent;
- no overlay-under-cleanup-root contract existed.

After the first implementation:

```text
6 passed, 64 deselected in 0.88s
```

A compatibility audit found that `submit_cutedsl_ab_replicates.sh` also invokes
the now-fail-closed matrix payload. Its executable regression was added before
the fix and reproduced the inherited ambient value:

```text
1 failed, 70 deselected in 0.85s
assert {'0'} == {'1'}
```

After adding the same scrub-and-force contract to that launcher:

```text
1 passed, 70 deselected in 0.73s
```

The review correction added two regressions before changing the harness. The
direct-Pyxis contract failed because the host-archive marker was absent:

```text
1 failed, 71 deselected in 0.48s
assert '# CUTEDSL_MCORE_HOST_ARCHIVE_START' in source
```

The outer canonical recheck contract failed independently because its marker
was absent:

```text
1 failed in 0.05s
assert '# CUTEDSL_MCORE_CANONICAL_POST_BOOTSTRAP_START' in source
```

The GREEN regression constructs a real linked Git worktree, rewrites its
`.git` file to use a relative gitdir, and proves all three relevant states:

- archiving from the original host topology succeeds at the exact commit;
- Git access fails after copying that worktree to a differently nested
  direct-container-style path;
- extracting the already-created run archive succeeds without Git metadata.

Together with the static ordering regression, the corrected focused result
was:

```text
3 passed, 70 deselected in 0.63s
```

The additional config-validation SRUN initially exposed two existing
per-SRUN invariants: every container entry must revalidate the locked runtime
interpreter and `NVTE_CUDA_ARCHS=100`. After applying those same checks to the
new SRUN, the wrapper suite passed `53 passed in 2.41s`.

The expanded wrapper suite then found one harness-compatibility regression:
placing the new shell function before the established six-line container TMPDIR
preamble made the preamble extractor execute `local` outside a function. The
function was moved immediately after the existing TMPDIR diagnostics and
assertions. The focused wrapper regression returned to `1 passed, 52
deselected`, and the final 282-test suite remained green.

The canonical artifact regression also executes the payload's rejection
function against an ABI-suffixed `helpers_cpp*.so`, and the non-editable matrix
regression executes the payload with both missing and false ambient values.

## Implementation

### Submission contract

- Both functional and factorial export-file blocks scrub ambient
  `UV_NO_EDITABLE` and add literal `UV_NO_EDITABLE=1`.
- The direct A/B submitter uses the same contract.
- The matrix exits immediately unless the exact value is `1`.

### Canonical source protection

- The nested MCore SHA is resolved from the pinned Bridge gitlink and checked
  against the canonical nested checkout HEAD.
- `find -name 'helpers_cpp*'` rejects every canonical artifact type, including
  ignored ABI shared objects and symlinks, before runtime work.
- The mounted canonical source is checked again inside the locked bootstrap
  before uv work and after the overlay build/import proof.
- Immediately after the bootstrap SRUN returns, the outer shell rechecks the
  host canonical helper directory and parent, Bridge, and MCore porcelain
  status with all untracked files and recursive submodule state enabled.
- Config resolution runs only after that outer recheck, in a separate SRUN
  with the same runtime-interpreter and SM100 assertions.

### Run-scoped overlay

- Before Pyxis can remap the repository, outer-shell `git archive` writes only
  content tracked by the pinned MCore commit to
  `${HOST_RUNTIME_DIR}/mcore-source.tar`.
- The mounted bootstrap reads `${CONTAINER_RUNTIME_DIR}/mcore-source.tar` and
  only extracts it into `${CONTAINER_RUNTIME_DIR}/mcore-overlay`; it performs
  no Git operation and therefore needs no canonical `.git` metadata.
- A matching uv-base `python3-config` is exposed through the run-scoped
  `runtime-bin`; the build `python3`, runtime interpreter, and extension suffix
  must agree.
- One `make -j1` builds the exact ABI-suffixed regular non-symlink helper.
- A suffixless helper is forbidden, `make -q` must pass, and the imported module
  origin must equal the overlay helper before GRPO config resolution.
- Overlay and tool directories are children of the existing runtime cleanup
  root, so the existing exit trap removes them on success, failure, or signal.

### Bounded evidence

`mcore_runtime.json` and `benchmark_manifest.json.mcore_runtime` contain exactly:

- source SHA and helper SHA256;
- tracked archive mode and non-editable boolean;
- Python extension suffix and relative helper path;
- overlay origin and matching-uv-base config-tool labels;
- `make -q` result and zero canonical artifact counts before/after.

No absolute overlay, repository, or internal filesystem path is published in
that object.

## Verification

Final focused harness after the review correction:

```text
73 passed in 12.08s
```

Final harness plus relevant HF-cache, recipe, collector, report, diagnostics,
wrapper, and Qwen3-235B suites:

```text
284 passed in 26.42s
```

Static checks passed:

- Bash syntax for the matrix, existing-Ray submitter, and direct A/B submitter;
- Ruff check and format check for the modified Python test;
- Python bytecode compile for the modified test;
- `git diff --check`.

The broader cluster-profile suite has a pre-existing source/test mismatch that
is outside this change: all three committed profiles export 11 keys while
`test_profile_scripts_export_exactly_required_keys` expects 10. The unchanged
suite result is `33 passed, 3 failed`; none of the profile files is in this
diff.

The repository lock supports Linux x86_64/aarch64 only, so `uv run --no-sync`
cannot resolve on this macOS host. Local pytest and Ruff commands used the
existing compatible project environment. This local result does not claim an
ARM helper compile.

## Remaining Linux gates

Authoritative acceptance still requires the documented remote gates after the
old job has stopped and canonical artifacts are safely quarantined:

1. pre-Tyche test-only export inspection;
2. one fixed three-update functional job;
3. two overlapping functional jobs from the same clean checkout;
4. only then, a fresh six-job official performance cohort.

The functional gate must prove an aarch64 ABI suffix, overlay-only Make paths,
no worker `g++` command, no missing `python3-config`, a clean canonical checkout
throughout, and three completed policy updates.
