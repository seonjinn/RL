# Task 2 Report: Production-Path Artifact Producer

## Status

Implemented the deterministic SFT validation-event producer without changing the
reviewed runtime CPU validation-cache implementation.

## Scope

- Added `examples/prepare_sft_validation_event.py`.
- Added `setup_data(..., load_validation=False)` to `examples/run_sft.py`.
- Added focused unit coverage in `tests/unit/algorithms/test_sft_validation_artifact.py`.

## Design

- The producer uses `setup_data`, `_build_sft_collate_fn`,
  `_validate_packed_validation_metadata`, and `_combine_validation_event_batches`.
- It creates a non-shuffled `StatefulDataLoader`, takes exactly four complete
  batches of 64 rows, validates packed metadata, preserves batch order, and
  emits only the Task 1 tensor schema.
- Eligibility is derived from actual config and loaded-dataset facts. It accepts
  only `megatron_sft_packed` with its exact production preprocessor, requires
  `data.shuffle=false`, `policy.dynamic_batching.enabled=false`, and the
  text-only Megatron `prepacked_sft_loss_mode=labels` path. It rejects unknown
  dataset or preprocessor contracts and any validation preprocessor. The
  `sequence_packing.enabled` flag is deliberately not used as raw online-packing
  evidence.
- Publication calls `save_validation_event(..., eligibility=eligibility)` and
  uses `ValidationArtifactFingerprint` built from explicit dataset, tokenizer,
  and container digests; internally derived preprocessing provenance; and
  checked-out Git and submodule revisions.

## RED Evidence

1. Added tests for `setup_data(load_validation=False)` before its production
   implementation. The tests prove validation config loading is skipped only
   when requested and remains the default otherwise.
2. Added producer tests before the producer implementation for four packed GBS64
   batches, runtime-combination equivalence, row order, token counts, payload
   digest, fail-closed dataset eligibility, repeatable manifest/tensor bytes,
   and Python/NumPy/Torch RNG preservation.
3. The expected failing-test runs could not reach test collection in this
   worktree:

   - `uv run pytest -q tests/unit/algorithms/test_sft_validation_artifact.py -k 'setup_data'`
     failed because the lockfile supports Linux only.
   - `python3 -m pytest -q tests/unit/algorithms/test_sft_validation_artifact.py -k 'setup_data or producer'`
     failed while importing `tests/unit/conftest.py`: `ModuleNotFoundError: No
     module named 'ray'`.

## GREEN Evidence

- `ruff check examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  passed with `Ruff: No issues found`.
- `ruff format --check examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  passed with all three files formatted.
- `pyright examples/prepare_sft_validation_event.py` passed with `0 errors, 0
  warnings, 0 informations`.
- `python3 -m compileall -q examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  passed.
- `git diff --check` passed.

The full focused pytest suite could not be run locally because this macOS
worktree has no Ray, Torch, OmegaConf, TorchData, or Transformers runtime and
the repository's `uv.lock` excludes macOS. The requested combined Pyright command
also exits nonzero for existing `examples/run_sft.py` TypedDict diagnostics and
missing third-party imports in this interpreter; the new producer file itself is
clean.

## Self-Review

- No call to `init_ray()` exists in the producer path.
- The existing validation-cache runtime code was not modified.
- The producer fails before publication when it cannot prove all required
  eligibility facts.
- Artifact payloads are tensor-only and use the Task 1 public serialization API.

## Remaining Concern

Run the focused pytest command in the supported Linux `uv` environment before
merging. That is the remaining validation gap; no production behavior was
observed locally because the required runtime dependencies are unavailable.

## Review Fixes

### Config Provenance

- The producer now derives `preprocessing_sha256` internally from a canonical
  JSON encoding of an explicitly selected subset of the fully resolved
  `MasterConfig` after Hydra overrides.
- The subset includes validation/default data processing, input length and text
  processing flags, tokenizer rendering config, maximum sequence length,
  sequence/dynamic batching, TP/CP and divisibility settings, packed loss mode,
  and validation batch count/global batch size/microbatch size.
- `--preprocessing-sha256` is now optional and acts only as an expected value.
  A mismatch fails before tokenizer construction, data loading, or publication.
- Added tests proving `data.max_input_seq_length` changes the digest while
  `logger.wandb.name` does not, plus a CLI pre-publication mismatch test.

### Source Provenance

- Fingerprint construction now runs `git status --porcelain=v1
  --untracked-files=all --ignore-submodules=all` in the root repository and in
  every recursive submodule before reading the source fingerprint.
- Added real temporary-Git tests for a clean recursive tree, tracked root
  changes, untracked root files, tracked submodule changes, and untracked files
  in a nested submodule.

### Production Path

- Added a Linux integration path that writes real `.jsonl.packed` records,
  loads `MegatronSFTPackedDataset` through `setup_data` and
  `AllTaskProcessedDataset`, and uses the unpatched production collate and
  `StatefulDataLoader`.
- The real-loader tests cover four complete GBS64 batches with token counts
  `(64, 128, 192, 256)`, row order across all 256 rows, 192-row input, and a
  255-row partial-final-batch input.
- The CLI integration test uses the real resolved Super V3 config, Hydra path
  overrides, real packed loading/collation/dataloader, and real artifact
  serialization. External tokenizer construction and Git fingerprinting are
  isolated in that test; Git provenance has separate real-repository coverage.
- The RNG unit loader deliberately consumes Python, NumPy, and Torch RNG state.
  The test compares complete snapshots, so removing producer restoration causes
  failure.

### Review RED/GREEN Evidence

The strengthened tests were written before their corresponding production
changes. Local RED/GREEN execution remains unavailable for the same environment
reason documented above:

- `python3 -m pytest -q tests/unit/algorithms/test_sft_validation_artifact.py`
  exits during conftest import with `ModuleNotFoundError: No module named 'ray'`.
- `uv run ...` cannot use this repository lockfile on macOS because its supported
  environments are Linux x86_64/aarch64 only.

Fresh locally available GREEN checks after the review fixes:

- `ruff format --check examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  passed.
- `ruff check examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  passed with `Ruff: No issues found`.
- `pyright examples/prepare_sft_validation_event.py` passed with zero errors.
- `pyright tests/unit/algorithms/test_sft_validation_artifact.py` passed with zero
  errors.
- `python3 -m compileall -q examples/prepare_sft_validation_event.py examples/run_sft.py tests/unit/algorithms/test_sft_validation_artifact.py`
  passed.
- `git diff --check` passed.
- Direct `pyrefly` execution is unavailable (`command not found`), and
  `uv run --group dev pyrefly check examples/prepare_sft_validation_event.py`
  exits because the lockfile supports Linux only. The producer is now explicitly
  listed in `pyrefly.toml` for the controller's Linux run.

The required Linux focused-suite GREEN result is pending the controller's CW
container run and should be appended here after that run.

## CW Job 13555872 Follow-Up

### RED Evidence

The controller ran the strengthened focused suite on CW Linux with:

`uv run --python <py3.13> --frozen --group test pytest -q tests/unit/algorithms/test_sft_validation_artifact.py`

The job collected 84 shard items, passed the first four preprocessing-provenance
tests, and failed first at
`test_source_fingerprint_accepts_clean_root_and_recursive_submodules`.
`_submodule_commits` rejected the first recursive submodule-status line because
its prefix was no longer a single space. Full log:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL_worktrees/sft-validation-precomputed-20260707/logs/validation-artifact-tests/20260707-223736-home/sna-val-artifact-uv4_13555872.out`

### Root Cause

The recursive temporary Git fixture is clean. A standalone reconstruction of
the same root -> child -> leaf submodule topology produced empty porcelain
status for the root and both submodules, while raw
`git submodule status --recursive` emitted two lines beginning with the valid
single-space clean marker.

Production `_git_output` normalized all Git stdout with `.strip()`. That removed
the meaningful leading space from the first clean submodule line before
`_submodule_commits` parsed it. The fixture was not misconstructed; production
parsing destroyed valid status syntax.

### Fix And Coverage

- Added explicit assertions that root, child, and leaf porcelain status are
  empty before fingerprint construction.
- Added an assertion that the exact recursive status contains two lines whose
  first character is the clean single-space marker; the assertion includes the
  raw status representation on failure.
- Split raw Git stdout from normalized scalar output. `_submodule_commits` now
  parses raw stdout, while commit-ID and porcelain truthiness callers preserve
  the prior stripped behavior.
- Tracked/untracked root and recursive-submodule rejection logic is unchanged.

Locally available checks after the fix:

- Ruff lint passed.
- Pyright passed with zero errors for the producer and focused test file.
- Python compileall passed.
- `git diff --check` passed.
- Local focused Pytest remains blocked before collection because Ray is absent;
  the controller will rerun the CW Linux suite from the signed fix commit.

### CW Linux GREEN Evidence

The controller reran the complete focused suite from signed commit
`c5739c27612966b6174a4d7034b0ff5f927da16b` in the reviewed NeMo-RL nightly
container on one CW H100 node:

```bash
uv run --python /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/.cache/uv-python/cpython-3.13.13-linux-x86_64-gnu/bin/python3.13 \
  --frozen --group test pytest -q \
  tests/unit/algorithms/test_sft_validation_artifact.py
```

- SLURM job: `13556555`
- State: `COMPLETED`, exit code `0:0`
- Runtime: `00:02:34` on `pool0-00014`
- Result: `84 passed, 3 warnings in 49.02s`
- CUDA rejection coverage executed and passed on H100.
- Real `.jsonl.packed` dataset loading, `AllTaskProcessedDataset`, production
  collate, `StatefulDataLoader`, exact four-batch ordering/token counts, partial
  input rejection, CLI wiring, recursive Git provenance, RNG restoration, and
  artifact hardening all passed.
- Log:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL_worktrees/sft-validation-precomputed-20260707/logs/validation-artifact-tests/20260707-225238-c5739/sna-val-artifact-c5739_13556555.out`

## Train-Derived Validation Provenance Fix

### RED Evidence

Regression tests were added before the implementation for:

- preprocessing digest sensitivity to `data.train.split_validation_size`,
- CLI rejection of positive train-derived validation before tokenizer loading or
  publication,
- acceptance of the normal Super explicit validation config with an absent or
  explicit-zero train split,
- rejection when no explicit validation config exists,
- rejection when `data.default.split_validation_size` contributes a positive
  effective train split, and
- fail-closed rejection when an unknown train dataset omits an explicit zero and
  its constructor default cannot be proven safe.

The pre-implementation Pyright run failed on the intentionally missing
`validate_validation_source_config` import, confirming the new contract was not
present.

### Fix

- Canonical preprocessing provenance now includes the fully resolved
  `data.train` configuration.
- A config-only validation-source preflight runs immediately after Hydra
  resolution and before preprocessing comparison, tokenizer construction, data
  loading, or publication.
- The preflight requires explicit validation data and resolves each train
  entry's effective `split_validation_size` using the same entry-over-default
  precedence as `setup_data`.
- Positive, negative, non-finite, boolean, or otherwise nonzero/unproven split
  values fail closed. The known `megatron_sft_packed` contract may omit the field
  because its production constructor default is zero; other train contracts must
  set zero explicitly.
- Producer eligibility repeats the same source preflight so direct helper use
  cannot bypass it.

### Local Checks

- Ruff format and lint passed for the producer and focused test file.
- Pyright passed with zero errors for both files.
- Python compileall passed.
- `git diff --check` passed.
- Local `uv run --group dev pyrefly check` remains blocked because the lockfile
  supports Linux only.
- Local focused Pytest remains blocked before collection by missing Ray. The
  controller will rerun the complete CW Linux focused suite from the signed fix
  commit.
