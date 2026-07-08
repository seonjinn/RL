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
