# Task 2 Hardening Report

## Status

`DONE_WITH_CONCERNS`: the pre-submission gate hardening is implemented, locally
verified where macOS permits, committed, and pushed. No cluster job or GPU
capability row was submitted. The direct-TE capability remains a raw MCore
candidate result and is not an integrated NeMo-RL attestation.

## Source State

- NeMo-RL base: `02bdf3ba90cd93cff9e905d42b811f6f3ef34891`
- NeMo-RL hardening implementation: `4d57dc35eee67caab6a51741bc17a9f4f7932320`
- MCore base: `564fa5e0c981d7e4cd09a594fcd98c6799bdbb88`
- MCore hardening candidate: `734679b358341fec37d98397944d2aac49601471`
- MCore candidate push: `origin/sj/thd-cg-hybrid-nemotron-20260731`
- NeMo-RL push target: `seonjinn/experiment/thd-cg-hybrid-nemotron-20260731`

Both repositories were clean and exactly at the requested bases before edits.
No history was rewritten and no unrelated changes were reverted.

## Implemented Hardening

### Run-unique distributed aggregation

The root runner now derives one run identity from the trusted Slurm job ID,
Slurm restart count, and SHA256 of the non-symlink submission intent. Per-rank
files are written below a directory containing that identity, so a later job or
requeue attempt cannot observe an earlier attempt's rank files.

Every rank payload now binds the run identity, candidate kind and SHA, row ID,
global rank, world size, node count, GPU count, literal pytest-node results,
and capability metadata. Rank 0 validates the exact schema and expected value
for every rank before aggregation. Device-binding capability fields may differ
by rank; all remaining semantic capability fields must be identical. The
published atomic result records the validated run identity.

Regression tests prove that an earlier scheduler run is assigned a distinct
directory and that a stale payload is rejected. A second regression permits
rank-specific device bindings but rejects one rank changing semantic capability
evidence.

### Persistent staged root validation

The immutable `RUNTIME_PHASE=stage` path now invokes the committed
`scripts/run_task2_root_tests.sh` runner with the staged Python environment.
The runner executes exactly these six modules before any stage job record,
immutable audit, completion marker, or `RUNTIME_STAGE_READY` publication:

- `tests/unit/experiments/test_validate_te_runtime.py`
- `tests/unit/experiments/test_runtime_attestation.py`
- `tests/unit/experiments/test_container_harness_hardening.py`
- `tests/unit/experiments/test_mcore_standalone_driver.py`
- `tests/unit/experiments/test_matrix_submitters.py`
- `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

The staged run disables bytecode and pytest cache generation and persists a
JUnit result under the immutable stage root. A harness regression executes the
runner against a controlled Python shim, verifies the exact argv, and verifies
that invocation precedes marker publication. This closes the gap exposed by
scratch job `5839681` (pytest absent) and staging job `5840488` (pytest present
but the suite not executed).

### Direct-TE probe

The MCore probe now reads the torchrun rank environment and calls
`torch.cuda.set_device(LOCAL_RANK)` before CUDA availability, current-device,
or device-count queries. No unrelated CUDA Graph test setup changed.

The `_reuse_graph_input_output_buffers=True` characterization still accepts
either the pinned TE rejection or a future accepted branch. If accepted, it
must now prove capture occurred, two changed inputs produce changed outputs,
both replays match eager results without entering Python `forward`, a deliberate
train/eval mismatch enters eager `forward` exactly once, and no parameter
gradient exists. The capability artifact records this accepted-branch evidence.
MCore production forward-only execution kinds were not implemented.

## TDD Evidence

- Baseline pure root runner suite: `26 passed`.
- Freshness/consensus regressions first failed because
  `derive_run_identity` did not exist, then passed after the runner change.
- Persistent-stage regression first failed because
  `run_task2_root_tests.sh` did not exist, then passed after the staged runner
  and pre-marker invocation were added.

## Local Verification

- Focused six-module root command using the available host pytest with parent
  conftest isolation: `229 passed in 46.71s`.
- Root `ruff check` on all changed Python files: passed.
- Root Python bytecode compilation on all changed Python files: passed.
- `bash -n` on both matrix submitters, both distributed wrappers, the staged
  test runner, and the runtime stage wrapper: passed.
- Root and MCore `git diff --check`: passed.
- MCore `ruff check tests/unit_tests/transformer/test_cuda_graphs.py`: passed
  with only the repository's existing top-level-settings deprecation warning.
- MCore bytecode compilation: passed.
- MCore commit signature: valid ED25519 Git signature; commit also contains the
  required sign-off.

## Limitations and Required Linux/GPU Follow-up

The canonical root `uv run pytest` command cannot run on this Apple Silicon
macOS host because the NeMo-RL lock supports only Linux x86_64/aarch64
environments. The host fallback required `--confcutdir=tests/unit/experiments`
because the full root conftest imports unavailable Ray. The passing host run
also emitted macOS pytest cleanup warnings for intentionally immutable fixture
directories; these were warnings, not test failures.

The required MCore `uv run isort` command cannot resolve on this host because
`nvidia-cudnn-frontend==1.26.0` has no macOS wheel. Imports were not changed;
the locally available ruff lint and compile checks passed. The two direct-TE
nodes require the pinned Linux container, CUDA, Transformer Engine, and eight
torchrun ranks, so they were not executed locally.

Before any capability claim, run the committed immutable stage path on Linux,
confirm its exact six-module JUnit result, then submit only
`te_eval_capability_8` and require eight joined ranks plus unanimous semantic
capability metadata. A passing candidate artifact must not authorize an
integrated NeMo-RL leaf until Task 6 pins the validated MCore SHA through
Bridge/NeMo-RL.
