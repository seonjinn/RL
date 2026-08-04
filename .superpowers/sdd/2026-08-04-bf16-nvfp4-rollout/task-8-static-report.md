# Task 8 Static Artifact Report

## Scope

Implemented only the static Task 8 smoke definitions and experiment planning
artifacts. No production code, recipe YAML, test-list file, or runtime report was
modified. No jobs were submitted.

## Artifacts

- Added W4A16 and W4A4 performance smoke scripts using the existing
  `tests/test_suites/llm/performance/common.env` contract.
- Added reproducible experiment setup and launch documentation.
- Added the four-run execution, monitoring, validation, and reporting plan.

## Script contract

- Two GRPO steps by default; checkpointing is forced off.
- `REFIT_TRANSPORT=null` selects legacy refit and
  `REFIT_TRANSPORT=nccl_reshard` selects NCCL-Reshard.
- NCCL forces non-colocated generation and uses a two-node train/two-node
  generation split. Megatron EP8 keeps the four-node B200 allocation valid.
- W4A4 fails before launch unless `NVFP4_CALIBRATION_ARTIFACT` names an existing
  file.
- Default log directories and W&B names include the quant mode and transport;
  the W&B project defaults to `sna-bf16-nvfp4-rollout` and supports an explicit
  override.
- Runtime assertions require the real ModelOpt worker, expected W4A16/W4A4
  method, two refit timing records, and two training-step records.
- Runtime assertions reject QARL/fake quantization, incomplete reloads, manifest
  errors, NaN/infinity values, NCCL agreement errors, and refit exceptions.

## Verification

Passed:

- `bash -n` on both smoke scripts.
- `git diff --check` before staging.
- Recipe composition and worker-resolution tests: `2 passed in 2.66s`.
- Static script contract checks for executable mode, two-step defaults,
  checkpoint disablement, both transport values, forced NCCL non-colocation,
  unique W&B/log controls, two refit/step assertions, and all required failure
  patterns.
- Early-failure checks: invalid `REFIT_TRANSPORT` and absent
  `NVFP4_CALIBRATION_ARTIFACT` both exited with status 2 before GRPO launch.
- W4A16 and W4A4 NCCL override configs passed
  `check_nccl_reshard_refit_support()` with the mode resolver substituted.

Local limitations:

- The repository lockfile supports Linux only, so `uv run --frozen pytest` cannot
  execute on this macOS host. The recipe test used the existing rollout
  worktree's environment with this worktree first on `PYTHONPATH`.
- ModelOpt is absent locally. The NCCL static validator run substituted only
  `resolve_nvfp4_real_quant_mode`; all other production validation executed.
- `tools/launch DRYRUN=1` cannot extract CONFIG blocks on macOS because its GNU
  `sed` basic-regex `\+` is not supported by BSD `sed`. The same command fails on
  an existing performance script. Equivalent extraction confirmed both new
  CONFIG blocks, and the launch dry-run remains a Linux/container gate.
- `shellcheck` is not installed locally.

## Runtime status

Not run. Task 8 GPU submission and `report.md` creation remain intentionally out
of scope for this static-only change.
