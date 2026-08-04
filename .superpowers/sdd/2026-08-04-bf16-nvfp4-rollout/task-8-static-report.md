# Task 8 Static Artifact Report

## Scope

Implemented the Task 8 smoke definitions, immutable model-revision recipe pins,
and experiment planning artifacts. No production Python code or runtime report
was modified. No GRPO job was submitted.

## Artifacts

- Added W4A16 and W4A4 performance smoke scripts using the existing
  `tests/test_suites/llm/performance/common.env` contract.
- Added reproducible experiment setup and launch documentation.
- Added the four-run execution, monitoring, validation, and reporting plan.

## Script contract

- Two GRPO steps by default; checkpointing is forced off.
- `REFIT_TRANSPORT=null` selects legacy refit and
  `REFIT_TRANSPORT=nccl_reshard` selects NCCL-Reshard.
- GCP-NRT maps the recipe's 16-GPU world to two 8-GPU B200 nodes. NCCL forces
  non-colocated generation and uses a one-node train/one-node generation split;
  Megatron EP8 keeps the eight-GPU training world valid.
- W4A4 fails before launch unless `NVFP4_CALIBRATION_ARTIFACT` names an existing
  file.
- Both recipes pin the Qwen3-30B-A3B revision used by W4A4 calibration.
- Legacy submission requires scheduler/application segment size 2; NCCL requires
  both to be 1.
- The scripts require `WANDB_API_KEY`. The documented launch mounts `/lustre`
  and uses a fresh commit/run-specific snapshot for every campaign.
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
- Recipe composition, revision, and worker-resolution tests passed before the
  final documentation hardening.
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
- A later focused pytest rerun stalled while importing the existing worker
  resolution stack in the reused macOS environment. `py_compile`, Ruff, direct
  config composition, and shell syntax validation passed; the full focused test
  remains a target-container gate.

## Runtime status

Not run. Task 8 GPU submission and `report.md` creation remain intentionally out
of scope for this static-only change.
