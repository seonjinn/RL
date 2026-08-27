# PR1-11 Draft Online Training Validation Matrix

**Audit boundary:** local repository evidence at
`cf683b75c159d73bdf7a4b67236887570f7ea6a6`; the submitted product source is
`1be8237816bfd78dad752dd5c1e0149ae2420301`. This review did not query SLURM
or W&B, submit work, or treat a configuration/capability declaration as runtime
evidence.

## Evidence grades

- `code` — an implementation or input exists, but this audit has no passing
  deterministic-test receipt for the exact claim.
- `unit` — a deterministic automated test has a passing receipt for the exact
  claim.
- `composed` — the exact Linux/container composition has a passing receipt.
- `scheduled` — a concrete, arm-bound scheduler preflight or job receipt exists.
- `runtime` — an identified GPU job crossed the stated behavioral gate.

The local macOS host cannot satisfy the Linux-only lockfile, so the focused
`uv run --frozen --extra mcore --group test pytest ...` command stopped before
collection with the lockfile platform error. Consequently, test source alone is
not promoted to `unit`, and no cell is promoted to `composed` or `runtime`.
Arm-bound durable submission receipts do support the limited `scheduled` grades
shown below; scheduler execution is not a training-runtime result.

| Drafter | CP1 packed | CP>1 unpacked | CP>1 packed | Repeated update/refit | Multi-node | Evidence grade | Evidence pointer | Next validation action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DFlash | `scheduled` — CP1-packed arms were allocated: static `6575002`, always `6575004`, fixed-10 `6575005`. | `code` — CP>1 without packing is an explicitly rejected layout, not an execution target. | `code` — CP2/CP4 packed setup and split-entrypoint paths exist. | `code` — the trainer exports draft tensors into the refit lane, but no two-refit receipt is retained. | `scheduled` — each submitted CP1-packed DFlash arm received a four-node allocation, then failed before driver gates. | `scheduled` for CP1-packed and multi-node; `code` otherwise | Durable receipts: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submissions`; submitted-input contract: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dflash-fixed10.yaml:14-34`; source tests: `tests/unit/models/policy/test_dflash_worker_validation.py::test_dflash_setup_allows_packed_cp_with_target_sp`, `::test_dflash_setup_requires_nemo_owned_packing_for_cp`, and `::test_packed_cp_draft_training_requires_split_entrypoint`; refit source: `nemo_rl/models/policy/workers/megatron_policy_worker.py::_iter_draft_weights_for_refit`. | Repair the compute-node Ray/Slurm discovery failure under an approved later run, then retain CUDA-graph, step-1, step-2, and **two successful refits followed by rollout** gates for one CP1-packed, four-node DFlash arm. |
| DSpark | `scheduled` — CP1-packed arms were allocated: static `6575010`, always `6575013`, fixed-10 `6575023`. | `code` — CP>1 unpacked is outside the documented supported topology; no contrary runtime receipt was found. | `code` — CP4 packed setup and matching-generation path exist. | `code` — the common draft refit lane applies to DSpark, but no two-refit receipt is retained. | `scheduled` — each submitted CP1-packed DSpark arm received a four-node allocation, then failed before driver gates. | `scheduled` for CP1-packed and multi-node; `code` otherwise | Durable receipts: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submissions`; submitted-input contract: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dspark-fixed10.yaml:14-38`; source tests: `tests/unit/models/policy/test_dflash_worker_validation.py::test_dspark_setup_allows_packed_cp4_target_sp_and_matching_generation`; `tests/unit/models/policy/test_policy_validation.py::test_megatron_packed_policy_accepts_capable_draft_provider`; refit source: `nemo_rl/models/policy/workers/megatron_policy_worker.py::_iter_draft_weights_for_refit`. | Repair the compute-node Ray/Slurm discovery failure under an approved later run, then retain CUDA-graph, step-1, step-2, and **two successful refits followed by rollout** gates for one CP1-packed, four-node DSpark arm. |

## Ownership and evidence limits

PR1-11 owns the source-side contract chain: typed draft configuration (PR1),
DFlash/DSpark integration (PR5-7), and runtime/refit plumbing (PR9-11). The
matrix does not convert that ownership into an online-training result.

The requested cross-PR trackers `#3750` and `#3757` are listed as external
validation pointers only. No local issue export establishes their status,
assignee, job mapping, or gate outcome, so neither issue is evidence for a
matrix grade. Their integrated packed-CP, repeat-refit, and multi-node gates
remain cross-PR follow-up work rather than a claim of this document.

The durable receipts map DFlash static/always/fixed-10 to `6575002`/`6575004`/
`6575005` and DSpark static/always/fixed-10 to `6575010`/`6575013`/`6575023`.
All six received four-node allocations and exited `1:0` after 18-19 seconds.
Their per-run Slurm logs are under the sibling artifacts tree. They failed
before the driver, W&B, CUDA-graph, step, update, refit, or rollout gates:
compute-node `ray.sub` could not find `sinfo`/`scontrol`, so automatic
`CPUS_PER_WORKER` detection failed. This is concrete `scheduled` evidence and
also concrete evidence that no training behavioral gate ran. The archived PR1
job `6298092` remains excluded because it validates an earlier typed-contract
head, not DFlash/DSpark online training on the submitted product SHA.

## Remaining gates

1. Run the focused DFlash/DSpark policy and refit test set in the pinned Linux
   container, preserving the command output; only then may exact covered cells
   become `unit`.
2. Preserve per-arm composition output for the CP1-packed four-node Q30 inputs;
   a source file alone is not `composed` evidence.
3. Resolve the compute-node `ray.sub` Slurm-command discovery failure before a
   later approved run; the existing scheduled jobs must not be modified or
   resubmitted in this scope.
4. For each drafter, retain the job ID, W&B URL/run ID, and gate log proving
   CUDA graph capture, steps 1 and 2, an update/refit, a second update/refit,
   and a subsequent rollout. Only that record qualifies the repeat-refit cell
   as `runtime`.
5. Run and retain a CP2 or CP4 packed split-entrypoint validation independently
   of the CP1 four-node run. The CP>1-unpacked result must remain an expected
   rejection, not a positive runtime target.

`research/qwen3_8b_draft_cadence_200step/PREPARATION_REPORT.md` separately
states that its prepared screen has no W&B run, SLURM job ID, throughput result,
or speedup. It is therefore excluded from all scheduled/runtime grades.
