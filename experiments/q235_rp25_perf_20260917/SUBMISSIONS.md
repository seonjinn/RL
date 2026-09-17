# Qwen3-235B frozen SpecDec submissions

All performance jobs use the official `grpo-qwen3-235b-16n4g.yaml` workload on
Lyris. The W&B project is `nvidia/sna-specdec`, and the group is
`q235-rp25-frozen-perf-20260917`.

## Baseline

| Scope | Job | State at receipt | W&B |
|---|---:|---|---|
| 1-step NUMA fix gate | 3086466 | COMPLETED, exit 0 | [zshyatn8](https://wandb.ai/nvidia/sna-specdec/runs/zshyatn8) |
| 20-step no-SpecDec | 3086570 | PENDING; estimated 09:17 PDT | created when the run starts |

## Frozen SpecDec matrix

Each 20-step job has an `afterok` dependency on its one-step gate.

| Arm | 1-step gate | 20-step job |
|---|---:|---:|
| EAGLE-3 K3 | 3086769 | 3086771 |
| EAGLE-3 K5 | 3086773 | 3086775 |
| DFlash B8 K5 | 3086777 | 3086779 |
| DFlash B8 K7 | 3086781 | 3086783 |
| DSpark B8 K5 | 3086785 | 3086787 |
| DSpark B8 K7 | 3086789 | 3086791 |
| DFlash B16 K11 | 3086793 | 3086795 |
| DFlash B16 K13 | 3086797 | 3086799 |
| DSpark B16 K11 | 3086801 | 3086803 |
| DSpark B16 K13 | 3086805 | 3086807 |

The first isolated-source gate, EAGLE-3 K3 job `3086659`, stopped before Ray
startup because `cp -a` attempted to preserve metadata that node-local
`/raid/scratch` does not support. The setup now uses a recursive content copy,
covered by an execution test that simulates an archive-metadata failure. All
other gates rendered from the affected revision were cancelled before startup
and replaced by the jobs above. Each replacement gate has a one-hour limit;
each 20-step job is protected by an `afterok` dependency on its gate.

## Checkpoint transfer receipt

- OCI-HSG upload job: `7216062`, COMPLETED, exit 0, 8 files, 9.196 GiB.
- First Lyris download attempts `3086656` and `3086691` exposed wrapper-only
  failures (`time` missing and `/bin/sh` rejecting `pipefail`).
- Corrected Lyris download job: `3086703`, 8/8 files and 9.196 GiB copied.
- Lyris staging root:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/drafters/q235-base-ptv2en-s25391-20260917`.
- Config receipt: DFlash/DSpark architectures and B8/B16 block sizes match all
  four expected exports.

## Ptyche readiness hedge

- Detached source worktree:
  `/home/sna/nemorl-q235-specdec-matrix-20260917`, revision `45a21c1b6`.
- The original Ptyche `batch` data-mover job `2843329` was cancelled after
  `sbatch --test-only` estimated November 28.
- Replacement PDX download: backfill job `2843331`, expected September 17 at
  08:50 PDT. It stages the same eight files and verifies the file count.
- A 16-node, one-hour Ptyche training scheduling probe estimated November 28.
  No duplicate Ptyche training matrix was submitted; Lyris remains the active
  measurement site.
