# Qwen3-235B frozen SpecDec submissions

All performance jobs use the official `grpo-qwen3-235b-16n4g.yaml` workload.
Lyris carries the complete matrix, including EAGLE-3, while Ptyche carries a
matched baseline plus DFlash/DSpark matrix on its high-priority `36x2-a01r`
partition. The W&B project is `nvidia/sna-specdec`, and the group is
`q235-rp25-frozen-perf-20260917`.

## Lyris baseline

| Scope | Job | State at receipt | W&B |
|---|---:|---|---|
| 1-step NUMA fix gate | 3086466 | COMPLETED, exit 0 | [zshyatn8](https://wandb.ai/nvidia/sna-specdec/runs/zshyatn8) |
| 20-step no-SpecDec | 3086570 | RUNNING since 06:34:41 PDT; initialization healthy at five minutes | [9mt3ocw9](https://wandb.ai/nvidia/sna-specdec/runs/9mt3ocw9) |

## Lyris frozen SpecDec matrix

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

## Ptyche frozen matrix

Ptyche uses source revision `948043afb137de13846e195678978b92fcbbbfe6`.
Every 20-step job has an `afterok` dependency on its one-step gate, and every
submission passed the launcher's `sbatch --test-only` check first.

| Arm | 1-step gate | 20-step job |
|---|---:|---:|
| Baseline, no SpecDec | 2843343 | 2843345 |
| DFlash B8 K5 | 2843347 | 2843349 |
| DFlash B8 K7 | 2843351 | 2843353 |
| DSpark B8 K5 | 2843355 | 2843357 |
| DSpark B8 K7 | 2843359 | 2843361 |
| DFlash B16 K11 | 2843363 | 2843365 |
| DFlash B16 K13 | 2843367 | 2843369 |
| DSpark B16 K11 | 2843371 | 2843373 |
| DSpark B16 K13 | 2843375 | 2843377 |

- Detached source worktree:
  `/home/sna/nemorl-q235-specdec-matrix-20260917`.
- The original Ptyche `batch` data-mover job `2843329` was cancelled after its
  scheduler estimate moved to November 28.
- Replacement backfill transfer job `2843331` completed: 8 files and
  9,874,628,326 bytes, with all DFlash/DSpark B8/B16 config receipts verified.
- A 16-node, one-hour probe on `36x2-a01r` estimated an immediate September 17
  window, so the actual matrix uses that partition instead of `batch`.
- The EAGLE-3 checkpoint is not staged on Ptyche. EAGLE-3 K3/K5 therefore stay
  on Lyris rather than mixing an unverified checkpoint transfer into this
  cohort.

### Ptyche one-step receipts

| Arm | Gate state | W&B |
|---|---|---|
| Baseline, no SpecDec | COMPLETED, exit 0 | [65bxq2im](https://wandb.ai/nvidia/sna-specdec/runs/65bxq2im) |
| DFlash B8 K5 | COMPLETED, exit 0 | [68u2ugzd](https://wandb.ai/nvidia/sna-specdec/runs/68u2ugzd) |
| DFlash B8 K7 | RUNNING | [kg88ulnp](https://wandb.ai/nvidia/sna-specdec/runs/kg88ulnp) |
| DSpark B8 K5 | RUNNING | [sfvuoy8u](https://wandb.ai/nvidia/sna-specdec/runs/sfvuoy8u) |
| DSpark B8 K7 | RUNNING | [tf1w5ng9](https://wandb.ai/nvidia/sna-specdec/runs/tf1w5ng9) |
| DFlash B16 K11 | RUNNING; W&B initialization pending | pending |

The completed baseline and DFlash K5 gates released their respective 20-step
dependencies (`2843345` and `2843349`).
