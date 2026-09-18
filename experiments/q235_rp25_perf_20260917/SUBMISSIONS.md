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
| 20-step no-SpecDec | 3086570 | COMPLETED, exit 0, 20/20 steps, 02:43:59 | [9mt3ocw9](https://wandb.ai/nvidia/sna-specdec/runs/9mt3ocw9) |

## Lyris frozen SpecDec matrix

Each 20-step job has an `afterok` dependency on its one-step gate.

| Arm | 1-step gate | 20-step job |
|---|---:|---:|
| NVIDIA EAGLE-3 K3 | 3090122, queued | 3090127, afterok, five hours |
| NVIDIA EAGLE-3 K5 | 3090124, queued | 3090130, afterok, five hours |
| DFlash B8 K5 | 3086777, COMPLETED, [ebfc2leq](https://wandb.ai/nvidia/sna-specdec/runs/ebfc2leq) | 3088187, pending, five hours |
| DFlash B8 K7 | 3086781, COMPLETED, [g4l0eptf](https://wandb.ai/nvidia/sna-specdec/runs/g4l0eptf) | 3088189, pending, five hours |
| DSpark B8 K5 | 3086785, COMPLETED, [glbjy493](https://wandb.ai/nvidia/sna-specdec/runs/glbjy493) | 3088191, pending, five hours |
| DSpark B8 K7 | 3086789, COMPLETED, [bbxp8zp8](https://wandb.ai/nvidia/sna-specdec/runs/bbxp8zp8) | 3088204, pending, five hours |
| DFlash B16 K11 | 3088220, replacement gate pending | 3088230, afterok, five hours |
| DFlash B16 K13 | 3088222, replacement gate pending | 3088232, afterok, five hours |
| DSpark B16 K11 | 3088224, replacement gate pending | 3088234, afterok, five hours |
| DSpark B16 K13 | 3088227, replacement gate pending | 3088236, afterok, five hours |

The first isolated-source gate, EAGLE-3 K3 job `3086659`, stopped before Ray
startup because `cp -a` attempted to preserve metadata that node-local
`/raid/scratch` does not support. The setup now uses a recursive content copy,
covered by an execution test that simulates an archive-metadata failure. All
other gates rendered from the affected revision were cancelled before startup
and replaced by the jobs above. Each replacement gate has a one-hour limit;
each 20-step job is protected by an `afterok` dependency on its gate. All ten
pending Lyris full runs were extended to the `gb200` partition maximum of five
hours after live runs exposed long-tail validation/refit steps.

The original EAGLE-3 gates and full runs used the RedHatAI speculator. Jobs
`3088183` and `3088185` were cancelled after partial progress when the requested
comparison was narrowed to NVIDIA's public
`nvidia/Qwen3-235B-A22B-Eagle3` checkpoint. The NVIDIA replacement uses pinned
revision `33f3c01ce807376d1171301b9a148b1b28f239ba`, staged under the Lyris
Hugging Face cache. Its `config.json`, `hf_quant_config.json`, and 620,791,032
byte weight file were checked before submission. Both replacement gates and
their dependent 20-step runs passed `sbatch --test-only` from clean source
revision `6700cf315de935e1e4d07af4992bd14528261801`.

The first full-run scripts were rendered before the Ray threshold fix. They
were replaced before execution by the job IDs in the table above and the old
full jobs `3086771`, `3086775`, `3086779`, `3086783`, `3086787`, `3086791`,
`3086795`, `3086799`, `3086803`, and `3086807` were cancelled. The replacement
matrix uses the clean source checkout
`/home/sna/nemorl-q235-specdec-matrix-v2-20260917` at revision
`eb323c5092d8d16955128c7c3153c2ba81c92cbf`, Ray's still-enabled 98% host
memory threshold, and five-hour allocations. Every replacement submission
passed `sbatch --test-only`.

The original B16 gates `3086793`, `3086797`, `3086801`, and `3086805` failed in
20 seconds because their old source checkout became dirty during an interrupted
submodule refresh, causing the launcher's clean-source preflight to exit before
Ray startup. Their empty Slurm logs and absence of driver logs confirm this was
not a model or drafter failure. The replacement gates in the table use the
isolated clean checkout.

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
| DFlash B8 K7 | COMPLETED, exit 0 | [kg88ulnp](https://wandb.ai/nvidia/sna-specdec/runs/kg88ulnp) |
| DSpark B8 K5 | COMPLETED, exit 0 | [sfvuoy8u](https://wandb.ai/nvidia/sna-specdec/runs/sfvuoy8u) |
| DSpark B8 K7 | COMPLETED, exit 0 | [tf1w5ng9](https://wandb.ai/nvidia/sna-specdec/runs/tf1w5ng9) |
| DFlash B16 K11 | COMPLETED, exit 0 | [zikaotsf](https://wandb.ai/nvidia/sna-specdec/runs/zikaotsf) |
| DFlash B16 K13 | COMPLETED, exit 0 | [2swwrr3r](https://wandb.ai/nvidia/sna-specdec/runs/2swwrr3r) |
| DSpark B16 K11 | COMPLETED, exit 0 | [86ikakze](https://wandb.ai/nvidia/sna-specdec/runs/86ikakze) |
| DSpark B16 K13 | COMPLETED, exit 0 | [yh93soqf](https://wandb.ai/nvidia/sna-specdec/runs/yh93soqf) |

All nine gates completed successfully and released their 20-step dependencies.
At the September 17 07:33 PDT receipt, jobs `2843345`, `2843349`, `2843353`,
`2843357`, `2843361`, `2843365`, `2843369`, and `2843373` were running in
parallel. Job `2843377` was the only full run still waiting for resources.

### Ptyche live 20-step W&B runs

| Arm | Job | W&B |
|---|---:|---|
| Baseline, no SpecDec | 2843345, COMPLETED, exit 0, 20/20 steps, 02:07:34 | [c7t0r1x8](https://wandb.ai/nvidia/sna-specdec/runs/c7t0r1x8) |
| DFlash B8 K5 | 2843349, FAILED after 12 steps; Ray 95% host-memory threshold | [hp6au1xv](https://wandb.ai/nvidia/sna-specdec/runs/hp6au1xv) |
| DFlash B8 K7 | 2843353, TIMEOUT after 17 completed steps | [2dajlyrg](https://wandb.ai/nvidia/sna-specdec/runs/2dajlyrg) |
| DSpark B8 K5 | 2843357, cancelled after 9 completed steps | [judua89b](https://wandb.ai/nvidia/sna-specdec/runs/judua89b) |
| DSpark B8 K7 | 2843361, TIMEOUT after 15 completed steps | [dcayf4kd](https://wandb.ai/nvidia/sna-specdec/runs/dcayf4kd) |
| DFlash B16 K11 | 2843365, cancelled after 9 completed steps | [oja1zxwa](https://wandb.ai/nvidia/sna-specdec/runs/oja1zxwa) |
| DFlash B16 K13 | 2843369, cancelled after 10 completed steps | [u1mz5th2](https://wandb.ai/nvidia/sna-specdec/runs/u1mz5th2) |
| DSpark B16 K11 | 2843373, cancelled after 9 completed steps | [pw2gcbp5](https://wandb.ai/nvidia/sna-specdec/runs/pw2gcbp5) |
| DSpark B16 K13 | 2843377, TIMEOUT after 9 completed steps | [c9c3kdgy](https://wandb.ai/nvidia/sna-specdec/runs/c9c3kdgy) |

Job `2843349` reached 12 completed steps, then Ray's 95% host-memory monitor
killed policy workers during refit on node `10.52.97.52`. The measured usage
was 908,306,677,760 bytes versus a 908,287,016,960-byte threshold, a 19.7 MB
overage with roughly 44.6 GiB of physical memory still free. A matched recovery
uses the still-enabled Ray monitor with a 98% threshold; it does not disable
memory protection or change the model/recipe configuration.

- DFlash B8 K5 recovery: job `2843994`, started on `36x2-a01r` after passing
  `sbatch --test-only`; W&B
  [syc0oid7](https://wandb.ai/nvidia/sna-specdec/runs/syc0oid7); source revision
  `6c5918598f78dbd9c51faeead30ab3cacaa4cbce`.
  Its pending allocation was extended from three to five hours.
- The recovery initialized all 16 nodes and completed eight steps without an
  OOM. It then stopped making progress in Step 9 generation: the driver log
  remained unchanged for more than 25 minutes and all 16 nodes reported 0%
  GPU utilization. Job `2843994` was cancelled at 11:38 PDT to release the
  idle allocation. This is a distinct generation stall, not a repeat of the
  original Ray host-memory kill.
- Recovery artifacts:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-rp25-perf-20260917/Qwen3-235B-DFlashK5-B8-20step-20260917T153815Z`.
- A clean DFlash B8 K5 replacement was submitted from revision
  `db70970e005a340d0ee7fbeb2fef378a41dcfe00` after `sbatch --test-only`:
  job `2844719`, artifacts
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q235-rp25-perf-20260917/Qwen3-235B-DFlashK5-B8-20step-20260917T183753Z`.
  The job started immediately and initialized W&B run
  [v8tubgyj](https://wandb.ai/nvidia/sna-specdec/runs/v8tubgyj); its first five
  minutes showed active checkpoint loading with no runtime traceback or OOM.
  The optional pending K11/K13 jobs were temporarily held so this required K5
  replacement can receive the released allocation first.

### Ptyche five-hour replacement matrix

The original three-hour allocations were too short for the official recipe's
long-tail validation and refit phases. The following independent five-hour
replacements were submitted from the clean worktree
`/home/sna/nemorl-q235-specdec-matrix-v2-20260917` at revision
`18bc46384d519785512683bafbd5a46402f78f4a`. Every submission first passed
`sbatch --test-only`; none is gated on another replacement job.

| Arm | Replacement job | State at receipt |
|---|---:|---|
| DFlash B8 K7 | 2844506 | running; [gcjrbhue](https://wandb.ai/nvidia/sna-specdec/runs/gcjrbhue) |
| DSpark B8 K5 | 2844508 | running; [cg0itgx5](https://wandb.ai/nvidia/sna-specdec/runs/cg0itgx5) |
| DSpark B8 K7 | 2844510 | running; [ipgtb07d](https://wandb.ai/nvidia/sna-specdec/runs/ipgtb07d) |
| DFlash B16 K11 | 2844512 | running; [5a7yr77j](https://wandb.ai/nvidia/sna-specdec/runs/5a7yr77j) |
| DFlash B16 K13 | 2844514 | pending; estimated 11:59 PDT |
| DSpark B16 K11 | 2844516 | pending; estimated 12:16 PDT |
| DSpark B16 K13 | 2844518 | pending; estimated 12:33 PDT |

Ptyche account `coreai_dlalgo_llm` had a measured user FairShare of `0.592`
before submission. The Lyris replacements remain queued as an independent
site-matched fallback; results from different sites will not share a baseline.
The first four replacements started together at 10:43 PDT and passed their
first 11 minutes without a traceback or OOM. The remaining three are held by
Ptyche's `MaxNodeRunMinsPerUser` limit and will become eligible as running jobs
release node-minutes.

All four active replacements completed their first step and entered Step 2.
Their first-step generation speedups versus the matched Ptyche gate baseline
are 1.33x for DFlash K7, 1.72x for DSpark K5, 1.66x for DSpark K7, and 1.04x
for DFlash K11. Reward, mean generation length, approximate entropy, and
generation KL remain in the matched baseline range, confirming that the
replacement jobs reproduce the gated configurations.

### Deep-refit correctness matrix, September 18

The final 20-step K3/K5/K7 matrix was submitted from the clean Ptyche worktree
`/home/sna/nemorl-q235-deep-refit-20260917` at revision
`ef08e3e114055828723d67ae94489ab754774b83`. This revision preserves a frozen
drafter across level-2 target refit, restores it after validation-triggered full
wakeups, recognizes both vLLM's native `drafter` owner and the DFlash/DSpark
`speculator` owner, and avoids copying large scratch buffers during refit.
Every arm passed `sbatch --test-only`; the jobs are independent.

| Arm | Job | State at submission |
|---|---:|---|
| DFlash B8 K3 | 2851882 | pending |
| DFlash B8 K5 | 2851884 | pending |
| DFlash B8 K7 | 2851886 | pending |
| DSpark B8 K3 | 2851888 | pending |
| DSpark B8 K5 | 2851890 | pending |
| DSpark B8 K7 | 2851892 | pending |
| EAGLE-3 K3 | 2851894 | pending |
| EAGLE-3 K5 | 2851896 | pending |
| EAGLE-3 K7 | 2851898 | pending |

The earlier DFlash K7 deep-refit run is diagnostic-only. Its Step 11 output was
corrupted immediately after validation because the no-refit training path did
not restore the frozen drafter after a full wakeup; Step 12 recovered after the
next target refit. The matrix above includes the validation-wakeup fix and is
the only cohort that will be used for final Steps 3--20 rankings.
