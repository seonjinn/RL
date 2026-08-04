# BF16 to NVFP4 Rollout Experiment Plan

## Objective

Produce two-step evidence that a plain BF16 Qwen3-30B-A3B Megatron policy can
refit real ModelOpt W4A16 and calibrated W4A4 vLLM rollout engines through both
legacy IPC and NCCL-Reshard without changing the trainer's storage format.

## Invariants

- Run an exact committed revision in a container on GCP-NRT B200.
- Keep `policy.quant_cfg=null`, BF16 policy storage, and the plain
  `MegatronPolicyWorker`.
- Run `grpo.max_num_steps=2` and `checkpointing.enabled=false`.
- Use `loss_fn.force_on_policy_ratio=false` and importance-sampling correction.
- Use the real ModelOpt generation worker and the expected W4A16 or W4A4 method.
- Require a provenance-checked W4A4 calibration artifact.
- Use a `batch` allocation with a four-hour limit and unique log/W&B names.
- Mount `/lustre`, export `WANDB_API_KEY`, and use a fresh
  `CODE_SNAPSHOT_DIRNAME` for every campaign.

## Pre-run gates

1. Run the recipe tests and focused Tasks 1-7 unit suites in the target
   container.
2. Run `bash -n` and static contract checks on both smoke scripts.
3. Run `git diff --check`, commit with DCO, push, and record the full commit SHA.
4. Run `tools/launch` with `DRYRUN=2` under a fresh campaign snapshot and
   inspect each generated `continue.sh` without executing it.
5. Generate and reopen the W4A4 artifact from the exact W4A4 snapshot. Record
   revision, quant config, dataset, sample count, sequence length, seed, path,
   size, and SHA256.
6. Confirm scheduler and application segment sizes are both 2 for legacy and
   both 1 for NCCL-Reshard before submitting committed code.

## Run matrix

| ID | Quant mode | Transport | Required environment | Expected topology |
| --- | --- | --- | --- | --- |
| `w4a16-legacy` | W4A16 | `null` | `SCHEDULER_SEGMENT_SIZE=2` | 2 B200 nodes x 8 GPUs, colocated, Megatron EP16 |
| `w4a4-legacy` | W4A4 | `null` | segment 2 and calibration artifact | 2 B200 nodes x 8 GPUs, colocated, Megatron EP16 |
| `w4a16-nccl` | W4A16 | `nccl_reshard` | `SCHEDULER_SEGMENT_SIZE=1` | 1 B200 train node + 1 B200 generation node, Megatron EP8 |
| `w4a4-nccl` | W4A4 | `nccl_reshard` | segment 1 and calibration artifact | 1 B200 train node + 1 B200 generation node, Megatron EP8 |

Use the exact launch commands in `README.md`. Submit one matrix entry per job so
job identity, logs, and W&B metadata remain unambiguous.

## Monitoring

For each submitted job:

1. Record job ID, full commit SHA, container, allocation account, config path,
   environment, Hydra overrides, log path, and W&B URL.
2. Monitor continuously for the first five minutes. Cancel and record the exact
   failure if setup selects a QARL/fake-quant worker, W4A4 lacks calibration,
   NCCL placement is colocated, or an exception appears.
3. After completion, preserve `run.log`, `metrics.json`, TensorBoard events, the
   code snapshot path, and the SLURM output.

## Validation

The script is the minimum pass gate. Independently confirm:

- expected real ModelOpt NVFP4 detection and quantization method;
- exactly two `timing/train/prepare_for_generation/transfer_and_update_weights`
  entries and two `train/loss` entries;
- no incomplete reload, incomplete manifest, mixed manifest, NaN/infinity,
  refit exception, or NCCL agreement error;
- finite reward, generation KL, token multiplication probability error, refit
  time, generation time, and total step time for both steps.

Initialization without both completed steps is a failure.

## Comparison and report

Create `report.md` only after runs exist. For each matrix entry, report status,
job metadata, artifact identity, reward, KL, importance ratio, generation time,
refit time, and end-to-end step time. Compare legacy against NCCL within each
quant mode and W4A16 against W4A4 within each transport. Record failures exactly
and state any remaining requirement for a longer accuracy run.

## Current status

Static artifacts only. No calibration job or smoke run has been submitted.
