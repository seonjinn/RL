# Qwen3-8B GBS128 / 32K frozen gate

Submitted on 2026-09-15, independently of the completed eleven-arm 200-step
GBS8 / OSL1024 online-cadence study. This is a three-step correctness and
rollout-share probe, not a completed performance result or online-training study.

| Arm | SLURM job | W&B run ID after initialization |
|---|---:|---|
| Baseline, no SpecDec | 7163384 | q8rp44-baseline-long-context-7163384 |
| DFlash K5 frozen | 7163385 | q8rp44-dflash-frozen-long-context-7163385 |
| DSpark K5 frozen | 7163386 | q8rp44-dspark-frozen-long-context-7163386 |

All jobs use `nemotron_n3_post`, `batch`, three-hour limits, one exclusive
four-GB200 node, and no dependency. Test-only checks passed before submission.
Planning IDs 7163377–7163379 are not submitted jobs. At 09:45 UTC all three
jobs were RUNNING for 5m34s. Initial five-minute monitoring passed without a
terminal failure; pinned dependency builds were still progressing. No GPU
training step or 32K memory success is yet established.

## Matched configuration

- Target: Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- New Draft: rp25-44000 B8 exports, same lineage as the 200-step cohort.
- DAPOMath17K; 16 prompts × 8 generations = GBS128.
- Input cap 2048; output cap 30720; total/model context cap 32768.
- Training TP2 / PP1 / CP1; microbatch1; logprob batch1, chunk256.
- Activation checkpointing enabled; sequence packing remains disabled.
- vLLM TP1, `max_num_seqs=8` for **all three arms**, memory utilization0.7.
- Existing PIECEWISE/eager-compilation graph settings and buckets retained.
- Frozen means `policy.draft.enabled=false`; SpecDec remains enabled for draft arms.
- Checkpointing and cadence-runtime receipts disabled for this short frozen probe.
- Reward shaping remains disabled. No minimum generation length is forced.
- W&B project `nvidia/sna-specdec`, group
  `q8-new-draft-gbs128-32k-frozen-gate-20260915`.

Runtime source `2628434ba6f76c0a26049ce5b0de4b913a359f77`; bundle SHA256
`ff60631ccb252207406bfebd15b358cd4c99aedb5243f678291ae167f8359cda`.
Same Aug18 nightly image and vLLM0.25.1 as the completed Q8 cohort. Source and
dependencies are staged into job-local scratch using the existing launcher.

Result root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/long-context-2628434ba-20260915/ARM/`

## Validation before expansion

Local renderer/launcher tests: 13 passed. Full inherited config resolves for all
three arms with correct GBS, length caps, dataset and disabled draft training.
Three GPU steps must still complete without OOM. Verify CUDA Graph capture and
dispatch, generation/logprob/training/refit timings, actual generated lengths,
truncation, reward, entropy, and generation/policy KL before a 20-step expansion.
An upper length cap does not ensure long outputs. Do not label generation the
bottleneck until measured; do not infer performance from allocation walltime.

Separate unresolved issue: the prior online resume checks failed on schedule
state-version validation. These new frozen gates do not validate or fix resume.

## Terminal outcome and replacement

All three GBS128 jobs FAILED before training, after about46minutes. The policy
initializer rejected positive `logprob_chunk_size=256` with missing
`policy.megatron_cfg.defer_fp32_logits=true`. This was a configuration assertion,
not measured CUDA OOM. No performance result may be reported for these jobs.

The user subsequently approved GBS512 /32K. See `GBS512_32K_20260915.md` for the
replacement contract; old submission IDs and immutable sources remain preserved.
