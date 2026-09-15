# Default-concurrency DFlash/DSpark KL controls

## Submitted

User approved two Qwen3-30B-A3B measurements with no `max_num_seqs` override.
Both use 20 steps, DAPOMath17K, GBS 2048, frozen New Draft Base step-44000
B8 checkpoints, K5, BF16, flashinfer_trtllm and FULL_AND_PIECEWISE. Hardware
remains eight nodes with four GB200 GPUs per node. Container and runtime
code are unchanged from the completed S64 cohort.

| Arm | Job ID | Account / partition | Time limit | Status at 09:02 UTC |
|---|---:|---|---|---|
| DFlash K5 default concurrency | 7162675 | nemotron_n3_post / batch_long | 8 hours | PENDING |
| DSpark K5 default concurrency | 7162677 | nemotron_n3_post / batch_long | 8 hours | PENDING |

Both have no dependency. Scheduler start estimates were September 15
10:07:58 UTC for DFlash and 09:56:48 UTC for DSpark; these are changeable
estimates, not reservations. Submission is not evidence of successful
initialization, graph replay, completed training or five-minute running-job
monitoring. That monitoring remains to be done after allocation starts.

The selected account had the highest observed user FairShare (0.824239).
The proven eight-hour measurement budget is retained because the four-hour
batch partition can be too short for this long-context workload. Checkpointing
remains disabled to match the frozen controls; a preempted run is incomplete.

## Exact comparison

Within each method, only `max_num_seqs=64` is removed, apart from run identity
and output paths. The S64 graph buckets are retained:

- DFlash: `[1,2,4,6,8,12,16,24,32,48,64,96,192,384]`.
- DSpark: `[1,2,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,160,192,320,384]`.

`default` is the engine default, not infinite concurrency. Equal sharding of
2048 rollouts across 32 TP1 engines can make S64 nonbinding. Record the
resolved engine limit, realized load and graph fallback before attributing
changes to concurrency. This control does not claim optimized graph coverage
for arbitrary larger batches.

Reference runs: [DFlash S64](https://wandb.ai/nvidia/sna-specdec/runs/8k92djcz),
[DSpark S64](https://wandb.ai/nvidia/sna-specdec/runs/1z42ngg1), and
[no-SpecDec baseline, default concurrency](https://wandb.ai/nvidia/sna-specdec/runs/6wl01vj0).
Collect exact Steps 3–20: Policy KL mean/median/max/spike steps, Generation KL,
probability-ratio diagnostics, reward, entropy, generated length, and
generation/E2E time and throughput. Policy KL is a sampled logprob diagnostic,
not a numerical measure of accuracy loss.

## Provenance

- Branch: `codex/q30-dapo-concurrency-20260912`.
- Source: `df2ede2af9c6cfaeee67a4635e3a7f515731ce5a`.
- Nine launcher/resolved-config tests passed, including the exact S64 delta.
- Shell syntax and whitespace checks passed.
- Source committed and pushed, then remote `git pull --ff-only` completed.
- Both `sbatch --test-only` checks passed before actual submission.
- Synthetic test-only IDs 7162673 and 7162676 are not submitted jobs.
- Copied sbatch files were checked again: no max_num_seqs override, 20 steps,
  GBS 2048, no dependency, and the retained capture lists above.
- Local sbatch and submission receipts: [DFlash](report/default_concurrency_20260915/dflash/)
  and [DSpark](report/default_concurrency_20260915/dspark/).

Durable root:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-dapo-concurrency-20260912/`

- DFlash directory: `Qwen3-30BA3B-DAPO40K-DFlashK5-Sdefault-20step-r4-20260915T090042Z`.
- DSpark directory: `Qwen3-30BA3B-DAPO40K-DSparkK5-Sdefault-20step-r4-20260915T090044Z`.

W&B project: `nvidia/sna-specdec`; group:
`q30-dapo-gbs2048-concurrency-20260912`. New run URLs will be recorded after
W&B initialization; no URL is inferred from a job ID.
