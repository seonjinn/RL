# Qwen3-8B DFlash policy/refit performance matrix

This harness compares fixed and online DFlash K7 training on the exact product
head `4d8a54538d694f81f65bf2b431c5b5ed6a3017ca`.

| Pair | Prompts | Generations/prompt | GBS | Train MBS | Logprob MBS | Replicates | First-arm order |
|---|---:|---:|---:|---:|---:|---:|---|
| `gbs32_mbs1` | 8 | 4 | 32 | 1 | 1 | 3 | fixed, online, fixed |
| `gbs64_mbs1` | 16 | 4 | 64 | 1 | 1 | 3 | fixed, online, fixed |
| `gbs64_mbs2` | 16 | 4 | 64 | 2 | 1 | 3 | fixed, online, fixed |

Each pair runs sequentially in one four-GPU allocation so fixed and online use
the same node. Three paired replicates per topology produce nine pair jobs and
18 W&B runs. The fixed/online/fixed first-arm order balances order effects
within every topology. Every replicate has unique W&B IDs and a unique durable
result directory. The runner disables Nsight profiling and changes only the
arm, GBS, MBS, prompt count, and run metadata. A container-side resolved-config
proof is required before either arm starts.

## Submission

Prepare a clean recursive checkout under `/home`, verify the immutable
container metadata, compare FairShare accounts, and export the variables
required by `submit_matrix.sh`. Forecast all nine allocations first:

```bash
bash research/qwen3_8b_dflash_refit_perf_matrix/submit_matrix.sh --test-only
```

Only after all forecasts are accepted, omit `--test-only` to submit. Monitor
the returned job IDs at intervals of at least 60 seconds for at least five
minutes. The harness itself does not select an account and does not submit any
job during local validation.

## Analysis

Merge the nine durable `wandb-runs.json` files into one manifest containing all
18 runs, preserving each entry's `replicate` field, then run:

```bash
python research/qwen3_8b_dflash_refit_perf_matrix/analyze_wandb.py \
  --manifest /lustre/path/all-wandb-runs.json \
  --output-dir /lustre/path/analysis
```

The analyzer merges W&B history records by `_step` and requires every step from
5 through 49 plus every required metric value. It fails with the missing steps
or metric observations instead of silently changing the window. Generation
throughput is the arithmetic mean of the canonical logged
`performance/generation_tokens_per_sec_per_gpu` values; it is never rebuilt
from tokens and time. The report contains all three paired deltas per topology,
their mean, sample standard deviation, and 95% t-confidence interval. It also
reports mean policy, refit, generation, and E2E time, acceptance, draft loss,
update/refit evidence, and peak allocated memory when a supported W&B metric
exists. A missing peak-memory metric is reported as `n/a`, never inferred.
