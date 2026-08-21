# Qwen3-8B DFlash policy/refit performance matrix

This harness compares fixed and online DFlash K7 training on the exact product
head `4d8a54538d694f81f65bf2b431c5b5ed6a3017ca`.

| Pair | Prompts | Generations/prompt | GBS | Train MBS | Logprob MBS | First arm |
|---|---:|---:|---:|---:|---:|---|
| `gbs32_mbs1` | 8 | 4 | 32 | 1 | 1 | fixed |
| `gbs64_mbs1` | 16 | 4 | 64 | 1 | 1 | online |
| `gbs64_mbs2` | 16 | 4 | 64 | 2 | 1 | fixed |

Each pair runs sequentially in one four-GPU allocation so fixed and online use
the same node. Alternating the first arm reduces, but does not eliminate, order
effects. The runner disables Nsight profiling and changes only the arm, GBS,
MBS, prompt count, and run metadata. A container-side resolved-config proof is
required before either arm starts.

## Submission

Prepare a clean recursive checkout under `/home`, verify the immutable
container metadata, compare FairShare accounts, and export the variables
required by `submit_matrix.sh`. Forecast all three allocations first:

```bash
bash research/qwen3_8b_dflash_refit_perf_matrix/submit_matrix.sh --test-only
```

Only after all forecasts are accepted, omit `--test-only` to submit. Monitor
the returned job IDs at intervals of at least 60 seconds for at least five
minutes. The harness itself does not select an account and does not submit any
job during local validation.

## Analysis

Merge the three durable `wandb-runs.json` files into one manifest containing
all six runs, then run:

```bash
python research/qwen3_8b_dflash_refit_perf_matrix/analyze_wandb.py \
  --manifest /lustre/path/all-wandb-runs.json \
  --output-dir /lustre/path/analysis
```

The analyzer excludes steps 0-4. E2E seconds per sample/token and generation
throughput are ratios of summed values. It reports mean policy, refit, and E2E
time, acceptance, draft loss, update/refit evidence, and peak allocated memory
when a supported W&B metric exists. A missing peak-memory metric is reported as
`n/a`, never inferred.
