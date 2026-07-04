# vLLM 0.24 Long-Context SpecDec Design

## Goal

Measure Qwen3-8B speculative-decoding performance at long output lengths while
keeping every speedup comparison matched to a baseline with the same context,
domain, sampling parameters, and runtime configuration.

## Context Profiles

| Profile | ISL | OSL | Total sequence | RoPE configuration |
|---|---:|---:|---:|---|
| Native 32K | 4,096 | 32,768 | 36,864 | Checkpoint default |
| Long 64K | 4,096 | 65,536 | 69,632 | YaRN factor 4 |
| Supported 128K | 4,096 | 126,976 | 131,072 | YaRN factor 4 |

The 128K profile means a supported total context of 131,072 tokens. It does
not request OSL 131,072 because ISL 4,096 plus that output would exceed Qwen3's
documented YaRN range.

## Runtime Contract

- Use the pinned vLLM 0.24.0 GB200 image and Qwen3-8B checkpoints.
- Keep Math/SWE, temperature 0/1, top_p 1.0, exact fixed output length,
  prefix caching, chunked prefill, Triton attention, eager execution, TP1, and
  the four-GPU throughput denominator used by the 32K replay.
- Compare baseline, Suffix K32, PARD K12, PARD-2 K15, and DFlash K15.
- Start 64K and 128K with BS1 and no benchmark-level warmup. Expand batch size
  only after the BS1 jobs establish wall-time and KV-cache headroom.
- Run one batch size per long-context job so a timeout cannot discard an
  earlier completed batch-size row.

## Model Views

Create lightweight model directories on Lustre. Every file except
`config.json` is an absolute symlink to the pinned Hugging Face snapshot.
`config.json` records:

```json
{
  "max_position_embeddings": 131072,
  "rope_parameters": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 1000000
  }
}
```

The target, PARD, PARD-2, DFlash, and DFlare checkpoints receive independent
views. This is required because applying a target-only `hf_overrides` object
does not update vLLM's separately loaded speculative draft model.

## Submission And Reporting

The long-context wrapper stages both profiles under separate result roots and
submits matched baseline/method jobs after `sbatch --test-only` succeeds.
Result JSON is written after each completed batch size. Reports keep native
32K and YaRN 64K/128K sections separate and never calculate speedup without a
matching baseline row.

AngelSlim DFlare remains separate from vLLM. Its Transformers-native runner
runs an autoregressive baseline and SpecDec serially, so 64K/128K cannot fit
the Lyris five-hour wall limit at the measured decode rate. It is not part of
the first long-context launch.

## Validation

- Unit-test model-view materialization and dry-run rendering.
- Verify the 64K and 128K commands contain exact ISL, OSL, max model length,
  batch size, YaRN view paths, and SLURM segment settings.
- Run shell syntax checks, pytest, Pyright, remote `--test-only`, then submit.
- Monitor all submitted jobs for at least five minutes and inspect early logs.
