# DSpark SWE Rollout-Only CUDA Graph Benchmark Design

## Goal

Measure Qwen3-235B DSpark V1/V2 rollout-only performance on the same
multi-turn SWE agent workload used by the canonical DFlash comparison, with
CUDA Graph enabled, and report speedup against both non-speculative decoding
and DFlash.

## Benchmark Cohort

The benchmark reuses the canonical NemoGym SWE2 rollout-only contract:

- target: `Qwen/Qwen3-235B-A22B-Thinking-2507`;
- dataset: `data/swe2/val-mini3.jsonl`;
- examples: three unseen Astropy instances `12907`, `13236`, and `13398`;
- one generation per prompt;
- natural-EOS multi-turn agent execution with at most eight turns;
- concurrency: eight;
- `max_new_tokens=131072`;
- `max_model_len=131072`;
- `temperature=1.0`, `top_p=1.0`, and seed `42`;
- one Lyris GB200 node with four GPUs;
- target TP4, PP1, EP1, BF16;
- vLLM 0.25.1.

This workload has variable input and output lengths. It must not be described
as a fixed-ISL/fixed-OSL synthetic benchmark.

## CUDA Graph Contract

All measured variants use CUDA Graph. `enforce_eager` is forbidden for both
smoke and production measurements. The launcher must preserve the canonical
FULL graph configuration and capture sizes:

```text
FULL: [6, 12, 24, 48, 96]
```

Graph capture and engine warmup occur before the measured rollout window.
Baseline, DFlash, and DSpark use the same target-engine graph settings.
Compilation or capture failures are reported as failures rather than silently
falling back to eager mode.

## Variants

The primary matrix contains:

| Label | Checkpoint | Method | Draft length |
|---|---|---|---:|
| baseline | none | non-speculative | 0 |
| DFlash V1 | `dflash_235bthink_main_v1` final release checkpoint | DFlash | K5 |
| DFlash V2 | `dflash_235bthink_v2/epoch1_end` | DFlash | K5 |
| DSpark B8 V1 | `dspark_235b_v3_b8/epoch0_end` | DSpark | K8 |
| DSpark B8 V2 | `dspark_235b_v3_b8/epoch1_end` | DSpark | K8 |
| DSpark B16 V1 | `dspark_235b_v3_b16/epoch0_end` | DSpark | K16 |
| DSpark B16 V2 | `dspark_235b_v3_b16/epoch1_end` | DSpark | K16 |

The DSpark checkpoints are the existing v2mix-trained checkpoints. The
general-only checkpoints are outside this matrix and require a separately
labeled ablation.

DFlash keeps its established K5 setting while DSpark uses its trained block
length. The report therefore compares deployable method configurations, not
equal-K algorithm cost.

## Runtime Compatibility

The pinned vLLM 0.25.1 runtime natively supports `dflash` and `dspark`.
DSpark checkpoint configuration resolves the verifier model, draft method,
block size, reduced vocabulary, Markov head, and bonus-anchor convention.
No Speculators runtime import, adapter, remote-code trust, or vLLM patch is
required.

Serving must use native vLLM DSpark inference. The Speculators
`scripts/launch_vllm.py` entry point is prohibited because it launches hidden
state extraction for training rather than rollout inference.

## Artifact Transfer

DSpark checkpoints currently reside on AWS-DFW. They are staged to Lyris
through the checksum-verifying PBSS/rclone data-mover workflow. Each staged
checkpoint must include:

- `config.json`;
- `config.py`;
- `model.safetensors`;
- a provenance manifest containing source path, source commit, file sizes, and
  SHA256 checksums.

The Lyris benchmark refuses to start if a staged checksum differs from the
manifest.

## Execution Gates

1. Validate the canonical baseline/DFlash provenance and launcher contract.
2. Validate source checkpoint symlinks and checkpoint file checksums on AWS.
3. Transfer and checksum-verify DSpark artifacts on Lyris.
4. Run a CUDA-Graph-enabled DSpark B8 V1 smoke with one SWE trajectory.
5. Confirm target and draft weights load, graph capture completes, speculative
   metrics are active, output is non-empty, and no eager fallback is present.
6. Submit the full matrix only after the smoke succeeds.
7. Monitor each submitted job for at least five minutes and inspect fatal log
   patterns.

## Metrics and Validity

For each completed variant, record:

- generation throughput in output tokens per second and per GPU;
- speedup versus the matched non-speculative baseline;
- speedup versus DFlash V1 and DFlash V2;
- generation wall time;
- runtime acceptance rate;
- mean accepted length;
- total generated tokens and completed trajectories;
- graph mode and capture sizes;
- job ID, runtime commit, checkpoint checksum, and result path.

Natural-EOS sampling can produce different token totals at temperature 1.0.
Throughput is valid only when all three assigned trajectories complete without
engine or environment errors. Token-count differences must remain visible in
the report; they must not be presented as identical-work latency speedup.

The existing canonical reference is:

| Variant | Throughput | Baseline-relative speedup |
|---|---:|---:|
| non-spec baseline, job `2451569` | 87 tok/s | 1.00x |
| DFlash V1 K5, job `2451570` | 116 tok/s | 1.33x |

These rows remain historical references. Newly launched rows are compared
against a newly matched baseline unless provenance proves every runtime,
dataset, and graph field is identical.

## Reporting

The experiment report and the existing DFlash training HTML page will record:

- the exact benchmark contract;
- checkpoint and runtime provenance;
- CUDA Graph evidence;
- final and partial result tables;
- failed attempts and their root causes;
- whether each speedup is throughput-based or identical-work latency-based.

No result is labeled final until the matched baseline and target row both
complete and expose valid speculative-decoding metrics.
