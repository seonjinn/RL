# Lyris Qwen3-235B SWE-Bench OSL32K Batch Sweep - 2026-06-12

## Why this sweep was added

Before this launch, Qwen3-235B-A22B OSL32K coverage was only the SWE-Bench
Verified bs1 pilot. That pilot showed suffix decoding was very strong, but it
did not answer the batch-size question for SWE-Bench or SWE-Bench-Verified.

## Submitted matrix

Submitted 50 jobs through
`experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_batch_sweep_20260612.sh`.

Tracker:
`latest_lyris_qwen235b_swebench_osl32k_batch_sweep_20260612_jobs.csv`

Refresh artifacts:

- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_status_20260612.md`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_live_progress_20260612.csv`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv`

| Dataset | Model | Batches | Methods |
|---|---|---:|---|
| SWE-Bench full | Qwen3-235B-A22B | 2, 4, 8, 16, 32 | baseline, suffix K32, PARD K5, native PARD-2 K1, Eagle-3 K3 |
| SWE-Bench-Verified | Qwen3-235B-A22B | 2, 4, 8, 16, 32 | baseline, suffix K32, PARD K5, native PARD-2 K1, Eagle-3 K3 |

Common runtime setup:

- Lyris `gb200`, account `coreai_dlalgo_llm`
- `MODEL=Qwen/Qwen3-235B-A22B`
- `TP=4`, `PP=1`, `KV_CACHE_DTYPE=fp8`
- `ISL=4096`, `OSL=32768`, `MAX_MODEL_LEN=40960`
- `max_num_batched_tokens=max(40960, ISL * batch_size)`
- PARD draft: `amd/PARD-Qwen3-0.6B`, K5, draft TP4
- Native PARD-2 draft: `amd/PARD2-Qwen3-8B`, K1, draft TP4, `method=pard2`
- Eagle-3 draft: `nvidia/Qwen3-235B-A22B-Eagle3`, K3, draft TP4

## Submission status

Job ID range: `2109517` to `2109580`.

The post-submit tracker check found all 50 expected rows:

| Dimension | Count |
|---|---:|
| Rows | 50 |
| SWE-Bench full | 25 |
| SWE-Bench-Verified | 25 |
| Each batch size | 10 |
| Each method | 10 |

At the first refresh, `2026-06-12T23:46:37+02:00`:

| State | Count |
|---|---:|
| RUNNING | 29 |
| PENDING | 21 |

At the second refresh, `2026-06-12T23:49:00+02:00`:

| State | Count |
|---|---:|
| RUNNING | 45 |
| PENDING | 5 |

At the latest refresh, `2026-06-13T00:01:14+02:00`:

| State | Count |
|---|---:|
| RUNNING | 50 |

No final `breakdown.json` metrics were available yet at this refresh. All
submitted Qwen3-235B OSL32K batch jobs are now running, with no readable log
errors in the live-progress scan.

Live log-tail signals are not final metrics, but they currently show suffix K32
as the strongest Qwen3-235B candidate on many larger-batch cells. Native PARD-2
K1 remains weak on acceptance length, especially on SWE-Bench-Verified, while
PARD K5 and Eagle-3 K3 are generally positive but below suffix in live
throughput.

## Existing Qwen3-235B OSL32K bs1 pilot readout

These are final SWE-Bench-Verified bs1 OSL32K rows from the earlier pilot and
should be treated as directional until the new batch sweep completes:

| Method | tok/s/GPU | Speedup | Acceptance |
|---|---:|---:|---:|
| baseline | 2.09 | 1.000x |  |
| suffix K32 | 12.60 | 6.042x | 82.63% |
| suffix K8 | 11.65 | 5.586x | 86.35% |
| Eagle-3 K3 | 5.19 | 2.488x | 54.28% |
| PARD K5 | 3.19 | 1.529x | 18.39% |
| native PARD-2 K1 | 1.88 | 0.899x | 11.53% |

The bs1 result says suffix is the strongest Qwen3-235B candidate so far.
Eagle-3 is positive but smaller, PARD is positive but much smaller, and native
PARD-2 needs improvement before it is competitive on this setup.
