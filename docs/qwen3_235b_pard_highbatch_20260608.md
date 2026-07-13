# Qwen3-235B PARD High-Batch Standalone Gate

Date: 2026-06-08

## Purpose

Extend the Qwen3-235B OpenMath standalone PARD sweep beyond batch 32 to batch
64 and 128. This checks whether the apparent static-K optimum changes when the
vLLM engine is more saturated.

## Submitted Jobs

All jobs use the same OpenMath prompt source and vLLM standalone wall-clock
gate:

- target model: `Qwen/Qwen3-235B-A22B`
- drafter: public `amd/PARD-Qwen3-0.6B`
- prompt file:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/openmath_reasoning_cot_conversations_50k.jsonl`
- `ISL=1024`, `OSL=1024`
- batch sizes: `64 128`
- target TP: `4`
- draft TP: `4` for PARD
- `max_model_len=4096`
- `max_num_seqs=128`
- `max_num_batched_tokens=393216`
- `gpu_memory_utilization=0.90`
- `enforce_eager=false`
- vLLM profiler disabled, custom all-reduce disabled

| Job | Mode | K | Status |
| ---: | --- | ---: | --- |
| `3211529` | baseline | 0 | completed |
| `3212856` | public PARD | 3 | completed |
| `3211531` | public PARD | 5 | completed |
| `3211982` | public PARD | 7 | completed |
| `3212858` | public PARD | 8 | completed |
| `3211532` | public PARD | 9 | completed |

Tracking file:

```text
latest_vllm_qwen235b_pard_openmath_bs64_128_k5_k9_20260608_070934_jobs.txt
latest_vllm_qwen235b_public_pard_openmath_bs64_128_k3_k8_jobs.txt
```

## Prior Batch-32 Context

| K | Batch 32 throughput | Speedup | Acceptance |
| ---: | ---: | ---: | ---: |
| 0 | `484.09 tok/s/GPU` | `1.000x` | n/a |
| 5 | `635.65 tok/s/GPU` | `1.313x` | `45.51%` |
| 7 | `631.05 tok/s/GPU` | `1.304x` | `35.80%` |
| 9 | `651.15 tok/s/GPU` | `1.345x` | `29.98%` |
| 12 | `590.71 tok/s/GPU` | `1.220x` | `22.84%` |

K9 is the best batch-32 point, but it has much weaker acceptance than K5. The
batch-64/128 gate determines whether K9 remains better when the engine is more
saturated, or whether the extra draft work starts losing to K5.

## Results

| Mode | K | Batch | Throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 64 | `807.19 tok/s/GPU` | `1.000x` | n/a |
| baseline | 0 | 128 | `1372.76 tok/s/GPU` | `1.000x` | n/a |
| public PARD | 3 | 64 | `1055.48 tok/s/GPU` | `1.308x` | `57.82%` |
| public PARD | 3 | 128 | `1708.11 tok/s/GPU` | `1.244x` | `58.30%` |
| public PARD | 5 | 64 | `1016.29 tok/s/GPU` | `1.259x` | `45.01%` |
| public PARD | 5 | 128 | `1605.71 tok/s/GPU` | `1.170x` | `44.77%` |
| public PARD | 7 | 64 | `965.73 tok/s/GPU` | `1.196x` | `35.18%` |
| public PARD | 7 | 128 | `1528.92 tok/s/GPU` | `1.114x` | `34.91%` |
| public PARD | 8 | 64 | `933.27 tok/s/GPU` | `1.156x` | `32.33%` |
| public PARD | 8 | 128 | `1429.09 tok/s/GPU` | `1.041x` | `31.44%` |
| public PARD | 9 | 64 | `957.51 tok/s/GPU` | `1.186x` | `28.90%` |
| public PARD | 9 | 128 | `1426.78 tok/s/GPU` | `1.039x` | `28.78%` |

## Interpretation

Batch 64/128 reverses the batch-32 K9 result. K9 was slightly better at batch
32, but at larger batches the extra draft work loses badly. The completed K3
and K8 gap-fill jobs make the high-batch standalone ordering clear:
K3 is best at both batch 64 and 128, K5 is second, and K7/K8/K9 are worse.
K3 also has the best acceptance in this OpenMath `ISL=1024/OSL=1024` gate.

This also explains why a static high K should not be promoted blindly into
NeMo-RL: K9 can win near batch 32 but becomes nearly flat over baseline at
batch 128. For standalone high-batch rollout pressure, prefer K3. For NeMo-RL
GBS512, however, static K5 still wins because the end-to-end engine and
training interaction differ from standalone OpenMath.

## NeMo-RL Follow-Up

The first aggressive NeMo-RL fixed256 gate used K9 because batch-32 OpenMath
had K9 as the best static point. That gate completed as job `3211503`:

| Mode | Job | Step window | Total speedup | Generation speedup | Acceptance |
| --- | ---: | --- | ---: | ---: | ---: |
| public PARD K9 | `3211503` | Step2-5 vs baseline `3210580` Step2-5 | `1.510x` | `2.115x` | `28.60%` |
| public PARD K5 | `3211706` | Step2-5 vs baseline `3210580` Step2-5 | `1.494x` | `2.047x` | `42.19%` |

The speedup is real on fixed256, but the low acceptance agrees with the
batch-64/128 standalone result. K7/K8/K9 should not be promoted as static
high-batch defaults. The matched public PARD K5 NeMo-RL gate completed with
nearly the same fixed256 speedup as K9 and much better acceptance, so K5 became
the better initial NeMo-RL static candidate. A 20-step K5 stability run on the
same GBS256 shape completed as job `3211900`.

That 20-step job is still GBS256, so it should be treated primarily as a
functional/stability check. For a more realistic Qwen3-235B performance gate,
a matched GBS512 pair was submitted next using the same non-colocated TP4
fixed256 shape, with `NUM_PROMPTS=16`, `NUM_GENERATIONS=32`, and
`TRAIN_GLOBAL_BATCH_SIZE=512`:

| Mode | Job | Status | Notes |
| --- | ---: | --- | --- |
| baseline | `3212012` | completed | matched no-SpecDec baseline; Step2-5 avg generation `115.57s`, E2E `8.35 tok/s/GPU` |
| public PARD K5 | `3212013` | failed before Step1 | checkpoint conversion/path-readiness race; no OOM or performance metric |
| public PARD K5 retry | `3212068` | failed before driver | Ray head/GCS startup failure on allocation; no OOM or performance metric |
| public PARD K5 retry2 | `3212209` | completed | Step2-5 vs `3212012`: total `1.810x`, generation `2.285x`, E2E `1.815x`, gen worker `2.287x`, acceptance `43.08%`; no OOM/fatal pattern |
| public PARD K3 | `3212919` | completed | Step2-5 vs `3212012`: total `1.597x`, generation `1.934x`, E2E `1.599x`, gen worker `1.934x`, acceptance `56.64%`; no OOM/fatal pattern |
| dynamic K5cap3 medium16 | `3213606` | completed | Step2-5 vs `3212012`: total `1.565x`, generation `1.907x`, E2E `1.575x`, gen worker `1.908x`, acceptance `56.83%`; selected K3 during dense `requests=32`, K5 only on small tail batches |

GBS512 conclusion: public PARD K5 remains strongly positive on the realistic
Qwen3-235B performance gate. The valid comparison is `3212012` vs `3212209`;
`3212013` and `3212068` were startup/checkpoint failures, not performance or
OOM results. The later K3 and dynamic-K medium16 jobs completed without OOM,
but they were static-K3-like and slower than K5. Use K3 as the standalone
high-batch winner and K5 as the current NeMo-RL GBS512 winner.
