# Lyris Qwen3-235B SWE-Bench - 2026-06-12

Current generated status, live rows, and final metrics are in
`docs/lyris_qwen235b_suffix_status_20260612.md`.

## Current State

The Qwen3-235B-A22B SWE-Bench Verified OSL1K and OSL32K pilots are complete.
The refreshed tracker now has 37 jobs total:

- Completed: 29
- Failed: 2
- Running: 6

The failed native PARD-2 probes have successful retry coverage. The newly
running jobs are the OSL64K pilot submitted at 11:43-11:44 CEST. The
13:41 CEST status check still had all six OSL64K pilot jobs running.

## OSL64K Pilot

These jobs use Lyris `gb200`, account `coreai_dlalgo_llm`, target
`Qwen/Qwen3-235B-A22B`, TP4, fp8 KV cache, one SWE-Bench Verified prompt,
`ISL=4096`, `OSL=65536`, `MAX_MODEL_LEN=73728`, and
`MAX_NUM_BATCHED_TOKENS=73728`.

| Job | Method | Tracker |
| ---: | --- | --- |
| `2104334` | baseline | `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_suffix_k32_pilot_20260612_jobs.txt` |
| `2104335` | suffix K32 | `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_suffix_k32_pilot_20260612_jobs.txt` |
| `2104336` | suffix K8 | `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_suffix_k8_pilot_20260612_jobs.txt` |
| `2104337` | PARD K5 | `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_pard_k5_pilot_20260612_jobs.txt` |
| `2104338` | Eagle-3 K3 | `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_eagle3_k3_pilot_20260612_jobs.txt` |
| `2104340` | native PARD-2 K1 | `latest_lyris_qwen235b_a22b_swebench_verified_osl64k_pard2_native_k1_pilot_20260612_jobs.txt` |

All six are now `RUNNING`. Current live partials, all with 0 completed rows so
far:

| Job | Method | Node | Live gen tok/s | Live acceptance |
| ---: | --- | --- | ---: | ---: |
| `2104334` | baseline | `lyris0162` | 8.2 |  |
| `2104335` | suffix K32 | `lyris0179` | 106.7 | 100.0% |
| `2104336` | suffix K8 | `lyris0006` | 60.4 | 100.0% |
| `2104337` | PARD K5 | `lyris0007` | 5.3 | 0.0% |
| `2104338` | Eagle-3 K3 | `lyris0009` | 4.4 | 0.8% |
| `2104340` | native PARD-2 K1 | `lyris0013` | 5.1 | 6.1% |

No OSL64K final metrics exist yet. The live rows are volatile and all
suffix candidates now sit far above the live baseline, while PARD, Eagle-3, and
native PARD-2 are still below it. This is still a log-tail signal rather than a
final `breakdown.json` measurement.

## Reviewed Configs

The OSL64K setup mirrors the successful OSL32K configuration:

- Suffix: `{"method": "suffix", "num_speculative_tokens": 32}` and K8 for the lower-K check.
- PARD: `amd/PARD-Qwen3-0.6B`, K5, draft TP4, `parallel_drafting=true`.
- Eagle-3: `nvidia/Qwen3-235B-A22B-Eagle3`, K3, draft TP4.
- Native PARD-2: `method=pard2`, `amd/PARD2-Qwen3-8B`, K1, draft TP4, `parallel_drafting=true`.

## Final OSL32K Readout

| Method | BS | tok/s/GPU | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1 | 2.09 | 1.000x |  |
| suffix K32 | 1 | 12.60 | 6.042x | 82.63% |
| suffix K8 | 1 | 11.65 | 5.586x | 86.35% |
| Eagle-3 K3 | 1 | 5.19 | 2.488x | 54.28% |
| PARD K5 | 1 | 3.19 | 1.529x | 18.39% |
| native PARD-2 K1 | 1 | 1.88 | 0.899x | 11.53% |

Interpretation: OSL32K strongly favors suffix decoding. Eagle-3 and PARD are
positive but far behind suffix. Native PARD-2 K1 has the best native PARD-2
acceptance at OSL32K, but it is still below baseline.
