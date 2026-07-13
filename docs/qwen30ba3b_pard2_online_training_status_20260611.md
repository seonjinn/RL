# Qwen3-30B-A3B PARD2 Online Training Status - 2026-06-11

## Current setup

- Target: `Qwen/Qwen3-30B-A3B`
- Drafter: local PARD2 CAT checkpoint
- SpecDec: `draft_model`, `parallel_drafting=true`, `K=5`
- Full-GRPO data: `OpenMathInstruct-2`
- Full-GRPO GBS: `512` (`16` prompts x `32` generations)
- Long-output setting: `max_new_tokens=4096`, `min_tokens=2048`, `max_model_len=8192`
- Runtime knobs: `TP1`, `draft_tensor_parallel_size=1`, `max_num_seqs=8`, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.70`, `enforce_eager=true`
- Current Qwen30 long-output jobs use sync on-policy logprob repair mode: vLLM request logprobs are omitted for SpecDec, and GRPO fills behavior logprobs from a fresh policy fprop after rollout. Exact request-logprob mode is available for correctness smoke tests through `PARD2_GRPO_MODE=strict_request_logprobs`, but vLLM V1 may disable SpecDec acceptance when request logprobs are enabled, so that mode should not be used for speedup claims.

## Current NeMo-RL PARD online implementation

- PARD/PARD-2 online training is implemented as a Megatron-attached draft model that uses a causal-LM draft path, not EAGLE hidden-state capture.
- `k_slot` training expands the draft sequence by K and trains hard next-token labels with optional CAT/PARD-2 weighting from `prev_logprobs`.
- PARD defaults `policy.draft.initial_refit=false`, so vLLM keeps the directly loaded HF drafter until an actual online training update is ready.
- Sequence-packing policy training is handled by a correctness-first fallback: the policy can run packed, while the PARD drafter forward uses the original padded `input_ids`. This avoids packed-boundary label mistakes, but it is not yet the final true-packed PARD fast path.
- Context parallelism for PARD k-slot online training is explicitly unsupported for now (`context_parallel_size=1` required).

## Qwen8 PARD2 online canary

Submitted a small Qwen3-8B pair to validate the same PARD2 online training and refit code path while the Qwen30 long-output jobs wait for SLURM priority.

CSV:

- `docs/qwen8_pard2_online_canary_steps_20260611.csv`
- `docs/qwen8_pard2_online_canary_summary_20260611.csv`
- `docs/qwen8_pard2_online_canary_comparison_20260611.csv`
- `docs/qwen8_pard2_online_canary_comparison_20260611.md`

| Job | Mode | Status | Steps | Draft training/refit | Draft loss | Token acceptance | Mean step time |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `3266304` | static PARD2 | completed | `2/2` | disabled | n/a | `29.18%` | `22.63s` |
| `3266305` | online PARD2 | completed | `2/2` | enabled on both steps | `2.0603` mean | `0.00%` | `25.90s` |

Functional result: the online path completed both GRPO steps with `Draft Training Enabled: True`, `Draft Refit This Step: True`, finite draft loss, no logged errors, and generation KL under `0.001`. The comparison result is intentionally not an advantage claim: this tiny two-step Qwen8 canary shows online PARD2 refit is functional, but acceptance was worse than the static canary (`0.00%` vs `29.18%`) and mean step time increased (`25.90s` vs `22.63s`). The running Qwen30 long-output jobs below are the target online-vs-static acceptance comparison.

2026-06-12 update: later public Qwen8 PARD-2 runs show the `0.00%` canary was a bad no-initial-refit/dummy-draft configuration point, not the current working online path. With `policy.draft.initial_refit=true` and `pard_token=151670`, Qwen8 online PARD-2 preserves acceptance. The 50-step OSL1024 matched steps `2-50` comparison is `47.54%` token acceptance for static-equivalent vs `47.16%` for interval-5 online, with interval-5 slightly faster in generation time but slightly slower end-to-end. See `docs/qwen8_pard2_online_initialrefit_diagnosis_20260612.md`.

Latest active Qwen30 long-output jobs are running as of `2026-06-11 21:26 PDT` / `2026-06-12 06:26 CEST`:

| Job | Mode | Queue state | Parsed step coverage |
| --- | --- | --- | --- |
| `3265386` | static PARD2, win2048, long OSL, 20 steps | `RUNNING` | step-2+ summary covers steps `2-19`; `17` completed rows and `1` incomplete row |
| `3265387` | online PARD2 interval 10, train/refit start step 1 | `RUNNING` | step-2+ summary covers steps `2-20`; `18` completed rows and `1` incomplete row |
| `3265388` | online PARD2 interval 10, train/refit start step 10 | `RUNNING` | step-2+ summary covers steps `2-19`; `17` completed rows and `1` incomplete row |
| `3274811` | online PARD2 interval 5, train/refit start step 5, submitted with `coreai_dlalgo_llm` | `RUNNING` | step-2+ summary has step `2` as incomplete; `0` completed rows yet |

Early step-2+ comparison is available, but it is not a final online-training conclusion because all three interval-10 jobs are still running and only `17-18` post-step-1 rows per variant are complete:

CSV:

- `docs/qwen30ba3b_pard2_online_long_output_win2048_steps_20260611.csv`
- `docs/qwen30ba3b_pard2_online_long_output_win2048_step2plus_summary_20260611.csv`
- `docs/qwen30ba3b_pard2_online_long_output_win2048_comparison_20260611.csv`
- `docs/qwen30ba3b_pard2_online_long_output_win2048_comparison_20260611.md`

| Mode | Completed summarized rows | Draft refits in summary | Token acceptance | Mean step time | Step speedup vs static |
| --- | ---: | ---: | ---: | ---: | ---: |
| static PARD2 | `17` | `0` | `48.65%` | `492.38s` | `1.000x` |
| online start step 1 | `18` | `1` with draft loss `1.9980` | `46.39%` | `489.50s` | `1.006x` |
| online start step 10 | `17` | `1` with draft loss `2.0842` | `46.91%` | `496.16s` | `0.992x` |

Current read: online refit is functional in both interval-10 online variants, but the comparison is still active and slightly uneven in completed-row count. Start-step-1 now has a small timing edge over static (`1.006x` step, `1.006x` E2E, `1.008x` generation-worker throughput), but acceptance is still lower (`-2.26 pp`), so this is not yet a clear online-training win. Start-step-10 is also lower acceptance (`-1.74 pp`) and total/E2E throughput remain below static (`0.992x` / `0.990x`). The interval-5/start-step-5 follow-up (`3274811`) has only an incomplete Step 2 row so far.

Related suffix Full-GRPO20 coverage: Qwen3-30B-A3B suffix K32 job `3266990` completed with `GBS=16`, `OSL=1024`, and `max_steps=20`. The final parsed summary has `20/20` completed rows, token acceptance `25.61%`, mean step time `70.72s`, and no latest error.

Additional apples-to-apples long-output Full-GRPO20 coverage was launched at `2026-06-11 21:33 PDT` / `2026-06-12 06:33 CEST` under account `coreai_dlalgo_llm`. These use the same `GBS=512`, `max_new_tokens=4096`, `min_tokens=2048`, `max_model_len=8192`, `max_num_seqs=8`, and 4-node Qwen30BA3B shape as the PARD2 matrix:

| Job | Mode | Queue state at `2026-06-11 21:35 PDT` | Tracker |
| --- | --- | --- | --- |
| `3274971` | no-SpecDec baseline, `policy.draft=false`, `max_steps=20` | `RUNNING` | `latest_qwen30ba3b_baseline_fullgrpo20_longosl_20260612_jobs.txt` |
| `3274972` | suffix decoding K32, long-output Full-GRPO20 | `RUNNING` | `latest_qwen30ba3b_suffix_fullgrpo20_longosl_20260612_jobs.txt` |

The `batch` partition currently enforces a `4:00:00` walltime limit, so these two jobs were submitted with `WALLTIME=04:00:00`. If either times out before 20 steps, the completed step rows can still be parsed, and the wrapper can be rerun with a lower `max_steps` or a partition/QoS that allows longer walltime.

## Apples-to-apples comparison rule

Throughput speedup is only reported when the benchmark shape matches exactly: target model, prompt JSONL, prompt offset/count, loaded/used prompt count, ISL/OSL, batch size, TP/PP, and total GPUs. This prevents mixing Lite, Verified, Full, or partial-slice results. Older Lite numbers without explicit prompt-count metadata remain useful as functional/performance evidence, but the final Lite baseline/PARD/Suffix comparison will use the newly submitted `n64` jobs below.

## SWE-Bench Lite standalone result

Prompt source: `data/swebench_lite_prompts_64.jsonl`

Final apple-to-apple CSV: `docs/swebench_specdec_standalone_summary_20260611.csv`

| Batch size | Baseline tok/s/GPU | PARD K5 tok/s/GPU | PARD speedup | PARD acceptance | Suffix K32 tok/s/GPU | Suffix speedup | Suffix acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 145.99 | 238.75 | 1.635x | 31.66% | 250.90 | 1.719x | 28.70% |
| 16 | 300.30 | 369.46 | 1.230x | 31.95% | 529.24 | 1.762x | 36.09% |

This confirms PARD and Suffix Decoding can be benchmarked on identical SWE-Bench Lite prompts in vLLM standalone mode. This is an acceptance/performance benchmark, not a full SWE-Bench correctness harness.

Completed jobs:

CSV: `docs/qwen30ba3b_swebench_lite_suffix_20260611_arctic_jobs.csv`
CSV: `docs/qwen30ba3b_swebench_lite_baseline_pard_20260611_n64_jobs.csv`

| Job | Mode | Prompt count | ISL | OSL | Batch sizes | Latest state |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `3264086` | suffix K32 | `64` | `4096` | `1024` | `8,16` | completed |
| `3264223` | baseline | `64` | `4096` | `1024` | `8,16` | completed |
| `3264224` | PARD K5 | `64` | `4096` | `1024` | `8,16` | completed |

These three Lite jobs use the same prompt file (`data/swebench_lite_prompts_64.jsonl`), offset `0`, prompt count `64`, ISL/OSL `4096/1024`, and batch sizes `8,16`; only the speculative decoding mode differs.

## SWE-Bench Verified / Full standalone jobs

Prompt files prepared on the remote NeMo-RL checkout:

- `data/swebench_verified_test.parquet`
- `data/swebench_full_test.parquet`
- `data/swebench_verified_prompts_all.jsonl` (`500` prompts)
- `data/swebench_full_test_prompts_all.jsonl` (`2294` prompts)
- `data/swebench_verified_prompts_64_seed0.jsonl` (`64` shuffled prompts)
- `data/swebench_full_test_prompts_64_seed0.jsonl` (`64` shuffled prompts)

The first submitted seed0 64-prompt slice jobs (`3263751`-`3263754`) were cancelled before running because the runner only loaded `max(batch_sizes)` prompts. The runner was updated to support `--prompt-count` and aggregate full chunks over the loaded prompt set.

Submitted all-prompt standalone vLLM jobs, matching the Lite setup (`Qwen/Qwen3-30B-A3B`, `TP1`, `ISL=4096`, `OSL=1024`, batch sizes `8,16`, baseline vs PARD K5):

CSV: `docs/qwen30ba3b_swebench_verified_full_all_standalone_jobs_20260611.csv`

| Job | Dataset | Mode | Prompt count loaded | Prompt count used per batch size | Latest state |
| --- | --- | --- | ---: | ---: | --- |
| `3263824` | SWE-Bench Verified | baseline | `500` | `496` | running |
| `3263825` | SWE-Bench Verified | PARD K5 | `500` | `496` | completed |
| `3263826` | SWE-Bench Full test | baseline | `2294` | `2288` | cancelled before completion |
| `3263827` | SWE-Bench Full test | PARD K5 | `2294` | `2288` | cancelled before completion |

The `prompt_count_used` is lower than loaded because the aggregate runner currently uses only full chunks for each batch size (`8` and `16`) to keep measured batch size stable.

Latest progress snapshot:

- Verified baseline has started and completed at least `bs=8` chunks `1-3/62` at about `132-134` output tokens/sec/GPU.
- Verified PARD completed all loaded prompts. Its result file is `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen30ba3b_swebench_verified_pard_allprompts_isl4096_osl1024_bs8_16_k5_20260611/breakdown.json`. The exact speedup is intentionally not reported yet because the matching Verified baseline job is still running.
- Full original test all-prompt jobs were cancelled because the baseline job was only at `bs=8` chunk `3/286` after about six minutes. At the observed `50-63s/chunk`, it cannot complete both `bs=8` and `bs=16` under the 4-hour walltime.

## SWE-Bench Full original representative slice

Submitted a tractable Full original representative slice using the same short-output settings as Lite/Verified (`ISL=4096`, `OSL=1024`, batch sizes `8,16`, prompt offset `0`, prompt count `256`):

CSV: `docs/qwen30ba3b_swebench_fulltest_slice_standalone_jobs_20260611.csv`

| Job | Dataset | Mode | Prompt offset | Prompt count | Latest state |
| --- | --- | --- | ---: | ---: | --- |
| `3263880` | SWE-Bench Full test | baseline | `0` | `256` | completed |
| `3263881` | SWE-Bench Full test | PARD K5 | `0` | `256` | completed |

This gives an original-SWE-Bench-domain baseline/PARD comparison that should finish within the current walltime. Additional offsets can be added after this first slice completes.

Final parser result:

CSV: `docs/qwen30ba3b_swebench_fulltest_slice_standalone_pard_k5_20260611.summary.csv`

| Batch size | Baseline tok/s/GPU | PARD tok/s/GPU | Speedup | Acceptance | Mean accept length |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 136.45 | 246.28 | 1.805x | 34.87% | 2.74 |
| 16 | 288.49 | 475.50 | 1.648x | 34.78% | 2.74 |

## SWE-Bench Verified long-output jobs

Submitted a small long-output Verified run to test the regime where SpecDec should matter more:

- Dataset: SWE-Bench Verified
- Prompt offset/count: `0` / `8`
- ISL/OSL: `4096` / `16384`
- Batch sizes: `1,2`
- `max_model_len=24576`, `max_num_seqs=2`, `gpu_memory_utilization=0.90`

CSV: `docs/qwen30ba3b_swebench_verified_longosl_standalone_jobs_20260611.csv`

| Job | Mode | Latest state |
| --- | --- | --- |
| `3263894` | baseline | completed |
| `3263895` | PARD K5 | completed |

Final result: baseline job `3263894` completed at `2026-06-11 08:41:24 PDT` and wrote both `bs=1` and `bs=2` aggregate rows. The long-output comparison below now includes final `bs=2` speedups.

## SWE-Bench Suffix Decoding paper-like comparison

Reference: `/Users/sna/Downloads/SpecDec_Benchmarking.pdf`, "An Empirical Study of Speculative Decoding on Software Engineering Tasks", arXiv:2604.26469v3.

Relevant paper settings/results:

- SWE-bench Verified, `Qwen/Qwen3-32B`, vLLM `0.12.0`, greedy decoding, max context `32768`, max generation `1024` tokens per agent turn, 8 concurrent threads, mini-swe-agent.
- Suffix Decoding uses `K=32`.
- Paper Qwen3-32B SWE-bench result: Suffix Decoding `1.66x` speedup with mean acceptance length `3.48`; PLD `1.05x` / `2.92`; Eagle-3 K3 `1.10x` / `1.87`; Eagle-3 K5 `1.53x` / `2.39`.

Submitted Qwen3-30B-A3B suffix K32 standalone jobs to compare against the active baseline/PARD2 jobs under the same prompt/batch setup. This is a throughput/acceptance comparison on prepared SWE prompts, not full mini-swe-agent patch correctness.

CSV: `docs/qwen30ba3b_swebench_suffix_paperlike_20260611_jobs.csv`

| Job | Dataset | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency | Latest state |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `3263988` | SWE-Bench Verified smoke | suffix K32 | `1` | `4096` | `128` | `1` | none | pending priority |
| `3263989` | SWE-Bench Verified | suffix K32 | `500` | `4096` | `1024` | `8,16` | afterok `3263988` | pending dependency |
| `3263990` | SWE-Bench Full original slice | suffix K32 | `256` | `4096` | `1024` | `8,16` | afterok `3263988` | pending dependency |
| `3263991` | SWE-Bench Verified long output | suffix K32 | `8` | `4096` | `16384` | `1,2` | afterok `3263988` | pending dependency |

The first smoke job `3263988` failed because the container was missing `arctic-inference==0.1.1`. The dependent jobs `3263989`-`3263991` were cancelled and replaced with an arctic-enabled run that installs the package into `${BENCH_ROOT}/.container_cache/arctic-inference-0.1.1` via `pip --target` before launching vLLM.

CSV: `docs/qwen30ba3b_swebench_suffix_paperlike_20260611_arctic_jobs.csv`

| Job | Dataset | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency | Latest state |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `3264022` | SWE-Bench Verified smoke | suffix K32 | `1` | `4096` | `128` | `1` | none | completed |
| `3264023` | SWE-Bench Verified | suffix K32 | `500` | `4096` | `1024` | `8,16` | afterok `3264022` | completed |
| `3264024` | SWE-Bench Full original slice | suffix K32 | `256` | `4096` | `1024` | `8,16` | afterok `3264022` | completed |
| `3264025` | SWE-Bench Verified long output | suffix K32 | `8` | `4096` | `16384` | `1,2` | afterok `3264022` | completed |

The replacement smoke completed generation. Parser sanity result is in `docs/qwen30ba3b_swebench_suffix_smoke_20260611_arctic.csv`: `bs=1`, `OSL=128`, output `30.87` tok/s/GPU, acceptance `55.00%`, mean acceptance length `2.43`. vLLM logs that async scheduling is disabled for suffix-based speculative decoding.

Also submitted a Qwen3-32B pilot to compare more directly with the paper's target model:

CSV: `docs/qwen3_32b_swebench_suffix_paperlike_20260611_jobs.csv`

| Job | Model | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `3263997` | Qwen3-32B | baseline | `64` | `4096` | `1024` | `8,16` | none |
| `3263998` | Qwen3-32B | suffix K32 | `64` | `4096` | `1024` | `8,16` | afterok `3263988` |

The first Qwen3-32B suffix job was tied to failed smoke `3263988`, so it was cancelled. The pilot was resubmitted:

CSV: `docs/qwen3_32b_swebench_suffix_paperlike_20260611_arctic_jobs.csv`

| Job | Model | Mode | Prompt count | ISL | OSL | Batch sizes | Dependency |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `3264026` | Qwen3-32B | baseline | `64` | `4096` | `1024` | `8,16` | none |
| `3264027` | Qwen3-32B | suffix K32 | `64` | `4096` | `1024` | `8,16` | afterok `3264022` now released |

Also submitted PLD jobs matching the paper's `N=4`, `K=5` setup:

CSV: `docs/swebench_pld_paperlike_20260611_jobs.csv`

| Job | Model | Dataset | Prompt count | ISL | OSL | Batch sizes | TP |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `3264041` | Qwen3-30B-A3B | SWE-Bench Verified | `500` | `4096` | `1024` | `8,16` | `1` |
| `3264042` | Qwen3-30B-A3B | SWE-Bench Full original slice | `256` | `4096` | `1024` | `8,16` | `1` |
| `3264043` | Qwen3-30B-A3B | SWE-Bench Verified long output | `8` | `4096` | `16384` | `1,2` | `1` |
| `3264044` | Qwen3-32B | SWE-Bench Verified pilot | `64` | `4096` | `1024` | `8,16` | `2` |

## SWE-Bench standalone comparison parsed

New aggregate CSVs:

- `docs/swebench_specdec_standalone_comparison_20260611.csv`
- `docs/swebench_specdec_standalone_summary_20260611.csv`

All rows below are apple-to-apple within each block: same model, prompt JSONL, prompt offset/count, ISL/OSL, TP, and batch size. Throughput is output tokens/sec/GPU.

Qwen3-30B-A3B, SWE-Bench Lite n64, ISL/OSL `4096/1024`:

| Batch | Baseline | PARD K5 | PARD speedup | Suffix K32 | Suffix speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 145.99 | 238.75 | 1.635x | 250.90 | 1.719x |
| 16 | 300.30 | 369.46 | 1.230x | 529.24 | 1.762x |

Qwen3-30B-A3B, SWE-Bench Verified all n500, ISL/OSL `4096/1024`:

| Batch | Baseline | PARD K5 | PARD speedup | Suffix K32 | Suffix speedup | PLD K5/N4 | PLD speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 133.26 | 240.97 | 1.808x | 229.95 | 1.726x | 168.24 | 1.262x |
| 16 | 267.67 | 460.00 | 1.719x | 490.93 | 1.834x | 341.59 | 1.276x |

Qwen3-30B-A3B, SWE-Bench Full original test slice n256, ISL/OSL `4096/1024`:

| Batch | Baseline | PARD K5 | PARD speedup | Suffix K32 | Suffix speedup | PLD K5/N4 | PLD speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 136.45 | 246.28 | 1.805x | 228.67 | 1.676x | 167.99 | 1.231x |
| 16 | 288.49 | 475.50 | 1.648x | 433.14 | 1.501x | 317.32 | 1.100x |

Qwen3-32B, SWE-Bench Verified pilot n64, ISL/OSL `4096/1024`:

| Batch | Baseline | Suffix K32 | Suffix speedup | PLD K5/N4 | PLD speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 68.94 | 99.43 | 1.442x | 76.50 | 1.110x |
| 16 | 140.93 | 222.66 | 1.580x | 149.04 | 1.058x |

Qwen3-30B-A3B, SWE-Bench Verified long-output n8, ISL/OSL `4096/16384`:

| Batch | Baseline | PARD K5 | PARD speedup | Suffix K32 | Suffix speedup | PLD K5/N4 | PLD speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.82 | 56.59 | 3.365x | 121.68 | 7.234x | 62.09 | 3.691x |
| 2 | 33.00 | 105.35 | 3.192x | 273.27 | 8.280x | 90.38 | 2.739x |

## Refresh - 2026-06-12 02:58 CEST

The HSG refresh path is working again after changing `scripts/refresh_pard2_swerl_active_status.sh` to use the existing SSH ControlMaster by default. The latest run wrote:

- `docs/pard2_swerl_active_status_20260611.csv`
- `docs/qwen8_pard2_online_canary_steps_20260611.csv`
- `docs/qwen8_pard2_online_canary_summary_20260611.csv`
- `docs/qwen8_pard2_online_canary_comparison_20260611.csv`
- `docs/qwen8_pard2_online_canary_comparison_20260611.md`

Earlier HSG queue state before the Qwen30 long-output jobs started:

| Job | Mode | State | Reason/start |
| --- | --- | --- | --- |
| `3265386` | Qwen30 static PARD-2 long-output win2048 | `PENDING` | `Priority`, start `2026-06-11T21:42:42` |
| `3265387` | Qwen30 online PARD-2 interval-10 start1 | `PENDING` | `Priority`, start `2026-06-11T21:42:42` |
| `3265388` | Qwen30 online PARD-2 interval-10 start10 | `PENDING` | `Priority`, start `2026-06-11T21:42:42` |
| `3266737` | suffix SWERL generation-only | `PENDING` | `Priority`, start `2026-06-11T22:20:00` |
| `3266990` | Qwen30 suffix K32 Full-GRPO20 smoke | `PENDING` | `Priority`, start `2026-06-11T21:42:42` |

Qwen8 PARD-2 canary result remains a functional validation, not a performance win:

| Mode | Steps | Draft refits | Acceptance | Mean step time | Gen worker tok/s/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| static PARD-2 | `2/2` | `0` | `29.18%` | `22.63s` | `52.63` |
| online PARD-2 | `2/2` | `2` | `0.00%` | `25.90s` | `47.62` |

Interpretation: the online drafter training/refit path executed with finite draft loss (`2.0603`) and no logged step errors, but this two-step Qwen8 canary made acceptance worse and slowed the step. The running Qwen30 long-output jobs are the required evidence before making a broader online-vs-static PARD-2 conclusion.

## Refresh - 2026-06-12 03:01 CEST

Latest refresh command: `scripts/refresh_pard2_swerl_active_status.sh`.

Historical note: at this earlier refresh the Qwen30 PARD-2 static/online long-output jobs had not started, and no Qwen30 ray-driver logs were available to parse. This has been superseded by the current running-job summary near the top of this document.

| Job | Mode | State | Reason |
| --- | --- | --- | --- |
| `3265386` | Qwen30 static PARD-2 long-output win2048 | `PENDING` | `Priority` |
| `3265387` | Qwen30 online PARD-2 interval-10 start1 | `PENDING` | `Priority` |
| `3265388` | Qwen30 online PARD-2 interval-10 start10 | `PENDING` | `Priority` |
| `3266737` | suffix SWERL generation-only | `PENDING` | `Priority` |
| `3266990` | Qwen30 suffix K32 Full-GRPO20 smoke | `PENDING` | `Priority` |

Qwen8 canary metrics are unchanged:

| Mode | Steps | Draft refits | Acceptance | Mean step time | Gen worker tok/s/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| static PARD-2 | `2/2` | `0` | `29.18%` | `22.63s` | `52.63` |
| online PARD-2 | `2/2` | `2` | `0.00%` | `25.90s` | `47.62` |

Current conclusion is still narrow: online PARD-2 training is functionally wired and executes, but the small Qwen8 canary is negative. The Qwen30 long-output online-vs-static run remains the required evidence for the target online-training performance claim.

## SWERL / Nemo-Gym suffix smoke

Suffix Decoding has also been staged for a functional SWERL/Nemo-Gym smoke:

- Added arm64 SWE sandbox formatter to `grpo_qwen3_235b_swe.yaml`.
- Created image-matched data: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_eagle3/data/swerebenchv2_arm64_image_matched_n8_nemogym_hf.jsonl`.
- Added `WANDB_ENABLED` launcher knob for token-free smoke runs.
- Added explicit async SpecDec behavior-logprob modes in `run_grpo_qwen3_235b_swe.sh`. Default async SpecDec mode is now generation-only (`SPECDEC_GRPO_MODE=stop_after_generation`) so the smoke records generation/acceptance metrics and stops before unsafe behavior-logprob reconstruction.
- Cancelled pre-patch job `3266023` before it started because it did not explicitly set the async SpecDec behavior-logprob mode.
- Submitted replacement job `3266737`: Qwen3-235B, total `32x4` GPUs as actor 24 + generation 8, async GRPO, suffix K32, `PPS=8`, `GPP=1`, `GBS=8`, one generation-only step.

The earlier `16x8` submit failed immediately because OCI/HSG exposes `gpu:4` nodes. Job `3266737` is pending priority with latest squeue estimated start `2026-06-12 02:30 PDT`.

## Long-output Full-GRPO old online result

Jobs:

- Static PARD SpecDec: `3263290`
- Online PARD training with previous initial-refit behavior: `3263291`

CSV: `docs/qwen30ba3b_pard2_online_advantage_longosl_oldinitialrefit_step2plus_summary.csv`

Step 2 result:

| Run | Total step | Generation | E2E tok/s/GPU | Weighted acceptance | Mean accept length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 481.66s | 398.41s | 222.23 | 52.64% | 3.63 |
| Online old initial-refit | 484.26s | 400.44s | 222.97 | 45.54% | 3.28 |

The old online run was cancelled after Step 2 because it was not testing the intended condition. The root cause is that `REFIT_DRAFT_WEIGHTS_PENDING` was initialized to online-draft enabled, so Step 1 generation refit overwrote vLLM's directly loaded HF PARD drafter with Megatron-exported PARD weights before any online training benefit could be measured. That made the online Step 1 and Step 2 acceptance lower than static.

## Long-output Full-GRPO static baseline current result

Job `3263290` produced static PARD metrics through Step 17 and then stopped during Step 18 generation. SLURM currently shows the top-level job as `PENDING/Priority`, so it appears to be waiting to resume/requeue rather than producing new log lines.

Latest parsed CSVs:

- `docs/qwen30ba3b_pard2_online_current_steps_20260611.csv`
- `docs/qwen30ba3b_pard2_online_current_step2plus_summary_20260611.csv`

Recent completed static steps:

| Step | Total step | Generation | E2E tok/s/GPU | Weighted acceptance | Mean accept length | Mean generation length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 487.64s | 411.99s | 263.21 | 51.07% | 3.55 | 3907.67 |
| 11 | 471.77s | 396.01s | 261.04 | 52.99% | 3.65 | 3752.48 |
| 12 | 519.78s | 442.42s | 244.21 | 39.35% | 2.97 | 3862.65 |
| 13 | 510.68s | 434.25s | 233.99 | 45.49% | 3.28 | 3623.55 |
| 14 | 490.92s | 414.75s | 248.54 | 44.07% | 3.20 | 3661.01 |
| 15 | 507.57s | 432.04s | 242.11 | 50.01% | 3.50 | 3694.05 |
| 16 | 521.40s | 443.24s | 237.89 | 41.72% | 3.09 | 3762.32 |
| 17 | 474.81s | 397.77s | 249.49 | 52.71% | 3.64 | 3616.28 |

Step 18 is present in the log but incomplete, so it is excluded from timing summaries.

Step 2-17 summary:

- Completed steps: `16`
- Mean total step time: `493.80s`
- Mean generation time: `415.78s`
- Mean E2E throughput: `248.88` tokens/sec/GPU
- Token-weighted acceptance: `46.29%`
- Mean weighted acceptance length: `3.34`

## Code fix applied remotely

Remote checkout:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606`

File:

`nemo_rl/algorithms/grpo.py`

Changes:

- PARD/PARD2 online training now defaults `initial_refit=false`, so vLLM keeps the HF drafter it loaded from `speculative_config.model` until a real online training update is available.
- Non-PARD draft types keep the old default `initial_refit=true`.
- `policy.draft.type=pard2` is now accepted as a PARD-family alias and defaults the draft loss to `pard2`.
- PARD/PARD2 CAT now uses an exclusive prefix product for k-slot weighting, so slot 0 has weight `1.0` and later slots are weighted by prior-token confidence instead of including their own target confidence.
- PARD online draft targets now explicitly mask sequence ends using `input_lengths`, avoiding rollaround labels from the first token.
- CAT weighting now refuses batches where `prev_logprobs` was skipped and replaced with zeros; GRPO marks this with `prev_logprobs_valid`.
- PARD online draft TP is validated to match target TP for now, because the current draft loss path uses the target tensor-parallel group.
- PARD online training now explicitly requires target `pipeline_model_parallel_size=1`; the previous PP>1 prototype could build non-owner dense PARD copies and retain unnecessary GPU memory.
- Added config knobs:
  - `policy.draft.initial_refit`
  - `policy.draft.initial_draft_refit`
  - `policy.draft.train_start_step`
  - `policy.draft.refit_start_step`
- Submit helpers now expose the start-step knobs as:
  - `POLICY_DRAFT_TRAIN_START_STEP`
  - `POLICY_DRAFT_REFIT_START_STEP`
  - Qwen30 long-output wrapper aliases: `ONLINE_TRAIN_START_STEP`, `ONLINE_REFIT_START_STEP`
- Submit helpers now expose PARD/PARD2 online loss knobs:
  - `POLICY_DRAFT_LOSS`
  - `POLICY_DRAFT_CAT_WEIGHTING`
  - `POLICY_DRAFT_CAT_MODE`
  - `POLICY_DRAFT_CAT_LOGPROB_KEY`
  - `POLICY_DRAFT_DROP_TOKEN_RATIO`
- Submit helpers now expose bounded PARD drafter-training knobs:
  - `POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH`
  - `POLICY_DRAFT_TRAINING_WINDOW`
- The Qwen30 long-output online wrapper defaults to `PARD_MAX_TRAINING_SEQUENCE_LENGTH=2048` and `PARD_TRAINING_WINDOW=tail`, so full-GRPO rollout remains long while the auxiliary k-slot drafter loss only forwards over a bounded tail window.
- The Qwen30 long-output wrapper now exposes `PARD2_GRPO_MODE=sync_repair|strict_request_logprobs|throughput_only`. The default `sync_repair` mode keeps the SpecDec speed/acceptance path active and uses synchronous on-policy logprob repair after rollout. `strict_request_logprobs` is available for correctness smoke tests but should not be used for speedup claims because vLLM can disable SpecDec acceptance for request-logprob runs.
- The sequence-packing fallback no longer builds a full padded-sequence dense PARD attention mask before applying the window; it only builds the draft-only causal mask after window selection.
- The base submit helper now preserves PARD-2 defaults: `POLICY_DRAFT_TYPE=pard2` defaults to `POLICY_DRAFT_LOSS=pard2` and `POLICY_DRAFT_CAT_WEIGHTING=true` unless explicitly overridden. This prevents silent downgrade to plain hard CE.
- The base submit helper now defaults `DRAFT_FORMAT=pard` to `SPECDEC_PARALLEL_DRAFTING=true` unless explicitly overridden, so the vLLM PARD path gets `parallel_drafting=true` by default.
- The base submit helper now fail-fast rejects PARD/PARD2 online `k_slot` training when `POLICY_DRAFT_MAX_TRAINING_SEQUENCE_LENGTH<=0`, because uncapped k-slot training expands the draft sequence by K and builds a dense draft attention mask.
- PARD hard-label draft loss now normalizes by a global draft-token denominator instead of the policy `global_valid_toks`. The count is computed once per global batch from the same PARD target/mask construction, all-reduced over data parallel ranks, and passed through the microbatches as `draft_global_valid_toks`.
- The previous tensor boolean normalization flag was removed from the draft loss hot path, avoiding a GPU-to-CPU sync during loss computation.

Validation:

- `python3 -m py_compile nemo_rl/algorithms/grpo.py` passed on the remote checkout.
- `python3 -m py_compile nemo_rl/models/megatron/draft/pard.py nemo_rl/models/megatron/draft/utils.py nemo_rl/models/megatron/train.py nemo_rl/models/megatron/setup.py nemo_rl/algorithms/grpo.py` passed on the remote checkout after the code-review fixes and the windowed-training patch.
- `python3 -m py_compile nemo_rl/algorithms/loss/loss_functions.py nemo_rl/algorithms/loss/utils.py nemo_rl/models/megatron/draft/pard.py nemo_rl/models/megatron/draft/__init__.py nemo_rl/models/policy/workers/megatron_policy_worker.py` passed on the remote checkout after the draft-token normalization patch.
- `bash -n` passed for `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` after the PARD parallel-drafting default and k-slot cap guard.
- `bash -n` passed for `experiments/eagle3_online/submit_qwen30ba3b_pard2_online_advantage_longosl_20260611.sh`.

## Follow-up job

The earlier pending online jobs `3263475` and `3265044` were cancelled before running because they were submitted before the PARD-2 CAT loss knobs were wired through the helper. Replacement jobs `3265200`/`3265201` and `3265329`/`3265330` were also cancelled before running because they were submitted before bounded/windowed PARD training and explicit SpecDec logprob-mode controls were wired through. They have been replaced with the following PARD-2 CAT windowed jobs:

| Job | Variant | Train/refit interval | Train/refit start step | GRPO logprob mode | Latest state |
| --- | --- | ---: | ---: | --- | --- |
| `3265386` | static PARD-2 CAT drafter, win2048 config present but no online train | n/a | n/a | sync repair | running |
| `3265387` | immediate-start PARD-2 CAT online, win2048 tail | `10` | `1` | sync repair | running |
| `3265388` | delayed-start PARD-2 CAT online, win2048 tail | `10` | `10` | sync repair | running |
| `3274811` | delayed/repeated PARD-2 CAT online, win2048 tail, submitted with `coreai_dlalgo_llm` | `5` | `5` | sync repair | running |

Latest queue check at `2026-06-11 21:26 PDT`: all four Qwen30 PARD2 jobs are `RUNNING`; the corrected SWERL suffix generation-only smoke `3266737` remains `PENDING/Priority`; suffix Full-GRPO20 job `3266990` is `COMPLETED`. Machine-readable status is in `docs/pard2_swerl_active_status_20260611.csv`, with the raw poll in `docs/pard2_swerl_active_status_20260611.txt`. The helper `scripts/refresh_pard2_swerl_active_status.sh` refreshes this status and fetches parsed PARD2 per-step/summary CSVs from ray-driver logs.

The immediate-start job tests whether a Step 1 online update helps or hurts later rollout acceptance. The delayed-start jobs keep the HF PARD2 drafter static through the early steps, then start online training/refit at Step 10 or Step 5; these are cleaner runs for testing whether online drafter adaptation helps after enough long-output rollout data has accumulated. Static comparison should use `3265386`; older static job `3263290` remains useful only as a non-strict generation/acceptance reference.

Remaining implementation risk: k-slot PARD training still expands the effective draft sequence to `training_window_length * K` and uses a dense attention mask. The new win2048 setting reduces the Qwen30 long-output online jobs from full `8192 * 5` draft length to `2048 * 5`, but this is still a bounded dense-window solution rather than a true varlen/sampled objective. For OSL 16K+ or aggressive K such as 32, the next step is sampled PARD training that avoids quadratic `window*K` attention entirely.

## Qwen8 PARD2 canary

While the Qwen30 20-step jobs wait in the 4-node queue, a smaller Qwen3-8B canary was submitted to exercise the same PARD2 online training/refit path on a 1-node shape:

| Job | Variant | Shape | Steps | K | Train window | Latest state |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `3266304` | static PARD2 canary | `1x4` GPUs | `2` | `5` | n/a | completed |
| `3266305` | online PARD2 canary | `1x4` GPUs | `2` | `5` | `256` tail | completed |

Job file: `latest_qwen8_pard2_online_canary_20260611_jobs.txt`.

This canary is not a performance substitute for Qwen30. It caught the PARD2 k-slot/CAT/refit path early and completed with finite online draft loss. Parsed outputs are:

- `docs/qwen8_pard2_online_canary_steps_20260611.csv`
- `docs/qwen8_pard2_online_canary_summary_20260611.csv`
- `docs/qwen8_pard2_online_canary_comparison_20260611.csv`
- `docs/qwen8_pard2_online_canary_comparison_20260611.md`
