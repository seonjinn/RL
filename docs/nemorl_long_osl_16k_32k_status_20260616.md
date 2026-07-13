# NeMo-RL Long OSL 16K/32K Status - 2026-06-16

Scope: NeMo-RL training runs where the generation/output cap is 16K or 32K. For the MathRL/SpecDec launchers this is `policy.generation.max_new_tokens`; for the SWE-RL NemoGym scaleout runs this is `policy.max_total_sequence_length`, with the generation cap tied to the sequence cap. Standalone vLLM OSL32K runs and `max_model_len=32768` fixed-output jobs are listed only as exclusions.

Remote refresh note, `2026-06-16 01:16 PDT`: OCI-HSG queue/log checks found no newer MathRL NeMo-RL job with `policy.generation.max_new_tokens=16384` or `32768`. Current OCI-HSG MathRL jobs are `max_new_tokens=1024`. Lyris is reachable via `login-lyris`; the active OSL32K jobs there are standalone vLLM benchmark jobs, not NeMo-RL. The Lyris integrated NeMo-RL matrix from 2026-06-13 used `max_new_tokens=1024` and is therefore excluded from this long-OSL table. Additional Lyris SWE-RL `len65568` NeMo-RL smoke retries were checked; they were pre-step setup/dependency failures or were cancelled before trajectory/step timing, so they do not add a 16K/32K OSL result.

## Summary

| Case | Jobs | Model | Max OSL | Status | Result |
|---|---:|---|---:|---|---|
| Matched Step 1 long decode | 3207578 / 3207579 | Qwen3-30B-A3B | 16384 | failed after Step 1 | PARD K3 was faster on Step 1: total step `1.330x`, E2E throughput `1.316x`, generation time `1.409x`, generation throughput `1.394x`. |
| Completed static PARD long decode | 3231517 | Qwen3-1.7B | 16384 | completed 20/20 | Static PARD K3 ran successfully; avg total step `260.14s`, generation `250.83s`, E2E `132.69 tok/s/GPU`, generation worker `137.96 tok/s/GPU`, token acceptance `74.90%`, mean accept len `3.27`. No matched baseline found. |
| Completed matched no-spec/static PARD | 3231642 / 3231622 | Qwen3-32B | 16384 | completed 12/12 | Static PARD K3 improved total step `1.106x`, E2E throughput `1.134x`, generation time `1.117x`, generation throughput `1.147x`; token acceptance `49.76%`, mean accept len `2.53`. |
| Completed matched no-spec/static PARD | 3231954 / 3231955 | Qwen3-32B | 16384 | completed 12/12 | Static PARD K3 improved total step `1.080x`, E2E throughput `1.095x`, generation time `1.083x`, generation throughput `1.099x`; token acceptance `50.28%`, mean accept len `2.44`. |
| Online PARD long decode | 3231518 | Qwen3-1.7B | 16384 | failed at Step 1 | Reached generation metrics, then failed in policy training logprob computation with CUDA OOM. No completed step timing. |
| SWE-RL 32K scaleout, async4/8 | 12561705 / 12561707 / 12561710 / 12561712 | Qwen3-235B-A22B-Thinking-2507 | 32768 | timed out after 4h, 6-9 steps | Best run was async4 FP8 KV job `12561710`: 9 steps, post-cold avg total `1026.02s`, exposed generation `681.38s`. Clean per-GBS-256-equivalent wall was ~`407s`, near 16K baseline `411.40s`, but spike-included real average was `+25%` slower. |
| SWE-RL 32K scaleout, async1 BF16 | 12612995 | Qwen3-235B-A22B-Thinking-2507 | 32768 | timed out after 4h, 10 step events | Post-cold mean total `1122.99s`, exposed generation `874.95s`, policy train `257.54s`; per-useful-work throughput was estimated `32%` better than the 16K 128-H100 baseline, but absolute step wall was `2.71x` longer. |
| Lyris SWE-RL long-context smoke retries | 2123407 / 2123638 / 2123875 / 2124030-2124032 / 2124206-2124208 | Qwen3-235B-A22B-Thinking-2507 | 65568 cap, not a valid 32K result | no completed step | These were follow-up smoke retries after a `65537 > 65536` vLLM boundary error. r23/r24 reached `SETUP COMPLETE` but no trajectory batch or Step 1 metric; r25 was cancelled during TransformerEngine build; r26/r27 failed during TransformerEngine dependency build. |
| Exact 32K MathRL/SpecDec NeMo-RL | none found | n/a | 32768 | no result | No completed or partial MathRL/PARD/Eagle/Suffix NeMo-RL training-step result with `max_new_tokens=32768` found in local docs or June SLURM history. |

## Details

The Qwen3-30B-A3B 16K run used `max_new_tokens=16384`, attempted `min_tokens=16384`, `max_model_len=20480`, `max_num_batched_tokens=32768`, GBS512, and 4 nodes x 4 GPUs. The launcher accepted the config and both baseline and PARD K3 emitted matched Step 1 timing blocks before failing in Step 2.

| Metric | Baseline 3207578 | PARD K3 3207579 | Speedup |
|---|---:|---:|---:|
| Total step time | `981.19s` | `737.72s` | `1.330x` |
| E2E throughput | `234.11 tok/s/GPU` | `308.08 tok/s/GPU` | `1.316x` |
| Generation time | `809.56s` | `574.42s` | `1.409x` |
| Generation throughput | `283.74 tok/s/GPU` | `395.67 tok/s/GPU` | `1.394x` |

Failure cause for 3207578/3207579: vLLM sleep/wake memory lifecycle at Step 2 generation start, specifically `CuMemAllocator.wake_up()` failing with CUDA OOM while remapping KV cache.

The Qwen3-1.7B static PARD run 3231517 used `max_new_tokens=16384`, `NRL_VLLM_GENERATION_MIN_TOKENS=16384`, `NRL_VLLM_GENERATION_IGNORE_EOS=true`, `max_model_len=18432`, GBS8, and completed all 20 steps. It is a stability proof for NeMo-RL + SpecDec at 16K OSL, but not a speedup proof because no matched baseline run was found.

The Qwen3-1.7B online PARD run 3231518 used the same 16K OSL shape with `train_interval=10` / `refit_interval=10`. It failed during `MegatronPolicyWorker.train()` while computing distributed logprobs:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.35 GiB.
```

The Qwen3-32B 16K OSL runs provide the cleanest completed baseline-relative NeMo-RL comparison found in this refresh. They used 4 nodes x 4 GPUs, `max_new_tokens=16384`, and completed all 12 steps for both no-spec and static PARD K3:

| Model | GBS | No-spec job | Static PARD job | Steps | No-spec total | PARD total | Step speedup | No-spec gen | PARD gen | Gen-time speedup | E2E speedup | Gen tok/s speedup | Acceptance | Mean accept len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-32B | 8 | 3231642 | 3231622 | 12/12 | `934.56s` | `844.86s` | `1.106x` | `903.41s` | `809.04s` | `1.117x` | `1.134x` | `1.147x` | `49.76%` | `2.53` |
| Qwen3-32B | 16 | 3231954 | 3231955 | 12/12 | `949.00s` | `878.92s` | `1.080x` | `914.10s` | `844.34s` | `1.083x` | `1.095x` | `1.099x` | `50.28%` | `2.44` |

Additional 16K OSL attempts had no completed step metrics:

| Jobs | Model | Shape | Status | Note |
|---|---|---|---|---|
| 3207762 / 3207763 | Qwen3-30B-A3B | GBS512, `max_new_tokens=16384`, mem70 | cancelled | No completed step metrics. |
| 3207808 / 3207809 | Qwen3-30B-A3B | GBS512, `max_new_tokens=16384`, system Python retry | failed | No completed step metrics. |
| 3231620 / 3231621 / 3231641 | Qwen3-30B-A3B | GBS8, no-spec/static/online | failed | No completed step metrics. |
| 3231623 / 3231956 | Qwen3-32B | online PARD INT10, GBS8/16 | failed | No completed step metrics. |
| 3231951 / 3231952 / 3231953 | Qwen3-30B-A3B | GBS16, no-spec/static/online | failed | No completed step metrics. |

## 32K SWE-RL Scaleout

The clearest true 32K NeMo-RL evidence is from the SWE-RL / NemoGym scaleout work under `experiments/scaleout_256h100_32k_async16`. These are not PARD/PARD-2/Eagle-3 SpecDec comparisons, but they are real NeMo-RL training runs with `policy.max_total_sequence_length=32768` on Qwen3-235B-A22B-Thinking-2507.

Initial async16 jobs `12550182` / `12550221` were too aggressive: cluster size, sequence length, GBS, and async age were all lifted together. They did not emit `Performance Metrics` blocks; Step 0 did not complete before the run was cancelled/timed out. The logs showed heavy prompt rejection: prompts longer than 32K caused `Prompt exceeds max_model_len` errors, with about `54%` 400 Bad Request waste. That is evidence that 32K was still too small for some long SWE agent traces.

The useful 32K scaleout comparison is the async4/async8 4h batch run:

| Job | Variant | Steps | Post-cold avg total | Exposed generation | Policy train | Logprobs | Real avg vs 16K baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| 12561705 | async4 BF16 KV | 7 | `1371.36s` | `1025.10s` | `255.66s` | `68.98s` | `+67%` slower per GBS-256 equivalent |
| 12561710 | async4 FP8 KV | 9 | `1026.02s` | `681.38s` | `253.35s` | `68.83s` | `+25%` slower per GBS-256 equivalent |
| 12561707 | async8 BF16 KV | 6 | `1164.19s` | `808.56s` | `256.69s` | `69.25s` | `+41%` slower per GBS-256 equivalent |
| 12561712 | async8 FP8 KV | 7 | `1074.96s` | `728.43s` | `255.48s` | `68.72s` | `+31%` slower per GBS-256 equivalent |

The main readout: training and logprobs were basically stable across variants; exposed generation carried almost all variance. FP8 KV helped generation substantially, especially at async4 (`1025.10s` -> `681.38s`, about `33.5%` lower exposed generation). However, long-tail SWE trajectories still produced spike steps around `1450-1700s`, so no config reached the 15-step target inside a 4h `batch` window.

A later 32K BF16 async=1 rerun, job `12612995`, reached 10 step events before 4h timeout:

| Job | Variant | Steps | Post-cold avg total | Exposed generation | Policy train | LogProb tok/s/GPU | Train TFLOPS/rank |
|---|---|---:|---:|---:|---:|---:|---:|
| 12612995 | async=1, 192 gen / 64 train, BF16 KV | 10 | `1122.99s` | `874.95s` | `257.54s` | `3160` | `150.3` |

Compared to the 16K 128-H100 baseline (`414.88s/step`, GBS256 x 16K), this 32K run used 4x the useful-work proxy (`512 x 32K`) and had 2.71x absolute step time, so normalized throughput was better. Operationally, though, the 4h partition wall and SWE long-tail remained the limiter; 15 steps would need either a longer partition or lower SWE agent timeout / max-turn caps.

The Qwen3-235B jobs 3228527-3228530 are not 16K/32K OSL runs. They are context-length ablations: `mlen16384-fixed256` and `mlen32768-fixed256`, meaning `max_model_len` was 16K/32K but `max_new_tokens=256`.

## Lyris SWE-RL `len65568` Smoke Retries

These jobs are useful operational evidence, but they are not benchmark results. They were launched after an earlier SWE-RL run hit `VLLMValidationError: You passed 65537 input tokens ... model's context length is only 65536`; the retry raised the sequence/model cap to `65568` so slightly-over-boundary SWE agent traces would not be rejected. That is larger than the requested 32K OSL bucket and, more importantly, none of these jobs emitted completed step timing.

| Job | Variant | State | Furthest observed point | Result |
|---:|---|---|---|---|
| 2123407 | PARD K5, r23 `len65568` | cancelled after `01:08:59` | `SETUP COMPLETE`; NemoGym env build | No trajectory batch, no Step 1 timing. |
| 2123638 | baseline, r24 `len65568` | cancelled after `00:56:53` | `SETUP COMPLETE`; NemoGym env build | No trajectory batch, no Step 1 timing. |
| 2123875 | PARD-2 K1, r25 `len65568` | cancelled after `00:22:59` | TransformerEngine build | Did not reach config/setup. |
| 2124030 / 2124031 / 2124032 | baseline / PARD / PARD-2, r26 `len65568` | failed after ~`35m` | dependency build | `RuntimeError: Error compiling objects for extension` while building TransformerEngine. |
| 2124206 / 2124207 / 2124208 | baseline / PARD / PARD-2, r27 `len65568` | failed/cancelled within ~`2m` | dependency build | `RuntimeError: Error when running CMake` while building TransformerEngine. |

Sources:

- `docs/qwen3_30ba3b_longosl16k_step1_metrics_20260607.csv`
- `docs/qwen3_30ba3b_fullgrpo20_status_20260606.md`
- `docs/oci_hsg_qwen17b_out16k_nemorl_summary_20260616.csv`
- `sacct` refresh on `oci-hsg-cs-001-vscode-02` for jobs 3207578, 3207579, 3207762, 3207763, 3207808, 3207809, 3231517, 3231518, 3231620-3231623, 3231641-3231642, 3231951-3231956
- `tmp/oci_hsg_qwen17b_out16k_20260609_logs/3231517-logs/ray-driver.log`
- `tmp/oci_hsg_qwen17b_out16k_20260609_logs/3231518-logs/ray-driver.log`
- `experiments/scaleout_256h100_32k_async16/report/README.md`
- `experiments/scaleout_256h100_32k_async16/report/data/per_step_breakdown.csv`
- `experiments/scaleout_256h100_32k_async16/report/data/per_step_breakdown_means_excl_cold.csv`
- `sacct` refresh on `login-lyris` for jobs 2123407, 2123638, 2123875, 2124030-2124032, 2124206-2124208
- Lyris driver logs under `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260614_lyris_swerl_qwen235b_fullgrpo_stagedsif_smoke_*_len65568*`
