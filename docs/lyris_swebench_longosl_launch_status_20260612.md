# Lyris SWE-Bench Long-OSL SpecDec Launch Status - 2026-06-12

## Review Before Submit

Reviewed before continuing the remaining Lyris launch:

- Local syntax validation passed for the Lyris base helper and long-OSL wrapper.
- Lyris SSH ControlMaster was active for `login-lyris`.
- Remote prompt data is present:
  - `data/swebench_verified_prompts_all.jsonl`: 500 prompts.
  - `data/swebench_full_test_prompts_all.jsonl`: 2294 prompts.
- Remote container exists:
  - `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm-hsg-ultra-rl-v0.20.2-pd42430.sqsh`
- Lyris HF cache contains the models/drafters used here:
  - `Qwen/Qwen3-30B-A3B`
  - `Qwen/Qwen3-8B`
  - `amd/PARD-Qwen3-0.6B`
  - `amd/PARD2-Qwen3-8B`
  - `RedHatAI/Qwen3-8B-speculator.eagle3`
- Generated SpecDec configs were checked:
  - Baseline: no speculative config.
  - Suffix: `{"method":"suffix","num_speculative_tokens":32}` with no draft model.
  - PARD: `draft_model`, `parallel_drafting=true`, K=5.
  - PARD-2: `draft_model`, `parallel_drafting=true`, K=5.
  - Eagle3: `method=eagle3`, K=3.
- Long-context vLLM setup exports `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

One cosmetic issue was found after review: generated log tags inherited a `20260611` suffix. This does not affect correctness or remote execution, and the local helper now uses configurable `RUN_TAG_DATE` for future launches.

## Submitted Matrix

Submitted on Lyris `gb200` with account `coreai_dlalgo_llm`.

The new long-OSL matrix covers:

- SWE-Bench Verified: 16K, 64K, 128K.
- SWE-Bench Full: 16K, 32K, 64K, 128K.
- Qwen3-30B-A3B: baseline, suffix K32, PARD K5.
- Qwen3-8B: baseline, suffix K32, PARD-2 K5, Eagle3 K3.

Verified 32K is intentionally not duplicated in this wrapper because it is already covered by the earlier `latest_lyris_*_swebench_verified_osl32k_*` trackers.

Batch sizing:

- OSL 16K/32K: `prompt_count=4`, `batch_sizes="1 2"`, `max_num_seqs=2`.
- OSL 64K: `prompt_count=2`, `batch_sizes=1`, `max_num_seqs=1`.
- OSL 128K: `prompt_count=1`, `batch_sizes=1`, `max_num_seqs=1`.

## Tracker Files

Use these local tracker files for collection:

- `latest_lyris_swebench_longosl_standalone_specdec_20260612_jobs.csv`: initial 10 jobs from the interrupted first launcher.
- `latest_lyris_swebench_verified_osl64k_qwen8_baseline_suffix_recovered_20260612_jobs.csv`: recovered Verified 64K qwen8 baseline/suffix jobs `2102223` and `2102225` from the interrupted launcher.
- `latest_lyris_swebench_verified_osl64k_qwen8_specdec_20260612_jobs.txt`: later Verified 64K qwen8 PARD-2/Eagle3 base-helper tracker.
- `latest_lyris_swebench_longosl_remaining_verified64_qwen8_20260612_jobs.csv`: remaining Verified 64K qwen8 PARD-2/Eagle3 jobs `2102300` and `2102301`.
- `latest_lyris_swebench_longosl_remaining_20260612_jobs.csv`: remaining Verified 128K plus Full SWE-Bench long-OSL jobs.
- `latest_lyris_swebench_verified_osl128k_qwen8_baseline_cuda_diag_20260612_jobs.txt`: Qwen3-8B Verified OSL128K TRITON_ATTN baseline diagnostic `2102797`.
- `latest_lyris_swebench_verified_osl128k_qwen8_baseline_flashattn_diag_20260612_jobs.txt`: Qwen3-8B Verified OSL128K FLASH_ATTN baseline diagnostic `2102987`.
- `latest_lyris_swebench_verified_osl128k_qwen8_flashattn_nodebug_after2102987_20260612_jobs.txt`: dependency-held Verified OSL128K FlashAttention no-debug jobs `2103047` and `2103048`, cancelled because `2102987` failed.
- `latest_lyris_swebench_full_osl128k_qwen8_flashattn_nodebug_after2102987_20260612_jobs.txt`: dependency-held Full OSL128K FlashAttention no-debug jobs `2103050` and `2103051`, cancelled because `2102987` failed.
- `latest_lyris_swebench_verified_osl96k_qwen8_baseline_maxctxdiag_20260612_jobs.txt`: Qwen3-8B Verified OSL96K baseline-only max-context diagnostic `2103166`.
- `latest_lyris_swebench_verified_osl96k_qwen8_specdec_after2103166_20260612_jobs.txt`: Qwen3-8B Verified OSL96K suffix K32 `2103212`, Eagle-3 K3 `2103213`, and PARD-2 K5 `2103214`, all submitted with `afterok:2103166`.

Total tracked jobs in the original long-OSL launch set: 49. The refreshed long-OSL status now tracks 62 jobs after adding Qwen8 OSL128K retries/diagnostics, cancelled FlashAttention follow-ups, and the OSL96K bracketing add-on.

## Status Snapshot

Immediately after submission:

- Running: 16 jobs.
- Pending: 33 jobs, all pending on SLURM priority.
- No submit-time failures observed.

Numeric SLURM gaps such as `2102224` are unrelated jobs from other users/accounts, not missing entries from this matrix.

## Refresh Snapshot

As of the 13:41 CEST refresh/poll:

- Manifest rows: 62 tracked jobs.
- Accounting states: 42 completed, 3 running, 9 failed, 4 cancelled, 4 timeout, 0 pending.
- Final metric rows: 63 completed benchmark rows.
- `2103166` completed successfully after 1:29:25 and wrote the Qwen3-8B Verified OSL96K baseline row: 37.13 tok/s/GPU.
- `2103212`, `2103213`, and `2103214` are still running against the OSL96K baseline. Current live partials, all with 0 completed rows so far: suffix K32 `2103212` on `lyris0018` at 13.5 gen tok/s and 4.7% acceptance; Eagle-3 K3 `2103213` on `lyris0062` at 7.0 gen tok/s and 0.0% acceptance; PARD-2 K5 `2103214` on `lyris0155` at 8.1 gen tok/s and 0.0% acceptance.
- Direct `squeue`/refresh at 13:41 CEST shows only three long-OSL jobs still running: `2103212`, `2103213`, and `2103214`.
- Newly terminal since the 12:36 refresh: Full OSL128K Qwen30 PARD K5 `2102334` timed out after 5:00:30 with 0 completed rows. This arm has no final benchmark result.
- Newly terminal since the 12:36 refresh: Full OSL128K Qwen8 PARD-2 K5 `2102349` timed out after 5:00:31 with 0 completed rows. This arm has no final benchmark result.
- A final duplicate-submission check found all 49 expected long-OSL combinations already represented in the manifest, and the checkpoint/dependency prewarm job `2100960` completed successfully. No duplicate jobs were submitted after this review. The 13:41 review also kept submissions closed because the only useful OSL96K evidence now depends on the three jobs already running.
- Newly terminal since the 12:15 refresh: Verified OSL128K Qwen8 PARD-2 K5 `2102307` timed out after 5:00:24 with 0 completed rows. This arm has no final benchmark result.
- Newly included in the final metric table: Verified OSL16K Qwen8 suffix K32 BS2 at 650.01 tok/s/GPU, 8.680x vs baseline, 92.01% acceptance.
- Newly terminal since the 12:09 refresh: Full OSL64K Qwen8 PARD-2 K5 `2102330` completed after 4:18:23. Its final row remains 12.92 tok/s/GPU, 0.354x vs baseline, 1.78% acceptance.
- Newly terminal since the 11:53 refresh: Verified OSL128K Qwen30 PARD K5 `2102304` timed out after 5:00:24. Its live progress row had 0 completed rows, so this arm is treated as no final benchmark result rather than a measured speedup.
- Newly finalized since the 11:30 refresh: Full OSL128K Qwen30 baseline `2102332` completed at 20.06 tok/s/GPU, making suffix K32 `2102333` a measured 3.973x speedup at 79.70 tok/s/GPU and 97.16% acceptance.
- Newly finalized since the 11:46 refresh: Full OSL64K Qwen30 PARD K5 `2102327` completed at 13.66 tok/s/GPU, 0.693x vs baseline, 10.99% acceptance; Full OSL32K Qwen8 PARD-2 K5 `2102323` completed its BS2 row at 33.52 tok/s/GPU, 0.447x vs baseline, 3.78% acceptance.
- Recently finalized: Full OSL128K Qwen8 Eagle-3 K3 `2102361` completed at 21.68 tok/s/GPU with 40.88% acceptance; the matching baseline is failed/unstable, so no final speedup is claimed.
- Recently finalized: Verified OSL96K Qwen8 baseline `2103166` completed at 37.13 tok/s/GPU.
- Recently finalized row retained from the 11:00 refresh: Verified OSL64K Qwen8 PARD-2 K5 `2102300` completed at 13.37 tok/s/GPU, 0.362x vs baseline, 2.61% acceptance.
- Recently finalized rows retained from earlier refreshes: Full OSL64K Qwen30 baseline `2102325` completed at 19.70 tok/s/GPU, making suffix K32 `2102326` a measured 1.863x speedup; Verified OSL128K Qwen30 baseline `2102302` completed at 19.97 tok/s/GPU, making suffix K32 `2102303` a measured 4.321x speedup; Verified OSL64K Qwen30 PARD K5 `2102222` completed below baseline at 0.763x; Full OSL64K Qwen8 Eagle-3 K3 `2102331` completed below baseline at 0.564x.

## Notes

This qwen8/qwen30 batch does not include Qwen3-235B. Qwen3-235B uses a separate reviewed TP4/fp8-KV setup; its OSL64K pilot is tracked in `docs/lyris_qwen235b_suffix_status_20260612.md`.
