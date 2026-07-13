# Lyris SpecDec 32K Launch Status - 2026-06-12

Checkpoint/dependency prewarm:

| Job | State | Account | Partition | Notes |
| --- | --- | --- | --- | --- |
| `2100960` | `COMPLETED`, `0:0`, elapsed `00:03:00` | `coreai_dlalgo_llm` | `gb200` | Downloaded/resolved all required HF targets and drafters; `arctic_inference_available=True`. |

Prewarmed repos:

| Role | Repo |
| --- | --- |
| Qwen30 target | `Qwen/Qwen3-30B-A3B` |
| Qwen30 Thinking target | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| Qwen8 target | `Qwen/Qwen3-8B` |
| Qwen14 target | `Qwen/Qwen3-14B` |
| PARD public drafter | `amd/PARD-Qwen3-0.6B` |
| PARD-2 public drafter | `amd/PARD2-Qwen3-8B` |
| PARD-2 public drafter | `amd/PARD2-Qwen3-14B` |
| Eagle-3 public drafter | `RedHatAI/Qwen3-8B-speculator.eagle3` |
| Eagle-3 public drafter | `RedHatAI/Qwen3-32B-speculator.eagle3` |
| Eagle-3 public drafter | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` |

Running OSL32K benchmark jobs:

| Job | Target | Method | Drafter | K | Batch sizes | State at launch check |
| --- | --- | --- | --- | --- | --- | --- |
| `2100972` | `Qwen/Qwen3-30B-A3B` | baseline | none | n/a | `1 2` | `RUNNING` on `lyris0266` |
| `2100976` | `Qwen/Qwen3-30B-A3B` | suffix | model-free suffix decoding | `32` | `1 2` | `RUNNING` on `lyris0008` |
| `2100980` | `Qwen/Qwen3-30B-A3B` | PARD | `amd/PARD-Qwen3-0.6B` | `5` | `1 2` | `RUNNING` on `lyris0247` |
| `2100973` | `Qwen/Qwen3-8B` | baseline | none | n/a | `1 2 4` | `RUNNING` on `lyris0166` |
| `2100977` | `Qwen/Qwen3-8B` | PARD-2 | `amd/PARD2-Qwen3-8B` | `5` | `1 2 4` | `RUNNING` on `lyris0015` |
| `2100979` | `Qwen/Qwen3-8B` | Eagle-3 | `RedHatAI/Qwen3-8B-speculator.eagle3` | `3` | `1 2 4` | `RUNNING` on `lyris0103` |
| `2100971` | `Qwen/Qwen3-14B` | baseline | none | n/a | `1 2` | `RUNNING` on `lyris0286` |
| `2100975` | `Qwen/Qwen3-14B` | PARD-2 | `amd/PARD2-Qwen3-14B` | `5` | `1 2` | `RUNNING` on `lyris0006` |
| `2100974` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | baseline | none | n/a | `1 2` | `RUNNING` on `lyris0040` |
| `2100978` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Eagle-3 | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | `1` | `1 2` | `RUNNING` on `lyris0013` |

Common shape:

| Field | Value |
| --- | --- |
| Dataset | SWE-Bench Verified prompt JSONL on Lyris |
| Prompt count | `4` |
| ISL / OSL | `4096 / 32768` |
| Max model length | `40960` |
| Wall limit | `05:00:00` |
| Container | `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm-hsg-ultra-rl-v0.20.2-pd42430.sqsh` |
| Remote root | `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark` |

Local tracking files:

| File | Contents |
| --- | --- |
| `latest_lyris_specdec_checkpoint_prewarm_20260611_jobs.txt` | Prewarm job and cache paths |
| `latest_lyris_qwen30ba3b_swebench_verified_osl32k_publicpard_20260611_jobs.txt` | Qwen30 baseline/suffix/PARD jobs |
| `latest_lyris_qwen3_8b_swebench_verified_osl32k_pard2_eagle3_20260611_jobs.txt` | Qwen8 baseline/PARD-2/Eagle-3 jobs |
| `latest_lyris_qwen3_14b_swebench_verified_osl32k_pard2_20260611_jobs.txt` | Qwen14 baseline/PARD-2 jobs |
| `latest_lyris_qwen30ba3b_thinking2507_swebench_verified_osl32k_eagle3_20260611_jobs.txt` | Qwen30 Thinking baseline/Eagle-3 jobs |

Notes:

- The launched Qwen30 PARD job uses the public HF drafter `amd/PARD-Qwen3-0.6B`.
- The locally trained Qwen30 CAT/TPP-mask checkpoint is still not staged on Lyris. The Lyris destination checked missing at `/lustre/fsw/coreai_dlalgo_llm/users/sna/qwen3_30ba3b_pard2_cat_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen30ba3b_math_k5_cat_tpp_mask_1024_resume_20260605_173358/checkpoint-32`; the HSG source probe hung and was killed locally.
- Early vLLM logs confirm active speculative configs for public PARD, public PARD-2, and public Eagle-3.

## Refresh - 2026-06-12 02:27 CEST

The refresh helper `scripts/refresh_lyris_specdec_32k_results.sh` now polls these jobs, writes a manifest/status snapshot, and extracts vLLM standalone metrics once `breakdown.json` files appear.

Latest generated files:

| File | Contents |
| --- | --- |
| `docs/lyris_specdec_32k_status_20260612.md` | Human-readable queue/result snapshot |
| `docs/lyris_specdec_32k_status_20260612.csv` | Queue/accounting status for the 10 jobs |
| `docs/lyris_specdec_32k_manifest_20260612.csv` | Job-to-log-dir and job-to-breakdown mapping |
| `docs/lyris_specdec_32k_live_progress_20260612.csv` | Live vLLM logger progress before final JSON |
| `docs/lyris_specdec_32k_metrics_20260612.csv` | Parsed metrics once any run completes |

Current refresh state: all 10 jobs are still `RUNNING` at about thirteen minutes elapsed, and no `breakdown.json` has been written yet. Early logs show active decode rather than startup failure.

Latest live vLLM logger snapshot, not final benchmark metrics:

| Label | Gen tok/s | Live acceptance | Mean accept len |
| --- | ---: | ---: | ---: |
| `qwen30_baseline` | `19.5` | n/a | n/a |
| `qwen30_suffix` | `199.9` | `99.4%` | `12.71` |
| `qwen30_pard` | `21.7` | `28.7%` | `2.43` |
| `qwen8_baseline` | `37.3` | n/a | n/a |
| `qwen8_pard2` | `16.0` | `3.9%` | `1.19` |
| `qwen8_eagle3` | `59.7` | `54.1%` | `2.62` |
| `qwen14_baseline` | `34.0` | n/a | n/a |
| `qwen14_pard2` | `15.5` | `2.1%` | `1.11` |
| `qwen30thinking_baseline` | `20.7` | n/a | n/a |
| `qwen30thinking_eagle3` | `19.0` | `0.5%` | `1.01` |

These live values are from the currently active BS=1 request windows. They are useful health/progress signals but should not be used as final speedups; final reporting should use `breakdown.json` after each batch-size sweep finishes.

HSG NeMo-RL status refresh through `scripts/refresh_pard2_swerl_active_status.sh` timed out on SSH this turn. The latest local HSG evidence remains the 2026-06-11 tracker: Qwen8 PARD-2 canaries completed; Qwen30 PARD-2 windowed long-output jobs and SWERL suffix smoke were pending priority at that snapshot.

## Update - 2026-06-12 02:54 CEST

Comparison policy:

- Primary method comparison is **method-best K on the same workload shape**, not forced same K. K is part of the method tuning surface; forcing suffix K32 down to PARD K5 only answers a secondary control question.
- Strict same-K controls are still running for suffix K5 on Qwen30, Qwen8, and Qwen14 so we can report both views: best reasonable K and same-K sanity check.
- Final K choice should be based on completed `breakdown.json` rows: highest output tok/s/GPU with non-degenerate acceptance and no startup/output failures. Live logger values below are only progress signals.

Additional K sweep jobs launched on Lyris:

| Job | Target | Method | Drafter | K | Batch sizes | State |
| --- | --- | --- | --- | ---: | --- | --- |
| `2101240` | `Qwen/Qwen3-30B-A3B` | suffix control | model-free suffix decoding | 5 | `1 2` | `RUNNING` |
| `2101239` | `Qwen/Qwen3-8B` | suffix control | model-free suffix decoding | 5 | `1 2 4` | `RUNNING` |
| `2101241` | `Qwen/Qwen3-14B` | suffix control | model-free suffix decoding | 5 | `1 2` | `RUNNING` |
| `2101263` | `Qwen/Qwen3-30B-A3B` | PARD | `amd/PARD-Qwen3-0.6B` | 9 | `1 2` | `RUNNING` |
| `2101264` | `Qwen/Qwen3-30B-A3B` | PARD | `amd/PARD-Qwen3-0.6B` | 11 | `1 2` | `RUNNING` |
| `2101266` | `Qwen/Qwen3-8B` | PARD-2 | `amd/PARD2-Qwen3-8B` | 9 | `1 2 4` | `RUNNING` |
| `2101267` | `Qwen/Qwen3-8B` | PARD-2 | `amd/PARD2-Qwen3-8B` | 11 | `1 2 4` | `RUNNING` |
| `2101268` | `Qwen/Qwen3-14B` | PARD-2 | `amd/PARD2-Qwen3-14B` | 9 | `1 2` | `RUNNING` |
| `2101269` | `Qwen/Qwen3-14B` | PARD-2 | `amd/PARD2-Qwen3-14B` | 11 | `1 2` | `RUNNING` |
| `2101270` | `Qwen/Qwen3-8B` | Eagle-3 | `RedHatAI/Qwen3-8B-speculator.eagle3` | 5 | `1 2 4` | `RUNNING` |
| `2101271` | `Qwen/Qwen3-8B` | Eagle-3 | `RedHatAI/Qwen3-8B-speculator.eagle3` | 9 | `1 2 4` | `RUNNING` |
| `2101272` | `Qwen/Qwen3-8B` | Eagle-3 | `RedHatAI/Qwen3-8B-speculator.eagle3` | 11 | `1 2 4` | `RUNNING` |
| `2101273` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Eagle-3 | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | 3 | `1 2` | `RUNNING` |
| `2101274` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Eagle-3 | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | 5 | `1 2` | `RUNNING` |
| `2101275` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Eagle-3 | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | 9 | `1 2` | `RUNNING` |
| `2101276` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Eagle-3 | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` | 11 | `1 2` | `RUNNING` |

Latest collector state:

- `scripts/refresh_lyris_specdec_32k_results.sh` now tracks 26 jobs, labels rows with `kN`, includes suffix K5 controls and high-K sweeps, and uses CSV-aware completed-run detection.
- `scripts/extract_vllm_standalone_breakdown_metrics.py` now uses the active SSH ControlMaster path; forcing a fresh Lyris SSH connection failed with `Permission denied (keyboard-interactive)`.
- Completed final metrics so far: Qwen30 suffix K32 only. It completed with `177.25 tok/s/GPU`, `93.92%` acceptance at BS1 and `311.28 tok/s/GPU`, `90.62%` acceptance at BS2.
- Current queue/accounting state: 1 completed, 25 running.

Latest live logger snapshot, not final metrics:

| Label | Gen tok/s | Live acceptance | Mean accept len | Note |
| --- | ---: | ---: | ---: | --- |
| `qwen30_pard_k5` | `73.3` | `86.0%` | `5.30` | volatile live window; final pending |
| `qwen30_pard_k9` | `30.8` | `19.1%` | `2.72` | final pending |
| `qwen30_pard_k11` | `32.3` | `17.1%` | `2.88` | final pending |
| `qwen8_pard2_k5` | `25.6` | `5.3%` | `1.27` | final pending |
| `qwen8_pard2_k11` | `21.5` | `1.8%` | `1.19` | K9 tail timed out in this refresh |
| `qwen14_pard2_k5` | `34.9` | `12.9%` | `1.64` | final pending |
| `qwen14_pard2_k9` | `20.0` | `1.0%` | `1.09` | final pending |
| `qwen14_pard2_k11` | `20.8` | `0.9%` | `1.10` | final pending |
| `qwen8_eagle3_k3` | `96.9` | `72.4%` | `3.17` | final pending |
| `qwen8_eagle3_k5` | `53.8` | `25.4%` | `2.27` | final pending |
| `qwen8_eagle3_k9` | `44.6` | `19.9%` | `2.79` | final pending |
| `qwen8_eagle3_k11` | `44.5` | `15.9%` | `2.75` | final pending |
| `qwen30thinking_eagle3_k1` | `28.9` | `55.1%` | `1.55` | final pending |
| `qwen30thinking_eagle3_k3` | `23.0` | `7.4%` | `1.22` | final pending |
| `qwen30thinking_eagle3_k5` | `22.0` | `5.8%` | `1.29` | final pending |
| `qwen30thinking_eagle3_k9` | `21.6` | `3.4%` | `1.31` | final pending |
| `qwen30thinking_eagle3_k11` | `21.2` | `3.3%` | `1.37` | final pending |

## Update - 2026-06-12 02:58 CEST

Latest refresh command: `scripts/refresh_lyris_specdec_32k_results.sh`.

Current final JSON state is unchanged: only Qwen30 suffix K32 has completed. Baseline and draft-model jobs are still running, so speedups against baseline are intentionally still blank in `docs/lyris_specdec_32k_metrics_20260612.csv`.

Current queue/accounting state:

- `1` completed: `2100976` (`qwen30_suffix_k32`)
- `25` running: all baselines, PARD/PARD-2, Eagle-3, suffix K5 controls, and high-K sweeps

Current final metrics:

| Label | Batch | tok/s/GPU | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: |
| `qwen30_suffix_k32` | 1 | `177.25` | `93.92%` | `10.01` |
| `qwen30_suffix_k32` | 2 | `311.28` | `90.62%` | `9.88` |

Latest live progress signal, still not final:

| Label | Gen tok/s | Live acceptance | Mean accept len |
| --- | ---: | ---: | ---: |
| `qwen30_pard_k5` | `51.3` | `60.2%` | `4.01` |
| `qwen30_pard_k9` | `21.3` | `13.2%` | `2.18` |
| `qwen30_pard_k11` | `21.4` | `10.8%` | `2.18` |
| `qwen8_pard2_k5` | `18.7` | `1.9%` | `1.09` |
| `qwen8_pard2_k9` | `17.2` | `1.1%` | `1.10` |
| `qwen8_pard2_k11` | `17.1` | `0.8%` | `1.09` |
| `qwen14_pard2_k5` | `27.1` | `5.4%` | `1.27` |
| `qwen14_pard2_k9` | `18.1` | `1.6%` | `1.14` |
| `qwen14_pard2_k11` | `18.4` | `1.2%` | `1.14` |
| `qwen8_eagle3_k3` | `60.5` | `46.3%` | `2.39` |
| `qwen8_eagle3_k5` | `98.2` | `47.9%` | `3.39` |
| `qwen8_eagle3_k9` | `93.5` | `28.7%` | `3.58` |
| `qwen8_eagle3_k11` | `86.4` | `22.4%` | `3.46` |
| `qwen30thinking_eagle3_k1` | `23.0` | `21.6%` | `1.22` |
| `qwen30thinking_eagle3_k3` | `20.7` | `3.4%` | `1.10` |
| `qwen30thinking_eagle3_k5` | `19.8` | `3.4%` | `1.17` |
| `qwen30thinking_eagle3_k9` | `18.7` | `1.6%` | `1.15` |
| `qwen30thinking_eagle3_k11` | `18.0` | `1.5%` | `1.17` |

Interim K readout from live logs:

- PARD Qwen30: K5 still looks better than K9/K11.
- PARD-2 Qwen8/Qwen14: K5/K9/K11 are all weak in live acceptance; higher K is not helping.
- Eagle-3 Qwen8: K5 currently has the best live gen tok/s, while K3 had the earlier method-native run; final JSON should choose.
- Eagle-3 Qwen30-thinking: K1 remains better than higher K in live signal.

## Update - 2026-06-12 03:01 CEST

Latest refresh command: `scripts/refresh_lyris_specdec_32k_results.sh`.

No new completed `breakdown.json` files appeared. Final metrics remain limited to `qwen30_suffix_k32`; all baseline, PARD, PARD-2, Eagle-3, suffix K5 control, and high-K sweep jobs are still running.

Current final metrics:

| Label | Batch | tok/s/GPU | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: |
| `qwen30_suffix_k32` | 1 | `177.25` | `93.92%` | `10.01` |
| `qwen30_suffix_k32` | 2 | `311.28` | `90.62%` | `9.88` |

Latest live progress signal, still not final:

| Label | Gen tok/s | Live acceptance | Mean accept len |
| --- | ---: | ---: | ---: |
| `qwen30_pard_k5` | `31.7` | `46.0%` | `3.30` |
| `qwen30_pard_k9` | `15.7` | `9.1%` | `1.82` |
| `qwen30_pard_k11` | `14.6` | `6.3%` | `1.69` |
| `qwen8_pard2_k5` | `15.5` | `1.7%` | `1.08` |
| `qwen8_pard2_k9` | `15.2` | `1.5%` | `1.13` |
| `qwen8_pard2_k11` | `15.8` | `1.5%` | `1.17` |
| `qwen14_pard2_k5` | `20.1` | `2.1%` | `1.10` |
| `qwen14_pard2_k9` | `14.8` | `1.2%` | `1.10` |
| `qwen14_pard2_k11` | `15.1` | `1.0%` | `1.11` |
| `qwen8_eagle3_k3` | `37.1` | `38.3%` | `2.15` |
| `qwen8_eagle3_k5` | `43.9` | `28.7%` | `2.44` |
| `qwen8_eagle3_k9` | `43.5` | `15.9%` | `2.43` |
| `qwen8_eagle3_k11` | `43.4` | `12.3%` | `2.35` |
| `qwen30thinking_eagle3_k1` | `20.3` | `9.7%` | `1.10` |
| `qwen30thinking_eagle3_k3` | `18.7` | `1.7%` | `1.05` |
| `qwen30thinking_eagle3_k5` | `17.8` | `1.1%` | `1.05` |
| `qwen30thinking_eagle3_k9` | `16.9` | `0.7%` | `1.06` |
| `qwen30thinking_eagle3_k11` | `16.3` | `0.5%` | `1.06` |

Interim readout is unchanged: higher K is not improving PARD/PARD-2 on this SWE-Bench 32K shape; the final decision still requires completed JSON rows for the baseline and drafter jobs.

## Update - 2026-06-12 03:10 CEST

Launched Qwen3-235B-A22B suffix decoding on Lyris to test whether the SWE-Bench
SuffixDecoding effect carries to a larger MoE target.

Active Qwen3-235B jobs:

| Job | Shape | Method | Key settings | State |
| ---: | --- | --- | --- | --- |
| `2101420` | SWE-Bench Verified OSL1K gate | baseline | `TP=4`, `fp8` KV, `n=16`, BS `1 2`, `max_model_len=8192` | RUNNING |
| `2101421` | SWE-Bench Verified OSL1K gate | suffix K32 | `TP=4`, `fp8` KV, `n=16`, BS `1 2`, `max_model_len=8192` | RUNNING |
| `2101422` | SWE-Bench Verified OSL32K pilot | baseline | `TP=4`, `fp8` KV, `n=2`, BS `1`, `max_model_len=40960` | RUNNING |
| `2101423` | SWE-Bench Verified OSL32K pilot | suffix K32 | `TP=4`, `fp8` KV, `n=2`, BS `1`, `max_model_len=40960` | RUNNING |

Tracker files:

- `latest_lyris_qwen235b_a22b_swebench_verified_osl1k_suffix_k32_r2_20260612_jobs.txt`
- `latest_lyris_qwen235b_a22b_swebench_verified_osl32k_suffix_k32_pilot_r2_20260612_jobs.txt`

The first Qwen3-235B submission attempt (`2101410`-`2101413`) was canceled before
metrics after vLLM warned that `max_num_batched_tokens` exceeded
`max_num_seqs * max_model_len`. The replacement `r2` jobs use tighter caps
(`16384` for OSL1K, `40960` for OSL32K).

Latest 32K refresh command: `scripts/refresh_lyris_specdec_32k_results.sh`.

Queue/metrics after refresh:

- `27` running, `1` completed in the 32K tracker set.
- The Qwen3-235B OSL32K pilot jobs are in the standard 32K refresh as
  `qwen235b_baseline` and `qwen235b_suffix_k32`.
- No Qwen3-235B `breakdown.json` metrics yet; both engines were still loading /
  early initializing at first refresh.
- Speedups should be reported only against the matching non-speculative
  baseline for the same model, prompt set, ISL/OSL, TP/PP, batch size, KV-cache
  dtype, and prompt count. The refresh Markdown now groups final metric rows by
  model and includes a `Speedup vs baseline` column once the baseline
  `breakdown.json` exists.

Current completed final rows:

| Label | Batch | tok/s/GPU | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: |
| `qwen14_suffix_k5` | 1 | `141.98` | `98.73%` | `5.88` |
| `qwen30_suffix_k32` | 1 | `177.25` | `93.92%` | `10.01` |
| `qwen8_eagle3_k3` | 1 | `69.38` | `61.77%` | `2.85` |
| `qwen8_suffix_k5` | 1 | `149.99` | `96.52%` | `5.52` |
| `qwen30_suffix_k32` | 2 | `311.28` | `90.62%` | `9.88` |
