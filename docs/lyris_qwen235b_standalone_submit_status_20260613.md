# Lyris Qwen3-235B Standalone SpecDec Submit Status - 2026-06-13

Refreshed at `2026-06-13T09:51:16+02:00`.

Latest fast refresh: `scripts/refresh_lyris_qwen235b_standalone_fast.py`.

## Launch Summary

All standalone vLLM jobs were submitted on Lyris with account `coreai_dlalgo_llm`, partition `gb200`, TP=4, FP8 KV cache, ISL=4096, and OSL=32768.

| Suite | Dataset | Methods | Batches | Jobs | Current state |
| --- | --- | --- | --- | ---: | --- |
| SWE standalone | SWE-Bench-Verified, SWE-Bench full | Suffix K8/K16, PARD K9/K11, PARD-2 K9/K11, Eagle-3 K9/K11 | 2, 4, 8, 16, 32 | 80 | 38 COMPLETED, 1 COMPLETING, 41 RUNNING |
| MATH500 standalone | MATH500 | baseline, Suffix K32, PARD K5, PARD-2 K1, Eagle-3 K3 | 1, 2 | 5 | 1 COMPLETED, 4 RUNNING |

## Current Results

- Fast refresh found `42` metrics rows from `41` completed `breakdown.json` files.
- SWE completed rows increased to `40` final metric rows. Suffix K8/K16 are complete for batch 2/4/8/16/32 on both SWE-Bench full and SWE-Bench-Verified.
- PARD K9/K11 now has 3 final SWE rows: full batch 2 K9, full batch 2 K11, and full batch 8 K9. These are `6.18-19.98` tok/s/GPU with `16.25%-18.12%` acceptance and still have no matched final baseline row.
- PARD-2 K9/K11 still has 0 final SWE rows. The jobs are running and logs parse cleanly, but live acceptance remains low.
- Eagle-3 K9/K11 now has 17 final SWE rows and is much more complete than PARD/PARD-2, though acceptance is still low for Qwen3-235B SWE OSL32K.
- Lyris MATH500 has final Suffix K32 rows for batch 1 and 2: `15.98` and `32.18` tok/s/GPU, with `83.79%` and `88.71%` acceptance.

## Trackers

- SWE suffix: `latest_lyris_qwen235b_swebench_osl32k_suffix_k8_k16_20260613_jobs.csv`
- SWE drafter: `latest_lyris_qwen235b_swebench_osl32k_drafter_k9_k11_20260613_jobs.csv`
- MATH500: `latest_lyris_qwen235b_math500_osl32k_specdec_20260613_jobs.txt`

## Lyris SWE-RL Setup Check

I rechecked the Lyris SWE-RL preflight after the ControlMaster came back. It still cannot be submitted safely because the Rui SWE-RL assets are missing at the Lyris paths used by the launcher:

| Check | Lyris result |
| --- | --- |
| `/lustre/fsw/portfolios/nemotron/users/ruit/evolution_rl` | missing |
| `test_assets/qwen-235B/run_grpo_qwen3_235b_swe_scale_gen.sh` | missing because repo path is missing |
| `ray.sub` in the Rui repo | missing because repo path is missing |
| `test_assets/qwen-235B/grpo_qwen3_235b_async_swe.yaml` | missing because repo path is missing |
| Rui SWE-Bench mcore/apptainer container under `/lustre/fsw/portfolios/coreai/users/ruit/enroot-images` | missing |

I did not submit Lyris SWE-RL jobs from this state. OCI-HSG SWE-RL is already submitted under `nemotron_n3_post`, and the 10-step baseline/PARD cells have started reaching `COMPLETING`.

## Local Artifacts

- `docs/lyris_qwen235b_standalone_fast_20260613.md`
- `docs/lyris_qwen235b_standalone_fast_20260613_status.csv`
- `docs/lyris_qwen235b_standalone_fast_20260613_metrics.csv`
- `docs/lyris_qwen235b_standalone_live_diagnostics_20260613.md`
- `docs/lyris_qwen235b_standalone_live_diagnostics_20260613.html`
- `docs/lyris_qwen235b_pard_pard2_triage_20260613.md`
- `docs/qwen235b_specdec_swe_math_status_20260613.md`
- `docs/qwen235b_specdec_swe_math_status_20260613.html`

## Current Read

The Lyris connection is working and vLLM standalone coverage is active. I am not resubmitting duplicate standalone jobs because all requested SWE batch/method cells and the MATH500 cells already have trackers and are either complete or still running. The priority is to keep refreshing until matched baseline rows and PARD/PARD-2 final rows appear.
