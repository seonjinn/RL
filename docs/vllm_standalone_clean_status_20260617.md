# vLLM Standalone Clean Status

Updated: `2026-06-17`

Scope: standalone vLLM benchmark artifacts for Math and SWE at `ISL=4096`, `OSL=32768`, temp `0.0` and `1.0`, with top-p `1.0` and top-k `-1` where the latest temp matrix records those sampling settings. This report intentionally separates clean result rows from failed launch provenance and old batch-sweep evidence.

## 2026-06-18 Refresh

The OCI-HSG qmath batch sweep is now fully collected: `112/112` jobs completed, `112` `breakdown.json` files were parsed, and `0` jobs failed. This covers Math MATH-500 batch sizes `4/8/16/32`, temperatures `0.0/1.0`, and models Qwen3-235B-A22B, Qwen3-30B-A3B, and Qwen3-8B. The refreshed status and clean report are:

- `docs/qmath_vllm_standalone_batch_sweep_status_20260617.md`
- `docs/qmath_vllm_standalone_batch_sweep_completed_metrics_live_20260617.csv`
- `docs/vllm_standalone_clean_results_20260617.html`

## Short Answer

Batch sizes `4/8/16/32` were not part of the latest temp `0/1` core matrix. The latest core matrix manifests and trackers use `batch_sizes=1 2` with `max_num_seqs=2`.

They were not generally OOM. Older `/lustre` batch sweeps have completed Qwen3-235B SWE rows for both `suffix_k32` and `eagle3_k3` at batch sizes `4`, `8`, `16`, and `32` on full and verified prompt sets. The one clear OOM evidence is narrower: initial Qwen3-30B-A3B `bs32` high-cap jobs used `max_num_batched_tokens=1310720`, failed during vLLM profiling with CUDA OOM, and were replaced by low-cap jobs using `131072`.

Lyris auth works, but no new retries were submitted because `/lustre` is currently quota-blocked for new log directories:

```text
mkdir: cannot create directory '/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/vllm-runs/clean_status_20260617_probe': Disk quota exceeded
```

I did not fall back to `/home`.

## Current Results

The latest clean core matrix covers Math for Qwen3-8B and Qwen3-30B-A3B, and SWE for Qwen3-235B-A22B, Qwen3-32B, and Qwen3-30B-A3B. The full machine-readable ledger is in `docs/vllm_standalone_clean_status_20260617.csv`.

Math latest core:

| Temp | Model | Best current standalone row | Notes |
| --- | --- | --- | --- |
| `0.0` | Qwen3-30B-A3B | `suffix_k32`: `206.214` tok/s/GPU, `7.147x`, `88.593%` acceptance | PARD K3/K5 also positive but only batch 1 in the latest public-PARD rows. |
| `0.0` | Qwen3-8B | `suffix_k32`: `312.852` tok/s/GPU, `5.660x`, `85.947%` acceptance | EAGLE-3 is positive at `1.766x`; PARD K3/K5 are near baseline. |
| `1.0` | Qwen3-30B-A3B | `suffix_k32`: `162.809` tok/s/GPU, `6.061x`, `83.850%` acceptance | PARD K3/K5 remain positive. |
| `1.0` | Qwen3-8B | `suffix_k32`: `272.290` tok/s/GPU, `5.322x`, `80.594%` acceptance | PARD-2 K3/K5 are below baseline; PARD K3 is positive. |

Qwen3-235B Math exists only as archived/partial standalone rows, not in the latest Math temp core. Temp `0.0` has completed EAGLE-3, PARD K5, and suffix rows, but the matched baseline timed out. Temp `1.0` has the same pattern. Those rows are usable for acceptance and raw throughput, but not for clean baseline-relative speedup.

SWE latest core:

| Temp | Model | Method | tok/s/GPU | Speedup | Acceptance |
| --- | --- | --- | --- | --- | --- |
| `0.0` | Qwen3-235B-A22B | `suffix_k32` | `17.264` | `5.643x` | `79.189%` |
| `0.0` | Qwen3-235B-A22B | `eagle3_k3` | `6.899` | `2.326x` | `54.579%` |
| `1.0` | Qwen3-235B-A22B | `suffix_k32` | `6.947` | `2.374x` | `52.841%` |
| `1.0` | Qwen3-235B-A22B | `eagle3_k3` | `4.789` | `1.550x` | `24.058%` |
| `0.0` | Qwen3-32B | `suffix_k32` | `80.230` | `6.527x` | `84.858%` |
| `1.0` | Qwen3-32B | `suffix_k32` | `27.879` | `2.174x` | `52.227%` |
| `0.0` | Qwen3-30B-A3B | `suffix_k32` | `243.290` | `8.348x` | `92.271%` |
| `1.0` | Qwen3-30B-A3B | `suffix_k32` | `237.720` | `8.261x` | `90.784%` |

These SWE result rows came from the latest temp matrix metrics file, but most final metrics were produced by later `/home` retries after the initial `/lustre` jobs failed or timed out. I kept that provenance explicit rather than calling them fresh `/lustre` results.

## Missing Or Failed Cells

Latest SWE temp core:

- Initial `/lustre` Qwen3-235B temp `0.0` jobs failed or timed out: baseline `2133873` timed out, suffix `2133874` failed, and EAGLE-3 `2133875` failed.
- Initial `/lustre` Qwen3-235B temp `1.0` jobs failed quickly: baseline `2133883`, suffix `2133884`, and EAGLE-3 `2133885`.
- Qwen3-32B and Qwen3-30B-A3B initial `/lustre` rows also failed in the first pass, then later `/home` retries produced the current metric rows.

Math:

- Qwen3-235B Math temp `0.0` baseline job `2113223` timed out after five hours. The spec rows completed, but speedups are blank without the matched baseline.
- Qwen3-235B Math temp `1.0` baseline job `2124147` also timed out after five hours. The spec rows completed, but speedups are blank.
- Latest Math temp core has Qwen3-8B and Qwen3-30B-A3B; it does not include a fresh Qwen3-235B Math core pass.

Batch-sweep files:

- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv` has completed Qwen3-235B SWE `suffix_k32` and `eagle3_k3` rows at larger batch sizes.
- Qwen3-235B baseline/PARD/PARD-2 rows in the corresponding status snapshot were still running or live telemetry only, with no final `breakdown.json` rows in that refresh. That is not OOM evidence.
- `docs/lyris_swebench_osl32k_batch_sweep_metrics_20260612.csv` is an early partial file for Qwen3-8B/Qwen3-30B-A3B; many rows were still running or replaced when captured.

## Larger Batch Evidence

Completed Qwen3-235B larger-batch rows from the older `/lustre` sweep:

| Dataset | Method | Batch sizes completed | tok/s/GPU range | Acceptance range |
| --- | --- | --- | --- | --- |
| SWE full | `eagle3_k3` | `4/8/16/32` | `15.624` to `126.831` | `46.474%` to `57.294%` |
| SWE full | `suffix_k32` | `4/8/16/32` | `27.971` to `223.154` | `76.244%` to `81.195%` |
| SWE verified | `eagle3_k3` | `4/8/16/32` | `18.654` to `122.584` | `51.454%` to `62.342%` |
| SWE verified | `suffix_k32` | `4/8/16/32` | `48.656` to `223.252` | `79.924%` to `86.201%` |

Conclusion: for Qwen3-235B suffix and EAGLE, batch sizes `4/8/16/32` are proven runnable in existing `/lustre` artifacts. They are absent from the latest temp matrix because that matrix was scoped to batch `1/2`.

## Retry Plan

Planned bounded retries, not submitted because `/lustre` cannot create new log directories:

| Domain | Model | Cells | Split |
| --- | --- | --- | --- |
| SWE | Qwen3-235B-A22B | temp `0.0/1.0` x baseline/suffix/EAGLE-3 x batch `1/2` | 12 separate jobs |
| Math | Qwen3-235B-A22B | temp `0.0/1.0` baseline batch `1` | 2 separate jobs |

The intended log root was `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark/vllm-runs/clean_status_20260617`. I stopped before submission to avoid writing under `/home` or creating misleading failed jobs.

## Sources

- `docs/vllm_standalone_clean_primary_20260617.csv`
- `docs/vllm_standalone_clean_supplemental_20260617.csv`
- `docs/lyris_math500_osl32k_temp01_home_retry_metrics_live_20260616.csv`
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics_live.csv`
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_status_live.csv`
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_manifest.csv`
- `docs/lyris_qwen235b_standalone_fast_20260613_metrics.csv`
- `docs/lyris_qwen235b_standalone_fast_20260613_status.csv`
- `docs/lyris_qwen235b_standalone_temp1rl_20260614_metrics.csv`
- `docs/lyris_qwen235b_standalone_temp1rl_20260614_status.csv`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_status_20260612.md`
- `docs/lyris_swebench_osl32k_batch_sweep_metrics_20260612.csv`
- `docs/lyris_swebench_osl32k_batch_sweep_launch_20260612.md`
