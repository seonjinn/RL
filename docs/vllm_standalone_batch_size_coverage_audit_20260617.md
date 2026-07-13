# vLLM Standalone Batch Size Coverage Audit

Updated: `2026-06-17`

Scope: structured local CSV artifacts under `docs/` whose metric/status/manifest/job schemas describe Qwen vLLM standalone SWEBench or Math500 runs. Target batch sizes were `1`, `2`, `4`, `8`, `16`, and `32`. Synthetic and OpenMath-only standalone files were excluded from this SWE/Math500 audit.

## Short Answer

No. `docs/vllm_standalone_clean_status_20260617.*` does not include every locally available tested batch size. It correctly states that the latest temp `0/1` core matrix is scoped to batch `1/2`, but it is not a complete inventory of older or adjacent standalone artifacts.

Final metric evidence in the scoped local artifacts covers Math batches `1/2` only, and SWE batches `1/2/4/8/16/32`. No local Math500 final metrics for batches `4/8/16/32` were found in scope.

The most direct clean-status gap is `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv`: final Qwen3-235B SWE `suffix_k32` and `eagle3_k3` rows exist at batch `2/4/8/16/32`, but clean status summarizes only `4/8/16/32`.

## Audit Counts

- Coverage rows written: `1951` from `50` structured source files.
- Evidence state counts: final `505`, partial `184`, manifest-only `1144`, failed `118`.
- Batch row counts: batch1 `615`, batch2 `582`, batch4 `197`, batch8 `192`, batch16 `190`, batch32 `175`.
- Domain row counts: Math `222`, SWE `1729`.
- Clean-status match counts: exact row `135`, same source/batch only `168`, omitted `1648`.
- Final metric clean-status match counts: exact row `87`, same source/batch only `32`, omitted `386`.

## Final Metric Batch Coverage

| Domain | Final metric batch sizes found | Clean-status interpretation |
| --- | --- | --- |
| Math | `1/2` | Clean status includes selected latest/archived batch `1/2` rows, but omits older Math500/OCI batch `1/2` sources. No scoped Math500 batch `4/8/16/32` final metrics were found. |
| SWE | `1/2/4/8/16/32` | Clean status includes latest core batch `1/2` plus selected Qwen3-235B batch `4/8/16/32`, but omits several tested SWE sources and omits batch `2` from the Qwen3-235B batch sweep summary. |

## Clean-Status Gaps By Final-Metric Source

| Source file | Domain | Final batches | Included by clean status | Omitted final batches | Omitted final rows |
| --- | --- | --- | --- | --- | --- |
| `docs/lyris_math500_osl32k_metrics_20260612.csv` | Math | `1/2` | `none` | `1/2` | `12` |
| `docs/lyris_qwen235b_standalone_fast_20260613_metrics.csv` | Math/SWE | `1/2/4/8/16/32` | `1/2` | `4/8/16/32` | `64` |
| `docs/lyris_qwen235b_standalone_temp1rl_20260614_metrics.csv` | Math/SWE | `1/2/4/8/16/32` | `1/2` | `4/8/16/32` | `64` |
| `docs/lyris_qwen235b_suffix_metrics_20260612.csv` | SWE | `1/2` | `none` | `1/2` | `42` |
| `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv` | SWE | `2/4/8/16/32` | `4/8/16/32` | `2` | `4` |
| `docs/lyris_specdec_32k_metrics_20260612.csv` | SWE | `1/2/4` | `none` | `1/2/4` | `54` |
| `docs/lyris_swebench_longosl_metrics_20260612.csv` | SWE | `1/2/4/8/16/32` | `none` | `1/2/4/8/16/32` | `70` |
| `docs/lyris_swebench_osl32k_batch_sweep_metrics_20260612.csv` | SWE | `4/8/16` | `none` | `4/8/16` | `6` |
| `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics.csv` | SWE | `1/2` | `none` | `1/2` | `10` |
| `docs/lyris_swebench_osl32k_temp01_home_retry_metrics_live_20260616.csv` | SWE | `1/2` | `none` | `1/2` | `21` |
| `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_r4_noprof_metrics.csv` | Math/SWE | `1/2` | `none` | `1/2` | `16` |
| `docs/oci_qwen235b_math500_drafter_k11_metrics_20260613.csv` | Math | `1/2` | `none` | `1/2` | `5` |
| `docs/oci_qwen235b_math500_drafter_k9_metrics_20260613.csv` | Math | `1/2` | `none` | `1/2` | `5` |
| `docs/oci_qwen235b_math500_osl32k_metrics_20260613.csv` | Math | `1/2` | `none` | `1/2` | `5` |
| `docs/oci_qwen235b_math500_suffix_py312_retry1_metrics_20260613.csv` | Math | `1/2` | `none` | `1/2` | `2` |
| `docs/oci_swebench_osl16k_smoke_metrics_20260612.csv` | SWE | `1/2` | `none` | `1/2` | `6` |

## Specific Batch Answer

- Batch `1`: final Math and SWE rows exist locally. Clean status includes selected latest/core rows, but omits older Math500, SWE long-OSL/specdec, OCI Math500, and OCI Qwen8 temp01 sources.
- Batch `2`: final Math and SWE rows exist locally. Clean status includes selected latest/core rows, but omits batch `2` from the Qwen3-235B SWE batch-sweep summary and omits several older/OCI sources.
- Batch `4`: final SWE rows exist locally. Clean status includes selected Qwen3-235B SWE batch-sweep rows, but omits Qwen3-8B/Qwen3-30B-A3B SWE sweep and long-OSL/specdec rows. No scoped Math500 final batch `4` rows were found.
- Batch `8`: final SWE rows exist locally. Clean status includes selected Qwen3-235B SWE batch-sweep rows, but omits other SWE sources. No scoped Math500 final batch `8` rows were found.
- Batch `16`: final SWE rows exist locally. Clean status includes selected Qwen3-235B SWE batch-sweep rows, but omits other SWE sources. No scoped Math500 final batch `16` rows were found.
- Batch `32`: final SWE rows exist locally in Qwen3-235B and long-OSL artifacts. Clean status includes selected Qwen3-235B SWE `4/8/16/32` rows, but omits other batch `32` sources. No scoped Math500 final batch `32` rows were found.

## Files Written

- `docs/vllm_standalone_batch_size_coverage_audit_20260617.csv`
- `docs/vllm_standalone_batch_size_coverage_audit_20260617.md`

