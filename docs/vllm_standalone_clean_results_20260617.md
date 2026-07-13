# Clean SpecDec Benchmark Results

Updated: `2026-06-22 01:57 PDT`

Scope: final JSON-backed standalone vLLM rows plus parsed NeMo-RL step metrics already present locally. Standalone primary table uses the latest 2026-06-16 Lyris core matrix plus the live 2026-06-17 OCI-HSG qmath batch sweep, with ISL/OSL `4096/32768`. NeMo-RL rows use parsed summaries and compute baseline-relative speedups within the same run group when a matched baseline exists.

## Key Findings

- SWE: `suffix_k32` is the strongest current method; it remains positive at temp 1.0, but Qwen3-32B and Qwen3-235B drop sharply versus temp 0.
- Math: `suffix_k32` dominates Qwen3-8B and Qwen3-30B; PARD K3/K5 is positive on Qwen3-30B, mixed on Qwen3-8B, and PARD-2 Qwen3-8B temp 1.0 is below baseline.
- qmath sweep: OCI-HSG is fully collected: 112/112 jobs completed, 112 breakdowns parsed, 0 failed.
- Batch detail: batch `4/8/16/32` rows are now included separately for qmath plus available SWE OSL32K batch-sweep artifacts; blank temp or speedup means the original artifact did not have matched temp/baseline metadata.
- EAGLE-3 is useful but smaller: strong on Qwen3-32B SWE, modest on Qwen3-30B, and weak on Math Qwen3-30B in the latest core matrix.
- NeMo-RL PerfCfg: latest Lyris performance-config rows are included; Qwen3-30B/Qwen3-32B produce usable metrics, while Qwen3-235B is still blocked at baseline Step 1 policy training by NCCL watchdog timeouts.
- NeMo-RL historical rows: Math/SWE gates remain available below, but they are separated from the current Lyris performance-config matrix.
- Online drafter: Qwen3-32B online PARD-2 completed as a correctness gate but is not a speedup win; Qwen3-8B online PARD-2 is also slower than static/baseline in the current parsed comparison.

## Standalone Primary Results

| Domain | Temp | Model | Method | Rows | Batches | ISL | OSL | Baseline tok/s/GPU | tok/s/GPU | Speedup | Acceptance | Mean accept len | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 0.0 | Qwen3-235B-A22B | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 26.9 | 56.3 | 2.21x | 60.6% | 2.82 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | pard2_k1 | 4 | 4/8/16/32 | 4096 | 32768 | 26.9 | 23.0 | 0.83x | 4.5% | 1.05 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 26.9 | 54.8 | 2.22x | 53.0% | 3.65 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 26.9 | 93.3 | 4.49x | 83.7% | 7.44 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 29.2 | 31.1 | 1.08x | 10.2% | 1.31 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-30B-A3B | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 244.5 | 254.2 | 1.04x | 9.3% | 1.28 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | pard_k3 | 1 | 1 | 4096 | 32768 | 29.2 | 37.3 | 1.92x | 70.7% | 3.12 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-30B-A3B | pard_k5 | 1 | 1 | 4096 | 32768 | 29.2 | 42.5 | 2.20x | 52.2% | 3.61 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-30B-A3B | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 244.5 | 269.2 | 1.27x | 45.0% | 3.25 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 29.2 | 206.2 | 7.15x | 88.6% | 8.40 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-30B-A3B | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 244.5 | 963.0 | 4.82x | 85.9% | 8.27 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 55.8 | 94.8 | 1.77x | 63.3% | 2.90 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-8B | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 430.2 | 684.0 | 1.48x | 70.7% | 3.12 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | pard2_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 430.2 | 163.9 | 0.39x | 0.1% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | pard_k3 | 1 | 1 | 4096 | 32768 | 55.8 | 38.2 | 1.03x | 50.7% | 2.52 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-8B | pard_k5 | 1 | 1 | 4096 | 32768 | 55.8 | 40.4 | 1.09x | 34.5% | 2.73 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-8B | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 430.2 | 407.6 | 0.98x | 47.8% | 3.39 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 55.8 | 312.9 | 5.66x | 85.9% | 8.09 | lyris_core_20260616 |
| Math | 0.0 | Qwen3-8B | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 430.2 | 1371.5 | 3.65x | 85.6% | 8.39 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 28.0 | 43.0 | 1.71x | 47.4% | 2.42 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | pard2_k1 | 4 | 4/8/16/32 | 4096 | 32768 | 28.0 | 23.5 | 0.88x | 1.9% | 1.02 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 28.0 | 29.7 | 1.16x | 26.9% | 2.34 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 28.0 | 66.8 | 3.23x | 68.4% | 5.21 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 29.1 | 32.5 | 1.12x | 10.7% | 1.32 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-30B-A3B | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 231.1 | 226.3 | 0.95x | 10.5% | 1.32 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | pard_k3 | 1 | 1 | 4096 | 32768 | 29.1 | 36.3 | 1.87x | 63.7% | 2.91 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-30B-A3B | pard_k5 | 1 | 1 | 4096 | 32768 | 29.1 | 36.9 | 1.91x | 41.1% | 3.05 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-30B-A3B | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 231.1 | 174.1 | 0.73x | 38.2% | 2.91 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 29.1 | 162.8 | 6.06x | 83.9% | 7.16 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-30B-A3B | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 231.1 | 750.8 | 3.92x | 80.8% | 6.96 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 54.4 | 99.6 | 1.93x | 68.6% | 3.06 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-8B | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 401.6 | 417.1 | 0.96x | 52.6% | 2.58 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | pard2_k3 | 2 | 1/2 | 4096 | 32768 | 54.4 | 36.5 | 0.70x | 25.3% | 1.76 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-8B | pard2_k5 | 2 | 1/2 | 4096 | 32768 | 54.4 | 31.0 | 0.62x | 12.4% | 1.62 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-8B | pard2_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 401.6 | 161.3 | 0.38x | 0.2% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | pard_k3 | 2 | 1/2 | 4096 | 32768 | 54.4 | 69.7 | 1.29x | 70.7% | 3.12 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-8B | pard_k5 | 2 | 1/2 | 4096 | 32768 | 54.4 | 57.5 | 1.05x | 33.4% | 2.67 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-8B | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 401.6 | 233.9 | 0.58x | 35.5% | 2.77 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 54.4 | 272.3 | 5.32x | 80.6% | 6.71 | lyris_core_20260616 |
| Math | 1.0 | Qwen3-8B | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 401.6 | 870.8 | 2.17x | 75.9% | 6.11 | oci_qmath_bsweep_20260617 |
| SWE | 0.0 | Qwen3-235B-A22B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 3.1 | 6.9 | 2.33x | 54.6% | 2.64 | lyris_core_20260616 |
| SWE | 0.0 | Qwen3-235B-A22B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 3.1 | 17.3 | 5.64x | 79.2% | 6.26 | lyris_core_20260616 |
| SWE | 0.0 | Qwen3-30B-A3B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 29.8 | 33.0 | 1.11x | 12.3% | 1.37 | lyris_core_20260616 |
| SWE | 0.0 | Qwen3-30B-A3B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 29.8 | 243.3 | 8.35x | 92.3% | 9.94 | lyris_core_20260616 |
| SWE | 0.0 | Qwen3-32B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 12.3 | 29.3 | 2.38x | 69.1% | 3.07 | lyris_core_20260616 |
| SWE | 0.0 | Qwen3-32B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 12.3 | 80.2 | 6.53x | 84.9% | 8.06 | lyris_core_20260616 |
| SWE | 1.0 | Qwen3-235B-A22B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 3.1 | 4.8 | 1.55x | 24.1% | 1.72 | lyris_core_20260616 |
| SWE | 1.0 | Qwen3-235B-A22B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 3.1 | 6.9 | 2.37x | 52.8% | 2.89 | lyris_core_20260616 |
| SWE | 1.0 | Qwen3-30B-A3B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 29.2 | 30.8 | 1.07x | 11.0% | 1.33 | lyris_core_20260616 |
| SWE | 1.0 | Qwen3-30B-A3B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 29.2 | 237.7 | 8.26x | 90.8% | 9.37 | lyris_core_20260616 |
| SWE | 1.0 | Qwen3-32B | eagle3_k3 | 2 | 1/2 | 4096 | 32768 | 12.4 | 22.8 | 1.86x | 49.0% | 2.47 | lyris_core_20260616 |
| SWE | 1.0 | Qwen3-32B | suffix_k32 | 2 | 1/2 | 4096 | 32768 | 12.4 | 27.9 | 2.17x | 52.2% | 3.21 | lyris_core_20260616 |

## Standalone Batch 4/8/16/32 Summary

| Domain | Temp | Model | Dataset | Method | Rows | Batches | ISL | OSL | GPUs | tok/s/GPU | Speedup | Acceptance | Mean accept len | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | baseline | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 26.9 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 56.3 | 2.21x | 60.6% | 2.82 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 23.0 | 0.83x | 4.5% | 1.05 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 54.8 | 2.22x | 53.0% | 3.65 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 93.3 | 4.49x | 83.7% | 7.44 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | baseline | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 244.5 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 254.2 | 1.04x | 9.3% | 1.28 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 269.2 | 1.27x | 45.0% | 3.25 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 963.0 | 4.82x | 85.9% | 8.27 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | baseline | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 430.2 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 684.0 | 1.48x | 70.7% | 3.12 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard2_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 163.9 | 0.39x | 0.1% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 407.6 | 0.98x | 47.8% | 3.39 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 1371.5 | 3.65x | 85.6% | 8.39 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | baseline | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 28.0 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 43.0 | 1.71x | 47.4% | 2.42 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 23.5 | 0.88x | 1.9% | 1.02 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 29.7 | 1.16x | 26.9% | 2.34 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 66.8 | 3.23x | 68.4% | 5.21 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | baseline | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 231.1 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 226.3 | 0.95x | 10.5% | 1.32 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 174.1 | 0.73x | 38.2% | 2.91 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 750.8 | 3.92x | 80.8% | 6.96 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | baseline | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 401.6 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 417.1 | 0.96x | 52.6% | 2.58 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard2_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 161.3 | 0.38x | 0.2% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard_k5 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 233.9 | 0.58x | 35.5% | 2.77 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 870.8 | 2.17x | 75.9% | 6.11 | oci_qmath_bsweep_20260617 |
| SWE |  | Qwen3-235B-A22B | SWE full | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 58.9 |  | 53.7% | 2.61 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 106.2 |  | 79.4% | 6.39 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 59.5 |  | 55.8% | 2.68 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 4 | 115.9 |  | 82.3% | 6.95 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | pard_k5 | 1 | 4 | 4096 | 32768 | 1 | 122.8 |  | 44.0% | 3.20 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 1 | 1332.3 |  | 89.5% | 9.46 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE verified | suffix_k32 | 3 | 4/8/16 | 4096 | 32768 | 1 | 1044.5 |  | 88.4% | 9.13 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | baseline | 3 | 4/8/16 | 4096 | 32768 | 1 | 348.3 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 1 | 678.5 | 1.41x | 63.3% | 2.90 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 1 | 1340.0 | 3.67x | 85.2% | 9.02 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | baseline | 3 | 4/8/16 | 4096 | 32768 | 1 | 348.4 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | eagle3_k3 | 4 | 4/8/16/32 | 4096 | 32768 | 1 | 728.3 | 1.45x | 64.0% | 2.92 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | suffix_k32 | 4 | 4/8/16/32 | 4096 | 32768 | 1 | 1811.7 | 4.91x | 91.1% | 10.12 | lyris_swe_longosl_osl32k_20260612 |

## Standalone Batch 4/8/16/32 Row Details

| Domain | Temp | Model | Dataset | Method | Batch | ISL | OSL | GPUs | tok/s/GPU | Speedup | Acceptance | Mean accept len | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | baseline | 4 | 4096 | 32768 | 4 | 7.9 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 4 | 4096 | 32768 | 4 | 15.9 | 2.03x | 57.0% | 2.71 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 4 | 4096 | 32768 | 4 | 5.0 | 0.63x | 3.2% | 1.03 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 4 | 4096 | 32768 | 4 | 17.9 | 2.28x | 55.7% | 3.79 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 4 | 4096 | 32768 | 4 | 51.6 | 6.57x | 89.8% | 8.80 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | baseline | 8 | 4096 | 32768 | 4 | 15.5 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 8 | 4096 | 32768 | 4 | 34.3 | 2.21x | 60.9% | 2.83 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 8 | 4096 | 32768 | 4 | 12.8 | 0.83x | 4.3% | 1.04 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 8 | 4096 | 32768 | 4 | 36.5 | 2.35x | 53.9% | 3.69 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 8 | 4096 | 32768 | 4 | 61.1 | 3.94x | 86.7% | 7.79 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | baseline | 16 | 4096 | 32768 | 4 | 23.5 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 16 | 4096 | 32768 | 4 | 65.4 | 2.78x | 65.3% | 2.96 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 16 | 4096 | 32768 | 4 | 25.1 | 1.07x | 5.1% | 1.05 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 16 | 4096 | 32768 | 4 | 59.3 | 2.52x | 50.8% | 3.54 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 16 | 4096 | 32768 | 4 | 121.5 | 5.16x | 80.4% | 6.78 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | baseline | 32 | 4096 | 32768 | 4 | 60.8 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 32 | 4096 | 32768 | 4 | 109.6 | 1.80x | 59.1% | 2.77 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 32 | 4096 | 32768 | 4 | 49.0 | 0.81x | 5.6% | 1.06 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 32 | 4096 | 32768 | 4 | 105.6 | 1.74x | 51.4% | 3.57 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 32 | 4096 | 32768 | 4 | 139.1 | 2.29x | 77.8% | 6.40 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | baseline | 4 | 4096 | 32768 | 4 | 70.5 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 4 | 4096 | 32768 | 4 | 74.0 | 1.05x | 8.7% | 1.26 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 4 | 4096 | 32768 | 4 | 97.3 | 1.38x | 42.4% | 3.12 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 4 | 4096 | 32768 | 4 | 483.2 | 6.85x | 91.7% | 9.32 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | baseline | 8 | 4096 | 32768 | 4 | 123.6 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 8 | 4096 | 32768 | 4 | 138.9 | 1.12x | 9.1% | 1.27 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 8 | 4096 | 32768 | 4 | 187.6 | 1.52x | 44.6% | 3.23 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 8 | 4096 | 32768 | 4 | 637.6 | 5.16x | 85.5% | 8.19 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | baseline | 16 | 4096 | 32768 | 4 | 276.8 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 16 | 4096 | 32768 | 4 | 253.7 | 0.92x | 9.5% | 1.29 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 16 | 4096 | 32768 | 4 | 368.6 | 1.33x | 47.0% | 3.35 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 16 | 4096 | 32768 | 4 | 1139.9 | 4.12x | 84.6% | 8.05 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | baseline | 32 | 4096 | 32768 | 4 | 507.1 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 32 | 4096 | 32768 | 4 | 550.3 | 1.09x | 9.9% | 1.30 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 32 | 4096 | 32768 | 4 | 423.2 | 0.83x | 46.1% | 3.31 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 32 | 4096 | 32768 | 4 | 1591.2 | 3.14x | 82.0% | 7.53 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | baseline | 4 | 4096 | 32768 | 4 | 151.8 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 4 | 4096 | 32768 | 4 | 193.0 | 1.27x | 68.4% | 3.05 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard2_k5 | 4 | 4096 | 32768 | 4 | 61.2 | 0.40x | 0.1% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard_k5 | 4 | 4096 | 32768 | 4 | 154.5 | 1.02x | 46.6% | 3.33 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | suffix_k32 | 4 | 4096 | 32768 | 4 | 663.5 | 4.37x | 89.6% | 9.05 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | baseline | 8 | 4096 | 32768 | 4 | 304.8 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 8 | 4096 | 32768 | 4 | 402.0 | 1.32x | 72.3% | 3.17 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard2_k5 | 8 | 4096 | 32768 | 4 | 120.3 | 0.39x | 0.2% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard_k5 | 8 | 4096 | 32768 | 4 | 320.5 | 1.05x | 52.0% | 3.60 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | suffix_k32 | 8 | 4096 | 32768 | 4 | 1476.5 | 4.84x | 87.6% | 8.92 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | baseline | 16 | 4096 | 32768 | 4 | 547.6 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 16 | 4096 | 32768 | 4 | 835.9 | 1.53x | 71.1% | 3.13 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard2_k5 | 16 | 4096 | 32768 | 4 | 230.5 | 0.42x | 0.1% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard_k5 | 16 | 4096 | 32768 | 4 | 536.8 | 0.98x | 49.0% | 3.45 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | suffix_k32 | 16 | 4096 | 32768 | 4 | 1617.8 | 2.95x | 83.7% | 8.07 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | baseline | 32 | 4096 | 32768 | 4 | 716.7 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 32 | 4096 | 32768 | 4 | 1305.1 | 1.82x | 70.9% | 3.13 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard2_k5 | 32 | 4096 | 32768 | 4 | 243.4 | 0.34x | 0.2% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | pard_k5 | 32 | 4096 | 32768 | 4 | 618.8 | 0.86x | 43.6% | 3.18 | oci_qmath_bsweep_20260617 |
| Math | 0.0 | Qwen3-8B | MATH-500 | suffix_k32 | 32 | 4096 | 32768 | 4 | 1728.3 | 2.41x | 81.5% | 7.51 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | baseline | 4 | 4096 | 32768 | 4 | 7.7 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 4 | 4096 | 32768 | 4 | 13.7 | 1.75x | 46.4% | 2.39 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 4 | 4096 | 32768 | 4 | 6.4 | 0.81x | 1.2% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 4 | 4096 | 32768 | 4 | 7.3 | 0.92x | 17.9% | 1.89 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 4 | 4096 | 32768 | 4 | 35.4 | 4.50x | 74.3% | 6.16 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | baseline | 8 | 4096 | 32768 | 4 | 14.7 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 8 | 4096 | 32768 | 4 | 25.2 | 1.62x | 50.2% | 2.51 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 8 | 4096 | 32768 | 4 | 12.4 | 0.80x | 1.5% | 1.02 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 8 | 4096 | 32768 | 4 | 19.9 | 1.28x | 30.7% | 2.54 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 8 | 4096 | 32768 | 4 | 54.6 | 3.52x | 70.0% | 5.52 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | baseline | 16 | 4096 | 32768 | 4 | 29.7 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 16 | 4096 | 32768 | 4 | 48.4 | 2.06x | 46.5% | 2.40 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 16 | 4096 | 32768 | 4 | 24.9 | 1.06x | 2.7% | 1.03 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 16 | 4096 | 32768 | 4 | 35.9 | 1.53x | 29.8% | 2.49 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 16 | 4096 | 32768 | 4 | 76.2 | 3.24x | 63.2% | 4.45 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | baseline | 32 | 4096 | 32768 | 4 | 59.8 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | eagle3_k3 | 32 | 4096 | 32768 | 4 | 84.6 | 1.39x | 46.5% | 2.40 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard2_k1 | 32 | 4096 | 32768 | 4 | 50.3 | 0.83x | 2.3% | 1.02 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | pard_k5 | 32 | 4096 | 32768 | 4 | 55.7 | 0.92x | 29.0% | 2.45 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-235B-A22B | MATH-500 | suffix_k32 | 32 | 4096 | 32768 | 4 | 100.8 | 1.66x | 66.1% | 4.71 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | baseline | 4 | 4096 | 32768 | 4 | 72.4 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 4 | 4096 | 32768 | 4 | 66.9 | 0.95x | 11.0% | 1.33 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 4 | 4096 | 32768 | 4 | 51.7 | 0.73x | 28.5% | 2.43 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 4 | 4096 | 32768 | 4 | 367.5 | 5.21x | 86.8% | 7.78 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | baseline | 8 | 4096 | 32768 | 4 | 130.7 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 8 | 4096 | 32768 | 4 | 128.9 | 1.04x | 10.2% | 1.31 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 8 | 4096 | 32768 | 4 | 99.1 | 0.80x | 36.6% | 2.83 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 8 | 4096 | 32768 | 4 | 664.0 | 5.37x | 81.9% | 7.22 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | baseline | 16 | 4096 | 32768 | 4 | 248.1 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 16 | 4096 | 32768 | 4 | 241.2 | 0.87x | 9.9% | 1.30 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 16 | 4096 | 32768 | 4 | 193.7 | 0.70x | 42.0% | 3.10 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 16 | 4096 | 32768 | 4 | 730.1 | 2.64x | 77.9% | 6.48 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | baseline | 32 | 4096 | 32768 | 4 | 473.1 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | eagle3_k3 | 32 | 4096 | 32768 | 4 | 468.3 | 0.92x | 10.9% | 1.33 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | pard_k5 | 32 | 4096 | 32768 | 4 | 351.9 | 0.69x | 45.9% | 3.29 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-30B-A3B | MATH-500 | suffix_k32 | 32 | 4096 | 32768 | 4 | 1241.7 | 2.45x | 76.5% | 6.38 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | baseline | 4 | 4096 | 32768 | 4 | 123.3 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 4 | 4096 | 32768 | 4 | 144.9 | 0.96x | 49.0% | 2.47 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard2_k5 | 4 | 4096 | 32768 | 4 | 59.5 | 0.39x | 0.1% | 1.00 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard_k5 | 4 | 4096 | 32768 | 4 | 95.2 | 0.63x | 32.7% | 2.64 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | suffix_k32 | 4 | 4096 | 32768 | 4 | 442.2 | 2.91x | 78.7% | 6.55 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | baseline | 8 | 4096 | 32768 | 4 | 218.6 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 8 | 4096 | 32768 | 4 | 263.7 | 0.87x | 60.7% | 2.82 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard2_k5 | 8 | 4096 | 32768 | 4 | 115.1 | 0.38x | 0.3% | 1.02 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard_k5 | 8 | 4096 | 32768 | 4 | 237.0 | 0.78x | 41.6% | 3.08 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | suffix_k32 | 8 | 4096 | 32768 | 4 | 569.8 | 1.87x | 79.0% | 6.32 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | baseline | 16 | 4096 | 32768 | 4 | 544.3 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 16 | 4096 | 32768 | 4 | 633.1 | 1.16x | 55.2% | 2.66 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard2_k5 | 16 | 4096 | 32768 | 4 | 226.0 | 0.41x | 0.1% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard_k5 | 16 | 4096 | 32768 | 4 | 199.8 | 0.36x | 31.5% | 2.58 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | suffix_k32 | 16 | 4096 | 32768 | 4 | 1046.0 | 1.91x | 73.4% | 5.86 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | baseline | 32 | 4096 | 32768 | 4 | 720.3 | 1.00x |  |  | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | eagle3_k3 | 32 | 4096 | 32768 | 4 | 626.5 | 0.87x | 45.5% | 2.36 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard2_k5 | 32 | 4096 | 32768 | 4 | 244.4 | 0.34x | 0.2% | 1.01 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | pard_k5 | 32 | 4096 | 32768 | 4 | 403.5 | 0.56x | 36.1% | 2.81 | oci_qmath_bsweep_20260617 |
| Math | 1.0 | Qwen3-8B | MATH-500 | suffix_k32 | 32 | 4096 | 32768 | 4 | 1425.0 | 1.99x | 72.3% | 5.73 | oci_qmath_bsweep_20260617 |
| SWE |  | Qwen3-235B-A22B | SWE full | eagle3_k3 | 4 | 4096 | 32768 | 4 | 15.6 |  | 46.5% | 2.39 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | suffix_k32 | 4 | 4096 | 32768 | 4 | 28.0 |  | 76.2% | 5.42 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | eagle3_k3 | 8 | 4096 | 32768 | 4 | 29.8 |  | 54.3% | 2.63 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | suffix_k32 | 8 | 4096 | 32768 | 4 | 63.6 |  | 81.2% | 6.84 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | eagle3_k3 | 16 | 4096 | 32768 | 4 | 63.4 |  | 57.3% | 2.72 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | suffix_k32 | 16 | 4096 | 32768 | 4 | 110.1 |  | 81.1% | 6.87 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | eagle3_k3 | 32 | 4096 | 32768 | 4 | 126.8 |  | 56.9% | 2.71 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE full | suffix_k32 | 32 | 4096 | 32768 | 4 | 223.2 |  | 78.9% | 6.46 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | eagle3_k3 | 4 | 4096 | 32768 | 4 | 18.7 |  | 55.3% | 2.66 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | suffix_k32 | 4 | 4096 | 32768 | 4 | 48.7 |  | 86.2% | 7.62 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | eagle3_k3 | 8 | 4096 | 32768 | 4 | 36.5 |  | 62.3% | 2.87 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | suffix_k32 | 8 | 4096 | 32768 | 4 | 67.1 |  | 80.1% | 6.35 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | eagle3_k3 | 16 | 4096 | 32768 | 4 | 60.4 |  | 51.5% | 2.54 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | suffix_k32 | 16 | 4096 | 32768 | 4 | 124.5 |  | 79.9% | 6.52 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | eagle3_k3 | 32 | 4096 | 32768 | 4 | 122.6 |  | 54.3% | 2.63 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-235B-A22B | SWE verified | suffix_k32 | 32 | 4096 | 32768 | 4 | 223.3 |  | 82.9% | 7.31 | lyris_qwen235b_swe_bsweep_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | pard_k5 | 4 | 4096 | 32768 | 1 | 122.8 |  | 44.0% | 3.20 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | suffix_k32 | 4 | 4096 | 32768 | 1 | 595.6 |  | 89.9% | 9.50 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | suffix_k32 | 8 | 4096 | 32768 | 1 | 1010.1 |  | 90.0% | 9.60 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | suffix_k32 | 16 | 4096 | 32768 | 1 | 1443.0 |  | 89.6% | 9.22 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE full | suffix_k32 | 32 | 4096 | 32768 | 1 | 2280.5 |  | 88.5% | 9.54 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE verified | suffix_k32 | 4 | 4096 | 32768 | 1 | 390.6 |  | 87.1% | 8.85 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE verified | suffix_k32 | 8 | 4096 | 32768 | 1 | 941.6 |  | 88.6% | 9.17 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-30B-A3B | SWE verified | suffix_k32 | 16 | 4096 | 32768 | 1 | 1801.4 |  | 89.5% | 9.36 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | baseline | 4 | 4096 | 32768 | 1 | 148.9 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | eagle3_k3 | 4 | 4096 | 32768 | 1 | 255.0 | 1.71x | 63.9% | 2.92 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | suffix_k32 | 4 | 4096 | 32768 | 1 | 1089.8 | 7.32x | 94.0% | 10.63 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | baseline | 8 | 4096 | 32768 | 1 | 290.9 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | eagle3_k3 | 8 | 4096 | 32768 | 1 | 391.3 | 1.34x | 60.5% | 2.82 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | suffix_k32 | 8 | 4096 | 32768 | 1 | 344.4 | 1.18x | 69.5% | 6.52 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | baseline | 16 | 4096 | 32768 | 1 | 605.1 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | eagle3_k3 | 16 | 4096 | 32768 | 1 | 717.6 | 1.19x | 64.1% | 2.92 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | suffix_k32 | 16 | 4096 | 32768 | 1 | 1512.8 | 2.50x | 87.5% | 9.12 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | eagle3_k3 | 32 | 4096 | 32768 | 1 | 1350.1 |  | 64.6% | 2.94 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE full | suffix_k32 | 32 | 4096 | 32768 | 1 | 2412.8 |  | 89.9% | 9.82 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | baseline | 4 | 4096 | 32768 | 1 | 148.8 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | eagle3_k3 | 4 | 4096 | 32768 | 1 | 229.5 | 1.54x | 63.1% | 2.89 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | suffix_k32 | 4 | 4096 | 32768 | 1 | 846.2 | 5.69x | 92.5% | 10.39 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | baseline | 8 | 4096 | 32768 | 1 | 295.1 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | eagle3_k3 | 8 | 4096 | 32768 | 1 | 462.8 | 1.57x | 63.0% | 2.89 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | suffix_k32 | 8 | 4096 | 32768 | 1 | 1397.8 | 4.74x | 91.9% | 10.24 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | baseline | 16 | 4096 | 32768 | 1 | 601.3 | 1.00x |  |  | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | eagle3_k3 | 16 | 4096 | 32768 | 1 | 738.4 | 1.23x | 64.9% | 2.95 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | suffix_k32 | 16 | 4096 | 32768 | 1 | 2597.4 | 4.32x | 92.1% | 10.29 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | eagle3_k3 | 32 | 4096 | 32768 | 1 | 1482.6 |  | 65.1% | 2.95 | lyris_swe_longosl_osl32k_20260612 |
| SWE |  | Qwen3-8B | SWE verified | suffix_k32 | 32 | 4096 | 32768 | 1 | 2405.2 |  | 88.1% | 9.56 | lyris_swe_longosl_osl32k_20260612 |

## NeMo-RL Results

| Domain | Run Group | Model | Method | Job | Done | Max | OSL | E2E step time | E2E time speedup | Generation time | Generation time speedup | E2E tok/s/GPU | E2E throughput speedup | Gen tok/s/GPU | Generation throughput speedup | Acceptance | Mean accept len | State | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math RL | qwen17B out16k | Qwen3-17B | pard_k3 | 3231517 | 20 | 20 | 16384 | 260.1s |  | 250.8s |  | 132.7 |  | 138.0 |  | 75.6% | 3.27 | complete | qwen17b_out16k_20260616 |
| Math RL | qwen17B out16k | Qwen3-17B | pard_k3 | 3231518 | 0 | 1 |  |  |  |  |  |  |  |  |  | 59.2% | 2.78 | no completed step | qwen17b_out16k_20260616 |
| Math RL | qwen235B reduced64 temp1 OSL256 | Qwen3-235B-A22B | baseline | 3321180 | 9 | 9 | 256 | 80.2s | 1.00x | 35.8s | 1.00x | 2.3 | 1.00x | 5.0 | 1.00x |  |  | complete | mathrl_qwen235b_reduced64_20260615 |
| Math RL | qwen235B reduced64 temp1 OSL256 | Qwen3-235B-A22B | pard_k3 | 3321423 | 9 | 9 | 256 | 69.2s | 1.16x | 23.3s | 1.54x | 2.6 | 1.16x | 7.7 | 1.53x | 61.8% | 2.85 | complete | mathrl_qwen235b_reduced64_20260615 |
| Math RL | qwen235B reduced64 temp1 OSL256 | Qwen3-235B-A22B | pard_k5 | 3321424 | 9 | 9 | 256 | 66.4s | 1.21x | 21.9s | 1.64x | 2.7 | 1.21x | 8.2 | 1.64x | 45.3% | 3.27 | complete | mathrl_qwen235b_reduced64_20260615 |
| Math RL | qwen32 online PARD-2 correctness | Qwen3-32B | online_pard2 | 3345352 | 5 | 5 | 1024 | 1506.5s |  | 1411.5s |  | 23.9 |  | 25.6 |  | 1.7% | 1.05 | complete | mathrl_online_pard2_20260616 |
| Math RL | qwen8 official PARD-2 online comparison | Qwen3-8B | baseline | 3288181 | 9 | 10 | 256 | 23.5s | 1.00x | 7.8s | 1.00x | 74.7 | 1.00x | 225.5 | 1.00x |  |  | complete | qwen8_pard2_online_20260613 |
| Math RL | qwen8 official PARD-2 online comparison | Qwen3-8B | online_pard2 | 3288183 | 9 | 10 | 256 | 35.3s | 0.66x | 13.3s | 0.59x | 50.1 | 0.67x | 132.7 | 0.59x | 2.6% | 1.03 | complete | qwen8_pard2_online_20260613 |
| Math RL | qwen8 official PARD-2 online comparison | Qwen3-8B | static_pard2 | 3288182 | 9 | 10 | 256 | 28.6s | 0.82x | 12.9s | 0.61x | 61.9 | 0.83x | 136.9 | 0.61x | 1.8% | 1.02 | complete | qwen8_pard2_online_20260613 |
| Math RL | step20 temp1 OSL1024 | Qwen3-235B-A22B | baseline | 3334220 | 8 | 20 | 1024 | 196.3s | 1.00x | 127.5s | 1.00x | 12.4 | 1.00x | 17.7 | 1.00x |  |  | partial_usable_with_errors | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-235B-A22B | eagle3 | 3333537 | 14 | 20 | 1024 | 119.6s | 1.64x | 66.2s | 1.93x | 20.8 | 1.67x | 34.2 | 1.93x | 47.4% | 2.42 | partial_usable_with_errors | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-235B-A22B | pard2 | 3333536 | 2 | 3 | 1024 | 336.8s | 0.58x | 155.6s | 0.82x | 7.6 | 0.61x | 14.5 | 0.82x | 5.4% | 1.16 | partial | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-235B-A22B | suffix | 3333717 | 14 | 20 | 1024 | 145.8s | 1.35x | 93.5s | 1.36x | 16.7 | 1.34x | 24.2 | 1.37x | 26.4% | 1.74 | partial_usable_with_errors | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-30B-A3B | baseline | 3334218 | 20 | 20 | 1024 | 290.2s | 1.00x | 211.3s | 1.00x | 125.7 | 1.00x | 172.6 | 1.00x |  |  | complete | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-30B-A3B | eagle3 | 3333528 | 20 | 20 | 1024 | 161.1s | 1.80x | 89.8s | 2.35x | 227.4 | 1.81x | 406.5 | 2.36x | 64.8% | 2.94 | complete | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-30B-A3B | pard | 3333526 | 20 | 20 | 1024 | 178.2s | 1.63x | 100.4s | 2.10x | 205.1 | 1.63x | 363.7 | 2.11x | 50.8% | 3.54 | complete | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-30B-A3B | pard2_k8 | 3333527 | 0 | 1 |  |  |  |  |  |  |  |  |  |  |  | no completed step | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-30B-A3B | suffix | 3333715 | 20 | 20 | 1024 | 179.2s | 1.62x | 108.9s | 1.94x | 204.0 | 1.62x | 335.9 | 1.95x | 35.6% | 2.64 | complete | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | baseline | 3334219 | 16 | 17 | 1024 | 528.2s | 1.00x | 480.8s | 1.00x | 69.3 | 1.00x | 76.1 | 1.00x |  |  | partial | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | eagle3 | 3333533 | 20 | 20 | 1024 | 297.6s | 1.78x | 248.2s | 1.94x | 122.6 | 1.77x | 147.0 | 1.93x | 46.7% | 2.40 | complete | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | online_pard2 | 3337769 | 0 | 1 |  |  |  |  |  |  |  |  |  |  |  | no completed step | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | pard | 3333531 | 20 | 20 | 1024 | 287.8s | 1.84x | 237.6s | 2.02x | 126.8 | 1.83x | 153.6 | 2.02x | 46.8% | 3.34 | complete | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | pard2_k14 | 3334113 | 10 | 11 | 1024 | 741.5s | 0.71x | 690.7s | 0.70x | 49.4 | 0.71x | 53.0 | 0.70x | 1.7% | 1.05 | partial | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | pard2_k8 | 3333532 | 0 | 1 |  |  |  |  |  |  |  |  |  |  |  | no completed step | mathrl_multimodel_20260616 |
| Math RL | step20 temp1 OSL1024 | Qwen3-32B | suffix | 3333716 | 18 | 19 | 1024 | 322.2s | 1.64x | 271.3s | 1.77x | 113.4 | 1.64x | 134.7 | 1.77x | 29.9% | 2.23 | partial | mathrl_multimodel_20260616 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-30B-A3B | Eagle-3 | 2152194 | 19 | 20 | 4096 | 338.2s | 1.26x |  |  | 1227.0 | 1.25x | 2510.4 | 1.26x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-30B-A3B | PARD K=5 | 2152139 | 19 | 20 | 4096 | 450.6s | 0.94x |  |  | 933.6 | 0.95x | 1894.6 | 0.95x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-30B-A3B | Suffix | 2152138 | 19 | 20 | 4096 | 477.2s | 0.89x |  |  | 882.0 | 0.90x | 1794.7 | 0.90x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-30B-A3B | baseline | 2152136 | 19 | 20 | 4096 | 425.6s | 1.00x |  |  | 983.3 | 1.00x | 1998.4 | 1.00x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-30B-A3B | baseline fuse_loss=false | 2152321 | 19 | 20 | 4096 | 416.6s | 1.02x |  |  | 1009.9 | 1.03x | 2052.3 | 1.03x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-32B | Eagle-3 | 2152196 | 19 | 20 | 4096 | 189.0s | 1.20x |  |  | 1114.6 | 1.19x | 2281.9 | 1.19x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-32B | PARD K=5 | 2152147 | 19 | 20 | 4096 | 244.6s | 0.93x |  |  | 881.4 | 0.94x | 1794.2 | 0.94x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-32B | PARD-2 | 2152218 | 19 | 20 | 4096 | 339.3s | 0.67x |  |  | 660.0 | 0.70x | 1344.2 | 0.70x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-32B | Suffix | 2152146 | 19 | 20 | 4096 | 494.9s | 0.46x |  |  | 431.7 | 0.46x | 873.3 | 0.46x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-32B | baseline | 2152144 | 19 | 20 | 4096 | 227.5s | 1.00x |  |  | 938.2 | 1.00x | 1910.5 | 1.00x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg async-1off OSL4096 | Qwen3-32B | baseline fuse_loss=false | 2152344 | 19 | 20 | 4096 | 238.6s | 0.95x |  |  | 912.6 | 0.97x | 1864.5 | 0.98x |  |  | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-30B-A3B | Eagle-3 | 2152193 | 19 | 20 | 4096 | 299.4s | 1.41x | 127.8s | 2.18x | 1371.6 | 1.39x | 3226.3 | 2.15x | 63.9% | 2.87 | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-30B-A3B | PARD K=5 | 2152135 | 19 | 20 | 4096 | 427.5s | 0.99x | 280.9s | 0.99x | 970.5 | 0.98x | 1478.0 | 0.99x | 36.6% | 2.29 | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-30B-A3B | PARD-2 | 2152222 | 0 | 20 | 4096 |  |  |  |  |  |  |  |  |  |  | FAILED; waiting metrics | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-30B-A3B | Suffix | 2152134 | 19 | 20 | 4096 | 415.2s | 1.02x | 240.0s | 1.16x | 999.7 | 1.01x | 1731.1 | 1.15x | 11.3% | 1.84 | COMPLETED; partial_usable_with_errors | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-30B-A3B | baseline | 2152132 | 19 | 20 | 4096 | 421.7s | 1.00x | 278.5s | 1.00x | 985.9 | 1.00x | 1500.1 | 1.00x |  |  | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-30B-A3B | baseline fuse_loss=false | 2152320 | 19 | 20 | 4096 | 449.7s | 0.94x | 299.3s | 0.93x | 924.5 | 0.94x | 1395.1 | 0.93x |  |  | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | Eagle-3 | 2152195 | 19 | 20 | 4096 | 395.2s | 1.36x | 192.6s | 1.72x | 1045.7 | 1.33x | 2150.3 | 1.69x | 45.2% | 2.33 | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | PARD K=5 | 2152143 | 19 | 20 | 4096 | 562.2s | 0.95x | 370.6s | 0.90x | 749.0 | 0.95x | 1138.2 | 0.90x | 31.8% | 2.00 | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | PARD-2 | 2152224 | 16 | 20 | 4096 | 802.7s | 0.67x | 601.7s | 0.55x | 515.6 | 0.66x | 688.1 | 0.54x | 1.0% | 1.05 | TIMEOUT; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | PARD-2 | 2152532 | 19 | 20 | 4096 | 873.4s | 0.61x | 677.2s | 0.49x | 479.6 | 0.61x | 623.9 | 0.49x | 1.0% | 1.05 | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | Suffix | 2152142 | 16 | 20 | 4096 | 788.6s | 0.68x | 580.0s | 0.57x | 537.5 | 0.68x | 733.0 | 0.58x | 7.9% | 1.56 | TIMEOUT; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | Suffix | 2152499 | 19 | 20 | 4096 | 592.5s | 0.90x | 386.1s | 0.86x | 695.4 | 0.89x | 1070.8 | 0.84x | 16.9% | 1.94 | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | baseline | 2152140 | 19 | 20 | 4096 | 536.1s | 1.00x | 331.8s | 1.00x | 784.9 | 1.00x | 1269.6 | 1.00x |  |  | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL4096 | Qwen3-32B | baseline fuse_loss=false | 2152343 | 19 | 20 | 4096 | 544.1s | 0.99x | 331.2s | 1.00x | 774.1 | 0.99x | 1272.9 | 1.00x |  |  | COMPLETED; partial_usable | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL8192 | Qwen3-235B-A22B | baseline | 2152151 | 0 | 20 | 8192 |  |  |  |  |  |  |  |  |  |  | FAILED; waiting metrics | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL8192 | Qwen3-235B-A22B | baseline | 2152615 | 0 | 3 | 8192 |  |  |  |  |  |  |  |  |  |  | FAILED; waiting metrics | lyris_perfcfg_20260618 |
| NeMo-RL PerfCfg | Lyris perfcfg sync OSL8192 | Qwen3-235B-A22B | baseline | 2152682 | 0 | 3 | 8192 |  |  |  |  |  |  |  |  |  |  | CANCELLED by 2001147693; waiting metrics | lyris_perfcfg_20260618 |
| SWE-RL | qwen235B step>=2 | Qwen3-235B-A22B | baseline | 3299487 | 8 | 8 |  | 1243.3s | 1.00x |  |  | 103.8 | 1.00x | 213.1 | 1.00x |  |  | complete | swerl_qwen235b_20260615 |
| SWE-RL | qwen235B step>=2 | Qwen3-235B-A22B | eagle3_k3 | 3299491 | 7 | 7 |  | 1389.8s | 0.89x |  |  | 50.1 | 0.48x | 103.5 | 0.49x |  |  | complete | swerl_qwen235b_20260615 |
| SWE-RL | qwen235B step>=2 | Qwen3-235B-A22B | pard_k5 | 3299489 | 5 | 5 |  | 1952.4s | 0.64x |  |  | 27.4 | 0.26x | 57.1 | 0.27x |  |  | complete | swerl_qwen235b_20260615 |
| SWE-RL | qwen30 ctx40k step1 | Qwen3-30B-A3B | baseline | 3344823 | 1 | 1 |  | 141.0s | 1.00x |  |  | 190.8 | 1.00x | 559.5 | 1.00x |  |  | complete | swerl_qwen30_ctx40k_20260616 |

## Supplemental / Archived Rows

These are valid parsed rows, but not mixed into the primary table because they use a different cluster/backend or older broader sweep settings.

| Domain | Temp | Model | Method | Rows | Batches | ISL | OSL | Baseline tok/s/GPU | tok/s/GPU | Speedup | Acceptance | Mean accept len | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math | 0.0 | Qwen3-8B | pard_k3 | 2 | 1/2 | 4096 | 32768 | 62.8 | 63.0 | 1.04x | 64.5% | 2.94 | oci_qwen8_pard1_20260616 |
| Math | 0.0 | Qwen3-8B | pard_k5 | 2 | 1/2 | 4096 | 32768 | 62.8 | 70.5 | 1.15x | 46.2% | 3.31 | oci_qwen8_pard1_20260616 |
| Math | 1.0 | Qwen3-8B | pard_k3 | 2 | 1/2 | 4096 | 32768 | 62.8 | 55.2 | 0.89x | 47.0% | 2.41 | oci_qwen8_pard1_20260616 |
| Math | 1.0 | Qwen3-8B | pard_k5 | 2 | 1/2 | 4096 | 32768 | 62.8 | 66.4 | 1.10x | 42.9% | 3.14 | oci_qwen8_pard1_20260616 |
| SWE | 0.0 | Qwen3-8B | pard_k3 | 2 | 1/2 | 4096 | 32768 | 58.2 | 68.1 | 1.19x | 72.6% | 3.18 | oci_qwen8_pard1_20260616 |
| SWE | 0.0 | Qwen3-8B | pard_k5 | 2 | 1/2 | 4096 | 32768 | 58.2 | 66.6 | 1.22x | 50.5% | 3.53 | oci_qwen8_pard1_20260616 |
| SWE | 1.0 | Qwen3-8B | pard_k3 | 2 | 1/2 | 4096 | 32768 | 60.4 | 56.6 | 0.97x | 57.4% | 2.72 | oci_qwen8_pard1_20260616 |
| SWE | 1.0 | Qwen3-8B | pard_k5 | 2 | 1/2 | 4096 | 32768 | 60.4 | 67.8 | 1.18x | 50.2% | 3.51 | oci_qwen8_pard1_20260616 |
| Math | 1.0 | Qwen3-235B-A22B | eagle3_k3 | 2 | mixed | 4096 | 32768 |  | 6.2 |  | 42.6% | 2.28 | archived_qwen235b_temp1_sweep |
| Math | 1.0 | Qwen3-235B-A22B | pard_k5 | 2 | mixed | 4096 | 32768 |  | 5.9 |  | 31.8% | 2.59 | archived_qwen235b_temp1_sweep |
| Math | 1.0 | Qwen3-235B-A22B | suffix_k32 | 2 | mixed | 4096 | 32768 |  | 15.4 |  | 66.3% | 5.12 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | eagle3_k11 | 10 | mixed | 4096 | 32768 |  | 28.7 | 1.23x | 7.5% | 1.83 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | eagle3_k9 | 10 | mixed | 4096 | 32768 |  | 28.7 | 1.21x | 9.1% | 1.82 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | pard2_k11 | 10 | mixed | 4096 | 32768 |  | 19.7 | 0.81x | 1.3% | 1.14 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | pard2_k9 | 10 | mixed | 4096 | 32768 |  | 19.7 | 0.83x | 1.8% | 1.16 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | pard_k11 | 10 | mixed | 4096 | 32768 |  | 22.1 | 1.00x | 6.4% | 1.71 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | pard_k9 | 10 | mixed | 4096 | 32768 |  | 21.8 | 0.98x | 7.2% | 1.64 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | suffix_k16 | 10 | mixed | 4096 | 32768 |  | 36.7 | 1.74x | 50.9% | 2.79 | archived_qwen235b_temp1_sweep |
| SWE | 1.0 | Qwen3-235B-A22B | suffix_k8 | 10 | mixed | 4096 | 32768 |  | 36.6 | 1.78x | 51.7% | 2.69 | archived_qwen235b_temp1_sweep |

## Sources

- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics_live.csv`
- `docs/lyris_math500_osl32k_temp01_home_retry_metrics_live_20260616.csv`
- `docs/qmath_vllm_standalone_batch_sweep_status_20260617.csv`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv`
- `docs/lyris_swebench_longosl_metrics_20260612.csv`
- `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_r4_noprof_metrics.csv`
- `docs/vllm_standalone_temp0_temp1_trends_20260616.csv`
- `docs/oci_hsg_mathrl_multimodel_specdec_step20_live_summary_20260616.csv`
- `docs/oci_hsg_mathrl_qwen32_online_pard2_hardce_r11_summary_20260616.csv`
- `docs/oci_hsg_mathrl_qwen235b_reduced64_temp1_pard_k3k5_summary_step2_10_20260615.csv`
- `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_completed_summary_stepge2_20260615.csv`
- `docs/oci_hsg_swerl_qwen30ba3b_baseline_ctx40k_3344823_summary_20260616.csv`
- `docs/qwen8_pard2_official_comparison_metrics_20260613.csv`
- `docs/oci_hsg_qwen17b_out16k_nemorl_summary_20260616.csv`
- `docs/lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv`
