# Qwen3-235B Math vLLM Standalone Results
ISL=4096, OSL=32768. B4/B8/B16/B32 rows are matched 6/17 qmath sweep rows. B1/B2 SpecDec rows are previous completed/reference rows against the 6/19 baseline unless otherwise noted.
|Temp|Batch|Method|tok/s/GPU|Baseline tok/s/GPU|Speedup|Acceptance|Mean accept len|Basis|
|---|---:|---|---:|---:|---:|---:|---:|---|
|0.0|1|baseline|1.91|1.91|1.00x|n/a|n/a|matched 6/19 aggregation|
|0.0|1|suffix_k32|15.31|1.91|8.00x*|83.8%|7.82|reference vs 6/19 baseline; cross-run|
|0.0|1|eagle3_k3|5.67|1.91|2.97x*|69.4%|3.08|reference vs 6/19 baseline; cross-run|
|0.0|1|pard_k5|4.78|1.91|2.50x*|56.1%|3.80|reference vs 6/19 baseline; cross-run|
|0.0|1|pard2_k1|1.60|1.91|0.84x*|3.1%|1.03|reference vs 6/19 baseline; cross-run|
|0.0|2|baseline|3.88|3.88|1.00x|n/a|n/a|matched 6/19 aggregation|
|0.0|2|suffix_k32|28.72|3.88|7.41x*|86.0%|8.31|reference vs 6/19 baseline; cross-run|
|0.0|2|eagle3_k3|10.94|3.88|2.82x*|69.7%|3.09|reference vs 6/19 baseline; cross-run|
|0.0|2|pard_k5|10.40|3.88|2.68x*|50.1%|3.50|reference vs 6/19 baseline; cross-run|
|0.0|4|baseline|7.86|7.86|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|0.0|4|suffix_k32|51.63|7.86|6.57x|89.8%|8.80|matched 6/17 qmath sweep|
|0.0|4|eagle3_k3|15.91|7.86|2.03x|57.0%|2.71|matched 6/17 qmath sweep|
|0.0|4|pard_k5|17.94|7.86|2.28x|55.7%|3.79|matched 6/17 qmath sweep|
|0.0|4|pard2_k1|4.98|7.86|0.63x|3.2%|1.03|matched 6/17 qmath sweep|
|0.0|8|baseline|15.52|15.52|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|0.0|8|suffix_k32|61.08|15.52|3.94x|86.7%|7.79|matched 6/17 qmath sweep|
|0.0|8|eagle3_k3|34.35|15.52|2.21x|60.9%|2.83|matched 6/17 qmath sweep|
|0.0|8|pard_k5|36.46|15.52|2.35x|53.9%|3.69|matched 6/17 qmath sweep|
|0.0|8|pard2_k1|12.80|15.52|0.83x|4.3%|1.04|matched 6/17 qmath sweep|
|0.0|16|baseline|23.52|23.52|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|0.0|16|suffix_k32|121.46|23.52|5.16x|80.4%|6.78|matched 6/17 qmath sweep|
|0.0|16|eagle3_k3|65.44|23.52|2.78x|65.3%|2.96|matched 6/17 qmath sweep|
|0.0|16|pard_k5|59.30|23.52|2.52x|50.8%|3.54|matched 6/17 qmath sweep|
|0.0|16|pard2_k1|25.08|23.52|1.07x|5.1%|1.05|matched 6/17 qmath sweep|
|0.0|32|baseline|60.84|60.84|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|0.0|32|suffix_k32|139.09|60.84|2.29x|77.8%|6.40|matched 6/17 qmath sweep|
|0.0|32|eagle3_k3|109.56|60.84|1.80x|59.1%|2.77|matched 6/17 qmath sweep|
|0.0|32|pard_k5|105.64|60.84|1.74x|51.4%|3.57|matched 6/17 qmath sweep|
|0.0|32|pard2_k1|49.01|60.84|0.81x|5.6%|1.06|matched 6/17 qmath sweep|
|1.0|1|baseline|1.93|1.93|1.00x|n/a|n/a|matched 6/19 aggregation|
|1.0|1|suffix_k32|7.55|1.93|3.91x*|59.2%|3.73|reference vs 6/19 baseline; cross-run|
|1.0|1|eagle3_k3|4.48|1.93|2.32x*|43.1%|2.29|reference vs 6/19 baseline; cross-run|
|1.0|1|pard_k5|4.54|1.93|2.35x*|34.7%|2.74|reference vs 6/19 baseline; cross-run|
|1.0|2|baseline|3.93|3.93|1.00x|n/a|n/a|matched 6/19 aggregation|
|1.0|2|suffix_k32|23.18|3.93|5.90x*|73.4%|6.52|reference vs 6/19 baseline; cross-run|
|1.0|2|eagle3_k3|7.99|3.93|2.03x*|42.0%|2.26|reference vs 6/19 baseline; cross-run|
|1.0|2|pard_k5|7.20|3.93|1.83x*|28.8%|2.44|reference vs 6/19 baseline; cross-run|
|1.0|4|baseline|7.73|7.73|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|1.0|4|suffix_k32|35.39|7.73|4.58x|74.3%|6.16|matched 6/17 qmath sweep|
|1.0|4|eagle3_k3|13.74|7.73|1.78x|46.4%|2.39|matched 6/17 qmath sweep|
|1.0|4|pard_k5|7.26|7.73|0.94x|17.9%|1.89|matched 6/17 qmath sweep|
|1.0|4|pard2_k1|6.39|7.73|0.83x|1.2%|1.01|matched 6/17 qmath sweep|
|1.0|8|baseline|14.72|14.72|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|1.0|8|suffix_k32|54.63|14.72|3.71x|70.0%|5.52|matched 6/17 qmath sweep|
|1.0|8|eagle3_k3|25.20|14.72|1.71x|50.2%|2.51|matched 6/17 qmath sweep|
|1.0|8|pard_k5|19.87|14.72|1.35x|30.7%|2.54|matched 6/17 qmath sweep|
|1.0|8|pard2_k1|12.44|14.72|0.85x|1.5%|1.02|matched 6/17 qmath sweep|
|1.0|16|baseline|29.69|29.69|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|1.0|16|suffix_k32|76.25|29.69|2.57x|63.2%|4.45|matched 6/17 qmath sweep|
|1.0|16|eagle3_k3|48.45|29.69|1.63x|46.5%|2.40|matched 6/17 qmath sweep|
|1.0|16|pard_k5|35.91|29.69|1.21x|29.8%|2.49|matched 6/17 qmath sweep|
|1.0|16|pard2_k1|24.93|29.69|0.84x|2.7%|1.03|matched 6/17 qmath sweep|
|1.0|32|baseline|59.83|59.83|1.00x|n/a|n/a|matched 6/17 qmath sweep|
|1.0|32|suffix_k32|100.75|59.83|1.68x|66.1%|4.71|matched 6/17 qmath sweep|
|1.0|32|eagle3_k3|84.56|59.83|1.41x|46.5%|2.40|matched 6/17 qmath sweep|
|1.0|32|pard_k5|55.67|59.83|0.93x|29.0%|2.45|matched 6/17 qmath sweep|
|1.0|32|pard2_k1|50.25|59.83|0.84x|2.3%|1.02|matched 6/17 qmath sweep|

Additional high-K B1/B2 rows are preserved in `docs/vllm_standalone_qwen235b_math_all_batches_20260619.csv`.
