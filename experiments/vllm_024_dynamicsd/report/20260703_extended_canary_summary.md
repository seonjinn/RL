# Qwen3-8B Extended SpecDec Canary

## vLLM 0.24

Configuration: Math, temperature 0, top-p 1, ISL 4096, OSL 256, BS4,
target TP1, four allocated GB200 GPUs, eager mode, Triton target attention.
Speedup is relative to the matched target-only baseline.

| Method | K | tok/s/GPU | Speedup | Acceptance | Mean accept length |
|---|---:|---:|---:|---:|---:|
| Baseline | - | 37.15 | 1.00x | - | - |
| Suffix | 32 | 438.59 | 11.81x | 95.70% | 15.95 |
| PARD | 12 | 140.09 | 3.77x | 71.14% | 9.54 |
| PARD-2 | 15 | 22.15 | 0.60x | 0.11% | 1.02 |
| DFlash | 15 | 179.27 | 4.83x | 73.95% | 12.09 |

PARD-2 loaded the official target projection, selected target layers
`(36, 29, 21, 13)`, and completed generation. Its very low acceptance is a
model-path correctness or feature-alignment issue, not a startup failure.

## AngelSlim Native Runtime Check

These jobs used the pinned AngelSlim Transformers runner with SDPA because
Flash Attention was unavailable in the image. They used natural prompt length
and a 128-token output cap, so they validate the code path but are not directly
comparable with the vLLM table. A separate fixed ISL4096/OSL32768 run follows.

| Method | Decode speedup | Baseline tok/s | SpecDec tok/s | Acceptance | Mean accept length |
|---|---:|---:|---:|---:|---:|
| DFlash | 3.89x | 3.27 | 12.70 | 57.40% | 9.61 |
| DFlare | 5.32x | 3.32 | 17.65 | 72.45% | 11.87 |

Raw results are stored in `20260703_q8_extended_canary/` and
`20260703_angelslim_canary/`.
