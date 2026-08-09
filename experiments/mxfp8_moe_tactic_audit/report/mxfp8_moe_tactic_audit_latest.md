# MXFP8 MoE Tactic Audit

## MICRO AUDIT COMPLETE: NO PROMOTION

The production-prepacked Qwen3-30B-A3B MXFP8 expert FC1/FC2 path was replayed with CUDA Graphs enabled. Eight representative routing profiles cover 95.31% of sampled MoE GPU time. The audit measured all 920 legal FC1/FC2 tactic pairs for every selected profile, producing 7,360 successful rows.

All rows were finite and deterministic, with `max_abs_error=0` and cosine similarity between 0.99999988 and 1.00000012. These checks establish micro-kernel numerical agreement for the replayed tensors; they are not a GSM8K or end-to-end accuracy claim.

The final same-job comparison repeated stock `(16,530)` and the five strongest common candidates 50 times per profile. Candidate `(32,574)` was the best stable direction: aggregate trace-weighted FC1+FC2 time improved by 1.52%, the binding lower weighted-median gain was 1.22%, and no profile regressed. It did not pass promotion because the required weighted-median gain is 2% and its maximum CV was 3.74%, above the 3% limit. No candidate cache was published, so runtime behavior remains unchanged.

![Stock-relative FC1+FC2 kernel throughput](mxfp8_moe_tactic_audit_actual_pair_speedup.png)

## Same-Run Candidate Comparison

| Tactic pair | Aggregate kernel throughput | Weighted-median gain | Maximum CV | Maximum profile regression | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `(16,530)` stock | 1.0000x | 0.00% | 4.70% | 0.00% | Baseline |
| `(32,232)` | 1.0045x | 0.02% | 3.22% | 1.77% | Reject |
| `(32,280)` | 1.0138x | 1.25% | 28.34% | 0.88% | Reject |
| `(32,550)` | 1.0136x | 1.25% | 40.86% | 0.88% | Reject |
| `(32,568)` | 1.0070x | 0.00% | 3.75% | 0.46% | Reject |
| `(32,574)` | 1.0152x | 1.22% | 3.74% | 0.00% | Reject |

The metric above is `weighted stock FC1+FC2 time / weighted candidate FC1+FC2 time`; it is a kernel-level ratio, not NeMo-RL generation throughput.

## Execution Record

| Stage | Job | Result |
| --- | ---: | --- |
| Routing trace and prepacked capture | `2549878` | Artifacts captured; job later failed during teardown because the driver environment lacked `megatron` |
| Full 920-pair audit | `2550024` | All 7,360 JSONL rows completed; job canceled only after raw data completion while NSys finalization remained active |
| Top-five repeat run | `2550079` | All 40 JSONL rows completed; superseded by the stock-inclusive run |
| Stock plus top-five, 50 repeats | `2550086` | Completed successfully, 48/48 rows |

## Qualification Status

- Weighted-gain gate: at least 2%.
- Stability gate: maximum CV at most 3%.
- Regression gate: no high-weight profile may regress by more than 1%.
- Result: no candidate passed every gate; do not enable an FC1/FC2 tactic override.
- End-to-end NeMo-RL and matched GSM8K candidate runs were intentionally not started because no candidate qualified for promotion.
- Final production provenance still requires rerunning the trace stage without the teardown failure.

The machine-readable summary is [mxfp8_moe_tactic_micro_audit_20260808.json](mxfp8_moe_tactic_micro_audit_20260808.json).
