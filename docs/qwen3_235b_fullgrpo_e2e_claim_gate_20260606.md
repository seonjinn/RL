# Qwen3-235B Full-GRPO E2E Claim Gate

Date: 2026-06-06 PDT, latest update 2026-06-07 PDT

Verdict: **positive_e2e_claim_allowed**.

A Qwen3-235B no-stop Full-GRPO E2E claim is allowed only when a matched
baseline/specdec pair has parsed logs and numeric E2E throughput or E2E
step-time speedup. Generation-only and stop-after-generation rows do not
satisfy this gate.

| Group | Candidate | Job | Baseline status | Candidate status | Verdict | Gen TPS speedup | E2E TPS speedup | E2E step speedup | Acceptance | Reason |
|---|---|---:|---|---|---|---:|---:|---:|---:|---|
| noncolocated_tp4_fixed256 | public_pard_k3 | 3209048 | parsed | parsed | claim_allowed_positive_e2e | 1.888000 | 1.420000 | 1.421000 | 57.5800 | matched no-stop Full-GRPO E2E speedup is positive on step2-5 against baseline 3209047 |
| sampling_step4 | local_pard2_cat_tpp_k5 | 3186511 | missing_log | missing_log | pending_metrics |  |  |  |  | baseline/specdec ray-driver metrics are not both parsed |
| sampling_step4 | dynamic_dpace_k5 | 3192180 | missing_log | missing_log | pending_metrics |  |  |  |  | baseline/specdec ray-driver metrics are not both parsed |
| sampling_step4 | dynamic_dpace_k3 | 3192438 | missing_log | missing_log | pending_metrics |  |  |  |  | baseline/specdec ray-driver metrics are not both parsed |
| fixed256_step20 | local_pard2_cat_tpp_k5 | 3186343 | missing_log | missing_log | pending_metrics |  |  |  |  | baseline/specdec ray-driver metrics are not both parsed |
| fixed256_step20 | public_pard_k5 | 3186344 | missing_log | missing_log | pending_metrics |  |  |  |  | baseline/specdec ray-driver metrics are not both parsed |

## Current Reading

- Pending rows: `5`.
- Positive E2E rows: `1`.
- Current safe claim: Qwen3-235B public PARD K3 no-stop Full-GRPO E2E benefit is proven for the fixed-256 non-colocated TP4 branch.
