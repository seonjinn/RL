# Qwen3-8B NeMo-RL SpecDec Completion Audit

Updated: 2026-06-04 07:15 PDT

## Bottom Line

The Qwen3-8B online drafter path is now working in NeMo-RL. Always-on K=1
recovers a real generate-path speedup, but full GRPO E2E speedup is small because
the measured generation slice is only about 4.37% of the steady-state step.

The result should be reported as:

| Metric | Result |
|---|---:|
| Best isolated Step-5 direct policy-generate K=1 speedup | 1.427x |
| Full 20-step K=1 generation-worker TPS speedup | 1.204x |
| Full 20-step K=1 direct policy-generate TPS speedup | 1.224x |
| Steady Step 5-20 K=1 direct policy-generate TPS speedup | 1.269x |
| Full 20-step K=1 E2E throughput speedup | 1.004x |
| Steady Step 5-20 K=1 E2E throughput speedup | 1.009x |
| Steady Step 5-20 baseline generation share | 4.37% |
| Steady Step 5-20 Amdahl E2E projection | 1.009x |
| K=1 acceptance, Step-5 isolation | 71.27% |
| K=1 acceptance, 20-step log-derived mean | about 61.2% |

## Objective Audit

| Requirement | Evidence | Status |
|---|---|---|
| Summarize Qwen3-8B NeMo-RL SpecDec result | `docs/qwen3_8b_r108_r109_timing_split_20step_window_summary.csv`, `docs/qwen3_8b_r108_r109_policy_generate_window_summary.csv` | Done |
| Compare against vLLM standalone speedup | `docs/eagle3_focus_vllm_standalone_metrics.csv`, `docs/qwen3_8b_nemorl_vs_vllm_root_cause_matrix.md` | Done |
| Explain why NeMo-RL E2E does not show standalone-style speedup | Timing split shows generation share 4.37%; measured E2E matches Amdahl projection | Done |
| Inspect and patch code-level timing issue | `nemo_rl/algorithms/grpo.py` moved `finish_generation()` out of generation timer; `nemo_rl/experience/rollouts.py` added direct `policy_generate_time_s` | Done |
| Verify online drafter weights reach vLLM drafter | r107 logs show `[draft] Loading 14 trainer-owned draft weights into vLLM drafter.` | Done |
| Bring NeMo-RL close to standalone performance where comparable | Direct generate path reaches 1.269x steady and 1.427x isolated; comparable K=1 standalone references are 1.299x exact-engine and 1.331x synthetic bs32 | Done with scope caveat |

## Comparison Table

| Workload | Config | Acceptance | Throughput speedup | Interpretation |
|---|---|---:|---:|---|
| vLLM standalone synthetic static, ISL=1000 OSL=512 bs32 | Qwen3-8B K=1 | N/A in original sweep | 1.331x | Optimistic synthetic fixed-token ceiling |
| vLLM standalone synthetic static, ISL=1000 OSL=512 bs32 | Qwen3-8B K=2 | 96.29% | 2.346x | High-acceptance synthetic ceiling, not representative of DAPOMath GRPO |
| vLLM standalone synthetic static, ISL=1000 OSL=512 bs32 | Qwen3-8B K=3 | N/A in original sweep | 2.145x | High synthetic speedup, but not matched by real prompts |
| vLLM standalone DAPOMath real prompt bs32 | Qwen3-8B K=1 | 61.11% | 0.665x | Real-prompt standalone regresses despite similar acceptance |
| vLLM standalone DAPOMath real prompt bs32 | Qwen3-8B K=3 | 36.62% | 0.853x | K=3 weak on real DAPOMath prompts |
| NeMo exact-engine/no-gate diagnostic r22 | Qwen3-8B K=1 | 64.61% | 1.299x | Useful native K=1 reference |
| NeMo online timing split r106/r107 Step 5 | Qwen3-8B K=1 | 71.27% | 1.427x direct generate | Confirms online drafter + refit works |
| NeMo online timing split r108/r109 Step 5-20 | Qwen3-8B K=1 | about 61.2% | 1.269x direct generate | Closest steady NeMo number to standalone output throughput |
| NeMo online timing split r108/r109 Step 5-20 | Qwen3-8B K=1 | about 61.2% | 1.009x E2E | Expected from 4.37% generation share |

## Root Cause

The low E2E speedup is not caused by SpecDec being disabled, online drafter
weights failing to load, or KV-cache quantization. The current dominant cause is
full-loop composition:

| Component | Finding |
|---|---|
| Direct generation | K=1 improves direct generate throughput by 1.269x steady-state and 1.427x in Step-5 isolation |
| Generation finish/sleep | Not accelerated by SpecDec; K=1 finish time was slower in the isolation run |
| GRPO non-generation work | Policy/reference logprobs and policy training dominate the step |
| Generation share | Only 4.37% in the Step 5-20 baseline window |
| E2E outcome | Measured 1.009x matches Amdahl projection, so large E2E gains are not expected without making the workload more decode-bound or overlapping/removing non-generation work |

## Recommendation

Use Qwen3-8B always-on K=1 as the only positive online SpecDec path. Do not use
K=3 for Qwen3-8B throughput reporting on this DAPOMath-style workload unless the
drafter or prompt/domain is changed. Report direct policy-generate speedup and
E2E speedup separately, with `generation_finish` split out.

## Artifacts

| Artifact | Purpose |
|---|---|
| `docs/qwen3_8b_r108_r109_policy_generate_window_speedups.png` | Steady direct policy-generate speedup chart |
| `docs/qwen3_8b_r108_r109_timing_split_20step_window_speedups.png` | Generation/E2E timing split speedup chart |
| `docs/qwen3_8b_r108_r109_timing_split_20step_window_summary.csv` | Windowed generation/E2E timing summary |
| `docs/qwen3_8b_r108_r109_policy_generate_window_summary.csv` | Windowed direct policy-generate summary |
| `docs/qwen3_8b_nemorl_vs_vllm_root_cause_matrix.md` | Detailed root-cause matrix |
| `experiments/eagle3_online/remote_patches_qwen8_timing_split_r106_r109.patch` | Remote code patch artifact |
