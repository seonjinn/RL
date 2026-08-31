# Qwen3-30B-A3B PTV2 frozen-first gate

This campaign compares the new PTV2 English DFlash, DSpark, and DFlash2
drafters against no-SpecDec baselines before any online drafter training is
enabled.

- Math: Qwen3-30B-A3B Base, official 4-node × 4-GPU GRPO performance recipe,
  20 optimizer steps, sequence packing enabled.
- SWE: Qwen3-30B-A3B Thinking-2507, rollout-only, the same 20 validation
  instances, seeds, sampling settings, and topology for every arm.
- Stable first gate: baseline, DFlash K7, DSpark K5.
- DFlash2: K7, isolated runtime cohort based on vLLM PR #52816. Its baseline
  must be rerun in that cohort and is never compared directly with the stable
  vLLM 0.25.1 baseline.

All rows are frozen: `policy.draft.enabled=false`; only the generation worker
loads the drafter. There is no drafter optimizer, online update, or draft refit.
