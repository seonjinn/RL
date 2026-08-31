# Qwen3-30B-A3B PTV2 frozen-first gate

This campaign compares the new PTV2 English DFlash, DSpark, and DFlash2
drafters against no-SpecDec baselines before any online drafter training is
enabled.

- Math: Qwen3-30B-A3B Base, official 4-node × 4-GPU GRPO performance recipe,
  20 optimizer steps, sequence packing enabled.
- SWE: Qwen3-30B-A3B Thinking-2507, rollout-only, the same 20 validation
  instances, seeds, sampling settings, and topology for every arm.
- Stable Math gate: baseline, DFlash K7, DSpark K5.
- Stable SWE gate: baseline, DFlash K3/K5/K7, DSpark K3/K5.
- DFlash2: K7, isolated runtime cohort based on vLLM PR #52816. Its baseline
  must be rerun in that cohort and is never compared directly with the stable
  vLLM 0.25.1 baseline.

All rows are frozen: `policy.draft.enabled=false`; only the generation worker
loads the drafter. There is no drafter optimizer, online update, or draft refit.

## 2026-08-31 execution status

- Completed SWE baseline: job `6731577`, W&B `y5dxoeps`.
- Completed SWE DFlash K7: job `6731579`, W&B `r180k01d`.
- Pending SWE K sweep: DFlash K3 `6739935`, DFlash K5 `6739937`,
  DSpark K3 `6739939`, and DSpark K5 `6739941`.
- Pending repaired Math gate: baseline `6739944`, DFlash K7 `6739946`, and
  DSpark K5 `6739948`.

The original Math jobs `6731294`, `6731296`, and `6731298` failed before model
startup because the generated batch script sourced a nonexistent
`/home/sna/script/export_env_vars.sh`. The launcher now resolves W&B credentials
before submission and exports them through Slurm. Original SWE DSpark job
`6731581` failed while rebuilding the editable NeMo Gym package from a shared
mutable checkout. SWE jobs now reuse the checksum-verified staged Gym venv with
`NRL_FORCE_REBUILD_VENVS=false`.
