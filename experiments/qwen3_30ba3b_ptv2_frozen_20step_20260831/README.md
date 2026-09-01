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

## Matched drafter-generation K sweep

The follow-up Math screen compares two weight generations at identical runtime
settings and `K` values:

- PTV2 step 25391: `ptv2_final/sd2en-q30-base-ptv2en-{dflash,dspark}-b8-16n`.
- Previous canonical cohort: the migrated Lyris DFlash `s4400` and DSpark
  `s5700` exports at checkpoint 14500 that produced the earlier canonical
  20-step `lyris14500` W&B runs.

Each cohort runs DFlash and DSpark with `K=1,2,3,5,7`. A matched no-SpecDec
baseline runs first; all 20 SpecDec jobs depend on its success. Every arm uses
20 GRPO steps. Every arm keeps the official Qwen3-30B-A3B 4n4g
performance recipe, frozen drafter mode, `flashinfer_trtllm`,
`FULL_AND_PIECEWISE`, target TP1, draft TP1, and `max_num_seqs=128`. CUDA Graph
capture sizes are selected per K to cover both the K-wide DSpark draft family
and the K+1-wide verifier/DFlash family without making low-K jobs capture the
largest K7 PIECEWISE graphs. `policy.offload_optimizer_for_refit=false` keeps
the optimizer on GPU during target refit so its CPU copy does not overlap the
vLLM sleep-weight backup on GB200 hosts.

The five-step sweep was submitted on 2026-08-31 under `nemotron_n3_post`.
Baseline job `6751603` is the gate; jobs `6751606` through `6751650` contain
the 20 SpecDec arms and carry `afterok:6751603`. The exact arm-to-job mapping
is recorded in `submissions/math_k_sweep_5step_20260901.tsv`.

That five-step baseline completed three steps and entered step 4 before all
four nodes hit host-memory OOM. The launcher had inherited
`policy.offload_optimizer_for_refit=true`; its optimizer CPU copy overlapped
vLLM's sleep-weight backup. Commit `dbb7a0bd7` pins the validated GB200 setting
to `false`.

The repaired 20-step matrix was submitted on 2026-08-31 under
`nemotron_n3_post`. Baseline job `6757938` is the gate; the 20 SpecDec jobs
`6757940` through `6757986` carry `afterok:6757938`. The exact mapping is in
`submissions/math_k_sweep_20step_20260901.tsv`.

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
