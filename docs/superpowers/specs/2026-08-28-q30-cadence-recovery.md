# Qwen3-30B-A3B Cadence Recovery Design

## Goal

Complete the 200-step DFlash and DSpark fixed-interval study without false
harness failures and without changing the official NeMo-RL performance recipe.

## Evidence

- DFlash fixed-20 reached step 14 and was terminated by the harness's fixed
  45-minute first-refit gate. It had not reached the first scheduled refit at
  step 20, so this was not a refit hang.
- The inherited performance recipe disables checkpointing. Existing DFlash
  jobs therefore cannot resume from steps 14, 65, or 66.
- OCI-HSG `batch` is limited to four hours, while `batch_long` permits seven
  days. The measured cadence requires roughly 12--14 hours for 200 steps.
- All three DSpark arms fail during first-step FAP generation with a CUDA
  illegal-memory access on GB200, before online drafter training or refit.
- The runtime is vLLM 0.25.1 with Model Runner V2, FlashInfer draft attention,
  and `FULL_AND_PIECEWISE` CUDA Graphs. Upstream vLLM change #48167 corrects
  Blackwell CUDA Graph support classification for non-causal draft attention;
  #48261 separately refreshes MRV2 speculator-prefill attention metadata.

## Decisions

1. Use `batch_long` with an 18-hour wall limit for the measurement jobs. Keep
   checkpointing disabled so checkpoint writes do not contaminate step timing.
2. Replace the first-refit wall-clock deadline with a liveness gate: wait until
   the requested refit marker, training-process exit, or SLURM wall limit.
3. Preserve stock vLLM 0.25.1 as the control. For DSpark only, create a
   node-local Python-package overlay and apply exactly #48167's non-causal
   attention guard. Fail closed if the installed source does not match.
4. Run a short corrected DSpark FAP canary. If #48167 alone still reproduces
   the illegal access, add #48261 in a second, separately identified A/B arm.
5. Do not claim legacy fixed-cadence resume safety until the applied serving
   drafter snapshot is durably restored. The long-partition experiment does
   not depend on that unfinished contract.

## Acceptance Criteria

- Rendered jobs request `batch_long` and 18 hours.
- Startup/CUDA Graph/step gates retain bounded diagnostics, while the first
  refit gate cannot kill a live training process solely because the interval is
  20.
- The DSpark overlay patch is exact, idempotent, provenance-logged, and rejects
  source drift.
- The existing experiment contract suite and shell syntax checks pass.
- A DSpark FAP canary reaches CUDA Graph completion and two GRPO steps without
  CUDA illegal-memory access before the 200-step matrix is resubmitted.
