# Qwen3-32B Thinking EAGLE3 K0-K5 Calibration Plan

**Goal:** Generate a matched vLLM 0.25.1 batch-size-aware K0-K5 schedule and
use it for a gated NeMo-RL final20 DynamicSD run.

## Constraints

- Keep the Qwen3-32B performance recipe controls unchanged.
- Profile K0-K5 before final20; do not use a heuristic schedule for claims.
- CUDA Graph remains enabled with `FULL_AND_PIECEWISE`.
- Commit and push to the private fork before cluster submission.
- Run scheduler test-only and monitor every submitted job for five minutes.

### Task 1: Implement The Pure Calibrator

- [ ] Add failing tests for profile validation and upstream goodput semantics.
- [ ] Implement typed profile parsing, interpolation, K selection, and ranges.
- [ ] Emit deterministic schedule JSON with raw-profile SHA-256.
- [ ] Verify focused pytest, Ruff, and Pyright.

### Task 2: Extend The Matrix To K5 DynamicSD

- [ ] Add fixed Thinking K4 as an explicit profiler/replay control.
- [ ] Add a DynamicSD K5 variant and a versioned max-K schedule field.
- [ ] Reject variant/schedule max-K mismatch before Hydra serialization.
- [ ] Preserve K0-K3 seed compatibility for smoke-only historical runs.

### Task 3: Build The Matched GPU Profiler

- [ ] Render OpenMathInstruct-2 prompts with the performance recipe template.
- [ ] Profile K0-K5 at batch sizes 1,4,16,32,64,128,192,256.
- [ ] Use twenty steady-state batches per cell and capture median ITL.
- [ ] Record K5 position-level acceptance and exact runtime provenance.
- [ ] Refuse artifact publication when any grid cell or metric is missing.

### Task 4: Run Calibration On Lyris

- [ ] Commit/push, pull on Lyris, and initialize recursive submodules.
- [ ] Pass exact scheduler test-only.
- [ ] Submit the profile job(s) and monitor for at least five minutes.
- [ ] Generate and review the immutable calibrated schedule artifact.

### Task 5: Promote To Final20

- [ ] Replay the schedule against fixed K0-K5 under matched profile controls.
- [ ] Allowlist the reviewed schedule SHA-256.
- [ ] Pass DynamicSD final20 test-only, submit, and monitor.
- [ ] Report exact completed steps 2-20 with W&B and artifact links.

