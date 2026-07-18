# Qwen3-235B DynamicSD K5 Profile Design

## Objective

Measure whether extending Qwen3-235B EAGLE3 DynamicSD from K0-K3 to K0-K5
improves NeMo-RL rollout performance. Preserve the existing performance recipe,
sampling policy, model revisions, and CUDA Graph execution so that K range is
the only intentional behavioral change.

## Fixed Experiment Contract

- Target: `Qwen/Qwen3-235B-A22B` at
  `8efa61729e24bd65b1d152b5ab5409052aa80e65`.
- Drafter: `RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3` at
  `3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87`.
- Runtime: vLLM 0.25.1, Model Runner V2, `FULL_AND_PIECEWISE` CUDA Graphs.
- Topology: target TP8 across two Lyris GB200 nodes, draft TP1.
- Sampling: temperature 1.0 and top-p 1.0.
- Workload: pinned OpenMathInstruct-2 prompts, 256 generated tokens per
  profiling request, and 20 measured batches per point.
- Batch-size grid: 1, 4, 8, 16, 32, 48, and 64.
- K grid: K0 through K5, measured in independent jobs.
- Engine limits: max model length 8192, max batched tokens 2048, max sequences
  128, and active profile endpoint 64.
- Prefix caching and chunked prefill remain enabled; Triton remains the MoE
  backend.

## CUDA Graph Coverage

The capture list must cover the verification shape at the largest active batch
for every K. K4 requires `64 * (4 + 1) = 320` tokens and K5 requires
`64 * (5 + 1) = 384` tokens. The matched profile and NeMo-RL variants therefore
use:

`[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384]`

K0-K3 will be re-profiled rather than mixed with the previous K0-K3 artifact.
This keeps global maximum K, capture descriptors, provenance, and acceptance
telemetry consistent within one immutable profile.

## Schedule Derivation

For every measured batch size and K, compute:

`goodput(K) = expected accepted length(K) / median inter-token latency(K)`

The expected accepted length is `1 + sum(position acceptance through K)`. The
calibrator linearly interpolates between measured batch-size points and selects
the K with maximum goodput. The resulting schedule records the exact profile
SHA-256 and vLLM/CUDA Graph contract.

## Promotion Gates

1. All 42 latency cells and the K5 position-level acceptance profile must be
   complete before schedule generation.
2. A two-step NeMo-RL smoke must confirm the requested K, selected K, verified
   draft width, acceptance metrics, and CUDA Graph mode without fallback.
3. A five-step run must complete before final promotion.
4. The reviewed schedule SHA-256 must be allowlisted before a 20-step run.
5. The reportable comparison uses steps 2-20 and the matched baseline and fixed
   K3 runs. It reports generation and E2E time, throughput per GPU, acceptance,
   accepted length, and generation-time ratio.

## Outcomes

- If K4 or K5 is selected, run the calibrated DynamicSD schedule through the
  full promotion sequence and compare it with fixed K3.
- If K3 remains optimal for all batch sizes, retain the profile as evidence and
  do not spend a 20-step allocation on a schedule equivalent to fixed K3.
- If K4/K5 causes OOM or eager fallback, record the exact capture failure and
  stop promotion rather than reducing CUDA Graph coverage silently.

## Validation

- Unit tests verify the Qwen3-235B K0-K5 grid and required capture endpoints.
- Submission `show` and `test-only` verify the two-node profile topology,
  `--segment=2`, absence of GRES, pinned revisions, and immutable output root.
- Completed profile artifacts are validated by the existing strict schema and
  schedule calibrator before any NeMo-RL submission.
