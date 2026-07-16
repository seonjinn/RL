# Qwen3-32B Thinking EAGLE3 DynamicSD Design

## Objective

Measure fixed EAGLE3 K1, K2, and K3 and a calibrated DynamicSD policy for
Qwen3-32B under the unmodified NeMo-RL performance recipe. Determine whether
K2 is a better fixed point than K1 or K3 and whether a batch-size-aware policy
can improve generation and end-to-end performance without changing sampling
or training behavior.

## Controlled Setup

All compared NeMo-RL runs use these controls:

- Target recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml`
- Target revision: Qwen3-32B snapshot
  `9216db5781bf21249d130ec9da846c4624c16137`
- Drafter: `RedHatAI/Qwen3-32B-Thinking-speculator.eagle3` revision
  `a1403e07b73a66fc9ef561463631c31864616933`
- Runtime: official vLLM 0.25.1 in the pinned NeMo-RL nightly container
- Topology: four Lyris GB200 nodes, four GPUs per node, `--segment=4`, no
  `--gres`, dependency, or singleton constraint
- Sampling: temperature 1.0 and top-p 1.0 from the performance recipe
- Maximum output length: 4096 from the performance recipe
- CUDA Graph: `enforce_eager=false`, `FULL_AND_PIECEWISE`, native capture
  sizing, and `VLLM_USE_V2_MODEL_RUNNER=1`
- Checkpoint saving: disabled
- Final metric window: completed steps 2 through 20 inclusive

Only the SpecDec token policy changes between candidate runs. Dataset,
batching, placement, target and drafter weights, policy training, and logprob
configuration remain unchanged.

## Experiment Matrix

| Candidate | Purpose | Initial gate | Final gate |
|---|---|---|---|
| Baseline K0 | Matched no-SpecDec control | Reuse validated run | Reuse final20 run `2405077` |
| Thinking EAGLE3 K1 | Current fixed winner | Reuse validated run | Reuse final20 run `2405078` |
| Thinking EAGLE3 K2 | Test the unmeasured midpoint | New smoke2 | New final20 after smoke5 |
| Thinking EAGLE3 K3 | Measure the higher accepted-length point | Reuse smoke2 evidence, then smoke5 | New final20 |
| DynamicSD K0/K1/K2/K3 | Select the fastest K for the active batch-size range | New smoke2 with transition evidence | New final20 |

K0 remains available to DynamicSD. Forcing K1 at a batch size where the
baseline is faster would make the policy structurally unable to select the
best measured mode.

## DynamicSD Calibration

Calibration sweeps fixed K0, K1, K2, and K3 over the active batch-size buckets
observed by the Qwen3-32B performance recipe. It uses the same target,
drafter, topology, sampling, CUDA Graph mode, and sequence-length controls as
the NeMo-RL comparison. The generated
`num_speculative_tokens_per_batch_size` schedule must:

1. cover every batch size from one through the largest profiled active batch;
2. contain non-overlapping, contiguous ranges;
3. select only K0, K1, K2, or K3;
4. select K0 when no speculative candidate exceeds baseline throughput;
5. preserve the raw profile rows and derivation metadata beside the schedule.

The final report records the selected schedule. DynamicSD is not described as
calibrated when any range is inferred without a matching profile row.

## CUDA Graph Handling

Fixed-K runs use the native vLLM 0.25.1 path without a runtime patch.
DynamicSD enables the existing opt-in
`experiments/vllm_0251_eagle3_perfcfg/apply_vllm0251_dynamic_sd_cg_fix.py`
post-sync patch only inside the run-specific venv. The patch prevents the
autoregressive drafter's CUDA Graph manager from incorrectly applying dynamic
decode shapes to its fixed one-token decode path.

DynamicSD requires native capture sizing. A launcher request combining
DynamicSD with the compact capture profile fails before submission. Runtime
validation must record the resolved CUDA Graph mode and reject eager fallback,
unexpected PIECEWISE-only downgrade, or a missing per-K graph path.

## Metrics And Comparison

The W&B project remains `nemo-rl-vllm0251-drafter-matrix`. Every final row
reports arithmetic means over the exact completed step intersection, with
steps 2-20 required for a final claim:

- E2E step time and baseline-relative time speedup
- generation time and baseline-relative time speedup
- E2E and generation throughput in tokens/s/GPU and their speedups
- policy-training and policy/reference-logprob time
- generation-time ratio of E2E time
- SpecDec acceptance rate and mean accepted length
- selected-K counts and fraction for K0, K1, K2, and K3 in DynamicSD
- resolved CUDA Graph mode, coverage, and eager-fallback count
- reward, generation length, KL error, and loss as correctness diagnostics

Logged NeMo-RL throughput metrics are authoritative; throughput is not
reconstructed from averaged token counts and times.

## Validation Gates

1. Unit tests fail before K2 and DynamicSD matrix support is added, then pass
   with exact Hydra overrides and invalid-combination rejection.
2. Shell syntax, Ruff, Pyright, and the focused matrix tests pass locally.
3. The exact cluster checkout is clean, pushed to the private fork, pulled on
   Lyris, and recursively initializes all submodules.
4. `show` and scheduler `test-only` validate the exact four-node topology.
5. Smoke2 reaches two completed steps and emits timing, throughput,
   acceptance, selected-K, and CUDA Graph telemetry.
6. Smoke5 confirms at least one expected DynamicSD K transition when the
   observed active batch sizes cross schedule ranges.
7. Final20 completes every step from 2 through 20 without runtime fallback,
   missing required metrics, or a changed controlled setting.

The final comparison remains preliminary until the baseline and every
reported candidate have the complete matched window.

## Deliverables

- K2 and DynamicSD variants in the isolated vLLM 0.25.1 drafter-matrix
  worktree
- Focused launcher and matrix contract tests
- Immutable calibration profile and derived DynamicSD schedule
- Lyris job IDs and direct W&B links
- Updated experiment report with fixed-K and DynamicSD tables

No changes are pushed to NVIDIA-NeMo/RL. Development commits and experiment
launches use only the private fork branch.
