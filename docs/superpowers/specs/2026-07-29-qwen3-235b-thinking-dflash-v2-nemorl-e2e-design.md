# Qwen3-235B Thinking DFlash v2 NeMo-RL E2E Benchmark Design

## Objective

Measure whether the Qwen3-235B DFlash v2 drafter improves generation and
end-to-end NeMo-RL performance when the verifier is
`Qwen/Qwen3-235B-A22B-Thinking-2507`.

The primary experiment is a matched Math NeMo-RL performance-recipe comparison.
A SWE-RL comparison follows only after the Math path proves that the DFlash
checkpoint, vLLM 0.25.1 runtime, refit, and policy update work together.

## Decision

Use a staged Math-first design:

1. Run no-SpecDec and DFlash v2 K3 with the same Math performance recipe.
2. Promote each arm through one-step, three-step, and twenty-step gates.
3. After the Math twenty-step comparison is valid, run the matched SWE-RL
   comparison with no-SpecDec and DFlash v2 K5.

This design isolates NeMo-RL integration and Amdahl effects before introducing
the longer and less deterministic OpenHands/SWE environment path.

## Fixed Runtime Inputs

| Input | Value |
|---|---|
| Verifier | `Qwen/Qwen3-235B-A22B-Thinking-2507` |
| Verifier snapshot | `/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5` |
| Drafter | `/home/sna/drafters/dflash_235bthink_v2` |
| Container | `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh` |
| Generation venv | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nrl_venvs_dynsd025` |
| vLLM family | 0.25.1 |
| Sampling | `temperature=1.0`, `top_p=1.0`, no top-k truncation |
| DFlash draft TP | 1 |
| DFlash attention | `FLASH_ATTN` |
| FlashInfer autotune | disabled |
| CUDA Graph mode | `FULL` |
| W&B | enabled; exact run URL recorded in the job ledger and HTML report |

The baseline uses the same verifier, container, topology, data, sampling,
sequence limits, seed policy, and CUDA Graph policy, with speculative decoding
disabled.

## Experiment Matrix

### Phase A: Math NeMo-RL Performance Recipe

| Arm | SpecDec | K | Purpose |
|---|---|---:|---|
| Math baseline | disabled | 0 | Matched target-only reference |
| Math DFlash v2 | DFlash v2 | 3 | Best measured DFlash v2 OpenMath setting |

The Math result already has a positive engine-level gate: DFlash v2 K3 reached
1.812x target-only throughput in the fresh 192-sequence OpenMath rollout.

Each arm advances through:

- one completed step for initialization, generation, refit, and optimizer
  compatibility;
- three completed steps for stable timing and SpecDec counters;
- twenty completed steps for the reported comparison.

Step 1 is treated as cold start. The primary steady-state window is steps 2-20.

### Phase B: SWE-RL

| Arm | SpecDec | K | Purpose |
|---|---|---:|---|
| SWE baseline | disabled | 0 | Matched OpenHands/SWE reference |
| SWE DFlash v2 | DFlash v2 | 5 | Best measured held-out SWE setting |

Phase B reuses the existing Thinking-2507 SWE2 configuration and the same
baseline/DFlash topology. The initial comparison advances through one and three
completed steps only after the collector and environment setup pass the
previous gate. A twenty-step SWE run is not part of this initial design because
the existing one-step SWERL allocation already reserves 2.5 hours and the Lyris
`gb200` partition is limited to 5 hours. A longer SWE study requires a separate
checkpoint-and-resume design after the three-step result is positive.

The existing one-step launcher is an input, not the final benchmark: it
currently disables W&B and must not be used for the reported twenty-step
comparison without enabling W&B and recording the exact run URL.

## CUDA Graph Contract

For each DFlash arm, the explicit capture list must contain `(K+1)` multiples
through `effective_max_num_seqs * (K+1)`.

- Math K3 requires four-token verification multiples.
- SWE K5 requires six-token verification multiples.

The effective `max_num_seqs`, resolved Hydra configuration, and final capture
list are written to the ledger before submission. A run is invalid if the
verification batch falls outside the captured sizes and silently uses eager
execution.

The baseline retains the same compilation mode and resource limits. CUDA Graph
startup time is reported separately from steady-state step metrics.

## Submission and Promotion Gates

Before each submission:

1. Confirm the source commit and launcher checksum.
2. Confirm verifier, drafter, container, venv, dataset, and cache paths.
3. Run the launcher in dry-run mode.
4. Run `sbatch --test-only` with the exact node count, partition, account,
   segment, and wall time.
5. Commit and push the experiment definition.

After submission:

1. Monitor scheduler state and driver logs for at least five minutes.
2. Reject the run on model/config mismatch, eager verification fallback,
   missing W&B URL, CUDA error, refit failure, empty trajectory batch, or
   incomplete policy update.
3. Promote to the next step count only when both baseline and DFlash arms pass
   the same gate.

The Math and SWE phases use separate run groups and ledgers so a SWE
infrastructure failure cannot invalidate the Math result.

## Required Metrics

The report records these metrics per arm:

- generation throughput in tokens/s/GPU;
- generation time and baseline-relative speedup;
- total E2E step time and baseline-relative speedup;
- E2E throughput in tokens/s/GPU and baseline-relative speedup;
- DFlash token acceptance rate;
- mean accepted length;
- total completion tokens;
- reward and success/pass metrics emitted by the task;
- completed steps, selected metric window, topology, max OSL, and W&B URL.

Speedups are computed only from the matched baseline in the same phase and
promotion tier.

## Interpretation

The twenty-step Math run is considered a performance win when:

- generation throughput improves by at least 1.20x;
- E2E throughput or E2E step time improves by at least 1.05x;
- all twenty steps complete without numerical or training instability;
- reward metrics show no clear regression requiring a correctness
  investigation.

Results below these thresholds are still retained. The timing breakdown is used
to distinguish weak DFlash acceptance from an Amdahl-limited E2E result.

The SWE comparison is reported with both performance and task outcome metrics.
Three-trajectory rollout-only results are supporting evidence, not a substitute
for the matched NeMo-RL run.

## Reporting

Every promoted run is added to the experiment ledger with:

- SLURM job ID and state;
- exact W&B run URL;
- source commit, launcher checksum, container, and model paths;
- resolved configuration and metric window;
- failure reason when a gate does not pass.

The final matched rows are added to
`docs/specdec_reports_index_latest.html` and regenerated into
`public/index.html`. Missing W&B links are shown as missing evidence rather than
replaced with a project-level or guessed run link.
