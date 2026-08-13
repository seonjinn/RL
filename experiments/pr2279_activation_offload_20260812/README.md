# PR 2279 activation CPU-offload evidence

This experiment validates NeMo-RL PR #2279 on the official Qwen3-30B-A3B
4-node/16-GPU GB200 performance workload.

The OFF and ON arms keep the same Transformer Engine CUDA-graph path and the
same `NVTE_CPU_OFFLOAD_V1=1` environment. The only factor is
`fine_grained_activation_offloading=false/null` versus
`true/["moe_act"]`. This dependency-matched design avoids attributing CUDA-graph
speed or CPU-offload protocol changes to activation offload itself.

## Evidence revisions

| Phase | Revision |
| --- | --- |
| Historical pre-fix runtime for jobs 2677022/2677024 | `2f39df66d6fd0a0b1b53cf472eb1599c6c05dfce` |
| Historical harness source | `4f0a1e45cb6bdee3a69ca019644411fd725c463b` |
| Final lifecycle runtime under validation | `01398467224921c058a70702cb4a8285eb98fc71` |
| Megatron-Bridge | `0c565c9a063bc8578ec094b4eb488594ed8b033a` |
| Megatron-LM | `d12f6c8c9aff51e166d872fd70151687a8e3f375` |
| Transformer Engine | `2.15.0+42b84005` |
| Lyris container SHA256 | `aa7621512376f5562a950708625f0db011f2a240d4fc1a612083489df4d8e8ab` |

The submitter requires the immutable evidence commit through
`EXPECTED_SOURCE_COMMIT`, verifies that it descends from the final lifecycle
runtime, and rejects runtime-file changes outside this experiment directory.

## Acceptance gates

- The ON-only lifecycle gate completes three optimizer updates with finite
  `train/loss` and `train/grad_norm` through step 3.
- At least one complete MCore summary contains ranks 0 through 15 and strictly
  positive `moe_act` and Total MiB values on every rank.
- Policy training, logprob, generation, and refit complete without a
  feature-specific exception.
- The source tree is clean and the exact source, runtime, dependency pins,
  config digest, container metadata, and command are recorded.
- Passing the ON-only gate proves activation and stability, not speedup.
- A performance claim requires dependency-matched OFF/ON arms at the same fixed
  runtime, same prompt/token workload, and replicated alternating or concurrent
  pairs.
- Same prompt/token workload across arms
- Report warmup-excluded policy, logprob, generation, refit, E2E throughput
- Report peak GPU/host memory from job telemetry when available

## Failure ledger

| Cluster | Job | Result | Root cause | Resolution |
| --- | --- | --- | --- | --- |
| Pre-Tyche | 2572885 | Harness failure | `sbatch --wrap` used `/bin/sh`; Bash `pipefail` was invalid | Changed focused unit wrapper to POSIX `set -eu` |
| Lyris | 2676543 | Harness failure | Same `/bin/sh` issue | Changed focused unit wrapper to POSIX `set -eu` |
| Lyris | 2676553 | Image failure | 2026-07-26 nightly `uv` could not resolve Python 3.13.14 | Stage a fresh immutable nightly before performance submission |
| Pre-Tyche | 2572912 | Setup timeout | TE source build targeted seven CUDA architectures and exceeded the 30-minute job limit before tests started | Reused the build cache and resubmitted with a one-hour limit as 2573195 |
| Lyris | 2676585 | Passed | Fresh nightly image staged with immutable metadata and SHA256 | Image SHA256 `aa7621512376f5562a950708625f0db011f2a240d4fc1a612083489df4d8e8ab` |
| Lyris | 2676732 | Smoke harness failure | The base image intentionally lacks the mcore-extra TE Python package; the smoke asserted too much | Validate base image uv/Python/torch/ray/CUDA only; TE is built in the source-pinned mcore venv |
| Lyris | 2676760 | Passed | Fresh image recognized 4 GB200 GPUs with torch 2.11.0+cu130, CUDA 13.0, Ray 2.56.1 | Use the immutable image for focused unit and performance jobs |
| Pre-Tyche | 2573195 | Harness failure | The focused file is marked `mcore`; NeMo-RL's unit-test hook excludes it unless `--mcore-only` is passed | Add `--mcore-only`, install the test dependency group in the source-pinned venv, and invoke that venv's Python directly |
| Lyris | 2676792 | Harness failure | Same missing `--mcore-only` selection caused zero collected tests after the dependency build completed | Reuse the completed cache with the corrected command |
| Pre-Tyche | 2573408 | Passed | Source-pinned Bridge/MCore/TE environment ran the corrected focused selection | 12 activation-offload setup tests passed |
| Lyris | 2677022, 2677024 | Execution passed; feature failed | Both 16-GB200 jobs completed 3/3 updates, but ON reported `moe_act=0.00 MB` and Total `0.00 MB` on ranks 0-15 | Invalidated as performance evidence; diagnose lifecycle and use arm-specific venv roots |
| Pre-Tyche | 2573688 | Expected RED | Forward-only schedule observed activation offload still enabled | Added forward-only suspension and restoration |
| Pre-Tyche | 2573756 | GREEN, incomplete design | Three initial lifecycle tests passed, but review found that a warmed manager could still lose cached chunks on later logprob phases | Added a stateful multi-update manager test and VPP atomicity tests |
| Pre-Tyche | 2573839 | Expected RED | `logprob -> train -> logprob -> train` reproduced cached forward-index consumption | Suspend both config-controlled initialization and the existing manager |
| Pre-Tyche | 2573853 | GREEN | Manager lifecycle and VPP atomicity tests passed 4/4 | Run the full focused regression set |
| Pre-Tyche | 2573862 | Passed | Runtime `aeeb3d6f5` passed all 56 selected training/setup tests | Reviewer corrected the test invariant from singleton absence to warmup/layout preservation |
| Pre-Tyche | 2573881 | Passed | Final runtime `013984672` passed 56/56 selected training/setup tests and completed `0:0` | Proceed to the ON-only GB200 lifecycle gate |
| Lyris | 2677648 | Passed | Final runtime `013984672` completed 3/3 Qwen3-30B-A3B updates; every rank reported positive `moe_act` offload and the automatic lifecycle checker accepted the run | Proceed to a controlled same-runtime OFF/ON performance comparison |
| GitHub CI | run 31650398312 | Infrastructure failure | Six functional shards hit the same flash-attn wheel HTTP/2 `refused stream` fetch error | No source change; rerun the external-network failures after approval |

## Root cause and fix

GRPO obtains current/reference logprobs before policy training. The logprob path
runs MCore with `torch.no_grad()` and `forward_only=True`, but the pre-fix model
still initialized fine-grained offload and the schedule reset its singleton.
That reset finalized an empty warmup layout, producing the 0.00-MB summary and
preventing policy training from discovering real saved activations.

The first fix disabled config-controlled initialization during forward-only
phases. Review then found a second lifecycle case: after the first real training
warmup, `TEGroupedMLP` retains its offload hook flags. A later logprob could
therefore advance cached chunks while the temporarily disabled config caused
the schedule to skip reset. The final fix also suspends the existing manager
and its cached chunks, preserves a pre-disabled nested state, restores in
`finally`, and discovers all VPP configs before mutating any. A first logprob may
create an empty singleton through cached MoE hooks, but it remains in warmup
with zero cached chunks so real training can initialize it.

## Post-fix GB200 lifecycle result

Lyris job `2677648` ran the official Qwen3-30B-A3B 4-node/16-GB200
performance workload at runtime `013984672`. It completed all three policy
updates and passed the automatic lifecycle checker.

| Gate | Result |
| --- | --- |
| `moe_act` summary | Positive on ranks 0-15 |
| Per-rank range | 3,246.56-5,205.24 MiB |
| Sum across ranks | 72,192.00 MiB (70.50 GiB) |
| Optimizer updates | 3/3 completed |
| Loss | `0.012618`, `-0.002424`, `-0.004723` |
| Gradient norm | `0.034980`, `0.037861`, `0.034533` |
| Automatic acceptance | `true` |

The MCore summary is the amount selected by the activation-offload manager; it
is not a measurement of net GPU-memory reduction. This ON-only gate establishes
that the fixed NeMo-RL lifecycle activates the feature and remains stable across
multiple GRPO steps. It does not establish a speedup or memory benefit.

## Historical diagnostic only

Jobs 2677022/2677024 are not evidence of activation-offload performance because
the ON arm offloaded zero bytes. Warm-step 2-3 observations are retained only
to document what was seen.

| Metric | OFF | ON | Diagnostic delta |
| --- | ---: | ---: | ---: |
| Step time | 181.085 s | 179.800 s | -0.71% |
| E2E throughput/GPU | 2336.735 tok/s | 2354.435 tok/s | +0.76% |
| Policy throughput/GPU | 5109.585 tok/s | 5133.500 tok/s | +0.47% |
| Logprob throughput/GPU | 20352.600 tok/s | 20370.155 tok/s | +0.09% |
| Generation throughput/GPU | 5931.200 tok/s | 6061.670 tok/s | +2.20% |

The pair had only two warmup-excluded samples, slightly different valid-token
workloads in steps 2-3, different node groups, and a recovered shared-venv
startup race. Across all three steps, E2E throughput was approximately 0.17%
worse with ON.

## Current validation

- Pre-Tyche focused unit: `2573408` passed 12/12 selected tests.
- Lyris Qwen3-30B-A3B historical pair: `2677022` / `2677024` completed,
  but the ON summary was zero-byte and performance is non-claimable.
- Pre-Tyche final lifecycle regression: `2573881` completed `0:0` and passed
  56/56 selected tests against runtime `013984672`.
- Lyris post-fix lifecycle gate: `2677648` completed `0:0`, passed 3/3 policy
  updates, and reported positive `moe_act` offload on all 16 ranks.
- Next gate: dependency-matched OFF/ON runs at the same runtime and fixed
  workload, followed by replicated measurements before a performance claim.
