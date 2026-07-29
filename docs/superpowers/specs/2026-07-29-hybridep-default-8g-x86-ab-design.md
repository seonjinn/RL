# HybridEP-Default MoE 8-GPU Recipe and CW A/B Design

## Goal

Make HybridEP the explicit default dispatcher for every supported MoE
performance recipe whose node shape is `*n8g`, while retaining matched
all-to-all recipes for performance and correctness comparisons. Validate the
result on CW-DFW H100 without changing `nemo_rl/utils/venvs.py`, and report
Policy training, LogProb, and end-to-end time and throughput for every
submitted workload.

## Scope

The HybridEP default applies to these MoE recipe families:

- Qwen3-30B-A3B: `grpo-qwen3-30ba3b-4n8g.yaml`;
- Qwen3-235B-A22B: `grpo-qwen3-235b-16n8g.yaml` and its
  `grpo-qwen3-235b-32n8g.yaml` derivative;
- Nemotron3 Super 120B-A12B:
  `grpo-nemotron3-super-120BA12B-32n8g.yaml`;
- DeepSeek-V3: `grpo-deepseek-v3-32n8g.yaml` and its
  `grpo-deepseek-v3-64n8g.yaml` derivative.

Dense `*n8g` recipes, including Qwen3-32B and Llama 3.1 8B, remain unchanged.
They do not contain experts and therefore cannot exercise HybridEP.

## Recipe Contract

Each base MoE recipe adds the same x86 NVL8 dispatcher block:

```yaml
policy:
  megatron_cfg:
    moe_token_dispatcher_type: flex
    moe_flex_dispatcher_backend: hybridep
    moe_hybridep_num_sms: 32
    env_vars:
      NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: "8"
      NUM_OF_TOKENS_PER_CHUNK_COMBINE_API: "128"
      NVLINK_DOMAIN_SIZE: "8"
      USE_MNNVL: "0"
```

Existing model-specific environment variables remain present. Derived
32-node and 64-node recipes inherit this block from their base recipe.

For each affected recipe family, preserve the prior dispatcher as a separate
`-alltoall.yaml` recipe. The baseline recipe inherits its HybridEP-default
counterpart and overrides only:

```yaml
policy:
  megatron_cfg:
    moe_token_dispatcher_type: alltoall
```

The baseline must remove inherited HybridEP-only backend, SM, and environment
settings through OmegaConf deletion overrides. A resolved-config contract test
must prove that every non-dispatcher field is identical between the two arms.

## Dependency and Runtime Contract

- Keep DeepEP pinned to
  `f725d29699f5bda9ba789456bb9579af69844685` for Linux x86_64.
- Build or reuse the immutable SM90 wheel with
  `TORCH_CUDA_ARCH_LIST=9.0` and `HYBRID_EP_MULTINODE=1`.
- Keep the current Megatron-Bridge and Megatron-LM gitlinks fixed.
- Do not change `nemo_rl/utils/venvs.py`.
- Avoid concurrent source rebuilds operationally: prepare the shared driver
  and actor environments before the A/B jobs, then run both arms with the same
  prebuilt environment and `NRL_FORCE_REBUILD_VENVS=false`.

The venv preparation rule is launch hygiene rather than part of x86 HybridEP
semantics. A source contract test compares `venvs.py` with the selected base
commit so a cache-isolation implementation cannot re-enter this change.

## Verification

Before cluster submission:

1. Add failing recipe-contract tests first.
2. Resolve every MoE HybridEP and all-to-all pair and compare the full
   configuration after excluding only dispatcher keys and run-label paths.
3. Verify all six MoE `*n8g` recipes resolve to `flex` plus `hybridep`, 32 SMs,
   and an eight-rank NVLink domain.
4. Verify the dense `*n8g` recipes do not select HybridEP.
5. Verify `nemo_rl/utils/venvs.py` has no diff from the selected base.
6. Run focused unit tests, recipe validation, shell syntax checks,
   `uv lock --check`, and `git diff --check`.
7. Commit with sign-off, push the exact branch, and pull it on CW before any
   submission.

## CW-DFW Test Gates

Use H100 nodes with eight GPUs per node and Lustre-backed code, environments,
model caches, and logs.

1. **Runtime gate:** import `deep_ep`, `deep_ep_cpp`, and `hybrid_ep_cpp` from
   the exact Ray runtime on an allocated H100.
2. **Small compatibility gate:** run matched three-step Qwen3-30B-A3B
   all-to-all and HybridEP arms.
3. **Representative multi-node gates:** after the small pair reaches the
   intended steps, submit matched short pairs for Qwen3-235B and Nemotron3
   Super using their valid performance topology. Submit DeepSeek-V3 only when
   a concrete model checkpoint replaces the recipe placeholder.
4. **Performance gate:** run matched 20-step arms for workloads that pass the
   compatibility gate and fit the available CW allocation.

Every job requires `sbatch --test-only`, the best current eligible FairShare
account, an exact clean pushed commit, and at least five minutes of monitoring.
Record node lists and scan bounded logs for actor loss, timeout, NCCL, CUDA,
RDMA, OOM, NaN, Inf, and model-config failures.

## Metrics

Clean A/B arms disable padding diagnostics. Use matched completed steps 5–20
for 20-step runs; for three-step compatibility runs, report all completed
steps but label the values as smoke-only.

For each workload and arm, report:

- Policy training mean, median, and total time;
- Policy training ratio-of-sums tokens/second/GPU;
- Policy and reference LogProb mean, median, and total time;
- Policy and reference LogProb ratio-of-sums tokens/second/GPU;
- end-to-end mean, median, and total step time;
- end-to-end ratio-of-sums tokens/second/GPU;
- generation time and throughput as supporting context;
- completed-step count and exact measurement window.

Use:

```text
throughput_change_percent =
    (HybridEP throughput / alltoall throughput - 1) × 100

time_reduction_percent =
    (1 - HybridEP time / alltoall time) × 100
```

Tokens/second/GPU must use the token count associated with the measured phase,
sum tokens across the selected window, divide by summed phase time, then
divide by the GPUs assigned to that phase. Do not average per-step throughput
ratios.

## Correctness and Overhead

For short-run numerical evidence, record reward, KL-related metrics, entropy,
gradient norm, validation accuracy, response length, and any non-finite value.
This is a smoke-level consistency check, not a convergence-equivalence claim.

Run HybridEP padding telemetry only in separate diagnostic jobs. Report the
weighted fake-token overhead,
`sum(added tokens) / sum(raw tokens)`, plus median, p95, and maximum per-call
overhead. Padding is transient activation, compute, and communication
overhead; it is not persistent storage overhead.

## Reporting

Continuously update the existing secret-free GitLab Pages report with:

- a concise conclusion for each workload;
- absolute Policy, LogProb, and end-to-end metrics;
- HybridEP-versus-all-to-all percentage changes;
- grouped bar charts with the exact step window in the caption;
- job IDs, terminal states, node lists, source and dependency commits,
  container and wheel hashes, recipe paths, and Lustre log paths;
- failure evidence and a clear distinction between compatibility smoke and
  valid performance data.

Large logs remain on Lustre. The Pages repository stores only HTML, small
structured summaries, plots, scripts, and configuration snapshots.
