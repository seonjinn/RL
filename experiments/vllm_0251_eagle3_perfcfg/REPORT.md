# NeMo-RL vLLM 0.25.1 Eagle-3 Results

## Status

The isolated upgrade is runtime-validated on Lyris GB200. Matched 20-step
baseline, native CUDA Graph, and compact CUDA Graph runs completed successfully.
All reported performance metrics average W&B steps 2 through 20 inclusive.

## Configuration

| Field | Value |
|---|---|
| NeMo-RL base | upstream `main` at `2ba2f0a73` |
| Run commit | `df2d6d7a1bc5d4aad046c0c1d2ed913b8ddb0744` |
| Launcher follow-up commit | `747f22eb0` (topology/metrics validation) |
| Branch | `sna/nemorl-vllm0251-eagle3-fullcg-20260715` |
| vLLM | `0.25.1`, official CUDA 13 wheel, Model Runner V2 |
| Cluster | Lyris, 4 nodes x 4 GB200, `--segment=4` |
| Recipe | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` |
| Target | `Qwen/Qwen3-30B-A3B` snapshot `ad44e777...` |
| Drafter | `RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3` snapshot `a7ec796d...` |
| Rollout | sync GRPO, 64 prompts x 32 generations, max OSL 4096, temperature/top-p 1.0/1.0 |
| Parallelism | generation TP1; training EP16; 16 GPUs total |
| Runtime | Triton MoE, prefix caching enabled, `enforce_eager=false` |
| CUDA Graph | `FULL_AND_PIECEWISE`; native vLLM 0.25.1 sizing unless noted |
| Checkpointing | disabled |

Only speculative decoding and the explicitly named CUDA Graph profile differ
between matched rows. The native profile delegates graph-shape derivation to
vLLM 0.25.1 MRv2. The compact profile captures
`(K + 1) * [1, 2, 4, 8, 16, 32, 64]` for a reduced-memory coverage ablation.
For this run, native resolved to 51 shapes from 1 through 512, while compact
resolved to only `[4, 8, 16, 32, 64, 128, 256]`.

## Final Results

| Run | E2E time | Time speedup | E2E tok/s/GPU | TPS speedup | Gen time | Gen time speedup | Gen tok/s/GPU | Gen TPS speedup | Gen ratio | Acceptance | Mean acceptance length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 181.56 s | 1.00x | 2,295.3 | 1.00x | 69.30 s | 1.00x | 5,996.3 | 1.00x | 38.2% | n/a | n/a |
| Eagle-3 K3, native | 150.35 s | **1.21x** | 2,765.2 | **1.20x** | 42.52 s | **1.63x** | 9,790.0 | **1.63x** | 28.3% | 64.82% | 2.944 |
| Eagle-3 K3, compact | 212.47 s | **0.85x** | 1,959.6 | **0.85x** | 103.12 s | **0.67x** | 4,049.7 | **0.68x** | 48.5% | 64.83% | 2.944 |

NeMo's acceptance-length metric includes the guaranteed target token. The
native run accepted 1.944 draft tokens per verification, corresponding to a
reported acceptance length of 2.944.

### Stage Breakdown

| Run | Policy time | Policy tok/s/GPU | Logprob time | Logprob tok/s/GPU | Refit time | Mean tokens/sample |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 80.96 s | 4,118.0 | 19.95 s | 20,861.5 | 5.19 s | 3,137.40 |
| Eagle-3 K3, native | 79.18 s | 4,208.3 | 19.52 s | 21,307.5 | 4.96 s | 3,135.42 |
| Eagle-3 K3, compact | 80.10 s | 4,158.1 | 19.81 s | 20,994.7 | 5.10 s | 3,137.06 |

Policy, logprob, and refit stages do not regress in the native Eagle run. Mean
generated length differs from baseline by less than 0.1%.

### CUDA Graph Comparison

| Profile | Capture shapes | Capture time | Memory/worker | Result |
|---|---|---:|---:|---|
| Baseline native | 51 shapes, 1-512 | 13-14 s | 0.59 GiB | reference |
| Eagle native | 51 shapes, 1-512 | 13-14 s | 1.00 GiB | 1.63x generation TPS |
| Eagle compact | `[4,8,16,32,64,128,256]` | 4 s | 0.35 GiB | 0.68x generation TPS |

Native and compact acceptance are identical, so the compact regression is a
systems issue rather than drafter quality. The reduced list saves 0.65 GiB per
worker but leaves common runtime shapes outside graph coverage. vLLM 0.25.1
MRv2 native sizing should remain the default.

## Runs

| Variant | Job | W&B | Elapsed | State |
|---|---:|---|---:|---|
| Baseline, native | `2396072` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-eagle3-perfcfg/runs/gohnwph0) | 1:08:17 | completed |
| Eagle-3 K3, native | `2396073` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-eagle3-perfcfg/runs/dnylr7ys) | 0:58:07 | completed |
| Eagle-3 K3, compact | `2396074` | [run](https://wandb.ai/nvidia/nemo-rl-vllm0251-eagle3-perfcfg/runs/50ci0it2) | 1:18:57 | completed |

## Validation

```text
17 focused tests passed
git diff --check passed
bash -n passed
Ruff passed on changed Python and test files
```

The official vLLM 0.25.1 source patch test is idempotent. DynamicSD remains
opt-in and is not part of this fixed-K3 comparison. The Ray-dependent unit
suite cannot run in the local macOS environment because Ray is not installed;
the 4-node GB200 smoke and 20-step runs are the runtime integration gate.
