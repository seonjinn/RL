# NeMo-RL worker-batch 32 generation speedup summary

Updated: 2026-06-04 PDT

## Question

Can NeMo-RL be configured so each vLLM generation engine sees about the same
batch size as vLLM standalone `bs=32`, and do we see generation speedup under
that setting?

## Batch matching rule

Approximate per-engine generation batch:

```text
requests_per_generation_engine ~= train_global_batch_size / number_of_generation_engines
number_of_generation_engines ~= total_generation_gpus / generation_tensor_parallel_size
```

This is not exactly identical to standalone `bs=32` because NeMo-RL uses real
rollout prompts, variable EOS/max-context stopping, policy/refit/logprob phases,
and vLLM worker lifecycle overhead. But it is the closest direct batch-shape
comparison for the generation engine.

## Qwen3-32B completed worker32 result

The completed Qwen3-32B worker32 run used:

| Item | Value |
| --- | --- |
| Jobs | baseline `3136000`, K=1 `3136001`, K=3 `3136002` |
| Slurm status | all completed |
| GPUs | `8x2` GPUs = 16 total GPUs |
| Generation TP | 1 |
| Generation engines | 16 |
| GBS | 512 |
| Estimated requests/engine | 32 |
| vLLM precision | BF16 generation, KV auto |
| Gate mode | always-on SpecDec |
| Drafter | public HF Qwen3-32B EAGLE3 drafter |

### Result

| Config | Matched scope | Mean generation throughput speedup | Mean E2E throughput speedup | Mean acceptance |
| --- | ---: | ---: | ---: | ---: |
| K=1 always | Step 1-3 generation, Step 1-2 E2E | 1.416x | 1.208x | 70.01% |
| K=3 always | Step 1-8 generation, Step 1-7 E2E | 1.646x | 1.312x | 45.82% |

Interpretation: yes, generation speedup is visible. Reducing Qwen3-32B from the
original `GBS=2048` shape to `GBS=512` improved the K=3 early generation signal
from the completed 20-step aggregate `1.356x` to about `1.646x` under worker
batch around 32.

## Comparison to vLLM standalone bs32

| Setting | K | Batch shape | Throughput speedup | Acceptance |
| --- | ---: | --- | ---: | ---: |
| vLLM standalone Qwen3-32B | 1 | bs32, ISL=1000, OSL=512 fixed decode | 1.625x | 79.9% |
| vLLM standalone Qwen3-32B | 3 | bs32, ISL=1000, OSL=512 fixed decode | 2.288x | 67.1% |
| NeMo-RL Qwen3-32B worker32 | 1 | GBS512 / 16 engines ~= 32 req/engine | 1.416x | 70.01% |
| NeMo-RL Qwen3-32B worker32 | 3 | GBS512 / 16 engines ~= 32 req/engine | 1.646x | 45.82% |

The remaining gap is mostly acceptance and workload mismatch:

- Standalone K=3 bs32 acceptance is `67.1%`.
- NeMo-RL worker32 K=3 acceptance is about `45.8%`.
- NeMo-RL rollouts are not fixed OSL=512 synthetic decoding; they are real
  rollout prompts with EOS/max-context stopping and extra orchestration overhead.

## Existing completed NeMo-RL always-on results for context

| Model | Shape | K | Generation speedup | E2E speedup | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-32B | GBS2048, about 128 req/engine | 1 | 1.346x | 1.154x | 69.45% |
| Qwen3-32B | GBS2048, about 128 req/engine | 3 | 1.356x | 1.181x | 45.28% |
| Qwen3-30B-A3B | GBS2048, about 128 req/engine | 1 | 1.344x | 1.099x | 57.51% |
| Qwen3-30B-A3B | GBS2048, about 128 req/engine | 3 | 1.177x | 1.068x | 31.85% |

## Newly submitted worker32 jobs

These are configured to match standalone-like bs32 at the generation-engine
level, but they have not produced results yet.

| Model | Jobs | Nodes/GPUs | Generation TP | Engines | GBS | Estimated req/engine | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B | `3153633` baseline, `3153634` K1, `3153635` K3 | 4 nodes x 4 GPUs | 1 | 16 | 512 | 32 | pending priority |
| Qwen3-235B-A22B | `3153149` baseline, `3153150` K1, `3153639` K3 | 32 nodes x 4 GPUs | 16 | 8 | 256 | 32 | pending priority |

Post-update online-drafter Step 2 jobs are also queued behind those Step 1 jobs:

| Model | Jobs | Dependency |
| --- | --- | --- |
| Qwen3-30B-A3B | `3153636` baseline, `3153637` K1, `3153638` K3 | after Step 1 jobs |
| Qwen3-235B-A22B | `3153640` baseline, `3153641` K1, `3153642` K3 | after Step 1 jobs |

## Bottom line

Generation speedup is visible when the NeMo-RL generation batch is shaped closer
to standalone `bs=32`. The clearest completed evidence is Qwen3-32B worker32:
K=1 reaches about `1.42x` generation speedup and K=3 reaches about `1.65x`.
This is better than the original GBS2048 shape, but still below standalone bs32,
especially for K=3, because acceptance is much lower in NeMo-RL.
