# Qwen3-30B-A3B GBS 2048 DSpark Study

This matched cohort uses the official
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`
workload unchanged: 64 prompts per step, 32 generations per prompt, GBS 2048,
4K maximum sequence length, sequence packing, and 4 nodes with 4 GPUs each.

The only runtime changes shared by every arm are BF16, FlashInfer TRTLLM MoE,
`FULL_AND_PIECEWISE` CUDA Graph mode, and `max_num_seqs=128`. SpecDec arms use
the PTV3-SWA step-44000 DSpark checkpoint. The ready matrix is a no-SpecDec
baseline plus fixed K3, K5, and K7, all for 20 steps.

Each arm uses 8-25 geometric CUDA Graph capture sizes. The lists include the
maximum target-verification and DSpark-draft shapes at 128 requests while
limiting padding to 2x. This replaces an exact-shape scheme that required
128-160 captures per arm and carried unnecessary startup and graph-memory cost.

DynamicSD is intentionally blocked in this vLLM 0.25.1 cohort. A result may be
called DynamicSD only after runtime counters prove both the scheduler-selected K
and a corresponding reduction in DSpark draft tokens or draft forward work.
The separate vLLM 0.29 standalone study owns that validation.

## Completed results

The following averages use W&B Steps 3-20. The baseline, K3, and K5 runs all
finished 20 steps with closely matched mean generation lengths (3,134-3,139
tokens per sample).

| Arm | SLURM | W&B | Generation TPS/GPU | Generation speedup | E2E TPS/GPU | E2E speedup | E2E step time |
|---|---:|---|---:|---:|---:|---:|---:|
| Baseline | 7088950 | [z6uj6prf](https://wandb.ai/nvidia/sna-specdec/runs/z6uj6prf) | 6,710.2 | 1.000x | 2,343.1 | 1.000x | 177.94 s |
| DSpark K3 | 7088952 | [y4z6wgzw](https://wandb.ai/nvidia/sna-specdec/runs/y4z6wgzw) | 9,990.4 | 1.489x | 2,685.0 | 1.146x | 155.00 s |
| DSpark K5 | 7088954 | [kdrqc5ia](https://wandb.ai/nvidia/sna-specdec/runs/kdrqc5ia) | 10,011.0 | 1.492x | 2,677.9 | 1.143x | 155.34 s |

K3 currently has the best E2E result by a small margin. K5 increases mean
accepted length from 2.694 to 3.203, but its extra verification work leaves
generation throughput effectively tied with K3 at this concurrency. K7 job
7088956 remains queued and must be added before the fixed-K sweep is complete.

The container already includes actor-specific environments. Jobs retain the
generation actor's prebuilt `/opt/ray_venvs/...VllmGenerationWorker/bin/python`
interpreter and prepare only the source-verified DSpark FAP overlay on
node-local `/raid/scratch`; they neither force the dependency-incomplete driver
interpreter onto generation actors nor rebuild a unique per-job uv venv.
