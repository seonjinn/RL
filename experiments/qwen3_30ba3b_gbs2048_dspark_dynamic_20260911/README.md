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

The container already includes actor-specific environments. Jobs retain the
generation actor's prebuilt `/opt/ray_venvs/...VllmGenerationWorker/bin/python`
interpreter and prepare only the source-verified DSpark FAP overlay on
node-local `/raid/scratch`; they neither force the dependency-incomplete driver
interpreter onto generation actors nor rebuild a unique per-job uv venv.
