# CW H100 comparison report

## Run status

| Model | Default transport | NCCL reshard |
| --- | --- | --- |
| Qwen3-32B 8n8g | `20697714` | Pending submission |
| Qwen3-235B-A22B 32n8g | `20697737` | Pending submission |

Both default-transport jobs were pending when their account was changed to
`coreai_dlalgo_llm`. Record the Slurm state, completed step count, CUDA errors,
accuracy proxies, and steady-state timing after both arms start.

## Results

No model run has completed yet.

## First attempt

Reshard jobs `20706452` (32B) and `20706513` (235B) were submitted under
`coreai_dlalgo_llm` and then canceled before training. The companion default
transport job failed during Ray deserialization because the launcher replaced
`PYTHONPATH` and hid the container's Megatron-LM path. This branch now includes
the path and checks `import megatron` at driver startup before GRPO begins.
Replacement job IDs will be recorded after submission.
