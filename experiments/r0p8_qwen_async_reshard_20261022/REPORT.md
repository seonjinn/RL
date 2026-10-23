# CW H100 comparison report

## Run status

| Model | Default transport | NCCL reshard |
| --- | --- | --- |
| Qwen3-32B 8n8g | `20707493` (running) | `20706924` (running) |
| Qwen3-235B-A22B 32n8g | `20708178` (new retry) | `20708180` (new retry) |

All active comparison jobs use `coreai_dlalgo_llm`. Record completed step
counts, CUDA errors, accuracy proxies, and steady-state timing as each arm
progresses.

## Results

The 32B NCCL arm has completed one training step. The paired default arm has
not produced a comparable step yet.

## First attempt

Reshard jobs `20706452` (32B) and `20706513` (235B) were submitted under
`coreai_dlalgo_llm` and then canceled before training. The companion default
transport job failed during Ray deserialization because the launcher replaced
`PYTHONPATH` and hid the container's Megatron-LM path. This branch now includes
the path and checks `import megatron` at driver startup before GRPO begins.

## H100 startup and first training result

Replacement job `20706924` (32B) passed container import and vLLM setup,
completed step 1/10, and entered step 2. At step 1 its average reward was
`0.6040`, generation KL error `0.0009`, and total step time `175.01 s`.
This is one stochastic sample, not an accuracy comparison.

Job `20706934` (235B) reached `nccl_reshard_refit`, but vLLM TP0 had only
`1.12 GiB` free when `packed_broadcast_consumer` requested a `1.60 GiB`
misc-parameter buffer. It was canceled after the OOM to release 32 nodes.
The default packed-buffer target is 2% of GPU memory. The launcher now sets
`NRL_REFIT_BUFFER_MEMORY_RATIO=0.005` for 235B on both comparison arms; both
policy producers and vLLM consumers receive the same value. It also sets
`HF_HUB_OFFLINE=1` to avoid Hub API rate limiting during cached model load.

Job `20707504` (235B, source `90fb0c7a0`) was submitted under
`coreai_dlalgo_llm` after `sbatch --test-only` succeeded. Its run directory is
`results/qwen3-235b-32n8g-async-1off-triton-reshard-10step-20261023T001716Z`.

Job `20707504` still allocated a `1.60 GiB` buffer and OOMed in the same
`packed_broadcast_consumer` call. Exporting the 0.5% ratio in the submit
environment had not passed that value to the vLLM TP worker. It was canceled.
The paired default job `20707495` had passed its initial refit but was also
canceled so the next two 235B runs use identical worker-level settings.

The run drivers now apply
`++policy.megatron_cfg.env_vars.NRL_REFIT_BUFFER_MEMORY_RATIO=0.005` and
`++policy.generation.vllm_cfg.env_vars.NRL_REFIT_BUFFER_MEMORY_RATIO=0.005`
for 235B. Local config composition confirmed the same value on both sides.
Job `20708180` was submitted from source `6bd99916e` after
`sbatch --test-only` succeeded. Its run directory is
`results/qwen3-235b-32n8g-async-1off-triton-reshard-10step-20261023T003306Z`.
Runtime result pending.
