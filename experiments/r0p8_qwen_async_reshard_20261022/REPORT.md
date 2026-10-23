# CW H100 comparison report

## Run status

| Model | Default transport | NCCL reshard |
| --- | --- | --- |
| Qwen3-32B 8n8g | `20707493` (completed, 10/10 steps) | `20706924` (completed, 10/10 steps) |
| Qwen3-235B-A22B 32n8g | `20708681` (completed, 10/10 steps) | `20708685` (completed, 10/10 steps) |

All comparison jobs used `coreai_dlalgo_llm` and finished with Slurm
`COMPLETED (0:0)`.

## Results

The 32B NCCL arm completed 10/10 steps and Slurm reported `COMPLETED (0:0)`.
The paired default arm also completed 10/10 steps with Slurm `COMPLETED (0:0)`.
Both used the same r0.8 source,
Triton generation setting, model cache, and 8-node H100 topology.

| Metric | Default refit | NCCL reshard |
| --- | ---: | ---: |
| Weight sync, mean steps 2–9 | 0.89375 s | 0.50375 s |
| Total step time, mean steps 2–9 | 150.20 s | 149.275 s |
| Reward, mean steps 2–9 | 0.49712 | 0.49657 |
| Generation KL error, mean steps 2–9 | 0.00097 | 0.00097 |
| Reward, mean all 10 steps | 0.51782 | 0.51669 |
| Generation KL error, mean all 10 steps | 0.00097 | 0.00096 |

NCCL weight sync was `43.6%` shorter (`1.77x` faster), while total step time
was only `0.6%` shorter. Step 1 includes startup and step 10 includes extra
end-of-run time (`272.40 s` default, `271.90 s` NCCL), so both are excluded
from steady-state timing. Reward and KL show no obvious short-run regression;
10 stochastic steps do not establish equal long-run convergence.

The 235B paired runs also completed 10/10 steps without the earlier vLLM
packed-buffer OOM. Both used the same r0.8 source, Triton generation backend,
32-node H100 topology, and `NRL_REFIT_BUFFER_MEMORY_RATIO="0.005"` in the
policy and vLLM worker environments. The transport setting differed.

| Metric | Default refit | NCCL reshard |
| --- | ---: | ---: |
| Weight sync, mean steps 2–9 | 5.825 s | 1.82125 s |
| Total step time, mean steps 2–9 | 120.00125 s | 116.48875 s |
| Reward, mean steps 2–9 | 0.59252 | 0.59106 |
| Generation KL error, mean steps 2–9 | 0.00695 | 0.00700 |
| Reward, mean all 10 steps | 0.59160 | 0.58887 |
| Generation KL error, mean all 10 steps | 0.00679 | 0.00681 |

NCCL weight sync was `68.7%` shorter (`3.20x` faster); mean step time was
`2.9%` shorter over steps 2–9. Step 1 includes startup and step 10 includes
end-of-run work, so neither is included in steady-state timing. Reward and
generation KL differ slightly between stochastic runs. These 10-step training
proxies show no clear accuracy regression, but cannot establish that long-run
accuracy or convergence is unchanged.

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
The default packed-buffer target is 2% of GPU memory. The first retry
exported `NRL_REFIT_BUFFER_MEMORY_RATIO=0.005` for 235B in both submit
environments; the later OOM below showed that this did not reach the vLLM TP
workers. The launchers also set `HF_HUB_OFFLINE=1` to avoid Hub API rate
limiting during cached model load.

Job `20707504` (235B, source `90fb0c7a0`) was submitted under
`coreai_dlalgo_llm` after `sbatch --test-only` succeeded. Its run directory is
`results/qwen3-235b-32n8g-async-1off-triton-reshard-10step-20261023T001716Z`.

Job `20707504` still allocated a `1.60 GiB` buffer and OOMed in the same
`packed_broadcast_consumer` call. Exporting the 0.5% ratio in the submit
environment had not passed that value to the vLLM TP worker. It was canceled.
The paired default job `20707495` had passed its initial refit but was also
canceled so the next two 235B runs use identical worker-level settings.

The run drivers then added worker-specific Hydra overrides for the policy and
vLLM generation environments. The first version used an unquoted `0.005`;
the corrected quoted version is described below.
Job `20708180` was submitted from source `6bd99916e` after
`sbatch --test-only` succeeded. Its run directory is
`results/qwen3-235b-32n8g-async-1off-triton-reshard-10step-20261023T003306Z`.
It failed before startup: Hydra parsed the unquoted CLI value `0.005` as a
float, while `MasterConfig` requires string environment values. Job
`20708178` on the default arm failed the same validation. Both drivers now
pass the quoted Hydra value `"0.005"`; local composition confirmed string
values on both policy and vLLM sides.

Job `20708685` was submitted from source `a6e3dcd57` under
`coreai_dlalgo_llm` after `sbatch --test-only` succeeded. Its run directory
is `results/qwen3-235b-32n8g-async-1off-triton-reshard-10step-20261023T004141Z`.
It completed all 10 steps without the earlier vLLM buffer OOM. The paired
default job `20708681` also completed all 10 steps. Their accuracy proxies
and steady-state timing are compared above.
