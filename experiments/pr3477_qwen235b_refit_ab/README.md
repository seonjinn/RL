# PR 3477 Qwen3-235B Refit A/B

This experiment measures whether PR 3477's NCCL-Reshard path works for BF16
training plus MXFP8 rollout on Qwen3-235B-A22B, and how much refit time it
saves versus the legacy non-colocated collective path.

The reportable pair uses 16 full GCP-NRT B200 nodes (128 GPUs), split evenly
between training and generation. The trainer uses TP2/PP4/CP2/EP16 on 64 GPUs.
Generation uses TP4/PP1/DP16 on 64 GPUs, with two independent vLLM engines per
8-GPU node. Only `policy.generation.refit_transport` differs between arms.

See [PLAN.md](PLAN.md) for the fixed setup and commands. Runtime metadata,
SLURM logs, and W&B run identifiers are written under the remote experiment
root printed by the submission script.

## Result

NCCL-Reshard reduced average refit time by 28.7%, from 18.71 to 13.35
seconds. This reduced E2E step time by 1.8% and increased E2E throughput by
1.7%. Generation, policy training, and logprob performance remained within
0.6% of the legacy arm.

The matched window is steps 3-20 inclusive. Both runs contain all 18 requested
E2E samples. Transfer/update is averaged over the 17 steps that performed a
real refit; step 11 was a no-op refit in both arms. Recipe-default periodic
validation ran at steps 10 and 20 in both arms and is included in E2E time.

| Metric | Legacy | NCCL-Reshard | Delta |
| --- | ---: | ---: | ---: |
| Refit total / step (s) | 18.71 | 13.35 | -28.7% |
| Refit transfer/update / event (s) | 19.81 | 14.13 | -28.7% |
| E2E step time (s) | 315.90 | 310.33 | -1.8% |
| E2E throughput (tokens/s/GPU) | 101.42 | 103.16 | +1.7% |

| Component | Legacy time (s) | NCCL time (s) | Time delta | Legacy throughput (tokens/s/GPU) | NCCL throughput (tokens/s/GPU) | Throughput delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation | 104.40 | 104.82 | +0.4% | 596.06 | 592.83 | -0.5% |
| Policy training | 99.64 | 99.38 | -0.3% | 624.59 | 625.31 | +0.1% |
| Policy and reference logprobs | 68.38 | 68.12 | -0.4% | 910.21 | 912.90 | +0.3% |

W&B runs:

- Legacy: [mmuggufa](https://wandb.ai/nvidia/sna-pr3477-qwen235b-refit-ab/runs/mmuggufa)
- NCCL-Reshard: [yx7bj48u](https://wandb.ai/nvidia/sna-pr3477-qwen235b-refit-ab/runs/yx7bj48u)

Both W&B runs are in the `finished` state. The NCCL SLURM job returned exit 1
after all 20 steps and W&B finalization because Python shutdown attempted to
initialize a second Ray CoreWorker in the already-connected driver process.
This teardown-only failure does not remove or truncate the measured history.

## Execution History

| Jobs | Outcome | Interpretation |
| --- | --- | --- |
| `507182`, `507183` | Failed during environment setup | The original container did not provide the lockfile-required Python 3.13.14 interpreter. |
| `507329`, `507330` | Cancelled | The direct container interpreter used NCCL 2.30.4 while the source lock required NCCL 2.30.7, so this pair was not a valid source/runtime comparison. |
| `507350`, `507351` | Failed during worker-venv setup | Builders on multiple nodes concurrently rebuilt the same Lustre venv and raced in `rmtree` and package installation. No model, refit, or training step ran. |
| `508251`, `508252` | Cancelled during worker-venv setup | Node-local venvs removed the directory race but repeated dependency fetches on every node. One `TransferQueue` fetch failed with `curl 56`/early EOF and another `uv sync` stalled. |
| `508298`, `508299` | Cancelled after topology audit | The coordinated venv path was valid, but the inherited EP16 and PP4 required 64 trainer ranks while the non-colocated split supplied 32. |
| `508312`, `508313` | Failed during vLLM initialization | TP4 placed two MXFP8 engines on each 8-GPU B200 node. One vLLM engine hit a symmetric-memory OOM and worker SIGSEGV during FlashInfer autotuning. |
| `508491`, `508492` | Failed during vLLM initialization | Disabling vLLM symmetric all-reduce did not remove the two-engine-per-node initialization failure. |
| `508531`, `508532` | Failed during vLLM initialization | TP8 placed one engine per node, but Qwen3-235B MXFP8 MoE scale shuffle does not support that partition: `shape '[128, 4096, 6]' is invalid for input of size 4194304`. |
| `508561`, `508562` | Failed during launcher preflight | `ray.sub` rejected 4-GPU requests on physical 8-GPU nodes before Ray started. |
| `508571`, `508572` | Failed during wrapper startup | Slurm copied the partial-node wrapper into its spool, so a wrapper-relative `ray.sub` path was invalid. The wrapper now receives the absolute repo path. |
| `508584`, `508585` | Cancelled during Ray startup | The partial allocation provided 112 CPUs/node, but `ray.sub` requested the physical `CPUTot=224` for each internal `srun`, so no Ray step could start. The experiment now passes the allocated 112 CPUs explicitly. |
| `508599`, `508600` | Cancelled during Ray startup | Concurrent partial-node Ray clusters shared physical nodes and produced duplicate membership (`72/64` worker units). Partial-node Ray is not isolated enough for this A/B. |
| `508634` | Failed during vLLM profile warmup | A forked vLLM rank aborted while CuTe DSL compiled FlashInfer's MXFP8 quantization kernel. The replacement uses the `spawn` worker start method so CUTLASS/MLIR state is initialized independently. |
| `508660` | Failed during policy worker creation | `spawn` fixed the CuTe DSL abort and vLLM initialized, but a stale incomplete worker venv lacked `megatron.bridge`. Replacement runs use a run-scoped worker-venv namespace shared only by the matched pair. |
| `508689` | Cancelled after vLLM native failure | Fresh worker venvs fixed the missing Megatron dependency, but one generation rank segfaulted during profile warmup and caused a TCPStore cancellation cascade. A clean retry reuses the completed venv and kernel cache on a different node set. |
| `508746` | Cancelled after deterministic vLLM native failure | One generation engine emitted `CUDA driver error: out of memory` from `CUDASymmetricMemory`, then all eight ranks segfaulted and the engine failed. The shell export did not reliably cover vLLM's internal non-leader Ray workers. The replacement injects the setting through `vllm_cfg.env_vars` and sequentially prefetches both worker venvs before model startup. |
| `508859` | Cancelled after isolating FlashInfer fusion failure | Sequential prefetch completed both vLLM and Megatron worker environments, but the same symmetric-memory crash remained. vLLM 0.25.1's `AllReduceFusionPass` initializes a FlashInfer symmetric workspace independently of `VLLM_ALLREDUCE_USE_SYMM_MEM`; the replacement disables this fusion pass identically in both A/B arms. |
| `508958`, `508983` | Legacy failed at its first refit; dependent NCCL arm cancelled | Disabling all-reduce fusion and using `spawn` let all 32 vLLM and 32 Megatron workers initialize and enter step 1. The spawned vLLM workers then lacked NeMo-RL's process-local MXFP8 config and loader patches, causing `global_fp8_config` to be `None` and an unquantized weight shape mismatch. The replacement serializes the config through `VllmConfig` and applies the patches in each internal worker before model creation. |
| `509337` | Passed 16/16 targeted vLLM unit tests | Validated spawned-worker config propagation, lifecycle ordering, and idempotent FP8 patch application in the GCP-NRT runtime. |
| `509340`, `509341` | Legacy cancelled during policy initialization; dependent NCCL arm cancelled | One Megatron rank blocked in `AutoTokenizer.from_pretrained` while reading the external Hugging Face API. The reportable pair sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after confirming that all required model artifacts are cached. |
| `509626`, `509627` | Legacy cancelled after step 1; dependent NCCL arm cancelled | The corrected 8-node topology reached training, but its TP4/PP2/DP4 generation mesh took 1,259.26 seconds for step 1 and could not finish 20 steps within the four-hour batch limit. |
| `509723` | Completed 20/20 steps | The 16-node legacy arm completed in 2:10:04. Its report window is steps 3-20 inclusive with no missing E2E, generation, policy-training, or logprob samples. |
| `509724` | Completed 20/20 steps; failed during teardown | All measured history and W&B finalization completed. During Python shutdown, Ray aborted on `Check failed: !core_worker_process` because a finalizer attempted a second CoreWorker initialization in the connected driver process. |

The reportable replacement returns to full-node allocation and uses TP4/PP1,
which provides enough generation data parallelism to complete 20 steps within
the GCP-NRT batch limit while preserving the TP4 MXFP8 MoE shard layout.

Only runs that reach measured GRPO steps are eligible for the performance
comparison.
