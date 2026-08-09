# PR 3477 Qwen3-235B Refit A/B

This experiment measures whether PR 3477's NCCL-Reshard path works for BF16
training plus MXFP8 rollout on Qwen3-235B-A22B, and how much refit time it
saves versus the legacy non-colocated collective path.

The pair uses 8 full GCP-NRT B200 nodes (64 GPUs). The trainer keeps the
upstream 32-GPU mesh. Generation uses TP4/PP2/DP4 so each vLLM engine occupies
one full 8-GPU node while preserving the TP4 MXFP8 MoE shard layout. Only
`policy.generation.refit_transport` differs between arms.

See [PLAN.md](PLAN.md) for the fixed setup and commands. Runtime metadata,
SLURM logs, and W&B run identifiers are written under the remote experiment
root printed by the submission script.

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

The reportable replacement returns to full-node allocation and combines the
recipe-native TP4 with PP2, yielding one 8-GPU generation engine per node.

Only runs that reach measured GRPO steps are eligible for the performance
comparison.
