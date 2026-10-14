# Release 0.8 HybridEP policy OOM investigation

Status: CW retry jobs submitted on 2026-10-14; GPU allocations pending. No new policy-training result is available yet.

## Existing MiniMax failure

The supplied driver log is `/lustre/fsw/portfolios/coreai/users/yukih/NeMo-RL/code_snapshots/grpo-minimax-m27-dapo-8n8g-automodel/20016923-logs/ray-driver.log`. It records the first policy training step failing in `AutomodelPolicyWorker.train()` at `self.optimizer.step()`, specifically PyTorch Adam's `_multi_tensor_adam` calling `torch._foreach_sqrt(device_exp_avg_sqs)`. The failed allocation was 144 MiB. GPU 0 had 79.11 GiB capacity and 17.38 MiB free; the policy process held 73.27 GiB including non-PyTorch memory, and another process held 5.71 GiB. The log also shows `expandable_segments` failing to map 20 MiB. This is optimizer-step peak memory, not a stack trace inside the HybridEP dispatch call.

That prior job used `git_meta=d2a4670d2a` and a `yukih-r0.8.0-bump-cudnn-frontend-nixl-d2a4670d2` container. The new runs use the user-provided `r0.8.0-19666f86c` image. A dispatcher comparison within the new image is needed before attributing the difference solely to HybridEP.

## First CW attempt

Image smoke job `20206465` completed successfully: eight H100 GPUs were visible and PyTorch was `2.13.0+cu130`. The image SHA256 is `a78e909f1877d48cc1046cb124e66a6fee81ddec99c461cd676b6e81993c51f6` (CPU hash job `20205982`).

Qwen HybridEP job `20206783` and DeepEP job `20207035` both failed before model loading. Their Ray clusters reached 32/32 workers, then `uv run` failed because the mounted checkout lacked `3rdparty/Gym-workspace/Gym/pyproject.toml`. The pinned image itself contains that submodule and `/opt/nemo-rl/.git/refs/heads/r0.8.0` is `19666f86ca3b1988880b2b8b9107c575608c4ab0`. The submit script now runs the driver from `/opt/nemo-rl`. The first MiniMax jobs were cancelled while still pending so they would not run the faulty command.

## Retry run matrix

| Job ID | Model | Dispatcher | Other change | Dependency | Status at submission |
| --- | --- | --- | --- | --- | --- |
| 20213842 | Qwen3.5 35B A3B, 4×8 H100 | HybridEP | None | None | Pending: QOSGrpNodeLimit |
| 20213860 | Qwen3.5 35B A3B, 4×8 H100 | DeepEP | None | afterany:20213842 | Pending: dependency |
| 20213844 | MiniMax M2.7, 8×8 H100 | HybridEP | None | None | Pending: Priority |
| 20213864 | MiniMax M2.7, 8×8 H100 | DeepEP | None | afterany:20213844 | Pending: dependency |
| 20213872 | MiniMax M2.7, 8×8 H100 | HybridEP | `+policy.optimizer.kwargs.foreach=false` | afterany:20213864 | Pending: dependency |

CPU `uv run` preflight job `20213922` is running. All training jobs use the shared `HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home` and the two release recipe topologies. Check each job's `manifest.txt` and `run.log` below `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/results` for exact commands and outcomes.

## Working hypothesis

MiniMax already has `train_micro_batch_size=1` and activation checkpointing enabled. HybridEP may raise the persistent GPU memory floor through its dispatch buffers, leaving insufficient room for AdamW's first-step foreach temporary tensors. This remains an inference until the new HybridEP and DeepEP runs reach the same optimizer step. `foreach=false` directly tests the allocation site; if it is insufficient, inspect GPU memory around optimizer state creation and the vLLM sleep residue before reducing sequence length or changing EP size.

Qwen uses Transformer Engine `FusedAdam`, so the MiniMax foreach control does not apply to its optimizer.
