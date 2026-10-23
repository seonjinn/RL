# CW H100 Llama3.1 FP8 reshard smoke

The first submission, `20706751`, was canceled while pending because the
shared launcher overwrote `PYTHONPATH` and hid Megatron-LM in the companion
Qwen experiment. The launcher now preserves Megatron-LM and checks its import
before GRPO starts.

Replacement job `20706978` ran under `coreai_dlalgo_llm` and passed the
Megatron import preflight. It failed during policy worker initialization in
`param_and_grad_buffer.reload_from_cpu`: PyTorch needed another `13.00 GiB`
on an H100 with `12.72 GiB` free. It did not reach NCCL refit or a training
step. This 2-node FP8 recipe needs a separate policy-memory adjustment before
it can validate the refit transport on H100.
