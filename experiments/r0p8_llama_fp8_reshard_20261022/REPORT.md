# CW H100 Llama3.1 FP8 reshard smoke

The first submission, `20706751`, was canceled while pending because the
shared launcher overwrote `PYTHONPATH` and hid Megatron-LM in the companion
Qwen experiment. The launcher now preserves Megatron-LM and checks its import
before GRPO starts. Replacement submission and runtime results pending.
