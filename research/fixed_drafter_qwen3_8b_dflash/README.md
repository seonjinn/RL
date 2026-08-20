# Fixed DFlash acceptance-drift experiment

This experiment keeps `z-lab/Qwen3-8B-DFlash-b16` fixed while ordinary BF16 GRPO updates only `Qwen/Qwen3-8B`. The public DFlash model card explicitly names that target and disables thinking. Target and tokenizer use revision `b968826d9c46dd6066d109eabc6255188de91218`; the drafter uses revision `9b41424b7109f9c5413454f481b09a82b85333f4` and official K=15.

The cross-arm contract is DAPOMath17K, seed 42, 8 prompts x 4 generations, global batch 32, microbatch 1, input at most 2048 tokens, at most 1024 new tokens, total length 4096, temperature 1, top-p 1, BF16, LR 1e-6, and 10 warmup iterations. Training is Megatron TP2/DP2. vLLM target and DFlash draft are both TP1 because the exact completed Qwen3-8B DFlash evidence uses TP1; TP2 is not assumed.

`policy.draft.enabled` is false. NeMo-RL therefore refits only target weights and loads the public drafter through `draft_load_config.load_format=auto`. Standard `vllm/spec_acceptance_rate`, `vllm/spec_acceptance_length`, and per-position metrics are enabled and persisted to TensorBoard. Four deterministic validation prompts are printed before and after each stage as a fixed qualitative panel.

Run only the staged sequence 1, 10, then 100. A later stage is allowed only after the previous job reaches a clean terminal state, performs a target refit, leaves the drafter fixed, and emits nonzero speculative counters.

The completed K=15 sequence is the TensorBoard-only baseline. The K=3 and K=5 one-step gates use `config_k3.yaml` and `config_k5.yaml`, selected with `DFLASH_K`. They log to W&B project `sna-nemo-rl-fixed-drafter`, group `qwen3-8b-dflash-fixed-drafter-k-sweep`, with deterministic K/step/seed names, tags, and exact artifact/runtime provenance. The runner rejects K=3 or K=5 stages longer than one step until both gates are terminal-green.
