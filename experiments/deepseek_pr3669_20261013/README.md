# DeepSeek-V3 BF16 TRT-LLM refit A/B

Compare 20-step DeepSeek-V3 BF16 GRPO runs on ptyche GB200 with vLLM 0.29.
The main comparison is `triton` versus `trtllm`, both on the same latest-main
plus PR #3669 source. The optional `trtllm-main` arm isolates the PR effect.

The recipe is `grpo-deepseek-v3-32n4g.yaml` with the historical DeepSeek
48-node topology: 32 training nodes, 16 non-colocated generation nodes, 4
GPUs/node, rollout TP16/PP1, Megatron PP8/EP16, GBS 512, maximum sequence
length 1536, and DeepSeek MTP speculative decoding with one token. The model
is the shared BF16 Hugging Face checkpoint under
`/lustre/fsw/coreai_dlalgo_llm/nemo-rl/custom_hf_checkpoints/DeepSeek-V3-BF16`.
The historical OCI-HSG checkpoint path is not mounted on ptyche, so these new
runs must be compared to each other rather than directly to old timings.

Set up one immutable checkout for latest main and one for the integration
branch under `/project/coreai_dlalgo_llm/sna/deepseek-pr3669-20261013/`.
The launcher verifies that its source is clean, writes a manifest, performs
`sbatch --test-only`, and then submits. The three arms do not depend on one
another. Each run writes `ray-driver.log` into its own result directory.

```bash
cd /project/coreai_dlalgo_llm/sna/deepseek-pr3669-20261013/pr
PAIR_ID=deepseek-vllm029-20261013 ARM=triton bash experiments/deepseek_pr3669_20261013/launch_ptyche.sh
PAIR_ID=deepseek-vllm029-20261013 ARM=trtllm bash experiments/deepseek_pr3669_20261013/launch_ptyche.sh
PAIR_ID=deepseek-vllm029-20261013 ARM=trtllm-main bash experiments/deepseek_pr3669_20261013/launch_ptyche.sh
```

W&B project: `nemo-rl-deepseek-pr3669-vllm029`.

Analyze completed jobs over the common steady-state window, excluding step 1.
Report generation, prepare/refit, policy training, logprob, E2E step time,
valid-token throughput per GPU, generation KL, and the exact source/image
SHAs. Do not infer a performance win from an incomplete or failed run.
