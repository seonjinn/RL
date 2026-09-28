# MXFP8 Latest-Main Validation

Re-run the matched MXFP8 precision matrix on GB200 after merging current open
MXFP8 PRs onto NeMo-RL main.

## Pinned Source

- NeMo-RL base: `ebbc8fdc8fd929390d1d7531e142d7b35333b222`
- Integration branch: `codex/mxfp8-latest-main-integration-20260928`
- Megatron-Bridge: `c04b99e74dc607d5cae46fc481ebab03e525f9f3`
- vLLM: `0.29.0`

The final integration SHA and immutable container SHA256 are recorded after
staging and before submission.

## Matrix

Models:

- Qwen3-30B-A3B
- Qwen3-235B-A22B
- Qwen3.5-35B-A3B
- Nemotron 3.5 Lightning 30B-A3B
- Nemotron3 Super, after the smaller-model gates pass

Modes: Sync and Async-1off.

Matched precision arms:

- BF16 training + BF16 rollout
- BF16 training + MXFP8 rollout
- MXFP8 training (`fp8_param=false`) + MXFP8 rollout
- MXFP8 training (`fp8_param=true`) + MXFP8 rollout

All arms retain each performance recipe's workload, GBS, parallelism, and two
logprob calculations. Only precision, refit transport, and required backend
settings differ.

## Procedure

1. Stage and verify the current NeMo-RL nightly image.
2. Run focused unit tests in the staged image.
3. Run two-step Qwen3-30B-A3B Sync and Async smoke tests.
4. Submit matched 20-step pairs, then the MXFP8-training arms.
5. Report steps 2-20 timing means, throughput, `gen_kl_error`, reward, and W&B
   links.

Use `ACTION=test-only` before every submission. Source is archived once and
expanded to node-local storage; build and JIT caches stay under
`/raid/scratch`.
