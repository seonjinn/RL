# Nemotron 3.5 Lightning attention A/B

Compare four precision combinations with FlashInfer and Triton attention on
Lyris GB200. All eight runs use the same committed NeMo-RL source, nightly
container, Async-1off recipe, GBS 512, 8 nodes x 4 GPUs, FlashInfer TRTLLM MoE,
and 20 updates. The only within-arm variable is the attention backend.

| Arm | Training | Rollout |
| --- | --- | --- |
| `bf16-bf16` | BF16 | BF16 |
| `bf16-mxfp8` | BF16 | MXFP8 routed experts |
| `mxfp8-default` | MXFP8, `fp8_param=true` | MXFP8 routed experts |
| `mxfp8-option-b` | Same plus TE op-fuser/grouped tensor | MXFP8 routed experts |

The FlashInfer arm retains vLLM's automatic attention selection, as in the
previous successful Lightning runs. Confirm `FLASHINFER` in worker logs before
including it in the comparison. Triton sets `TRITON_ATTN` explicitly. The MoE
backend remains `flashinfer_trtllm` in both arms; it is not the attention
backend.

Run `git pull --ff-only` on the Lyris source checkout, create one `git archive`
tarball from the committed HEAD, and set `CONTAINER`, `SOURCE_ARCHIVE`,
`SOURCE_COMMIT`, `RESULT_ROOT`, and `WANDB_API_KEY`. For each pair, run
`submit-lyris.sh <arm> <backend> test-only` before submission, then replace
`test-only` with `submit`. The launcher calls `git pull --ff-only` and verifies
the expected commit before every real submission.

Analyze completed W&B steps 2-20 only. Report per-metric valid counts, E2E,
policy, policy/reference logprob, exposed generation wait, refit, logged
throughputs, `gen_kl_error`, reward, and output-token counts. Exposed generation
is unhidden wait, not full vLLM decode latency. Reject nonfinite-KL arms as
accuracy-qualified performance results even if all 20 steps finish.
